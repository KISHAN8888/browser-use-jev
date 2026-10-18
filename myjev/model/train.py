"""Train the Choice scorer (LoRA + MLP head) on Jev-format records, then calibrate temperatures.

python -m myjev.model.train --train data/train.jsonl --val data/val.jsonl --out runs/q4b_10k \
    --model Qwen/Qwen3-4B-Base --train-limit 10000 --no-rules

Milestones (--milestones 5000,10000 --bench bench.jsonl [--bench-ids ids.txt]): whenever the number of training
records consumed (counted across epochs, checked at optimizer-step boundaries) crosses a milestone, and always at the
end of training, the live model is snapshotted to <out>/milestones/m<N>/ (same format as best/), calibrated on the
calib set (temperatures stored in that snapshot's config), used in memory to predict the bench
(m<N>/bench_pred.jsonl) and scored with myjev.evaluate (m<N>/bench_eval.json). One compact line per milestone is
appended to <out>/milestones.jsonl (written last, so a line means the milestone's files are complete). Milestone work
runs in eval mode (no dropout) and restores the torch RNG state, so it does not change the training trajectory.

Continuation (--init-from <ckpt_dir>, a best/ or milestones/m<N>/ dir): the LoRA adapter + head weights of that
checkpoint are loaded into the freshly built model (base model, marker, include_rules, max_len and LoRA r/alpha must
match its config.json, else ValueError), then trained with a fresh optimizer and a fresh warmup + cosine-to-0 schedule
over the new run (--epochs defaults to 1). The loaded model is evaluated on val before the first update (eval tag
"init", saved as best/ so the continuation only replaces it if val improves); the start event logs init_from and a
checksum of the loaded parameters.

python -m myjev.model.train --init-from /vol/runs/q4b_full_ms/milestones/m22320 --train ... --lr 1e-4 --no-rules
"""

import argparse
import collections
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch

from .scorer import ChoiceScorer, collate, load_backbone, loss_fn, save_checkpoint, soft_targets, to_device
from .serialize import encode, gold_targets, marker_token


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prepare(records, tok, args):
    """Encode records -> list of (encoded, [question names], [gold key lists], n_tokens, weight)."""
    out = []
    for r in records:
        golds = gold_targets(r)
        enc = encode(r["request"], tok, args.max_len, not args.no_rules, args.marker)
        names = [q for q, g in golds.items() if set(g) & set(enc["questions"][q]["keys"])]
        if "operation" not in names:
            continue
        n = len(enc["prefix"]) + sum(len(enc["questions"][q]["ids"]) for q in names)
        out.append((enc, names, [golds[q] for q in names], n, float(r.get("meta", {}).get("weight", 1.0))))
    return out


def cached_prepare(path, records, tok, args, role, limit):
    """prepare() with an on-disk cache keyed by tokenizer, encoding options and the source file (md5 + size)."""
    if not args.enc_cache:
        return prepare(records, tok, args)
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    # Include the serializer source so editing serialize.py can never serve stale encodings.
    from . import serialize
    code = hashlib.md5(Path(serialize.__file__).read_bytes()).hexdigest()
    key = [args.model, args.max_len, not args.no_rules, args.marker, h.hexdigest(), os.path.getsize(path), limit,
           role, args.data_seed, bool(args.val), code]
    p = Path(args.enc_cache) / (hashlib.md5(json.dumps(key).encode()).hexdigest() + ".pt")
    if p.exists():
        return torch.load(p, weights_only=False)
    ex = prepare(records, tok, args)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ex, p.with_suffix(".tmp"))
    p.with_suffix(".tmp").replace(p)
    return ex


def batches(examples, micro_bs, shuffle, seed):
    """Length-grouped micro-batches: shuffle, sort within mega-chunks of 64 micro-batches, shuffle batches."""
    idx = list(range(len(examples)))
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(idx)
    chunk = micro_bs * 64
    out = []
    for i in range(0, len(idx), chunk):
        part = sorted(idx[i : i + chunk], key=lambda j: examples[j][3])
        out += [part[k : k + micro_bs] for k in range(0, len(part), micro_bs)]
    if shuffle:
        rng.shuffle(out)
    return out


def run_batch(model, examples, ids, args, device, dtype):
    """-> logits, targets, meta, per-question weights (a record's weight repeated per question), non-pad tokens."""
    items = [(examples[j][0], examples[j][1]) for j in ids]
    golds = [g for j in ids for g in examples[j][2]]
    weights = torch.tensor([examples[j][4] for j in ids for _ in examples[j][1]], device=device)
    batch, meta = collate(items, args.pad_id, pack=not args.no_pack, dtype=dtype)
    targets = soft_targets(meta, golds, batch["q_index"].shape[1]).to(device)
    n_tok = sum(examples[j][3] + (len(examples[j][1]) - 1) * len(examples[j][0]["prefix"]) * args.no_pack for j in ids)
    logits = model(to_device(batch, device, dtype))
    return logits, targets, meta, weights, n_tok


@torch.no_grad()
def collect_logits(model, examples, args, device, dtype):
    """Logits/targets per question for evaluation and calibration."""
    model.eval()
    rows = []
    for ids in batches(examples, args.eval_bs, False, 0):
        logits, targets, meta, _, _ = run_batch(model, examples, ids, args, device, dtype)
        for i, m in enumerate(meta):
            k = len(m["keys"])
            rows.append((m["kind"], logits[i, :k].float().cpu(), targets[i, :k].cpu()))
    model.train()
    return rows


def evaluate(rows, temps=None):
    temps = temps or {}
    stats = {}
    for kind in ("operation", "target", "all"):
        sel = [r for r in rows if kind in ("all", r[0])]
        if not sel:
            continue
        nll = acc = 0.0
        for k, lg, t in sel:
            logp = torch.log_softmax(lg / temps.get(k, 1.0), -1)
            nll += -(t * logp).sum().item()
            acc += float(t[lg.argmax()] > 0)
        stats[kind] = {"nll": nll / len(sel), "acc": acc / len(sel), "n": len(sel)}
    return stats


def fit_temperatures(rows):
    """One temperature per question kind, minimizing NLL with LBFGS on log T (all questions in one padded tensor)."""
    temps = {}
    for kind in ("operation", "target"):
        sel = [r for r in rows if r[0] == kind]
        if not sel:
            temps[kind] = 1.0
            continue
        K = max(len(lg) for _, lg, _ in sel)
        L = torch.full((len(sel), K), float("-inf"))
        T = torch.zeros((len(sel), K))
        for i, (_, lg, t) in enumerate(sel):
            L[i, : len(lg)], T[i, : len(t)] = lg, t
        valid = torch.isfinite(L)
        L = L.masked_fill(~valid, 0)  # re-insert -inf only after scaling, so no inf/nan reaches d/dT
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            logp = torch.log_softmax((L / log_t.exp()).masked_fill(~valid, float("-inf")), -1)
            loss = -(T * logp.masked_fill(~valid, 0)).sum() / len(sel)
            loss.backward()
            return loss

        opt.step(closure)
        temps[kind] = float(log_t.detach().exp().clamp(0.05, 20.0))
    return temps


def parse_milestones(spec):
    return sorted({int(x) for x in spec.split(",") if x.strip()}) if spec else []


def load_bench(path, ids_path=None, limit=None):
    """Bench records with meta.id filled in (line index if absent, as predict.py/evaluate.py do), optionally
    restricted to the ids listed in ids_path (one per line), then to the first `limit`."""
    bench = []
    for i, r in enumerate(read_jsonl(path)):
        meta = r.get("meta") or {}
        bench.append(r | {"meta": meta | {"id": meta.get("id", i)}})
    if ids_path:
        ids = {l.strip() for l in open(ids_path, encoding="utf-8") if l.strip()}
        bench = [r for r in bench if str(r["meta"]["id"]) in ids]
    return bench[:limit]


def score_bench(bench, pred_path, name):
    """myjev.evaluate report for one prediction file: overall + by bench part / source + by operation."""
    from .. import evaluate as ev
    rows, missing = ev.evaluate(bench, ev.load_jsonl(pred_path))
    return {name: {"overall": ev.summarize(rows), "missing": missing, ev.PART_KEY: ev.group_by(rows, ev.PART_KEY),
                   "operation": ev.group_by(rows, "operation")}}


def load_trained_weights(model, ckpt, device):
    """Copy a checkpoint's LoRA adapter + head weights into the live model in place (the PeftModel and its trainable
    flags are kept, so training can continue). Errors if any LoRA tensor of the model or the file is not matched."""
    from peft import set_peft_model_state_dict
    from peft.utils import load_peft_weights
    ckpt = Path(ckpt)
    sd = load_peft_weights(str(ckpt / "adapter"), device=str(device))
    res = set_peft_model_state_dict(model.backbone, sd)
    missing = [k for k in res.missing_keys if "lora_" in k]
    if missing or res.unexpected_keys:
        raise ValueError(f"adapter {ckpt / 'adapter'} does not fit the model: missing {missing[:5]} "
                         f"({len(missing)}), unexpected {list(res.unexpected_keys)[:5]} ({len(res.unexpected_keys)})")
    model.head.load_state_dict(torch.load(ckpt / "head.pt", map_location=device))
    return len(sd)


def trained_checksum(model):
    """md5 over the trainable (LoRA) + head tensors, name-sorted, as fp32 bytes: identifies loaded/saved weights."""
    h = hashlib.md5()
    named = [(n, p) for n, p in model.backbone.named_parameters() if p.requires_grad]
    named += [("head." + n, p) for n, p in model.head.named_parameters()]
    for n, p in sorted(named, key=lambda x: x[0]):
        h.update(n.encode())
        h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def check_init_config(cfg, args):
    """--init-from compatibility: everything that changes the weights' meaning must match the checkpoint."""
    want = {"base_model": args.model, "include_rules": not args.no_rules, "max_len": args.max_len,
            "lora_r": args.lora_r, "lora_alpha": args.lora_alpha}
    if args.marker:
        want["marker"] = args.marker
    bad = {k: (cfg.get(k), v) for k, v in want.items() if k in cfg and cfg[k] != v}
    if "base_model" not in cfg:
        bad["base_model"] = (None, args.model)
    if bad:
        raise ValueError(f"--init-from {args.init_from} does not match this run: "
                         + ", ".join(f"{k}: checkpoint {a!r} vs run {b!r}" for k, (a, b) in bad.items())
                         + " (pass matching --model/--marker/--max-len/--no-rules/--lora-r/--lora-alpha)")


def main(argv=None, on_milestone=None, on_start=None):
    """on_milestone(line: dict, snapshot_dir: Path) is called after each milestone's files are written.
    on_start(model) is called once right before the first optimizer step (tests)."""
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", help="default: 5%% of --train held out")
    p.add_argument("--calib", help="calibration file (default: --val)")
    p.add_argument("--out", required=True)
    p.add_argument("--model", help="base model (default: --init-from's base_model, else Qwen/Qwen3-1.7B-Base)")
    p.add_argument("--init-from", help="checkpoint dir (adapter/, head.pt, config.json) to continue training from")
    p.add_argument("--train-limit", type=int, help="nested seed-fixed subset size (5000/10000)")
    p.add_argument("--resume-step", type=int, default=0,
                   help="resume an interrupted run: rebuild the identical batch order and LR schedule, skip the first N "
                        "optimizer steps (load their weights with --init-from; optimizer moments restart)")
    p.add_argument("--val-limit", type=int, help="first N val records only (short sanity runs)")
    p.add_argument("--calib-limit", type=int, help="first N calib records only")
    p.add_argument("--data-seed", type=int, default=0, help="fixes the subset order; keep constant across the curve")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--epochs", type=float, help="default 2, or 1 with --init-from")
    p.add_argument("--batch-size", type=int, default=16, help="effective batch (records)")
    p.add_argument("--micro-bs", type=int, default=4)
    p.add_argument("--eval-bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=float, default=0.05)
    p.add_argument("--brier", type=float, default=0.5)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--attn", default="sdpa", help="sdpa | eager | flash_attention_2 (flash needs --no-pack)")
    p.add_argument("--no-pack", action="store_true", help="one row per question instead of shared-prefix packing")
    p.add_argument("--no-rules", action="store_true", help="drop the constant NEXT_ACTION/TARGET rules (~700 tokens)")
    p.add_argument("--marker", help="override marker token")
    p.add_argument("--max-steps", type=int, help="stop after N optimizer steps (smoke tests)")
    p.add_argument("--eval-every", type=int, default=0, help="also eval every N optimizer steps")
    p.add_argument("--no-grad-ckpt", action="store_true", help="disable gradient checkpointing (faster, more memory)")
    p.add_argument("--enc-cache", help="dir to cache tokenized examples in (e.g. /vol/enc)")
    p.add_argument("--fp32", action="store_true", help="fp32 weights (CPU smoke test)")
    p.add_argument("--milestones", help="records seen (across epochs), e.g. 5000,10000,22320; end of training is "
                                        "always one too. Snapshot + calibrate + bench-score at each")
    p.add_argument("--bench", help="records to predict + score at each milestone")
    p.add_argument("--bench-ids", help="file with one record id per line: milestone bench restricted to these ids")
    p.add_argument("--bench-limit", type=int, help="first N bench records (after --bench-ids) only")
    p.add_argument("--on-milestone-cmd", help="shell command run after each milestone (env MYJEV_MILESTONE_DIR)")
    args = p.parse_args(argv)
    init_cfg = None
    if args.init_from:
        init_cfg = json.loads((Path(args.init_from) / "config.json").read_text())
        args.model = args.model or init_cfg.get("base_model")
    args.model = args.model or "Qwen/Qwen3-1.7B-Base"
    args.epochs = args.epochs if args.epochs is not None else (1 if args.init_from else 2)
    if init_cfg is not None:
        check_init_config(init_cfg, args)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.fp32 or device.type == "cpu" else torch.bfloat16
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.jsonl", "a", encoding="utf-8")

    def emit(**kw):
        kw["time"] = time.strftime("%H:%M:%S")
        print(json.dumps(kw), flush=True)
        log.write(json.dumps(kw) + "\n")
        log.flush()

    tok = AutoTokenizer.from_pretrained(args.model)
    args.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    args.marker = args.marker or marker_token(tok)
    if init_cfg is not None and init_cfg.get("marker", args.marker) != args.marker:
        raise ValueError(f"--init-from {args.init_from} does not match this run: marker: checkpoint "
                         f"{init_cfg['marker']!r} vs run {args.marker!r} (pass --marker)")

    records = read_jsonl(args.train)
    order = list(range(len(records)))
    random.Random(args.data_seed).shuffle(order)  # same permutation for every size -> nested subsets
    if args.val:
        val_records = read_jsonl(args.val)
    else:
        n_val = max(1, len(records) // 20)
        val_records = [records[i] for i in order[-n_val:]]
        order = order[:-n_val]
    train_records = [records[i] for i in order[: args.train_limit]]
    val_records = val_records[: args.val_limit]

    t0 = time.time()
    train_ex = cached_prepare(args.train, train_records, tok, args, "train", args.train_limit)
    val_ex = cached_prepare(args.val or args.train, val_records, tok, args, "val", args.val_limit)
    if args.calib:
        calib_records = read_jsonl(args.calib)[: args.calib_limit]
        calib_ex = cached_prepare(args.calib, calib_records, tok, args, "calib", args.calib_limit)
    else:
        calib_ex = val_ex
    milestones = parse_milestones(args.milestones) if (args.milestones or args.bench) else None
    bench = load_bench(args.bench, args.bench_ids) if args.bench and milestones is not None else []
    if args.bench_ids:
        emit(event="bench_ids", matched=len(bench))
    # Longest first: similar lengths per batch, and the biggest batch (bench rows reach ~6.3k tokens, longer than any
    # training row) runs first, so an out-of-memory error shows up at the first milestone, not at the end of the run.
    bench.sort(key=lambda r: -len(json.dumps(r["request"])))
    bench = bench[: args.bench_limit or None]
    lens = sorted(e[3] for e in train_ex)
    emit(event="data", train=len(train_ex), val=len(val_ex), calib=len(calib_ex), marker=args.marker,
         tokens_p50=lens[len(lens) // 2], tokens_p95=lens[int(len(lens) * 0.95)], tokens_max=lens[-1],
         tokens_mean=round(statistics.mean(lens), 1), encode_s=round(time.time() - t0, 1),
         milestones=milestones, bench=len(bench))

    backbone = load_backbone(args.model, dtype, args.attn)
    backbone.config.use_cache = False
    if not args.no_grad_ckpt:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules="all-linear", bias="none")
    backbone = get_peft_model(backbone, lora)
    backbone.print_trainable_parameters()
    model = ChoiceScorer(backbone, backbone.config.hidden_size).to(device)
    model.head.float()
    init_info = {}
    if args.init_from:
        n_loaded = load_trained_weights(model, args.init_from, device)
        init_info = {"init_from": args.init_from, "init_tensors": n_loaded, "init_checksum": trained_checksum(model),
                     "init_step": init_cfg.get("step"), "init_records_seen": init_cfg.get("records_seen")}
    model.train()

    accum = max(1, args.batch_size // args.micro_bs)
    steps_per_epoch = math.ceil(len(train_ex) / args.micro_bs / accum)
    total = args.max_steps or max(1, math.ceil(steps_per_epoch * args.epochs))
    lora_params = [p for p in model.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora_params, "lr": args.lr},
                             {"params": model.head.parameters(), "lr": args.head_lr}], weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup * total), total)
    for _ in range(args.resume_step):  # fast-forward the LR schedule to where the interrupted run stopped
        sched.step()
    config = {k: v for k, v in vars(args).items()} | {"base_model": args.model, "head_dim": 256,
                                                      "include_rules": not args.no_rules,
                                                      "temperatures": {"operation": 1.0, "target": 1.0}}
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    emit(event="start", steps=total, accum=accum, steps_per_epoch=steps_per_epoch, device=str(device), gpu=gpu,
         micro_bs=args.micro_bs, grad_ckpt=not args.no_grad_ckpt, epochs=args.epochs, lr=args.lr, head_lr=args.head_lr,
         trainable=sum(p.numel() for p in lora_params), **init_info)

    best, step, epoch = float("inf"), 0, 0
    def checkpoint_if_best(tag):
        nonlocal best
        stats = evaluate(collect_logits(model, val_ex, args, device, dtype))
        emit(event="eval", tag=tag, step=step, **stats)
        if stats["all"]["nll"] < best:
            best = stats["all"]["nll"]
            save_checkpoint(model, tok, out / "best", config | {"val": stats, "step": step})
            emit(event="saved", step=step, val_nll=best)

    seen, t_train = 0, time.time()  # records consumed by optimizer steps so far (across epochs)

    def run_milestone(label):
        """Snapshot + calibrate + bench-score the live model, then hand it back to training untouched."""
        from .predict import Predictor, predict_records
        t_ms = time.time()
        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        d = out / "milestones" / f"m{label}"
        info = {"milestone": label, "records_seen": seen, "epoch": epoch, "epoch_frac": round(seen / len(train_ex), 4),
                "step": step}
        rows = collect_logits(model, calib_ex, args, device, dtype)
        temps = fit_temperatures(rows)
        calib = {"before": evaluate(rows), "after": evaluate(rows, temps)}
        save_checkpoint(model, tok, d, config | info | {"temperatures": temps, "calib": calib})
        line = info | {"elapsed_s": round(t_ms - t_train, 1), "temperatures": temps,
                       "calib_nll_after": round(calib["after"]["all"]["nll"], 4)}
        if bench:
            model.eval()
            pred = Predictor.from_model(model, tok, config | {"temperatures": temps}, device, dtype, not args.no_pack)
            with open(d / "bench_pred.jsonl", "w", encoding="utf-8") as f:
                predict_records(pred, bench, f, max(1, args.eval_bs // 2))  # half batch: bench rows are longer
            report = score_bench(bench, d / "bench_pred.jsonl", f"m{label}")
            (d / "bench_eval.json").write_text(json.dumps(report, indent=1))
            s = report[f"m{label}"]["overall"]
            line |= {"target_acc": s["target_acc"], "step_acc": s["step_acc"], "op_acc": s["operation_acc"],
                     "ece_op": s["ece_operation"], "ece_target": s["ece_target"], "n": s["n"],
                     "target_acc_macro": s["target_acc_macro"], "missing": report[f"m{label}"]["missing"]}
        line["milestone_s"] = round(time.time() - t_ms, 1)
        with open(out / "milestones.jsonl", "a", encoding="utf-8") as f:  # last: a line means the files are complete
            f.write(json.dumps(line) + "\n")
        emit(event="milestone", **line)
        model.train()
        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state_all(rng_cuda)
        if args.on_milestone_cmd:
            subprocess.run(args.on_milestone_cmd, shell=True, env=os.environ | {"MYJEV_MILESTONE_DIR": str(d)})
        if on_milestone:
            on_milestone(line, d)

    if args.init_from:  # baseline: the loaded model on this run's val; best/ starts as the loaded model
        checkpoint_if_best("init")
    if on_start:
        on_start(model)
    window = collections.deque(maxlen=10)  # (seconds, tokens) of the last 10 steps
    while step < total:
        mb = batches(train_ex, args.micro_bs, True, args.seed + epoch)
        if epoch == 0:  # largest padded micro-batch first, so an OOM shows up at step 1
            k = max(range(len(mb)), key=lambda b: len(mb[b]) * max(train_ex[j][3] for j in mb[b]))
            mb.insert(0, mb.pop(k))
        if epoch == 0 and args.resume_step:  # same order as the interrupted run; drop what it already trained on
            skipped, mb = mb[: args.resume_step * accum], mb[args.resume_step * accum :]
            step, seen = args.resume_step, sum(len(ids) for ids in skipped)
            if milestones is not None:
                milestones = [m for m in milestones if m > seen]
            emit(event="resumed", step=step, records_seen=seen, lr=sched.get_last_lr()[0])
        for i in range(0, len(mb), accum):
            t_step = time.time()
            group = mb[i : i + accum]
            w_group = max(1e-8, sum(train_ex[j][4] * len(train_ex[j][1]) for ids in group for j in ids))
            loss_sum = ce_sum = nq = n_tok = 0.0
            for ids in group:
                logits, targets, _, w, nt = run_batch(model, train_ex, ids, args, device, dtype)
                loss, ce = loss_fn(logits, targets, args.brier, w)
                scale = w.sum().item() / w_group  # -> weighted mean over all questions of the optimizer step
                (loss * scale).backward()
                loss_sum += loss.item() * scale
                ce_sum += ce.sum().item()
                nq += len(ce)
                n_tok += nt
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            seen += sum(len(ids) for ids in group)
            window.append((time.time() - t_step, n_tok))
            if step % 10 == 0 or step <= 3 or step == total:
                mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
                dt = sum(d for d, _ in window)
                emit(event="step", step=step, epoch=epoch, loss=round(loss_sum, 4), ce=round(ce_sum / nq, 4),
                     lr=sched.get_last_lr()[0], tokens=int(n_tok), s_per_step=round(dt / len(window), 2),
                     tok_per_s=round(sum(t for _, t in window) / dt), max_mem_gb=round(mem, 1))
            if milestones is not None:
                crossed = [m for m in milestones if seen >= m]
                if crossed or step >= total:
                    milestones = [m for m in milestones if seen < m]
                    run_milestone(seen if step >= total else max(crossed))
            if args.eval_every and step % args.eval_every == 0:
                checkpoint_if_best(f"step{step}")
            if step >= total:
                break
        epoch += 1
        checkpoint_if_best(f"epoch{epoch}")

    # Calibrate the best checkpoint: reload its adapter + head weights into the live model.
    load_trained_weights(model, out / "best", device)
    rows = collect_logits(model, calib_ex, args, device, dtype)
    temps = fit_temperatures(rows)
    cfg_path = out / "best" / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["temperatures"] = temps
    cfg["calib"] = {"before": evaluate(rows), "after": evaluate(rows, temps)}
    cfg_path.write_text(json.dumps(cfg, indent=2))
    emit(event="calibrated", temperatures=temps, **{f"calib_{k}": v for k, v in cfg["calib"].items()})
    log.close()


if __name__ == "__main__":
    main()
