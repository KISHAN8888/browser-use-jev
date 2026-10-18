"""Training helpers: per-record loss weights, vectorised temperature fit."""
import pytest
import torch

from myjev.model.scorer import loss_fn
from myjev.model.train import evaluate, fit_temperatures


def test_weighted_loss():
    torch.manual_seed(0)
    logits = torch.randn(3, 4)
    logits[2, 3] = float("-inf")  # padded candidate
    targets = torch.tensor([[1.0, 0, 0, 0], [0, 0.5, 0.5, 0], [0, 0, 1.0, 0]])
    per_q = torch.stack([loss_fn(logits[i : i + 1], targets[i : i + 1], 0.5)[0] for i in range(3)])
    plain, ce = loss_fn(logits, targets, 0.5)
    assert torch.allclose(plain, per_q.mean()) and ce.shape == (3,)
    assert torch.allclose(loss_fn(logits, targets, 0.5, torch.ones(3))[0], plain)
    w = torch.tensor([0.3, 0.3, 1.0])  # e.g. one record of weight 0.3 with two questions, one of weight 1.0
    weighted = loss_fn(logits, targets, 0.5, w)[0]
    assert torch.allclose(weighted, (w * per_q).sum() / w.sum())
    assert torch.allclose(loss_fn(logits, targets, 0.5, torch.tensor([0.0, 0.0, 2.0]))[0], per_q[2])


def fit_temperatures_reference(rows):
    """The original per-question loop implementation."""
    temps = {}
    for kind in ("operation", "target"):
        sel = [r for r in rows if r[0] == kind]
        if not sel:
            temps[kind] = 1.0
            continue
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = sum(-(t * torch.log_softmax(lg / log_t.exp(), -1)).sum() for _, lg, t in sel) / len(sel)
            loss.backward()
            return loss

        opt.step(closure)
        temps[kind] = float(log_t.detach().exp().clamp(0.05, 20.0))
    return temps


@pytest.mark.parametrize("scale", [0.3, 1.0, 4.0])
def test_fit_temperatures_matches_reference(scale):
    g = torch.Generator().manual_seed(int(scale * 10))
    rows = []
    for i in range(300):
        k = int(torch.randint(2, 40, (1,), generator=g))
        lg = torch.randn(k, generator=g) * scale
        t = torch.zeros(k)
        gold = torch.randint(0, k, (2,), generator=g).unique()
        if i % 3:  # informative but miscalibrated logits -> a finite optimal temperature
            lg[gold] += 2.0 * scale
        t[gold] = 1.0 / len(gold)
        rows.append(("operation" if i % 4 == 0 else "target", lg, t))
    new, old = fit_temperatures(rows), fit_temperatures_reference(rows)
    nll_new, nll_old = evaluate(rows, new), evaluate(rows, old)
    for kind in ("operation", "target"):  # same optimum up to LBFGS tolerance (the NLL is flat near it)
        assert new[kind] == pytest.approx(old[kind], rel=2e-2), (kind, new, old)
        assert nll_new[kind]["nll"] == pytest.approx(nll_old[kind]["nll"], abs=1e-4)
        assert nll_new[kind]["nll"] < evaluate(rows)[kind]["nll"] - 1e-3  # calibration actually helped
    assert fit_temperatures([])["operation"] == 1.0


def _tiny_backbone(name, dtype, attn="sdpa"):
    """Randomly initialised 2-layer SmolLM2 (seeded by train.main's torch.manual_seed): fast, deterministic."""
    import transformers
    cfg = transformers.AutoConfig.from_pretrained(name)
    cfg.num_hidden_layers = 2
    return transformers.AutoModel.from_config(cfg, attn_implementation=attn, dtype=dtype)


def _smoke_records(path):
    import json
    import random
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from smoke_cpu import GOLDS, snapshot
    from myjev.jevformat import make_record
    rng = random.Random(0)
    recs = []
    for i, gold in enumerate(GOLDS):  # 8 records
        hist = [{"action": "Where from?", "kind": "fill", "text": "Zürich", "page_changed": True}] if i % 2 else []
        recs.append(make_record(snapshot(rng, i), "Find business flights from Zürich to London", hist, gold,
                                {"id": f"smoke-{i}", "source": "a" if i < 4 else "b"}))
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")


def test_milestones_fire_and_do_not_change_training(tmp_path, monkeypatch):
    import json
    transformers = pytest.importorskip("transformers")
    name = "HuggingFaceTB/SmolLM2-135M"
    try:
        transformers.AutoConfig.from_pretrained(name)
    except Exception as e:  # offline
        pytest.skip(str(e))
    from myjev.model import train
    monkeypatch.setattr(train, "load_backbone", _tiny_backbone)
    data = tmp_path / "records.jsonl"
    _smoke_records(data)
    ids = tmp_path / "ids.txt"
    ids.write_text("smoke-0\nsmoke-2\nsmoke-5\nsmoke-6\nsmoke-7\n")
    # 8 records, micro-bs 2, batch 2 -> 2 records per optimizer step, 4 steps per epoch, 8 steps, 16 records seen.
    common = ["--train", str(data), "--val", str(data), "--model", name, "--max-len", "1024", "--micro-bs", "2",
              "--batch-size", "2", "--eval-bs", "4", "--epochs", "2", "--fp32"]
    fired = []
    out_a = tmp_path / "a"
    train.main(common + ["--out", str(out_a), "--milestones", "3,9,16", "--bench", str(data), "--bench-ids", str(ids)],
               on_milestone=lambda line, d: fired.append((line, d)))
    train.main(common + ["--out", str(tmp_path / "b")])

    lines = [json.loads(l) for l in (out_a / "milestones.jsonl").read_text().splitlines()]
    assert lines == [f[0] for f in fired]
    # m3 fires at step 2 (4 seen, epoch 0); m9 at step 5 (10 seen, after the epoch boundary); end of training at 16
    # (milestone 16 coincides with it -> one snapshot, not two).
    assert [(l["milestone"], l["records_seen"], l["step"], l["epoch"]) for l in lines] == \
        [(3, 4, 2, 0), (9, 10, 5, 1), (16, 16, 8, 1)]
    for l in lines:
        d = out_a / "milestones" / f"m{l['milestone']}"
        assert (d / "adapter").is_dir() and (d / "head.pt").exists()
        cfg = json.loads((d / "config.json").read_text())
        assert cfg["records_seen"] == l["records_seen"] and cfg["step"] == l["step"] and "after" in cfg["calib"]
        assert set(cfg["temperatures"]) == {"operation", "target"}
        assert len((d / "bench_pred.jsonl").read_text().splitlines()) == 5 and l["n"] == 5 and l["missing"] == 0
        report = json.loads((d / "bench_eval.json").read_text())[f"m{l['milestone']}"]
        assert report["overall"]["n"] == 5 and set(report["part"]) == {"a", "b"}
        for k in ("target_acc", "step_acc", "op_acc", "ece_op", "ece_target", "elapsed_s"):
            assert k in l

    def events(out):
        return [json.loads(l) for l in (out / "train_log.jsonl").read_text().splitlines()]
    ev_a, ev_b = events(out_a), events(tmp_path / "b")
    steps_a = [(e["step"], e["loss"], e["ce"]) for e in ev_a if e["event"] == "step"]
    steps_b = [(e["step"], e["loss"], e["ce"]) for e in ev_b if e["event"] == "step"]
    assert steps_a == steps_b and steps_a[-1][0] == 8  # training continued after each milestone, identically
    kinds = [e["event"] for e in ev_a]
    assert kinds.index("milestone") < max(i for i, k in enumerate(kinds) if k == "step")
    assert kinds[-1] == "calibrated"  # best/ + final calibration still happen
    import torch
    ha = torch.load(out_a / "best" / "head.pt")
    hb = torch.load(tmp_path / "b" / "best" / "head.pt")
    assert all(torch.equal(ha[k], hb[k]) for k in ha)
    assert json.loads((out_a / "best" / "config.json").read_text())["step"] == \
        json.loads((tmp_path / "b" / "best" / "config.json").read_text())["step"]


def _tiny_setup(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    name = "HuggingFaceTB/SmolLM2-135M"
    try:
        transformers.AutoConfig.from_pretrained(name)
    except Exception as e:  # offline
        pytest.skip(str(e))
    from myjev.model import train
    monkeypatch.setattr(train, "load_backbone", _tiny_backbone)
    data = tmp_path / "records.jsonl"
    _smoke_records(data)
    common = ["--train", str(data), "--val", str(data), "--max-len", "1024", "--micro-bs", "2",
              "--batch-size", "2", "--eval-bs", "4", "--fp32"]
    return train, name, data, common


def test_init_from_loads_exactly_and_continues(tmp_path, monkeypatch):
    import json
    from myjev.model.predict import Predictor, predict_records
    train, name, data, common = _tiny_setup(tmp_path, monkeypatch)
    # Run A: 6 steps with high LRs so the weights move far from init; the end-of-training milestone m12 snapshots the
    # final model and predicts the bench with it.
    out_a = tmp_path / "a"
    train.main(common + ["--model", name, "--out", str(out_a), "--epochs", "2", "--max-steps", "6", "--lr", "5e-3",
                         "--head-lr", "2e-2", "--milestones", "1000", "--bench", str(data)])
    snap = out_a / "milestones" / "m12"
    cfg_a = json.loads((snap / "config.json").read_text())
    pred_a = [json.loads(l) for l in (snap / "bench_pred.jsonl").read_text().splitlines()]

    # Run B: --init-from the snapshot (base model taken from its config). Before any update, the live model must
    # reproduce A's snapshot predictions exactly.
    seen = {}

    def on_start(model):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        model.eval()
        pred = Predictor.from_model(model, tok, cfg_a, "cpu", torch.float32, True)
        bench = train.load_bench(str(data))
        bench.sort(key=lambda r: -len(json.dumps(r["request"])))  # same order + batches as the milestone bench
        import io
        f = io.StringIO()
        predict_records(pred, bench, f, max(1, 4 // 2))
        seen["pred"] = [json.loads(l) for l in f.getvalue().splitlines()]
        seen["checksum"] = train.trained_checksum(model)
        seen["trainable"] = [n for n, p in model.backbone.named_parameters() if p.requires_grad]
        model.train()

    out_b = tmp_path / "b"
    train.main(common + ["--out", str(out_b), "--init-from", str(snap), "--max-steps", "4", "--lr", "1e-3"],
               on_start=on_start)
    strip = lambda ps: [(p["id"], p["answers"]) for p in ps]
    assert strip(seen["pred"]) == strip(pred_a)  # exact: same choices and identical probabilities
    assert seen["trainable"] and all("lora_" in n for n in seen["trainable"])

    ev_b = [json.loads(l) for l in (out_b / "train_log.jsonl").read_text().splitlines()]
    start = next(e for e in ev_b if e["event"] == "start")
    assert start["init_from"] == str(snap) and start["init_checksum"] == seen["checksum"] and start["epochs"] == 1
    assert start["steps"] == 4 and start["init_tensors"] > 0
    # The loaded model's val metrics equal A's final eval of the same weights (val = train file here).
    ev_a = [json.loads(l) for l in (out_a / "train_log.jsonl").read_text().splitlines()]
    eval_a = [e for e in ev_a if e["event"] == "eval"][-1]
    eval_init = next(e for e in ev_b if e["event"] == "eval")
    assert eval_init["tag"] == "init" and eval_init["all"] == eval_a["all"]

    # (b) training continues from there: fresh warmup schedule, loss moves, weights change.
    steps_b = [e for e in ev_b if e["event"] == "step"]
    assert [e["step"] for e in steps_b] == [1, 2, 3, 4]
    assert len({e["loss"] for e in steps_b}) > 1
    assert steps_b[0]["lr"] < 1e-3 and steps_b[-1]["lr"] == pytest.approx(0.0, abs=1e-12)  # fresh warmup+cosine to 0
    saved = [e["step"] for e in ev_b if e["event"] == "saved"]
    assert saved[0] == 0  # the loaded model is best/ until val improves
    cfg_b = json.loads((out_b / "best" / "config.json").read_text())
    assert cfg_b["init_from"] == str(snap) and cfg_b["base_model"] == name
    assert cfg_b["step"] == 4  # val improved during continuation (deterministic on CPU) -> updated weights saved
    head_a, head_b = torch.load(snap / "head.pt"), torch.load(out_b / "best" / "head.pt")
    assert any(not torch.equal(head_a[k], head_b[k]) for k in head_a)


def test_init_from_mismatch_errors(tmp_path, monkeypatch):
    train, name, data, common = _tiny_setup(tmp_path, monkeypatch)
    out_a = tmp_path / "a"
    train.main(common + ["--model", name, "--out", str(out_a), "--max-steps", "1"])
    ck = str(out_a / "best")
    with pytest.raises(ValueError, match=r"base_model: checkpoint 'HuggingFaceTB/SmolLM2-135M' vs run 'Qwen/Qwen3-4B"):
        train.main(common + ["--out", str(tmp_path / "x"), "--init-from", ck, "--model", "Qwen/Qwen3-4B-Base"])
    with pytest.raises(ValueError, match=r"marker: checkpoint .* vs run '<zz>'"):
        train.main(common + ["--out", str(tmp_path / "y"), "--init-from", ck, "--marker", "<zz>"])
    with pytest.raises(ValueError, match="include_rules"):
        train.main(common + ["--out", str(tmp_path / "z"), "--init-from", ck, "--no-rules"])
    with pytest.raises(ValueError, match="lora_r"):
        train.main(common + ["--out", str(tmp_path / "w"), "--init-from", ck, "--lora-r", "8"])


def test_resume_step_reproduces_interrupted_run(tmp_path, monkeypatch):
    """Resuming at step K from the step-K snapshot replays the same batches and LR: step K+1 has A's exact loss/LR."""
    import json
    train, name, data, common = _tiny_setup(tmp_path, monkeypatch)
    args = common + ["--model", name, "--epochs", "1", "--lr", "5e-3", "--head-lr", "2e-2", "--lora-dropout", "0"]
    out_a = tmp_path / "a"  # 8 records, micro-bs 2, batch 2 -> 4 steps; milestone 4 records -> snapshot after step 2
    train.main(args + ["--out", str(out_a), "--milestones", "4"])
    snap = out_a / "milestones" / "m4"
    assert json.loads((snap / "config.json").read_text())["step"] == 2

    out_b = tmp_path / "b"
    train.main(args + ["--out", str(out_b), "--init-from", str(snap), "--resume-step", "2", "--milestones", "4"])
    ev_a = [json.loads(l) for l in (out_a / "train_log.jsonl").read_text().splitlines()]
    ev_b = [json.loads(l) for l in (out_b / "train_log.jsonl").read_text().splitlines()]
    resumed = next(e for e in ev_b if e["event"] == "resumed")
    assert resumed["step"] == 2 and resumed["records_seen"] == 4
    steps_a = {e["step"]: e for e in ev_a if e["event"] == "step"}
    steps_b = {e["step"]: e for e in ev_b if e["event"] == "step"}
    assert sorted(steps_b) == [3, 4]
    assert steps_b[3]["loss"] == pytest.approx(steps_a[3]["loss"], rel=1e-5)  # same weights + same batch
    assert steps_b[3]["lr"] == pytest.approx(steps_a[3]["lr"]) and steps_b[4]["lr"] == pytest.approx(steps_a[4]["lr"])
    ms_b = [e for e in ev_b if e["event"] == "milestone"]
    assert [m["records_seen"] for m in ms_b] == [8]  # the already-passed milestone (4) is not re-run; end still is
