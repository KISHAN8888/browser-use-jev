"""Run training / prediction / scoring on Modal GPUs (pay per second; free $30/month credit covers the first curve).

One-time (Git Bash: MSYS_NO_PATHCONV=1 stops it rewriting the leading-slash volume path):
    modal token new                                   # link this laptop to your Modal account
    modal volume create myjev-data
    MSYS_NO_PATHCONV=1 modal volume put myjev-data data/final_v2 /data   # once per dataset rebuild

Runs (--detach so closing the laptop does not stop the job). Read at `modal run` time on the laptop:
MYJEV_GPU (H100 | L40S | A100-80GB, default H100) and MYJEV_DATA (dataset dir on the volume, default /vol/data).
    modal run modal_app.py::sanity                                  # 4B, 25 steps: speed, memory, $ projection
    MYJEV_GPU=L40S modal run modal_app.py::sanity
    modal run modal_app.py::sanity --micro-bs 2 --extra=--no-grad-ckpt
    modal run --detach modal_app.py::curve                          # 4B at 5k then 10k (nested subsets)
    modal run --detach modal_app.py::full                           # 4B on all train records
    modal run --detach modal_app.py::one --limit 10000 --name q4b_10k_b
    modal run --detach modal_app.py::fullrun                        # 4B, all records, bench-scored milestones
    modal run --detach modal_app.py::continue_run                   # continue m22320 on /vol/data_v3 (docs/RUNBOOK.md)
Results:
    modal volume get myjev-data runs runs_modal                     # predictions, logs, eval json
    python scripts/milestones.py                                    # fullrun milestones table, mid-run
Stop early (milestone snapshots/results already committed to the volume survive):
    modal app list            # find the running app's id (ap-...)
    modal app stop --yes <ap-id>    # or "Stop app" in the Modal dashboard
"""

import json
import os
import statistics
import subprocess
import sys
import time

import modal

GPU = os.environ.get("MYJEV_GPU", "H100")
DATA = os.environ.get("MYJEV_DATA", "/vol/data")
RUNS = "/vol/runs"
ENC = "/vol/enc"  # tokenization cache
MODEL = "Qwen/Qwen3-4B-Base"
PRICE = {"H100": 3.95, "L40S": 1.95, "A100-80GB": 2.50}  # $/GPU-hour, approximate Modal list prices (check modal.com/pricing)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.14.1", "transformers==5.18.0", "peft==0.21.2", "accelerate==1.15.0",
        "huggingface_hub", "hf_transfer", "tqdm", "numpy",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf"})
    .add_local_python_source("myjev")
)
app = modal.App("myjev", image=image)
vol = modal.Volume.from_name("myjev-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("myjev-hf-cache", create_if_missing=True)


def sh(args):
    print("$", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def sh_watch(args, out, every=30):
    """sh() for training: commits the volume whenever <out>/milestones.jsonl gains a line (train.py appends it after a
    milestone's files are complete), so milestone results can be downloaded mid-run and survive `modal app stop`.
    Done here, in the Modal function's own process, rather than via a hook inside train.py, because vol.commit() is
    only reliable from the function's process and training stays a subprocess (crash isolation, GPU memory freed)."""
    print("$", " ".join(args), flush=True)
    path = f"{out}/milestones.jsonl"
    count = lambda: sum(1 for _ in open(path, encoding="utf-8")) if os.path.exists(path) else 0
    done = count()  # lines from earlier runs into the same dir
    proc = subprocess.Popen(args)
    while True:
        try:
            rc = proc.wait(timeout=every)
        except subprocess.TimeoutExpired:
            rc = None
        n = count()
        if n > done:
            done = n
            vol.commit()
            print(f"[watch] milestone line {n} written -> volume committed", flush=True)
        if rc is not None:
            break
    if rc:
        raise subprocess.CalledProcessError(rc, args)


def read_log(out):
    """Events of the latest run in <out>/train_log.jsonl (the log is appended to on re-runs)."""
    path = f"{out}/train_log.jsonl"
    if not os.path.exists(path):
        return []
    events = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    starts = [i for i, e in enumerate(events) if e["event"] == "data"]
    return events[starts[-1]:] if starts else events


@app.function(gpu=GPU, volumes={"/vol": vol, "/hf": hf_cache}, timeout=12 * 3600)  # caps a runaway run's cost
def train_and_score(model: str, limit: int, name: str, data: str = DATA, micro_bs: int = 8, extra: str = "",
                    pred_limit: int = 0, milestones: str = "", bench_limit: int = 0, init_from: str = ""):
    """Train one run (--no-rules; limit <= 0 = all records), predict the bench, score it -> /vol/runs/<name>/.
    Prediction + scoring also run if training dies after saving a checkpoint. Returns the train log events.
    milestones="5000,10000": also snapshot + calibrate + score the bench while training (see train.py), restricted to
    <data>/bench_jev_answered_ids.txt when that file exists; the volume is committed after each milestone.
    init_from=<ckpt dir on /vol>: continue training from that checkpoint (train.py --init-from)."""
    files = {s: f"{data}/{s}.jsonl" for s in ("train", "val", "calib", "bench")}
    missing = [f for f in files.values() if not os.path.exists(f)]
    assert not missing, f"missing {missing}; upload the dataset first (see modal_app.py docstring)"
    out, t0 = f"{RUNS}/{name}", time.time()
    if init_from:
        assert os.path.exists(f"{init_from}/adapter") and os.path.exists(f"{init_from}/head.pt"),             f"--init-from {init_from}: no adapter/ + head.pt there"
        assert not os.path.abspath(init_from).startswith(os.path.abspath(out) + "/"),             f"name {name!r} would write into the run that holds {init_from}; pick a new name"
    ms_args = []
    if milestones:
        ids = f"{data}/bench_jev_answered_ids.txt"
        ms_args = ["--milestones", milestones, "--bench", files["bench"], *(["--bench-ids", ids] if os.path.exists(ids) else []),
                   *(["--bench-limit", str(bench_limit)] if bench_limit else [])]
        print("milestone bench restricted to", ids if os.path.exists(ids) else "nothing (all bench records)", flush=True)
    sh([sys.executable, "-c", "import torch; p=torch.cuda.get_device_properties(0); print(p.name, p.total_memory//2**30, 'GiB')"])
    try:
        sh_watch([sys.executable, "-m", "myjev.model.train", "--train", files["train"], "--val", files["val"],
                  "--calib", files["calib"], "--out", out, "--model", model, "--micro-bs", str(micro_bs), "--no-rules",
                  "--enc-cache", ENC, *(["--train-limit", str(limit)] if limit > 0 else []), *ms_args,
                  *(["--init-from", init_from] if init_from else []), *extra.split()],
                 out)
    finally:
        vol.commit()
        hf_cache.commit()
        cfg = f"{out}/best/config.json"
        if os.path.exists(cfg) and os.path.getmtime(cfg) > t0:  # a checkpoint from this run, not a previous one
            if "calib" not in json.load(open(cfg)):
                print("WARNING: training stopped before calibration; predicting with temperature 1.0 (uncalibrated)")
            sh([sys.executable, "-m", "myjev.model.predict", "--ckpt", f"{out}/best", "--data", files["bench"],
                "--out", f"{out}/bench_pred.jsonl", *(["--limit", str(pred_limit)] if pred_limit else [])])
            sh([sys.executable, "-m", "myjev.evaluate", "--records", files["bench"], "--pred",
                f"{name}={out}/bench_pred.jsonl", "--by", "operation", "--by", "source", "--json-out", f"{out}/bench_eval.json"])
            vol.commit()
    return read_log(out)


def summary(log, gpu):
    """One line: speed/memory of the sanity run and the projected cost of a 10k-record x 2-epoch run."""
    ev = lambda kind: [e for e in log if e["event"] == kind]
    data, start, steps = ev("data")[0], ev("start")[0], ev("step")
    warm = [e for e in steps if e["step"] > 10] or steps  # windows that exclude the first (longest, warm-up) steps
    s_step = statistics.median(e["s_per_step"] for e in warm)
    tps = statistics.median(e["tok_per_s"] for e in warm)
    hours = data["tokens_mean"] * 10000 * 2 / tps / 3600
    cost = f" ~${hours * PRICE[gpu]:.0f}" if gpu in PRICE else ""
    return (f"SANITY gpu={start['gpu']} ({gpu}) micro_bs={start['micro_bs']} grad_ckpt={start['grad_ckpt']} "
            f"s/step={s_step} tok/s={tps} peak={max(e['max_mem_gb'] for e in steps)}GB "
            f"-> 10k x 2 epochs ~{hours:.1f} h{cost} (training only, approximate prices)")


@app.local_entrypoint()
def sanity(micro_bs: int = 8, extra: str = ""):
    """4B on 400 records for 25 steps, 100 val/calib/bench records: pipeline check + speed/memory/$ projection.
    Compare e.g. `--micro-bs 2 --extra=--no-grad-ckpt` against the default (micro_bs 8, grad ckpt on)."""
    name = f"sanity_4b_{GPU}_mb{micro_bs}" + ("_nockpt" if "--no-grad-ckpt" in extra else "")
    log = train_and_score.with_options(gpu=GPU).remote(
        MODEL, 400, name, DATA, micro_bs, f"--max-steps 25 --val-limit 100 --calib-limit 100 {extra}", pred_limit=100)
    print(summary(log, GPU))


@app.function(timeout=24 * 3600)
def run_sizes(sizes: list, data: str, gpu: str, micro_bs: int, extra: str):
    """Runs remotely (CPU-only orchestrator) so the sequence survives the laptop disconnecting."""
    for n in sizes:
        train_and_score.with_options(gpu=gpu).remote(MODEL, n, f"q4b_{n or 'full'}", data, micro_bs, extra)


@app.local_entrypoint()
def curve(micro_bs: int = 8, extra: str = ""):
    """4B at 5k then 10k records (nested subsets, same data seed). Decide on 20k after comparing the two."""
    run_sizes.remote([5000, 10000], DATA, GPU, micro_bs, extra)


@app.local_entrypoint()
def full(micro_bs: int = 8, extra: str = ""):
    """4B on all train records."""
    run_sizes.remote([0], DATA, GPU, micro_bs, extra)


@app.local_entrypoint()
def one(model: str = MODEL, limit: int = 10000, name: str = "", micro_bs: int = 8, extra: str = ""):
    train_and_score.with_options(gpu=GPU).remote(model, limit, name or f"{model.split('/')[-1]}_{limit}", DATA, micro_bs, extra)


def milestone_table(lines):
    """Milestone lines (dicts from milestones.jsonl / train log) -> printable table."""
    pct = lambda x: "-" if x is None or x != x else f"{100 * x:.1f}"
    num = lambda x: "-" if x is None or x != x else f"{x:.3f}"
    rows = [f"{'milestone':>9}{'seen':>7}{'ep':>6}{'step':>6}{'elapsed':>9}{'TARGET':>8}{'step':>7}{'op':>7}"
            f"{'ECEop':>7}{'ECEtg':>7}{'n':>6}{'ms_s':>7}"]
    for l in lines:
        rows.append(f"{l['milestone']:>9}{l['records_seen']:>7}{l.get('epoch_frac', l['epoch']):>6}{l['step']:>6}"
                    f"{l['elapsed_s'] / 3600:>8.2f}h{pct(l.get('target_acc')):>8}{pct(l.get('step_acc')):>7}"
                    f"{pct(l.get('op_acc')):>7}{num(l.get('ece_op')):>7}{num(l.get('ece_target')):>7}"
                    f"{l.get('n', '-'):>6}{l.get('milestone_s', '-'):>7}")
    return "\n".join(rows)


@app.local_entrypoint()
def fullrun(micro_bs: int = 8, milestones: str = "5000,10000,22320", name: str = "q4b_full_ms", extra: str = "",
            bench_limit: int = 0, pred_limit: int = 0):
    """4B on all train records, 2 epochs, one GPU job that scores the bench at milestones of records seen (plus the end
    of training) and commits the volume after each, so `python scripts/milestones.py` shows results mid-run and the run
    can be stopped early (`modal app stop --yes <ap-id>`) without losing them. Launch with `modal run --detach`.
    micro_bs 8 + gradient checkpointing (default) measured best on H100. Timeout 7 h: expected ~5 h (~$20), cap ~$28."""
    log = train_and_score.with_options(gpu=GPU, timeout=7 * 3600).remote(
        MODEL, 0, name, DATA, micro_bs, extra, milestones=milestones, bench_limit=bench_limit, pred_limit=pred_limit)
    print(milestone_table([e for e in log if e["event"] == "milestone"]))


@app.local_entrypoint()
def continue_run(init_from: str = f"{RUNS}/q4b_full_ms/milestones/m22320", data: str = "/vol/data_v3",
                 name: str = "q4b_cont_v3", milestones: str = "5000,10000", train_limit: int = 0, epochs: float = 1,
                 lr: float = 1e-4, head_lr: float = 5e-4, micro_bs: int = 8, extra: str = "", bench_limit: int = 0,
                 pred_limit: int = 0):
    """Continue training a saved checkpoint (LoRA adapter + head) on a corrected dataset: same model, fresh optimizer,
    fresh warmup + cosine-to-0 schedule over this run. Bench, val/calib and bench_jev_answered_ids.txt all come from
    `data`. Milestones + volume commits as in fullrun; the loaded model is val-evaluated first (eval tag "init") and is
    best/ until val improves. Launch with `modal run --detach`. Timeout 5 h (~$20 cap on H100)."""
    args = f"--epochs {epochs} --lr {lr} --head-lr {head_lr} {extra}"
    log = train_and_score.with_options(gpu=GPU, timeout=5 * 3600).remote(
        MODEL, train_limit, name, data, micro_bs, args, milestones=milestones, bench_limit=bench_limit,
        pred_limit=pred_limit, init_from=init_from)
    start = next((e for e in log if e["event"] == "start"), {})
    print("init_from", start.get("init_from"), "checksum", start.get("init_checksum"), "steps", start.get("steps"))
    for e in log:
        if e["event"] == "eval":
            print(f"eval {e['tag']:>8} step {e['step']:>5} val nll {e['all']['nll']:.4f} acc {e['all']['acc']:.4f}")
    print(milestone_table([e for e in log if e["event"] == "milestone"]))


@app.local_entrypoint()
def resume_run(init_from: str = f"{RUNS}/q4b_cont_v3/milestones/m10000", resume_step: int = 625,
               data: str = "/vol/data_v3", name: str = "q4b_cont_v3_resume", milestones: str = "",
               lr: float = 1e-4, head_lr: float = 5e-4, micro_bs: int = 8, pred_limit: int = 20):
    """Resume an interrupted continuation: identical batch order + LR schedule as q4b_cont_v3, skipping the first
    `resume_step` optimizer steps, weights from the step-`resume_step` snapshot. Uses .spawn() and returns at once, so
    a laptop network drop cannot stop the app (launch with `modal run --detach`); the end of training is always a
    milestone, so results land in /vol/runs/<name>/milestones.jsonl. Watch with scripts/milestones.py --name <name>."""
    args = f"--epochs 1 --lr {lr} --head-lr {head_lr} --resume-step {resume_step}"
    call = train_and_score.with_options(gpu=GPU, timeout=3 * 3600).spawn(
        MODEL, 0, name, data, micro_bs, args, milestones=milestones or "999999", bench_limit=0,
        pred_limit=pred_limit, init_from=init_from)
    print("spawned", call.object_id, "-> results in", f"{RUNS}/{name}/milestones.jsonl")
