# Runbook

Commands for every stage, in order. Paths are relative to the repo root. On Windows Git Bash, prefix any
`modal volume` command whose remote path starts with `/` with `MSYS_NO_PATHCONV=1` (otherwise Git Bash rewrites
the path), and set `PYTHONUTF8=1`.

## 0. Install

```bash
uv venv --python 3.12 && source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -e ".[model,serve,modal,dev]"                # torch: pick the wheel for your CUDA / CPU
python -m playwright install chromium                        # only for the Mind2Web / WebChain converters
pytest -q tests --ignore=tests/test_mind2web.py
```

GPU box without Modal: `bash scripts/setup_gpu.sh` (installs pinned torch / transformers / peft, downloads the base model).

## 1. Build the dataset (free, CPU, slow for Mind2Web / WebChain)

Converted records land in `data/processed/<source>/`; raw downloads are cached in `data/raw/`.

```bash
python -m myjev.convert.mind2web --split train        --out data/processed/mind2web_train_full/train.jsonl
python -m myjev.convert.mind2web --split test_website --out data/processed/mind2web/test_website.jsonl
python -m myjev.convert.mind2web --split test_domain  --out data/processed/mind2web/test_domain.jsonl
python -m myjev.convert.weblinx  --split train
python -m myjev.convert.weblinx  --split test_web
python -m myjev.convert.webchain                                   # resumable: add --resume after a stop
python -m myjev.convert.nnetnav  --source live --limit 8000        # expects data/raw/nnetnav_live/train.jsonl
python -m myjev.convert.nnetnav  --source wa   --limit 8000        # expects data/raw/nnetnav_wa/train.jsonl
python -m myjev.convert.gobrowse --limit 3000                      # expects parquet shards in data/raw/gobrowse_wa/
python -m myjev.build_dataset --out data/final_v3 \
       --expect-bench-sha256 f95b894c4161ad7c059624916e6f7e4db9460bbd16c3b16a485481534436bd37
```

- NNetNav (`stanfordnlp/nnetnav-live`, `stanfordnlp/nnetnav-wa`) and Go-Browse (`apurvaga/go-browse-wa`) are read
  from local files: download them from the Hugging Face Hub into the paths above first.
- `build_dataset` reads the benchmark label audit from `data/audit/bench_audit.jsonl` (shipped in the HF dataset
  under `extras/`). The released v3 build also used `--reuse-splits <previous build>/stats.json` to keep the
  val/calib websites and train rows of an earlier build; without it the train/val/calib selection can differ
  slightly. `--expect-bench-sha256` fails the build if the benchmark is not byte-identical to the frozen one.
- Live web pages change (WebChain re-fetches CSS), so a rebuild may not be byte-identical. The released files on
  Hugging Face are the reference.

## 2. Jev baseline (paid TypeSafe API, hard dollar cap)

```bash
cp .env.example .env        # fill TYPESAFE_API_KEY (and optionally TYPESAFE_MODEL, default jev-latest)
python -m myjev.run_jev --records data/frozen/bench_v1.jsonl --out runs/jev_bench_v1.jsonl --budget 2.0 --dry-run
python -m myjev.run_jev --records data/frozen/bench_v1.jsonl --out runs/jev_bench_v1.jsonl --budget 2.0 --workers 4
```

`--budget` is a hard cap over the whole spend ledger (`data/jev_calls.jsonl`), checked before every call. Re-running
resumes and never re-pays for finished items. The published run cost ~$1.7; 106 requests were rejected as too large
and are listed in `runs/jev_bench_v1.errors.jsonl`.

## 3. Train on Modal (paid GPU)

```bash
modal token new
modal volume create myjev-data
MSYS_NO_PATHCONV=1 modal volume put myjev-data data/final_v3 /data_v3
MSYS_NO_PATHCONV=1 modal volume put myjev-data data/frozen/bench_jev_answered_ids.txt /data_v3/bench_jev_answered_ids.txt
MSYS_NO_PATHCONV=1 modal volume put myjev-data data/frozen/bench_v1.jsonl /data_v3/bench.jsonl
MYJEV_DATA=/vol/data_v3 modal run modal_app.py::sanity            # 25 steps: speed, memory, $ projection
MYJEV_DATA=/vol/data_v3 modal run --detach modal_app.py::fullrun    # all train records, bench-scored milestones
python scripts/milestones.py --name q4b_full_ms                     # watch milestones (free)
modal app list && modal app stop --yes <app-id>                     # stop early; committed milestones survive
modal volume get myjev-data runs/<run>/milestones/<mN> runs_modal/<run>/<mN>
```

The published model came from two runs: `fullrun` on an earlier build (stopped after epoch 1), then
`continue_run` (1 epoch on v3, LR 1e-4 / head 5e-4) and `resume_run` after an interruption. See
`docs/PIPELINE.md` section 5 and the entry-point docstrings in `modal_app.py` (paths there default to the
original run names; pass `--init-from`, `--data`, `--name` for your own).

Without Modal, on any CUDA box: `DATA=data/final_v3 MICRO_BS=8 bash scripts/train_curve.sh`.

## 4. Predict, evaluate, compare

```bash
python -m myjev.model.predict --ckpt <ckpt dir> --data data/frozen/bench_v1.jsonl --out runs/ours_bench.jsonl
python - <<'EOF'      # restrict the benchmark to the 3,239 steps Jev answered
ids = set(open("data/frozen/bench_jev_answered_ids.txt").read().split())
import json
with open("data/frozen/bench_v1.jsonl", encoding="utf-8") as f, \
     open("data/frozen/bench_v1_jev_answered.jsonl", "w", encoding="utf-8") as g:
    for line in f:
        if json.loads(line)["meta"]["id"] in ids:
            g.write(line)
EOF
python scripts/mind2web_table.py --records data/frozen/bench_v1_jev_answered.jsonl \
       --pred Jev=runs/jev_bench_v1.jsonl --pred Ours=runs/ours_bench.jsonl
python scripts/typefield_score.py --records data/frozen/bench_v1_jev_answered.jsonl \
       --pred Jev=runs/jev_bench_v1.jsonl --pred Ours=runs/ours_bench.jsonl
python -m myjev.evaluate --records data/frozen/bench_v1_jev_answered.jsonl \
       --pred Jev=runs/jev_bench_v1.jsonl --pred Ours=runs/ours_bench.jsonl --by operation --by source --by tags
python -m myjev.baselines --records data/frozen/bench_v1_jev_answered.jsonl --train data/final_v3/train.jsonl \
       --out-dir runs/baselines
```

## 5. Serve inside jev-ultrafast

```bash
MYJEV_CKPT=<ckpt dir> uvicorn myjev.model.server:app --host 0.0.0.0 --port 8000
```

jev-ultrafast posts to `https://api.typesafe.ai/v1/systemone` in `jev_ultrafast/model.py`; change that one URL to
`http://<host>:8000/v1/systemone`. The auth header is ignored. Typed text is still produced by jev-ultrafast's
separate text model.
