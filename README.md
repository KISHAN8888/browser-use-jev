# myjev

An open decision model and dataset for [browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast).
It answers the same typed questions that TypeSafe's hosted **Jev** model answers on every browser step (which
operation, which element), in the same request/response format, so jev-ultrafast can use it by changing one URL.

- Model: [`<hf-user>/myjev-decision-4b`](https://huggingface.co/<hf-user>/myjev-decision-4b) (Qwen3-4B-Base + LoRA + scoring head)
- Dataset: [`<hf-user>/myjev-browser-decisions`](https://huggingface.co/datasets/<hf-user>/myjev-browser-decisions)
  (21,646 train steps on 301 websites + a frozen 3,345-step benchmark on 84 unseen websites)

## Why

Jev is a closed API. This repo shows that an open 4B model, trained only on public web-agent datasets converted
into jev-ultrafast's exact request format, can match or beat it on held-out websites, run locally, and expose
calibrated probabilities.

## Results

Benchmark: human-labelled steps on websites never seen in training (Mind2Web test_website + test_domain, WebLINX
test_web), in jev-ultrafast's exact request format. The 106 steps Jev's API rejected as too large are excluded for
both systems (3,239 steps scored).

**Headline (Mind2Web Cross-Website + Cross-Domain, 2,074 steps, 62 unseen websites): Step SR 64.8% (ours) vs
57.8% (Jev 1.13), +7.0 points, paired 95% CI 4.9–9.2; ours ahead on 44 of 62 websites.**

| | Cross-Website (660) | | | Cross-Domain (1,414) | | | WebLINX test_web (1,165) | | | All (3,239) | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | Ele. Acc | Op. Acc | Step SR | Ele. Acc | Op. Acc | Step SR | Ele. Acc | Op. Acc | Step SR | Ele. Acc | Op. Acc | Step SR |
| TypeSafe Jev 1.13 | 66.2 | 79.5 | 63.2 | 56.4 | 80.6 | 55.3 | 51.9 | 47.9 | 32.6 | 57.2 | 68.6 | 48.7 |
| Ours (Qwen3-4B) | 73.8 | 90.5 | 71.4 | 62.8 | 91.1 | 61.7 | 62.6 | 79.1 | 57.2 | 65.2 | 86.6 | 62.1 |

Caveats (from a fairness audit, details in [`docs/RESULTS.md`](docs/RESULTS.md)):

- The all-splits gap (62.1 vs 48.7) **overstates** our lead. WebLINX test_web penalises Jev for dataset
  conventions it never saw (scroll-while-reading as gold, click-then-type labelled CLICK, chat-transcript goals);
  our model was trained on WebLINX train. With scroll-gold steps dropped and lenient click-then-type scoring:
  56.9 vs 64.4 overall, 51.5 vs 61.5 on WebLINX.
- Jev is still better on TYPE_TEXT steps (77.1 vs 65.6) and SELECT (81.2 vs 39.6, only 48 steps).
- Candidate sets are viewport-filtered (median 34 elements), so numbers are not comparable to published Mind2Web
  results. Step-level accuracy is not live task success; DONE/BLOCKED are trained but not benchmarked.
- Calibration (ECE action / element): ours 0.017 / 0.021, Jev 0.062 / 0.077. Latency: ours ~105 ms on an H100
  (batched benchmark scoring) vs Jev 321 ms p50 over the API; not a like-for-like comparison.

## How it works

```
open datasets ──convert──► Jev-format records ──clean/split──► train / val / calib / frozen bench
                                                                 │                 │
                              Qwen3-4B + LoRA + scoring head ◄───┘                 │
                                         │ train (Modal H100) → calibrate          │
                                         ▼                                         ▼
                                predictions ──────────── same scorer ──────── Jev API predictions
```

- **Format** (`myjev/jevformat.py`): builds the exact request jev-ultrafast sends, verified byte-identical to
  upstream (`tests/test_parity.py`). Format notes: [`docs/jev_ultrafast_format.md`](docs/jev_ultrafast_format.md).
- **Converters** (`myjev/convert/`): Mind2Web and WebChain pages are rendered in headless Chromium and run through
  jev-ultrafast's real `snapshot.js`; WebLINX, NNetNav and Go-Browse are converted from accessibility trees.
- **Dataset build** (`myjev/build_dataset.py`, `myjev/relabel.py`): dedupe, website-disjoint splits, caps,
  per-record weights, click-then-type label fix, frozen audited benchmark.
- **Model** (`myjev/model/`): Qwen3-4B-Base body, LoRA r16/α32 on all linear layers, a 2560→256→1 MLP head
  scoring each option at a `<|box_end|>` marker token; all questions of a request packed into **one forward
  pass**; per-question-kind temperature calibration. No text generation.
- **Server** (`myjev/model/server.py`): `POST /v1/systemone`, Jev-shaped answers.

Full pipeline description: [`docs/PIPELINE.md`](docs/PIPELINE.md). Step-by-step commands: [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

## Quickstart

Requires Python ≥ 3.12. A CUDA GPU is recommended for inference (the 4B base model is ~8 GB in bf16; on CPU it
runs in fp32 and needs ~16 GB RAM).

```bash
git clone https://github.com/<github-user>/myjev && cd myjev
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[model,serve]"         # add ,modal,dev for training on Modal / running tests

# model checkpoint (adapter + head + tokenizer + config; the base model downloads automatically)
huggingface-cli download <hf-user>/myjev-decision-4b --local-dir ckpt/myjev-decision-4b
# dataset
huggingface-cli download <hf-user>/myjev-browser-decisions --repo-type dataset --local-dir data/hf
```

Score the benchmark:

```bash
python -m myjev.model.predict --ckpt ckpt/myjev-decision-4b --data data/hf/data/benchmark.jsonl \
       --out runs/ours_bench.jsonl
python -m myjev.evaluate --records data/hf/data/benchmark.jsonl --pred ours=runs/ours_bench.jsonl --by operation
```

From Python:

```python
import json
from myjev.model.predict import Predictor

pred = Predictor("ckpt/myjev-decision-4b")            # cuda if available, else cpu
record = json.loads(open("data/hf/data/benchmark.jsonl", encoding="utf-8").readline())
answers = pred.answer([record["request"]])[0]          # list of Jev request bodies in, one dict per request out
print(answers["operation"]["choice"], answers["operation"]["confidence"])
```

Use it inside jev-ultrafast:

```bash
MYJEV_CKPT=ckpt/myjev-decision-4b uvicorn myjev.model.server:app --host 0.0.0.0 --port 8000
```

Then in jev-ultrafast's `jev_ultrafast/model.py` replace `https://api.typesafe.ai/v1/systemone` with
`http://<host>:8000/v1/systemone` (the auth header is ignored).

## Rebuild the dataset from the sources

```bash
uv pip install -e ".[dev]" && python -m playwright install chromium
python -m myjev.convert.mind2web --split test_website --out data/processed/mind2web/test_website.jsonl
# ... the other converters, see docs/RUNBOOK.md section 1
python -m myjev.build_dataset --out data/final_v3 \
       --expect-bench-sha256 f95b894c4161ad7c059624916e6f7e4db9460bbd16c3b16a485481534436bd37
```

Mind2Web and WebChain conversion renders every step in Chromium (hours on a laptop). Live pages and source
updates mean a rebuild may differ slightly from the released files; the HF dataset is the reference.

## Train on Modal

```bash
uv pip install -e ".[model,modal]" && modal token new && modal volume create myjev-data
MSYS_NO_PATHCONV=1 modal volume put myjev-data data/final_v3 /data_v3     # MSYS_NO_PATHCONV only in Git Bash
MYJEV_DATA=/vol/data_v3 modal run modal_app.py::sanity                    # 25 steps: speed, memory, $ estimate
MYJEV_DATA=/vol/data_v3 modal run --detach modal_app.py::fullrun          # full run with bench-scored milestones
```

Measured on one H100: ~5 s/step, peak 25.5 GB; one epoch over 21.6k records is roughly 2 GPU-hours. Modal bills
per second; check current prices before launching. The data dir on the volume must contain `train.jsonl`,
`val.jsonl`, `calib.jsonl`, `bench.jsonl` (rename the HF files `validation`→`val`, `calibration`→`calib`,
`benchmark`→`bench`). Details: `modal_app.py` docstrings and `docs/RUNBOOK.md` section 3.

## Reproduce the Jev baseline (your own TypeSafe key, paid)

Jev's raw API outputs are not redistributed. To regenerate them:

```bash
cp .env.example .env                     # set TYPESAFE_API_KEY
python -m myjev.run_jev --records data/hf/data/benchmark.jsonl --out runs/jev_bench_v1.jsonl --budget 2.0 --dry-run
python -m myjev.run_jev --records data/hf/data/benchmark.jsonl --out runs/jev_bench_v1.jsonl --budget 2.0 --workers 4
```

The published run used `jev-latest` = jev-1.13.0 and cost ~$1.7; later Jev versions may score differently.
Compare on the steps Jev answered (`benchmark_jev_answered_ids.txt`) with `scripts/mind2web_table.py`
(see `docs/RUNBOOK.md` section 4).

## Tests

```bash
uv pip install -e ".[model,dev]"
pytest -q tests --ignore=tests/test_mind2web.py   # test_mind2web.py needs network + Chromium
```

## Licences

- Code: Apache-2.0 ([`LICENSE`](LICENSE)).
- Vendored files from browser-use/jev-ultrafast (`myjev/vendor/`, `tests/upstream/`): MIT, see
  [`myjev/vendor/LICENSE-jev-ultrafast`](myjev/vendor/LICENSE-jev-ultrafast) and [`tests/upstream/LICENSE`](tests/upstream/LICENSE).
- Dataset and model weights (on Hugging Face): CC-BY-NC-SA-4.0, because the training data includes WebLINX
  (CC-BY-NC-SA-4.0). Other sources: Mind2Web CC-BY-4.0, WebChain CC-BY-4.0, NNetNav Apache-2.0, Go-Browse MIT
  (repository licence). The base model Qwen3-4B-Base is Apache-2.0.

This project is not affiliated with TypeSafe or Browser Use. "Jev" refers to TypeSafe's hosted model, used here
only as a baseline through its public API.

## Acknowledgements and citation

Built on: Mind2Web (Deng et al., 2023), WebLINX (Lù et al., 2024), WebChain, NNetNav (Murty et al., 2024),
Go-Browse (Gandhi et al., 2025), browser-use/jev-ultrafast, and Qwen3 (Qwen Team, 2025). Please cite the original
datasets if you use the data.

```bibtex
@misc{myjev2026,
  title  = {myjev: an open decision model for jev-ultrafast browser agents},
  author = {<github-user>},
  year   = {2026},
  url    = {https://github.com/<github-user>/myjev}
}
```
