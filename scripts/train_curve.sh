#!/usr/bin/env bash
# Learning curve on a plain GPU box (Modal: see modal_app.py): Qwen3-4B at 5k then 10k records (nested subsets, same
# data seed), --no-rules; predict + score the benchmark after each. Decide on 20k / all after comparing the two.
#   bash scripts/train_curve.sh            (defaults: data/final_v2/{train,val,calib,bench}.jsonl)
# Env: DATA (dir), SIZES (default "5000 10000"), MODEL, MICRO_BS (default 4; 8 on 80GB cards), EXTRA (extra train args).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

DATA=${DATA:-data/final_v2}; TRAIN=$DATA/train.jsonl; VAL=$DATA/val.jsonl; CALIB=$DATA/calib.jsonl; BENCH=$DATA/bench.jsonl
SIZES=${SIZES:-"5000 10000"}; MODEL=${MODEL:-Qwen/Qwen3-4B-Base}; MICRO_BS=${MICRO_BS:-4}; EXTRA=${EXTRA:-}
for f in "$TRAIN" "$VAL" "$CALIB" "$BENCH"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
mkdir -p runs

run() {  # $1 model  $2 size  $3 name
  local out="runs/$3"
  echo "=== $3: $1 on $2 records -> $out ($(date))"
  python -m myjev.model.train --train "$TRAIN" --val "$VAL" --calib "$CALIB" --out "$out" --model "$1" \
    --train-limit "$2" --micro-bs "$MICRO_BS" --no-rules --enc-cache runs/enc $EXTRA 2>&1 | tee "runs/$3.train.log"
  python -m myjev.model.predict --ckpt "$out/best" --data "$BENCH" --out "$out/bench_pred.jsonl" 2>&1 | tee "runs/$3.predict.log"
  python -m myjev.evaluate --records "$BENCH" --pred "$3=$out/bench_pred.jsonl" --json-out "$out/bench_eval.json"     | tee "runs/$3.eval.log"
}

for n in $SIZES; do run "$MODEL" "$n" "q4b_${n}"; done

echo "=== summary"
preds=(); for d in runs/q*/; do [ -f "$d/bench_pred.jsonl" ] && preds+=(--pred "$(basename "$d")=$d/bench_pred.jsonl"); done
[ -f runs/jev_bench.jsonl ] && preds+=(--pred "jev=runs/jev_bench.jsonl")
for b in runs/baselines/*.jsonl; do [ -f "$b" ] && preds+=(--pred "$(basename "$b" .jsonl)=$b"); done
python -m myjev.evaluate --records "$BENCH" "${preds[@]}" --by operation --by elements --by tags --by source --json-out runs/summary.json
