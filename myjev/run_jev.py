"""Run Jev on a records file and write predictions in the evaluator format. Resumable and budget-capped.

    python -m myjev.run_jev --records bench.jsonl --out runs/jev_bench.jsonl --budget 0.60 --dry-run
    python -m myjev.run_jev --records bench.jsonl --out runs/jev_bench.jsonl --budget 0.60 --workers 4
"""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .evaluate import load_jsonl, record_id
from .jev_client import PRICE_PER_M_INPUT, Jev, estimate_tokens, spent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--budget", type=float, required=True, help="hard cap in USD over the whole ledger")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-errors", action="store_true", help="re-run items that previously returned an API error")
    args = parser.parse_args()

    records = load_jsonl(args.records)[: args.limit]
    out = Path(args.out)
    errors_path = out.with_suffix(".errors.jsonl")
    done = {str(json.loads(line)["id"]) for line in out.open(encoding="utf-8")} if out.exists() else set()
    if errors_path.exists() and not args.retry_errors:
        done |= {str(json.loads(line)["id"]) for line in errors_path.open(encoding="utf-8")}
    todo = [(record_id(r, i), r) for i, r in enumerate(records) if record_id(r, i) not in done]
    estimate = sum(estimate_tokens(r["request"]) for _, r in todo) * PRICE_PER_M_INPUT / 1e6
    print(f"{len(todo)} to run ({len(done)} already done); estimated ${estimate:.4f}; ledger so far ${spent():.4f}")
    if args.dry_run or not todo:
        return

    jev, lock = Jev(budget_usd=args.budget), threading.Lock()
    out.parent.mkdir(parents=True, exist_ok=True)

    def run(item):
        rid, record = item
        result, entry = jev.ask(record["request"], tag=f"{Path(args.records).stem}:{rid}")
        line = {"id": rid, "answers": result.get("answers", {}), "latency_ms": entry["latency_ms"],
                "model": result.get("model"), "usage": entry["usage"]}
        with lock, out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")

    def run_safe(item):
        try:
            run(item)
        except RuntimeError as error:
            if "Budget cap" in str(error) or "Ledger" in str(error):
                raise
            # A per-item API error (e.g. max_tokens_exceeded) is a real Jev failure on that step: record it.
            with lock, errors_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"id": item[0], "error": str(error)[:300]}) + "\n")

    with ThreadPoolExecutor(args.workers) as pool:
        for future in [pool.submit(run_safe, item) for item in todo]:
            try:
                future.result()
            except Exception as error:  # budget/ledger problems stop the whole run
                print(error)
                pool.shutdown(cancel_futures=True)
                break
    print(f"ledger total ${spent():.4f}")


if __name__ == "__main__":
    main()
