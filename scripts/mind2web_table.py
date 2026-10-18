"""Results in the standard Mind2Web format: Ele. Acc, Op. Acc, Step SR per test split.

- Ele. Acc: the element the model actually chose (in the head of the operation it chose) is a gold element.
  Measured over steps whose gold action has an element. A select option counts by its element (the dropdown);
  the chosen option is part of the operation, as in Mind2Web.
- Op. Acc: the action type is right. Mind2Web's Op. F1 also scores typed text; neither Jev nor our model writes
  text (jev-ultrafast uses a separate text model), so for these systems Op. F1 reduces to action-type accuracy.
- Step SR: action right AND (if the gold action has an element) exactly the gold element/option.

    python scripts/mind2web_table.py --records data/frozen/bench_v1_jev_answered.jsonl \
        --pred Jev=runs/jev_bench_v1.jsonl --pred Ours=runs_modal/cont5k/bench_pred.jsonl
"""

import argparse
import json
from collections import defaultdict

SPLITS = [("mind2web:test_website", "Cross-Website"), ("mind2web:test_domain", "Cross-Domain"),
          ("weblinx:test_web", "WebLINX test_web")]


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def element(key):
    return None if key is None else str(key).split(":")[0]


def score(record, answers):
    gold = record["gold"]
    op = (answers.get("operation") or {}).get("choice")
    gold_key = gold["operation"].lower() + "_target"
    chosen = (answers.get(op.lower() + "_target") or {}).get("choice") if op else None
    has_element = gold_key in gold
    ele = element(chosen) in {element(k) for k in gold[gold_key]} if has_element else None
    op_ok = op == gold["operation"]
    step = op_ok and (not has_element or chosen in gold[gold_key])
    return ele, op_ok, step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--pred", action="append", required=True, help="name=path")
    args = parser.parse_args()
    records = {str(r["meta"]["id"]): r for r in load(args.records)}
    results = {}
    for spec in args.pred:
        name, path = spec.split("=", 1)
        preds = {str(p["id"]): p.get("answers", {}) for p in load(path)}
        cells = defaultdict(lambda: {"ele": [], "op": [], "step": []})
        for rid, r in records.items():
            if rid not in preds:
                continue
            ele, op_ok, step = score(r, preds[rid])
            for part in (r["meta"].get("bench_part"), "all"):
                c = cells[part]
                if ele is not None:
                    c["ele"].append(ele)
                c["op"].append(op_ok)
                c["step"].append(step)
        results[name] = cells

    pct = lambda xs: f"{100 * sum(xs) / len(xs):.1f}" if xs else "-"
    head = "| |" + "".join(f" {label} | | |" for _, label in SPLITS + [("all", "All")])
    sub = "| |" + " Ele. Acc | Op. Acc | **Step SR** |" * (len(SPLITS) + 1)
    print(head + "\n" + sub + "\n|---|" + "---|" * 3 * (len(SPLITS) + 1))
    for name, cells in results.items():
        row = f"| {name} |"
        for part, _ in SPLITS + [("all", "All")]:
            c = cells[part]
            row += f" {pct(c['ele'])} | {pct(c['op'])} | **{pct(c['step'])}** |"
        print(row)
    print("\nsteps per split: " + ", ".join(
        f"{label} {len(next(iter(results.values()))[part]['step'])}" for part, label in SPLITS + [("all", "All")]))


if __name__ == "__main__":
    main()
