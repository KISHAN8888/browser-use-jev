"""Secondary step accuracy that also accepts TYPE_TEXT into the exact field a gold CLICK points at.

Some human recordings label "click the text field" (then type) as CLICK. In jev-ultrafast, TYPE_TEXT replaces a
field's text directly, so typing into that same field is an equally valid next step. The strict step accuracy in
myjev.evaluate stays the headline; this is reported next to it, applied identically to every system.

    python scripts/typefield_score.py --records data/frozen/bench_v1_jev_answered.jsonl \
        --pred jev=runs/jev_bench_v1.jsonl --pred ours=runs_modal/m22320/bench_pred.jsonl
"""

import argparse
import json
from collections import defaultdict


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def step_ok(record, answers, accept_type_into_clicked_field):
    gold = record["gold"]
    op = (answers.get("operation") or {}).get("choice")
    key = gold["operation"].lower() + "_target"
    if op == gold["operation"]:
        return key not in gold or (answers.get(key) or {}).get("choice") in gold[key]
    if accept_type_into_clicked_field and gold["operation"] == "CLICK" and op == "TYPE_TEXT":
        typed = (answers.get("type_text_target") or {}).get("choice")
        editable = record["request"]["questions"].get("type_text_target", {}).get("criteria", {})
        return typed in gold.get("click_target", []) and typed in editable
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--pred", action="append", required=True, help="name=path")
    args = parser.parse_args()
    records = {str(r["meta"]["id"]): r for r in load(args.records)}
    print(f"{'system':<14}{'n':>6}{'strict':>9}{'+typefield':>12}{'changed':>9}   by part (strict -> +typefield)")
    for spec in args.pred:
        name, path = spec.split("=", 1)
        preds = {str(p["id"]): p.get("answers", {}) for p in load(path)}
        ids = [i for i in records if i in preds]
        parts = defaultdict(lambda: [0, 0, 0])
        strict = lenient = 0
        for i in ids:
            r = records[i]
            s, l = step_ok(r, preds[i], False), step_ok(r, preds[i], True)
            strict += s
            lenient += l
            p = parts[r["meta"].get("bench_part", r["meta"].get("source"))]
            p[0] += 1
            p[1] += s
            p[2] += l
        by_part = "  ".join(f"{k}: {100 * v[1] / v[0]:.1f}->{100 * v[2] / v[0]:.1f}" for k, v in sorted(parts.items()))
        n = len(ids)
        print(f"{name:<14}{n:>6}{100 * strict / n:>9.1f}{100 * lenient / n:>12.1f}{lenient - strict:>9}   {by_part}")


if __name__ == "__main__":
    main()
