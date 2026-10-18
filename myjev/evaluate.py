"""Score predictions (Jev, our model, or a baseline) against gold records, all with the same rules.

Prediction file: one JSON per line, {"id", "answers": {question: {"choice", "probabilities"}}, "latency_ms"}.
Records file: {"request", "gold", "meta"} per line; a record's id is meta["id"] if present, else its line index.

Step is correct when the operation is right and, if that operation has a target, the target chosen in the
gold operation's head is one of the gold indices. This mirrors jev-ultrafast: only the head of the
chosen operation is ever executed.

MAIN METRIC: `target_acc` = accuracy of the target chosen in the GOLD operation's head (steps whose gold operation
has a target). It is the fair headline number across sources: operation accuracy (and so step accuracy) can be
gamed by source shortcuts, e.g. Mind2Web never has SCROLL/WAIT/DONE gold, so recognising a Mind2Web page is enough
to rule those out. Every accuracy is reported micro (over steps) and website-macro (mean of per-website accuracies,
so one big site cannot dominate); a per-source / per-bench-part table is always printed.

    python -m myjev.evaluate --records bench.jsonl --pred jev.jsonl [--pred ours.jsonl ...] [--by meta.source]
"""

import argparse
import json
import math
import random
import re
from collections import defaultdict


def record_id(record, index):
    return str(record["meta"].get("id", index))


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def element_bucket(n):
    return "<50" if n < 50 else "50-149" if n < 150 else "150+"


def categories(record):
    meta, gold = record["meta"], record["gold"]
    n = len(record["request"]["state"]["elements"])
    cats = {"all": "all", "operation": gold["operation"], "elements": element_bucket(n)}
    for key in ("source", "bench_part", "split", "domain", "website", "site_key"):
        if key in meta:
            cats[key] = str(meta[key])
    for tag in meta.get("tags", []):
        cats["tag:" + tag] = "yes"
    return cats


def score_step(record, answers):
    """Returns per-question (correct, max_prob, gold_prob) items and the step result."""
    gold = record["gold"]
    items = []
    op = answers.get("operation") or {}
    op_probs = op.get("probabilities", {})
    op_ok = op.get("choice") == gold["operation"]
    items.append(("operation", op_ok, max(op_probs.values(), default=0.0), op_probs.get(gold["operation"], 0.0)))
    target_key = gold["operation"].lower() + "_target"
    target_ok = None
    if target_key in gold:
        target = answers.get(target_key) or {}
        probs = target.get("probabilities", {})
        target_ok = target.get("choice") in gold[target_key]
        gold_mass = sum(probs.get(t, 0.0) for t in gold[target_key])
        items.append(("target", target_ok, max(probs.values(), default=0.0), gold_mass))
    step_ok = op_ok and (target_ok is None or target_ok)
    return items, op_ok, target_ok, step_ok


def candidate_identity(candidate):
    """(normalised visible label, current_value) of a target candidate; None if the label is empty."""
    if not isinstance(candidate, dict):
        return None
    label = re.sub(r"^\[[^\]]*\]\s*", "", str(candidate.get("element", "")))
    label = re.sub(r"\s+", " ", label).strip().lower()
    return (label, str(candidate.get("current_value", ""))) if label else None


def lenient_target_ok(record, answers):
    """Secondary metric: the chosen target is also accepted when its normalised label and current value equal a gold
    candidate's (duplicate controls: checkbox + link for one facet, two identical buttons). None if no target."""
    gold = record["gold"]
    head = gold["operation"].lower() + "_target"
    if head not in gold:
        return None
    choice = (answers.get(head) or {}).get("choice")
    if choice in gold[head]:
        return True
    criteria = record["request"]["questions"].get(head, {}).get("criteria", {})
    ident = candidate_identity(criteria.get(choice))
    return ident is not None and any(candidate_identity(criteria.get(t)) == ident for t in gold[head])


def ece(pairs, bins=15):
    if not pairs:
        return float("nan")
    total, buckets = len(pairs), defaultdict(list)
    for conf, ok in pairs:
        buckets[min(int(conf * bins), bins - 1)].append((conf, ok))
    return sum(len(b) / total * abs(sum(c for c, _ in b) / len(b) - sum(o for _, o in b) / len(b)) for b in buckets.values())


def bootstrap_ci(values, n=1000, seed=0):
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def evaluate(records, predictions):
    by_id = {str(p["id"]): p for p in predictions}
    rows, missing = [], 0
    for i, record in enumerate(records):
        pred = by_id.get(record_id(record, i))
        if pred is None:
            missing += 1
            continue
        answers = pred.get("answers", {})
        items, op_ok, target_ok, step_ok = score_step(record, answers)
        lenient = lenient_target_ok(record, answers)
        lenient_step = op_ok and (lenient is None or lenient)
        rows.append((record, items, op_ok, target_ok, step_ok, pred.get("latency_ms"), lenient, lenient_step))
    return rows, missing


def site_of(record):
    meta = record["meta"]
    if meta.get("site_key"):
        return meta["site_key"]
    from .build_dataset import website
    return website(record)


def macro(rows, index):
    """Mean over websites of the per-website accuracy of row[index] (rows with None are skipped)."""
    per_site = defaultdict(list)
    for row in rows:
        if row[index] is not None:
            per_site[site_of(row[0])].append(row[index])
    accs = [sum(v) / len(v) for v in per_site.values()]
    return sum(accs) / len(accs) if accs else float("nan")


def _mean(values):
    return sum(values) / len(values) if values else float("nan")


def summarize(rows):
    steps = [r[4] for r in rows]
    ops = [r[2] for r in rows]
    targets = [r[3] for r in rows if r[3] is not None]
    calib = defaultdict(list)
    nll = defaultdict(list)
    for _, items, *_ in rows:
        for kind, ok, conf, gold_p in items:
            calib[kind].append((conf, ok))
            nll[kind].append(-math.log(max(gold_p, 1e-6)))
    latencies = sorted(r[5] for r in rows if r[5] is not None)
    lo, hi = bootstrap_ci([float(s) for s in steps])
    return {
        "n": len(rows),
        "step_acc": sum(steps) / len(steps) if steps else float("nan"),
        "step_ci95": [lo, hi],
        "operation_acc": sum(ops) / len(ops) if ops else float("nan"),
        "target_acc": sum(targets) / len(targets) if targets else float("nan"),
        "n_sites": len({site_of(r[0]) for r in rows}),
        "step_acc_macro": macro(rows, 4),
        "operation_acc_macro": macro(rows, 2),
        "target_acc_macro": macro(rows, 3),
        "target_acc_lenient": _mean([r[6] for r in rows if len(r) > 6 and r[6] is not None]),
        "target_acc_lenient_macro": macro([r for r in rows if len(r) > 6], 6),
        "step_acc_lenient": _mean([r[7] for r in rows if len(r) > 7]),
        "ece_operation": ece(calib["operation"]),
        "ece_target": ece(calib["target"]),
        "nll_operation": sum(nll["operation"]) / max(1, len(nll["operation"])),
        "nll_target": sum(nll["target"]) / max(1, len(nll["target"])),
        "latency_p50_ms": latencies[len(latencies) // 2] if latencies else None,
        "latency_p95_ms": latencies[int(len(latencies) * 0.95)] if latencies else None,
    }


PART_KEY = "part"  # bench_part when present, else source


def group_by(rows, key):
    groups = defaultdict(list)
    for row in rows:
        if key == "tags":  # one group per difficulty tag; a step can be in several
            for tag in row[0]["meta"].get("tags", []):
                groups[tag].append(row)
            continue
        cats = categories(row[0])
        value = cats.get("bench_part", cats.get("source")) if key == PART_KEY else cats.get(key)
        if value is not None:
            groups[value].append(row)
    return {value: summarize(group) for value, group in sorted(groups.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--pred", action="append", required=True, help="name=path or path")
    parser.add_argument("--by", action="append", default=[], help="operation, elements, source, website, tags")
    parser.add_argument("--json-out")
    args = parser.parse_args()
    records = load_jsonl(args.records)
    report = {}
    for spec in args.pred:
        name, path = spec.split("=", 1) if "=" in spec else (spec, spec)
        rows, missing = evaluate(records, load_jsonl(path))
        entry = {"overall": summarize(rows), "missing": missing}
        for key in [PART_KEY] + args.by:
            entry[key] = group_by(rows, key)
        report[name] = entry
    print_table(report, args.by)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)


def fmt(x, pct=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{100 * x:.1f}" if pct else f"{x:.3f}"


def print_table(report, keys):
    print("TARGET = target accuracy in the gold operation's head (main metric; operation accuracy can be gamed by")
    print("source shortcuts). micro = over steps, macro = mean over websites. TGTlen = lenient target accuracy (secondary:")
    print("also accepts a candidate with the same visible label + current value as a gold one, i.e. duplicate controls).")
    print(f"{'system':<22}{'n':>6}{'sites':>6}{'TARGET':>8}{'TGTmac':>8}{'step':>7}{'stepmac':>8}{'ci95':>12}{'op':>7}{'opmac':>7}{'TGTlen':>7}{'TLmac':>7}"
          f"{'ECEop':>7}{'ECEtg':>7}{'p50ms':>7}")
    for name, entry in report.items():
        s = entry["overall"]
        ci = f"{fmt(s['step_ci95'][0])}-{fmt(s['step_ci95'][1])}"
        print(f"{name:<22}{s['n']:>6}{s['n_sites']:>6}{fmt(s['target_acc']):>8}{fmt(s['target_acc_macro']):>8}{fmt(s['step_acc']):>7}"
              f"{fmt(s['step_acc_macro']):>8}{ci:>12}{fmt(s['operation_acc']):>7}{fmt(s['operation_acc_macro']):>7}"
              f"{fmt(s['target_acc_lenient']):>7}{fmt(s['target_acc_lenient_macro']):>7}"
              f"{fmt(s['ece_operation'], False):>7}{fmt(s['ece_target'], False):>7}{str(s['latency_p50_ms'] or '-'):>7}"
              + (f"  (missing {entry['missing']})" if entry["missing"] else ""))
    print("\nby source / bench part: TARGET micro/macro | step micro/macro (n steps, sites)")
    values = sorted({v for entry in report.values() for v in entry.get(PART_KEY, {})})
    for name, entry in report.items():
        print(f"  {name}")
        for value in values:
            s = entry.get(PART_KEY, {}).get(value)
            if s:
                print(f"    {value[:26]:<26} {fmt(s['target_acc']):>6}/{fmt(s['target_acc_macro']):<6} | "
                      f"{fmt(s['step_acc']):>6}/{fmt(s['step_acc_macro']):<6} ({s['n']}, {s['n_sites']})")
    for key in keys:
        print(f"\nstep accuracy by {key}")
        values = sorted({v for entry in report.values() for v in entry.get(key, {})})
        print(f"{'':<18}" + "".join(f"{name[:12]:>14}" for name in report))
        for value in values:
            cells = []
            for entry in report.values():
                s = entry.get(key, {}).get(value)
                cells.append(f"{fmt(s['step_acc'])} ({s['n']})" if s else "-")
            print(f"{value[:18]:<18}" + "".join(f"{c:>14}" for c in cells))


if __name__ == "__main__":
    main()
