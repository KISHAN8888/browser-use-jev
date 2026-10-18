"""No-learning baselines, written in the evaluator's prediction format.

- majority: the most common operation in a train file; uniform over target candidates.
- random:   uniform over operations and over target candidates (seeded).
- overlap:  a hand rule. Score each target candidate by word overlap between its label and the goal.
            If the goal has a quoted/plain value not yet present in any filled field, prefer TYPE_TEXT
            into the best-overlapping empty field; otherwise CLICK the best-overlapping element.

    python -m myjev.baselines --records bench.jsonl --train train.jsonl --out-dir runs/baselines
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from .evaluate import load_jsonl, record_id

WORD = re.compile(r"[a-z0-9]+")
STOP = set("the a an to of in on for and or with from by at is be my me i you your this that it as".split())


def words(text):
    return {w for w in WORD.findall(str(text).lower()) if w not in STOP}


def uniform(options, choice=None):
    options = list(options)
    probs = {o: 1 / len(options) for o in options}
    return {"choice": choice if choice is not None else options[0], "probabilities": probs}


def one_hot(options, choice, mass=0.9):
    options = list(options)
    if len(options) == 1:
        mass = 1.0
    rest = (1 - mass) / max(1, len(options) - 1)
    return {"choice": choice, "probabilities": {o: (mass if o == choice else rest) for o in options}}


def target_questions(request):
    return {name: q for name, q in request["questions"].items() if name != "operation"}


def majority(record, top_op, rng):
    ops = record["request"]["questions"]["operation"]["criteria"]
    choice = top_op if top_op in ops else next(iter(ops))
    answers = {"operation": one_hot(ops, choice)}
    for name, q in target_questions(record["request"]).items():
        answers[name] = uniform(q["criteria"], rng.choice(list(q["criteria"])))
    return answers


def random_baseline(record, _top_op, rng):
    ops = record["request"]["questions"]["operation"]["criteria"]
    answers = {"operation": uniform(ops, rng.choice(list(ops)))}
    for name, q in target_questions(record["request"]).items():
        answers[name] = uniform(q["criteria"], rng.choice(list(q["criteria"])))
    return answers


def overlap(record, _top_op, rng):
    request = record["request"]
    goal = words(request["questions"]["operation"]["instructions"]["goal"])
    ops = request["questions"]["operation"]["criteria"]
    answers, best = {}, {}
    for name, q in target_questions(request).items():
        scores = {i: len(goal & words(c.get("element", ""))) + 0.01 * rng.random() for i, c in q["criteria"].items()}
        pick = max(scores, key=scores.get)
        answers[name] = one_hot(q["criteria"], pick, mass=0.6)
        best[name] = (scores[pick], q["criteria"][pick])
    filled = " ".join(str(e.get("value", "")) for e in request["state"]["elements"]).lower()
    unfilled_goal_words = {w for w in goal if w not in filled}
    type_best = best.get("type_text_target")
    if type_best and type_best[0] >= 1 and not type_best[1].get("current_value") and unfilled_goal_words:
        op = "TYPE_TEXT"
    elif "CLICK" in ops:
        op = "CLICK"
    else:
        op = next(iter(ops))
    answers["operation"] = one_hot(ops, op, mass=0.6)
    return answers


BASELINES = {"majority": majority, "random": random_baseline, "overlap": overlap}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--train", help="train file for the majority operation")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    records = load_jsonl(args.records)
    if args.train:  # streamed: the train file can be several GB in memory
        with open(args.train, encoding="utf-8") as f:
            ops = Counter(json.loads(line)["gold"]["operation"] for line in f if line.strip())
    else:
        ops = Counter(r["gold"]["operation"] for r in records)
    top_op = ops.most_common(1)[0][0]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in BASELINES.items():
        rng = random.Random(args.seed)
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for i, record in enumerate(records):
                f.write(json.dumps({"id": record_id(record, i), "answers": fn(record, top_op, rng), "latency_ms": 0}) + "\n")
    print(f"wrote {', '.join(BASELINES)} to {out_dir} (majority operation: {top_op})")


if __name__ == "__main__":
    main()
