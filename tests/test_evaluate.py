import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from test_parity import SNAPSHOT
from myjev.jevformat import make_record
from myjev.evaluate import evaluate, summarize
from myjev import baselines
import random


def records():
    golds = [["e3"], ["e4"], ["e5"], ["scroll_down"], ["DONE"], ["e2", "e4"]]
    return [make_record(SNAPSHOT, "Fly from Zürich to London", [], g, {"id": i, "source": "t"}) for i, g in enumerate(golds)]


def test_perfect_and_wrong_predictions():
    recs = records()
    perfect = []
    for r in recs:
        answers = {"operation": {"choice": r["gold"]["operation"], "probabilities": {r["gold"]["operation"]: 1.0}}}
        for k, v in r["gold"].items():
            if k != "operation":
                answers[k] = {"choice": v[-1], "probabilities": {v[-1]: 1.0}}
        perfect.append({"id": r["meta"]["id"], "answers": answers})
    s = summarize(evaluate(recs, perfect)[0])
    assert s["step_acc"] == 1 and s["operation_acc"] == 1 and s["target_acc"] == 1 and s["ece_operation"] == 0
    # Right operation, wrong target in the gold head -> step wrong.
    wrong = json.loads(json.dumps(perfect))
    wrong[0]["answers"]["type_text_target"] = {"choice": "1", "probabilities": {"1": 1.0}}
    s = summarize(evaluate(recs, wrong)[0])
    assert s["operation_acc"] == 1 and abs(s["step_acc"] - 5 / 6) < 1e-9


def test_baselines_produce_every_question():
    for r in records():
        for fn in baselines.BASELINES.values():
            answers = fn(r, "CLICK", random.Random(0))
            assert set(answers) == set(r["request"]["questions"])
            for a in answers.values():
                assert a["choice"] in a["probabilities"] and abs(sum(a["probabilities"].values()) - 1) < 1e-6


def answer(record, target=None):
    gold = record["gold"]
    answers = {"operation": {"choice": gold["operation"], "probabilities": {gold["operation"]: 1.0}}}
    head = gold["operation"].lower() + "_target"
    if head in gold:
        choice = target or gold[head][0]
        answers[head] = {"choice": choice, "probabilities": {choice: 1.0}}
    return {"id": record["meta"]["id"], "answers": answers}


def test_website_macro_average():
    # Site "big" has 3 steps (all right), site "small" 1 step (wrong) -> micro 3/4, macro (1 + 0) / 2.
    recs = [make_record(SNAPSHOT, "g", [], ["e4"], {"id": i, "site_key": "big" if i < 3 else "small", "source": "t"}) for i in range(4)]
    preds = [answer(r) for r in recs[:3]] + [answer(recs[3], "2")]
    s = summarize(evaluate(recs, preds)[0])
    assert s["step_acc"] == 0.75 and s["step_acc_macro"] == 0.5 and s["target_acc_macro"] == 0.5 and s["n_sites"] == 2


def test_lenient_target_accepts_identical_duplicate_control():
    snap = json.loads(json.dumps(SNAPSHOT))
    snap["actions"].append({"id": "e7", "kind": "click", "node": 6, "role": "link", "label": "Search", "value": ""})
    snap["actions"].append({"id": "e8", "kind": "click", "node": 7, "role": "link", "label": "", "value": ""})
    rec = make_record(snap, "g", [], ["e4"], {"id": 0, "source": "t"})
    criteria = rec["request"]["questions"]["click_target"]["criteria"]
    dup = next(k for k, c in criteria.items() if c["element"].endswith("Search") and k not in rec["gold"]["click_target"])
    other = next(k for k, c in criteria.items() if c["element"].endswith("Nonstop"))
    s = summarize(evaluate([rec], [answer(rec, dup)])[0])
    assert s["target_acc"] == 0 and s["target_acc_lenient"] == 1 and s["step_acc_lenient"] == 1
    s = summarize(evaluate([rec], [answer(rec, other)])[0])
    assert s["target_acc_lenient"] == 0
