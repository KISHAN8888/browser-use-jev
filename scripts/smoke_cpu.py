"""CPU end-to-end smoke test: 8 synthetic records -> train 2 steps -> calibrate -> predict -> validate.

python scripts/smoke_cpu.py [--model HuggingFaceTB/SmolLM2-135M] [--out runs/smoke]
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

from myjev.jevformat import make_record
from myjev.model.predict import validate


def snapshot(rng, i):
    actions = [
        {"id": "e1", "kind": "fill", "node": 1, "role": "combobox", "label": "Where from?", "value": rng.choice(["", "Zürich"])},
        {"id": "e2", "kind": "click", "node": 1, "role": "combobox", "label": "Open Where from?", "value": ""},
        {"id": "e3", "kind": "fill", "node": 2, "role": "combobox", "label": "Where to?", "value": ""},
        {"id": "e4", "kind": "click", "node": 3, "role": "button", "label": "Search", "value": ""},
        {"id": "e5", "kind": "select", "node": 4, "role": "combobox", "label": "Cabin → Business", "value": "b", "current_value": "Economy"},
        {"id": "e6", "kind": "select", "node": 4, "role": "combobox", "label": "Cabin → First", "value": "f", "current_value": "Economy"},
        {"id": "e7", "kind": "click", "node": 5, "role": "checkbox", "label": "Nonstop", "value": "on", "checked": "false"},
        {"id": "e8", "kind": "click", "node": 6, "role": "link", "label": "Search", "value": ""},  # equivalent to e4
        {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560},
        {"id": "wait", "kind": "wait", "label": "Wait for the page to update"},
    ]
    return {"url": f"https://example.test/flights/{i}", "title": "Flights", "text": "From\nTo\nSearch\n" * 20, "actions": actions}


GOLDS = [["e3"], ["e4", "e8"], ["e5"], ["e7"], ["scroll_down"], ["DONE"], ["e2"], ["wait"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--out", default="runs/smoke")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    records = []
    for i, gold in enumerate(GOLDS):
        hist = [{"action": "Where from?", "kind": "fill", "text": "Zürich", "page_changed": True}] if i % 2 else []
        r = make_record(snapshot(rng, i), "Find business flights from Zürich to London", hist, gold, {"id": f"smoke-{i}"})
        assert r is not None, gold
        records.append(r)
    data = out / "records.jsonl"
    data.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")

    py = sys.executable
    run = lambda *a: subprocess.run([py, "-m", *a], check=True)
    run("myjev.model.train", "--train", str(data), "--val", str(data), "--out", str(out / "ckpt"),
        "--model", args.model, "--max-len", "1024", "--micro-bs", "2", "--batch-size", "4", "--max-steps", "2", "--fp32")
    run("myjev.model.predict", "--ckpt", str(out / "ckpt" / "best"), "--data", str(data), "--out", str(out / "pred.jsonl"))
    n = validate(out / "pred.jsonl", data)
    cfg = json.loads((out / "ckpt" / "best" / "config.json").read_text())
    print(f"SMOKE OK: {n} predictions validated; temperatures={cfg['temperatures']}")
    print("first prediction:", (out / "pred.jsonl").read_text(encoding="utf-8").splitlines()[0][:400])


if __name__ == "__main__":
    main()
