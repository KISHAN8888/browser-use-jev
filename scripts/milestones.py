"""Download a run's milestones.jsonl (+ train log) from the Modal volume and print the milestones table. Free: no GPU.

python scripts/milestones.py [--name q4b_full_ms] [--out runs_modal]   # -> runs_modal/<name>/milestones.jsonl
python scripts/milestones.py --local runs_modal/q4b_full_ms            # print an already-downloaded copy
Snapshots: modal volume get myjev-data runs/<name>/milestones/m10000 runs_modal/<name>/milestones/m10000
"""
import argparse
import json


def parse(line):
    """A line still being written mid-run may be incomplete; skip it instead of crashing."""
    try:
        return json.loads(line) if line.strip() else None
    except json.JSONDecodeError:
        return None
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modal_app import milestone_table  # noqa: E402


def fetch(volume_path, dest):
    """modal volume get <path> - (stdout); returns the text, or None if the file does not exist (yet)."""
    r = subprocess.run([sys.executable, "-m", "modal", "volume", "get", "myjev-data", volume_path, "-"],
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode:
        print(f"(no {volume_path} yet: {r.stderr.strip()[-200:]})")
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(r.stdout, encoding="utf-8")
    return r.stdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="q4b_full_ms")
    p.add_argument("--out", default="runs_modal")
    p.add_argument("--local", help="run dir already on disk; skip the download")
    args = p.parse_args()
    local = Path(args.local or Path(args.out) / args.name)
    if not args.local:
        fetch(f"runs/{args.name}/milestones.jsonl", local / "milestones.jsonl")
        fetch(f"runs/{args.name}/train_log.jsonl", local / "train_log.jsonl")
    ms = local / "milestones.jsonl"
    lines = [x for x in map(parse, ms.read_text(encoding="utf-8").splitlines()) if x] if ms.exists() else []
    print(milestone_table(lines) if lines else "no milestones yet")
    log = local / "train_log.jsonl"
    if log.exists():
        events = [x for x in map(parse, log.read_text(encoding="utf-8").splitlines()) if x]
        start = [e for e in events if e.get("event") == "start"]
        last = [e for e in events if e.get("event") == "step"]
        if last:
            total = start[-1]["steps"] if start else "?"
            e = last[-1]
            print(f"progress: step {e['step']}/{total} epoch {e['epoch']} loss {e['loss']} "
                  f"{e['s_per_step']} s/step at {e['time']} (container clock)")


if __name__ == "__main__":
    main()
