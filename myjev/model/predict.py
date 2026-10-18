"""Answer every Choice question of every request with a trained checkpoint.

python -m myjev.model.predict --ckpt runs/q17b_10k/best --data data/benchmark.jsonl --out runs/q17b_10k/pred.jsonl

Output, one line per record:
  {"id": meta.id or line index,
   "answers": {"operation": {"choice": str, "probabilities": {opt: p}}, "<op>_target": {...}, ...},
   "latency_ms": float}
latency_ms = encode + forward + softmax for that record (batch size 1 by default, like a live agent).
"""

import argparse
import json
import math
import time

import torch

from .scorer import collate, load_checkpoint, to_device
from .serialize import encode


class Predictor:
    def __init__(self, ckpt, device=None, attn="sdpa"):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model, self.tok, self.config = load_checkpoint(ckpt, self.device, self.dtype, attn)
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.pack = attn != "flash_attention_2"

    @classmethod
    def from_model(cls, model, tok, config, device, dtype, pack=True):
        """Wrap an already-loaded (e.g. in-training) ChoiceScorer; config needs max_len, marker, temperatures.
        The caller is responsible for model.eval() / model.train()."""
        self = cls.__new__(cls)
        self.model, self.tok, self.config, self.device, self.dtype, self.pack = model, tok, config, device, dtype, pack
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        return self

    @torch.no_grad()
    def answer(self, requests):
        """List of Jev request bodies -> list of {question: {"choice", "probabilities", "confidence"}}."""
        c = self.config
        encs = [encode(r, self.tok, c["max_len"], c.get("include_rules", True), c["marker"]) for r in requests]
        batch, meta = collate([(e, list(e["questions"])) for e in encs], self.pad_id, self.pack, self.dtype)
        logits = self.model(to_device(batch, self.device, self.dtype)).float().cpu()
        out, i = [], 0
        for e in encs:
            answers = {}
            for _ in e["questions"]:
                m = meta[i]
                t = c["temperatures"].get(m["kind"], 1.0)
                probs = torch.softmax(logits[i, : len(m["keys"])] / t, -1).tolist()
                k = max(range(len(probs)), key=probs.__getitem__)
                answers[m["name"]] = {"choice": m["keys"][k], "probabilities": dict(zip(m["keys"], probs)),
                                      "confidence": probs[k]}
                i += 1
            out.append(answers)
        return out


def predict_records(pred, records, f, batch_size=1, start_index=0):
    """Answer `records` in chunks of batch_size and write prediction lines to file object f.
    Record ids: meta.id, else start_index + position. Returns (joint_correct, n_with_gold)."""
    correct = n = 0
    for s in range(0, len(records), batch_size):
        chunk = records[s : s + batch_size]
        t = time.perf_counter()
        answers = pred.answer([r["request"] for r in chunk])
        ms = (time.perf_counter() - t) * 1000 / len(chunk)
        for j, (r, a) in enumerate(zip(chunk, answers)):
            for v in a.values():
                v.pop("confidence")
            rid = r.get("meta", {}).get("id", start_index + s + j)
            f.write(json.dumps({"id": rid, "answers": a, "latency_ms": round(ms, 2)}, ensure_ascii=False) + "\n")
            g = r.get("gold")
            if g:  # quick joint accuracy: operation and (if any) target both right
                op = a["operation"]["choice"]
                tq = op.lower() + "_target"
                ok = op == g["operation"] and (tq not in g or a.get(tq, {}).get("choice") in g[tq])
                correct += ok
                n += 1
    return correct, n


def validate(pred_path, data_path, limit=None):
    """Check predictions against the first `limit` requests: every question answered, keys match, probs sum to 1."""
    data = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()][:limit]
    preds = [json.loads(l) for l in open(pred_path, encoding="utf-8") if l.strip()]
    assert len(preds) == len(data), f"{len(preds)} predictions for {len(data)} records"
    for i, (p, r) in enumerate(zip(preds, data)):
        assert p["id"] == r.get("meta", {}).get("id", i), (p["id"], i)
        assert isinstance(p["latency_ms"], float)
        qs = r["request"]["questions"]
        assert set(p["answers"]) == set(qs), (i, set(p["answers"]), set(qs))
        for q, a in p["answers"].items():
            assert set(a) == {"choice", "probabilities"}, a.keys()
            assert set(a["probabilities"]) == set(qs[q]["criteria"]), (i, q)
            assert math.isclose(sum(a["probabilities"].values()), 1.0, abs_tol=1e-4)
            assert a["choice"] == max(a["probabilities"], key=a["probabilities"].get)
    return len(preds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--device")
    p.add_argument("--limit", type=int, help="first N records only (sanity runs)")
    args = p.parse_args()

    pred = Predictor(args.ckpt, args.device, args.attn)
    records = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()][: args.limit]
    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as f:
        correct, n = predict_records(pred, records, f, args.batch_size)
    print(json.dumps({"records": len(records), "joint_acc": round(correct / n, 4) if n else None,
                      "seconds": round(time.time() - t0, 1), "validated": validate(args.out, args.data, args.limit)}))


if __name__ == "__main__":
    main()
