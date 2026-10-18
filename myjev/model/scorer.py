"""Choice scorer: causal-LM body + tiny MLP head on each candidate's marker token.

score_k = MLP(h[marker_k]); softmax over the candidates of one question.

Packing: a request's questions share one prefix. In packed mode a row is
prefix + suffix_1 + suffix_2 + ..., with a 4D additive mask so each suffix sees only the prefix and
itself, and position ids restarting at len(prefix) for every suffix. This equals running
prefix + suffix_q separately per question (tested in tests/test_serialize.py) at a fraction of the
compute. Needs sdpa/eager attention; with flash_attention_2 use pack=False (one row per question).
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChoiceScorer(nn.Module):
    def __init__(self, backbone, hidden_size, head_dim=256):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Linear(hidden_size, head_dim), nn.GELU(), nn.Linear(head_dim, 1))
        nn.init.zeros_(self.head[2].weight)  # step 0: all scores equal -> uniform softmax
        nn.init.zeros_(self.head[2].bias)

    def forward(self, batch):
        h = self.backbone(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"], use_cache=False,
        ).last_hidden_state
        hm = h[batch["marker_row"], batch["marker_pos"]].float()  # (M, H); head runs in fp32
        scores = self.head(hm).squeeze(-1)  # (M,)
        logits = scores[batch["q_index"]]  # (Q, Kmax)
        return logits.masked_fill(~batch["q_mask"], float("-inf"))


def collate(items, pad_id, pack=True, dtype=torch.float32):
    """items: list of (encoded, [question names]) -> tensors. Returns batch + per-question meta."""
    rows = []  # each row: (prefix, [(name, q)])
    for enc, names in items:
        qs = [(n, enc["questions"][n]) for n in names]
        rows += [(enc["prefix"], qs)] if pack else [(enc["prefix"], [q]) for q in qs]
    L = max(len(p) + sum(len(q["ids"]) for _, q in qs) for p, qs in rows)
    B = len(rows)
    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    position_ids = torch.zeros((B, L), dtype=torch.long)
    allow = torch.zeros((B, L, L), dtype=torch.bool) if pack else None
    lengths = []
    m_row, m_pos, q_slices, meta = [], [], [], []
    for b, (prefix, qs) in enumerate(rows):
        P = len(prefix)
        ids = list(prefix)
        pos = list(range(P))
        if pack:
            allow[b, :P, :P] = torch.ones(P, P, dtype=torch.bool).tril()
        for name, q in qs:
            s, n = len(ids), len(q["ids"])
            ids += q["ids"]
            pos += range(P, P + n)
            if pack:
                allow[b, s : s + n, :P] = True
                allow[b, s : s + n, s : s + n] = torch.ones(n, n, dtype=torch.bool).tril()
            start = len(m_pos)
            m_row += [b] * len(q["markers"])
            m_pos += [s + m for m in q["markers"]]
            q_slices.append((start, len(m_pos)))
            meta.append({"name": name, "keys": q["keys"], "kind": q["kind"]})
        input_ids[b, : len(ids)] = torch.tensor(ids)
        position_ids[b, : len(ids)] = torch.tensor(pos)
        lengths.append(len(ids))
    if pack:
        idx = torch.arange(L)
        allow[:, idx, idx] = True  # padding rows attend to themselves (avoids NaN)
        mask = torch.zeros((B, 1, L, L), dtype=dtype).masked_fill(~allow[:, None], torch.finfo(dtype).min)
    else:
        mask = (torch.arange(L)[None] < torch.tensor(lengths)[:, None]).long()
    K = max(e - s for s, e in q_slices)
    q_index = torch.zeros((len(q_slices), K), dtype=torch.long)
    q_mask = torch.zeros((len(q_slices), K), dtype=torch.bool)
    for i, (s, e) in enumerate(q_slices):
        q_index[i, : e - s] = torch.arange(s, e)
        q_mask[i, : e - s] = True
    batch = {
        "input_ids": input_ids, "attention_mask": mask, "position_ids": position_ids,
        "marker_row": torch.tensor(m_row), "marker_pos": torch.tensor(m_pos),
        "q_index": q_index, "q_mask": q_mask,
    }
    return batch, meta


def soft_targets(meta, golds, K):
    """golds: list of gold key lists aligned with meta -> (Q, K) uniform mass over gold indices."""
    t = torch.zeros((len(meta), K))
    for i, (m, g) in enumerate(zip(meta, golds)):
        hits = [m["keys"].index(k) for k in g if k in m["keys"]]
        t[i, hits] = 1.0 / len(hits)
    return t


def loss_fn(logits, targets, brier_weight=0.5, weights=None):
    """Soft-target CE + brier_weight * Brier, weighted mean over questions (weights: (Q,), default 1).
    Returns (loss, ce_per_question)."""
    logp = F.log_softmax(logits, -1)
    valid = torch.isfinite(logits)
    ce = -(targets * logp.masked_fill(~valid, 0)).sum(-1)
    brier = ((logp.exp() - targets) ** 2).masked_fill(~valid, 0).sum(-1)
    w = torch.ones_like(ce) if weights is None else weights.to(ce)
    return (w * (ce + brier_weight * brier)).sum() / w.sum().clamp_min(1e-8), ce.detach()


def to_device(batch, device, dtype=None):
    out = {k: v.to(device) for k, v in batch.items()}
    if dtype is not None and out["attention_mask"].is_floating_point():
        out["attention_mask"] = out["attention_mask"].to(dtype)  # finfo(fp32).min -> -inf in bf16 is fine
    return out


def load_backbone(name, dtype, attn="sdpa"):
    from transformers import AutoModel
    return AutoModel.from_pretrained(name, dtype=dtype, attn_implementation=attn)


def save_checkpoint(model, tok, out_dir, config):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(out / "adapter")
    torch.save(model.head.state_dict(), out / "head.pt")
    tok.save_pretrained(out / "tokenizer")
    (out / "config.json").write_text(json.dumps(config, indent=2))


def load_checkpoint(ckpt, device="cpu", dtype=torch.bfloat16, attn="sdpa", merge=False):
    """merge=False by default: merge_and_unload in bf16 rounds the merged weights and shifts logits slightly, and the
    temperatures were fit on the unmerged model. Unmerged LoRA costs a little latency but scores exactly as calibrated."""
    from peft import PeftModel
    from transformers import AutoTokenizer
    ckpt = Path(ckpt)
    config = json.loads((ckpt / "config.json").read_text())
    tok = AutoTokenizer.from_pretrained(ckpt / "tokenizer")
    base = load_backbone(config["base_model"], dtype, attn)
    backbone = PeftModel.from_pretrained(base, ckpt / "adapter")
    if merge:
        backbone = backbone.merge_and_unload()
    model = ChoiceScorer(backbone, base.config.hidden_size, config.get("head_dim", 256))
    model.head.load_state_dict(torch.load(ckpt / "head.pt", map_location="cpu"))
    return model.to(device).eval(), tok, config
