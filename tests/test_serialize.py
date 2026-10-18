"""Serialization tests: one marker per candidate, gold never truncated, ordering, packing == separate rows."""
import copy
import pytest

from myjev.jevformat import make_record
from myjev.model.serialize import encode, gold_targets


class FakeTok:
    """Whitespace-ish tokenizer: one id per character chunk of 4. Enough to test budgets and ordering."""
    bos_token_id = None
    eos_token = "<eos>"

    def __init__(self):
        self.vocab = {"<eos>": 0, "<|box_end|>": 1}
        self.inv = {0: "<eos>", 1: "<|box_end|>"}

    def get_vocab(self):
        return self.vocab

    def convert_tokens_to_ids(self, t):
        return self.vocab[t]

    def _id(self, piece):
        if piece not in self.vocab:
            self.vocab[piece] = len(self.vocab)
            self.inv[self.vocab[piece]] = piece
        return self.vocab[piece]

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[self._id(t[i : i + 4]) for i in range(0, len(t), 4)] for t in texts]}

    def decode(self, ids):
        return "".join(self.inv[i] for i in ids)


def snapshot(n_links=30, text_len=4000):
    actions = [
        {"id": "e1", "kind": "fill", "node": 1, "role": "combobox", "label": "Where from?", "value": "Zürich"},
        {"id": "e2", "kind": "click", "node": 1, "role": "combobox", "label": "Open Where from?", "value": "Zürich"},
        {"id": "e3", "kind": "select", "node": 2, "role": "combobox", "label": "Cabin → Business", "value": "b", "current_value": "Economy"},
    ]
    actions += [{"id": f"l{i}", "kind": "click", "node": 10 + i, "role": "link", "label": f"Result link number {i}", "value": ""}
                for i in range(n_links)]
    actions += [{"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560},
                {"id": "wait", "kind": "wait", "label": "Wait for the page to update"}]
    return {"url": "https://x.test/", "title": "X", "text": "lorem ipsum " * (text_len // 12), "actions": actions}


HIST = [{"action": "Where from?", "kind": "fill", "text": "Zürich", "page_changed": True}] * 5


def rec(gold_ids, **kw):
    r = make_record(snapshot(**kw), "Fly to London in business", HIST, gold_ids, {"id": "t"})
    assert r is not None
    return r


def test_markers_match_candidates():
    tok = FakeTok()
    r = rec(["l7"])
    enc = encode(r["request"], tok, max_len=100000)
    assert set(enc["questions"]) == set(r["request"]["questions"])
    for name, q in enc["questions"].items():
        crit = r["request"]["questions"][name]["criteria"]
        assert q["keys"] == list(crit)
        assert len(q["markers"]) == len(crit) == q["ids"].count(1)
        assert all(q["ids"][m] == 1 for m in q["markers"])
    assert 1 not in enc["prefix"]
    assert enc["stats"] == {"text_cut": 0, "rows_cut": 0, "hist_cut": 0, "cap": 48, "len": enc["stats"]["len"]}


def test_ordering():
    tok = FakeTok()
    r = rec(["l7"])
    enc = encode(r["request"], tok, max_len=100000)
    prefix = tok.decode(enc["prefix"])
    order = [prefix.index(s) for s in ("Goal: Fly", "Rules:", "Recent actions:", "URL: https://x.test/", "Page text:", "Elements:", "[32] link")]
    assert order == sorted(order)
    suffix = tok.decode(enc["questions"]["click_target"]["ids"])
    assert suffix.startswith("\n\nQuestion: click_target (operation CLICK)")
    assert suffix.index("[1] Open Where from?") < suffix.index("[10] Result link number 7")
    op = tok.decode(enc["questions"]["operation"]["ids"])
    assert "SCROLL_DOWN: Scroll down" in op and "DONE:" in op


@pytest.mark.parametrize("max_len", [3000, 1500, 900, 400])
def test_truncation_keeps_all_candidates_and_gold(max_len):
    tok = FakeTok()
    r = rec(["l29"], n_links=60, text_len=8000)
    full = encode(r["request"], tok, max_len=100000)
    enc = encode(r["request"], tok, max_len=max_len)
    assert enc["stats"]["text_cut"] > 0
    for name, q in enc["questions"].items():
        assert q["keys"] == full["questions"][name]["keys"]
        assert len(q["markers"]) == len(q["keys"])
    gold = gold_targets(r)
    assert gold["click_target"] == ["32"] and "32" in enc["questions"]["click_target"]["keys"]
    if enc["stats"]["cap"] == 48:  # unless rows had to shrink, the budget holds exactly
        assert enc["stats"]["len"] <= max_len
    # page text is cut before any element row
    if enc["stats"]["rows_cut"]:
        assert "Page text:\n\n\nElements:" in tok.decode(enc["prefix"])


def test_truncation_order_text_then_rows():
    tok = FakeTok()
    r = rec(["e2"], n_links=60, text_len=8000)
    full = encode(r["request"], tok, max_len=100000)
    small = encode(r["request"], tok, max_len=full["stats"]["len"] - 50)
    assert small["stats"]["text_cut"] >= 50 and small["stats"]["rows_cut"] == 0


def test_packed_equals_separate():
    """Shared-prefix packing must give the same logits as one row per question (real tiny model)."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from myjev.model.scorer import ChoiceScorer, collate
    name = "HuggingFaceTB/SmolLM2-135M"
    try:
        tok = transformers.AutoTokenizer.from_pretrained(name)
        cfg = transformers.AutoConfig.from_pretrained(name)
    except Exception as e:  # offline
        pytest.skip(str(e))
    cfg.num_hidden_layers = 2
    torch.manual_seed(0)
    body = transformers.AutoModel.from_config(cfg, attn_implementation="sdpa", dtype=torch.float32).eval()
    model = ChoiceScorer(body, cfg.hidden_size)
    torch.nn.init.normal_(model.head[2].weight)
    encs = [encode(rec(["l3"], n_links=n)["request"], tok, max_len=1024) for n in (5, 12)]
    items = [(e, list(e["questions"])) for e in encs]
    with torch.no_grad():
        bp, mp = collate(items, tok.eos_token_id, pack=True)
        bs, ms = collate(items, tok.eos_token_id, pack=False)
        lp, ls = model(bp), model(bs)
    assert [m["name"] for m in mp] == [m["name"] for m in ms]
    assert torch.allclose(lp, ls, atol=1e-4), (lp - ls).abs().nan_to_num(0).max()
