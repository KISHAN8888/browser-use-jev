"""Turn a Jev request into token ids for the Choice scorer. Pure python; the tokenizer is passed in.

Layout (candidates last so they attend to everything):

    prefix   = [bos] goal + rules -> recent actions -> page url/title/text -> element table
    suffix_q = "Question: <name> (operation <OP>)" + one row per candidate, each row ending in MARKER

One (state, question) sequence is prefix + suffix_q. Every question of a request shares the same
prefix, so the model can run them all in one forward pass (block attention mask, see scorer.py);
that is mathematically identical to running each prefix + suffix_q on its own.

Budget: len(prefix) + max_q len(suffix_q) <= max_len. The budget is computed from ALL questions in
the request (not only the ones trained on), so train and inference sequences are identical.
Over budget we cut, in order: page text (tail first), element rows that are no question's candidate
(bottom first), remaining element rows, recent actions, then the per-row token cap of candidate rows.
Candidates (hence gold) are never dropped.

Marker token: we reuse an existing special token instead of adding a new one, so the embedding
matrix is never resized and the LoRA checkpoint stays adapter-only. For Qwen3 we use
`<|box_end|>` (a vision token that never occurs in text-only training data); otherwise the first
available of MARKER_CANDIDATES, falling back to eos. The marker's input embedding hardly matters:
the score is read from its *output* hidden state, which LoRA adapts.
"""

MARKER_CANDIDATES = ["<|box_end|>", "<|quad_end|>", "<|object_ref_end|>", "<|endoftext|>"]
ROW_CAP = 48  # max tokens per element / candidate row (long link texts are cut)


def marker_token(tok):
    vocab = tok.get_vocab()
    for t in MARKER_CANDIDATES:
        if t in vocab:
            return t
    return tok.eos_token


def _clip(s, n=200):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def element_row(e):
    row = f"[{e['index']}] {e.get('role', '')} '{_clip(e.get('label', ''))}'"
    if e.get("value"):
        row += f" value='{_clip(e['value'], 80)}'"
    for k in ("checked", "selected", "expanded"):
        if k in e:
            row += f" {k}={e[k]}"
    if e.get("operations"):
        row += " ops=" + ",".join(e["operations"])
    if e.get("options"):
        labels = [_clip(o["label"].split(" → ")[-1], 40) for o in e["options"][:10]]
        row += " options=[" + " | ".join(labels) + ("" if len(e["options"]) <= 10 else " | …") + "]"
    return row


def candidate_row(key, value):
    if not isinstance(value, dict):  # operation question: {"CLICK": "Click an element ..."}
        return f"{key}: {_clip(value)}"
    row = _clip(value.get("element", f"[{key}]"))
    if value.get("role"):
        row += f" ({value['role']})"
    if value.get("current_value"):
        row += f" value='{_clip(value['current_value'], 80)}'"
    for k, v in value.items():
        if k not in ("element", "role", "current_value"):
            row += f" {k}={_clip(v, 40)}"
    return row


def history_row(h):
    row = f"- {h.get('kind') or ''} '{_clip(h.get('action') or '', 120)}'"
    if h.get("text"):
        row += f" text='{_clip(h['text'], 80)}'"
    if h.get("page_changed") is not None:
        row += " (page changed)" if h["page_changed"] else " (page unchanged)"
    return row


def _rules(questions):
    """Unique rule strings across all questions, in order (NEXT_ACTION, then TARGET)."""
    seen = []
    for q in questions.values():
        rules = q.get("instructions", {}).get("rules", [])
        for r in [rules] if isinstance(rules, str) else rules:
            if r and r not in seen:
                seen.append(r)
    return seen


def _goal(questions):
    for q in questions.values():
        if q.get("instructions", {}).get("goal"):
            return q["instructions"]["goal"]
    return ""


def encode(request, tok, max_len=4096, include_rules=True, marker=None):
    """Return {"prefix": ids, "questions": {name: {"ids", "markers", "keys", "kind"}}, "stats": {...}}.

    `markers` are positions inside the question's suffix ids; `keys` are the criteria keys in order.
    """
    marker_id = tok.convert_tokens_to_ids(marker or marker_token(tok))
    state, questions = request["state"], request["questions"]
    enc = lambda texts: tok(texts, add_special_tokens=False)["input_ids"] if texts else []

    head = f"Goal: {_goal(questions)}\n"
    if include_rules:
        head += "\nRules:\n" + "\n\n".join(_rules(questions)) + "\n"
    page = state.get("page", {})
    hist = [history_row(h) for h in state.get("recent_actions", [])]
    elems = state.get("elements", [])

    head_ids = ([tok.bos_token_id] if tok.bos_token_id is not None else []) + enc([head])[0]
    hist_ids = enc(["\nRecent actions:"] + ["\n" + r for r in hist] if hist else ["\nRecent actions: none"])
    page_ids = enc([f"\nURL: {page.get('url', '')}\nTitle: {page.get('title', '')}\nPage text:\n"])[0]
    text_ids = enc([page.get("text", "")])[0] if page.get("text") else []
    elem_head = enc(["\n\nElements:"])[0]
    elem_ids = [ids[:ROW_CAP] for ids in enc(["\n" + element_row(e) for e in elems])]

    # Question suffixes: rows kept as separate id lists so the row cap can shrink later.
    qrows, candidate_elems = {}, set()
    for name, q in questions.items():
        crit = q.get("criteria", {})
        op = q.get("instructions", {}).get("operation")
        title = enc([f"\n\nQuestion: {name}" + (f" (operation {op})" if op else "") + "\nOptions:"])[0]
        rows = enc(["\n" + candidate_row(k, v) for k, v in crit.items()])
        qrows[name] = (title, rows, list(crit.keys()), "operation" if name == "operation" else "target")
        if name != "operation":
            candidate_elems.update(str(k).split(":")[0] for k in crit)

    def suffix_len(cap):
        return max((len(t) + sum(min(len(r), cap) + 1 for r in rows) for t, rows, _, _ in qrows.values()), default=0)

    cap = ROW_CAP
    keep = [True] * len(elems)
    fixed = lambda: len(head_ids) + sum(map(len, hist_ids)) + len(page_ids) + len(elem_head) + sum(
        len(ids) for ids, k in zip(elem_ids, keep) if k)
    over = fixed() + len(text_ids) + suffix_len(cap) - max_len
    stats = {"text_cut": 0, "rows_cut": 0, "hist_cut": 0, "cap": cap}
    if over > 0:  # 1. page text
        cut = min(over, len(text_ids))
        text_ids = text_ids[: len(text_ids) - cut]
        stats["text_cut"], over = cut, over - cut
    for only_non_candidates in (True, False):  # 2./3. element rows, bottom first
        for i in reversed(range(len(elems))):
            if over <= 0:
                break
            if keep[i] and (not only_non_candidates or str(elems[i]["index"]) not in candidate_elems):
                keep[i] = False
                over -= len(elem_ids[i])
                stats["rows_cut"] += 1
    while over > 0 and len(hist_ids) > 1:  # 4. history, oldest first
        over -= len(hist_ids.pop(1))
        stats["hist_cut"] += 1
    while over > 0 and cap > 4:  # 5. shrink candidate rows (never drop them)
        cap //= 2
        over = fixed() + len(text_ids) + suffix_len(cap) - max_len
    stats["cap"] = cap

    prefix = head_ids + sum(hist_ids, []) + page_ids + text_ids + elem_head
    prefix += sum((ids for ids, k in zip(elem_ids, keep) if k), [])
    out = {}
    for name, (title, rows, keys, kind) in qrows.items():
        ids, markers = list(title), []
        for r in rows:
            ids += r[:cap]
            markers.append(len(ids))
            ids.append(marker_id)
        out[name] = {"ids": ids, "markers": markers, "keys": keys, "kind": kind}
    stats["len"] = len(prefix) + max((len(q["ids"]) for q in out.values()), default=0)
    return {"prefix": prefix, "questions": out, "stats": stats}


def gold_targets(record):
    """{question: [gold keys]} for the questions we train on: operation + the gold op's target head."""
    gold = record["gold"]
    out = {"operation": [gold["operation"]]}
    tq = gold["operation"].lower() + "_target"
    if gold.get(tq) and tq in record["request"]["questions"]:
        out[tq] = list(gold[tq])
    return out
