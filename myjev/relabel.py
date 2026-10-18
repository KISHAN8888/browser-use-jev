"""Label-convention fixes for train/val/calib (never bench). Pure functions; build_dataset applies them.

Why: in jev-ultrafast TYPE_TEXT replaces a field's text directly (no prior click) and SELECT picks an option of a
native <select> directly. Source traces record "click the field, then type": the CLICK step's state and the TYPE
step's state are nearly identical (same URL in 94-100% of pairs, median element-set Jaccard 1.0 on 5 of 6 sources;
focus is not visible in the Jev state) but the gold flips CLICK -> TYPE_TEXT. That teaches CLICK on type steps.

Drop reasons (`plan_drops`):
- click_then_type:    CLICK on an editable element (offers TYPE_TEXT) immediately followed by TYPE_TEXT into the
                      same element. Repeated clicks on that field directly before it (CLICK X, CLICK X, TYPE X) are
                      dropped too (same reason).
- click_then_select:  CLICK on an element immediately followed by SELECT of an option of the same element. (Native
                      <select>s offer no CLICK in jev-ultrafast, so this is ~0 in practice; kept for completeness.)
- enter_on_field:     Mind2Web ENTER (press Enter in a field) is converted to CLICK on that field ("Open X"), which
                      does not submit anything in jev-ultrafast (no key presses; you CLICK Search). Unfixable label.
- type_already_filled: TYPE_TEXT into a field whose current value already equals the typed text (known from the
                      next step's history or Mind2Web's value); NEXT_ACTION/TARGET say not to do this.
Plus mild per-operation loss-weight multipliers (`op_weight`, OP_WEIGHT), multiplied into meta.weight in train.

Element identity across consecutive states (`same_element`): equal normalised label and equal role (labels are
accessible names and stay stable; indices shift between snapshots). Unlabelled elements match only on equal
index + role. "Immediately followed" (`adjacent`): the next recorded step of the same trajectory has order + 1, or
its last history entry is this very click ("Open <label>" or "<label>") -- WebLINX orders by chat turn_index, whose
gaps are utterances, not actions.
"""

import re

CLICK, TYPE, SELECT = "CLICK", "TYPE_TEXT", "SELECT"
DROPDOWN_ROLES = {"combobox", "listbox"}
OPTION_ROLES = {"option", "menuitem", "menuitemradio", "menuitemcheckbox", "treeitem"}
REASONS = ("click_then_type", "click_then_select", "enter_on_field", "type_already_filled")

# Loss-weight multipliers by gold operation. final_v2 train class mass (source weights) TYPE 13.5% / SELECT 2.1% vs
# bench 14.5% / 2.8%: priors already roughly match, so keep the boost mild (TYPE ~17%, SELECT ~3% of mass).
OP_WEIGHT = {"TYPE_TEXT": 1.25, "SELECT": 1.5}


def norm(text):
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def target_elements(record):
    """Elements targeted by the gold answer (a SELECT option "4:2" -> element "4"). [] for target-less ops."""
    gold = record["gold"]
    head = gold["operation"].lower() + "_target"
    by_index = {str(e.get("index")): e for e in record["request"]["state"]["elements"]}
    out = []
    for t in gold.get(head, []):
        e = by_index.get(str(t).split(":")[0])
        if e is not None:
            out.append({"index": str(e.get("index")), "label": norm(e.get("label")), "role": e.get("role"),
                        "value": norm(e.get("value")), "ops": list(e.get("operations", [])),
                        "expanded": e.get("expanded")})
    return out


def step_order(meta, fallback=None):
    """Position of a step in its trajectory: step_index (most sources), turn_index (WebLINX), else `fallback`."""
    for key in ("step_index", "turn_index"):
        if isinstance(meta.get(key), int):
            return meta[key]
    return fallback


def summarize(record, order):
    """Compact, JSON-able view of one step: everything the relabel decisions (and the audit) need."""
    history = record["request"]["state"].get("recent_actions") or []
    last = history[-1] if history else None
    raw_op = record["meta"].get("gold_op_raw") or {}
    return {"order": order, "op": record["gold"]["operation"], "targets": target_elements(record),
            "last_action": {"action": norm(last.get("action")), "kind": last.get("kind"), "text": last.get("text")}
            if last else None,
            "url": record["request"]["state"]["page"].get("url"),
            "enter": raw_op.get("original_op") == "ENTER",
            "typed": raw_op.get("value") if raw_op.get("op") == "TYPE" else None}


def same_element(a, b):
    if a["label"] and b["label"]:
        return a["label"] == b["label"] and (a.get("role") == b.get("role") or not a.get("role") or not b.get("role"))
    return not a["label"] and not b["label"] and a["index"] == b["index"] and a.get("role") == b.get("role")


def is_editable(element):
    return TYPE in element["ops"]


def is_dropdown(element):
    return element.get("role") in DROPDOWN_ROLES or element.get("expanded") is not None or SELECT in element["ops"]


def history_is_click_on(step, element):
    """Does `step`'s last history entry record a click on `element`?"""
    last = step.get("last_action")
    if not last or last.get("kind") != "click" or not element["label"]:
        return False
    return last["action"] in (element["label"], "open " + element["label"])


def adjacent(cur, nxt, click_element=None):
    if isinstance(cur.get("order"), int) and isinstance(nxt.get("order"), int) and nxt["order"] - cur["order"] == 1:
        return True
    return click_element is not None and history_is_click_on(nxt, click_element)


def typed_text(cur, nxt):
    """Text typed at TYPE step `cur`: the source's own value, else the next step's last history entry."""
    if cur.get("typed") is not None:
        return cur["typed"]
    last = (nxt or {}).get("last_action")
    if not last or last.get("kind") != "fill":
        return None
    same_label = bool(cur["targets"]) and last["action"] == cur["targets"][0]["label"]
    return last.get("text") if same_label or adjacent(cur, nxt) else None


def classify_pair(cur, nxt):
    """Reason to drop CLICK step `cur` given the next step `nxt` of the same trajectory, or None."""
    if cur["op"] != CLICK or nxt is None or nxt["op"] not in (TYPE, SELECT):
        return None
    for c in cur["targets"]:
        if not adjacent(cur, nxt, c):
            continue
        for t in nxt["targets"]:
            if not same_element(c, t):
                continue
            if nxt["op"] == TYPE and is_editable(c):
                return "click_then_type"
            if nxt["op"] == SELECT:
                return "click_then_select"
    return None


def classify_single(cur, nxt):
    """Reasons that need only the step itself (and the next step for the typed text)."""
    if cur["op"] == CLICK and cur.get("enter") and any(is_editable(t) for t in cur["targets"]):
        return "enter_on_field"
    if cur["op"] == TYPE and cur["targets"]:
        text = typed_text(cur, nxt)
        value = cur["targets"][0]["value"]
        if text is not None and value and norm(text) == value:
            return "type_already_filled"
    return None


def plan_drops(steps):
    """steps: summaries of one trajectory (any order). Returns {position in the input list: reason}."""
    order = sorted(range(len(steps)), key=lambda i: (steps[i]["order"] is None, steps[i]["order"]))
    drops = {}
    for k, a in enumerate(order):
        nxt = steps[order[k + 1]] if k + 1 < len(order) else None
        reason = classify_single(steps[a], nxt) or classify_pair(steps[a], nxt)
        if reason:
            drops[a] = reason
    # CLICK X, CLICK X, TYPE X: earlier clicks on the same field right before a dropped click_then_type go too.
    for k in range(len(order) - 2, -1, -1):
        a, b = order[k], order[k + 1]
        if a in drops or drops.get(b) != "click_then_type" or steps[a]["op"] != CLICK:
            continue
        if any(is_editable(x) and same_element(x, y) and adjacent(steps[a], steps[b], x)
               for x in steps[a]["targets"] for y in steps[b]["targets"]):
            drops[a] = "click_then_type"
    return drops


def op_weight(op, table=None):
    return (OP_WEIGHT if table is None else table).get(op, 1.0)


def parse_op_weights(text):
    """'TYPE_TEXT=1.5,SELECT=2' -> {'TYPE_TEXT': 1.5, 'SELECT': 2.0}."""
    out = {}
    for part in filter(None, (p.strip() for p in str(text or "").split(","))):
        key, _, value = part.partition("=")
        out[key.strip().upper()] = float(value)
    return out
