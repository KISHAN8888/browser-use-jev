"""Build the exact TypeSafe request that browser-use/jev-ultrafast sends, plus gold labels.

Ported from jev_ultrafast/model.py (`action_space`, `choose`) at commit 1231850 (MIT, see
vendor/LICENSE-jev-ultrafast). The request body must stay byte-for-byte compatible with that code,
so do not "improve" anything here without checking upstream.

Input is a snapshot in snapshot.js shape:
    {"url", "title", "text", "actions": [{"id", "kind", "node", "role", "label", "value", ...}, ...]}
where kind is click | fill | select | scroll | wait. Every converter produces this shape.
"""

from .vendor.questions import NEXT_ACTION, TARGET

OPERATION_KINDS = {"click": "CLICK", "fill": "TYPE_TEXT", "select": "SELECT"}
OPERATION_LABELS = {
    "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
    "TYPE_TEXT": "Enter or replace text in an editable field. A small LLM will supply the value from the goal.",
    "SELECT": "Select an observed dropdown value.",
}
TERMINAL = {"DONE": "Every requirement is visibly satisfied.", "BLOCKED": "No supported operation can progress."}
MAX_ACTIONS = 250


def action_space(actions):
    """Verbatim port of jev_ultrafast.model.action_space."""
    elements, indices, targets, controls = [], {}, {}, {}
    for action in actions:
        kind = action["kind"]
        if kind not in OPERATION_KINDS:
            controls[action["id"].upper()] = action
            continue
        node = action["node"]
        if node not in indices:
            index = str(len(elements) + 1)
            indices[node] = index
            element = {k: action[k] for k in ("role", "value", "checked", "selected", "expanded") if k in action}
            element.update(index=index, label=action["label"].split(" → ")[0], operations=[])
            if kind == "select":
                element["value"] = action.get("current_value", "")
                element["options"] = []
            elements.append(element)
        index = indices[node]
        operation = OPERATION_KINDS[kind]
        group = targets.setdefault(operation, {})
        element = elements[int(index) - 1]
        if operation not in element["operations"]:
            element["operations"].append(operation)
        target = index
        if kind == "select":
            target = f"{index}:{len(element['options']) + 1}"
            element["options"].append({"index": target, "label": action["label"], "value": action["value"]})
        group[target] = action
    return elements, targets, controls


def build_request(snapshot, goal, history, model="jev-latest"):
    """The request body of jev_ultrafast.model.choose, without the HTTP call.

    Returns (body, targets, controls) so callers can map gold action ids to answers.
    """
    elements, targets, controls = action_space(snapshot["actions"])
    operations = {key: OPERATION_LABELS[key] for key in targets}
    operations.update({key: value["label"] for key, value in controls.items()})
    operations.update(TERMINAL)
    questions = {
        "operation": {"type": "choice", "criteria": operations, "instructions": {"goal": goal, "rules": NEXT_ACTION}}
    }
    for operation, candidates in targets.items():
        questions[operation.lower() + "_target"] = {
            "type": "choice",
            "criteria": {
                index: {
                    "element": f"[{index}] {a['label']}",
                    "current_value": a.get("current_value", a.get("value", "")),
                    **{k: a[k] for k in ("role", "checked", "selected", "expanded") if k in a},
                }
                for index, a in candidates.items()
            },
            "instructions": {"goal": goal, "operation": operation, "rules": [NEXT_ACTION, TARGET]},
        }
    body = {
        "model": model,
        "state": {
            "page": {k: snapshot[k] for k in ("url", "title", "text")},
            "elements": elements,
            "recent_actions": [
                {k: h.get(k) for k in ("action", "kind", "text", "page_changed")} for h in history[-10:]
            ],
        },
        "questions": questions,
    }
    return body, targets, controls


def gold_answers(gold_ids, targets, controls):
    """Map gold snapshot action ids (e.g. ["e12"], ["scroll_down"], ["DONE"]) to question answers.

    Several ids are allowed when elements are equivalent (multi-gold). All must agree on the
    operation. Returns {"operation": "CLICK", "click_target": ["7"]} or None if a gold id is
    not observable in this snapshot (the step must then be dropped).
    """
    operation, answers = None, []
    for gold in gold_ids:
        if gold in TERMINAL:
            op, target = gold, None
        elif gold.upper() in controls:
            op, target = gold.upper(), None
        else:
            hit = [(op, t) for op, group in targets.items() for t, a in group.items() if a["id"] == gold]
            if not hit:
                return None
            op, target = hit[0]
        if operation not in (None, op):
            return None
        operation = op
        if target is not None and target not in answers:
            answers.append(target)
    if operation is None:
        return None
    result = {"operation": operation}
    if answers:
        result[operation.lower() + "_target"] = answers
    return result


def make_record(snapshot, goal, history, gold_ids, meta):
    """One training/eval record: the exact Jev request plus gold answers and provenance."""
    body, targets, controls = build_request(snapshot, goal, history)
    gold = gold_answers(gold_ids, targets, controls)
    if gold is None:
        return None
    return {"request": body, "gold": gold, "meta": meta}
