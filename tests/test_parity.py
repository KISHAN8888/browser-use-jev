"""build_request must equal the body jev_ultrafast.model.choose sends (upstream clone required)."""
import os, sys, json, pathlib, pytest

# Defaults to the pinned copy in tests/upstream; set JEV_ULTRAFAST_DIR to test against a fresh clone.
UPSTREAM = os.environ.get("JEV_ULTRAFAST_DIR", str(pathlib.Path(__file__).parent / "upstream"))

from myjev.jevformat import build_request, gold_answers

SNAPSHOT = {
    "url": "https://example.test/flights", "title": "Flights", "text": "From\nTo\nSearch",
    "actions": [
        {"id": "e1", "kind": "fill", "node": 1, "role": "combobox", "label": "Where from?", "value": "Zürich"},
        {"id": "e2", "kind": "click", "node": 1, "role": "combobox", "label": "Open Where from?", "value": "Zürich"},
        {"id": "e3", "kind": "fill", "node": 2, "role": "combobox", "label": "Where to?", "value": ""},
        {"id": "e4", "kind": "click", "node": 3, "role": "button", "label": "Search", "value": ""},
        {"id": "e5", "kind": "select", "node": 4, "role": "combobox", "label": "Cabin → Business", "value": "b", "current_value": "Economy"},
        {"id": "e6", "kind": "click", "node": 5, "role": "checkbox", "label": "Nonstop", "value": "on", "checked": "false"},
        {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560},
        {"id": "wait", "kind": "wait", "label": "Wait for the page to update"},
    ],
}
HISTORY = [{"action": "Where from?", "kind": "fill", "text": "Zürich", "page_changed": True, "extra": 1}]


def test_body_matches_upstream(monkeypatch):
    # Load only model.py; the package __init__ pulls in the browser harness we do not need.
    import types, importlib
    pkg = types.ModuleType("jev_ultrafast"); pkg.__path__ = [str(pathlib.Path(UPSTREAM) / "jev_ultrafast")]
    monkeypatch.setitem(sys.modules, "jev_ultrafast", pkg)
    model = importlib.import_module("jev_ultrafast.model")
    captured = {}
    def fake_post(url, key, body):
        captured["body"] = body
        return {"answers": {"operation": {"choice": "WAIT", "probabilities": {k: (1.0 if k == "WAIT" else 0.0) for k in body["questions"]["operation"]["criteria"]}, "confidence": 1.0}}, "model": "x"}
    monkeypatch.setattr(model, "post_json", fake_post)
    monkeypatch.setenv("TYPESAFE_API_KEY", "x")
    monkeypatch.delenv("TYPESAFE_MODEL", raising=False)
    model.choose(SNAPSHOT, "Fly to London", HISTORY)
    ours, _, _ = build_request(SNAPSHOT, "Fly to London", HISTORY)
    assert json.dumps(ours, sort_keys=True) == json.dumps(captured["body"], sort_keys=True)


def test_gold_mapping():
    _, targets, controls = build_request(SNAPSHOT, "g", [])
    assert gold_answers(["e3"], targets, controls) == {"operation": "TYPE_TEXT", "type_text_target": ["2"]}
    assert gold_answers(["e5"], targets, controls) == {"operation": "SELECT", "select_target": ["4:1"]}
    assert gold_answers(["e2", "e4"], targets, controls) == {"operation": "CLICK", "click_target": ["1", "3"]}
    assert gold_answers(["scroll_down"], targets, controls) == {"operation": "SCROLL_DOWN"}
    assert gold_answers(["DONE"], targets, controls) == {"operation": "DONE"}
    assert gold_answers(["e99"], targets, controls) is None
    assert gold_answers(["e1", "e4"], targets, controls) is None  # mixed operations
