"""backend_node_id mapping of the Mind2Web converter on a handmade page (no network, no dataset)."""
import asyncio
import json

import pytest

from myjev.convert import mind2web as m

HTML = """<html backend_node_id="1"><body backend_node_id="2">
  <iframe backend_node_id="3" title="ad"/>
  <a backend_node_id="10"><span backend_node_id="11"><text backend_node_id="12">Flights</text></span></a>
  <button backend_node_id="20" aria_label="Search now"><text backend_node_id="21">Go</text></button>
  <label backend_node_id="30"><input backend_node_id="31" type="text" placeholder="From" input_value="Zurich"/></label>
  <select backend_node_id="40"><option backend_node_id="41" value="e" option_selected="true">Economy</option>
    <option backend_node_id="42" value="b">Business</option></select>
  <div backend_node_id="50"><text backend_node_id="51">Andorra</text></div>
  <div backend_node_id="60"><a backend_node_id="61">One</a><a backend_node_id="62">Two</a></div>
</body></html>"""


def row(bnids, op="CLICK", value="", step=1):
    cands = [json.dumps({"tag": "x", "backend_node_id": b, "is_original_target": i == 0}) for i, b in enumerate(bnids)]
    return {"action_uid": "u", "raw_html": HTML, "operation": json.dumps({"original_op": op, "value": value, "op": op}),
            "pos_candidates": cands, "website": "example", "domain": "Travel", "subdomain": "Airlines",
            "annotation_id": "a", "confirmed_task": "Fly somewhere", "target_action_index": str(step),
            "action_reprs": ["[input]   -> TYPE: Zurich", "[span]  Flights -> CLICK", "[select]  -> SELECT: Business"],
            "target_action_reprs": "x"}


def convert(*rows):
    """Run the real async converter (headless Chromium, offline) on handmade rows, in input order."""
    pytest.importorskip("playwright.async_api")
    out = {}
    asyncio.run(m.convert_all(list(rows), "t", lambda i, r: out.__setitem__(i, r), concurrency=2))
    return [out[i] for i in range(len(rows))]


def answer(rec):
    g = rec["gold"]
    targets = g[g["operation"].lower() + "_target"]
    crit = rec["request"]["questions"][g["operation"].lower() + "_target"]["criteria"]
    return g["operation"], [crit[t]["element"] for t in targets]


def test_exact_ancestor_descendant_select():
    results = iter(convert(row(["20"]), row(["11"]), row(["30"], "TYPE", "Zurich"),
                           row(["40"], "SELECT", "Business"), row(["11", "10"])))
    rec, reason, _ = next(results)
    assert reason == "kept" and answer(rec) == ("CLICK", ["[2] Search now"]) and rec["meta"]["gold_match"] == "exact"

    rec, reason, _ = next(results)  # span inside <a> (no href in dump)
    assert reason == "kept" and answer(rec) == ("CLICK", ["[1] Flights"]) and rec["meta"]["gold_via_ancestor"]

    rec, reason, _ = next(results)  # label wrapping the input
    assert reason == "kept" and answer(rec) == ("TYPE_TEXT", ["[3] From"]) and rec["meta"]["gold_match"] == "descendant"
    assert rec["request"]["state"]["elements"][2]["value"] == "Zurich"

    rec, reason, _ = next(results)
    assert reason == "kept" and answer(rec) == ("SELECT", ["[4:1] Economy Business → Business"])

    rec, reason, _ = next(results)  # nested candidates = same control
    assert reason == "kept" and answer(rec) == ("CLICK", ["[1] Flights"])


def test_drops():
    empty = row(["20"])
    empty["pos_candidates"] = []
    reasons = [r for _, r, _ in convert(row(["50"]), row(["60"]), row(["999"]), row(["40"], "SELECT", "Economy"), empty)]
    assert reasons == ["gold_not_interactive", "ambiguous_descendants", "gold_not_in_html",
                       "select_option_not_found", "no_pos_candidates"]


def test_history_and_helpers():
    assert m.build_history(row([])["action_reprs"], 3) == [
        {"action": "input", "kind": "fill", "text": "Zurich", "page_changed": True},
        {"action": "Flights", "kind": "click", "text": None, "page_changed": True},
        {"action": "select → Business", "kind": "select", "text": None, "page_changed": True}]
    assert m.fix_self_closing('<iframe a="1"/><img b="2"/>') == '<iframe a="1"></iframe><img b="2"/>'
