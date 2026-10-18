import copy
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from test_build_dataset import make_processed, read, write
from myjev import build_dataset as bd
from myjev import relabel as rl
from myjev.jevformat import make_record

SNAP = {
    "url": "https://example.test/flights", "title": "Flights", "text": "From\nTo\nSearch",
    "actions": [
        {"id": "from_fill", "kind": "fill", "node": 1, "role": "combobox", "label": "Where from?", "value": "Zürich"},
        {"id": "from_click", "kind": "click", "node": 1, "role": "combobox", "label": "Open Where from?", "value": "Zürich"},
        {"id": "to_fill", "kind": "fill", "node": 2, "role": "combobox", "label": "Where to?", "value": ""},
        {"id": "to_click", "kind": "click", "node": 2, "role": "combobox", "label": "Open Where to?", "value": ""},
        {"id": "search", "kind": "click", "node": 3, "role": "button", "label": "Search", "value": ""},
        {"id": "cabin_biz", "kind": "select", "node": 4, "role": "combobox", "label": "Cabin → Business", "value": "b",
         "current_value": "Economy"},
        # A node is never both click and select in snapshot.js; a wrapper/trigger with the select's name can be.
        {"id": "cabin_click", "kind": "click", "node": 5, "role": "combobox", "label": "Cabin", "value": "Economy"},
        {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560},
        {"id": "wait", "kind": "wait", "label": "Wait for the page to update"},
    ],
}


def step(gold, order, history=(), meta=None, snap=SNAP):
    record = make_record(snap, "Fly from Zürich to London", list(history), [gold], dict(meta or {}))
    assert record is not None, gold
    return rl.summarize(record, order)


def fill(label, text):
    return {"action": label, "kind": "fill", "text": text, "page_changed": None}


def click(label):
    return {"action": label, "kind": "click", "text": None, "page_changed": None}


def test_summary_targets_and_order():
    s = step("cabin_biz", 3)
    assert s["op"] == "SELECT" and s["targets"][0]["label"] == "cabin" and s["targets"][0]["index"] == "4"
    assert step("to_click", 0)["targets"][0]["ops"] == ["TYPE_TEXT", "CLICK"]
    assert step("DONE", 0)["targets"] == []
    assert rl.step_order({"step_index": 4}) == 4 and rl.step_order({"turn_index": 7}) == 7
    assert rl.step_order({}, fallback=12) == 12


def test_click_then_type_same_field_dropped():
    traj = [step("to_click", 0), step("to_fill", 1, [click("Open Where to?")]), step("search", 2, [fill("Where to?", "London")])]
    assert rl.plan_drops(traj) == {0: "click_then_type"}
    assert rl.plan_drops(list(reversed(traj))) == {2: "click_then_type"}  # input order does not matter


def test_click_then_type_other_field_or_button_kept():
    assert rl.plan_drops([step("from_click", 0), step("to_fill", 1)]) == {}
    assert rl.plan_drops([step("search", 0), step("to_fill", 1)]) == {}
    assert rl.plan_drops([step("to_click", 0), step("search", 1)]) == {}  # e.g. date picker: click opens a widget


def test_adjacency_by_history_when_order_has_gaps():
    # WebLINX: turn_index gaps are chat utterances; the next step's history proves the click came right before.
    assert rl.plan_drops([step("to_click", 3), step("to_fill", 6, [click("Open Where to?")])]) == {0: "click_then_type"}
    assert rl.plan_drops([step("to_click", 3), step("to_fill", 6, [click("Search")])]) == {}


def test_repeated_clicks_before_type_dropped():
    traj = [step("search", 0), step("to_click", 1), step("to_click", 2, [click("Open Where to?")]),
            step("to_fill", 3, [click("Open Where to?")])]
    assert rl.plan_drops(traj) == {1: "click_then_type", 2: "click_then_type"}


def test_click_then_select_same_control():
    assert rl.plan_drops([step("cabin_click", 0), step("cabin_biz", 1, [click("Cabin")])]) == {0: "click_then_select"}
    assert rl.plan_drops([step("search", 0), step("cabin_biz", 1)]) == {}


def test_enter_on_field():
    meta = {"gold_op_raw": {"original_op": "ENTER", "op": "CLICK", "value": ""}}
    assert rl.plan_drops([step("from_click", 0, meta=meta), step("search", 1)]) == {0: "enter_on_field"}
    assert rl.plan_drops([step("search", 0, meta=meta), step("DONE", 1)]) == {}  # ENTER on a button is a click


def test_type_already_filled():
    # "Where from?" already holds "Zürich": typing Zürich again is a no-op label; typing something else is fine.
    assert rl.plan_drops([step("from_fill", 0), step("search", 1, [fill("Where from?", " zürich ")])]) == {0: "type_already_filled"}
    assert rl.plan_drops([step("from_fill", 0), step("search", 1, [fill("Where from?", "Basel")])]) == {}
    assert rl.plan_drops([step("from_fill", 0)]) == {}  # typed text unknown (last step) -> keep
    m2w = {"gold_op_raw": {"original_op": "TYPE", "op": "TYPE", "value": "Zürich"}}
    assert rl.plan_drops([step("from_fill", 0, meta=m2w)]) == {0: "type_already_filled"}
    assert rl.plan_drops([step("to_fill", 0), step("search", 1, [fill("Where to?", "")])]) == {}  # empty field


def test_same_element_rules():
    a = {"index": "3", "label": "", "role": "textbox"}
    assert rl.same_element(a, dict(a)) and not rl.same_element(a, dict(a, index="4"))
    assert not rl.same_element({"index": "1", "label": "to", "role": "textbox"}, {"index": "1", "label": "to", "role": "button"})
    assert rl.same_element({"index": "1", "label": "to", "role": "textbox"}, {"index": "9", "label": "to", "role": "textbox"})


def test_op_weights():
    assert rl.parse_op_weights("type_text=1.5, SELECT=2") == {"TYPE_TEXT": 1.5, "SELECT": 2.0}
    assert rl.parse_op_weights("") == {}
    assert rl.op_weight("TYPE_TEXT") == rl.OP_WEIGHT["TYPE_TEXT"] and rl.op_weight("CLICK") == 1.0
    assert rl.op_weight("SELECT", {}) == 1.0


def click_type_traj(source, site, task, n_pairs, extra_meta=None):
    out, history = [], []
    for i in range(n_pairs):
        for j, gold in enumerate(("to_click", "to_fill")):
            meta = {"source": source, "website": site, "task_id": task, "step_index": 2 * i + j, **(extra_meta or {})}
            r = make_record(SNAP, f"goal {task}", history, [gold], meta)
            out.append(r)
            history = history + [click("Open Where to?") if j == 0 else fill("Where to?", f"x{i}")]
    return out


def test_build_applies_fix_to_pool_not_bench(tmp_path):
    processed = tmp_path / "processed"
    make_processed(processed)
    pool = read(processed / "webchain" / "train.jsonl")
    for k in range(30):
        pool += click_type_traj("webchain", f"chain{k}.net", f"CT{k}", 2)
    write(processed / "webchain" / "train.jsonl", pool)
    bench = read(processed / "mind2web" / "test_website.jsonl") + click_type_traj("mind2web", "tinybench.com", "BT", 2)
    write(processed / "mind2web" / "test_website.jsonl", bench)
    common = ["--min-file-age-s", "0", "--val-steps", "60", "--calib-steps", "60", "--bench-site-cap", "50",
              "--train-site-cap", "150", "--max-small-site-steps", "40", "--audit", str(tmp_path / "none.jsonl")]
    out_on, out_off = tmp_path / "on", tmp_path / "off"
    stats_on = bd.build(bd.parse_args(["--out", str(out_on)] + common), processed=processed, log=lambda *a: None)
    bd.build(bd.parse_args(["--out", str(out_off), "--no-fix-labels"] + common), processed=processed, log=lambda *a: None)
    on = {n: read(out_on / f"{n}.jsonl") for n in ("train", "val", "calib", "bench")}
    off = {n: read(out_off / f"{n}.jsonl") for n in ("train", "val", "calib", "bench")}

    assert (out_on / "bench.jsonl").read_bytes() == (out_off / "bench.jsonl").read_bytes()
    assert any(r["meta"].get("task_id") == "BT" and r["gold"]["operation"] == "CLICK" for r in on["bench"])
    for name in ("train", "val", "calib"):
        ids_on = {r["meta"]["id"] for r in on[name]}
        ids_off = {r["meta"]["id"] for r in off[name]}
        assert ids_on <= ids_off
        dropped = [r for r in off[name] if r["meta"]["id"] not in ids_on]
        assert all(r["meta"].get("task_id", "").startswith("CT") and r["gold"]["operation"] == "CLICK" for r in dropped)
        assert sum(stats_on["relabel"]["dropped"][name].values()) == len(dropped)
        assert not any(r["meta"].get("task_id", "").startswith("CT") and r["gold"]["operation"] == "CLICK" for r in on[name])
    assert sum(sum(v.values()) for v in stats_on["relabel"]["dropped"].values()) > 0
    # Weights: source weight x op multiplier in train only.
    base = {r["meta"]["id"]: r["meta"]["weight"] for r in off["train"]}
    for r in on["train"]:
        assert r["meta"]["weight"] == round(base[r["meta"]["id"]] * rl.op_weight(r["gold"]["operation"]), 6)
    assert {r["meta"]["weight"] for r in on["train"] if r["gold"]["operation"] == "TYPE_TEXT" and r["meta"]["source"] == "webchain"} == {rl.op_weight("TYPE_TEXT")}
    assert not any("weight" in r["meta"] for n in ("val", "calib", "bench") for r in on[n])
    assert stats_on["relabel"]["train_class_mass"]["after_op_weights"]["TYPE_TEXT"] > \
        stats_on["relabel"]["train_class_mass"]["before_op_weights"]["TYPE_TEXT"]


def test_reuse_splits_keeps_assignment(tmp_path):
    processed = tmp_path / "processed"
    make_processed(processed)
    pool = read(processed / "webchain" / "train.jsonl")
    for k in range(30):
        pool += click_type_traj("webchain", f"chain{k}.net", f"CT{k}", 2)
    write(processed / "webchain" / "train.jsonl", pool)
    common = ["--min-file-age-s", "0", "--val-steps", "60", "--calib-steps", "60", "--bench-site-cap", "50",
              "--train-site-cap", "150", "--max-small-site-steps", "40", "--audit", str(tmp_path / "none.jsonl")]
    prev, new = tmp_path / "prev", tmp_path / "new"
    s_prev = bd.build(bd.parse_args(["--out", str(prev), "--no-fix-labels"] + common), processed=processed, log=lambda *a: None)
    # Different seed would pick different val/calib sites; --reuse-splits must override that.
    s_new = bd.build(bd.parse_args(["--out", str(new), "--seed", "7", "--reuse-splits", str(prev / "stats.json")] + common),
                     processed=processed, log=lambda *a: None)
    assert s_new["val_sites"] == s_prev["val_sites"] and s_new["calib_sites"] == s_prev["calib_sites"]
    prev_train = [r["meta"]["id"] for r in read(prev / "train.jsonl")]
    new_train = [r["meta"]["id"] for r in read(new / "train.jsonl")]
    dropped = sum(s_new["relabel"]["dropped"]["train"].values())
    assert dropped > 0 and len(new_train) == len(prev_train) - dropped
    assert new_train == [i for i in prev_train if i in set(new_train)]  # subset, same order
    prev_vc = {r["meta"]["id"] for n in ("val", "calib") for r in read(prev / f"{n}.jsonl")}
    new_vc = {r["meta"]["id"] for n in ("val", "calib") for r in read(new / f"{n}.jsonl")}
    assert new_vc <= prev_vc and not new_vc & set(prev_train)
