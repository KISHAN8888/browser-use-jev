import json, sys, pathlib, random
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from test_parity import SNAPSHOT
from myjev.jevformat import make_record
from myjev import build_dataset as bd

GOLDS = [["e3"], ["e4"], ["e5"], ["scroll_down"]]
HISTORY = [{"action": "Where from?", "kind": "fill", "text": "Zürich", "page_changed": True},
           {"action": "span", "kind": "click", "text": None, "page_changed": True}]


def traj(source, site, task, n, done=True, blocked=False):
    """One trajectory of n steps; last step DONE (or BLOCKED proxy)."""
    out = []
    for i in range(n):
        last = i == n - 1
        gold = (["BLOCKED"] if blocked else ["DONE"]) if last and (done or blocked) else GOLDS[i % len(GOLDS)]
        meta = {"source": source, "website": site, "task_id": task, "step_index": i, "blocked_proxy": blocked and last}
        r = make_record(SNAPSHOT, f"goal {source} {site} {task} {i}", HISTORY if i else [], gold, meta)
        assert r is not None, gold
        r["request"]["state"]["page"]["title"] = '<text backend_node_id="1">Real title</text>'

        out.append(r)
    return out


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def make_processed(root):
    bench = traj("mind2web", "bigshop.com", "b1", 100, done=False) + traj("mind2web", "bigshop.com", "b2", 100, done=False) \
        + traj("mind2web", "tinybench.com", "t1", 10, done=False)
    write(root / "mind2web" / "test_website.jsonl", bench)
    live = []
    for k in range(3):  # big live sites -> bench held-out candidates and train
        for t in range(40):
            live += traj("nnetnav_live", f"big{k}.com", f"L{k}_{t}", 5, blocked=(t % 10 == 0))
    for k in range(30):  # small live sites -> val/calib candidates
        live += traj("nnetnav_live", f"small{k}.org", f"S{k}", 6)
    live += traj("nnetnav_live", "bigshop.com", "leak", 5)  # on a bench website -> must be removed
    write(root / "nnetnav_live" / "train.jsonl", live)
    m2w = []
    for k in range(30):
        m2w += traj("mind2web", f"m2w{k}", f"A{k}", 8, done=False)
    write(root / "mind2web_train_full" / "train.jsonl", m2w)
    chain = []
    for k in range(30):  # human-labelled source with DONE/SCROLL -> val/calib rare ops
        chain += traj("webchain", f"chain{k}.net", f"C{k}", 6)
    write(root / "webchain" / "train.jsonl", chain)
    wa = []
    for t in range(400):
        wa += traj("nnetnav_wa", "shopping_admin", f"W{t}", 4, done=False)
    write(root / "nnetnav_wa" / "train.jsonl", wa)
    write(root / "mind2web" / "test_iid_unused.jsonl", traj("mind2web", "x", "x", 2))  # not a pool/bench file


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_build_end_to_end(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    make_processed(processed)
    audit = tmp_path / "audit.jsonl"
    rows = [{"id": "mind2web:test_website:200", "gold": {"operation": "TYPE_TEXT", "type_text_target": ["2"]}, "equivalent_keys": ["1"]},
            {"id": "mind2web:test_website:201", "gold": {"operation": "DONE"}, "equivalent_keys": ["9"]}]  # stale: gold differs
    audit.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    args = bd.parse_args(["--out", str(tmp_path / "out"), "--min-file-age-s", "0", "--val-steps", "60", "--calib-steps", "60",
                          "--bench-site-cap", "50", "--train-site-cap", "150", "--max-small-site-steps", "20", "--audit", str(audit)])
    stats = bd.build(args, processed=processed, log=lambda *a: None)
    splits = {name: read(tmp_path / "out" / f"{name}.jsonl") for name in ("train", "val", "calib", "bench")}

    # page_changed normalised, titles stripped of tags, bare-tag history labels made explicit.
    for records in splits.values():
        for r in records:
            history = r["request"]["state"]["recent_actions"]
            assert all(h.get("page_changed") is None for h in history)
            assert r["request"]["state"]["page"]["title"] == "Real title"
            assert all(h["action"] != "span" for h in history) and (not history or history[1]["action"] == "(unlabeled span)")
    # Bench: per-site cap; held-out live sites present; no blocked proxies.
    bench_sites = Counter(r["meta"]["site_key"] for r in splits["bench"])
    assert max(bench_sites.values()) <= 50 and bench_sites["bigshop"] == 50
    assert {r["meta"]["bench_part"] for r in splits["bench"]} == {"mind2web:test_website"}
    # Audit equivalents merged only when the audited gold still matches (line 200 = tinybench step 0).
    by_id = {r["meta"]["id"]: r for r in splits["bench"]}
    assert stats["bench_audit_equivalents_merged"] == 1
    assert by_id["mind2web:test_website:200"]["gold"]["type_text_target"] == ["2", "1"]
    assert "audit_equivalents" not in by_id["mind2web:test_website:201"]["meta"]
    assert not any(r["meta"].get("blocked_proxy") for name in ("bench", "val", "calib") for r in splits[name])
    # Whole trajectories kept: bigshop has two 100-step trajectories -> the 50 sampled steps are one prefix.
    assert len({r["meta"]["task_id"] for r in splits["bench"] if r["meta"]["site_key"] == "bigshop"}) == 1
    # No train/val/calib step on a bench website, and the leak was counted.
    for name in ("train", "val", "calib"):
        assert not {r["meta"]["site_key"] for r in splits[name]} & set(bench_sites)
    assert stats["bench_overlap_removed"]["nnetnav_live"]["steps"] >= 5
    # val/calib: disjoint sites, both sources, contain rare ops; no WebArena.
    assert not {r["meta"]["site_key"] for r in splits["val"]} & {r["meta"]["site_key"] for r in splits["calib"]}
    for name in ("bench", "val", "calib"):
        assert not {r["meta"]["source"] for r in splits[name]} & {"nnetnav_live", "nnetnav_wa"}
    for name in ("val", "calib"):
        assert {r["meta"]["source"] for r in splits[name]} == {"webchain", "mind2web"}
        assert stats[name]["rare_op_frac"] >= 0.08
    # train: site cap, WebArena share, weights.
    train_sites = Counter(r["meta"]["site_key"] for r in splits["train"])
    assert max(n for site, n in train_sites.items() if site != "shopping_admin") <= 150
    assert train_sites["shopping_admin"] > 150  # WebArena: share cap only, no site cap
    wa = sum(r["meta"]["source"] == "nnetnav_wa" for r in splits["train"]) / len(splits["train"])
    assert wa <= 0.36
    from myjev.relabel import op_weight  # default --op-weights multiply the source weight
    weights = {(r["meta"]["source"], bool(r["meta"].get("blocked_proxy")), round(r["meta"]["weight"] / op_weight(r["gold"]["operation"]), 6))
               for r in splits["train"]}
    assert weights == {("nnetnav_live", False, 0.5), ("nnetnav_live", True, 0.3), ("nnetnav_wa", False, 0.5),
                       ("mind2web", False, 1.0), ("webchain", False, 1.0)}
    assert stats["concentration"]["train_after_cap"]["gini"] <= stats["concentration"]["train_before_cap"]["gini"]
    assert stats["source_leakage"]["accuracy"] >= 0  # probe ran
    assert all(r["meta"]["website"] != "x" for records in splits.values() for r in records)


def test_fit_cap_and_sample_site():
    assert bd.fit_cap([10, 100, 300], None, 120) == 120
    assert bd.fit_cap([10, 100, 300], 150) == 70  # 10 + 70 + 70
    steps = [bd.Step(site="s", traj=f"t{t}", order=i) for t in range(10) for i in range(7)]
    got = bd.sample_site(steps, 30, random.Random(0))
    assert len(got) == 30
    per_traj = Counter(s.traj for s in got)
    assert sorted(per_traj.values()).count(7) == 4  # four whole trajectories + a 2-step prefix of a fifth
    assert bd.sample_site(steps, 100, random.Random(0)) == steps


def test_source_leakage_detects_separable_sources():
    rng = random.Random(0)
    feats, labels = [], []
    for i in range(400):
        src = "a" if i % 2 else "b"
        f = [rng.random() for _ in bd.FEATURES]
        f[0] = 1.0 if src == "a" else 0.0  # has_title separates the sources
        feats.append(f)
        labels.append(src)
    out = bd.source_leakage(feats, labels)
    assert out["accuracy"] > 0.95 and out["top_features"][0][0] == "has_title"
