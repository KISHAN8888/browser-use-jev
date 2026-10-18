"""Assemble final splits from converted sources (data/processed/<source>/*.jsonl).

- bench:  every part in BENCH_PARTS below that exists on disk (Mind2Web test_website + test_domain, WebLINX
          test_web; human labels only -- NNetNav is never used in bench/val/calib after the label audit). At most --bench-site-cap steps per
          website (seeded, whole trajectories kept together where possible) so no site dominates; a part with a
          `max_steps` budget lowers its per-site cap until it fits (keeps as many websites as possible).
          meta.bench_part says which part a step came from.
- val, calib: whole small websites (<= --max-small-site-steps) from all human-labelled (non-NNetNav, non-WebArena) sources, quota
          split evenly across sources, sites with DONE/SCROLL/WAIT preferred until each split has
          >= --min-rare-frac of those ops (where available). WebArena sites only ever go to train.
- train:  the rest, deduplicated, per-website cap (--train-site-cap) on real-website sources only, WebArena limited
          only by its share cap (--webarena-max-share),
          DONE downsampled, shuffled with a fixed seed so `--train-limit N` prefixes are nested subsets.

Leakage control:
- Any train/val/calib step whose website (normalised by `website()`) appears in ANY bench part is removed.
- `recent_actions[*].page_changed` is set to None for every source (Mind2Web always had True, AX-tree sources
  None, so it identified the source; jev-ultrafast does send it, but our data cannot know it reliably).
- A trivial logistic-regression probe on cheap surface features reports how identifiable each source still is.

BLOCKED labels from NNetNav `stop [N/A]` / Go-Browse report_infeasible are proxies (meta.blocked_proxy). They are
excluded from bench/val/calib. In train they are kept with meta.weight = --blocked-proxy-weight (0.3); every
other NNetNav train step has meta.weight = --noisy-source-weight (0.5; audit found ~25% of NNetNav labels wrong);
the rest 1.0. The trainer may use meta.weight as a per-example loss weight.
Label-convention fixes (--fix-labels, default on; myjev/relabel.py) apply to train/val/calib only, never bench: drop
CLICK-on-field steps immediately followed by TYPE_TEXT into that field (jev-ultrafast types directly), CLICK-on-dropdown
followed by SELECT of it, Mind2Web ENTER-on-field clicks, and TYPE_TEXT into a field already holding that text. Decisions
are made on whole source trajectories (before dedupe/caps) and applied after the split selection, so splits are the
same sites as without the fix minus the dropped steps. Train meta.weight is also multiplied by --op-weights
(TYPE_TEXT 1.25, SELECT 1.5 by default); val/calib stay unweighted.
Bench steps listed in data/audit/bench_audit.jsonl with `equivalent_keys` get those keys added to the gold head
(meta.audit_equivalents = True).

Every record gets meta.id = "<source>:<file stem>:<line>", meta.site_key and meta.tags (difficulty categories,
chosen from features only, never from any model's errors).

Memory: files are indexed in one streaming pass (byte offsets + a few small fields per step), splits are decided on
the index, and records are re-read one at a time by offset when writing.

    python -m myjev.build_dataset --out data/final_v2
"""

import argparse
import hashlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from . import relabel

PROCESSED = Path("data/processed")
WEBARENA_SOURCES = {"nnetnav_wa", "gobrowse_wa"}

# Benchmark parts. "file" parts are taken from data/processed/<dir>/<file> whenever that file exists;
# "heldout_sites" parts hold out whole websites of a training source (steps taken = sites x per-site cap).
# max_steps (optional) is a budget for that part, met by lowering its per-site cap.
BENCH_PARTS = {
    "mind2web:test_website": {"dir": "mind2web", "file": "test_website.jsonl"},
    "mind2web:test_domain": {"dir": "mind2web", "file": "test_domain.jsonl", "max_steps": 1500},
    "weblinx:test_web": {"dir": "weblinx", "file": "test_web.jsonl", "max_steps": 1200},
    # Disabled after the label audit (data/audit/summary.md: ~25% of NNetNav labels wrong). To re-enable:
    # "nnetnav_live:heldout": {"heldout_sites": "nnetnav_live", "max_steps": 360},
}
# Synthetic-agent labels (audit: ~25% wrong). Never in bench/val/calib; down-weighted in train.
NOISY_SOURCES = {"nnetnav_live", "nnetnav_wa"}
AUDIT_FILE = Path("data/audit/bench_audit.jsonl")
BARE_TAGS = {"span", "svg", "i", "a", "div", "img", "button", "label", "li", "path", "input"}
# Only these files feed train/val/calib (other test files of a source are ignored, not leaked into train).
POOL_FILE = re.compile(r"^(train|valid|dev)[\w\-]*\.jsonl$")
RARE_OPS = {"DONE", "SCROLL_DOWN", "SCROLL_UP", "WAIT", "BLOCKED"}  # what val/calib must contain some of
TRAJ_KEYS = ("annotation_id", "task_id", "traj_id", "trajectory_id", "episode_id", "demo_id", "demo", "session_id")


def load(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


SUFFIXES = {"com", "org", "net", "co", "uk", "io", "gov", "edu", "us", "ca", "de", "fr", "in", "au", "jp", "info", "ai", "app"}


def website(record):
    """Core site name shared across sources: 'www.imdb.com', 'imdb' and 'm.imdb.com' all become 'imdb',
    and every subdomain of a site counts as that site (stricter held-out splits)."""
    meta = record["meta"]
    site = meta.get("website") or meta.get("host")
    if not site:
        url = record["request"]["state"]["page"].get("url", "")
        site = re.sub(r"^https?://", "", url).split("/")[0] or "unknown"
    labels = [label for label in str(site).lower().split(":")[0].split(".") if label]
    while len(labels) > 1 and labels[-1] in SUFFIXES:
        labels.pop()
    return labels[-1] if labels else "unknown"


def dedupe_key(record):
    request = record["request"]
    key = json.dumps([request["questions"]["operation"]["instructions"]["goal"], request["state"]["page"].get("url"),
                      [e.get("label") for e in request["state"]["elements"]],
                      [h.get("action") for h in request["state"]["recent_actions"]], record["gold"]], sort_keys=True)
    return hashlib.sha1(key.encode()).hexdigest()


def traj_key(record, source):
    meta = record["meta"]
    for key in TRAJ_KEYS:
        if meta.get(key) not in (None, ""):
            return f"{source}:{meta[key]}"
    return f"{source}:goal:{record['request']['questions']['operation']['instructions']['goal']}"


def tags(record):
    """Difficulty categories from features only."""
    request, gold = record["request"], record["gold"]
    elements = request["state"]["elements"]
    goal = request["questions"]["operation"]["instructions"]["goal"]
    out = []
    if len(elements) >= 150:
        out.append("many_elements")
    labels = Counter(str(e.get("label", "")).strip().lower() for e in elements)
    target_key = gold["operation"].lower() + "_target"
    if target_key in gold:
        criteria = request["questions"][target_key]["criteria"]
        gold_labels = {str(criteria[t]["element"]).split("] ", 1)[-1].strip().lower() for t in gold[target_key] if t in criteria}
        if any(labels[label] > 1 for label in gold_labels):
            out.append("duplicate_label_distractor")
    if len(re.findall(r"\d", goal)) >= 2 or len(re.findall(r"\b(and|with|under|between|filter|sort)\b", goal.lower())) >= 2:
        out.append("multi_constraint")
    roles = {e.get("role") for e in elements}
    if "combobox" in roles or any(e.get("options") for e in elements):
        out.append("dropdown_or_autocomplete")
    if any(re.search(r"\b(date|calendar|check-?in|depart)", str(e.get("label", "")).lower()) for e in elements):
        out.append("date_widget")
    if len(request["state"]["recent_actions"]) >= 3:
        out.append("long_history")
    if gold["operation"] in {"DONE", "BLOCKED"}:
        out.append("terminal_decision")
    return out


def clean_goal(goal):
    """NNetNav-live has ~1.4% goals like '_instruction  Search for X_'."""
    return re.sub(r"^\W*_?instruction\W*", "", goal).strip(" _\n") or goal


def normalise(record):
    """Remove fields that identify the source rather than the page (see module docstring), strip literal tags from
    titles (Mind2Web saved <title>s like '<text backend_node_id=..>Real title</text>') and make bare-tag history
    labels ('span', 'svg', ...) consistent. Idempotent."""
    page = record["request"]["state"]["page"]
    if page.get("title"):
        page["title"] = re.sub(r"<[^>]+>", "", page["title"]).strip()
    for action in record["request"]["state"].get("recent_actions", []):
        if "page_changed" in action:
            action["page_changed"] = None
        label = str(action.get("action") or "").strip().lower()
        if label in BARE_TAGS:
            action["action"] = f"(unlabeled {label})"
    return record


def load_audit(path):
    """id -> equivalent target keys judged acceptable by the label audit."""
    out = {}
    if path and Path(path).exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if row.get("equivalent_keys"):
                        out[row["id"]] = ([str(k) for k in row["equivalent_keys"]], row.get("gold"))
    return out


def merge_equivalents(record, keys, audited_gold=None):
    """Adds audit-equivalent keys to the gold head; skipped if the record's gold no longer matches the audited one
    (the audit refers to ids of an earlier build)."""
    gold = record["gold"]
    if audited_gold is not None and audited_gold != gold:
        return False
    head = gold["operation"].lower() + "_target"
    if head in gold and keys:
        gold[head] = list(dict.fromkeys(gold[head] + keys))
        record["meta"]["audit_equivalents"] = True
        return True
    return False


def finish(record, rid):
    for question in record["request"]["questions"].values():
        if "goal" in question.get("instructions", {}):
            question["instructions"]["goal"] = clean_goal(question["instructions"]["goal"])
    normalise(record)
    record["meta"]["id"] = rid
    record["meta"]["tags"] = tags(record)
    record["meta"]["site_key"] = website(record)
    return record


# ---------------------------------------------------------------- surface features for the source-leakage probe

FEATURES = ["has_title", "url_root_only", "url_has_port", "url_ip_or_cloud_host", "url_has_query", "log_n_elements",
            "log_text_len", "n_history", "hist_page_changed_true", "hist_page_changed_none", "hist_text_frac",
            "n_target_heads", "frac_elements_expanded", "frac_elements_empty_label", "log_goal_len"]


def surface_features(record):
    state = record["request"]["state"]
    page, elements, history = state["page"], state["elements"], state.get("recent_actions", [])
    url = str(page.get("url") or "")
    host_path = re.sub(r"^https?://", "", url)
    host, _, path = host_path.partition("/")
    nh = len(history)
    goal = record["request"]["questions"]["operation"]["instructions"]["goal"]
    return [
        float(bool(page.get("title"))),
        float(path in ("", "#") and "?" not in url),
        float(":" in host),
        float(bool(re.search(r"^\d+\.\d+\.\d+\.\d+|amazonaws|localhost", host))),
        float("?" in url),
        math.log1p(len(elements)),
        math.log1p(len(page.get("text") or "")),
        float(nh),
        sum(h.get("page_changed") is True for h in history) / nh if nh else 0.0,
        sum(h.get("page_changed") is None for h in history) / nh if nh else 0.0,
        sum(bool(h.get("text")) for h in history) / nh if nh else 0.0,
        float(len(record["request"]["questions"]) - 1),
        sum("expanded" in e for e in elements) / max(1, len(elements)),
        sum(not str(e.get("label", "")).strip() for e in elements) / max(1, len(elements)),
        math.log1p(len(goal)),
    ]


def source_leakage(features, labels, seed=0, iters=400, lr=0.5, l2=1e-3):
    """Multinomial logistic regression (numpy) predicting the source from surface features. 70/30 split.
    Returns accuracy, balanced accuracy, majority baseline and features ranked by max |standardised weight|."""
    import numpy as np

    classes = sorted(set(labels))
    if len(classes) < 2:
        return None
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray([classes.index(label) for label in labels])
    idx = np.random.default_rng(seed).permutation(len(y))
    cut = int(0.7 * len(y))
    tr, te = idx[:cut], idx[cut:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xs = (X - mu) / sd
    W = np.zeros((X.shape[1], len(classes)))
    b = np.zeros(len(classes))
    Y = np.eye(len(classes))[y[tr]]
    for _ in range(iters):
        z = Xs[tr] @ W + b
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        g = p - Y
        W -= lr * (Xs[tr].T @ g / len(tr) + l2 * W)
        b -= lr * g.mean(0)
    pred = (Xs[te] @ W + b).argmax(1)
    acc = float((pred == y[te]).mean())
    per_class = [float((pred[y[te] == c] == c).mean()) for c in range(len(classes)) if (y[te] == c).any()]
    majority = float(np.bincount(y[te], minlength=len(classes)).max() / len(te))
    strength = np.abs(W).max(1)
    order = np.argsort(-strength)
    top = [(FEATURES[i], round(float(strength[i]), 3), {classes[c]: round(float(W[i, c]), 2) for c in range(len(classes))})
           for i in order[:6]]
    return {"accuracy": acc, "balanced_accuracy": sum(per_class) / len(per_class), "majority_baseline": majority,
            "classes": classes, "n_test": int(len(te)), "top_features": top}


# ---------------------------------------------------------------- index (streaming pass 1)

class Step:
    """What the split logic needs to know about one step; the record itself stays on disk."""
    __slots__ = ("file", "offset", "rid", "source", "site", "traj", "order", "op", "blocked", "features", "part")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def index_file(path, file_no, source_dir, part=None):
    with open(path, "rb") as f:
        line_no = 0
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            line_no += 1
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            meta = record["meta"]
            source = meta.get("source") or source_dir
            normalise(record)
            yield record, Step(file=file_no, offset=offset, rid=f"{source_dir}:{path.stem}:{line_no - 1}", source=source,
                               site=website(record), traj=traj_key(record, source),
                               order=(meta.get("step_index") if isinstance(meta.get("step_index"), int) else line_no),
                               op=record["gold"]["operation"], blocked=bool(meta.get("blocked_proxy")),
                               features=surface_features(record), part=part)


def discover(processed, min_age_s):
    """Bench files and pool files present on disk. Files modified < min_age_s ago are skipped (still being written)."""
    bench_files, pool_files, skipped = {}, [], []
    file_parts = {(spec["dir"], spec["file"]): name for name, spec in BENCH_PARTS.items() if "file" in spec}
    now = time.time()
    for source_dir in sorted(p for p in processed.iterdir() if p.is_dir()):
        for path in sorted(source_dir.glob("*.jsonl")):
            if path.name.endswith(".log.jsonl") or path.name.startswith("_"):
                continue
            if now - path.stat().st_mtime < min_age_s:
                skipped.append(str(path))
                continue
            part = file_parts.get((source_dir.name, path.name))
            if part:
                bench_files[part] = path
            elif POOL_FILE.match(path.name):
                pool_files.append(path)
            else:
                skipped.append(str(path))
    return bench_files, pool_files, skipped


# ---------------------------------------------------------------- sampling helpers

def fit_cap(sizes, budget, cap=None):
    """Largest per-group cap c (<= cap) with sum(min(n, c)) <= budget (water-filling)."""
    sizes = list(sizes)
    hi = max(sizes, default=0) if cap is None else min(cap, max(sizes, default=0))
    if budget is None or sum(min(n, hi) for n in sizes) <= budget:
        return hi
    lo = 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(min(n, mid) for n in sizes) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def sample_site(steps, cap, rng):
    """At most `cap` steps, whole trajectories first (seeded order), then a prefix of one more trajectory."""
    if cap is None or len(steps) <= cap:
        return list(steps)
    trajs = defaultdict(list)
    for s in steps:
        trajs[s.traj].append(s)
    keys = sorted(trajs)
    rng.shuffle(keys)
    out, rest = [], []
    for k in keys:
        if len(out) + len(trajs[k]) <= cap:
            out += trajs[k]
        else:
            rest.append(k)
    if len(out) < cap and rest:
        out += sorted(trajs[rest[0]], key=lambda s: s.order)[: cap - len(out)]
    return out


def cap_by_site(steps, cap, rng, caps=None):
    by_site = defaultdict(list)
    for s in steps:
        by_site[s.site].append(s)
    out = []
    for site in sorted(by_site):
        out += sample_site(by_site[site], (caps or {}).get(site, cap), rng)
    return out


def concentration(steps):
    sizes = sorted(Counter(s.site for s in steps).values())
    n, total = len(sizes), sum(sizes)
    if not total:
        return {"sites": 0}
    gini = sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(sizes)) / (n * total)
    entropy = -sum(x / total * math.log(x / total) for x in sizes)
    return {"steps": total, "sites": n, "gini": round(gini, 3), "entropy_norm": round(entropy / math.log(n), 3) if n > 1 else 1.0,
            "effective_sites": round(math.exp(entropy), 1), "top_site_share": round(sizes[-1] / total, 3),
            "top5": [(site, k) for site, k in Counter(s.site for s in steps).most_common(5)]}


def pick_val_calib(pool, targets, max_small, min_rare_frac, rng):
    """Whole small sites with no WebArena/NNetNav step into val/calib. Each split gets an even share per source (water-filled by what
    is available); sites containing RARE_OPS are preferred while a split is below min_rare_frac."""
    sizes, rare, src_votes = Counter(), Counter(), defaultdict(Counter)
    for s in pool:
        if not s.blocked:
            sizes[s.site] += 1
            rare[s.site] += s.op in RARE_OPS
        src_votes[s.site][s.source] += 1
    site_source = {site: votes.most_common(1)[0][0] for site, votes in src_votes.items()}
    excluded = {s.site for s in pool if s.source in WEBARENA_SOURCES | NOISY_SOURCES}
    small = [site for site, n in sizes.items() if n <= max_small and site not in excluded and site != "unknown"]
    by_source = defaultdict(list)
    for site in sorted(small):
        by_source[site_source[site]].append(site)
    available = {src: sum(sizes[x] for x in sites) / len(targets) for src, sites in by_source.items()}
    held = {name: set() for name in targets}
    count, rare_count = Counter(), Counter()
    quotas = {name: {src: min(avail, fit_cap(available.values(), target)) for src, avail in available.items()}
              for name, target in targets.items()}
    for src in sorted(by_source):
        with_rare = [x for x in by_source[src] if rare[x]]
        without = [x for x in by_source[src] if not rare[x]]
        rng.shuffle(with_rare)
        rng.shuffle(without)
        got = Counter()
        while with_rare or without:
            open_splits = [n for n in targets if got[n] < quotas[n][src]]
            if not open_splits:
                break
            name = min(open_splits, key=lambda n: got[n] / max(1, quotas[n][src]))
            need_rare = rare_count[name] < 1.25 * min_rare_frac * max(1, count[name])
            site = (with_rare if (need_rare and with_rare) or not without else without).pop()
            held[name].add(site)
            got[name] += sizes[site]
            count[name] += sizes[site]
            rare_count[name] += rare[site]
    return held


# ---------------------------------------------------------------- stats while writing

class Describer:
    def __init__(self):
        self.n, self.ops, self.sources, self.parts, self.sites, self.tags, self.elements = 0, Counter(), Counter(), Counter(), Counter(), Counter(), []
        self.blocked_proxy = 0

    def add(self, record):
        meta = record["meta"]
        self.n += 1
        self.ops[record["gold"]["operation"]] += 1
        self.sources[meta.get("source")] += 1
        if meta.get("bench_part"):
            self.parts[meta["bench_part"]] += 1
        self.sites[meta["site_key"]] += 1
        self.tags.update(meta["tags"])
        self.elements.append(len(record["request"]["state"]["elements"]))
        self.blocked_proxy += bool(meta.get("blocked_proxy"))

    def result(self):
        el = sorted(self.elements)
        rare = sum(v for k, v in self.ops.items() if k in RARE_OPS)
        out = {"n": self.n, "operations": self.ops.most_common(), "rare_op_frac": round(rare / max(1, self.n), 3),
               "sources": self.sources.most_common(), "websites": len(self.sites), "tags": self.tags.most_common(),
               "median_elements": el[len(el) // 2] if el else 0, "blocked_proxy": self.blocked_proxy,
               "top_sites": self.sites.most_common(10)}
        if self.parts:
            out["bench_parts"] = self.parts.most_common()
            out["bench_part_websites"] = None
        return out


def describe(records):
    d = Describer()
    for r in records:
        d.add(r)
    return d.result()


# ---------------------------------------------------------------- main

def build(args, processed=PROCESSED, log=print):
    rng = random.Random(args.seed)
    out = Path(args.out)
    op_weights = relabel.parse_op_weights(args.op_weights) if args.fix_labels else {}
    bench_files, pool_files, skipped = discover(processed, args.min_file_age_s)
    files = []
    for s in skipped:
        log(f"skip (not a bench/pool file, or modified < {args.min_file_age_s}s ago): {s}")

    # Pass 1: index.
    bench_raw, pool, seen = [], [], set()
    dup = Counter()
    trajectories = defaultdict(list)  # (file, traj) -> [(rid, source, summary)] of every pool step, for relabel
    for part, path in sorted(bench_files.items()):
        files.append(path)
        for _, step in index_file(path, len(files) - 1, path.parent.name, part):
            bench_raw.append(step)
    for path in pool_files:
        files.append(path)
        for record, step in index_file(path, len(files) - 1, path.parent.name):
            if args.fix_labels:
                summary = relabel.summarize(record, relabel.step_order(record["meta"], step.order))
                trajectories[(step.file, step.traj)].append((step.rid, step.source, summary))
            keys = {dedupe_key(record)}
            for k in ("action_uid", "dedupe_key"):
                if record["meta"].get(k):
                    keys.add(f"{k}:{record['meta'][k]}")
            if keys & seen:
                dup[step.source] += 1
                continue
            seen |= keys
            pool.append(step)
    del seen
    log(f"indexed bench parts {sorted(bench_files)}; pool {len(pool)} steps from {[str(p) for p in pool_files]}; duplicates dropped {dict(dup)}")
    drops = {}
    for items in trajectories.values():
        for i, reason in relabel.plan_drops([summary for _, _, summary in items]).items():
            drops[items[i][0]] = reason
    del trajectories

    # Bench: file parts, then held-out whole sites of a training source.
    file_sites = {s.site for s in bench_raw}
    for part, spec in BENCH_PARTS.items():
        if "heldout_sites" not in spec:
            continue
        src_sizes = Counter(s.site for s in pool if s.source == spec["heldout_sites"] and not s.blocked)
        all_sizes = Counter(s.site for s in pool)
        candidates = sorted(site for site, n in src_sizes.items()
                            if n > args.max_small_site_steps and site not in file_sites and site not in {"google", "unknown"})
        rng.shuffle(candidates)
        taken = 0
        for site in candidates:
            if taken >= spec.get("max_steps", 0):
                break
            for s in pool:
                if s.site == site and s.source == spec["heldout_sites"]:
                    s.part = part
                    bench_raw.append(s)
            taken += min(src_sizes[site], args.bench_site_cap)
            log(f"bench {part}: held out site {site} ({all_sizes[site]} steps)")
    bench_sites = {s.site for s in bench_raw}
    bench_candidates = [s for s in bench_raw if not s.blocked and s.source not in NOISY_SOURCES]
    bench_blocked = len(bench_raw) - len(bench_candidates)

    # Overlap: remove every pool step on a bench website.
    overlap_steps, overlap_sites = Counter(), defaultdict(set)
    kept = []
    for s in pool:
        if s.site in bench_sites:
            overlap_steps[s.source] += 1
            overlap_sites[s.source].add(s.site)
        else:
            kept.append(s)
    pool = kept
    overlap = {src: {"steps": n, "sites": sorted(overlap_sites[src])} for src, n in overlap_steps.items()}

    bench = []
    bench_caps = {}
    for part in sorted({s.part for s in bench_candidates}):
        steps = [s for s in bench_candidates if s.part == part]
        sizes = Counter(s.site for s in steps)
        cap = fit_cap(sizes.values(), BENCH_PARTS[part].get("max_steps"), args.bench_site_cap)
        bench_caps[part] = cap
        bench += cap_by_site(steps, cap, rng)
    bench_before = concentration(bench_candidates)
    bench = cap_by_site(bench, args.bench_site_cap, rng)  # a site shared by two parts still gets one cap

    # val / calib.
    reused = load_reuse(args.reuse_splits) if args.reuse_splits else None
    if reused:
        held = {"val": set(reused["val_sites"]), "calib": set(reused["calib_sites"])}
    else:
        held = pick_val_calib(pool, {"val": args.val_steps, "calib": args.calib_steps}, args.max_small_site_steps,
                              args.min_rare_frac, rng)
    val = [s for s in pool if s.site in held["val"] and not s.blocked]
    calib = [s for s in pool if s.site in held["calib"] and not s.blocked]
    train_pool = [s for s in pool if s.site not in held["val"] | held["calib"]]

    # train caps.
    train_before = concentration(train_pool)
    wa_before = sum(s.source in WEBARENA_SOURCES for s in train_pool) / max(1, len(train_pool))
    non_wa = cap_by_site([s for s in train_pool if s.source not in WEBARENA_SOURCES], args.train_site_cap, rng)
    wa_all = [s for s in train_pool if s.source in WEBARENA_SOURCES]
    allowed = int(args.webarena_max_share / (1 - args.webarena_max_share) * len(non_wa))
    # WebArena (~5 sites, nearly all WAIT/SELECT examples) is limited only by its overall share, not the site cap;
    # the share budget is water-filled over its sites so the biggest are trimmed first.
    wa_cap = fit_cap(Counter(s.site for s in wa_all).values(), allowed)
    train = non_wa + cap_by_site(wa_all, wa_cap, rng)
    done = [s for s in train if s.op == "DONE"]
    other = [s for s in train if s.op != "DONE"]
    rng.shuffle(done)
    keep_done = int(args.done_max_frac * len(other) / (1 - args.done_max_frac))
    train = other + done[:keep_done]
    rng.shuffle(train)
    reuse_stats = None
    if reused:
        # Exactly the previous train rows, in the previous order (continued training must not see new rows).
        by_rid = {s.rid: s for s in train_pool}
        train = [by_rid[rid] for rid in reused["train_ids"] if rid in by_rid]
        reuse_stats = {"from": str(args.reuse_splits), "previous_train": len(reused["train_ids"]),
                       "previous_train_missing_now": len(reused["train_ids"]) - len(train)}
        log(f"reuse splits from {args.reuse_splits}: val/calib sites fixed, train = previous train rows "
            f"({reuse_stats['previous_train_missing_now']} of {len(reused['train_ids'])} not found in the pool)")
        if reuse_stats["previous_train_missing_now"]:
            raise SystemExit("previous train rows missing from the pool: processed files changed since that build")

    # Label fixes: drop planned steps from train/val/calib (bench untouched).
    relabel_stats = {"enabled": bool(args.fix_labels), "op_weights": op_weights, "dropped": {}, "ops_before": {}}
    if args.fix_labels:
        for name, steps in (("train", train), ("val", val), ("calib", calib)):
            relabel_stats["ops_before"][name] = dict(Counter(s.op for s in steps).most_common())
            per = Counter((s.source, drops[s.rid]) for s in steps if s.rid in drops)
            relabel_stats["dropped"][name] = {f"{src}:{reason}": n for (src, reason), n in sorted(per.items())}
        train = [s for s in train if s.rid not in drops]
        val = [s for s in val if s.rid not in drops]
        calib = [s for s in calib if s.rid not in drops]
    train_after = concentration(train)
    wa_after = sum(s.source in WEBARENA_SOURCES for s in train) / max(1, len(train))

    # Pass 2: write.
    splits = {"train": train, "val": val, "calib": calib, "bench": bench}
    handles = {}
    stats = {}
    audit = load_audit(args.audit)
    merged = 0
    mass = Counter()
    try:
        for name, steps in splits.items():
            d = Describer()
            path = out / f"{name}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                for s in steps:
                    if s.file not in handles:
                        handles[s.file] = open(files[s.file], "rb")
                    h = handles[s.file]
                    h.seek(s.offset)
                    record = finish(json.loads(h.readline()), s.rid)
                    record["meta"].setdefault("source", s.source)
                    if s.part:
                        record["meta"]["bench_part"] = s.part
                    if name == "train":
                        base = (args.blocked_proxy_weight if s.blocked else
                                args.noisy_source_weight if s.source in NOISY_SOURCES else 1.0)
                        record["meta"]["weight"] = round(base * relabel.op_weight(s.op, op_weights), 6)
                        mass[(base, s.op)] += 1
                    if name == "bench" and s.rid in audit:
                        merged += merge_equivalents(record, *audit[s.rid])
                    d.add(record)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats[name] = d.result()
    finally:
        for h in handles.values():
            h.close()
    if args.expect_bench_sha256:
        got = hashlib.sha256((out / "bench.jsonl").read_bytes()).hexdigest()
        assert got == args.expect_bench_sha256, f"bench.jsonl sha256 {got} != expected {args.expect_bench_sha256}"
        log(f"bench.jsonl sha256 matches {got}")
    stats["bench"]["bench_part_websites"] = {p: len({s.site for s in bench if s.part == p}) for p in sorted({s.part for s in bench})}

    before = Counter()
    after = Counter()
    for (base, op), n in mass.items():
        before[op] += base * n
        after[op] += base * relabel.op_weight(op, op_weights) * n
    relabel_stats["train_class_mass"] = {
        "before_op_weights": {op: round(v / sum(before.values()), 4) for op, v in before.most_common()},
        "after_op_weights": {op: round(v / sum(after.values()), 4) for op, v in after.most_common()}}

    final = train + val + calib + bench
    leakage = source_leakage([s.features for s in final], [s.source for s in final], seed=args.seed)
    stats.update({
        "bench_overlap_removed": overlap,
        "bench_per_part_site_cap": bench_caps,
        "bench_blocked_proxy_excluded": bench_blocked,
        "valcalib_blocked_proxy_excluded": sum(s.blocked for s in pool if s.site in held["val"] | held["calib"]),
        "train_blocked_proxy_weighted": sum(s.blocked for s in train),
        "train_noisy_source_weighted": sum(s.source in NOISY_SOURCES and not s.blocked for s in train),
        "bench_audit_equivalents_merged": merged,
        "relabel": relabel_stats,
        "reuse_splits": reuse_stats,
        "done_dropped": max(0, len(done) - keep_done),
        "concentration": {"bench_before_cap": bench_before, "bench_after_cap": concentration(bench),
                          "train_before_cap": train_before, "train_after_cap": train_after},
        "webarena_share": {"before": round(wa_before, 3), "after": round(wa_after, 3), "wa_site_cap": wa_cap},
        "source_leakage": leakage,
        "val_sites": sorted(held["val"]), "calib_sites": sorted(held["calib"]),
        "bench_sites": sorted({s.site for s in bench}),
        "args": vars(args),
    })
    (out / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    report(stats, log)
    return stats


def load_reuse(stats_path):
    """Split assignment of a previous build: its val/calib sites (stats.json) and train row ids (train.jsonl next to it)."""
    stats_path = Path(stats_path)
    prev = json.loads(stats_path.read_text(encoding="utf-8"))
    ids = []
    with open(stats_path.parent / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.append(json.loads(line)["meta"]["id"])
    return {"val_sites": prev["val_sites"], "calib_sites": prev["calib_sites"], "train_ids": ids}


def report(stats, log=print):
    for name in ("train", "val", "calib", "bench"):
        s = stats[name]
        log(f"{name:<6} {s['n']:>6} steps {s['websites']:>4} sites  rare-op {100 * s['rare_op_frac']:.1f}%  "
            f"median elements {s['median_elements']}  sources {s['sources']}")
        log(f"       ops {s['operations']}")
    log(f"bench parts {stats['bench'].get('bench_parts')}  websites per part {stats['bench'].get('bench_part_websites')}  "
        f"per-part site cap {stats['bench_per_part_site_cap']}")
    log("removed train/val/calib steps on bench websites: "
        + ", ".join(f"{src} {v['steps']} steps / {len(v['sites'])} sites" for src, v in stats["bench_overlap_removed"].items()))
    log(f"blocked_proxy: {stats['bench_blocked_proxy_excluded']} excluded from bench, {stats['valcalib_blocked_proxy_excluded']} "
        f"from val/calib, {stats['train_blocked_proxy_weighted']} kept in train with weight 0.3; "
        f"{stats['train_noisy_source_weighted']} other NNetNav train steps with weight 0.5; "
        f"{stats['bench_audit_equivalents_merged']} bench steps got audit-equivalent gold keys")
    c = stats["concentration"]
    for k in ("train_before_cap", "train_after_cap", "bench_before_cap", "bench_after_cap"):
        log(f"{k:<17} {c[k]}")
    log(f"webarena share in train {stats['webarena_share']}; DONE dropped {stats['done_dropped']}")
    rl = stats.get("relabel") or {}
    if rl.get("enabled"):
        for name, per in rl["dropped"].items():
            log(f"relabel {name}: ops before fix {rl['ops_before'][name]}; dropped {sum(per.values())} {per}")
        log(f"relabel op weights {rl['op_weights']}; train class mass (source weights only) "
            f"{rl['train_class_mass']['before_op_weights']} -> with op weights {rl['train_class_mass']['after_op_weights']}")
    lk = stats["source_leakage"]
    if lk:
        log(f"source-leakage probe: acc {lk['accuracy']:.3f} (balanced {lk['balanced_accuracy']:.3f}, majority {lk['majority_baseline']:.3f}, "
            f"n_test {lk['n_test']}) top features {[(f, w) for f, w, _ in lk['top_features']]}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/final_v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-steps", type=int, default=600)
    parser.add_argument("--calib-steps", type=int, default=600)
    parser.add_argument("--max-small-site-steps", type=int, default=150)
    parser.add_argument("--min-rare-frac", type=float, default=0.08, help="DONE/SCROLL/WAIT share wanted in val/calib")
    parser.add_argument("--bench-site-cap", type=int, default=120)
    parser.add_argument("--train-site-cap", type=int, default=600)
    parser.add_argument("--webarena-max-share", type=float, default=0.35)
    parser.add_argument("--blocked-proxy-weight", type=float, default=0.3)
    parser.add_argument("--noisy-source-weight", type=float, default=0.5, help="meta.weight of NNetNav train steps")
    parser.add_argument("--audit", default=str(AUDIT_FILE), help="label audit with equivalent_keys merged into bench gold")
    parser.add_argument("--fix-labels", action=argparse.BooleanOptionalAction, default=True,
                        help="apply myjev/relabel.py fixes + op weights to train/val/calib (never bench)")
    parser.add_argument("--op-weights", default=",".join(f"{k}={v}" for k, v in relabel.OP_WEIGHT.items()),
                        help="train meta.weight multipliers by gold operation, e.g. TYPE_TEXT=1.5,SELECT=2")
    parser.add_argument("--expect-bench-sha256", default=None,
                        help="fail unless bench.jsonl has this sha256 (frozen bench_v1: f95b894c...bd37)")
    parser.add_argument("--reuse-splits", default=None,
                        help="stats.json of a previous build: reuse its val/calib sites and exactly its train rows")
    parser.add_argument("--done-max-frac", type=float, default=0.08, help="cap DONE share in train")
    parser.add_argument("--min-file-age-s", type=float, default=120, help="skip files modified more recently (still being written)")
    return parser.parse_args(argv)


def main():
    build(parse_args())


if __name__ == "__main__":
    main()
