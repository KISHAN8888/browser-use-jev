"""Multimodal-Mind2Web (osunlp/Multimodal-Mind2Web) -> Jev records.

Each Mind2Web row is one step. We render its `raw_html` offline in headless Chromium, scroll the gold
element to the viewport centre, run jev-ultrafast's real snapshot.js, map snapshot actions back to
Mind2Web `backend_node_id`s and emit `make_record(...)`.

Usage:
    python -m myjev.convert.mind2web --split test_website --out data/processed/mind2web/test_website.jsonl
    python -m myjev.convert.mind2web --split train --sample 300 --out data/processed/mind2web/train_sample300.jsonl

Only the small columns are fetched from the Hub (parquet column projection over HTTP range reads), so
the screenshot column (~95% of the bytes) is never downloaded. They are cached in data/raw/mind2web/.

Rendering choices / approximations (also documented in the report):
- Viewport 1280x1000 (Mind2Web screenshots are 1280 wide).
- raw_html is a DOM dump without CSS, scripts, images or hrefs. Before snapshotting we repair what
  snapshot.js needs: `aria_label` -> `aria-label`, `<a>` without href gets href="#", `input_value`
  -> .value, `input_checked` -> .checked, `option_selected` -> .selected; XML-style `<div/>`/`<iframe/>`
  are expanded (otherwise the HTML parser swallows the rest of the page); whitespace-only text nodes
  from the dump's pretty-printing are removed (they leak into labels).
- The page URL is not in the dataset; we serve the HTML at a synthetic https://<website>/ URL.
"""

import argparse
import asyncio
import json
import os
import pathlib
import random
import re
import statistics
import sys
import time
from collections import Counter

from ..jevformat import make_record

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "mind2web"
SNAPSHOT_JS = (pathlib.Path(__file__).resolve().parents[1] / "vendor" / "snapshot.js").read_text(encoding="utf-8")
REPO = "datasets/osunlp/Multimodal-Mind2Web"
COLUMNS = ["action_uid", "raw_html", "operation", "pos_candidates", "website", "domain", "subdomain",
           "annotation_id", "confirmed_task", "action_reprs", "target_action_index", "target_action_reprs"]
VIEWPORT = {"width": 1280, "height": 1000}
OP_KIND = {"CLICK": "click", "HOVER": "click", "ENTER": "click", "TYPE": "fill", "SELECT": "select"}

# Make Mind2Web's attribute-renamed DOM dump look like the live DOM snapshot.js expects.
REPAIR_JS = """() => {
  for (const e of document.querySelectorAll('[aria_label]'))
    if (!e.hasAttribute('aria-label')) e.setAttribute('aria-label', e.getAttribute('aria_label'));
  for (const a of document.querySelectorAll('a:not([href])')) a.setAttribute('href', '#');
  for (const e of document.querySelectorAll('input[input_value],textarea[input_value]'))
    try { e.value = e.getAttribute('input_value'); } catch (_) {}
  for (const e of document.querySelectorAll('input[input_checked]'))
    e.checked = e.getAttribute('input_checked') === 'true';
  for (const o of document.querySelectorAll('option[option_selected]'))
    if (o.getAttribute('option_selected') === 'true') o.selected = true;
  // The dump is pretty-printed; its indentation would leak into labels as "Songs \\n      navArrow".
  const w = document.createTreeWalker(document.documentElement, NodeFilter.SHOW_TEXT), dead = [];
  while (w.nextNode()) {
    const n = w.currentNode;
    if (!n.textContent.trim() && !(n.parentElement && n.parentElement.closest('pre,textarea'))) dead.push(n);
  }
  for (const n of dead) n.remove();
}"""

# Put the gold at a seeded random height in the viewport, not always the centre: a fixed position would
# leak the answer through its place in the element table.
SCROLL_JS = """([bnid, frac]) => {
  const e = document.querySelector(`[backend_node_id="${bnid}"]`);
  if (!e) return null;
  e.scrollIntoView({block: 'start', inline: 'nearest'});
  // Near the end of a page scrollIntoView cannot bring e to the top, so move by the real offset.
  const r = e.getBoundingClientRect();
  window.scrollBy(0, r.top - Math.max(0, frac * (innerHeight - r.height - 20)));
  return [scrollX, scrollY];
}"""

# For every snapshot action node: its backend_node_id (walking up if needed).
# For every gold backend_node_id: the snapshot nodes that are the element itself, its ancestors
# (nearest first) and its descendants, plus visibility diagnostics for drop reasons.
MAP_JS = """({nodes, gold}) => {
  const M = window.__jevFast.nodes, nodeOf = new Map();
  const bn = {};
  for (const n of nodes) {
    let e = M.get(n), via = false;
    if (e) nodeOf.set(e, n);
    while (e && !e.hasAttribute('backend_node_id')) { e = e.parentElement; via = true; }
    bn[n] = {bnid: e ? e.getAttribute('backend_node_id') : null, via};
  }
  const golds = {};
  for (const g of gold) {
    const e = document.querySelector(`[backend_node_id="${g}"]`);
    if (!e) { golds[g] = {found: false}; continue; }
    const anc = [], desc = [];
    for (let p = e.parentElement; p; p = p.parentElement) if (nodeOf.has(p)) anc.push(nodeOf.get(p));
    for (const d of e.querySelectorAll('*')) if (nodeOf.has(d)) desc.push(nodeOf.get(d));
    const r = e.getBoundingClientRect();
    golds[g] = {found: true, self: nodeOf.has(e) ? nodeOf.get(e) : null, anc, desc, tag: e.tagName.toLowerCase(),
      visible: e.checkVisibility({checkOpacity: true, checkVisibilityCSS: true}) && !e.closest('[aria-hidden="true"],[inert]'),
      in_viewport: r.width > 0 && r.height > 0 && r.bottom > 0 && r.top < innerHeight && r.right > 0 && r.left < innerWidth};
  }
  return {bn, golds};
}"""

NESTED_JS = """([a, b]) => { const M = window.__jevFast.nodes, x = M.get(a), y = M.get(b);
  return !!(x && y && (x.contains(y) || y.contains(x))); }"""


# ---------------------------------------------------------------- data loading

def fetch_split(split, cache_dir=RAW_DIR):
    """Download only the needed columns of one split to <cache_dir>/<split>.parquet and return the path."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{split}.parquet"
    if out.exists():
        return out
    fs = HfFileSystem()
    shards = sorted(p for p in fs.ls(f"{REPO}/data", detail=False) if p.split("/")[-1].startswith(split + "-"))
    if not shards:
        raise SystemExit(f"no shards for split {split!r}")
    tables = []
    for i, shard in enumerate(shards):
        t0 = time.time()
        with fs.open(shard, block_size=8 << 20) as f:
            tables.append(pq.ParquetFile(f).read(columns=COLUMNS))
        print(f"[fetch] {split} shard {i + 1}/{len(shards)} rows={tables[-1].num_rows} {time.time() - t0:.0f}s",
              file=sys.stderr)
    tmp = out.with_suffix(".tmp")
    pq.write_table(pa.concat_tables(tables), tmp, compression="zstd")
    os.replace(tmp, out)
    return out


def select_uids(split, limit=None, sample=None, seed=0):
    """action_uids of the rows to convert (file order; --sample is a seeded random subset)."""
    import pyarrow.parquet as pq
    uids = pq.ParquetFile(fetch_split(split)).read(columns=["action_uid"]).column(0).to_pylist()
    if sample:
        uids = random.Random(seed).sample(uids, min(sample, len(uids)))
    if limit:
        uids = uids[:limit]
    return uids


def iter_rows(split, uids):
    """Stream the selected rows as dicts. Never materialises the whole split (train raw_html is ~1 GB)."""
    import pyarrow.parquet as pq
    wanted = set(uids)
    for batch in pq.ParquetFile(fetch_split(split)).iter_batches(batch_size=16):
        for i, uid in enumerate(batch.column("action_uid").to_pylist()):
            if uid in wanted:
                yield batch.slice(i, 1).to_pylist()[0]


def load_rows(split, limit=None, sample=None, seed=0):
    return list(iter_rows(split, select_uids(split, limit, sample, seed)))


# ---------------------------------------------------------------- pure helpers (unit-tested)

REPR = re.compile(r"^\[(?P<tag>[^\]]*)\]\s*(?P<label>.*?)\s*->\s*(?P<op>[A-Z]+)(?::\s?(?P<value>.*))?$", re.S)


def parse_repr(text):
    """'[span]  TikTok Series -> CLICK' -> ('span', 'TikTok Series', 'CLICK', None)."""
    m = REPR.match(text.strip())
    if not m:
        return None, text.strip(), "CLICK", None
    return m["tag"], m["label"].strip(), m["op"], (m["value"] if m["value"] is not None else None)


def build_history(action_reprs, step_index):
    """Previous steps of the task as jev-ultrafast history entries (last 10)."""
    history = []
    for text in action_reprs[:step_index]:
        tag, label, op, value = parse_repr(text)
        kind = OP_KIND.get(op, "click")
        label = label or tag or "element"
        if kind == "select" and value:
            label = f"{label} → {value}"  # snapshot.js select action label shape
        history.append({"action": label, "kind": kind, "text": value if kind == "fill" else None, "page_changed": True})
    return history[-10:]


VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source",
             "track", "wbr"}
SELF_CLOSING = re.compile(r"""<([a-zA-Z][\w:.-]*)((?:\s+[^\s"'<>/=]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s"'<>]+))?)*)\s*/>""")


def fix_self_closing(html):
    """Mind2Web dumps write empty elements XML-style (`<iframe .../>`, `<div/>`). The HTML parser ignores
    `/>` on non-void elements, so an `<iframe/>` or `<textarea/>` swallows the rest of the page as text
    and `<div/>` nests every following sibling. Expand them to `<tag ...></tag>`."""
    return SELF_CLOSING.sub(lambda m: m.group(0) if m.group(1).lower() in VOID_TAGS
                            else f"<{m.group(1)}{m.group(2)}></{m.group(1)}>", html)


def parse_candidates(pos_candidates):
    """[(backend_node_id, is_original_target), ...] with the original target first."""
    out = []
    for raw in pos_candidates:
        c = json.loads(raw)
        out.append((str(c["backend_node_id"]), bool(c.get("is_original_target"))))
    out.sort(key=lambda x: not x[1])
    return out


def _norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip().casefold()


def match_gold(gold_info, actions_by_node, op, value):
    """Find the snapshot action for one gold element.

    Order: the element itself, then its nearest snapshot ancestor (clicking a span inside a link clicks
    the link), then a *unique* snapshot descendant (a label/li wrapping one input/select).
    Returns (action, how) or (None, reason).
    """
    kind = OP_KIND.get(op)
    if kind is None:
        return None, f"unsupported_op_{op}"

    def pick(node):
        acts = [a for a in actions_by_node.get(node, []) if a["kind"] == kind]
        if kind == "select":
            v = _norm(value)
            acts = [a for a in acts if _norm(a["label"].split(" → ", 1)[-1]) == v or _norm(a["value"]) == v]
        return acts[0] if acts else None

    if not gold_info.get("found"):
        return None, "gold_not_in_html"
    if gold_info.get("self") is not None and (a := pick(gold_info["self"])):
        return a, "exact"
    for node in gold_info.get("anc", []):
        if a := pick(node):
            return a, "ancestor"
    hits = {}
    for node in gold_info.get("desc", []):
        if (a := pick(node)) and node not in hits:
            hits[node] = a
    if len(hits) == 1:
        return next(iter(hits.values())), "descendant"
    if len(hits) > 1:
        return None, "ambiguous_descendants"
    # Diagnose why.
    if not gold_info.get("visible"):
        return None, "gold_not_visible"
    if not gold_info.get("in_viewport"):
        return None, "gold_not_in_viewport"
    if kind == "select":
        has_select = any(a["kind"] == "select" for n in [gold_info.get("self")] + gold_info.get("anc", []) + gold_info.get("desc", [])
                         for a in actions_by_node.get(n, []))
        return None, "select_option_not_found" if has_select else "select_not_native"
    related = [gold_info.get("self")] + gold_info.get("anc", []) + gold_info.get("desc", [])
    if any(actions_by_node.get(n) for n in related if n is not None):
        return None, "kind_mismatch"
    return None, "gold_not_interactive"


# ---------------------------------------------------------------- browser side (async Playwright)

async def snapshot_page(page, html, url, gold_bnids, frac=0.5):
    """Render html at url, scroll gold into view at height `frac`, run snapshot.js and the mapping script."""
    page._m2w_html = fix_self_closing(html)
    await page.goto(url, wait_until="domcontentloaded")
    await page.evaluate(REPAIR_JS)
    scroll = None
    for b in gold_bnids:
        scroll = await page.evaluate(SCROLL_JS, [b, frac])
        if scroll is not None:
            break
    snap = await page.evaluate(SNAPSHOT_JS)
    nodes = sorted({a["node"] for a in snap["actions"] if "node" in a})
    mapping = await page.evaluate(MAP_JS, {"nodes": nodes, "gold": gold_bnids})
    return snap, mapping, scroll


async def new_page(browser):
    page = await browser.new_page(viewport=VIEWPORT, java_script_enabled=True)
    page.set_default_timeout(30000)

    async def route(r):
        if r.request.resource_type == "document" and r.request.is_navigation_request():
            await r.fulfill(status=200, content_type="text/html; charset=utf-8", body=page._m2w_html)
        else:
            await r.abort()  # offline: no CSS/images/scripts/fonts from the network

    await page.route("**/*", route)
    return page


def synthetic_url(website):
    host = (website or "example").strip().lower().replace(" ", "")
    if "." not in host:
        host = f"www.{host}.com"
    return f"https://{host}/"


async def convert_row(page, row, split):
    """Returns (record_or_None, reason, info)."""
    t0 = time.time()
    op_raw = json.loads(row["operation"])
    op = op_raw.get("op") or op_raw.get("original_op")
    step = int(row["target_action_index"])
    info = {"op": op, "seconds": None}
    if not row["pos_candidates"]:
        return None, "no_pos_candidates", info
    cands = parse_candidates(row["pos_candidates"])
    gold_bnids = [b for b, _ in cands]
    # Render failures propagate: convert_all must replace the page (a half-finished navigation would
    # otherwise interrupt the next step's goto) and retries once.
    frac = random.Random(row["action_uid"]).uniform(0.05, 0.95)
    snap, mapping, scroll = await snapshot_page(page, row["raw_html"], synthetic_url(row["website"]), gold_bnids, frac)
    elements = [a for a in snap["actions"] if "node" in a]
    by_node = {}
    for a in elements:
        by_node.setdefault(a["node"], []).append(a)

    matched, reasons = [], []
    for b in gold_bnids:
        a, how = match_gold(mapping["golds"][b], by_node, op, op_raw.get("value", ""))
        (matched.append((b, a, how)) if a else reasons.append(how))
    info["n_elements"] = len(elements)
    info["seconds"] = time.time() - t0
    if not matched:
        reason = reasons[0]
        if reason in ("gold_not_interactive", "gold_not_in_viewport") and snap.get("omitted_actions"):
            reason = "gold_maybe_beyond_250"
        return None, reason, info

    primary = matched[0]
    gold_ids = [primary[1]["id"]]
    multi = False
    for b, a, how in matched[1:]:
        if a["id"] in gold_ids:
            continue
        if a["node"] == primary[1]["node"] or await page.evaluate(NESTED_JS, [a["node"], primary[1]["node"]]):
            gold_ids.append(a["id"])  # nested pos_candidates = the same control
            multi = True

    meta = {
        "source": "mind2web",
        "dataset": "osunlp/Multimodal-Mind2Web",
        "split": split,
        "website": row["website"],
        "domain": row["domain"],
        "subdomain": row["subdomain"],
        "annotation_id": row["annotation_id"],
        "action_uid": row["action_uid"],
        "step_index": step,
        "n_steps_in_task": len(row["action_reprs"]),
        "n_actions_in_snapshot": len(elements),
        "omitted_actions": snap.get("omitted_actions", 0),
        "gold_op_raw": op_raw,
        "gold_action_repr": row["target_action_reprs"],
        "gold_backend_node_ids": [b for b, _, _ in matched],
        "gold_match": primary[2],
        "gold_scrolled_into_view": bool(scroll and (scroll[0] or scroll[1])),
        "scroll_y": scroll[1] if scroll else None,
        "gold_viewport_frac": round(frac, 3),
        "gold_via_ancestor": primary[2] == "ancestor" or any(mapping["bn"][str(a["node"])]["via"]
                                                           for _, a, _ in matched),
        "multi_gold": multi,
        "url_synthetic": True,
        "viewport": [VIEWPORT["width"], VIEWPORT["height"]],
    }
    snapshot = {k: snap[k] for k in ("url", "title", "text")}
    snapshot["actions"] = snap["actions"]
    rec = make_record(snapshot, row["confirmed_task"], build_history(row["action_reprs"], step), gold_ids, meta)
    info["seconds"] = time.time() - t0
    if rec is None:
        return None, "gold_answers_none", info
    return rec, "kept", info


async def convert_all(rows, split, on_result, concurrency=3, timeout=90):
    """Convert rows with `concurrency` pages in one headless Chromium; calls on_result(i, (rec, reason, info)).

    Each step has a hard timeout (a pathological page must not stall the run); the page is then
    replaced, and the browser relaunched if it died.
    """
    from playwright.async_api import async_playwright
    jobs = enumerate(rows)  # shared lazy iterator; rows may be a generator
    async with async_playwright() as pw:
        state = {"browser": await pw.chromium.launch(headless=True)}

        async def fresh_page(old=None):
            if old is not None:
                try:
                    await asyncio.wait_for(old.close(), 10)
                except Exception:  # noqa: BLE001
                    pass
            if not state["browser"].is_connected():
                state["browser"] = await pw.chromium.launch(headless=True)
            return await new_page(state["browser"])

        async def worker():
            page = await fresh_page()
            for i, row in jobs:
                for attempt in range(2):
                    try:
                        result = await asyncio.wait_for(convert_row(page, row, split), timeout)
                        break
                    except Exception as e:  # noqa: BLE001  -- goto timeout, crashed page, closed browser
                        reason = "timeout" if isinstance(e, asyncio.TimeoutError) else "render_error"
                        result = (None, reason, {"op": json.loads(row["operation"]).get("op"), "seconds": None,
                                                 "error": repr(e)[:300], "attempts": attempt + 1})
                        page = await fresh_page(page)
                on_result(i, result)
            await page.close()

        await asyncio.gather(*(worker() for _ in range(max(1, concurrency))))
        await state["browser"].close()


def summarize(results, wall, split, n_workers):
    recs = [r for r, reason, _ in results if r]
    reasons = Counter(reason for _, reason, _ in results)
    n_el = [r["meta"]["n_actions_in_snapshot"] for r in recs]
    secs = [i["seconds"] for _, _, i in results if i.get("seconds")]
    pct = lambda xs, q: sorted(xs)[min(len(xs) - 1, int(q * len(xs)))] if xs else None  # noqa: E731
    buckets = Counter()
    for n in n_el:
        buckets[next(f"{lo}-{hi}" for lo, hi in [(0, 10), (11, 25), (26, 50), (51, 100), (101, 150), (151, 250), (251, 10**9)] if lo <= n <= hi)] += 1
    k = max(1, len(recs))
    return {
        "split": split,
        "rows": len(results),
        "kept": len(recs),
        "kept_pct": round(100 * len(recs) / max(1, len(results)), 1),
        "by_reason": dict(reasons.most_common()),
        "op_dist_all": dict(Counter(i["op"] for _, _, i in results).most_common()),
        "op_dist_kept": dict(Counter(r["meta"]["gold_op_raw"]["op"] for r in recs).most_common()),
        "drop_reason_by_op": dict(
            Counter(f"{i['op']}:{reason}" for _, reason, i in results if reason != "kept").most_common()),
        "gold_operation_kept": dict(Counter(r["gold"]["operation"] for r in recs).most_common()),
        "gold_match_kept": dict(Counter(r["meta"]["gold_match"] for r in recs).most_common()),
        "n_elements": {"min": min(n_el, default=None), "p10": pct(n_el, .1), "median": pct(n_el, .5),
                       "p90": pct(n_el, .9), "max": max(n_el, default=None),
                       "mean": round(statistics.mean(n_el), 1) if n_el else None, "buckets": dict(sorted(buckets.items(), key=lambda x: int(x[0].split('-')[0])))},
        "omitted_actions_gt0_pct": round(100 * sum(r["meta"]["omitted_actions"] > 0 for r in recs) / k, 1),
        "gold_scrolled_into_view_pct": round(100 * sum(r["meta"]["gold_scrolled_into_view"] for r in recs) / k, 1),
        "gold_via_ancestor_pct": round(100 * sum(r["meta"]["gold_via_ancestor"] for r in recs) / k, 1),
        "multi_gold_pct": round(100 * sum(r["meta"]["multi_gold"] for r in recs) / k, 1),
        "seconds_per_step_per_worker": {"mean": round(statistics.mean(secs), 3) if secs else None,
                                        "median": round(pct(secs, .5), 3) if secs else None,
                                        "p90": round(pct(secs, .9), 3) if secs else None},
        "error_samples": [i["error"] for _, _, i in results if i.get("error")][:5],
        "workers": n_workers,
        "wall_seconds": round(wall, 1),
        "wall_seconds_per_step": round(wall / max(1, len(results)), 3),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", required=True, help="train | test_website | test_domain | test_task")
    ap.add_argument("--limit", type=int, help="first N rows")
    ap.add_argument("--sample", type=int, help="random N rows (seeded)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3, help="concurrent pages in one Chromium")
    ap.add_argument("--timeout", type=float, default=90, help="hard per-step timeout (s)")
    ap.add_argument("--resume", action="store_true", help="skip rows already in <out>.log.jsonl and append")
    ap.add_argument("--retry-errors", action="store_true", help="with --resume: redo render_error/timeout rows")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", help="stats.json to merge into (default: <out dir>/stats.json)")
    ap.add_argument("--samples-out", help="also write the first 5 records pretty-printed here")
    a = ap.parse_args(argv)

    uids = select_uids(a.split, a.limit, a.sample, a.seed)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_name(out.stem + ".log.jsonl")  # one line per processed row (kept or dropped)
    done = set()
    if a.resume and log_path.exists():
        last = {}
        for line in log_path.open(encoding="utf-8"):
            if line.strip():
                e = json.loads(line)
                last[e["action_uid"]] = e["reason"]
        redo = {"render_error", "timeout", "worker_error"} if a.retry_errors else set()
        done = {u for u, reason in last.items() if reason not in redo}
    todo = [u for u in uids if u not in done]
    started = []  # action_uid of the i-th row handed to convert_all

    def rows():
        for r in iter_rows(a.split, todo):
            started.append(r["action_uid"])
            yield r
    mode = "a" if a.resume else "w"
    print(f"[{a.split}] {len(uids)} rows, {len(done)} already done, {len(todo)} to go", file=sys.stderr)
    t0, n = time.time(), [0, 0]
    with open(out, mode, encoding="utf-8") as f, open(log_path, mode, encoding="utf-8") as log:
        def on_result(i, result):
            rec, reason, info = result
            if rec:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            log.write(json.dumps({"action_uid": started[i], "reason": reason, "info": info}) + "\n")
            log.flush()
            n[0] += 1
            n[1] += reason == "kept"
            if n[0] % 50 == 0 or n[0] == len(todo):
                print(f"[{a.split}] {n[0]}/{len(todo)} kept={n[1]} {time.time() - t0:.0f}s", file=sys.stderr)

        asyncio.run(convert_all(rows(), a.split, on_result, a.workers, a.timeout))
    wall = time.time() - t0

    wanted = set(uids)
    recs, logged = {}, {}
    for line in out.open(encoding="utf-8"):
        r = json.loads(line)
        if r["meta"]["action_uid"] in wanted:
            recs[r["meta"]["action_uid"]] = {"meta": r["meta"], "gold": r["gold"]}
    for line in log_path.open(encoding="utf-8"):
        e = json.loads(line)
        if e["action_uid"] in wanted:
            logged[e["action_uid"]] = e
    results = [(recs.get(u) if e["reason"] == "kept" else None, e["reason"], e["info"]) for u, e in logged.items()]
    summary = summarize(results, wall, a.split, a.workers)
    summary["wall_seconds_per_step"] = round(wall / max(1, len(todo)), 3)  # this invocation only
    stats_path = pathlib.Path(a.stats) if a.stats else out.parent / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    stats[out.stem] = summary
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    if a.samples_out:
        samples = []
        with open(out, encoding="utf-8") as fh:
            for line in fh:
                samples.append(json.loads(line))
                if len(samples) == 5:
                    break
        pathlib.Path(a.samples_out).write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("rows", "kept", "kept_pct", "by_reason")}, indent=1), file=sys.stderr)


if __name__ == "__main__":
    main()
