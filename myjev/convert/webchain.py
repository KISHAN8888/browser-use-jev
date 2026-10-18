"""WebChain v2 (webagentlab/webchain, CC-BY-4.0) -> Jev records via the real snapshot.js.

Route ("html_snapshot"): every WebChain step links a full pre-action DOM dump (`html_dom_url`,
outerHTML of the live page with inline <style> blocks and `data-imean-axt-id`s). We render it in
headless Chromium (scripts stripped, network blocked except cached stylesheets), locate the gold element
with the recorded CSS `selector` (checked against the recorded tag-only `dom_path`), scroll it to its
recorded viewport height, run jev-ultrafast's real `myjev/vendor/snapshot.js` and map the gold to snapshot
action ids with Mind2Web's matcher (`mind2web.match_gold`: element itself, nearest snapshot ancestor,
unique descendant). The AX-tree files are not needed: they exist for exactly the same steps as the HTML,
so they cannot rescue a step whose HTML is missing.

Usage:
    python -m myjev.convert.webchain --limit 20000 --per-site-cap 80 --workers 2 --resume \
        --out data/processed/webchain

Outputs in --out: train.jsonl, samples.json, stats.json, plus progress.log.jsonl (one line per processed
step: kept or dropped + history label) which makes the run resumable after a kill.

Sampling (diversity first): trajectories are taken round-robin over `primary_host`s (English queries
first within a host, seeded shuffle otherwise); at most --per-site-cap kept steps per host.

Mapping of WebChain actions: click -> CLICK, hover -> CLICK (flag `hover_as_click`), type -> TYPE_TEXT,
select (selector points at an <option>) -> SELECT on its <select>. drag/press_enter/paste/copy/
double_click/right_click/back/launchApp have no Jev equivalent and are dropped (and left out of history).
DONE: WebChain has no success/completion field and no page state after the last action (each HTML is
the state *before* its action), so no DONE steps are synthesised (`done_synthetic` would mark them).

Fidelity notes (also in stats.json):
- Stylesheets are re-fetched from their original URLs (static CDN assets, cached on disk, --no-css to
  disable); many are gone (403/404), then the page renders with inline <style> only.
- Scripts are stripped, so input .value typed earlier is only present if serialized as an attribute.
- Viewport = recorded viewport clamped to 1024..1920 x 600..1200.
- `layout_match`: rendered gold box within 25% of the recorded box size (a cheap fidelity check).
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import ctypes
import gzip
import hashlib
import json
import os
import pathlib
import random
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from urllib.parse import urljoin

from ..jevformat import make_record
from .axcommon import dedupe_key, history_item
from .mind2web import MAP_JS, SCROLL_JS, SNAPSHOT_JS, match_gold

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "webchain"
META_DIR = RAW_DIR / "meta"
HTML_DIR = RAW_DIR / "html"
CSS_DIR = RAW_DIR / "css"
HF_BASE = "https://huggingface.co/datasets/webagentlab/webchain/resolve/main/data/seed_sft/metadata/"
ACTION_COLS = ["trace_uid", "source_step_index", "action_type", "html_dom_url", "selector", "dom_path", "value",
               "scroll_x", "scroll_y", "page_x", "page_y",
               "title", "href", "viewport", "rect", "input_text", "attributes"]
TRACE_COLS = ["uid", "query", "primary_host", "web_type", "intent_type", "source_trace_id"]
OP_MAP = {"click": "CLICK", "hover": "HOVER", "type": "TYPE", "select": "SELECT"}  # mind2web.OP_KIND keys
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) myjev-webchain-converter (research; cached, polite)"
MAX_HTML = 15 << 20
MAX_CSS = 4 << 20
MAX_CSS_PER_PAGE = 12


# ---------------------------------------------------------------- pure helpers (unit-tested)

VERB = re.compile(r"^\s*(点击|输入|选择|悬停|拖动|Click|Input|Select|Hover|Type|Drag)\b\s*", re.I)


def strip_verb(title):
    """'点击 新加坡 (SIN)' / 'Click FILTERS' / 'Input liverpool' -> the object part."""
    return VERB.sub("", title or "", count=1).strip()


_SCRIPTS = [("zh", re.compile(r"[一-鿿]")), ("ja", re.compile(r"[぀-ヿ]")),
            ("ko", re.compile(r"[가-힯]")), ("ru", re.compile(r"[Ѐ-ӿ]")),
            ("ar", re.compile(r"[؀-ۿ]")), ("th", re.compile(r"[฀-๿]"))]


_EN_WORDS = {"the", "a", "an", "to", "for", "and", "of", "in", "on", "with", "me", "my", "find", "from", "is",
             "are", "search", "please", "help", "that", "this", "under", "at", "by", "or", "can", "you", "i"}


def detect_language(text, threshold=0.3):
    """Cheap guess. Mostly-ASCII text is 'en' if it has an English function word (else 'other': French,
    German...); otherwise the dominant non-Latin script (zh/ja/ko/ru/ar/th) or 'other'. A few quoted CJK
    words inside an English task ("find '1段奶粉' formula") stay 'en'."""
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return "unknown"
    non_ascii = [c for c in letters if ord(c) > 127]
    if len(non_ascii) / len(letters) < threshold:
        words = re.findall(r"[a-z]+", (text or "").lower())
        return "en" if len(words) < 4 or _EN_WORDS.intersection(words) else "other"
    s = "".join(non_ascii)
    counts = {lang: len(rx.findall(s)) for lang, rx in _SCRIPTS}
    if "ja" in counts and counts["ja"] > 0 and counts["ja"] >= counts["zh"] * 0.2:
        return "ja"
    best = max(counts, key=counts.get)
    return best if counts[best] else "other"


_SCRIPT_TAG = re.compile(r"<script\b[^>]*>.*?</script\s*>", re.S | re.I)
_REFRESH = re.compile(r"<meta[^>]+http-equiv\s*=\s*[\"']?refresh[^>]*>", re.I)


def prepare_html(html):
    """Strip <script> blocks and meta refresh so the dump renders as captured (no app code re-running)."""
    return _REFRESH.sub("", _SCRIPT_TAG.sub("", html))


_LINK = re.compile(r"<link\b[^>]*>", re.I)
_ATTR = re.compile(r"""([\w:-]+)\s*=\s*("[^"]*"|'[^']*'|[^\s"'>]+)""")
_BASE = re.compile(r"<base\b[^>]*href\s*=\s*[\"']([^\"']+)", re.I)


def stylesheet_urls(html, page_url):
    """Absolute stylesheet URLs linked from the dump (rel=stylesheet, or preload as=style)."""
    m = _BASE.search(html)
    base = urljoin(page_url, m.group(1)) if m else page_url
    out = []
    for tag in _LINK.findall(html):
        attrs = {k.lower(): v.strip("\"'") for k, v in _ATTR.findall(tag)}
        rel = attrs.get("rel", "").lower().split()
        if "stylesheet" in rel or ("preload" in rel and attrs.get("as", "").lower() == "style"):
            href = attrs.get("href", "").replace("&amp;", "&")
            if href and not href.startswith("data:"):
                u = urljoin(base, href)
                if u.startswith("http") and u not in out:
                    out.append(u)
    return out[:MAX_CSS_PER_PAGE]


def clamp_viewport(vp):
    w = int((vp or {}).get("width") or 1280)
    h = int((vp or {}).get("height") or 900)
    return {"width": min(max(w, 1024), 1920), "height": min(max(h, 600), 1200)}


def gold_frac(rect, viewport, seed):
    """SCROLL_JS puts the gold top at frac*(innerHeight - h - 20). Use the recorded viewport position
    (the human's real scroll), else a seeded random height (never a fixed one: that would leak)."""
    try:
        y, h = float(rect["y"]), float(rect["height"])
        span = viewport["height"] - h - 20
        if span > 0 and 0 <= y <= viewport["height"]:
            return round(min(1.0, max(0.0, y / span)), 4), True
    except (TypeError, KeyError, ValueError):
        pass
    return round(random.Random(seed).uniform(0.05, 0.95), 4), False


def _jl(s, default=None):
    if not s:
        return default
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return default


def fallback_label(step):
    """History label for a step that was not rendered (or dropped): element text from the metadata."""
    attrs = (_jl(step.get("attributes"), {}) or {}).get("data", {}) or {}
    for cand in (attrs.get("aria-label"), step.get("value") if step["action_type"] != "type" else None,
                 strip_verb(step.get("title"))):
        if cand and str(cand).strip():
            return str(cand).strip()[:200]
    role = attrs.get("role") or (attrs.get("node") or {}).get("name", "element")
    return str(role).lower() or "element"


def history_entry(step, label=None):
    """jev-ultrafast history item for a previous step (same shape as the other converters)."""
    op = step["action_type"]
    if op not in OP_MAP:
        return None
    label = label or fallback_label(step)
    if op == "type":
        return history_item(label, "fill", step.get("input_text") or step.get("value") or "")
    if op == "select":
        opt = step.get("input_text") or step.get("value") or ""
        if opt and " → " not in label:
            label = f"{label} → {opt}"
        return history_item(label, "select")
    return history_item(label, "click")


def plan_order(traces, seed=0):
    """Round-robin order of trace uids over hosts: every host gets its 1st trajectory before any gets a 2nd.

    traces: iterable of dicts with uid, primary_host, language. English tasks first within a host."""
    rng = random.Random(seed)
    by_host = collections.defaultdict(list)
    for t in traces:
        by_host[t["primary_host"]].append(t)
    queues = {}
    for host, ts in by_host.items():
        ts = sorted(ts, key=lambda t: t["uid"])
        rng.shuffle(ts)
        ts.sort(key=lambda t: t.get("language") != "en")  # stable: English first, shuffled within
        queues[host] = collections.deque(t["uid"] for t in ts)
    hosts = sorted(queues)
    rng.shuffle(hosts)
    order = []
    while hosts:
        nxt = []
        for h in hosts:
            order.append(queues[h].popleft())
            if queues[h]:
                nxt.append(h)
        hosts = nxt
    return order


def free_memory_gb(commit=False):
    """Available physical memory in GB (what PowerShell's Get-CimInstance Win32_OperatingSystem reports as
    FreePhysicalMemory, plus standby cache), or with commit=True the remaining commit charge (RAM + pagefile):
    Windows kills/fails allocations when commit runs out, which is what happened to earlier jobs."""
    if sys.platform == "win32":
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return (m.ullAvailPageFile if commit else m.ullAvailPhys) / 2**30
    try:
        import psutil
        return psutil.virtual_memory().available / 2**30
    except ImportError:
        return 99.0


# ---------------------------------------------------------------- metadata

def load_metadata():
    """(traces: {uid: dict}, steps: {uid: [step dicts sorted by step index]}). Downloads ~530 MB once."""
    import pyarrow.parquet as pq
    META_DIR.mkdir(parents=True, exist_ok=True)
    slim = META_DIR / "actions_slim.parquet"
    for name in ("traces.parquet", "actions.parquet"):
        p = META_DIR / name
        if not p.exists() and not (name == "actions.parquet" and slim.exists()):
            print(f"[meta] downloading {name}", file=sys.stderr)
            tmp = p.with_suffix(".tmp")
            urllib.request.urlretrieve(HF_BASE + name, tmp)
            os.replace(tmp, p)
    if not slim.exists():
        t = pq.read_table(META_DIR / "actions.parquet", columns=ACTION_COLS)
        pq.write_table(t, slim.with_suffix(".tmp"), compression="zstd")
        os.replace(slim.with_suffix(".tmp"), slim)
    traces = {r["uid"]: r for r in pq.read_table(META_DIR / "traces.parquet", columns=TRACE_COLS).to_pylist()}
    for r in traces.values():
        r["language"] = detect_language(r["query"])
    return traces, Steps(pq.read_table(slim))


class Steps:
    """Read-only {trace_uid: [step dicts sorted by step index]} backed by one sorted Arrow table.

    Only Jev-mappable steps (click/hover/type/select) are kept; the others are neither converted nor put
    in history. Materialising all 317k steps as Python dicts took 1.4 GB, too much next to other jobs;
    this keeps ~150 MB and builds one trajectory's dicts on demand."""

    def __init__(self, table):
        import pyarrow as pa
        import pyarrow.compute as pc
        self.unsupported = collections.Counter(
            {k: v for k, v in zip(*[c.to_pylist() for c in pc.value_counts(table["action_type"]).flatten()])
             if k not in OP_MAP})
        table = table.filter(pc.is_in(table["action_type"], value_set=pa.array(list(OP_MAP))))
        self.table = table.sort_by([("trace_uid", "ascending"), ("source_step_index", "ascending")])
        uids = self.table["trace_uid"].to_pylist()
        has_html = pc.not_equal(self.table["html_dom_url"], "").to_pylist()
        self.index, self.usable = {}, set()
        for i, u in enumerate(uids):
            start, n = self.index.get(u, (i, 0))
            self.index[u] = (start, n + 1)
            if has_html[i]:
                self.usable.add(u)
        del uids, has_html, table
        pa.default_memory_pool().release_unused()  # give the read/sort scratch (~500 MB) back to the OS

    def __contains__(self, uid):
        return uid in self.index

    def __getitem__(self, uid):
        start, n = self.index[uid]
        return self.table.slice(start, n).to_pylist()

    def __len__(self):
        return len(self.index)


# ---------------------------------------------------------------- polite cached fetching

class Fetcher:
    """Disk-cached HTTP GET with bounded concurrency (<= 8) and retries. Bodies are stored gzip-compressed;
    dead links are cached as `.dead` markers (with the reason) so they are never re-requested. Concurrent
    requests for the same URL wait for one download."""

    def __init__(self, root, concurrency=8, retries=3, max_bytes=MAX_HTML, sem=None, timeout=40):
        self.root, self.retries, self.max_bytes, self.timeout = pathlib.Path(root), retries, max_bytes, timeout
        self.host_timeouts = collections.Counter()  # breaker: stop asking hosts that keep timing out
        self.sem = sem or threading.BoundedSemaphore(max(1, min(8, concurrency)))  # share to cap total
        self.counts = collections.Counter()
        self._locks, self._guard = {}, threading.Lock()

    def _path(self, url):
        h = hashlib.sha1(url.encode("utf-8")).hexdigest()
        return self.root / h[:2] / h

    def ensure(self, url):
        """Download into the cache if needed. Returns None if cached/available, else the dead reason."""
        p = self._path(url)
        gz, dead = p.with_suffix(".gz"), p.with_suffix(".dead")
        if gz.exists():
            return None
        if dead.exists():
            return dead.read_text(encoding="utf-8")
        with self._guard:
            lock = self._locks.setdefault(url, threading.Lock())
        with lock:
            try:
                if gz.exists():
                    return None
                if dead.exists():
                    return dead.read_text(encoding="utf-8")
                return self._download(url, p)
            finally:
                with self._guard:
                    self._locks.pop(url, None)

    def _download(self, url, p):
        gz, dead = p.with_suffix(".gz"), p.with_suffix(".dead")
        reason = "fetch_error"
        netloc = urllib.parse.urlsplit(url).netloc
        if self.host_timeouts[netloc] >= 3:
            self.counts["host_skipped"] += 1
            return "host_unresponsive"  # not cached as dead: a later run may retry
        for attempt in range(self.retries):
            try:
                with self.sem:
                    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "identity"})
                    with urllib.request.urlopen(req, timeout=self.timeout) as r:
                        body = r.read(self.max_bytes + 1)
                if len(body) > self.max_bytes:
                    reason = "too_large"
                    break
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_suffix(f".tmp{threading.get_ident()}")
                tmp.write_bytes(gzip.compress(body, 6))
                os.replace(tmp, gz)
                self.counts["fetched"] += 1
                return None
            except urllib.error.HTTPError as e:
                reason = f"http_{e.code}"
                if e.code in (400, 401, 403, 404, 410, 451):
                    break
            except Exception as e:  # noqa: BLE001  -- timeouts, resets, DNS
                reason = f"net_{type(e).__name__}"
                if "timeout" in reason.lower() or "timed out" in repr(e):
                    self.host_timeouts[netloc] += 1
            time.sleep(1.5 * (attempt + 1))
        if reason.startswith("http_4") or reason == "too_large":
            p.parent.mkdir(parents=True, exist_ok=True)
            dead.write_text(reason, encoding="utf-8")  # permanent: never re-requested
            self.counts["dead"] += 1
        else:
            self.counts["transient_fail"] += 1  # network/5xx: not cached, retried by a later --resume
        return reason

    def get(self, url):
        """Returns (text, None) or (None, reason)."""
        err = self.ensure(url)
        if err:
            return None, err
        gz = self._path(url).with_suffix(".gz")
        return gzip.decompress(gz.read_bytes()).decode("utf-8", errors="replace"), None


# ---------------------------------------------------------------- browser side

# Mark the gold with backend_node_id="gold" so Mind2Web's SCROLL_JS / MAP_JS work unchanged.
# 1. the recorded CSS selector, accepted if its tag path equals the recorded tag-only dom_path
#    (several hits: the one whose page box best overlaps the recorded box);
# 2. a unique selector hit whose tag path differs;
# 3. elements on the dom_path: unique text match, else best box overlap (IoU >= 0.5) with the recorded
#    page box (CSS-in-JS dumps often have class names that no longer match the recorded selector),
#    else the only element on that path.
# A clicked <label> is replaced by its control (clicking a label activates the control).
FIND_JS = r"""([sel, path, text, box]) => {
  const tp = e => { const t=[]; for (let x=e; x && x!==document.documentElement; x=x.parentElement) t.unshift(x.tagName.toLowerCase()); return t.join('>'); };
  const norm = s => (s||'').replace(/\s+/g,' ').trim().toLowerCase();
  const iou = e => {
    if (!box) return 0;
    const r = e.getBoundingClientRect(), x = r.left + scrollX, y = r.top + scrollY;
    const ix = Math.max(0, Math.min(x + r.width, box[0] + box[2]) - Math.max(x, box[0]));
    const iy = Math.max(0, Math.min(y + r.height, box[1] + box[3]) - Math.max(y, box[1]));
    const inter = ix * iy, uni = r.width * r.height + box[2] * box[3] - inter;
    return uni > 0 ? inter / uni : 0;
  };
  const best = es => { let b = null, s = -1; for (const e of es) { const v = iou(e); if (v > s) { s = v; b = e; } } return [b, s]; };
  let cands = [];
  try { cands = sel ? [...document.querySelectorAll(sel)] : []; } catch (_) {}
  let hits = path ? cands.filter(e => tp(e) === path) : cands, how = 'selector', score = null;
  if (hits.length > 1 && box) { const [b, s] = best(hits); if (s > 0) { hits = [b]; how = 'selector_box'; score = s; } }
  if (!hits.length && cands.length === 1) { hits = cands; how = 'selector_path_mismatch'; }
  if (!hits.length && path) {
    let byPath = [];
    try { byPath = [...document.querySelectorAll(path.split('>').join(' > '))]; } catch (_) {}
    const t = norm(text);
    const m = t ? byPath.filter(e => norm(e.innerText || e.value || e.getAttribute('aria-label')) === t) : [];
    const [b, s] = byPath.length ? best(byPath) : [null, 0];
    if (m.length === 1) { hits = m; how = 'dom_path_text'; }
    else if (b && s >= 0.5) { hits = [b]; how = 'dom_path_box'; score = s; }
    else if (byPath.length === 1) { hits = byPath; how = 'dom_path_unique'; }
  }
  if (!hits.length) return {how: null, n_selector: cands.length};
  let e = hits[0];
  if (e.tagName === 'OPTION' && e.closest('select')) e = e.closest('select');
  if (e.tagName === 'LABEL' && e.control) { e = e.control; how += '+label_control'; }
  for (const x of document.querySelectorAll('[backend_node_id]')) x.removeAttribute('backend_node_id');
  e.setAttribute('backend_node_id', 'gold');
  const opt = hits[0].tagName === 'OPTION' ? (hits[0].label || hits[0].textContent || '').trim() : null;
  return {how, n_selector: cands.length, n_hits: hits.length, tag: e.tagName.toLowerCase(), option: opt,
          iou: score === null ? null : Math.round(score * 100) / 100};
}"""

POINT_JS = """([nodes, px, py]) => { const M = window.__jevFast.nodes;
  return nodes.filter(n => { const e = M.get(n); if (!e) return false; const r = e.getBoundingClientRect();
    const x = r.left + scrollX, y = r.top + scrollY; return px >= x && px <= x + r.width && py >= y && py <= y + r.height; }); }"""

RECT_JS = """() => { const e = document.querySelector('[backend_node_id="gold"]');
  if (!e) return null; const r = e.getBoundingClientRect(); return [r.x, r.y, r.width, r.height]; }"""


class Renderer:
    """One headless Chromium, at most 2 pages (each in its own context, recreated every --page-life steps to
    bound memory). Serves the dump for the top document, stylesheets from the CSS cache, and aborts every
    other request (offline: no scripts, images, fonts, XHR)."""

    def __init__(self, pw, pages=2):
        self.pw, self.browser = pw, None
        self.n_pages = max(1, min(2, pages))
        self.lock = asyncio.Lock()

    async def ensure_browser(self):
        async with self.lock:
            if self.browser is None or not self.browser.is_connected():
                self.browser = await self.pw.chromium.launch(
                    headless=True, args=["--disable-gpu", "--disable-extensions", "--mute-audio", "--no-first-run",
                                         "--js-flags=--max-old-space-size=512"])

    async def new_page(self):
        await self.ensure_browser()
        ctx = await self.browser.new_context(viewport={"width": 1280, "height": 900}, bypass_csp=True,
                                             java_script_enabled=True, service_workers="block")
        page = await ctx.new_page()
        page.set_default_timeout(30000)
        page._wc = {"html": "", "css": {}}

        async def route(r, *_):
            req = r.request
            try:
                if req.resource_type == "document":
                    if req.frame.parent_frame is None:
                        return await r.fulfill(status=200, content_type="text/html; charset=utf-8",
                                               body=page._wc["html"])
                    return await r.fulfill(status=200, content_type="text/html", body="")  # iframes: empty
                css = page._wc["css"].get(req.url)
                if req.resource_type == "stylesheet" and css is not None:
                    return await r.fulfill(status=200, content_type="text/css; charset=utf-8", body=css)
                await r.abort()
            except Exception:  # noqa: BLE001  -- page closed mid-request
                pass

        await page.route("**/*", route)
        return page

    @staticmethod
    async def close_page(page):
        if page is None:
            return
        try:
            await asyncio.wait_for(page.context.close(), 10)
        except Exception:  # noqa: BLE001
            pass

    async def close(self):
        try:
            await asyncio.wait_for(self.browser.close(), 20)
        except Exception:  # noqa: BLE001
            pass


async def render_step(page, html, css, url, viewport, sel, path, text, frac, box=None):
    """Render one dump and return (find, snapshot, mapping, scroll, rect)."""
    page._wc = {"html": html, "css": css}
    await page.set_viewport_size(viewport)
    try:
        await page.goto(url, wait_until="load", timeout=30000)
    except Exception as e:  # noqa: BLE001
        if "Timeout" not in repr(e):
            raise
        # 'load' can stall on a hung subresource; the DOM is there, carry on.
    find = await page.evaluate(FIND_JS, [sel, path, text, box])
    scroll = None
    if find.get("how"):
        scroll = await page.evaluate(SCROLL_JS, ["gold", frac])
    snap = await page.evaluate(SNAPSHOT_JS)
    if snap is None:
        raise RuntimeError("snapshot returned null (no <body>)")
    nodes = sorted({a["node"] for a in snap["actions"] if "node" in a})
    mapping = await page.evaluate(MAP_JS, {"nodes": nodes, "gold": ["gold"] if find.get("how") else []})
    rect = await page.evaluate(RECT_JS) if find.get("how") else None
    return find, snap, mapping, scroll, rect


# ---------------------------------------------------------------- conversion of one step

def page_box(rect, scroll_y, scroll_x=0):
    """Recorded viewport rect + recorded scroll -> page-coordinate box [x, y, w, h] (None if unknown)."""
    try:
        return [float(rect["x"]) + float(scroll_x or 0), float(rect["y"]) + float(scroll_y or 0),
                float(rect["width"]), float(rect["height"])]
    except (TypeError, KeyError, ValueError):
        return None


def layout_match(rendered, recorded, tol=0.25):
    try:
        w, h = rendered[2], rendered[3]
        rw, rh = float(recorded["width"]), float(recorded["height"])
        return bool(rw > 0 and rh > 0 and abs(w - rw) <= tol * rw + 2 and abs(h - rh) <= tol * rh + 2)
    except (TypeError, KeyError, ValueError, IndexError):
        return None


async def convert_step(page, step, trace, history, html_text, css_map, css_info):
    """Returns (record_or_None, reason, info). info carries the history label of the gold."""
    t0 = time.time()
    op = OP_MAP[step["action_type"]]
    info = {"op": step["action_type"], "seconds": None}
    viewport = clamp_viewport(_jl(step["viewport"], {}))
    rect = _jl(step["rect"], {})
    uid_step = f"{trace['uid']}:{step['source_step_index']}"
    frac, frac_recorded = gold_frac(rect, viewport, uid_step)
    elem_text = step.get("value") if step["action_type"] in ("click", "hover") else ""
    find, snap, mapping, scroll, grect = await render_step(
        page, html_text, css_map, step["href"] or f"https://{trace['primary_host']}/", viewport,
        step["selector"] or "", step["dom_path"] or "", elem_text or "", frac, page_box(rect, step.get("scroll_y"), step.get("scroll_x")))
    elements = [a for a in snap["actions"] if "node" in a]
    info["n_elements"] = len(elements)
    if not find.get("how"):
        info["seconds"] = time.time() - t0
        return None, "gold_not_found_in_dom", info
    by_node = collections.defaultdict(list)
    for a in elements:
        by_node[a["node"]].append(a)
    value = find.get("option") or step.get("input_text") or step.get("value") or ""
    action, how = match_gold(mapping["golds"]["gold"], by_node, op, value)
    if how == "ambiguous_descendants" and step.get("page_x") is not None:
        # The human clicked a container holding several controls: keep the one under the recorded click point.
        kind = {"TYPE": "fill", "SELECT": "select"}.get(op, "click")
        desc = [n for n in mapping["golds"]["gold"].get("desc", []) if any(a["kind"] == kind for a in by_node.get(n, []))]
        under = await page.evaluate(POINT_JS, [desc, step["page_x"], step["page_y"]])
        if len(under) == 1:
            action, how = next(a for a in by_node[under[0]] if a["kind"] == kind), "descendant_point"
    info["seconds"] = time.time() - t0
    if action is None:
        reason = how
        if reason in ("gold_not_interactive", "gold_not_in_viewport") and snap.get("omitted_actions"):
            reason = "gold_maybe_beyond_250"
        return None, reason, info
    info["label"] = action["label"]
    page_text = snap.get("text", "")
    meta = {
        "source": "webchain",
        "dataset": "webagentlab/webchain",
        "route": "html_snapshot",
        "website": trace["primary_host"],
        "host": (re.sub(r"^https?://([^/]+).*$", r"\1", step["href"] or "") or None),
        "web_type": trace.get("web_type"),
        "intent_type": trace.get("intent_type"),
        "task_id": trace["uid"],
        "source_trace_id": trace.get("source_trace_id"),
        "step_index": step["source_step_index"],
        "language": trace["language"],
        "page_language": detect_language(page_text),
        "non_english": trace["language"] != "en" or detect_language(page_text) not in ("en", "unknown"),
        "gold_action_raw": {k: step.get(k) for k in ("action_type", "title", "value", "input_text", "selector",
                                                     "dom_path")},
        "gold_action_raw_rect": rect or None,
        "gold_find": find["how"],
        "gold_find_iou": find.get("iou"),
        "gold_match": how,
        "gold_via_ancestor": how == "ancestor",
        "hover_as_click": step["action_type"] == "hover",
        "n_actions_in_snapshot": len(elements),
        "omitted_actions": snap.get("omitted_actions", 0),
        "scroll_y": scroll[1] if scroll else None,
        "gold_viewport_frac": frac,
        "gold_position_recorded": frac_recorded,
        "layout_match": layout_match(grect, rect),
        "css": css_info,
        "viewport": [viewport["width"], viewport["height"]],
        "done_synthetic": False,
        "input_values_from_attributes_only": True,
        "url_synthetic": not bool(step["href"]),
    }
    snapshot = {k: snap[k] for k in ("url", "title", "text")}
    snapshot["actions"] = snap["actions"]
    rec = make_record(snapshot, trace["query"], history, [action["id"]], meta)
    if rec is None:
        return None, "gold_answers_none", info
    rec["meta"]["dedupe_key"] = dedupe_key(rec)
    return rec, "kept", info


# ---------------------------------------------------------------- driver

class Run:
    def __init__(self, a):
        self.a = a
        self.out = pathlib.Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.train_path, self.log_path = self.out / "train.jsonl", self.out / "progress.log.jsonl"
        self.kept_by_host = collections.Counter()
        self.inflight = collections.Counter()
        self.done = {}      # (trace_uid, step_index) -> log entry
        self.seen = set()   # dedupe keys
        self.kept_total = 0
        if a.resume:
            self._load_resume()
        else:
            for p in (self.train_path, self.log_path):
                if p.exists():
                    p.unlink()
        self.html = Fetcher(HTML_DIR, a.fetch_concurrency)
        # One semaphore for HTML and CSS: never more than --fetch-concurrency (<= 8) open downloads in total.
        self.cssf = Fetcher(CSS_DIR, retries=1, max_bytes=MAX_CSS, sem=self.html.sem, timeout=12)
        from concurrent.futures import ThreadPoolExecutor
        self.pool = ThreadPoolExecutor(max(1, min(8, a.fetch_concurrency)), thread_name_prefix="prefetch")
        self.now_pool = ThreadPoolExecutor(4, thread_name_prefix="now")    # the step being rendered
        self.css_pool = ThreadPoolExecutor(6, thread_name_prefix="css")
        self.t0 = time.time()
        self.n_processed = 0
        self.mem_pauses = 0

    def _load_resume(self):
        if self.log_path.exists():
            for line in self.log_path.open(encoding="utf-8"):
                try:
                    e = json.loads(line)
                except ValueError:
                    continue  # torn last line after a kill
                self.done[(e["trace"], e["step"])] = e
        for k in [k for k, e in self.done.items() if e["reason"] in set(self.a.retry_reasons or [])]:
            del self.done[k]  # redo these (e.g. after a matcher improvement); stats use the last log line
        recs = set()
        if self.train_path.exists():
            good = []
            for line in self.train_path.open(encoding="utf-8"):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue  # torn last line
                if r["meta"]["dedupe_key"] in self.seen:
                    continue
                self.seen.add(r["meta"]["dedupe_key"])
                good.append(line if line.endswith("\n") else line + "\n")
                self.kept_by_host[r["meta"]["website"]] += 1
                key = (r["meta"]["task_id"], r["meta"]["step_index"])
                recs.add(key)
                if self.done.get(key, {}).get("reason") != "kept":  # killed between the two writes
                    self.done[key] = {"trace": key[0], "step": key[1], "reason": "kept", "host": r["meta"]["website"],
                                      "label": None, "info": {"op": r["meta"]["gold_action_raw"]["action_type"]}}
            tmp = self.train_path.with_suffix(".tmp")
            tmp.write_text("".join(good), encoding="utf-8")
            os.replace(tmp, self.train_path)
        for k in [k for k, e in self.done.items() if e["reason"] == "kept" and k not in recs]:
            del self.done[k]  # logged as kept but the record was lost: redo
        self.kept_total = sum(self.kept_by_host.values())
        print(f"[resume] {len(self.done)} steps done, {self.kept_total} kept", file=sys.stderr)

    def full(self):
        return bool(self.a.limit) and self.kept_total >= self.a.limit

    def host_open(self, host):
        return self.kept_by_host[host] + self.inflight[host] < self.a.per_site_cap

    async def wait_memory(self):
        while free_memory_gb() < self.a.min_free_gb or free_memory_gb(commit=True) < self.a.min_commit_gb:
            self.mem_pauses += 1
            print(f"[mem] {free_memory_gb():.1f} GB RAM / {free_memory_gb(True):.1f} GB commit free "
                  f"(< {self.a.min_free_gb} / {self.a.min_commit_gb}); pausing 20s", file=sys.stderr,
                  flush=True)
            await asyncio.sleep(20)

    def fetch_css(self, html, url):
        urls = stylesheet_urls(html, url) if not self.a.no_css else []
        css = {}
        for u, (text, _) in zip(urls, self.css_pool.map(self.cssf.get, urls)):
            if text is not None:
                css[u] = text
        return css, {"linked": len(urls), "loaded": len(css)}

    def prefetch(self, steps_of_trace):
        """Start downloading a trajectory's dumps into the disk cache (nothing is held in memory)."""
        loop = asyncio.get_running_loop()
        for s in steps_of_trace:
            if s["html_dom_url"] and s["action_type"] in OP_MAP and \
                    (s["trace_uid"], s["source_step_index"]) not in self.done:
                loop.run_in_executor(self.pool, self.html.ensure, s["html_dom_url"])

    def emit(self, trace, step, result):
        rec, reason, info = result
        if rec is not None:
            if rec["meta"]["dedupe_key"] in self.seen:
                rec, reason = None, "duplicate"
            else:
                self.seen.add(rec["meta"]["dedupe_key"])
                self.train.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self.train.flush()
                self.kept_by_host[trace["primary_host"]] += 1
                self.kept_total += 1
        e = {"trace": trace["uid"], "step": step["source_step_index"], "reason": reason,
             "host": trace["primary_host"], "label": info.get("label"), "info": info}
        self.log.write(json.dumps(e, ensure_ascii=False) + "\n")
        self.log.flush()
        self.done[(trace["uid"], step["source_step_index"])] = e
        self.n_processed += 1
        if self.n_processed % 100 == 0:
            el = time.time() - self.t0
            print(f"[run] processed={self.n_processed} kept={self.kept_total} hosts={len(self.kept_by_host)} "
                  f"{el / self.n_processed:.2f}s/step free={free_memory_gb():.1f}GB "
                  f"html={dict(self.html.counts)} css={dict(self.cssf.counts)}", file=sys.stderr, flush=True)

    async def run(self, traces, steps):
        from playwright.async_api import async_playwright
        order = plan_order([traces[u] for u in traces if u in steps.usable and
                            (not self.a.english_only or traces[u]["language"] == "en")], self.a.seed)
        print(f"[plan] {len(order)} trajectories over {len({traces[u]['primary_host'] for u in order})} hosts",
              file=sys.stderr)
        queue = collections.deque(order)
        self.train = open(self.train_path, "a", encoding="utf-8", newline="\n")
        self.log = open(self.log_path, "a", encoding="utf-8", newline="\n")

        def next_trace():
            # Skip finished trajectories and hosts at their cap. No await inside: atomic in the event loop.
            while queue:
                uid = queue.popleft()
                if not self.host_open(traces[uid]["primary_host"]):
                    continue
                if all((uid, s["source_step_index"]) in self.done for s in steps[uid]):
                    continue
                return uid
            return None

        async with async_playwright() as pw:
            renderer = Renderer(pw, self.a.workers)

            async def worker():
                page, used = await renderer.new_page(), 0
                while not self.full():
                    uid = next_trace()
                    if uid is None:
                        break
                    trace = traces[uid]
                    tsteps = steps[uid]
                    self.prefetch(tsteps)
                    if queue:
                        self.prefetch(steps[queue[0]])  # one trajectory ahead
                    history = []
                    for step in tsteps:
                        key = (uid, step["source_step_index"])
                        prev = self.done.get(key)
                        if prev is None:
                            if self.full() or not self.host_open(trace["primary_host"]):
                                break  # budget reached: stop this trajectory here
                            if step["action_type"] not in OP_MAP:
                                self.emit(trace, step, (None, f"unsupported_op_{step['action_type']}",
                                                        {"op": step["action_type"]}))
                            else:
                                self.inflight[trace["primary_host"]] += 1
                                try:
                                    result, page = await self.do_step(renderer, page, step, trace, history)
                                finally:
                                    self.inflight[trace["primary_host"]] -= 1
                                self.emit(trace, step, result)
                                used += 1
                            prev = self.done[key]
                        h = history_entry(step, prev.get("label"))
                        if h:
                            history.append(h)
                    if used >= self.a.page_life:
                        await renderer.close_page(page)
                        page, used = await renderer.new_page(), 0
                await renderer.close_page(page)

            await asyncio.gather(*(worker() for _ in range(renderer.n_pages)))
            await renderer.close()
        for pool in (self.pool, self.now_pool, self.css_pool):
            pool.shutdown(wait=False, cancel_futures=True)
        self.train.close()
        self.log.close()

    async def do_step(self, renderer, page, step, trace, history):
        """Returns ((record, reason, info), page) -- the page is replaced after a render failure."""
        info = {"op": step["action_type"]}
        if not step["html_dom_url"]:
            return (None, "no_html_url", info), page
        loop = asyncio.get_running_loop()
        t0 = time.time()
        html, err = await loop.run_in_executor(self.now_pool, self.html.get, step["html_dom_url"])
        info["t_html"] = round(time.time() - t0, 2)
        if html is None:
            info["error"] = err
            return (None, "dead_link", info), page
        t0 = time.time()
        css_map, css_info = await loop.run_in_executor(self.now_pool, self.fetch_css, html, step["href"] or "")
        info["t_css"] = round(time.time() - t0, 2)
        html = prepare_html(html)
        reason = "render_error"
        for attempt in range(2):
            await self.wait_memory()
            try:
                result = await asyncio.wait_for(
                    convert_step(page, step, trace, list(history[-10:]), html, css_map, css_info), self.a.timeout)
                result[2].update(t_html=info["t_html"], t_css=info["t_css"])
                return result, page
            except Exception as e:  # noqa: BLE001  -- goto timeout, crashed page, closed browser
                info.update(error=repr(e)[:300], attempts=attempt + 1)
                reason = "timeout" if isinstance(e, asyncio.TimeoutError) else "render_error"
                await renderer.close_page(page)
                page = await renderer.new_page()
        return (None, reason, info), page


# ---------------------------------------------------------------- stats / samples

def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def _rate(flags):
    return {"n": len(flags), "pct": round(100 * sum(bool(f) for f in flags) / len(flags), 1) if flags else None}


def write_stats(out, traces, steps, run_info):
    out = pathlib.Path(out)
    log, recs = {}, []
    for line in (out / "progress.log.jsonl").open(encoding="utf-8"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        log[(e["trace"], e["step"])] = e
    samples, sample_hosts = [], set()
    for line in (out / "train.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        m = r["meta"]
        recs.append({"website": m["website"], "op": r["gold"]["operation"], "raw": m["gold_action_raw"]["action_type"],
                     "n": len(r["request"]["state"]["elements"]), "lang": m["language"], "plang": m["page_language"],
                     "route": m["route"], "find": m["gold_find"], "match": m["gold_match"],
                     "layout": m["layout_match"], "css": m["css"], "recorded": m["gold_position_recorded"],
                     "omitted": m["omitted_actions"], "hist": len(r["request"]["state"]["recent_actions"])})
        # 5 samples on 5 different websites, a new operation whenever possible, small pages preferred.
        if len(samples) < 5 and m["website"] not in sample_hosts and len(r["request"]["state"]["elements"]) <= 60:
            ops = {s["gold"]["operation"] for s in samples}
            if r["gold"]["operation"] not in ops or len(recs) > 3000:
                samples.append(r)
                sample_hosts.add(m["website"])
    reasons = collections.Counter(e["reason"] for e in log.values())
    per_site = collections.Counter(r["website"] for r in recs)
    secs = [e["info"].get("seconds") for e in log.values() if e["info"].get("seconds")]
    n_el = [r["n"] for r in recs]
    k = max(1, len(recs))
    dead = sum(1 for e in log.values() if e["reason"] == "dead_link")
    fetch_attempts = sum(1 for e in log.values() if e["reason"] not in ("no_html_url",)
                         and not e["reason"].startswith("unsupported_op_"))
    css_linked = sum(r["css"]["linked"] for r in recs)
    stats = {
        "source": "webchain",
        "dataset": "webagentlab/webchain (CC-BY-4.0)",
        "route": dict(collections.Counter(r["route"] for r in recs)),
        "steps_processed": len(log),
        "kept": len(recs),
        "dropped_total": len(log) - reasons["kept"],
        "dropped_by_reason": {k_: v for k_, v in reasons.most_common() if k_ != "kept"},
        "drop_reason_by_op": dict(collections.Counter(f"{e['info'].get('op')}:{e['reason']}" for e in log.values()
                                                      if e["reason"] != "kept").most_common(40)),
        "operation_distribution": dict(collections.Counter(r["op"] for r in recs).most_common()),
        "raw_action_distribution_kept": dict(collections.Counter(r["raw"] for r in recs).most_common()),
        "raw_action_distribution_processed": dict(collections.Counter(e["info"].get("op") for e in log.values()).most_common()),
        "distinct_websites": len(per_site),
        "websites_available": len({t["primary_host"] for u, t in traces.items() if u in steps.usable}),
        "trajectories_available": len(steps.usable),
        "unsupported_actions_in_dataset_not_converted": dict(steps.unsupported.most_common()),
        "trajectories_touched": len({t for t, _ in log}),
        "steps_per_site": {"min": min(per_site.values(), default=0), "p25": pct(list(per_site.values()), .25),
                           "median": pct(list(per_site.values()), .5), "p75": pct(list(per_site.values()), .75),
                           "max": max(per_site.values(), default=0),
                           "sites_at_cap": sum(v >= run_info["per_site_cap"] for v in per_site.values()),
                           "top": dict(per_site.most_common(15))},
        "languages_task": dict(collections.Counter(r["lang"] for r in recs).most_common()),
        "languages_page": dict(collections.Counter(r["plang"] for r in recs).most_common()),
        "elements_per_step": {"min": min(n_el, default=None), "p25": pct(n_el, .25), "median": pct(n_el, .5),
                              "p75": pct(n_el, .75), "p95": pct(n_el, .95), "max": max(n_el, default=None)},
        "omitted_actions_gt0_pct": round(100 * sum(r["omitted"] > 0 for r in recs) / k, 1),
        "history_len_median": pct([r["hist"] for r in recs], .5),
        "gold_find": dict(collections.Counter(r["find"] for r in recs).most_common()),
        "gold_match": dict(collections.Counter(r["match"] for r in recs).most_common()),
        "fidelity": {
            "layout_match_pct": round(100 * sum(bool(r["layout"]) for r in recs) / k, 1),
            "gold_position_recorded_pct": round(100 * sum(r["recorded"] for r in recs) / k, 1),
            "css_linked_total": css_linked,
            "css_loaded_pct": round(100 * sum(r["css"]["loaded"] for r in recs) / max(1, css_linked), 1),
            "pages_with_all_css_pct": round(100 * sum(r["css"]["loaded"] == r["css"]["linked"] for r in recs) / k, 1),
            "layout_match_pct_all_css_loaded": _rate([r["layout"] for r in recs
                                                      if r["css"]["loaded"] == r["css"]["linked"] > 0]),
            "layout_match_pct_some_css_missing": _rate([r["layout"] for r in recs
                                                        if r["css"]["loaded"] < r["css"]["linked"]]),
            "layout_match_pct_no_external_css": _rate([r["layout"] for r in recs if r["css"]["linked"] == 0]),
        },
        "dead_links": {"html_dead": dead, "html_fetch_attempts": fetch_attempts,
                       "rate_pct": round(100 * dead / max(1, fetch_attempts), 2),
                       "dead_reasons": dict(collections.Counter(e["info"].get("error") for e in log.values()
                                                                if e["reason"] == "dead_link").most_common(10))},
        "done_steps": {"added": 0, "why": "WebChain has no success/completion field and no page state after the "
                                          "last action (each HTML is the pre-action state)"},
        "seconds_per_step_render": {"median": round(pct(secs, .5), 3) if secs else None,
                                    "p90": round(pct(secs, .9), 3) if secs else None},
        "run": run_info,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "samples.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "webchain"), help="output directory")
    ap.add_argument("--limit", type=int, default=20000, help="stop after this many kept steps (0 = no limit)")
    ap.add_argument("--per-site-cap", type=int, default=80, help="max kept steps per primary_host")
    ap.add_argument("--workers", type=int, default=2, help="Chromium pages (max 2)")
    ap.add_argument("--fetch-concurrency", type=int, default=8, help="parallel downloads (max 8)")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/progress.log.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retry-reasons", nargs="*", default=["render_error", "timeout", "dead_link"],
                    help="with --resume: redo steps previously dropped for these reasons")
    ap.add_argument("--timeout", type=float, default=90, help="hard per-step timeout (s)")
    ap.add_argument("--page-life", type=int, default=60, help="recreate each page after N steps")
    ap.add_argument("--min-free-gb", type=float, default=0.8, help="pause while available RAM is below this")
    ap.add_argument("--min-commit-gb", type=float, default=3.0, help="pause while free commit charge is below this")
    ap.add_argument("--no-css", action="store_true", help="do not fetch external stylesheets")
    ap.add_argument("--english-only", action="store_true", help="skip non-English tasks entirely")
    ap.add_argument("--stats-only", action="store_true", help="just rewrite stats.json/samples.json")
    a = ap.parse_args(argv)
    a.workers = max(1, min(2, a.workers))
    traces, steps = load_metadata()
    run_info = {"limit": a.limit, "per_site_cap": a.per_site_cap, "workers": a.workers, "css": not a.no_css,
                "seed": a.seed}
    if not a.stats_only:
        r = Run(a)
        t0 = time.time()
        asyncio.run(r.run(traces, steps))
        wall = time.time() - t0
        run_info.update(wall_seconds_this_invocation=round(wall, 1), processed_this_invocation=r.n_processed,
                        wall_seconds_per_step=round(wall / max(1, r.n_processed), 3), memory_pauses=r.mem_pauses,
                        html_fetch=dict(r.html.counts), css_fetch=dict(r.cssf.counts))
    s = write_stats(a.out, traces, steps, run_info)
    print(json.dumps({k: s[k] for k in ("kept", "dropped_by_reason", "distinct_websites", "operation_distribution",
                                        "dead_links")}, indent=1, ensure_ascii=False), file=sys.stderr)


if __name__ == "__main__":
    main()
