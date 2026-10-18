"""WebChain converter on tiny handmade inputs (no network, no dataset download)."""
import asyncio
import json

import pytest

from myjev.convert import webchain as w

HTML = """<html><head><title>Shop</title><script>document.body.innerHTML='wiped'</script>
<link rel="stylesheet" href="/static/a.css"><link rel="preload" as="style" href="https://cdn.x.com/b.css">
<link rel="icon" href="/fav.ico"><meta http-equiv="refresh" content="0;url=https://evil"></head>
<body>
  <nav><a href="/flights" class="nav-a"><span class="lbl">Flights</span></a></nav>
  <form id="f"><label>From <input id="from" type="text" value="Zurich"></label>
    <select id="cabin"><option value="e" selected>Economy</option><option value="b">Business</option></select>
    <button type="submit" class="go">Search</button></form>
  <div style="height:3000px"></div>
  <a href="/deep" id="deep">Deep link</a>
  <div class="row" style="position:absolute;left:0;top:500px;height:40px;width:600px">
    <a href="/p1" style="position:absolute;left:0;top:0;width:100px;height:20px">One</a>
    <a href="/p2" style="position:absolute;left:200px;top:0;width:100px;height:20px">Two</a></div>
</body></html>"""


def step(i, op, selector, dom_path, value="", input_text="", title="", rect=None, scroll_y=0.0, point=None):
    return {"trace_uid": "t1", "source_step_index": i, "action_type": op, "html_dom_url": "u%d" % i,
            "selector": selector, "dom_path": dom_path, "value": value, "title": title,
            "href": "https://shop.example.com/", "viewport": json.dumps({"width": 1280, "height": 800}),
            "rect": json.dumps(rect) if rect else "", "input_text": input_text, "attributes": "",
            "scroll_x": 0.0, "scroll_y": scroll_y, "page_x": point[0] if point else None,
            "page_y": point[1] if point else None}


TRACE = {"uid": "t1", "query": "Find a business flight from Zurich", "primary_host": "shop.example.com",
         "web_type": "travel", "intent_type": "Simple IR", "source_trace_id": "s", "language": "en"}


# ---------------------------------------------------------------- pure helpers

def test_strip_verb_and_language():
    assert w.strip_verb("点击 新加坡 (SIN)") == "新加坡 (SIN)"
    assert w.strip_verb("Click FILTERS") == "FILTERS"
    assert w.strip_verb("Input liverpool") == "liverpool"
    assert w.strip_verb("Clicker game") == "Clicker game"
    assert w.detect_language("Find a return flight from Singapore") == "en"
    assert w.detect_language("Find ‘1段奶粉’ baby formula") == "en"  # a few CJK chars in an English task
    assert w.detect_language("帮我找一个去曼谷的航班") == "zh"
    assert w.detect_language("東京へのフライトを探してください") == "ja"
    assert w.detect_language("서울행 항공편") == "ko"
    assert w.detect_language("Trouvez un hôtel à Genève très élégant près du lac") == "other"
    assert w.detect_language("") == "unknown"


def test_prepare_html_and_stylesheets():
    out = w.prepare_html(HTML)
    assert "<script" not in out and "http-equiv" not in out and "Flights" in out
    urls = w.stylesheet_urls(HTML, "https://shop.example.com/a/b")
    assert urls == ["https://shop.example.com/static/a.css", "https://cdn.x.com/b.css"]
    base = '<base href="https://other.org/x/"><link href="s.css" rel="stylesheet">'
    assert w.stylesheet_urls(base, "https://shop.example.com/") == ["https://other.org/x/s.css"]


def test_viewport_and_gold_frac():
    assert w.clamp_viewport({"width": 2108, "height": 1293}) == {"width": 1920, "height": 1200}
    assert w.clamp_viewport({"width": 800, "height": 400}) == {"width": 1024, "height": 600}
    vp = {"width": 1280, "height": 800}
    frac, rec = w.gold_frac({"y": 390, "height": 20}, vp, "k")
    assert rec and abs(frac - 390 / 760) < 1e-3
    f1, r1 = w.gold_frac({}, vp, "k")
    f2, _ = w.gold_frac(None, vp, "k")
    assert not r1 and f1 == f2 and 0.05 <= f1 <= 0.95  # seeded, deterministic
    assert w.gold_frac({"y": 5000, "height": 20}, vp, "k")[1] is False  # off-screen record


def test_history_entries():
    s = step(1, "type", "#from", "body>form>label>input", value="liverpool", input_text="liverpool",
             title="Input liverpool")
    assert w.history_entry(s, "From") == {"action": "From", "kind": "fill", "text": "liverpool", "page_changed": None}
    c = step(2, "click", "a", "body>a", value="Deals", title="Click Deals")
    assert w.history_entry(c) == {"action": "Deals", "kind": "click", "text": None, "page_changed": None}
    sel = step(3, "select", "#cabin>option:nth-child(2)", "body>select>option", value="Business", input_text="Business")
    assert w.history_entry(sel, "Cabin")["action"] == "Cabin → Business"
    assert w.history_entry(sel, "Cabin → Business")["action"] == "Cabin → Business"
    assert w.history_entry(step(4, "drag", "", "")) is None
    attrs = step(5, "click", "", "", title="点击")
    attrs["attributes"] = json.dumps({"data": {"aria-label": "Close dialog"}})
    assert w.fallback_label(attrs) == "Close dialog"


def test_plan_order_round_robin_english_first():
    traces = [{"uid": f"a{i}", "primary_host": "a.com", "language": "en" if i else "zh"} for i in range(4)]
    traces += [{"uid": "b0", "primary_host": "b.com", "language": "en"},
               {"uid": "c0", "primary_host": "c.com", "language": "zh"}]
    order = w.plan_order(traces, seed=1)
    assert sorted(order) == sorted(t["uid"] for t in traces)
    assert set(order[:3]) == {order[0], order[1], order[2]} and {o[0] for o in order[:3]} == {"a", "b", "c"}
    a_order = [o for o in order if o.startswith("a")]
    assert a_order[-1] == "a0"  # the non-English task of a.com comes last
    assert w.plan_order(traces, seed=1) == order


def test_page_box():
    assert w.page_box({"x": 10, "y": 20, "width": 5, "height": 6}, 100.0, 3) == [13.0, 120.0, 5.0, 6.0]
    assert w.page_box({}, 0) is None


def test_layout_match():
    assert w.layout_match([0, 0, 100, 20], {"width": 104, "height": 21}) is True
    assert w.layout_match([0, 0, 300, 20], {"width": 100, "height": 20}) is False
    assert w.layout_match(None, {"width": 1, "height": 1}) is None


def test_fetcher_cache_and_dead(tmp_path, monkeypatch):
    f = w.Fetcher(tmp_path)
    calls = []

    def fake(url, p):
        calls.append(url)
        if "dead" in url:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.with_suffix(".dead").write_text("http_404", encoding="utf-8")
            return "http_404"
        import gzip
        p.parent.mkdir(parents=True, exist_ok=True)
        p.with_suffix(".gz").write_bytes(gzip.compress(b"<html>ok</html>"))
        return None

    monkeypatch.setattr(f, "_download", fake)
    assert f.get("https://x/ok") == ("<html>ok</html>", None)
    assert f.get("https://x/ok") == ("<html>ok</html>", None)
    assert f.get("https://x/dead") == (None, "http_404")
    assert f.get("https://x/dead") == (None, "http_404")
    assert calls == ["https://x/ok", "https://x/dead"]  # cached, dead links never re-requested


# ---------------------------------------------------------------- offline Chromium, real snapshot.js

def run_steps(steps, history=()):
    pytest.importorskip("playwright.async_api")
    from playwright.async_api import async_playwright

    async def go():
        out = []
        async with async_playwright() as pw:
            r = w.Renderer(pw, 1)
            page = await r.new_page()
            for s in steps:
                out.append(await w.convert_step(page, s, TRACE, list(history), w.prepare_html(HTML), {},
                                                {"linked": 0, "loaded": 0}))
            await r.close_page(page)
            await r.close()
        return out

    return asyncio.run(go())


def answer(rec):
    g = rec["gold"]
    key = g["operation"].lower() + "_target"
    return g["operation"], [rec["request"]["questions"][key]["criteria"][t]["element"] for t in g[key]]


def test_convert_steps_offline():
    hist = [w.history_entry(step(0, "click", "", "", value="Home"))]
    res = run_steps([
        step(1, "click", ".nav-a >span:nth-child(1)", "body>nav>a>span", value="Flights", title="Click Flights",
             rect={"x": 8, "y": 10, "width": 45, "height": 18}),
        step(2, "type", "#from", "body>form>label>input", input_text="Basel", title="Input Basel"),
        step(3, "select", "#cabin >option:nth-child(2)", "body>form>select>option", value="Business",
             input_text="Business"),
        step(4, "hover", "#deep", "body>a", value="Deep link", rect={"x": 8, "y": 600, "width": 70, "height": 18}),
        step(5, "click", ".nope", "body>form>button", value="Search"),          # selector stale -> dom_path
        step(6, "click", ".missing", "body>main>div", value="x"),                 # not in the dump
        step(7, "click", "label", "body>form>label", value="From"),              # label -> its input
        step(8, "click", ".stale", "body>div>a", rect={"x": 200, "y": 100, "width": 100, "height": 20},
             scroll_y=400),                                                      # recorded box picks "Two"
        step(9, "click", ".row", "body>div", point=(250, 510)),                 # container: click point -> "Two"
        step(10, "click", ".row", "body>div"),                                   # no point: ambiguous
    ], history=hist)
    (r9, why9, _), (r10, why10, _) = res[8:]
    assert why9 == "kept" and r9["meta"]["gold_match"] == "descendant_point" and answer(r9)[1][0].endswith("] Two")
    assert r10 is None and why10 == "ambiguous_descendants"
    (r1, why1, i1), (r2, why2, _), (r3, why3, _), (r4, why4, _), (r5, why5, _), (r6, why6, _) = res[:6]
    (r7, why7, _), (r8, why8, _) = res[6:8]
    assert why7 == "kept" and r7["meta"]["gold_find"] == "selector+label_control" and answer(r7)[1][0].endswith("Open From")
    assert why8 == "kept" and r8["meta"]["gold_find"] == "dom_path_box" and answer(r8)[1][0].endswith("] Two")
    assert why1 == "kept" and answer(r1) == ("CLICK", ["[1] Flights"]) and r1["meta"]["gold_match"] == "ancestor"
    assert i1["label"] == "Flights"
    m = r1["meta"]
    assert m["source"] == "webchain" and m["route"] == "html_snapshot" and m["website"] == "shop.example.com"
    assert m["task_id"] == "t1" and m["step_index"] == 1 and m["language"] == "en" and m["done_synthetic"] is False
    assert m["gold_action_raw"]["action_type"] == "click" and m["gold_position_recorded"]
    assert r1["request"]["state"]["recent_actions"] == [
        {"action": "Home", "kind": "click", "text": None, "page_changed": None}]
    assert r1["request"]["state"]["page"]["url"] == "https://shop.example.com/"
    assert r1["request"]["state"]["page"]["title"] == "Shop"
    assert "wiped" not in r1["request"]["state"]["page"]["text"]  # scripts stripped

    assert why2 == "kept" and answer(r2) == ("TYPE_TEXT", ["[2] From"])
    assert why3 == "kept" and r3["gold"]["operation"] == "SELECT"
    assert answer(r3)[1][0].endswith("→ Business")
    assert why4 == "kept" and r4["meta"]["hover_as_click"] and answer(r4)[1] == ["[" + r4["gold"]["click_target"][0]
                                                                                   + "] Deep link"]
    assert r4["meta"]["scroll_y"] > 0  # scrolled to the recorded height
    assert why5 == "kept" and r5["meta"]["gold_find"] == "dom_path_text" and answer(r5)[1][0].endswith("Search")
    assert r6 is None and why6 == "gold_not_found_in_dom"
