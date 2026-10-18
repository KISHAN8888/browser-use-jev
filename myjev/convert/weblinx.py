"""WebLINX 1.1 / BrowserGym version (McGill-NLP/weblinx-browsergym, CC-BY-NC-SA-4.0) -> Jev records.

Route: BrowserGym AX tree + the recorded viewport geometry (no Chromium).

Per step the dataset ships the page the human navigator saw *before* acting, as a CDP accessibility tree
whose nodes carry ``browsergym_id`` (= the page's ``data-webtasks-id``, = the action's ``uid``), plus
``extra_element_properties`` with each element's bbox in CSS px at recording time. The replay gives the
viewport size of every browser action. So we can do what snapshot.js does and keep only elements whose
centre is inside the real viewport, and only the text of boxes that intersect it, with the real scroll
offset (scroll_up / scroll_down offered exactly when the window can scroll). The saved HTML would let us
run the real snapshot.js, but it has no CSS (external stylesheets are not saved), so layout, visibility
and the viewport would all be wrong; the AX route keeps the real geometry.

Goal: WebLINX has no single upfront task; an instructor chats with the navigator. The goal for a step is
every instructor ``say`` turn before that step, oldest first / most recent last, one per line. Pure
greetings ("Hi", "Hello there!") are skipped. If longer than GOAL_MAX chars the oldest utterances are
cut (the most recent one is always kept; ``meta.goal_truncated``).

Actions: click -> CLICK (inner non-interactable targets mapped to the nearest interactable ancestor
within 3 levels, flagged); textInput -> TYPE_TEXT on that field; scroll -> SCROLL_DOWN / SCROLL_UP by the
sign of the y delta (only if the window can scroll that way); submit -> CLICK on the form's submit button
if it follows a text input and the form has exactly one / one submit-like button (``meta.from_submit``),
dropped when it follows a click (the click already is the step). Navigator ``say``, load, hover and tab
operations are dropped (counted). WebLINX 1.1 has no ``change`` (select) actions, so there is no SELECT.
History: previous click / fill / scroll steps of the same demo (last 10), labelled from the AX node when
that step's tree was loaded, else from the replay's element attributes.

Usage:
    python -m myjev.convert.weblinx --split train --per-site-cap 80 --out data/processed/weblinx/train.jsonl
    python -m myjev.convert.weblinx --split test_web --out data/processed/weblinx/test_web.jsonl
    (add --resume to continue an interrupted run)
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import os
import pathlib
import random
import re
import shutil
import sys
import time

from ..jevformat import make_record
from .axcommon import control_history, dedupe_key, history_item, website
from .axtree import (SNAPSHOT_ROLES, _descendants, axtree_to_snapshot, clean_label, history_label, page_text,
                     parse_axtree, resolve_gold)

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/raw/weblinx/bg'
REPO = 'McGill-NLP/weblinx-browsergym'
ROUTE = 'browsergym_axtree+recorded_viewport'
GOAL_MAX = 2000
GREETING = re.compile(r"^\W*(hi+|hello+|hey+|hey there|hi there|hello there|good (morning|afternoon|evening)|"
                      r"greetings)\W*(navigator|there)?\W*$", re.I)
SKIP_ROLES = {'InlineTextBox', 'LineBreak'}
PROMOTE_ROLES = {'generic', 'none', 'presentation', 'Ignored', 'IgnoredRole'}
INTERACTIVE = set(SNAPSHOT_ROLES) | {'DisclosureTriangle'}
SUBMITTY = re.compile(r'search|submit|go\b|send|sign ?in|log ?in|continue|next|find|apply|save|post|enter|ok\b',
                      re.I)
ELEMENT_INTENTS = {'click', 'textinput', 'submit', 'scroll'}
LOOKAHEAD = 3  # demos downloaded ahead of the one being converted


# --------------------------------------------------------------------------------------------- AX tree

def _prop(v):
    """CDP AXValue -> python value for the text dialect."""
    if not isinstance(v, dict):
        return None
    return v.get('value')


def _fmt(key, val):
    if isinstance(val, bool):
        return f'{key}={val}'
    if isinstance(val, (int, float)):
        return f'{key}={val}'
    return f'{key}={str(val)!r}'


def viewport_geometry(props, viewport):
    """-> (scroll_y, page_height) from the document element's recorded bbox, or (0, None)."""
    if not viewport:
        return 0.0, None
    vw = viewport[0]
    best = None
    for v in props.values():
        b = v.get('bbox')
        if not b or b[0] != 0 or b[2] < 0.9 * vw:
            continue
        if best is None or b[3] > best[3]:
            best = b
    if best is None:
        return 0.0, None
    return max(0.0, -best[1]), best[3]


def _in_viewport(bbox, viewport, centre):
    if not bbox or not viewport:
        return None
    x, y, w, h = bbox
    vw, vh = viewport
    if centre:
        cx, cy = x + w / 2, y + h / 2
        return w > 0 and h > 0 and 0 <= cx < vw and 0 <= cy < vh
    if w <= 0 or h <= 0:
        return None  # zero-size wrappers say nothing about their (possibly overflowing) children
    return x < vw and x + w > 0 and y < vh and y + h > 0


def cdp_to_axtext(ax, props, viewport):
    """CDP AX tree (BrowserGym dump with ``browsergym_id``) -> BrowserGym-dialect text for parse_axtree.

    Out-of-viewport nodes are emitted with ``hidden=True`` so axtree_to_snapshot skips them and their text:
    interactable roles when their centre is outside the viewport (snapshot.js rule), other boxes when they
    do not intersect it at all. Ignored and nameless generic nodes are not emitted (their children move
    up); their bids go to ``alias`` -> nearest emitted ancestor bid, because humans often click such inner
    <div>/<span>s. Returns (text, offscreen_bids, alias, n_without_geometry).
    """
    nodes = ax.get('nodes', [])
    by_id = {n['nodeId']: n for n in nodes}
    child_set = {c for n in nodes for c in n.get('childIds', [])}
    roots = [n for n in nodes if not n.get('parentId') and n['nodeId'] not in child_set]
    if not roots and nodes:
        roots = [nodes[0]]
    lines, offscreen, alias, seen, no_geom = [], set(), {}, set(), 0
    stack = [(r, 0 if i == 0 else 1, None) for i, r in reversed(list(enumerate(roots)))]
    while stack:  # iterative DFS: real pages nest deeper than Python's recursion limit
        n, depth, anc_bid = stack.pop()
        if n['nodeId'] in seen:
            continue
        seen.add(n['nodeId'])
        role = _prop(n.get('role')) or ''
        if role in SKIP_ROLES:
            continue
        name = _prop(n.get('name')) or ''
        bid = n.get('browsergym_id')
        promote = n.get('ignored') or (role in PROMOTE_ROLES and not str(name).strip())
        if promote:
            if bid:
                alias[bid] = anc_bid or ''  # '' = no emitted ancestor with a bid (not interactable)
        else:
            pr = {p['name']: _prop(p.get('value')) for p in n.get('properties', []) if isinstance(p, dict)}
            parts = []
            val = _prop(n.get('value'))
            if val is not None and val != '':
                parts.append(_fmt('value', str(val)))
            for key in ('checked', 'pressed'):
                if pr.get(key) is not None:
                    parts.append(_fmt(key, str(pr[key]).lower()))
            for key in ('selected', 'expanded', 'disabled', 'focused', 'readonly', 'required', 'multiselectable'):
                if pr.get(key) is not None:
                    parts.append(_fmt(key, bool(pr[key])))
            for key in ('hasPopup', 'autocomplete', 'editable', 'valuetext'):
                if pr.get(key) not in (None, '', 'false', False):
                    parts.append(_fmt(key, pr[key]))
            hidden = bool(pr.get('hidden'))
            if bid and not hidden:
                p = props.get(bid)
                bbox = p.get('bbox') if p else None
                inside = _in_viewport(bbox, viewport, centre=role in INTERACTIVE)
                if inside is None and role in INTERACTIVE and viewport and p is not None:
                    inside = bool(p.get('visibility'))
                if inside is None and p is None:
                    no_geom += 1
                if inside is False:
                    hidden = True
                    offscreen.add(bid)
            if hidden:
                parts.append('hidden=True')
            head = ('\t' * depth) + (f'[{bid}] ' if bid else '') + f'{role} {str(name)!r}'
            lines.append(head + (', ' + ', '.join(parts) if parts else ''))
            depth += 1
            anc_bid = bid or anc_bid
        for c in reversed(n.get('childIds', [])):
            if c in by_id:
                stack.append((by_id[c], depth, anc_bid))
    return '\n'.join(lines), offscreen, alias, no_geom


def build_page(ax, props, viewport, url, title=None):
    """-> (snapshot, index, info) for one recorded page."""
    text, offscreen, alias, no_geom = cdp_to_axtext(ax, props, viewport)
    nodes = parse_axtree(text)
    scroll_y, height = viewport_geometry(props, viewport)
    more_below = True if height is None or not viewport else scroll_y + viewport[1] < height - 2
    snapshot, index = axtree_to_snapshot(nodes, url, title=title, scrolled=scroll_y > 0, more_below=more_below)
    index['offscreen'] = offscreen
    index['alias'] = alias
    info = {'scroll_y': round(scroll_y, 1), 'page_height': round(height, 1) if height else None,
            'viewport': list(viewport) if viewport else None, 'nodes_without_geometry': no_geom}
    return snapshot, index, info


def dom_ancestors(dom, uid, max_up=8):
    """bids of the DOM ancestors of uid, nearest first, from a CDP DOMSnapshot (attribute ``bid``, which
    BrowserGym set to the page's data-webtasks-id)."""
    strings = dom.get('strings', [])
    try:
        val = strings.index(uid)
    except ValueError:
        return []
    keys = {i for i, x in enumerate(strings) if x in ('bid', 'data-webtasks-id')}
    for doc in dom.get('documents', []):
        attrs, parents = doc['nodes'].get('attributes', []), doc['nodes'].get('parentIndex', [])
        bid_of = {}
        hit = None
        for i, a in enumerate(attrs):
            for j in range(0, len(a) - 1, 2):
                if a[j] in keys:
                    bid_of[i] = a[j + 1]
                    if a[j + 1] == val:
                        hit = i
        if hit is None:
            continue
        out, i = [], parents[hit] if hit < len(parents) else -1
        while i is not None and i >= 0 and len(out) < max_up:
            if i in bid_of:
                out.append(strings[bid_of[i]])
            i = parents[i] if i < len(parents) else -1
        return out
    return []


def label_control(index, bid, max_up=2):
    """Click on (inside) a <label>: the single control it contains -> its click action id, or None."""
    nodes = index['nodes']
    node = index['nodes_by_id'].get(bid)
    for _ in range(max_up + 1):
        if node is None:
            return None
        if node['role'] == 'LabelText':
            ctl = [nodes[j]['id'] for j in _descendants(nodes, node['index'])
                   if nodes[j]['id'] in index['targets'] and 'click' in index['targets'][nodes[j]['id']]]
            return index['targets'][ctl[0]]['click'] if len(ctl) == 1 else None
        node = nodes[node['parent']] if node['parent'] is not None else None
    return None


def gold_blocker(index, bid):
    """Reason the gold node is invisible to snapshot.js, walking up ancestors; None if visible."""
    node = index['nodes_by_id'].get(bid)
    nodes = index['nodes']
    while node is not None:
        if node['id'] in index['offscreen']:
            return 'gold_offscreen'
        if node['hidden']:
            return 'gold_hidden'
        node = nodes[node['parent']] if node['parent'] is not None else None
    return None


# --------------------------------------------------------------------------------------------- goal/history

def is_greeting(utt):
    return bool(GREETING.match(utt.strip()))


def build_goal(utterances, max_chars=GOAL_MAX):
    """Instructor utterances (oldest first) -> (goal, n_used, truncated)."""
    utts = [re.sub(r'[ \t]+', ' ', u).strip() for u in utterances]
    utts = [u for u in utts if u and not is_greeting(u)]
    if not utts:
        return '', 0, False
    kept, total = [], 0
    for u in reversed(utts):
        if kept and total + len(u) + 1 > max_chars:
            break
        kept.append(u)
        total += len(u) + 1
    kept.reverse()
    goal = '\n'.join(kept)
    if len(goal) > max_chars:  # a single huge utterance: keep its end (most recent text)
        goal = goal[-max_chars:]
    return goal, len(kept), len(kept) < len(utts) or len(goal) < sum(len(u) for u in kept)


def _attrs(element):
    a = (element or {}).get('attributes') or {}
    if isinstance(a, str):
        try:
            import ast
            a = ast.literal_eval(a)
        except Exception:
            a = {}
    return a if isinstance(a, dict) else {}


def replay_label(element, op):
    """Best-effort history label from a replay element (used when that step's AX tree was not loaded)."""
    element = element or {}
    a = _attrs(element)
    tag = (element.get('tagName') or '').lower()
    name = (a.get('aria-label') or a.get('placeholder') or a.get('title') or a.get('alt') or '').strip()
    if not name and tag not in ('input', 'textarea', 'select'):
        name = (element.get('textContent') or '').strip()
    if not name:
        name = a.get('name') or a.get('value') or tag or 'element'
    label = clean_label(name)[:100]
    editable = tag in ('input', 'textarea') and a.get('type', 'text') not in (
        'button', 'submit', 'checkbox', 'radio', 'reset', 'image', 'file', 'hidden', 'range', 'color')
    if op == 'click' and editable:
        return 'Open ' + label
    return label


def format_raw(intent, args):
    parts = ', '.join(f'{k}={json.dumps(v, ensure_ascii=False)}' for k, v in args.items())
    return f'{intent}({parts})'[:500]


def submit_control(index, form_bid):
    """Submit button inside a form: the only button, or the only submit-like one. -> (action id, bid)."""
    node = index['nodes_by_id'].get(form_bid)
    if node is None:
        return None, None
    buttons = []
    for j in _descendants(index['nodes'], node['index']):
        n = index['nodes'][j]
        if n['role'] == 'button' and n['id'] in index['targets'] and 'click' in index['targets'][n['id']]:
            buttons.append(n)
    if len(buttons) != 1:
        buttons = [b for b in buttons if SUBMITTY.search(b['name'] or '')]
    if len(buttons) != 1:
        return None, None
    return index['targets'][buttons[0]['id']]['click'], buttons[0]['id']


# --------------------------------------------------------------------------------------------- one demo

def step_keys(steps):
    return sorted(steps, key=int)


def candidate_site(step):
    return website(((step.get('tab') or {}).get('url')) or '')


def convert_demo(demo_id, steps, replay, load_page, split, site_ok, counters, load_dom=None):
    """Convert one demonstration.

    steps: metadata[split][demo_id] ({step_key: step_dict}); replay: replay.json 'data' list (indexable by
    step key) or None; load_page(step) -> (ax_json, props) or None; site_ok(site, n_kept_in_this_demo) -> bool
    (per-site cap);
    counters: collections.Counter for drop reasons. Returns a list of (record, site).
    """
    out, history, utterances = [], [], []
    kept_here = collections.Counter()
    prev_intent = None
    for key in step_keys(steps):
        s = steps[key]
        intent, args = s['intent'], s.get('args') or {}
        turn = replay[int(key)] if replay is not None and int(key) < len(replay) else None
        action = (turn or {}).get('action') or {}
        element = (action.get('arguments') or {}).get('element')
        md = (action.get('arguments') or {}).get('metadata') or {}
        if intent == 'say':
            if args.get('speaker') == 'instructor':
                utterances.append(args.get('utterance') or '')
            else:
                counters['navigator_say'] += 1
            continue
        if intent not in ELEMENT_INTENTS:
            counters['tab_op' if intent.startswith('tab') else intent] += 1
            prev_intent = intent
            continue
        url = ((s.get('tab') or {}).get('url')) or md.get('url') or ''
        site = website(url)
        label_from_ax = None
        hist_after = None
        reason, rec = None, None
        try:
            if not url or url.startswith(('about:', 'chrome')):
                reason = 'no_url'
            elif not site_ok(site, kept_here[site]):
                reason = 'site_cap'
            elif not s.get('axtree_path') or not s.get('has_full_snapshot', True):
                reason = 'no_snapshot'
            elif s.get('_stale'):
                reason = 'stale_snapshot'
            else:
                goal, n_utt, truncated = build_goal(utterances)
                if not goal:
                    reason = 'no_instructor_goal'
                else:
                    page = load_page(s)
                    if page is None:
                        reason = 'snapshot_missing'
                    else:
                        ax, props = page
                        vw, vh = md.get('viewportWidth', -1), md.get('viewportHeight', -1)
                        viewport = (vw, vh) if vw and vh and vw > 0 and vh > 0 else None
                        snapshot, index, info = build_page(ax, props, viewport, url)
                        rec, reason, label_from_ax = _gold_record(
                            intent, args, element, snapshot, index, info, prev_intent, goal,
                            history, demo_id, key, s, split, site, url, n_utt, truncated, viewport,
                            (lambda s=s: load_dom(s)) if load_dom else None)
        except Exception as e:  # one bad page must not kill a long run
            reason = 'error_' + type(e).__name__
        if rec is not None:
            out.append((rec, site))
            kept_here[site] += 1
        else:
            counters[reason] += 1
        # history (Jev's recent_actions): what the navigator actually did
        if intent == 'click':
            hist_after = history_item(label_from_ax or replay_label(element, 'click'), 'click')
        elif intent == 'textinput':
            hist_after = history_item(label_from_ax or replay_label(element, 'fill'), 'fill', args.get('value'))
        elif intent == 'scroll' and args.get('y'):
            hist_after = control_history('scroll_down' if args['y'] > 0 else 'scroll_up')
        if hist_after:
            history.append(hist_after)
        prev_intent = intent
    return out


def _gold_record(intent, args, element, snapshot, index, info, prev_intent, goal, history, demo_id, key, s,
                 split, site, url, n_utt, truncated, viewport, get_dom=None):
    """-> (record | None, drop_reason | None, history label from the AX node | None)."""
    flags = {'gold_mapped_to_ancestor': False, 'from_submit': False, 'gold_via_dom_ancestor': False}
    label = None
    uid = args.get('uid')
    target = uid if uid in index['nodes_by_id'] else index['alias'].get(uid)  # inner <div> -> emitted ancestor
    if target is None and uid and get_dom is not None:  # not in the AX tree at all (aria-hidden icon, ...)
        dom = get_dom()
        for anc in dom_ancestors(dom, uid) if dom else []:
            t = anc if anc in index['nodes_by_id'] else index['alias'].get(anc)
            if t is not None:
                target = t
                flags['gold_via_dom_ancestor'] = True
                break
    if target == '':
        return None, 'gold_not_interactable', None
    promoted = target is not None and target != uid
    if target is not None and target in index['nodes_by_id']:
        node = index['nodes_by_id'][target]
        label = history_label(node, 'fill' if intent == 'textinput' else 'click')
    if intent == 'scroll':
        y = args.get('y') or 0
        if y > 0:
            gold = 'scroll_down' if any(a['id'] == 'scroll_down' for a in snapshot['actions']) else None
            if gold is None:
                return None, 'scroll_down_not_offered', None
        elif y < 0:
            gold = 'scroll_up' if any(a['id'] == 'scroll_up' for a in snapshot['actions']) else None
            if gold is None:
                return None, 'scroll_up_not_offered', None
        else:
            return None, 'scroll_horizontal', None
    elif intent == 'submit':
        if prev_intent == 'click':
            return None, 'submit_after_click', None
        if target not in index['nodes_by_id']:
            return None, 'gold_not_in_axtree', None
        gold, bid = submit_control(index, target)
        if gold is None:
            return None, 'submit_no_control', None
        why = gold_blocker(index, bid)
        if why:
            return None, why, None
        flags['from_submit'] = True
        label = None
    else:
        if (_attrs(element).get('type') or '').lower() == 'password':
            return None, 'password_field', label
        if target not in index['nodes_by_id']:
            return None, 'gold_not_in_axtree', None
        why = gold_blocker(index, target)
        if why:
            return None, why, label
        op = 'fill' if intent == 'textinput' else 'click'
        gold, why, mapped = resolve_gold(index, target, op)
        if gold is None and why == 'gold_not_interactable' and op == 'click':
            gold = label_control(index, target)
            mapped = gold is not None
        if gold is None:
            return None, why, label
        mapped = mapped or (promoted and op == 'click')
        flags['gold_mapped_to_ancestor'] = mapped
        if mapped:
            label = next((a['label'] for a in snapshot['actions'] if a['id'] == gold), label)
    meta = {'source': 'weblinx', 'split': split, 'website': site, 'url': url, 'demo_id': demo_id,
            'turn_index': int(key), 'route': ROUTE, 'gold_action_raw': format_raw(intent, args),
            'goal_n_utterances': n_utt, 'goal_truncated': truncated,
            'page_file': (s.get('axtree_path') or '').split('/')[-1], 'zoom': s.get('zoom'),
            'viewport_known': viewport is not None, 'scroll_y': info['scroll_y'],
            'n_actions_in_snapshot': sum(1 for a in snapshot['actions'] if a['id'].startswith('e')),
            'approx_from_axtree': True, 'annotator_id': s.get('annotator_id'),
            'uses_ai_output': s.get('uses_ai_output'), **flags}
    rec = make_record(snapshot, goal, history[-10:], [gold], meta)
    if rec is None:
        return None, 'gold_not_observable', label
    rec['meta']['dedupe_key'] = dedupe_key(rec)
    return rec, None, label


def mark_stale(steps):
    """Flag element steps whose page file is the same as the previous element step's (state not refreshed)."""
    last = None
    for key in step_keys(steps):
        s = steps[key]
        if s['intent'] in ELEMENT_INTENTS | {'hover'}:
            ax = s.get('axtree_path')
            if ax and ax == last and s['intent'] != 'scroll':
                s['_stale'] = True
            last = ax


# --------------------------------------------------------------------------------------------- IO

KEEP_DIRS = ('axtrees/', 'extra_element_properties/', 'dom_snapshots/')


def fetch_demo(demo, retries=8):
    """Download demonstrations_zip/<demo>.zip (one Hub request per demo: per-file downloads hit the anonymous
    rate limit), extract replay.json + AX trees + element properties + DOM snapshots, delete the zip
    (screenshots/pages are skipped). Raises after repeated failures so a run stops instead of silently
    dropping steps; re-run with --resume."""
    import zipfile
    from huggingface_hub import hf_hub_download
    dest = RAW / 'demonstrations' / demo
    if (dest / '.complete').exists():
        return dest
    for attempt in range(retries):
        try:
            zp = pathlib.Path(hf_hub_download(REPO, f'demonstrations_zip/{demo}.zip', repo_type='dataset',
                                              local_dir=str(RAW)))
            break
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(min(300, 15 * 2 ** attempt))
    with zipfile.ZipFile(zp) as z:
        for info in z.infolist():
            name = info.filename.split(demo + '/', 1)[-1] if info.filename.startswith(demo + '/') else info.filename
            if name == 'replay.json' or name.startswith(KEEP_DIRS):
                target = dest / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(target, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
    zp.unlink(missing_ok=True)
    (dest / '.complete').write_text('', encoding='utf-8')
    return dest


def fetch(rel):
    """Local path of demonstrations/<rel> (after fetch_demo), or None if the demo does not have it."""
    local = RAW / 'demonstrations' / rel
    return local if local.exists() else None


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


class Out:
    """Appends records per finished demo; keeps the counts needed for --per-site-cap, --limit, stats, resume."""

    def __init__(self, path, resume):
        self.path = pathlib.Path(path)
        self.progress_path = self.path.with_suffix('.progress.json')
        self.site_kept = collections.Counter()
        self.ops = collections.Counter()
        self.sizes, self.samples, self.seen = [], [], set()
        self.flags = collections.Counter()
        self.kept, self.dupes = 0, 0
        self.drops = collections.Counter()
        self.done = set()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if resume and self.path.exists():
            with open(self.path, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self._count(json.loads(line))
            if self.progress_path.exists():
                p = load_json(self.progress_path)
                self.done = set(p['done_demos'])
                self.drops.update(p['drops'])
                self.dupes = p.get('dupes', 0)
        else:
            self.path.write_text('', encoding='utf-8')
        self.fh = open(self.path, 'a', encoding='utf-8', newline='\n')

    def _count(self, rec):
        m = rec['meta']
        self.seen.add(m['dedupe_key'])
        self.kept += 1
        self.site_kept[m['website']] += 1
        self.ops[rec['gold']['operation']] += 1
        self.sizes.append(len(rec['request']['state']['elements']))
        for f in ('gold_mapped_to_ancestor', 'from_submit', 'goal_truncated'):
            if m.get(f):
                self.flags[f] += 1
        if len(self.samples) < 300:
            self.samples.append(rec)

    def add_demo(self, demo_id, items, counters, cap, limit):
        for rec, site in items:
            if limit and self.kept >= limit:
                counters['limit'] += 1
                continue
            if cap and self.site_kept[site] >= cap:
                counters['site_cap'] += 1
                continue
            if rec['meta']['dedupe_key'] in self.seen:
                self.dupes += 1
                counters['duplicate'] += 1
                continue
            self.fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
            self._count(rec)
        self.fh.flush()
        os.fsync(self.fh.fileno())
        self.drops.update(counters)
        self.done.add(demo_id)
        tmp = self.progress_path.with_suffix('.tmp')
        tmp.write_text(json.dumps({'done_demos': sorted(self.done), 'drops': dict(self.drops),
                                   'dupes': self.dupes}), encoding='utf-8')
        os.replace(tmp, self.progress_path)

    def stats(self, split, total_demos):
        sizes = sorted(self.sizes) or [0]
        pct = lambda p: sizes[min(len(sizes) - 1, int(p * len(sizes)))]
        per_site = sorted(self.site_kept.values())
        quality = {k: v for k, v in self.drops.items() if k not in NON_STEP_REASONS}
        return {
            'source': 'weblinx', 'split': split, 'route': ROUTE, 'kept': self.kept,
            'demos_processed': len(self.done), 'demos_in_split': total_demos,
            'dropped_by_reason': dict(self.drops.most_common()),
            'dropped_element_steps_by_reason': dict(collections.Counter(quality).most_common()),
            'operation_distribution': dict(self.ops.most_common()), 'flags': dict(self.flags),
            'elements_per_step': {'min': sizes[0], 'p25': pct(.25), 'median': pct(.5), 'p75': pct(.75),
                                  'p95': pct(.95), 'max': sizes[-1]},
            'distinct_websites': len(self.site_kept),
            'steps_per_site': {'min': per_site[0] if per_site else 0,
                               'median': per_site[len(per_site) // 2] if per_site else 0,
                               'max': per_site[-1] if per_site else 0},
            'websites': dict(self.site_kept.most_common()),
            'duplicates_removed': self.dupes,
        }

    def close(self):
        self.fh.close()


# reasons that are not failed conversions of an element step (different action type or a sampling choice)
NON_STEP_REASONS = {'navigator_say', 'load', 'hover', 'tab_op', 'site_cap', 'limit'}


def pick_samples(samples, n=5):
    picked, by_op = [], collections.defaultdict(list)
    for r in samples:
        by_op[r['gold']['operation']].append(r)
    while len(picked) < n and any(by_op.values()):
        for op in list(by_op):
            if by_op[op] and len(picked) < n:
                picked.append(by_op[op].pop(0))
    return picked


def merge_json(path, split, value):
    data = load_json(path) if path.exists() else {}
    data[split] = value
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', default='train',
                    help='train | valid | test_iid | test_web | test_cat | test_geo | test_vis')
    ap.add_argument('--limit', type=int, default=0, help='max kept steps (0 = all)')
    ap.add_argument('--per-site-cap', type=int, default=0, help='max kept steps per website (0 = no cap)')
    ap.add_argument('--out', help='output jsonl (default data/processed/weblinx/<split>.jsonl)')
    ap.add_argument('--resume', action='store_true', help='continue: skip demos already in <out>.progress.json')
    ap.add_argument('--keep-raw', action='store_true', help='keep downloaded demo files (default: delete)')
    ap.add_argument('--workers', type=int, default=2, help='parallel demo downloads')
    ap.add_argument('--seed', type=int, default=0, help='demo order shuffle (spreads the site cap)')
    args = ap.parse_args(argv)
    out_path = pathlib.Path(args.out or ROOT / f'data/processed/weblinx/{args.split}.jsonl')
    meta_path = RAW / 'metadata.json'
    if not meta_path.exists():
        from huggingface_hub import hf_hub_download
        hf_hub_download(REPO, 'metadata.json', repo_type='dataset', local_dir=str(RAW))
    metadata = load_json(meta_path)[args.split]
    demos = sorted(metadata)
    random.Random(args.seed).shuffle(demos)
    out = Out(out_path, args.resume)
    pool = cf.ThreadPoolExecutor(args.workers)
    t0, n_new = time.time(), 0

    cap = args.per_site_cap
    futures = {}  # rel path -> download future (prefetch of the next demos)

    def is_open(demo):
        steps = metadata[demo]
        return any(not cap or out.site_kept[candidate_site(steps[k])] < cap
                   for k in steps if steps[k]['intent'] in ELEMENT_INTENTS)

    def prefetch(start):
        for d in demos[start:start + LOOKAHEAD]:
            if d not in out.done and d not in futures and is_open(d):
                futures[d] = pool.submit(fetch_demo, d)

    for di, demo in enumerate(demos):
        if demo in out.done:
            continue
        if args.limit and out.kept >= args.limit:
            break
        steps = metadata[demo]
        mark_stale(steps)
        counters = collections.Counter()
        prefetch(di)
        fut = futures.pop(demo, None)
        if not is_open(demo):  # every candidate is on a capped site: nothing to convert
            if fut is not None:
                fut.result()
            for k in step_keys(steps):
                s = steps[k]
                if s['intent'] == 'say':
                    if s['args'].get('speaker') != 'instructor':
                        counters['navigator_say'] += 1
                elif s['intent'] in ELEMENT_INTENTS:
                    counters['site_cap'] += 1
                else:
                    counters['tab_op' if s['intent'].startswith('tab') else s['intent']] += 1
            out.add_demo(demo, [], counters, cap, args.limit)
            if not args.keep_raw:
                shutil.rmtree(RAW / 'demonstrations' / demo, ignore_errors=True)
            continue
        fut.result() if fut is not None else fetch_demo(demo)
        files = {}
        for k in steps:
            for f in ('axtree_path', 'extra_props_path'):
                if steps[k].get(f):
                    files[steps[k][f]] = fetch(steps[k][f])
        files[f'{demo}/replay.json'] = fetch(f'{demo}/replay.json')
        rp = files.get(f'{demo}/replay.json')
        replay = load_json(rp)['data'] if rp else None
        page_cache = {}

        def load_page(s):
            k = (s['axtree_path'], s.get('extra_props_path'))
            if k not in page_cache:
                a, p = files.get(s['axtree_path']), files.get(s.get('extra_props_path'))
                page_cache.clear()  # pages are used in order; keep memory flat
                page_cache[k] = (load_json(a), load_json(p) if p else {}) if a else None
            return page_cache[k]

        def site_ok(site, pending):
            return not cap or out.site_kept[site] + pending < cap

        def load_dom(s):  # only fetched when a gold node is missing from the AX tree
            rel = s.get('dom_object_path')
            path = fetch(rel) if rel else None
            return load_json(path) if path else None

        items = convert_demo(demo, steps, replay, load_page, args.split, site_ok, counters, load_dom)
        before = out.kept
        out.add_demo(demo, items, counters, cap, args.limit)
        n_new += out.kept - before
        if not args.keep_raw:
            shutil.rmtree(RAW / 'demonstrations' / demo, ignore_errors=True)
        if (di + 1) % 10 == 0 or di == len(demos) - 1:
            print(f'[{di + 1}/{len(demos)}] demos, kept {out.kept} ({len(out.site_kept)} sites), '
                  f'{time.time() - t0:.0f}s', flush=True)
    for f in futures.values():
        f.cancel()
    pool.shutdown()
    if not args.keep_raw:  # prefetched demos that were never converted
        for d in futures:
            shutil.rmtree(RAW / 'demonstrations' / d, ignore_errors=True)
    out.close()
    stats = out.stats(args.split, len(demos))
    merge_json(out_path.parent / 'stats.json', args.split, stats)
    merge_json(out_path.parent / 'samples.json', args.split, pick_samples(out.samples))
    json.dump({k: stats[k] for k in ('kept', 'demos_processed', 'dropped_by_reason', 'operation_distribution',
                                     'distinct_websites', 'steps_per_site', 'elements_per_step')},
              sys.stdout, indent=1, ensure_ascii=False)
    print()


if __name__ == '__main__':
    main()
