"""NNetNav-live / NNetNav-WA (stanfordnlp/nnetnav-{live,wa}, Apache-2.0) -> Jev records.

Each row is one (observation, previous actions, model output) step. The observation is a WebArena-style
AX tree (BrowserGym dialect for -live, WebArena dialect for -WA). Rows are best-of-N policy samples over
exploration states, so the same state appears several times; records are deduped on
(goal, page, elements, gold) — history is not part of the key.

Usage:
    python -m myjev.convert.nnetnav --source live --limit 3000 --out data/processed/nnetnav_live
    python -m myjev.convert.nnetnav --source wa --limit 0 --no-write   # full-dataset stats only
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

from .axtree import axtree_to_snapshot, clean_label, history_label, parse_axtree, parse_line, resolve_gold
from .axcommon import Sink, control_history, history_item, iter_lines, website

ROOT = pathlib.Path(__file__).resolve().parents[2]
DROP_VERBS = {'hover', 'press', 'goto', 'go_back', 'go_forward', 'new_tab', 'tab_focus', 'close_tab',
              'page_focus', 'page_close'}

_ACTION = re.compile(r'^(?P<verb>[a-z_]+)\b\s*(?P<args>.*)$', re.S)


def parse_action(text: str):
    """NNetNav action string -> dict(verb, id, text, dir, answer) or None if unparseable."""
    text = text.strip().strip('`').strip()
    if '\n' in text and re.match(r'^[a-z]+\n', text):  # ```python\n...```
        text = text.split('\n', 1)[1].strip()
    m = _ACTION.match(text)
    if not m:
        return None
    verb, args = m.group('verb'), m.group('args').strip()
    if verb == 'click' or verb == 'hover':
        a = re.match(r'^\[([^\]]+)\]', args)
        return {'verb': verb, 'id': a.group(1).strip()} if a else None
    if verb == 'type':
        a = re.match(r'^\[([^\]]+)\]\s*\[(.*)\]\s*\[([01])\]\s*$', args, re.S) or \
            re.match(r'^\[([^\]]+)\]\s*\[(.*)\]\s*$', args, re.S)
        return {'verb': verb, 'id': a.group(1).strip(), 'text': a.group(2).strip()} if a else None
    if verb == 'scroll':
        a = re.search(r'\b(down|up)\b', args, re.I)
        return {'verb': verb, 'dir': a.group(1).lower()} if a else None
    if verb == 'stop':
        a = re.match(r'^\[(.*)\]\s*$', args, re.S)
        return {'verb': verb, 'answer': a.group(1).strip() if a else args}
    if verb in DROP_VERBS:
        return {'verb': verb}
    return None


def extract_gold(output: str):
    s = re.sub(r'<\|[a-z_]+\|>', '', output).replace('assistant\n', '', 1)
    verbs = '|'.join(['click', 'type', 'scroll', 'stop'] + sorted(DROP_VERBS))
    acts = re.findall(r'`+\s*(?:[a-z]+\n)?\s*((?:%s)\b.*?)\s*`+' % verbs, s, re.S)
    if acts:
        return acts[-1]
    blocks = [b for b in re.findall(r'```+\s*(.*?)\s*```+', s, re.S) if b.strip()]
    if blocks:
        return blocks[-1]
    m = re.search(r'next action I will perform is\s*`*\s*(.+?)\s*`*\s*$', s, re.S)
    return m.group(1) if m else None


def split_prompt(user: str):
    """User message -> (tree_text, url, objective, previous_actions_text)."""
    i, j, k = user.rfind('\nURL:'), user.rfind('\nOBJECTIVE:'), user.rfind('\nPREVIOUS ACTIONS:')
    if min(i, j, k) < 0:
        return None
    obs = user[:i]
    o = obs.find('OBSERVATION:')
    if o >= 0:
        obs = obs[o + len('OBSERVATION:'):]
    lines = [l for l in obs.split('\n') if not re.match(r'^Tab \d+', l)]
    return '\n'.join(lines), user[i + 5:j].strip(), user[j + 11:k].strip(), user[k + 18:].strip()


def parse_history(prev: str, nodes_by_id):
    """PREVIOUS ACTIONS block -> (history items, n_steps, scrolled_down)."""
    items, n, scrolled = [], 0, False
    for entry in re.split(r'\n(?=\d+: )', '\n' + prev):
        entry = entry.strip()
        m = re.match(r'^\d+: (.*)$', entry, re.S)
        if not m or m.group(1).strip() == 'None' or m.group(1).startswith('The previous prediction'):
            continue
        n += 1
        body = m.group(1)
        act, _, desc = body.partition(' where [')
        a = parse_action(act)
        if not a:
            continue
        if a['verb'] == 'scroll':
            items.append(control_history('scroll_' + a['dir']))
            scrolled |= a['dir'] == 'down'
            continue
        if a['verb'] not in ('click', 'type'):
            continue  # stop/goto/tab/hover/press have no Jev equivalent
        desc = desc.split('] is ', 1)[1] if '] is ' in desc else ''
        node = None
        if re.match(r"^[A-Za-z][\w-]*\s?['\"]", desc):  # WA dialect: role 'name' props
            node = parse_line(desc)
        if node is None:  # live dialect: just the name; borrow the role if the id is still on the page
            cur = nodes_by_id.get(a['id'])
            if cur is not None and clean_label(cur['name']) == clean_label(desc):
                node = cur
            else:
                node = {'role': '', 'name': desc, 'props': {}, 'readonly': False, 'editable': False}
        label = history_label(node, 'click' if a['verb'] == 'click' else 'fill') or clean_label(desc)
        if a['verb'] == 'type':
            items.append(history_item(label, 'fill', a.get('text')))
        else:
            items.append(history_item(label, 'click'))
    return items, n, scrolled


def convert_row(row, source, split, sink):
    user = next((m['content'] for m in reversed(row['messages']) if m['role'] == 'user'), None)
    parts = split_prompt(user or '')
    if parts is None:
        return sink.drop('bad_prompt')
    tree, url, goal, prev = parts
    if url.startswith(('about:', 'chrome')) or not url:
        return sink.drop('blank_page')
    raw = extract_gold(row['output'])
    a = parse_action(raw) if raw else None
    if a is None:
        return sink.drop('unparseable_action')
    if a['verb'] in DROP_VERBS:
        return sink.drop('action_' + a['verb'])
    nodes = parse_axtree(tree)
    history, n_steps, scrolled = parse_history(prev, {n['id']: n for n in nodes if n['id']})
    snapshot, index = axtree_to_snapshot(nodes, url, scrolled=scrolled)
    meta = {'source': f'nnetnav_{source}', 'split': split, 'website': website(url, webarena=source == 'wa'),
            'url': url, 'task_id': row.get('task_name') or row.get('id'), 'step_index': n_steps,
            'gold_action_raw': raw.strip()[:500], 'blocked_proxy': False, 'gold_mapped_to_ancestor': False}
    verb = a['verb']
    if verb == 'stop':
        ans = a['answer'].strip().strip('$').strip().upper()
        if ans in ('N/A', 'NA', 'N\\A'):
            gold, meta['blocked_proxy'] = 'BLOCKED', True
        elif not ans:
            return sink.drop('stop_empty_answer')
        else:
            gold = 'DONE'
    elif verb == 'scroll':
        if a['dir'] == 'up' and not scrolled:
            return sink.drop('scroll_up_not_offered')
        gold = 'scroll_' + a['dir']
    else:
        op = 'click' if verb == 'click' else 'fill'
        if verb == 'type' and a['id'] in index['native_select']:
            op = 'select'  # WebArena types into a <select> to pick the matching option
        gold, reason, mapped = resolve_gold(index, a['id'], op, a.get('text'))
        if gold is None:
            return sink.drop(reason)
        meta['gold_mapped_to_ancestor'] = mapped
    sink.add(snapshot, goal, history[-10:], [gold], meta)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', choices=['live', 'wa'], required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--input', help='jsonl path (default data/raw/nnetnav_<source>/<split>.jsonl)')
    ap.add_argument('--limit', type=int, default=3000, help='max kept steps (0 = all)')
    ap.add_argument('--out', help='output dir (default data/processed/nnetnav_<source>)')
    ap.add_argument('--no-write', action='store_true', help='only compute stats (written to <out>/stats_full.json)')
    args = ap.parse_args(argv)
    inp = pathlib.Path(args.input or ROOT / f'data/raw/nnetnav_{args.source}/{args.split}.jsonl')
    out = pathlib.Path(args.out or ROOT / f'data/processed/nnetnav_{args.source}')
    sink = Sink(out, f'nnetnav_{args.source}', args.limit, write=not args.no_write)
    for line in iter_lines(inp):
        if sink.full:
            break
        if not line.strip():
            continue
        sink.rows += 1
        try:
            row = json.loads(line.decode('utf-8', errors='replace'))
        except json.JSONDecodeError:
            sink.drop('bad_json')
            continue
        convert_row(row, args.source, args.split, sink)
    stats = sink.close()
    if args.no_write:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'stats_full.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding='utf-8')
    json.dump({k: stats[k] for k in ('rows_read', 'kept', 'dropped_by_reason', 'operation_distribution',
                                     'distinct_websites', 'dedupe')}, sys.stdout, indent=1, ensure_ascii=False)
    print()


if __name__ == '__main__':
    main()
