"""Go-Browse-WA (apurvaga/go-browse-wa, MIT repo) -> Jev records.

Rows are BrowserGym steps on WebArena sites, ordered by trajectory (``step_number`` 0..traj_length-1).
Only successful trajectories (``traj_reward == 1``) are kept. History labels are resolved against the
AX tree of the step where each previous action was taken.

Usage:
    python -m myjev.convert.gobrowse --limit 3000 --out data/processed/gobrowse_wa
    python -m myjev.convert.gobrowse --limit 0 --no-write   # full-dataset stats only
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import re
import sys

import pyarrow.parquet as pq

from .axtree import axtree_to_snapshot, clean_label, history_label, parse_axtree, resolve_gold
from .axcommon import Sink, control_history, history_item, website

ROOT = pathlib.Path(__file__).resolve().parents[2]


def parse_action(text: str):
    """BrowserGym action string -> (name, args, kwargs) or None (unparseable / several actions)."""
    text = text.strip()
    if text in ('noop', 'send_msg_to_user', 'report_infeasible'):  # bare names without a call
        return text, [], {}
    try:
        tree = ast.parse(text, mode='exec')
    except SyntaxError:
        # Unescaped quotes in free text, e.g. send_msg_to_user('I've found ...'), or trailing junk.
        m = re.match(r"^(send_msg_to_user|report_infeasible)\((['\"])(.*)\2\)\s*$", text, re.S)
        if m:
            return m.group(1), [m.group(3)], {}
        m = re.match(r"^(click|fill|select_option)\(\s*(['\"])([^'\"]+)\2\s*(?:,\s*(['\"])(.*?)\4)?\s*\)", text, re.S)
        if m and '\n' not in text:
            args = [m.group(3)] + ([m.group(5)] if m.group(5) is not None else [])
            if m.group(1) == 'click' and len(args) > 1:  # click('456', 'left')
                args, kw = args[:1], {'button': args[1]}
                return 'click', args, kw
            return m.group(1), args, {}
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr) or not isinstance(tree.body[0].value, ast.Call):
        return None
    call = tree.body[0].value
    if not isinstance(call.func, ast.Name):
        return None
    try:
        args = [ast.literal_eval(a) for a in call.args]
        kwargs = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
    except ValueError:
        return None
    return call.func.id, args, kwargs


def split_prompt(user: str):
    """User message -> (goal, tree_text) or None."""
    g = re.search(r'^# Goal\n(.*?)(?=^# )', user, re.S | re.M)
    t = re.search(r'^# Current page Accessibility Tree\n(.*?)(?=^# )', user, re.S | re.M)
    if not g or not t:
        return None
    return g.group(1).strip(), t.group(1)


def gold_for(name, args, kwargs, index, scrolled, is_last):
    """-> (gold_id, drop_reason, extra_meta)."""
    if name == 'click':
        if kwargs.get('button', 'left') != 'left' or kwargs.get('modifiers'):
            return None, 'click_nonleft_or_modifier', {}
        gold, reason, mapped = resolve_gold(index, args[0], 'click')
        return gold, reason, {'gold_mapped_to_ancestor': mapped}
    if name == 'fill':
        gold, reason, _ = resolve_gold(index, args[0], 'fill')
        return gold, reason, {}
    if name == 'select_option':
        opt = args[1] if len(args) > 1 else kwargs.get('options')
        if isinstance(opt, (list, tuple)):
            if len(opt) != 1:
                return None, 'select_multiple', {}
            opt = opt[0]
        gold, reason, _ = resolve_gold(index, args[0], 'select', str(opt))
        return gold, reason, {}
    if name == 'scroll':
        dy = args[1] if len(args) > 1 else kwargs.get('delta_y', 0)
        if dy > 0:
            return 'scroll_down', None, {}
        if dy < 0:
            return ('scroll_up', None, {}) if scrolled else (None, 'scroll_up_not_offered', {})
        return None, 'scroll_horizontal', {}
    if name == 'noop':
        return 'wait', None, {}
    if name == 'send_msg_to_user':
        # Non-final messages are repeats of the final answer on the finished page; keep, but flag.
        return 'DONE', None, {} if is_last else {'done_not_final': True}
    if name == 'report_infeasible':
        return 'BLOCKED', None, {'blocked_proxy': True}
    return None, 'action_' + name, {}


def history_for(name, args, kwargs, index):
    """History item for an executed action, or None if Jev has no equivalent."""
    if name in ('click', 'fill', 'select_option') and args:
        bid = str(args[0])
        node = index['nodes_by_id'].get(bid)
        if node is None:
            return None
        if name == 'fill':
            return history_item(history_label(node, 'fill'), 'fill', str(args[1]) if len(args) > 1 else None)
        if name == 'select_option':
            opt = args[1] if len(args) > 1 else kwargs.get('options')
            opt = opt[0] if isinstance(opt, (list, tuple)) and opt else opt
            return history_item(f"{history_label(node, 'select')} → {clean_label(str(opt))}", 'select')
        if bid in index['options']:  # click on a native <select> option
            sel = index['nodes'][node['parent']] if node['parent'] is not None else None
            while sel is not None and sel['role'] != 'combobox':
                sel = index['nodes'][sel['parent']] if sel['parent'] is not None else None
            if sel is not None:
                return history_item(f"{history_label(sel, 'select')} → {clean_label(node['name'])}", 'select')
        return history_item(history_label(node, 'click'), 'click')
    if name == 'scroll':
        dy = args[1] if len(args) > 1 else kwargs.get('delta_y', 0)
        return control_history('scroll_down' if dy > 0 else 'scroll_up') if dy else None
    if name == 'noop':
        return control_history('wait')
    return None


def iter_rows(paths, shard_done=lambda path: False):
    """Rows of all shards in order; a shard is abandoned once shard_done(path) is true."""
    cols = ['step_idx', 'step_data', 'traj_reward', 'next_step_idx', 'traj_length', 'step_number']
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256, columns=cols):
            if shard_done(path):
                break
            for row in batch.to_pylist():
                yield path, row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', default=str(ROOT / 'data/raw/gobrowse_wa'), help='dir with parquet shards')
    ap.add_argument('--limit', type=int, default=3000, help='max kept steps (0 = all)')
    ap.add_argument('--out', default=str(ROOT / 'data/processed/gobrowse_wa'))
    ap.add_argument('--no-write', action='store_true', help='only compute stats (written to <out>/stats_full.json)')
    args = ap.parse_args(argv)
    paths = sorted(pathlib.Path(args.input).glob('*.parquet'))
    sink = Sink(args.out, 'gobrowse_wa', args.limit, write=not args.no_write)
    traj = None  # {'id', 'history', 'scrolled', 'next'}
    skipped_unsuccessful = 0
    # Shards are grouped by website, so a pilot takes an equal quota from every shard.
    quota = -(-args.limit // len(paths)) if args.limit else 0
    kept_at_start = {}

    def shard_done(path):
        kept_at_start.setdefault(path, sink.kept)
        return bool(quota) and sink.kept - kept_at_start[path] >= quota

    for path, row in iter_rows(paths, shard_done):
        if sink.full:
            break
        if row['traj_reward'] != 1:
            skipped_unsuccessful += 1
            continue
        sink.rows += 1
        if row['step_number'] == 0 or traj is None or traj['next'] != row['step_idx']:
            if row['step_number'] != 0:
                sink.drop('trajectory_start_missing')
                traj = None
                continue
            traj = {'id': f"gb_{row['step_idx']}", 'history': [], 'scrolled': False}
        traj['next'] = row['next_step_idx']
        user = next((m['content'] for m in row['step_data']['prompt'] if m['role'] == 'user'), '')
        parts = split_prompt(user)
        try:
            raw = json.loads(row['step_data']['completion'][0]['content'])['action']
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            raw = None
        parsed = parse_action(raw) if isinstance(raw, str) else None
        if parts is None:
            sink.drop('bad_prompt')
            continue
        goal, tree = parts
        nodes = parse_axtree(tree)
        root = next((n for n in nodes if n['role'] == 'RootWebArea'), None)
        url = (root or {}).get('props', {}).get('url', '') if root else ''
        url = url if isinstance(url, str) else ''
        snapshot, index = axtree_to_snapshot(nodes, url, scrolled=traj['scrolled'])
        if parsed is not None and parsed[0] in ('click', 'fill', 'select_option') and not parsed[1]:
            name, a, kw = parsed  # bid passed as keyword, e.g. click(bid='12')
            parsed = (name, [kw.pop('bid')] + ([kw.pop('value', kw.pop('options', None))] if name != 'click' else []),
                      kw) if 'bid' in kw else None
        if parsed is not None and parsed[0] == 'scroll' and len(parsed[1]) < 2 and 'delta_y' not in parsed[2]:
            parsed = None
        if parsed is None:
            sink.drop('unparseable_action')
        else:
            name, a, kw = parsed
            gold, reason, extra = gold_for(name, a, kw, index, traj['scrolled'], row['next_step_idx'] == -1)
            if gold is None:
                sink.drop(reason)
            else:
                meta = {'source': 'gobrowse_wa', 'split': 'train', 'website': website(url, webarena=True),
                        'url': url, 'task_id': traj['id'],
                        'goal_hash': hashlib.sha1(goal.encode('utf-8')).hexdigest()[:12],
                        'step_index': row['step_number'], 'traj_length': row['traj_length'],
                        'gold_action_raw': raw[:500], 'blocked_proxy': False, 'gold_mapped_to_ancestor': False}
                meta.update(extra)
                sink.add(snapshot, goal, traj['history'][-10:], [gold], meta)
            item = history_for(name, a, kw, index)
            if item:
                traj['history'].append(item)
            if name == 'scroll' and (a[1] if len(a) > 1 else kw.get('delta_y', 0)) > 0:
                traj['scrolled'] = True
    stats = sink.close()
    stats['rows_skipped_unsuccessful_traj'] = skipped_unsuccessful
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / ('stats_full.json' if args.no_write else 'stats.json')).write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding='utf-8')
    json.dump({k: stats[k] for k in ('rows_read', 'kept', 'dropped_by_reason', 'operation_distribution',
                                     'distinct_websites', 'dedupe', 'rows_skipped_unsuccessful_traj')},
              sys.stdout, indent=1, ensure_ascii=False)
    print()


if __name__ == '__main__':
    main()
