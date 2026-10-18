"""Shared plumbing for the AX-tree converters (NNetNav, Go-Browse): website names, history items,
dedupe, output files and stats."""

from __future__ import annotations

import collections
import hashlib
import json
import pathlib
from urllib.parse import urlparse

from ..jevformat import make_record

# WebArena self-hosted sites, keyed by port (hosts are throwaway EC2 names).
WEBARENA_PORTS = {7770: 'shopping', 7780: 'shopping_admin', 9999: 'reddit', 8023: 'gitlab', 3000: 'map',
                  8888: 'wikipedia', 4399: 'homepage', 443: None, 80: None}
CONTROL_LABELS = {'scroll_down': 'Scroll down', 'scroll_up': 'Scroll up', 'wait': 'Wait for the page to update'}


def website(url: str, webarena: bool = False) -> str:
    try:
        u = urlparse(url)
    except ValueError:
        return 'unknown'
    if not u.hostname:
        return u.scheme or 'unknown'
    if webarena or (u.port in WEBARENA_PORTS and WEBARENA_PORTS[u.port]):
        name = WEBARENA_PORTS.get(u.port)
        if name:
            return name
    host = u.hostname.lower()
    return host[4:] if host.startswith('www.') else host


def history_item(action, kind, text=None):
    return {'action': action, 'kind': kind, 'text': text, 'page_changed': None}


def control_history(gold_id):
    kind = 'wait' if gold_id == 'wait' else 'scroll'
    return history_item(CONTROL_LABELS[gold_id], kind)


def dedupe_key(record):
    s = record['request']['state']
    payload = [record['request']['questions']['operation']['instructions']['goal'], s['page'], s['elements'],
               record['gold']]
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


class Sink:
    """Collects records, dedupes, counts drops, writes train.jsonl / samples.json / stats.json."""

    def __init__(self, out_dir, source, limit=0, write=True):
        self.out = pathlib.Path(out_dir)
        self.source, self.limit, self.write = source, limit, write
        self.kept, self.rows = 0, 0
        self.dropped = collections.Counter()
        self.ops = collections.Counter()
        self.sizes = []
        self.websites = collections.Counter()
        self.flags = collections.Counter()
        self.seen = set()
        self.dupes = 0
        self.samples = []
        self.fh = None
        if write:
            self.out.mkdir(parents=True, exist_ok=True)
            self.fh = open(self.out / 'train.jsonl', 'w', encoding='utf-8', newline='\n')

    @property
    def full(self):
        return bool(self.limit) and self.kept >= self.limit

    def drop(self, reason):
        self.dropped[reason] += 1

    def add(self, snapshot, goal, history, gold_ids, meta):
        """Returns True if kept."""
        meta = dict(meta, n_actions_in_snapshot=sum(1 for a in snapshot['actions'] if a['id'].startswith('e')),
                    approx_from_axtree=True)
        record = make_record(snapshot, goal, history, gold_ids, meta)
        if record is None:
            self.drop('gold_not_observable')
            return False
        key = dedupe_key(record)
        if key in self.seen:
            self.dupes += 1
            self.drop('duplicate')
            return False
        self.seen.add(key)
        record['meta']['dedupe_key'] = key
        self.kept += 1
        self.ops[record['gold']['operation']] += 1
        self.sizes.append(len(record['request']['state']['elements']))
        self.websites[meta.get('website')] += 1
        for f in ('blocked_proxy', 'gold_mapped_to_ancestor'):
            if meta.get(f):
                self.flags[f] += 1
        if self.fh:
            self.fh.write(json.dumps(record, ensure_ascii=False) + '\n')
        if len(self.samples) < 200:
            self.samples.append(record)
        return True

    def stats(self):
        sizes = sorted(self.sizes) or [0]
        pct = lambda p: sizes[min(len(sizes) - 1, int(p * len(sizes)))]
        buckets = collections.Counter()
        for n in self.sizes:
            buckets['0-25' if n <= 25 else '26-50' if n <= 50 else '51-100' if n <= 100 else
                    '101-150' if n <= 150 else '151-200' if n <= 200 else '201-250'] += 1
        total_candidates = self.kept + self.dupes
        return {
            'source': self.source,
            'rows_read': self.rows,
            'kept': self.kept,
            'dropped_total': sum(self.dropped.values()),
            'dropped_by_reason': dict(self.dropped.most_common()),
            'operation_distribution': dict(self.ops.most_common()),
            'flags': dict(self.flags),
            'elements_per_step': {'min': sizes[0], 'p25': pct(.25), 'median': pct(.5), 'p75': pct(.75),
                                  'p95': pct(.95), 'max': sizes[-1],
                                  'buckets': {k: buckets[k] for k in ('0-25', '26-50', '51-100', '101-150',
                                                                      '151-200', '201-250')}},
            'distinct_websites': len(self.websites),
            'top_websites': dict(self.websites.most_common(30)),
            'dedupe': {'duplicates_removed': self.dupes, 'candidates': total_candidates,
                       'duplicate_ratio': round(self.dupes / total_candidates, 4) if total_candidates else 0},
        }

    def close(self):
        stats = self.stats()
        if self.fh:
            self.fh.close()
            # 5 samples spread over operations
            picked, by_op = [], collections.defaultdict(list)
            for r in self.samples:
                by_op[r['gold']['operation']].append(r)
            while len(picked) < 5 and any(by_op.values()):
                for op in list(by_op):
                    if by_op[op] and len(picked) < 5:
                        picked.append(by_op[op].pop(0))
            (self.out / 'samples.json').write_text(json.dumps(picked, ensure_ascii=False, indent=2), encoding='utf-8')
            (self.out / 'stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
        return stats


def iter_lines(path, chunk=1 << 24, retries=5):
    """Yield the byte lines of a big file, reopening and seeking on transient OSErrors.

    Long single-handle reads of multi-GB files intermittently failed on Windows with
    "OSError: [Errno 22] Invalid argument"; plain chunked reads with reopen are robust.
    """
    import time
    pos, tail, failures = 0, b'', 0
    while True:
        try:
            with open(path, 'rb') as f:
                f.seek(pos)
                while True:
                    b = f.read(chunk)
                    if not b:
                        if tail:
                            yield tail
                        return
                    pos += len(b)
                    parts = (tail + b).split(b'\n')
                    tail = parts.pop()
                    yield from parts
        except OSError:
            failures += 1
            if failures > retries:
                raise
            time.sleep(1)
