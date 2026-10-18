"""Accessibility-tree text -> snapshot.js-shaped dict.

Parses the two AX-tree text dialects found in open web-agent datasets:

* BrowserGym / NNetNav-live style::

      RootWebArea 'Title', focused, url='https://...'
      \t[89] textbox 'Search' value='abc', clickable, visible, focused
      \t\tStaticText 'abc'
      \t[166] combobox '' value='Car', hasPopup='menu', expanded=False
      \t\t[167] option 'Bike', selected=False

* WebArena native style (NNetNav-WA)::

      [81] RootWebArea 'Title' focused: True
      \t[6917] combobox 'Search' autocomplete: both hasPopup: listbox required: False expanded: False
      \t\t[7869] StaticText 'sunscreen'
      \t[6920] button 'Search' disabled: True

and turns it into the snapshot dict consumed by ``myjev.jevformat.make_record``, mimicking
``myjev/vendor/snapshot.js`` as far as an AX tree allows:

* only snapshot.js roles are interactable (button, link, checkbox, radio, switch, tab, menuitem,
  menuitemradio, option, gridcell, combobox, textbox, searchbox, spinbutton; ``DisclosureTriangle``
  i.e. ``<summary>`` maps to button);
* editable fields (textbox/searchbox/spinbutton, editable combobox) -> ``fill`` + ``click`` "Open <name>";
* a native ``<select>`` (combobox whose option/menuitem children all carry ``selected``) -> one
  ``select`` action per unselected option, label ``"<name> → <option>"``, ``current_value``;
  its option nodes are consumed (not separate clicks);
* checkbox/radio/switch carry ``checked`` "true"/"false"; aria ``selected``/``expanded`` passed through;
* disabled nodes and hidden subtrees are skipped; gridcells containing a button are skipped;
* max 250 actions, ids e1..eN, then ``scroll_down`` / ``scroll_up`` / ``wait``;
* page text = StaticText names (plus names of content-named nodes whose StaticText was pruned),
  newline-joined, cut at 6000 chars exactly like snapshot.js.

Known gaps (an AX tree has no geometry or viewport):

* ``scroll_down`` is appended unless the caller says otherwise (default: always), because AX dumps
  do not tell us whether there is more page below. ``scroll_up`` only when ``scrolled=True``
  (converters pass "history contains a scroll down").
* The whole tree is used, not only the in-viewport part, so element lists and text are larger
  than what snapshot.js would show on the same page.
* Native-select option ``value`` is the option label (the HTML ``value`` attribute is not in the tree).
* Role mapping is approximate: AX roles come from implicit semantics, snapshot.js only honours
  explicit ``role`` attributes for e.g. gridcell/option/menuitem.
* Private-use icon-font glyphs are stripped from names and labels are capped at 500 chars.
"""

from __future__ import annotations

import ast
import re

SNAPSHOT_ROLES = ['button', 'link', 'checkbox', 'radio', 'switch', 'tab', 'menuitem', 'menuitemradio',
                  'option', 'gridcell', 'combobox', 'textbox', 'searchbox', 'spinbutton']
ROLE_ALIASES = {'DisclosureTriangle': 'button'}
EDITABLE_ROLES = {'textbox', 'searchbox', 'spinbutton'}
CHECKABLE_ROLES = {'checkbox', 'radio', 'switch', 'menuitemradio'}
SELECTED_ROLES = {'tab', 'option', 'gridcell', 'menuitemradio', 'treeitem'}
# Roles whose accessible name normally comes from their contents. BrowserGym prunes StaticText
# children that repeat the parent's name, so we put the name back into the page text.
CONTENT_NAMED = {'link', 'button', 'heading', 'cell', 'gridcell', 'columnheader', 'rowheader', 'menuitem',
                 'menuitemradio', 'menuitemcheckbox', 'tab', 'option', 'treeitem', 'LabelText', 'label',
                 'strong', 'emphasis', 'term', 'definition', 'caption', 'paragraph', 'listitem', 'code',
                 'Pre', 'time', 'superscript', 'subscript', 'mark', 'DescriptionListTerm',
                 'DescriptionListDetail', 'tooltip', 'status', 'checkbox', 'radio', 'switch',
                 'Legend', 'Figcaption', 'blockquote', 'insertion', 'deletion'}
TEXT_ROLES = {'StaticText', 'text'}
MAX_ACTIONS = 250
MAX_LABEL = 500
MAX_TEXT = 6000
PUA = re.compile('[-]')

_LINE = re.compile(r"^(?P<indent>\t*)(?:\[(?P<id>[^\]\s]+)\]\s*)?(?P<role>[A-Za-z][\w-]*)(?P<rest>.*)$")
_QUOTED = r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\""
_PROP = re.compile(r"\s*,?\s*(?P<key>[A-Za-z_][\w-]*)(?:(?:=|:\s)(?P<val>" + _QUOTED + r"|[^,\s]+))?")


def _unquote(q: str) -> str:
    try:
        v = ast.literal_eval(q)
        if isinstance(v, str):
            return v
    except Exception:
        pass
    return q[1:-1]


def _value(v):
    if v is None:
        return True
    if v[:1] in ("'", '"'):
        return _unquote(v)
    return {'True': True, 'False': False}.get(v, v)


def parse_line(line: str):
    """One AX-tree line -> node dict or None."""
    m = _LINE.match(line.rstrip('\r\n'))
    if not m:
        return None
    rest = m.group('rest')
    name = ''
    s = rest.lstrip(' ')
    if s[:1] in ("'", '"'):
        q = re.match(_QUOTED, s)
        if q:
            name = _unquote(q.group(0))
            rest = s[q.end():]
        else:  # unterminated (truncated) name
            name, rest = s[1:], ''
    props = {}
    pos = 0
    while pos < len(rest):
        p = _PROP.match(rest, pos)
        if not p or p.end() == pos:
            pos += 1
            continue
        props[p.group('key')] = _value(p.group('val'))
        pos = p.end()
    return {'id': m.group('id'), 'role': m.group('role'), 'name': name, 'props': props,
            'depth': len(m.group('indent'))}


def parse_axtree(text: str):
    """AX-tree text -> list of nodes in document order with parent/children links.

    Node: {id, role, name, props, depth, parent (index|None), children [indices], index}.
    Convenience fields: value, checked, selected, expanded, disabled, focused, hidden, editable.
    """
    nodes, stack = [], []
    for line in text.split('\n'):
        if not line.strip():
            continue
        node = parse_line(line)
        if node is None:
            continue
        node['index'] = len(nodes)
        node['children'] = []
        while stack and nodes[stack[-1]]['depth'] >= node['depth']:
            stack.pop()
        node['parent'] = stack[-1] if stack else None
        if node['parent'] is not None:
            nodes[node['parent']]['children'].append(node['index'])
        stack.append(node['index'])
        p = node['props']
        node['value'] = p.get('value') if isinstance(p.get('value'), str) else None
        for key in ('checked', 'selected', 'expanded', 'pressed'):
            v = p.get(key)
            node[key] = None if v is None else str(v).lower()
        node['disabled'] = p.get('disabled') in (True, 'true')
        node['hidden'] = p.get('hidden') in (True, 'true')
        node['focused'] = p.get('focused') in (True, 'true')
        node['readonly'] = p.get('readonly') in (True, 'true')
        node['editable'] = 'editable' in p
        nodes.append(node)
    return nodes


def _descendants(nodes, i):
    out, todo = [], list(reversed(nodes[i]['children']))
    while todo:
        j = todo.pop()
        out.append(j)
        todo.extend(reversed(nodes[j]['children']))
    return out


def clean_label(name: str) -> str:
    return re.sub(r'\s+', ' ', PUA.sub('', name or '')).strip()[:MAX_LABEL]


def _role(node):
    return ROLE_ALIASES.get(node['role'], node['role'])


def _select_options(nodes, i):
    """Option nodes of a native <select>, or None if node i is not one."""
    node = nodes[i]
    if _role(node) != 'combobox' or 'autocomplete' in node['props'] or node['editable']:
        return None
    opts = []
    for j in node['children']:
        child = nodes[j]
        if child['role'] in ('option', 'menuitem'):
            opts.append(j)
        elif child['role'] in ('menu', 'listbox', 'MenuListPopup', 'group'):
            opts.extend(k for k in child['children'] if nodes[k]['role'] in ('option', 'menuitem'))
    if not opts or any(nodes[j]['selected'] is None for j in opts):
        return None
    return opts


def _is_editable(nodes, i):
    node, role = nodes[i], _role(nodes[i])
    if node.get('readonly'):
        return False
    if role in EDITABLE_ROLES:
        return True
    if role == 'combobox':
        p = node['props']
        return 'autocomplete' in p or 'editable' in p or p.get('hasPopup') in ('listbox', 'grid')
    return False


def _inner_text(nodes, i):
    return ' '.join(clean_label(nodes[j]['name']) for j in _descendants(nodes, i)
                    if nodes[j]['role'] in TEXT_ROLES and nodes[j]['name'].strip()).strip()


def axtree_to_snapshot(nodes, url, title=None, scrolled=False, more_below=True):
    """Build a snapshot.js-shaped dict from parsed nodes.

    Returns (snapshot, index) where index = {
        'targets': {ax_id: {'click': eid, 'fill': eid, 'open': eid, 'select': {option_name_lower: eid}}},
        'options': {option_ax_id: eid | 'selected'}  (options of native selects),
        'native_select': set(ax_id), 'omitted': set(ax_id) (cut by the 250 cap),
        'skipped': {ax_id: reason} (disabled/hidden), 'nodes_by_id': {ax_id: node},
    }
    """
    if title is None:
        root = next((n for n in nodes if n['role'] == 'RootWebArea'), None)
        title = root['name'] if root else ''
    hidden = set()
    for n in nodes:
        if n['hidden'] or (n['parent'] is not None and n['parent'] in hidden):
            hidden.add(n['index'])
    consumed, actions = set(), []
    targets, options, native, skipped = {}, {}, set(), {}
    node_seq = 0
    for n in nodes:
        i, role = n['index'], _role(n)
        if i in consumed or role not in SNAPSHOT_ROLES:
            continue
        if i in hidden:
            skipped[n['id']] = 'hidden'
            continue
        if n['disabled']:
            skipped[n['id']] = 'disabled'
            continue
        if role == 'gridcell' and any(_role(nodes[j]) == 'button' for j in _descendants(nodes, i)):
            skipped[n['id']] = 'gridcell_with_button'
            continue
        node_seq += 1
        label = clean_label(n['name']) or role
        base = {'node': node_seq, 'role': role, 'label': label}
        for key in ('checked', 'selected', 'expanded'):
            if n[key] is not None and (key != 'selected' or role in SELECTED_ROLES):
                base[key] = n[key]
        if role in ('checkbox', 'radio') and 'checked' not in base:
            base['checked'] = 'false'
        if role in ('checkbox', 'radio') and base['checked'] not in ('true', 'false'):
            base['checked'] = 'true' if base['checked'] == 'mixed' else 'false'
        opts = _select_options(nodes, i)
        entry = targets.setdefault(n['id'], {}) if n['id'] else {}
        if opts is not None:
            base.pop('expanded', None)  # Chromium adds expanded=False to every <select>; no aria attr
            if not clean_label(n['name']):  # snapshot.js name() falls back to the options' text content
                label = clean_label(' '.join(nodes[j]['name'] for j in opts)) or role
                base['label'] = label
            native.add(n['id'])
            consumed.update(opts)
            for j in opts:
                consumed.update(_descendants(nodes, j))
            selected = [clean_label(nodes[j]['name']) for j in opts if nodes[j]['selected'] == 'true']
            current = ', '.join(selected) if selected else clean_label(n['value'] or '')
            entry['select'] = {}
            for j in opts:
                o = nodes[j]
                oname = clean_label(o['name'])
                if o['selected'] == 'true':
                    options[o['id']] = 'selected'
                    continue
                if o['disabled'] or j in hidden:
                    continue
                a = {**base, 'kind': 'select', 'value': oname, 'current_value': current,
                     'label': label + ' → ' + oname}
                actions.append(a)
                entry['select'].setdefault(oname.lower(), a)
                if o['id']:
                    options[o['id']] = a
            continue
        editable = _is_editable(nodes, i)
        if editable:
            value = n['value'] if n['value'] is not None else (
                n['props'].get('valuetext') if isinstance(n['props'].get('valuetext'), str) else _inner_text(nodes, i))
        elif role in ('checkbox', 'radio'):
            value = 'on'
        elif role == 'combobox':
            value = n['value'] if n['value'] is not None else _inner_text(nodes, i)
        else:
            value = ''
        value = PUA.sub('', value or '')
        a = {**base, 'kind': 'fill' if editable else 'click', 'value': value}
        actions.append(a)
        if editable:
            entry['fill'] = a
            o = {**base, 'kind': 'click', 'value': value, 'label': 'Open ' + label}
            actions.append(o)
            entry['open'] = entry['click'] = o
        else:
            entry['click'] = a
    omitted = set()
    kept = actions[:MAX_ACTIONS]
    for k, a in enumerate(kept):
        a['id'] = 'e' + str(k + 1)
    for a in actions[MAX_ACTIONS:]:
        a['id'] = None
    # resolve action objects -> ids
    resolved = {}
    for ax_id, entry in targets.items():
        r = {}
        for key in ('click', 'fill', 'open'):
            if key in entry:
                if entry[key]['id']:
                    r[key] = entry[key]['id']
                else:
                    omitted.add(ax_id)
        if 'select' in entry:
            r['select'] = {k: a['id'] for k, a in entry['select'].items() if a['id']}
            if len(r['select']) < len(entry['select']):
                omitted.add(ax_id)
        if r:
            resolved[ax_id] = r
    opt_ids = {k: (v if v == 'selected' else v['id']) for k, v in options.items()}
    controls = []
    if more_below:
        controls.append({'id': 'scroll_down', 'kind': 'scroll', 'label': 'Scroll down', 'delta': 560})
    if scrolled:
        controls.append({'id': 'scroll_up', 'kind': 'scroll', 'label': 'Scroll up', 'delta': -560})
    controls.append({'id': 'wait', 'kind': 'wait', 'label': 'Wait for the page to update'})
    snapshot = {'url': url, 'title': clean_label(title) if title else '', 'text': page_text(nodes, hidden | consumed),
                'actions': kept + controls, 'omitted_actions': max(0, len(actions) - MAX_ACTIONS)}
    index = {'targets': resolved, 'options': opt_ids, 'native_select': native, 'omitted': omitted,
             'skipped': skipped, 'nodes_by_id': {n['id']: n for n in nodes if n['id']}, 'nodes': nodes}
    return snapshot, index


def page_text(nodes, hidden=()):
    """Visible text like snapshot.js: trimmed text nodes, newline-joined, cut at 6000 chars."""
    words, length = [], 0
    has_text = [False] * len(nodes)
    for n in reversed(nodes):  # mark nodes that have StaticText descendants
        if n['parent'] is not None and (n['role'] in TEXT_ROLES or has_text[n['index']]):
            has_text[n['parent']] = True
    skip = set(hidden)
    for n in nodes:  # text inside editable fields is a value, not a page text node
        if n['parent'] is not None and (n['parent'] in skip):
            skip.add(n['index'])
        elif _role(n) in EDITABLE_ROLES or (_role(n) == 'combobox' and 'autocomplete' in n['props']):
            skip.update(n['children'])
    for n in nodes:
        if length >= MAX_TEXT:
            break
        if n['index'] in skip:
            continue
        if n['role'] in TEXT_ROLES or (n['role'] in CONTENT_NAMED and not has_text[n['index']]):
            value = PUA.sub('', n['name']).strip()
            if value:
                words.append(value)
                length += len(value)
    return '\n'.join(words)[:MAX_TEXT]


def nearest_interactable(index, ax_id, max_up=3):
    """Closest ancestor (<= max_up levels) of ax_id that has a snapshot action."""
    node = index['nodes_by_id'].get(ax_id)
    nodes = index['nodes']
    for _ in range(max_up):
        if node is None or node['parent'] is None:
            return None
        node = nodes[node['parent']]
        if node['id'] in index['targets']:
            return node['id']
    return None


def resolve_gold(index, ax_id, op, option=None):
    """Map an agent action on AX node ax_id to a snapshot action id.

    op: 'click' | 'fill' | 'select'. Returns (action_id | None, reason | None, mapped_to_ancestor).
    """
    ax_id = str(ax_id)
    t = index['targets'].get(ax_id)
    if op == 'click':
        if ax_id in index['options']:
            eid = index['options'][ax_id]
            if eid == 'selected':
                return None, 'select_already_selected', False
            return (None, 'gold_beyond_250', False) if eid is None else (eid, None, False)
        if ax_id in index['native_select']:
            return None, 'click_native_select', False
        if t and 'click' in t:
            return t['click'], None, False
        why = _missing(index, ax_id)
        if why == 'gold_not_interactable':
            anc = nearest_interactable(index, ax_id)
            if anc and 'click' in index['targets'][anc]:
                return index['targets'][anc]['click'], None, True
        return None, why, False
    if op == 'fill':
        if t and 'fill' in t:
            return t['fill'], None, False
        why = _missing(index, ax_id)
        return None, ('type_non_editable' if why in ('gold_not_interactable', None) else why), False
    if op == 'select':
        if ax_id not in index['native_select']:
            return None, _missing(index, ax_id) or 'select_non_native', False
        key = clean_label(option or '').lower()
        sel = (t or {}).get('select', {})
        if key in sel:
            return sel[key], None, False
        node = index['nodes_by_id'][ax_id]
        for j in _descendants(index['nodes'], node['index']):
            o = index['nodes'][j]
            if clean_label(o['name']).lower() == key and o['selected'] == 'true':
                return None, 'select_already_selected', False
        return None, 'select_option_not_found', False
    return None, 'unsupported_op', False


def _missing(index, ax_id):
    if ax_id in index['omitted']:
        return 'gold_beyond_250'
    if ax_id in index['skipped']:
        return 'gold_' + index['skipped'][ax_id]
    if ax_id not in index['nodes_by_id']:
        return 'gold_not_in_tree'
    return 'gold_not_interactable'


def node_is_editable(node):
    """Editability from a single node (no children needed)."""
    return _is_editable([node], 0)


def history_label(node, op):
    """Label jev-ultrafast would log for an action on this node (best effort).

    Clicks on editable fields are logged with the "Open <name>" label, like snapshot.js.
    """
    if node is None:
        return None
    label = clean_label(node['name']) or _role(node)
    if op == 'click' and node_is_editable(node):
        return 'Open ' + label
    return label
