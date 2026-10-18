"""WebLINX-BrowserGym converter on tiny handmade CDP AX trees / metadata (no network)."""
import collections

from myjev.convert import weblinx as W


def ax_node(nid, role, name='', bid=None, children=(), parent=None, ignored=False, props=None, value=None):
    n = {'nodeId': nid, 'ignored': ignored, 'role': {'type': 'role', 'value': role},
         'name': {'type': 'computedString', 'value': name}, 'childIds': list(children),
         'properties': [{'name': k, 'value': {'type': 'boolean', 'value': v}} for k, v in (props or {}).items()]}
    if parent:
        n['parentId'] = parent
    if bid:
        n['browsergym_id'] = bid
    if value is not None:
        n['value'] = {'type': 'string', 'value': value}
    return n


# Viewport 1000x600, page 2000 tall scrolled by 100. "Below" link is off-screen.
AX = {'nodes': [
    ax_node('1', 'RootWebArea', 'Shop', children=['2']),
    ax_node('2', 'generic', '', bid='html', parent='1', children=['3', '4', '5', '7', '9', '11', '13', '14']),
    ax_node('3', 'textbox', 'Search', bid='q', parent='2', props={'editable': 'plaintext', 'focused': True}),
    ax_node('4', 'button', 'Go', bid='go', parent='2', children=['6']),
    ax_node('6', 'generic', '', bid='go-inner', parent='4'),
    ax_node('5', 'link', 'Below', bid='below', parent='2'),
    ax_node('7', 'paragraph', '', bid='p', parent='2', children=['8']),
    ax_node('8', 'StaticText', 'Visible text', parent='7'),
    ax_node('9', 'paragraph', '', bid='p2', parent='2', children=['10']),
    ax_node('10', 'StaticText', 'Far away text', parent='9'),
    ax_node('11', 'form', '', bid='form', parent='2', children=['12']),
    ax_node('12', 'button', 'Search now', bid='sb', parent='11'),
    ax_node('13', 'LabelText', '', bid='lbl', parent='2', children=['15']),
    ax_node('15', 'checkbox', 'Remember me', bid='cb', parent='13', props={'checked': False}),
    ax_node('14', 'generic', '', bid='orphan', parent='2', ignored=True),
]}
PROPS = {
    'html': {'visibility': 0.3, 'bbox': [0, -100, 1000, 2000], 'clickable': 0},
    'q': {'visibility': 1, 'bbox': [10, 10, 300, 30], 'clickable': True},
    'go': {'visibility': 1, 'bbox': [320, 10, 50, 30], 'clickable': True},
    'go-inner': {'visibility': 1, 'bbox': [325, 15, 20, 20], 'clickable': 0},
    'below': {'visibility': 0, 'bbox': [10, 900, 100, 20], 'clickable': True},
    'p': {'visibility': 1, 'bbox': [10, 100, 500, 40], 'clickable': 0},
    'p2': {'visibility': 0, 'bbox': [10, 1500, 500, 40], 'clickable': 0},
    'form': {'visibility': 1, 'bbox': [10, 200, 500, 100], 'clickable': 0},
    'sb': {'visibility': 1, 'bbox': [20, 210, 100, 30], 'clickable': True},
    'lbl': {'visibility': 1, 'bbox': [20, 400, 200, 20], 'clickable': True},
    'cb': {'visibility': 1, 'bbox': [20, 400, 20, 20], 'clickable': True},
}
VP = (1000, 600)


def page():
    return W.build_page(AX, PROPS, VP, 'https://www.shop.com/x')


def test_viewport_filter_and_text():
    snap, index, info = page()
    labels = [a['label'] for a in snap['actions']]
    assert 'Below' not in labels and 'below' in index['offscreen']
    assert 'Search' in labels and 'Open Search' in labels and 'Go' in labels
    assert 'Visible text' in snap['text'] and 'Far away text' not in snap['text']
    ids = [a['id'] for a in snap['actions']]
    assert 'scroll_down' in ids and 'scroll_up' in ids  # 100 + 600 < 2000, scrolled by 100
    assert info['scroll_y'] == 100


def test_no_scroll_when_page_fits():
    props = dict(PROPS, html={'visibility': 1, 'bbox': [0, 0, 1000, 500], 'clickable': 0})
    snap, _, _ = W.build_page(AX, props, VP, 'https://shop.com/')
    ids = [a['id'] for a in snap['actions']]
    assert 'scroll_down' not in ids and 'scroll_up' not in ids


def test_gold_resolution():
    snap, index, info = page()
    common = dict(snapshot=snap, index=index, info=info, prev_intent='say', goal='Find shoes', history=[],
                  demo_id='d', key='5', s={'axtree_path': 'd/axtrees/page-5-0.json'}, split='train',
                  site='shop.com', url='https://www.shop.com/x', n_utt=1, truncated=False, viewport=VP)
    # click on an inner (promoted) div of the button -> the button, flagged
    rec, why, label = W._gold_record('click', {'uid': 'go-inner'}, {}, **common)
    assert why is None and rec['gold']['operation'] == 'CLICK' and rec['meta']['gold_mapped_to_ancestor']
    tgt = rec['gold']['click_target'][0]
    assert rec['request']['questions']['click_target']['criteria'][tgt]['element'].endswith('Go')
    # textInput -> TYPE_TEXT on the field
    rec, why, label = W._gold_record('textinput', {'uid': 'q', 'value': 'shoes'}, {}, **common)
    assert rec['gold']['operation'] == 'TYPE_TEXT' and label == 'Search'
    # off-screen and missing targets are dropped with a reason
    assert W._gold_record('click', {'uid': 'below'}, {}, **common)[1] == 'gold_offscreen'
    assert W._gold_record('click', {'uid': 'nope'}, {}, **common)[1] == 'gold_not_in_axtree'
    assert W._gold_record('click', {'uid': 'p'}, {}, **common)[1] == 'gold_not_interactable'
    # password fields are invisible to snapshot.js
    pw = {'attributes': {'type': 'password'}, 'tagName': 'INPUT'}
    assert W._gold_record('textinput', {'uid': 'q', 'value': 'x'}, pw, **common)[1] == 'password_field'
    # click on a <label> -> its checkbox
    rec, why, _ = W._gold_record('click', {'uid': 'lbl'}, {}, **common)
    tgt = rec['gold']['click_target'][0]
    assert rec['request']['questions']['click_target']['criteria'][tgt]['role'] == 'checkbox'
    # submit after typing -> the form's only button; after a click -> dropped
    rec, why, _ = W._gold_record('submit', {'uid': 'form'}, {}, **dict(common, prev_intent='textinput'))
    assert rec['meta']['from_submit'] and rec['gold']['operation'] == 'CLICK'
    assert W._gold_record('submit', {'uid': 'form'}, {}, **dict(common, prev_intent='click'))[1] == \
        'submit_after_click'
    # scroll by sign of y
    assert W._gold_record('scroll', {'x': 0, 'y': 300}, None, **common)[0]['gold'] == {'operation': 'SCROLL_DOWN'}
    assert W._gold_record('scroll', {'x': 0, 'y': -300}, None, **common)[0]['gold'] == {'operation': 'SCROLL_UP'}
    assert W._gold_record('scroll', {'x': 50, 'y': 0}, None, **common)[1] == 'scroll_horizontal'


def test_dom_ancestor_fallback():
    strings = ['bid', 'icon', 'go', 'class', 'x']
    dom = {'strings': strings, 'documents': [{'nodes': {
        'parentIndex': [-1, 0, 1], 'attributes': [[], [0, 2], [3, 4, 0, 1]]}}]}
    assert W.dom_ancestors(dom, 'icon') == ['go']
    assert W.dom_ancestors(dom, 'missing') == []
    snap, index, info = page()
    rec, why, _ = W._gold_record('click', {'uid': 'icon'}, {}, snap, index, info, 'say', 'g', [], 'd', '1', {},
                                 'train', 'shop.com', 'u', 1, False, VP, get_dom=lambda: dom)
    assert why is None and rec['meta']['gold_via_dom_ancestor']


def test_goal_builder():
    goal, n, trunc = W.build_goal(['Hi', 'Hello there!', 'Open amazon.com', 'Find red shoes size 9'])
    assert goal == 'Open amazon.com\nFind red shoes size 9' and n == 2 and not trunc
    goal, n, trunc = W.build_goal(['a' * 1500, 'b' * 1000], max_chars=2000)
    assert goal == 'b' * 1000 and trunc
    assert W.build_goal(['Hi'])[0] == ''


def test_convert_demo_history_and_drops():
    steps = {
        '0': {'intent': 'say', 'args': {'speaker': 'instructor', 'utterance': 'Hello'}},
        '1': {'intent': 'say', 'args': {'speaker': 'navigator', 'utterance': 'Hi'}},
        '2': {'intent': 'say', 'args': {'speaker': 'instructor', 'utterance': 'Search shoes on shop.com'}},
        '3': {'intent': 'load', 'args': {'url': 'https://www.shop.com/'}, 'tab': {'url': 'https://www.shop.com/'}},
        '4': {'intent': 'textinput', 'args': {'uid': 'q', 'value': 'shoes'}, 'tab': {'url': 'https://www.shop.com/x'},
              'axtree_path': 'd/axtrees/page-4-0.json', 'extra_props_path': 'd/e/page-4-0.json'},
        '5': {'intent': 'click', 'args': {'uid': 'go-inner'}, 'tab': {'url': 'https://www.shop.com/x'},
              'axtree_path': 'd/axtrees/page-5-0.json', 'extra_props_path': 'd/e/page-5-0.json'},
        '6': {'intent': 'tabcreate', 'args': {}, 'tab': {'url': 'about:blank'}},
        '7': {'intent': 'click', 'args': {'uid': 'go'}, 'tab': {'url': 'https://www.shop.com/x'}},  # no snapshot
    }
    md = {'viewportWidth': 1000, 'viewportHeight': 600, 'url': 'https://www.shop.com/x'}
    replay = [{'type': 'chat'}] * 4 + [{'type': 'browser', 'action': {'intent': 'x', 'arguments': {'metadata': md}}}] * 4
    counters = collections.Counter()
    W.mark_stale(steps)
    out = W.convert_demo('d', steps, replay, lambda s: (AX, PROPS), 'train', lambda site, n: True, counters)
    assert len(out) == 2
    (r1, site1), (r2, _) = out
    assert site1 == 'shop.com'
    assert r1['request']['questions']['operation']['instructions']['goal'] == 'Search shoes on shop.com'
    assert r1['request']['state']['recent_actions'] == []
    assert r2['request']['state']['recent_actions'] == [
        {'action': 'Search', 'kind': 'fill', 'text': 'shoes', 'page_changed': None}]
    assert r2['meta']['turn_index'] == 5 and r2['meta']['source'] == 'weblinx' and r2['meta']['split'] == 'train'
    assert counters == {'navigator_say': 1, 'load': 1, 'tab_op': 1, 'no_snapshot': 1}
    # per-site cap
    counters = collections.Counter()
    out = W.convert_demo('d', steps, replay, lambda s: (AX, PROPS), 'train', lambda site, n: n < 1, counters)
    assert len(out) == 1 and counters['site_cap'] == 2


def test_replay_label():
    assert W.replay_label({'tagName': 'INPUT', 'attributes': {'aria-label': 'Email'}}, 'click') == 'Open Email'
    assert W.replay_label({'tagName': 'DIV', 'attributes': {}, 'textContent': '  Sign in '}, 'click') == 'Sign in'
    assert W.replay_label({'tagName': 'BUTTON', 'attributes': "{'title': 'Close'}"}, 'click') == 'Close'
