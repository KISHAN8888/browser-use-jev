"""AX-tree -> snapshot.js mimicry, on small handmade trees in each dialect (no network)."""
from myjev.convert.axtree import axtree_to_snapshot, parse_axtree, parse_line, resolve_gold
from myjev.convert import gobrowse, nnetnav
from myjev.convert.axcommon import website
from myjev.jevformat import make_record

# BrowserGym dialect (Go-Browse): no ids on StaticText, `key=value` props, options with selected=.
BROWSERGYM = """RootWebArea 'OpenStreetMap', focused, url='http://ec2-1-2-3-4.compute.amazonaws.com:3000/directions'
\t[44] banner ''
\t\t[46] link 'OpenStreetMap logo OpenStreetMap', url='http://x:3000/'
\t\t\t[49] image 'OpenStreetMap logo', url='http://x/logo.svg'
\t[158] textbox 'From'
\t[163] textbox 'To' value='Times Square', focused
\t\tStaticText 'Times Square'
\t[166] combobox '' value='Car (OSRM)', hasPopup='menu', expanded=False
\t\t[167] option 'Bicycle (OSRM)', selected=False
\t\t[168] option 'Car (OSRM)', selected=True
\t\t[169] option 'Foot (OSRM)', selected=False
\t[171] button 'Go'
\t[172] button 'Save changes', disabled=True
\t[173] checkbox 'Squash commits', checked='false'
\t[174] switch 'Forks', checked='true'
\t[175] tab 'Open 6', selected=True, controls='x'
\t[176] spinbutton 'Size limit', disabled=True, valuemin=0, valuemax=0, valuetext=''
\t[180] paragraph ''
\t\tStaticText 'Hosting is supported by'
\t\t[193] link 'UCL', url='https://www.ucl.ac.uk/'
\t[200] gridcell 'Mon 3'
\t\t[201] button '3'
"""

# WebArena dialect (NNetNav-WA): ids everywhere, `key: value` props, native select as menuitems.
WEBARENA = """Tab 0 (current): Search results

[5485] RootWebArea "Search results for:'sunscreen'" focused: True
\t[6721] link 'My Account'
\t[6665] link'store logo'
\t\t[6729] img 'one_stop_market_logo'
\t[6917] combobox '\\ue615 Search' autocomplete: both hasPopup: listbox required: False expanded: False
\t\t[7869] StaticText 'sunscreen'
\t[6920] button 'Search' disabled: True
\t[6691] StaticText "Search results for:'sunscreen'"
\t[6766] combobox 'Sort By' hasPopup: menu expanded: False
\t\t[6865] menuitem 'Product Name' selected: False
\t\t[6866] menuitem 'Price' selected: False
\t\t[6867] menuitem 'Relevance' selected: True
\t[6900] menuitem '\\ue622 Beauty & Personal Care' hasPopup: menu
\t[6950] listitem ''
\t\t[6951] link 'Sunscreen SPF 50'
\t\t\t[6952] StaticText 'Sunscreen SPF 50'
\t[6990] checkbox 'Agree' checked: true
\t[6995] link 'Hidden' hidden: True
"""

# NNetNav-live dialect: BrowserGym plus clickable/visible flags.
LIVE = """RootWebArea 'Wolfram|Alpha', focused, url='https://www.wolframalpha.com/'
\t[55] button 'UPGRADE TO PRO', clickable, visible, hasPopup='menu', expanded=False
\t\tStaticText 'UPGRADE TO PRO'
\t\t[57] image '', visible
\t[89] textbox 'WolframAlpha input field', clickable, visible, focused
\t[206] listitem '', clickable, visible
"""


def snap(text, **kw):
    nodes = parse_axtree(text)
    root = next(n for n in nodes if n['role'] == 'RootWebArea')
    return axtree_to_snapshot(nodes, root['props'].get('url', 'http://x/'), **kw)


def by_label(snapshot):
    return {a['label']: a for a in snapshot['actions']}


def test_parse_line_dialects():
    n = parse_line("\t\t[163] textbox 'To' value='Times', focused, required")
    assert (n['id'], n['role'], n['name'], n['depth']) == ('163', 'textbox', 'To', 2)
    assert n['props'] == {'value': 'Times', 'focused': True, 'required': True}
    n = parse_line("[6917] combobox 'Search' autocomplete: both hasPopup: listbox required: False")
    assert n['props'] == {'autocomplete': 'both', 'hasPopup': 'listbox', 'required': False}
    n = parse_line("[6665] link'store logo'")
    assert (n['role'], n['name']) == ('link', 'store logo')
    n = parse_line("[5485] RootWebArea \"Search results for:'sunscreen'\" focused: True")
    assert n['name'] == "Search results for:'sunscreen'"
    n = parse_line("\tStaticText 'it\\'s'")
    assert (n['id'], n['role'], n['name']) == (None, 'StaticText', "it's")


def test_browsergym_snapshot_shape():
    s, idx = snap(BROWSERGYM)
    acts = s['actions']
    assert s['title'] == 'OpenStreetMap'
    assert [a['id'] for a in acts[:-2]] == ['e%d' % i for i in range(1, len(acts) - 1)]
    assert [a['id'] for a in acts[-2:]] == ['scroll_down', 'wait']
    labels = by_label(s)
    # editable field twice: fill + "Open <name>" click, sharing a node
    assert labels['To']['kind'] == 'fill' and labels['To']['value'] == 'Times Square'
    assert labels['Open To']['kind'] == 'click' and labels['Open To']['node'] == labels['To']['node']
    # native select: one select per unselected option, named by option texts when unlabeled
    sel = [a for a in acts if a['kind'] == 'select']
    assert [a['value'] for a in sel] == ['Bicycle (OSRM)', 'Foot (OSRM)']
    assert sel[0]['label'] == 'Bicycle (OSRM) Car (OSRM) Foot (OSRM) → Bicycle (OSRM)'
    assert all(a['current_value'] == 'Car (OSRM)' and 'expanded' not in a for a in sel)
    assert not any(a.get('role') == 'option' for a in acts)  # options consumed
    # disabled excluded; checkbox/switch state; tab selected
    assert 'Save changes' not in labels and 'Size limit' not in labels
    assert labels['Squash commits']['checked'] == 'false' and labels['Squash commits']['value'] == 'on'
    assert labels['Forks']['checked'] == 'true'
    assert labels['Open 6']['selected'] == 'true'
    # gridcell containing a button is skipped, the button is kept
    assert 'Mon 3' not in labels and '3' in labels
    # text: StaticText plus content-named nodes whose text was pruned; no select options, no field values
    assert 'Hosting is supported by' in s['text'] and 'UCL' in s['text'] and 'Go' in s['text']
    assert 'Bicycle (OSRM)' not in s['text'] and 'Times Square' not in s['text']
    # gold mapping
    assert resolve_gold(idx, '163', 'click')[0] == labels['Open To']['id']
    assert resolve_gold(idx, '163', 'fill')[0] == labels['To']['id']
    assert resolve_gold(idx, '166', 'select', 'Foot (OSRM)')[0] == sel[1]['id']
    assert resolve_gold(idx, '169', 'click')[0] == sel[1]['id']  # clicking an option == selecting it
    assert resolve_gold(idx, '168', 'click')[1] == 'select_already_selected'
    assert resolve_gold(idx, '166', 'click')[1] == 'click_native_select'
    assert resolve_gold(idx, '172', 'click')[1] == 'gold_disabled'
    assert resolve_gold(idx, '49', 'click')[0] == by_label(s)['OpenStreetMap logo OpenStreetMap']['id']  # image in link
    assert resolve_gold(idx, '999', 'click')[1] == 'gold_not_in_tree'
    assert resolve_gold(idx, '171', 'fill')[1] == 'type_non_editable'


def test_webarena_snapshot_shape():
    s, idx = snap(WEBARENA, scrolled=True)
    labels = by_label(s)
    assert s['title'] == "Search results for:'sunscreen'"
    assert labels['Search']['kind'] == 'fill' and labels['Search']['value'] == 'sunscreen'  # PUA glyph stripped
    assert labels['Open Search']['role'] == 'combobox'
    assert 'store logo' in labels
    sel = [a for a in s['actions'] if a['kind'] == 'select']
    assert [a['label'] for a in sel] == ['Sort By → Product Name', 'Sort By → Price']
    assert labels['Beauty & Personal Care']['role'] == 'menuitem'
    assert labels['Agree']['checked'] == 'true'
    assert 'Hidden' not in labels
    assert [a['id'] for a in s['actions'][-3:]] == ['scroll_down', 'scroll_up', 'wait']
    assert 'sunscreen' not in s['text'].split('\n')  # field value is not page text
    assert "Search results for:'sunscreen'" in s['text']
    assert resolve_gold(idx, '6952', 'click')[0] == labels['Sunscreen SPF 50']['id']  # StaticText -> link
    assert resolve_gold(idx, '6950', 'click')[1] == 'gold_not_interactable'
    assert resolve_gold(idx, '6766', 'select', 'price')[0] == sel[1]['id']


def test_live_snapshot_and_record():
    s, idx = snap(LIVE, more_below=False)
    assert [a['label'] for a in s['actions']] == ['UPGRADE TO PRO', 'WolframAlpha input field',
                                                  'Open WolframAlpha input field', 'Wait for the page to update']
    assert s['actions'][0]['expanded'] == 'false'
    gold, reason, _ = resolve_gold(idx, '89', 'fill')
    rec = make_record(s, 'Find water chemistry', [], [gold], {'source': 'test'})
    assert rec['gold'] == {'operation': 'TYPE_TEXT', 'type_text_target': ['2']}
    els = rec['request']['state']['elements']
    assert els[1]['operations'] == ['TYPE_TEXT', 'CLICK']
    assert set(rec['request']['questions']['operation']['criteria']) == {'CLICK', 'TYPE_TEXT', 'WAIT', 'DONE', 'BLOCKED'}
    assert resolve_gold(idx, '206', 'click')[1] == 'gold_not_interactable'  # clickable div, no role


def test_text_cap_and_action_cap():
    lines = ["RootWebArea 'T', url='http://a/'"] + [f"\t[{i}] link 'L{i}'" for i in range(1, 301)] + \
            ["\tStaticText '" + 'x' * 5000 + "'", "\tStaticText '" + 'y' * 5000 + "'", "\tStaticText 'z'"]
    s, idx = snap('\n'.join(lines))
    assert sum(1 for a in s['actions'] if a['id'].startswith('e')) == 250 and s['omitted_actions'] == 50
    assert len(s['text']) == 6000 and 'z' not in s['text']
    assert resolve_gold(idx, '260', 'click')[1] == 'gold_beyond_250'


def test_nnetnav_parsing():
    assert nnetnav.parse_action('type [89] [water [hot] chemistry] [1]') == \
        {'verb': 'type', 'id': '89', 'text': 'water [hot] chemistry'}
    assert nnetnav.parse_action('scroll [direction=down]') == {'verb': 'scroll', 'dir': 'down'}
    assert nnetnav.parse_action('stop [N/A]') == {'verb': 'stop', 'answer': 'N/A'}
    assert nnetnav.parse_action('go_back') == {'verb': 'go_back'}
    out = "<|start_header_id|>assistant<|end_header_id|>\nblah ```click [3]``` then In summary, ```stop [$12]``````\n<|eot_id|>"
    assert nnetnav.extract_gold(out) == 'stop [$12]'
    prev = ("1: None\n2: type [6917] [sunscreen ] where [6917] is combobox 'Search' autocomplete: both\n"
            "3: scroll [down]\n4: click [6951] where [6951] is link 'Sunscreen SPF 50'\n5: hover [1] where [1] is x")
    hist, n, scrolled = nnetnav.parse_history(prev, {})
    assert n == 4 and scrolled
    assert [(h['action'], h['kind'], h['text']) for h in hist] == [
        ('Search', 'fill', 'sunscreen'), ('Scroll down', 'scroll', None), ('Sunscreen SPF 50', 'click', None)]


def test_gobrowse_parsing():
    assert gobrowse.parse_action("fill('12', 'a \"b\"')") == ('fill', ['12', 'a "b"'], {})
    assert gobrowse.parse_action("send_msg_to_user('I've done it')") == ('send_msg_to_user', ["I've done it"], {})
    assert gobrowse.parse_action("click('456', 'left')}") == ('click', ['456'], {'button': 'left'})
    assert gobrowse.parse_action("click('1')\nclick('2')") is None
    s, idx = snap(BROWSERGYM)
    assert gobrowse.gold_for('scroll', [0, 300], {}, idx, False, False)[0] == 'scroll_down'
    assert gobrowse.gold_for('scroll', [0, -300], {}, idx, False, False)[1] == 'scroll_up_not_offered'
    assert gobrowse.gold_for('noop', [], {}, idx, False, False)[0] == 'wait'
    assert gobrowse.gold_for('select_option', ['166', ['Foot (OSRM)']], {}, idx, False, False)[0] is not None
    assert gobrowse.gold_for('report_infeasible', ['x'], {}, idx, False, True)[:2] == ('BLOCKED', None)
    assert gobrowse.history_for('click', ['167'], {}, idx)['kind'] == 'select'
    assert gobrowse.history_for('click', ['158'], {}, idx)['action'] == 'Open From'


def test_website_names():
    assert website('http://ec2-3-148.us-east-2.compute.amazonaws.com:7780/admin', webarena=True) == 'shopping_admin'
    assert website('http://ec2-3-148.compute.amazonaws.com:8023/', webarena=True) == 'gitlab'
    assert website('https://www.booking.com/city/us.html') == 'booking.com'
