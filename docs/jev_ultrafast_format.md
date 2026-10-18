# Jev request format used by browser-use/jev-ultrafast

Source: `jev_ultrafast/model.py` (`action_space`, `choose`) and `jev_ultrafast/snapshot.js`, commit 1231850.
Every training record we build must reproduce this request, plus the gold answer for each question.

## One step = one request

```json
{
  "model": "jev-latest",
  "state": {
    "page": {"url": "https://www.google.com/travel/flights", "title": "Google Flights", "text": "<visible page text, max 6000 chars>"},
    "elements": [
      {"index": "1", "role": "combobox", "label": "Where from?", "value": "Zürich", "operations": ["TYPE_TEXT", "CLICK"]},
      {"index": "2", "role": "combobox", "label": "Where to?",   "value": "",       "operations": ["TYPE_TEXT", "CLICK"]},
      {"index": "3", "role": "button",   "label": "Search",                          "operations": ["CLICK"]},
      {"index": "4", "role": "combobox", "label": "Cabin class", "value": "Economy", "operations": ["SELECT"],
       "options": [{"index": "4:1", "label": "Cabin class → Business", "value": "business"}]}
    ],
    "recent_actions": [
      {"action": "Where from?", "kind": "fill", "text": "Zürich", "page_changed": true}
    ]
  },
  "questions": {
    "operation": {
      "type": "choice",
      "criteria": {
        "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
        "TYPE_TEXT": "Enter or replace text in an editable field. A small LLM will supply the value from the goal.",
        "SELECT": "Select an observed dropdown value.",
        "SCROLL_DOWN": "Scroll down",
        "WAIT": "Wait for the page to update",
        "DONE": "Every requirement is visibly satisfied.",
        "BLOCKED": "No supported operation can progress."
      },
      "instructions": {"goal": "<user task>", "rules": "<NEXT_ACTION prompt>"}
    },
    "click_target":     {"type": "choice", "criteria": {"1": {"element": "[1] Open Where from?", "current_value": "Zürich", "role": "combobox"}, "3": {"element": "[3] Search", "current_value": "", "role": "button"}}, "instructions": {"goal": "...", "operation": "CLICK", "rules": ["<NEXT_ACTION>", "<TARGET>"]}},
    "type_text_target": {"type": "choice", "criteria": {"1": {...}, "2": {...}}, "instructions": {"goal": "...", "operation": "TYPE_TEXT", "rules": ["..."]}},
    "select_target":    {"type": "choice", "criteria": {"4:1": {...}}, "instructions": {"goal": "...", "operation": "SELECT", "rules": ["..."]}}
  }
}
```

## Rules that matter for the dataset

- **Operations are dynamic.** `SCROLL_UP` only if scrollY > 0, `SCROLL_DOWN` only if more page below, `WAIT` always,
  `DONE` and `BLOCKED` always. CLICK/TYPE_TEXT/SELECT only if some element supports them.
- **One target question per available operation**, each listing only compatible elements. Only the head matching the
  chosen operation is used. Training label: the gold operation + the gold target in that operation's head.
- **Elements**: visible, enabled, in-viewport, interactable (`a[href], button, input, textarea, select, summary,
  contenteditable`, plus 14 ARIA roles). Password/file/hidden inputs excluded. Max 250 actions per page.
- **Editable fields appear twice**: `fill` (TYPE_TEXT) and `click` with label `"Open <name>"`.
- **SELECT targets are options**, indexed `"<element>:<n>"`, one per unselected option.
- **Page text**: visible text nodes in viewport only, joined by newlines, cut at 6000 chars.
- **History**: last 10 actions, fields `action` (label), `kind`, `text`, `page_changed`.
- Typed text is not Jev's job; a separate LLM writes it. Our model only needs operation + target.
- Prompts `NEXT_ACTION`, `TARGET` are in `jev_ultrafast/questions.py`; copy them verbatim into every record.
