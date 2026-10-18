# myjev: end-to-end pipeline

An open decision model that replaces TypeSafe's **Jev** inside **browser-use/jev-ultrafast**. It answers the same
typed questions Jev answers (which operation, which element), in the same request/response format, and beats Jev
on a held-out benchmark of unseen websites. Defensible headline after a fairness audit: Mind2Web, 2,074 steps on
62 unseen sites, Step SR 64.8% vs 57.8% (+7.0, 95% CI 4.9–9.2); see `docs/RESULTS.md`.

```
open datasets ──convert──► Jev-format records ──clean/split──► train / val / calib / frozen bench
                                                                 │                 │
                              Qwen3-4B + LoRA + scoring head ◄───┘                 │
                                         │ train (Modal H100) → calibrate          │
                                         ▼                                         ▼
                                predictions ──────────── same scorer ──────── Jev API predictions
```

---

## 1. The task: what Jev does in jev-ultrafast

jev-ultrafast drives a real browser. On every step it:

1. runs `snapshot.js` on the page → visible text (≤6,000 chars) plus a table of on-screen interactable elements
   (`[index] role · label · value`, ≤250 actions; password/file/hidden inputs excluded);
2. sends **one request** to Jev with that state, the user's goal, the last 10 actions, and these **Choice questions**:
   - `operation`: which of the available operations (CLICK, TYPE_TEXT, SELECT, SCROLL_DOWN, SCROLL_UP, WAIT, DONE,
     BLOCKED). Only operations possible on the page are offered.
   - one `<op>_target` question per element operation (`click_target`, `type_text_target`, `select_target`),
     each listing only the elements that operation can act on. A text field appears as both a TYPE_TEXT target
     and a CLICK target ("Open <name>"); a dropdown offers one SELECT target per option (`"4:2"`).
3. executes the chosen operation on the target from that operation's question. Text to type is written by a
   separate small LLM; Jev never generates text.

Jev answers every question with a choice, a probability per option and a confidence. Our model must do the same.
Exact format: `docs/jev_ultrafast_format.md`.

---

## 2. Record format (shared by every stage)

One record = one browser step:

```json
{"request": <the exact request jev-ultrafast sends to Jev>,
 "gold":    {"operation": "TYPE_TEXT", "type_text_target": ["2"]},
 "meta":    {"id", "source", "website", "task/trajectory id", "step_index", "weight", "flags", "tags", ...}}
```

- `myjev/jevformat.py` builds `request` from a snapshot-shaped dict. It is ported from jev-ultrafast's
  `model.py` and verified **byte-identical** to upstream (`tests/test_parity.py` compares against the pinned
  upstream copy in `tests/upstream/`).
- `gold` may list several equivalent targets (multi-gold); all must share one operation.
- SCROLL / WAIT / DONE / BLOCKED have no target.

---

## 3. Dataset curation

### 3.1 Sources and how each was converted

| Source | Licence | How converted | Fidelity to real `snapshot.js` | Labels |
|---|---|---|---|---|
| **Mind2Web** (train, test_website, test_domain) | CC-BY-4.0 | Saved raw HTML rendered offline in headless Chromium; the **real** `snapshot.js` is run; gold `backend_node_id` mapped to the snapshot index (`myjev/convert/mind2web.py`) | exact script; no CSS (layout/visibility can differ) | human |
| **WebChain** | CC-BY-4.0 | Saved page HTML (with CSS refetch) rendered + real `snapshot.js` (`webchain.py`) | exact script | human |
| **WebLINX** (train, test_web) | CC-BY-NC-SA-4.0 | BrowserGym accessibility tree + recorded element boxes, so the on-screen rule of `snapshot.js` is reproduced (`weblinx.py`); goal = instructor messages so far | close | human |
| **NNetNav** live + WA | Apache-2.0 | Accessibility tree → snapshot-shaped rows (`axtree.py`, `nnetnav.py`) | approximate (whole page, no viewport) | agent-generated, noisier |
| **Go-Browse** (successful trajectories only) | MIT | same AX route (`gobrowse.py`) | approximate | agent-generated |

Repairs applied during conversion (Mind2Web): every `<a>` gets an `href` (otherwise snapshot.js ignores links),
`aria_label`→`aria-label`, saved input values/checkboxes/selected options restored, self-closing tags expanded,
pretty-print whitespace removed. The gold element is scrolled to a **seeded random height** in the viewport (not
the centre), so its position in the element table does not leak the answer.

Action mapping: click/hover/enter → CLICK, type/fill → TYPE_TEXT, select_option → SELECT (matched option),
scroll → SCROLL_DOWN/UP, noop → WAIT, stop/send_msg_to_user → DONE, `stop [N/A]`/report_infeasible → BLOCKED
(flagged `blocked_proxy`). goto/go_back/tab/hover-only/press steps are dropped (no Jev equivalent). Steps whose
gold element isn't in the snapshot are dropped, with the reason counted.

### 3.2 Cleaning, website normalisation and splits (`myjev/build_dataset.py`)

1. **Dedupe** identical steps (NNetNav-live has best-of-N duplicates).
2. **Website key** normalised across sources: `www.imdb.com`, `imdb`, `m.imdb.com` → `imdb` (subdomains count
   as the same site), so the same website can never sit on both sides of a split.
3. **Benchmark** = Mind2Web test_website + test_domain + WebLINX test_web, **capped at 120 steps per website**
   (per-part budgets), human labels only. Any training/val/calib step on a benchmark website is removed.
4. **Val / calib**: whole small websites from human-labelled sources, held out (~600 steps each).
5. **Train caps**: ≤600 steps per real website; WebArena (5 sites) limited to ≤35% of train; DONE downsampled to ≤8%.
6. **Per-record weights** (`meta.weight`): NNetNav 0.5 (≈25% wrong labels in the audit), BLOCKED proxies 0.3,
   TYPE_TEXT ×1.25, SELECT ×1.5.
7. **Shortcut hygiene**: `page_changed` set to None everywhere, HTML tags stripped from titles, bare-tag history
   labels (`span`, `svg`) replaced by `(unlabeled span)`.
8. **Difficulty tags** chosen from features only (never from any model's errors): many_elements,
   duplicate_label_distractor, multi_constraint, dropdown_or_autocomplete, date_widget, long_history.

### 3.3 Label-consistency fix (`myjev/relabel.py`, applied to train/val/calib only)

A cross-source audit found that "click the text field, then type into it" was labelled CLICK in 27–71% of cases
depending on the source, while jev-ultrafast types directly (TYPE_TEXT replaces the field's text). This taught
the first model a click bias (TYPE_TEXT step accuracy 48.6%). The fix drops: CLICK-on-field immediately followed
by TYPE into the same field (579), Mind2Web "Enter on field" clicks (9), and TYPE into a field already holding
that text (86). It raised TYPE_TEXT step accuracy to ~66–68%.

### 3.4 Benchmark audit and freeze

- A Sonnet agent audited 200 benchmark steps by hand: Mind2Web labels ≈95% correct or equivalent (the practical
  ceiling); NNetNav ≈65% → **NNetNav is excluded from the benchmark**. Equivalent alternatives found by the audit
  are added as extra gold keys for those steps.
- The benchmark is **frozen** (`data/frozen/bench_v1.jsonl`, read-only, SHA-256
  `f95b894c4161ad7c059624916e6f7e4db9460bbd16c3b16a485481534436bd37`); later data changes only touch train/val/calib.

### 3.5 Final data

| Split | Steps | Websites |
|---|---|---|
| train (`data/final_v3`) | 21,646 | 301 (Mind2Web, NNetNav live/WA, Go-Browse, WebLINX, WebChain) |
| val | 627 | 21 (human-labelled) |
| calib | 652 | 28 (human-labelled) |
| bench (frozen) | 3,345 (3,239 answered by Jev) | 84, unseen in training |

---

## 4. Model architecture

**Backbone:** `Qwen/Qwen3-4B-Base` (decoder-only LM, hidden size 2560), loaded as the transformer body only
(`AutoModel`, no LM head: we never generate text). bf16 weights, SDPA attention, gradient checkpointing.

**Adapter:** LoRA on **all linear layers** (attention and MLP projections), r = 16, α = 32, dropout 0.05.
Trainable parameters ≈ 33M (LoRA + head) out of ~4B; the base weights stay frozen.

**Scoring head:** for each candidate option, the hidden state at a **marker token** placed at the end of that
option's row goes through a small MLP:

```
score_k = Linear(256→1)( GELU( Linear(2560→256)( h[marker_k] ) ) )        # run in fp32
P(option k) = softmax over the options of that question ( score_k / T_kind )
```

The last layer is zero-initialised, so at step 0 every question is a uniform distribution. The marker is
`<|box_end|>`, an unused Qwen3 vision token, so the vocabulary never changes and checkpoints stay adapter-only.

**Input serialisation** (`myjev/model/serialize.py`), one causal sequence, ordered so candidates come last and
can attend to everything before them:

```
Goal: <task>
Recent actions: <last 10: label, kind, typed text>
URL / Title / Page text: <visible text, trimmed first if over budget>
Elements:
[1] combobox 'Where from?' value='Zürich'
[2] combobox 'Where to?'
...
Question: type_text_target (operation TYPE_TEXT)
Options:
[1] Where from? ... <|box_end|>
[2] Where to? ...   <|box_end|>
```

- The constant jev-ultrafast instruction text (~600 tokens) is left out (`--no-rules`), so ~20% fewer tokens.
- Budget 4,096 tokens. Over budget, it cuts in this order: page text → element rows that aren't candidates →
  other element rows → oldest history → the per-row cap (48 tokens). **Candidates are never dropped.**
- Mean training sequence ≈ 1,900 tokens; benchmark rows reach ~6,300 tokens with all questions packed.

**Packing:** all questions of one request share the same prefix (state). They are packed into one sequence,
`prefix + question₁ + question₂ + …`, with a 4D attention mask so each question sees only the prefix and itself,
and position ids restarting after the prefix. This is mathematically equal to running each question separately
(tested in `tests/test_serialize.py`) and answers the whole request in **one forward pass**.

**Output:** a Jev-shaped answer per question, `{"choice", "probabilities", "confidence"}`.
`myjev/model/server.py` serves `POST /v1/systemone`, so jev-ultrafast can use it by changing one URL.

---

## 5. Training

**Objective** (per question, weighted by `meta.weight`, weighted mean over the optimizer step):

```
loss = soft-target cross-entropy + 0.5 · Brier score
```

The target is one-hot (or uniform over multi-gold). Brier penalises confident mistakes, which helps calibration.
We train on the `operation` question and the gold operation's target question for each record.

**Optimisation:** AdamW (no weight decay); LR 2e-4 for LoRA and 1e-3 for the head (first run), 1e-4 and 5e-4
(continuation). 5% warmup, cosine to 0. Effective batch 16 records (micro-batch 8 × 2 accumulation),
grad-clip 1.0. Micro-batches are grouped by length; the largest one runs first so an out-of-memory error shows at
step 1. The encoded data is cached per (tokenizer, max_len, rules, file hash, serializer code).

**Infrastructure:** Modal, one H100 80GB (`modal_app.py`). Measured ~4.5–5.5 s/step, ~5,500 tokens/s, peak memory
25.5 GB. Runs are detached and write results to a Modal volume.

**Mid-run milestones:** at chosen counts of records seen (e.g. 5k, 10k, end) training pauses. It saves a snapshot,
fits calibration on the calib split, scores the benchmark with the in-memory model and commits the results. So
results stream in, and a run can be stopped early without losing anything. Milestones are verified not to
change the training trajectory (`tests/test_train.py`).

**What we actually ran:**

| Run | Data | Schedule | Result (Step SR) |
|---|---|---|---|
| 1. `q4b_full_ms` | v2, 22,320 steps | planned 2 epochs; stopped after epoch 1 (step 1,395) once the click bias was found | 5k 49.0 → 10k 54.2 → 22.3k **59.4** |
| 2. quick check `q4b_cont_check5k` | v3 (label-fixed), first 5k | continue from run 1's 22.3k snapshot, 1 epoch | **62.1** (TYPE 48.6 → 67.5) |
| 3. `q4b_cont_v3` + `_resume` | v3, all 21,646 | continue from the 22.3k snapshot, 1 epoch, LR 1e-4 → 0; interrupted at step 760 by a laptop network drop, resumed exactly from the step-625 snapshot (`--resume-step`) | **62.1** final (Cross-Website 71.4) |

Continuation (`--init-from`) loads the adapter and head exactly (tested: identical predictions before any
update), with a fresh optimizer. Resume (`--resume-step K`) rebuilds the identical batch order and LR schedule
and skips the first K steps (tested: step K+1 reproduces the original loss and LR exactly).

---

## 6. Validation, checkpoint selection and calibration

- **Val** (627 steps, held-out websites): validation NLL after each epoch; the best checkpoint by NLL is `best/`.
  When continuing, the loaded model is evaluated first and stays `best/` until val improves.
- **Calibration:** after training (and at every milestone) we fit **one temperature per question kind**
  (operation, target) on the calib split, by minimising NLL (LBFGS on log T, vectorised). Temperatures rescale
  confidence without changing any choice. Final: T_operation ≈ 1.13, T_target ≈ 1.36.
- Val, calib and bench are disjoint by website, and all are disjoint from train.

---

## 7. Evaluating Jev on the same benchmark

- `myjev/run_jev.py` sends each frozen benchmark request to the TypeSafe API (`jev-latest` = jev-1.13.0). It is
  resumable, uses one HTTP client per worker, retries network errors and keeps a thread-safe spend ledger
  (`data/jev_calls.jsonl`) with a hard dollar cap.
- 106 of 3,345 requests were rejected by Jev as too large (`max_tokens_exceeded`). These are **excluded for both
  systems**; the comparison uses the 3,239 steps Jev answered (`data/frozen/bench_jev_answered_ids.txt`).
- Total Jev spend for the benchmark: ~$1.7.

---

## 8. Metrics

Standard Mind2Web reporting (`scripts/mind2web_table.py`), per split (Cross-Website, Cross-Domain, WebLINX test_web):

- **Ele. Acc**: the element the model actually chose is a gold element.
- **Op. Acc**: the action type is right (Mind2Web's Op. F1 reduces to this because neither system writes text).
- **Step SR**: right action **and** right element. **Headline metric.**

Also: calibration error (ECE, 15 bins, operation and element), latency, a per-website macro average, and a lenient
step score that also accepts typing directly into a field the human clicked (`scripts/typefield_score.py`).
Baselines (`myjev/baselines.py`): always the most common operation (3.4% Step SR), random (5.0%), a word-overlap
rule (18.0%).

---

## 9. Results (frozen bench, 3,239 steps, 84 unseen websites)

| | Ele. Acc | Op. Acc | **Step SR** | ECE op / elem | Latency |
|---|---|---|---|---|---|
| TypeSafe Jev 1.13 | 57.2 | 68.6 | **48.7** | 0.062 / 0.077 | 321 ms |
| **Ours (Qwen3-4B + LoRA)** | **65.2** | **86.6** | **62.1** | **0.017 / 0.021** | ~105 ms |

**Caveat:** the all-splits gap overstates our lead. WebLINX penalises Jev for conventions it was never shown
(reading-scrolls as gold, click-then-type labelled CLICK, chat-transcript goals). Mind2Web-only: Jev 57.8 vs ours
64.8. Full audit and per-split numbers: `docs/RESULTS.md`. Jev remains better on TYPE_TEXT (77.1 vs 65.6) and SELECT
(81.2 vs 39.6).

---

## 10. Known limitations

- ~45% of training data (NNetNav, Go-Browse, WebLINX) comes from accessibility trees rather than the real
  `snapshot.js` output.
- Mind2Web pages render without CSS; titles are often empty and URLs synthetic, so a probe can identify the source
  from surface features (~82%). This is why element-level metrics are reported, not operation accuracy alone.
- Benchmark labels have one gold answer; ~11% of steps have equally valid alternatives (partly handled by audit
  keys and the lenient scores). Label ceiling ≈95%.
- DONE/BLOCKED decisions are trained but not yet benchmarked (no trustworthy human labels): see `docs/TODO.md`.
- Step-level accuracy is not task success in a live browser; a live jev-ultrafast evaluation is future work.

---

## 11. Reproduce

```bash
# data (local, free)
python -m myjev.convert.mind2web --split test_website ...    # and the other converters
python -m myjev.build_dataset --out data/final_v3 --reuse-splits data/final_v2/stats.json \
       --expect-bench-sha256 f95b894c4161ad7c059624916e6f7e4db9460bbd16c3b16a485481534436bd37
# Jev baseline (paid, capped)
python -m myjev.run_jev --records data/frozen/bench_v1.jsonl --out runs/jev_bench_v1.jsonl --budget 2.0
# training (Modal)
MSYS_NO_PATHCONV=1 modal volume put myjev-data data/final_v3 /data_v3
modal run --detach modal_app.py::fullrun          # or ::continue_run / ::resume_run
python scripts/milestones.py --name <run>         # watch results mid-run
# results table
python scripts/mind2web_table.py --records data/frozen/bench_v1_jev_answered.jsonl \
       --pred Jev=runs/jev_bench_v1.jsonl --pred Ours=runs_modal/final/bench_pred.jsonl
```

Related docs: `docs/jev_ultrafast_format.md` (request format), `docs/dataset_plan.md`, `docs/RESULTS.md`,
`docs/TODO.md` (DONE/BLOCKED test set, ~100k data plan), `docs/RUNBOOK.md` (runbook).
