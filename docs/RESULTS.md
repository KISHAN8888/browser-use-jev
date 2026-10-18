# Results: open decision model vs TypeSafe Jev in jev-ultrafast's request format

Benchmark: 3,239 human-labelled browser steps on 84 websites never seen in training (Mind2Web test_website +
test_domain, WebLINX test_web), in browser-use/jev-ultrafast's exact request format. Frozen file
`data/frozen/bench_v1.jsonl` (sha256 f95b894c4161ad7c059624916e6f7e4db9460bbd16c3b16a485481534436bd37).
106 of 3,345 steps that Jev's API rejected as too large (`max_tokens_exceeded`) are excluded for both systems.

Model: Qwen3-4B-Base + LoRA (r16) + candidate-scoring head, one forward pass, no text generation; temperature-calibrated.
Checkpoint: `/vol/runs/q4b_cont_v3_resume/milestones/m21646` (Modal volume `myjev-data`).

## Headline (after the fairness audit, 2026-10-05)

**Mind2Web (Cross-Website + Cross-Domain, 2,074 steps, 62 unseen websites): ours 64.8% vs Jev 57.8% Step SR,
+7.0 points (paired 95% CI 4.9–9.2); ours ahead on 44 of 62 websites.** Ours is ~3× faster with ~4× lower
calibration error. Jev remains better on TYPE_TEXT and SELECT.

The all-splits number below (62.1 vs 48.7) **overstates the gap** and should not be the headline. An adversarial
audit found that WebLINX test_web penalises Jev for dataset conventions it was never shown:
- WebLINX has no DONE action, so "keep scrolling while reading" is gold; in 11–12 of 14 sampled cases the answer
  was already on screen (Jev answered DONE). 249 scroll-gold steps: Jev 5.1%, ours 66.4% (learned from WebLINX train).
- "Click the field, then type" is still labelled CLICK in the bench (fixed in train only); typing directly is valid.
- WebLINX goals are concatenated chat transcripts (passwords, "use ChatGPT" side-trips) with noisy history; Jev
  degrades as the transcript grows; our model trained on the same format.
- Minor: `page_changed` blanked (~0.3 pt against Jev); Mind2Web rendered without CSS (affects both; our model saw
  the same rendering in training).

| Slice | Jev | Ours |
|---|---|---|
| All splits, as published below | 48.7 | 62.1 |
| All, scroll-gold steps dropped + lenient click-then-type | 56.9 | 64.4 |
| **Mind2Web only** | **57.8** | **64.8** |
| WebLINX, scroll dropped + lenient | 51.5 | 61.5 |

Checked and clean: scoring (independently re-implemented, all cells reproduced), id matching, symmetric exclusion of
the 106 Jev-rejected steps, no leakage (website, URL host, task/demo ids, goals, page tables). Note: candidate sets are
viewport-filtered (median 34 elements), so these numbers are not comparable to published Mind2Web results.

## Full table, all splits (see caveats above)

| | Cross-Website | | | Cross-Domain | | | WebLINX test_web | | | All | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | Ele. Acc | Op. Acc | **Step SR** | Ele. Acc | Op. Acc | **Step SR** | Ele. Acc | Op. Acc | **Step SR** | Ele. Acc | Op. Acc | **Step SR** |
| TypeSafe Jev 1.13 | 66.2 | 79.5 | **63.2** | 56.4 | 80.6 | **55.3** | 51.9 | 47.9 | **32.6** | 57.2 | 68.6 | **48.7** |
| **Ours (Qwen3-4B)** | **73.8** | **90.5** | **71.4** | **62.8** | **91.1** | **61.7** | **62.6** | **79.1** | **57.2** | **65.2** | **86.6** | **62.1** |

Steps: Cross-Website 660 (10 sites), Cross-Domain 1,414 (52), WebLINX test_web 1,165 (26).
Step SR 95% CI: ours 60.5–63.7, Jev 47.1–50.3.

## Calibration and speed

| | ECE action ↓ | ECE element ↓ | Latency p50 |
|---|---|---|---|
| Jev 1.13 | 0.062 | 0.077 | 321 ms (API) |
| Ours | **0.017** | **0.021** | ~105 ms (H100, batched benchmark scoring) |

## Notes
- Op. Acc is action-type accuracy: neither system writes typed text (jev-ultrafast uses a separate text model), so
  Mind2Web's Op. F1 reduces to action type.
- Lenient scoring that also accepts typing directly into a field the human clicked: Jev 52.8%, ours 64.0% Step SR.
- Where Jev is still better: TYPE_TEXT steps (77.1% vs 65.6%) and SELECT (81.2% vs 39.6%, only 48 steps).
  Where we are much better: CLICK (62.2% vs 47.2%) and SCROLL_DOWN (66.4% vs 5.1%).
- Training data: 21,646 steps from 301 websites (Mind2Web train, WebLINX train, WebChain, NNetNav, Go-Browse),
  none on benchmark websites; click-then-type label inconsistencies removed (`myjev/relabel.py`).
- Reproduce the table: `python scripts/mind2web_table.py --records data/frozen/bench_v1_jev_answered.jsonl
  --pred Jev=runs/jev_bench_v1.jsonl --pred Ours=runs_modal/final/bench_pred.jsonl`
