# Deferred work

## DONE / BLOCKED test set (do at the end, after the main comparison works)

**Why:** after the 200-step label audit (`data/audit/summary.md`), NNetNav labels were ~25% wrong, so all NNetNav steps
were removed from the bench. The remaining bench (Mind2Web test_website + test_domain, WebLINX test_web) is human-labelled
but has almost no DONE or BLOCKED gold steps: Mind2Web never records the finished page, WebLINX has no DONE action.
So the model is trained on DONE/BLOCKED but we cannot yet score those two decisions fairly. Retraining is not needed
to add this; it is a test-set-only addition.

**Plan (pick one or both):**
1. Hand-built set of ~200 states, labelled by a Sonnet subagent and spot-checked:
   - DONE: final page states of completed human tasks (WebChain completed trajectories, Mind2Web last steps where the
     final page is available), plus near-miss "not done yet" states (one step before completion) as negatives.
   - BLOCKED: real login walls, captchas, paywalls, "Access Denied" pages (e.g. from AgentTrek states, or rendered
     pages), plus look-alike pages that are NOT blocked as negatives.
2. AgentRewardBench (`McGill-NLP/agent-reward-bench`, ~1.3k trajectories with expert success labels) for
   "is the task actually complete" judgements.

**Report** DONE/BLOCKED precision and recall separately from the main step accuracy, for Jev and our model.

## Scale training data toward ~100k steps (decide after the 5k → 10k learning curve)

**Trigger:** only if benchmark accuracy is still clearly rising from 5k → 10k (→ 20k) training steps.
Current train: 22,320 steps, 301 websites (`data/final_v2`). Converted pool before cleaning: ~36.5k.

| source | how | expected add | labels |
|---|---|---|---|
| WebChain (`myjev/convert/webchain.py`) | raise per-site cap 80 → ~200, breadth-first over its 428 sites | ~60k | human |
| WebLINX (`myjev/convert/weblinx.py`) | rerun train without per-site cap | ~+4.5k (→ ~10k) | human |
| NNetNav live/WA + Go-Browse | full conversion (`--limit 0`), keep capped as a minority | ~25k used | agent, noisier (0.5 weight) |
| AgentTrek | only if licence is confirmed | ~45k | agent |

**Bottleneck:** WebChain converts at ~30 steps/min locally (download ~1 MB page + render + snapshot.js per step),
so ~60k ≈ 33 h. Options: (1) parallel Modal CPU containers (~50 workers, ~1–2 h; measure cost on a small batch
first), or (2) two nights on the laptop for free. The frozen bench (`data/frozen/bench_v1.jsonl`) must not change;
only train/val/calib grow.
