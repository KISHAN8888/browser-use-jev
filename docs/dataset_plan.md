# Dataset plan: open browser-decision dataset in Jev (jev-ultrafast) format

Licence target: non-commercial, open release (inherits CC-BY-NC-SA-4.0 from WebLINX).
Input format: see `jev_ultrafast_format.md`.

## Size (staged; grow only if the learning curve says so)
- Train: start at 5k–10k cleaned steps. Train at 2.5k → 5k → 10k and plot benchmark accuracy.
  Add more data only if the curve is still rising at 10k.
- Benchmark: ~2k steps, all from websites unseen in training (±2 pts at 95% CI). Eval only, never trained on.
- Rough cost at 10k steps, 2 epochs, Qwen3-4B LoRA: ~$4–5 on a rented 4090, ~$8–10 on an A100/H100.
  Jev on the 2k benchmark: < $1. Dataset building: free (open data).
- Keep tokens per step down: trim page text, keep only in-viewport elements.

## Sources
| source | use | est. usable steps | licence |
|---|---|---|---|
| NNetNav-live, NNetNav-WA | train | ~75k after dedupe | Apache-2.0 |
| Go-Browse-WA (traj_reward=1) | train (WAIT/SELECT/DONE coverage) | ~26k | MIT (repo) |
| Mind2Web + Multimodal-Mind2Web train | train, real snapshot.js tables | ~7.7k | CC-BY-4.0 / openrail |
| WebLINX (browsergym) | train | ~20k (TBD after action filtering) | CC-BY-NC-SA-4.0 |
| MM-Mind2Web test_website / test_domain | benchmark | 1,019 / 4,060 | CC-BY-4.0 |
| AgentTrek | only after licence is confirmed | ~45k | unstated |
| WebChain v2 | round 2 | up to ~200k | CC-BY-4.0 |
| AgentRewardBench | DONE/BLOCKED eval | ~1.3k trajectories | unstated |

## Pipeline
1. Download sources.
2. Normalise to one intermediate schema: task, url, history, elements[role,name,value,state], page_text, gold_op, gold_target(s), website, domain, source.
3. Rebuild real snapshot.js tables from raw HTML where available (Mind2Web): headless browser + `snapshot.js`, map gold `backend_node_id` → index. Other sources: AX tree → row format.
4. Render Jev requests: dynamic operation question + one target question per available operation; verbatim NEXT_ACTION/TARGET prompts. SCROLL/WAIT/DONE/BLOCKED have no target.
5. Clean: gold element must be in the table, drop goto/tab/back, dedupe best-of-N, successful trajectories only, ≤250 actions, downsample DONE.
6. Fill gaps: BLOCKED (mined login/captcha/denied pages + synthetic splices), WAIT (noop, loading states).
7. Label audit: 300-step manual + LLM check; multi-gold for equivalent elements.
8. Splits by website and domain; all WebArena on one side.

## Benchmark difficulty (chosen by features up front, never by Jev's errors)
Unseen site/domain, ≥150 elements, near-duplicate distractors, multi-constraint goals, autocomplete/date pickers/dropdowns,
history-dependent steps, DONE vs not-yet, BLOCKED. Report overall and per-category.
