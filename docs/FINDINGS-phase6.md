# Phase 6 — nothing to schedule

Date: 2026-10-13. 16 agents, 32x32, **density 0.25**, peer goals hidden,
20M agent-steps, 5 seeds, five arms on one architecture and one checkpoint.

| arm | success | agents-on-goal | collisions | comm rate |
|---|---|---|---|---|
| none | 0.5312 | 0.9025 | 0.1204 | 0.000 |
| full | 0.4688 | 0.8779 | 0.1191 | 1.000 |
| random | 0.4719 | 0.8658 | 0.1183 | 0.250 |
| distance | 0.5125 | 0.8902 | 0.1221 | 0.250 |
| **learned** | 0.5031 | 0.9025 | **0.1379** | 0.242 |

**GATE: FAILED.** The learned scheduler is *worse* on collisions than every
baseline, including budget-matched coin flips.

## Why, measured directly

| diagnostic | value | reading |
|---|---|---|
| `voi_mean` | 9e-4 nats | the policy barely changes between hearing messages and silence |
| `selection_advantage` | 1.01 (5 seeds) | selection is indistinguishable from random (1.0 = chance) |
| `comm_rate` | 0.242 | the budget binds correctly |

The chain is unambiguous. At this density **messages carry almost no
information** — the KL between `pi(.|obs, messages)` and `pi(.|obs, silence)`
is about a thousandth of a nat. With no value to measure, the
value-of-information loss has no signal, the scheduler cannot rank senders,
and it selects at chance. The machinery works; there is simply nothing for it
to do.

Note that `full` (0.1191) and `none` (0.1204) are also indistinguishable here.
The Phase 5 collision benefit does not exist at this density, so the premise
for *selective* communication is absent before the scheduler is even involved.

## The regime mismatch that obscured this

`scripts/phase5_ablation.py` set `a16` to **density 0.20**;
`scripts/phase6_ablation.py` set it to **0.25**; the `a16h` dataset is 0.25.
The two phases are therefore different experiments, and Phase 5's numbers were
documented under the wrong density.

| | d = 0.20 (8 seeds) | d = 0.25 (5 seeds) |
|---|---|---|
| none, collisions | 0.1009 | 0.1204 |
| full, collisions | **0.0760** | 0.1191 |
| effect | **−25%, p<0.001, d=2.51** | none |

That is a genuine, measurable **density dependence**, and it runs opposite to
the intuition that more congestion means more need to coordinate. The `voi_mean`
measurement explains it: at d=0.25 the binding constraint is raw congestion,
not coordination. Messages cannot route an agent around a jam that has no free
route. Communication helps where *coordination* is the bottleneck, not where
*space* is.

## Fixes

1. **Both ablations now read the regime from the warm-start checkpoint**
   instead of a hard-coded preset. Checkpoints already record `map_spec`,
   `n_agents`, `observe_peer_goals`; they now also record `max_steps` and `fov`.
2. **A training-environment mismatch raises** rather than running quietly
   (`allow_env_mismatch=True` to override deliberately), with a test.
3. **`max_steps` is adopted from the checkpoint.** Training defaulted to 128
   while the demos were generated at 192 — enough to change how many instances
   time out, and therefore success rate, with no change in policy quality.

## The decisive next experiment

Run Phase 6 at **d=0.20**, where communication demonstrably carries value:

* If `voi_mean` is materially higher there, the VoI signal is real and the
  scheduler has something to learn from. If it then beats budget-matched
  random, the project has its positive result.
* If `voi_mean` is still ~1e-3 at d=0.20 while full-comm nonetheless beats
  no-comm, then the collision benefit is **not** mediated by per-step message
  content — more likely the comm pathway acts as a regulariser or changes the
  representation. That would be a different and more interesting paper, and
  `voi_mean` is the measurement that distinguishes them.

Either way the diagnostic answers it, which is what it was built for.
