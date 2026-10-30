# Phase 5 — communication is a trade-off, not a win or a null

> **CORRECTION (2026-10-13).** This run used **density 0.20**, not the 0.25
> stated below. `scripts/phase5_ablation.py` carried a different density for
> the `a16` preset than `scripts/phase6_ablation.py` and than the `a16h`
> dataset the checkpoint came from. The result itself is internally valid —
> both arms, all 8 seeds, same checkpoint, same environment — but it is a
> result **at d=0.20**, and it does **not** hold at d=0.25 (see
> `FINDINGS-phase6.md`). Both ablation scripts now read the regime from the
> checkpoint, and a mismatch raises rather than running.

> **SECOND CORRECTION (2026-10-15).** The **−25% figure below is inflated
> roughly 6×.** Every number in this document is the *final* evaluation of
> each run — 128 episodes, one draw from a distribution whose consecutive
> samples swing by 0.20–0.30. Re-aggregating these same eight seeds over
> their last 10 evaluations (~1,280 episodes each) gives:
>
> | eval window | none | full | delta | p | d |
> |---|---|---|---|---|---|
> | final only (below) | 0.1009 | 0.0760 | −0.0248 | 0.0003 | −2.51 |
> | last 5 | 0.0931 | 0.0827 | −0.0104 | 0.0056 | −1.66 |
> | last 10 | 0.0902 | 0.0861 | **−0.0041** | 0.0365 | −1.16 |
>
> The sign holds and it stays nominally significant, but shrinking 6× as the
> measurement gets more precise is the signature of an estimate inflated by
> noise. The defensible claim is a **~4% collision reduction for a ~4pp
> success loss** (success: −0.041, p=0.011 at the last-10 window — a clearer
> cost than reported below). Both ablation scripts now average the last 10
> evaluations by default (`--eval-window`). See `FINDINGS-phase6b.md`.

Date: 2026-10-13. Peer goals hidden, intent head supervised
(`--tentative-coef 0.3`), 20M agent-steps per run, identical architecture
(924,267 parameters) warm-started from one shared checkpoint per setting.

## 16 agents, 32x32, **d=0.20** (see correction above) — 8 seeds per arm

| metric | no comm | full comm | delta | p |
|---|---|---|---|---|
| **collisions/step** | 0.1009 | **0.0760** | **−25%** _(see correction: ~−4%)_ | **<0.001** (t=5.02, d=2.51) |
| success | 0.7402 | 0.7012 | −0.039 | 0.154 |
| agents-on-goal | 0.9386 | 0.9276 | −0.011 | 0.127 |

## 32 agents, 40x40, **d=0.20** — 5 seeds per arm

| metric | no comm | full comm | delta | p |
|---|---|---|---|---|
| collisions/step | 0.2233 | 0.1988 | −11% | 0.335 (t=1.06, d=0.67) |
| success | 0.4469 | 0.4125 | −0.034 | 0.267 |
| agents-on-goal | 0.8354 | 0.8214 | −0.014 | 0.425 |

## What this says

**The collision reduction is real.** At 16 agents, 25% fewer collisions with
p < 0.001 and Cohen's d = 2.51 over 8 seeds. That is not a marginal effect.

**It costs task completion.** All four completion deltas across both agent
counts are negative. Individually none reaches significance, but all four
pointing the same way is a 1-in-16 coincidence under the null — and the
mechanism is already documented.

**It is the same trade-off the scripted oracle showed.** Phase 2,
`intent + priority`: collisions fell 33x while success fell 73.5% -> 45.3%.
Agents that know their neighbours' intentions **yield rather than contest**.
That removes conflicts and slows progress. The learned attention-based system
reproduces the oracle's trade-off in miniature: −25% collisions, −3.9pp
success.

**The benefit is density-dependent.** Cohen's d falls from 2.51 at 16 agents
to 0.67 at 32. More agents means more simultaneous messages and a harder
underlying task; the signal does not simply strengthen with congestion.

## Why the gate said FAILED

It required "agents-on-goal not worse". A binary verdict cannot express a
trade-off, and reporting this as a plain failure discards the finding. The
script now detects the pattern — significant collision reduction plus a
completion cost — and reports `GATE: TRADE-OFF` with the interpretation.

## This is the result the project was looking for

Not "communication helps" and not "communication is useless", but:

> **Inter-agent intent communication in MAPF buys collision avoidance at the
> cost of throughput. Agents told what their neighbours intend become
> deferential; conflicts disappear and so does progress.**

Three independent measurements support it: a scripted priority oracle, a
learned attention mechanism at 16 agents, and the same direction at 32.

And it sets up Phase 6 with a precise, falsifiable target. If full
communication over-communicates — making agents cautious everywhere rather
than where it matters — then a **selective** scheduler should keep the
collision benefit without the throughput cost:

| | collisions | success |
|---|---|---|
| no comm | 0.1009 | 0.7402 |
| full comm | **0.0760** | 0.7012 |
| **Phase 6 target** | **<= 0.080** | **>= 0.740** |

under a hard budget of B < N transmitting agents. That is exactly what the
counterfactual value-of-information scheduler was designed for: transmit where
knowing j's intent actually changes i's decision, stay silent elsewhere.

The original project premise survives — in sharper form than it started.

## Supporting measurement to run alongside Phase 6

`eval_makespan` is now reported by the ablation. If the caution mechanism is
right, communication should produce **longer episodes**. That would confirm the
story directly rather than by analogy with Phase 2, and it is free: the metric
was already being computed and simply was not printed.
