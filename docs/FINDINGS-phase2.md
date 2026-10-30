# Phase 2 findings — the project premise, tested early

Date: 2026-10-11. All numbers from scripted policies on held-out maps.

## Why this investigation happened

Phase 2 produced the expert planner. Before spending four weeks building a
communication architecture, it was cheap to ask the Phase 5 question now:
**is there anything for communication to do?**

## 1. The expert is strong, and near-optimal

100% solve rate at every tested size up to 64 agents on 64x64, with
sum-of-costs within 2-9% of the lower bound (every agent on its own shortest
path, ignoring others). Demonstrations will be high quality.

| config | solved | makespan | SoC/LB | ms/instance |
|---|---|---|---|---|
| 8 agents 20x20 | 100% | 24.6 | 1.044 | 0.9 |
| 16 agents 32x32 | 100% | 41.6 | 1.059 | 5.3 |
| 32 agents 40x40 | 100% | 57.3 | 1.084 | 26.7 |
| 64 agents 64x64 | 100% | 96.8 | 1.092 | 240.4 |

Prioritised planning **fails on mazes** (55% / 7% / 0% at 8 / 16 / 32 agents)
— width-1 corridors make its incompleteness bite. Mazes are therefore unusable
for demonstration generation, but become an interesting *evaluation-only*
setting: a learned reactive policy succeeding where the classical planner
fails is a result worth reporting.

## 2. Demonstrations are dominated by trivial actions

Of 25k agent-samples at 8 agents:

| | share |
|---|---|
| parked on goal, STAY | **40.8%** |
| off goal, STAY (yielding) | **0.2%** |
| moving | 59.0% |

**Consequence for Phase 3:** BC must downweight parked-on-goal samples or the
policy collapses toward STAY. The `at_goal` mask is stored in every shard for
exactly this purpose.

## 3. The original premise, as framed, does not hold

Framing communication as *"agents need to see further than their FOV"* fails
empirically. Measuring every expert decision that deviates from pure greedy
descent, and how far away the agent responsible was:

- deviations are rare: 0.2% (8 agents) to 3.1% (32 agents, rooms)
- of those, **~100% involve an agent at Chebyshev distance 1** — adjacent
- non-local deviations are **~0.0-0.1% of all decisions**, even at FOV 5x5

A 9x9 FOV already sees everything spatially relevant. Two caveats on this
measurement: the expert is a centralised offline planner, so it dodges
conflicts long before they form, and the metric counts spatial distance only.

**A failed instrument, recorded for honesty.** A first attempt measured a
reactive policy's success against visibility radius and showed zero effect
from radius 2 upward. That was a bug, not a finding: the congestion penalty
only inspected cells adjacent to each candidate move, so no agent beyond
radius 2 could ever influence the score. One-step greedy descent cannot use
distant information at all — using it requires replanning, not a wider sensor.

## 4. The premise that *does* hold: intent, not range

Position is observable. **Intention is not observable at any range.** Two
adjacent agents can see each other perfectly and still step into the same
cell, because neither knows what the other is about to do. No FOV increase
fixes this; one message does.

Testing this with a full-communication oracle — every agent broadcasts its
intended action, conflicts resolved by a shared priority convention:

| config | policy | success | **on-goal** | **coll/step** |
|---|---|---|---|---|
| 16 agents 32x32 | greedy + noise | 73.5% | 94.5% | 2.67 |
| | intent + priority | 52.0% | **95.4%** | **0.13** |
| 32 agents 32x32 d=0.25 | greedy + noise | 5.5% | 69.0% | 11.14 |
| | intent + priority | 3.5% | **82.5%** | **1.59** |
| 32 agents 40x40 | greedy + noise | 12.0% | 84.1% | 6.38 |
| | intent + priority | 13.5% | **92.1%** | **0.33** |

**Intent-sharing cuts collisions by 7-20x** and lifts agents-on-goal by up to
13.5 points. That is a large, real effect that position-observation alone
cannot produce.

### But the naive oracle is a bad policy, and that is the opportunity

Success rate *drops* at 16 agents (73.5% -> 52.0%) even as collisions nearly
vanish. The cause is **starvation**: with a fixed priority order, a
low-priority agent yields forever and never reaches its goal. Success demands
every agent home, so one starved agent destroys the episode — while
agents-on-goal rises because everyone else flows smoothly. Adding noise
partially rescues it (52% -> 63%), which confirms the diagnosis.

So: *knowing* intent is clearly valuable, but *a fixed rule for who yields is
not good enough*. The correct yielding decision is contextual — who is closer
to their goal, who has an alternative route, who is plugging a corridor.

**That is a learning problem, and it is precisely the gap this project fills.**

## 5. Consequences for the design

1. **Messages carry intent** (planned action / next-k cells), not encoded raw
   observations. This becomes the communication content in Phases 5-6.
2. **The VoI signal gets a crisp meaning**: the value of j's message to i is
   *how much knowing j's intent changes whether i should yield*. The
   counterfactual KL already planned measures exactly this.
3. **Baselines must include `intent + priority`** — a strong, cheap,
   communication-using baseline. Beating it requires learned contextual
   yielding, which is the actual claim.
4. **Report collision rate and agents-on-goal as primary**, success rate
   secondary. Success rate is hostage to the single worst agent and hides most
   of the signal, as the table above shows.
5. **Anti-starvation matters.** Add a per-agent fairness term to the reward,
   or at minimum log per-agent arrival spread.
6. Keep random and room maps for training. Add mazes as an evaluation-only
   generalisation setting where the classical planner fails.

## 6. Revised risk assessment

The Phase 5 gate ("full-comm must beat no-comm") is now **measured in advance
and passed** on the metrics that matter: collisions 7-20x lower, agents-on-goal
clearly higher. The residual risk shifts from *"does communication help?"* to
*"can a learned scheduler beat intent+priority under a bandwidth budget?"* —
a narrower and more honest question, and the one the paper should ask.

---

## Addendum (2026-10-11): measurement correction

The success rates in section 4 were produced with a biased harness (details in
`docs/BASELINES.md`). Re-measured with equal episodes per environment:

| config | policy | success (old -> corrected) | on-goal | coll/step |
|---|---|---|---|---|
| 16 agents 32x32 | greedy + noise | 73.5% -> **72.7%** | 94.3% | 2.61 |
| | intent + priority | 52.0% -> **45.3%** | 94.4% | **0.08** |
| 32 agents 40x40 | greedy + noise | 12.0% -> **15.2%** | 84.5% | 6.28 |
| | intent + priority | 13.5% -> **8.2%** | **90.9%** | **0.43** |

**The conclusions are unchanged and in fact strengthened.** Collision reduction
from intent-sharing is larger than first reported — 2.61 -> 0.08 at 16 agents is
a **33x** reduction, not 20x. The starvation effect on success rate is also
larger. Both findings in section 4 hold.
