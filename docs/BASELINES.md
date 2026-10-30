# Baseline results

All numbers: held-out map pool (`pool_seed=999`), 64 environments x 4 episodes
each = 256 episodes, `max_steps=160`. Reproduce with
`python scripts/eval_scripted.py`.

> **Measurement correction (2026-10-11).** An earlier version of this file used
> a harness that stepped until `N` episodes had completed and then kept the
> first `N`. That is **biased toward success**: solved episodes end as soon as
> the last agent arrives, while failures run to the step limit and finish last,
> so truncation preferentially discards failures. Measured inflation on a
> 16-agent config: **81.2% reported vs 58.2% true**. Every number below was
> re-measured with `eval/metrics.run_episodes`, where each environment
> contributes an equal number of episodes regardless of their duration.

## Scripted policies

| config | policy | success | agents-on-goal | collisions/step |
|---|---|---|---|---|
| 8 agents 20x20 | random | 0.0% | 0.4% | 1.34 |
| | greedy | 19.1% | 71.4% | 2.59 |
| | greedy + noise | **95.7%** | 98.8% | 1.19 |
| | intent + priority | 74.2% | 95.9% | **0.04** |
| | intent + pri + noise | 81.6% | 96.7% | 0.09 |
| 16 agents 32x32 | random | 0.0% | 0.1% | 3.07 |
| | greedy | 1.6% | 58.9% | 6.21 |
| | greedy + noise | **72.7%** | 94.3% | 2.61 |
| | intent + priority | 45.3% | 94.4% | **0.08** |
| | intent + pri + noise | 53.1% | 94.2% | 0.23 |
| 32 agents 40x40 | random | 0.0% | 0.1% | 6.04 |
| | greedy | 0.0% | 44.6% | 16.30 |
| | greedy + noise | 15.2% | 84.5% | 6.28 |
| | intent + priority | 8.2% | **90.9%** | **0.43** |
| | intent + pri + noise | **17.2%** | 91.0% | 0.58 |

## Learned policies (16 agents, 32x32, d=0.20, GPU)

| policy | success | agents-on-goal | collisions/step |
|---|---|---|---|
| BC alone, epoch 3 (env-selected) | 80.9% | 97.2% | 2.03 |
| BC alone, epoch 19 (over-trained) | 28.9% | 83.8% | 3.41 |
| **MAPPO, BC warm start, 20M steps** | **88.3%** | **99.1%** | **0.096** |

> **Correction.** An earlier version reported BC at 67.7% from a 4-epoch CPU
> smoke model, and a 20-epoch GPU run scored 28.9%. Both were artefacts of
> selecting on validation loss, which falls monotonically while closed-loop
> success collapses (see `docs/FINDINGS-phase5.md` section 4). Selecting on
> environment performance gives **80.9%** — BC does beat `greedy + noise`.

**MAPPO with no communication beats every baseline on both axes**: higher
success than `greedy + noise` (88.3% vs 72.7%) and lower collisions than the
`intent + priority` oracle (0.096 vs 0.23). RL added 59 points over its own
behaviour-cloned start.

This result forced a change of regime for Phase 5 onward. With collisions
already at 0.096 there is no headroom for communication to capture, because
observation channel 2 shows agents their neighbours' goals 69% of the time —
which is their intent. Phases 5–8 run with `observe_peer_goals=False` at
density 0.25. See `docs/FINDINGS-phase5.md`.

## What the table says

**`greedy + noise` is a strong baseline up to 16 agents** and BC does not beat
it there (90.1% vs 95.7%; 67.7% vs 72.7%). This is the expected behaviour-
cloning failure mode: the policy imitates the expert's state distribution, and
small errors compound into states the expert never visited. Phase 4 (RL
finetuning) exists to fix exactly this.

**BC wins decisively where it is hard.** At 32 agents it more than doubles
greedy+noise (38.0% vs 15.2%) with agents-on-goal at 93.5% vs 84.5% — and it
was trained only on *16*-agent data, so that is zero-shot transfer to twice the
agent count. It also transfers to room maps (57.3%) having seen only random
maps. The observation is purely local, so nothing in the network is tied to a
particular agent count or map size.

**Argmax is catastrophic** for every learned policy (2.6% at 16 agents vs 67.7%
sampled). Two deterministic agents will contest the same cell indefinitely.
Always report sampled actions; treat greedy decoding as a diagnostic only.

**The collision gap is the opportunity.** BC sits at 2.52 collisions/step at 16
agents; `intent + priority` achieves 0.23 with a far worse success rate. Neither
is good at both. A learned policy *with* intent communication should reach the
low collision rate without inheriting the starvation that wrecks the
priority rule's success — that is the project's claim, stated as a measurable
target.
