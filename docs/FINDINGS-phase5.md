# Phase 5 findings — MAPPO closed the gap, and three harness bugs

Date: 2026-10-11. GPU runs on an RTX 4060.

## 1. MAPPO beats every baseline, with no communication at all

`runs/ppo_a16`: recurrent MAPPO, BC warm start, 20M agent-steps, 16 agents,
32x32, density 0.20.

| policy | success | agents-on-goal | collisions/step |
|---|---|---|---|
| greedy + noise | 72.7% | 94.3% | 2.61 |
| intent + priority oracle | 45.3% | 94.4% | 0.23 |
| BC alone (epoch 3) | 80.9% | 97.2% | 2.03 |
| **MAPPO, no communication** | **88.3%** | **99.1%** | **0.096** |

Phase 4's gate (beat 72.7%) passed decisively. RL added **59 points** over its
own behaviour-cloned starting point, which validates the warm-start-then-
finetune methodology.

## 2. ...which leaves communication almost nothing to do

Collisions are already at **0.096**. The scripted full-communication oracle
managed 0.23, and the best achievable was around 0.08. There is no meaningful
headroom left for a message to capture.

**Why:** observation channel 2 is "other agents' goals". Measured at 16
agents with a 9x9 FOV:

| | share of timesteps |
|---|---|
| agent sees >= 1 peer in its FOV | 71.1% |
| agent sees >= 1 **peer's goal** in its FOV | **69.2%** |
| sees both a peer and that peer's goal | 61.5% |

Knowing where a neighbour is headed *is* its intent. We were handing it over
for free, 69% of the time. The Phase 2 conclusion — that intent is the thing
communication can supply — was right; we had simply left a channel open that
supplied it already.

This is also an unrealistic assumption. A real robot cannot observe another
robot's destination; it has to be told. So `MAPFConfig.observe_peer_goals=False`
removes an unrealistic affordance rather than manufacturing difficulty. The
channel is zeroed rather than deleted, so model capacity is unchanged.

**Revised regime for Phases 5–8:** peer goals hidden, density 0.25, and agent
counts extended to 32 and 64. The expert still solves 100% of instances at all
three sizes (SoC/LB 1.066–1.092), so demonstrations remain near-optimal.

## 3. Three harness bugs, all of which produced misleading output

### 3a. A false PASS from gameable gate criteria

The first ablation reported **GATE: PASSED** on two runs that had both
collapsed to ~1% agents-on-goal. The criteria were "collisions lower" and
"agents-on-goal not worse" — both trivially satisfied by a policy that does
nothing, because **frozen agents never collide**.

Fixed: a validity floor. Both arms must reach >= 50% agents-on-goal before any
comparison is attempted; otherwise the script exits `INCONCLUSIVE`. A
comparison between two broken policies is not a result.

### 3b. Warm start was optional, and silently absent

`--warm-start` defaulted to `None`, so the ablation trained from scratch.

| | agents-on-goal after 20M steps |
|---|---|
| from scratch | **0.2%** |
| BC warm start | **99.1%** |

BC warm start is not an optimisation, it is the difference between working and
not working. `--warm-start` is now a required argument.

### 3c. Synchronised environment resets

Every environment was created at t=0 and timed out on the same step, so
rollout batches were highly correlated and most iterations contained no
completed episodes at all (training statistics logged `nan`).

Measured over 20 rollout iterations of 32 steps:

| | iterations containing a finished episode |
|---|---|
| synchronised | 5 / 20 |
| staggered (`stagger_resets=True`) | **20 / 20** |

## 4. Behaviour cloning: validation loss points the wrong way

| epoch | val_loss | accuracy | env success |
|---|---|---|---|
| 3 | 0.0804 | 98.24% | **80.9%** |
| 7 | 0.0630 | 98.57% | 50.0% |
| 11 | 0.0570 | 98.69% | 37.1% |
| 15 | 0.0549 | 98.75% | 33.6% |
| 19 | 0.0543 | 98.75% | **28.9%** |

Validation loss falls monotonically while closed-loop success collapses by
**52 points**. Validation loss scores single-step accuracy on the *expert's*
state distribution; closed-loop performance depends on the policy's *own*
distribution, where small errors compound into states the expert never
visited. Fitting the expert more tightly makes the policy more brittle.

Fixed: the BC trainer now evaluates in the environment every epoch, selects
`best.pt` on `agents-on-goal`, and early-stops on patience against that metric.

This also corrects the baseline table. BC alone reaches **80.9%**, not 28.9% —
it does beat `greedy + noise`. An earlier version of `docs/BASELINES.md`
reported the over-trained checkpoint.

This table is worth a figure in the paper. It is a clean, measured
demonstration of why validation loss is the wrong early-stopping signal for
imitation learning in sequential tasks.
