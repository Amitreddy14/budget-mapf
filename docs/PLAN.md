# BUDGET-MAPF — project plan

**Budgeted Inter-Agent Communication with Counterfactual Value-of-Information
Scheduling for Multi-Agent Path Finding**

Started 2026-10-11 · target: results frozen by ~2026-11-23, then paper writing.

---

## 1. Why v1 failed

The v1 report describes a system the v1 code never implemented. `env.py`
defines `PredatorsPrey`; there are no per-agent goals, no obstacles and no
MAPF conflicts. Beyond that, nothing in the pipeline *could* have learned:

| Defect | Consequence |
|---|---|
| Q-head is `Dense(n_actions, activation='relu')` | Q >= 0 always, but rewards are `-distance`. The value function is unrepresentable. |
| Target network `model_` never synced | Every TD target bootstraps off a permanently random net. |
| Comm gate compiled but `.fit()` never called | "Learned scheduling" was a random sigmoid. |
| Mixer fed `np.zeros` global state, rebuilt per sample, never trained | QMIX was an untrained random projection. |
| Each agent regressed onto the *mixed* value as its own target | Loss on the wrong quantity; no gradient through the mixer. |
| Every agent received the full global state | **No partial observability => nothing to communicate.** The premise was void. |
| epsilon horizon 100k steps vs 500 steps actually run | Policy stayed ~100% random (`epsilon=0.9996` in the logs). |
| Replay capacity 100; `episode_number` default 5 | Saved results CSVs contain one row. |

No experiment was ever run that could have worked. There is therefore no
evidence against the original idea — only an untested one.

## 2. Contribution for v2

Framing matters: *"QMIX + a binary comm gate"* is already covered by SchedNet,
IC3Net, ATOC, TarMAC and MAGIC, and on the MAPF side by PRIMAL, DCC and
SCRIMP. Rebuilding the v1 design faithfully yields a working project but a
thin paper. We keep the communication-efficiency theme and replace the
mechanism.

**Primary claim.** A communication scheduler that selects which agents
transmit **their intentions** under a *hard* bandwidth budget, trained with a
dense **counterfactual value-of-information** signal rather than only the
sparse task reward.

> Revised after Phase 2 measurement (`docs/FINDINGS-phase2.md`). The original
> framing -- communication lets agents *see further* -- was tested and failed:
> ~100% of expert conflict decisions involve an adjacent agent, already inside
> the FOV. The framing that survives is that **position is observable but
> intention is not, at any range.** An intent-broadcast oracle cuts collisions
> 7-20x, so communication has clear measured value; but a fixed priority rule
> for *who yields* starves low-priority agents, which is exactly the contextual
> decision a learned policy must make.

- **Hard budget via Gumbel-Top-k + Lagrangian dual.** Exactly `B` of `N`
  agents transmit per timestep. A dual variable drives the realised rate onto
  the budget, so there is no hand-tuned penalty coefficient trading off
  against task reward.
- **Counterfactual VoI.** Periodically recompute a receiver's policy with and
  without a given message; `VoI(i->j) = KL(pi_with || pi_without)`. Regress
  the sender's priority score onto aggregated VoI. This is the direct fix for
  why v1's gate learned nothing: it had no gradient path at all.

**Enabling method (not claimed as novel).** Expert bootstrapping: behaviour-clone
from a classical MAPF planner, then RL-finetune. This is what makes the success
rates publishable and supplies an optimality-gap metric.

**Evaluation contribution.** Robustness under realistic channel conditions —
packet loss, latency, limited range — which the MARL-comms literature largely
ignores.

## 3. Architecture

```
  per agent i                                      centralised (training only)
  ───────────                                      ────────────────────────────
  FOV tensor (6 x F x F) ──► CNN encoder
  vector feats (9)       ──►    │
                                ▼
                            GRU  h_i ──────────────► message m_i = f_msg(h_i)
                                │                          │
                                │                          ▼
                                │                  scheduler score s_i
                                │                          │
                                │                  Gumbel-Top-k  (budget B)
                                │                          │  ◄── Lagrangian λ
                                │                          ▼
                                │                  selected senders
                                │                          │
                                ▼                          ▼
                        multi-head attention ◄──── messages within radius R
                                │
                                ▼
                       policy π(a|h_i, c_i)        critic V(s)  ◄── global state
                                │
                                ▼
                       action ──► env ──► reward
                                           │
                     counterfactual VoI ◄──┘  (KL between π with / without m)
```

Observation channels: obstacles · other agents · other agents' goals · own
goal · own cost-to-go field · downhill set. The cost-to-go field is the key
input — it converts "navigate to an unseen goal" into "follow the local
gradient", freeing capacity for the part that is actually hard: negotiating
with other agents.

## 4. Measured starting point

See `docs/BASELINES.md`. Success rate on held-out maps:

| Config | random | greedy | greedy+noise |
|---|---|---|---|
| 8 agents, 20x20 | 0.0% | 22.5% | 94.0% |
| 16 agents, 32x32 | 0.0% | 0.5% | 61.5% |
| 32 agents, 40x40 | 0.0% | 0.0% | 9.0% |

**Headline experiments run at 16 and 32 agents.** Below that, `greedy+noise`
already reaches 94%, so a learned policy matching it proves nothing.

## 5. Phases and gates

Each phase ends at a measurable gate. A failed gate stops the next phase.

| # | Phase | Days | Gate |
|---|---|---|---|
| 0 | Repo scaffold | Oct 11 | ✅ package, config, tests, CI-able |
| 1 | Vectorised POMDP MAPF env | Oct 11 | ✅ 14/14 tests, 390k agent-steps/s, greedy solves |
| 2 | Expert planner + demos | Oct 11 | ✅ expert 100% to 64 agents, SoC/LB 1.02–1.09; 920k samples cached |
| 3 | CNN+GRU backbone + BC | Oct 11 | ✅ 90.1% @ 8 agents, 38.0% @ 32 (zero-shot) |
| 4 | Recurrent MAPPO, no comms | Oct 11 | ✅ **88.3% success, 0.096 coll** — beats every baseline |
| 5 | Comm module (none / full / intent) | Oct 11–13 | ✅ **TRADE-OFF**: −25% collisions (p<0.001, d=2.51, n=8), −3.9pp success |
| 6 | Budgeted VoI scheduler | Oct 13–22 | ⚠️ FAILED at d=0.25 (voi_mean 9e-4 — no signal to schedule); retry at d=0.20 |
| 7 | QMIX baseline + ablations | Nov 9–14 | full baseline table, 3 seeds |
| 8 | Robustness + scalability | Nov 15–19 | loss/latency/radius sweeps; 64 agents |
| 9 | Results consolidation | Nov 20–23 | seed-averaged figures with CIs |

### Phase 5 risk — retired early

This gate was measured at Phase 2 cost with scripted policies and **passed**:
an intent-broadcast oracle cuts collisions from 2.67 to 0.13 per step at 16
agents and lifts agents-on-goal by up to 13.5 points. Communication has
demonstrable value.

The residual risk is narrower and more honest: *can a learned scheduler beat
`intent + priority` under a bandwidth budget?* That baseline is strong on
collisions but starves low-priority agents, so the headroom is in **contextual
yielding** — deciding who should give way based on who is closer to their
goal, who has an alternative route, who is blocking a corridor.

## 6. Baselines for the results table

1. greedy (uncoordinated) · 2. greedy + noise · 3. **intent + priority oracle**
· 4. BC only · 5. MAPPO no-comm · 6. MAPPO full-comm · 7. random-k scheduler
(budget-matched) · 8. distance-heuristic scheduler · 9. SchedNet-style weight
scheduler · 10. v1 binary gate, correctly implemented · 11. **ours** · 12.
QMIX variants of 5/6/11 · 13. classical planner (optimality reference)

Two baselines carry the argument. **Baseline 7 (random-k)** isolates whether
the scheduler learned anything beyond respecting the budget. **Baseline 3
(intent + priority)** is the cheap non-learned way to use communication — if
we cannot beat it, the learning adds nothing.

## 7. Metrics

**Primary: collision rate and agents-on-goal fraction.** Secondary: success
rate, makespan, sum-of-costs, suboptimality vs expert, **communication rate**
(fraction of possible messages) and bits/agent/step, robustness curves. Also
log per-agent arrival spread, to catch starvation.

Success rate is demoted deliberately. It is hostage to the single worst agent:
`intent + priority` at 16 agents nearly eliminates collisions and improves
agents-on-goal, yet its success rate *falls* because one starved agent ruins
each episode. Reporting success alone would have hidden the entire effect.

## 8. Compute

Training on the local RTX 4060 (8 GB). The env sustains ~390k agent-steps/s on
CPU, so the GPU forward pass is the bottleneck — networks stay small (CNN
~100k params). BC warm-start is what makes the RL budget fit in the schedule;
training from scratch would not.

## 9. Standing risks

- **Phase 5 negative result** — mitigations above.
- **Schedule.** Three contributions in six weeks is tight. If it slips, cut
  Phase 8 scale-up first, then the QMIX baseline. Never cut Phase 6.
- **Scheduler collapse.** The Top-k may degenerate to a fixed agent subset.
  Watch the entropy of the selection distribution; add an entropy bonus.
- **VoI cost.** Counterfactual recomputation doubles forward passes. Subsample
  it (every K steps, random agent subset).
