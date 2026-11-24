# Phase 6b — the measurement was the result

Phase 6 was run twice, at d=0.25 and at the matched d=0.20, and failed its
gate both times. The second failure was supposed to discriminate between two
explanations. It discriminated against both of them, and against the way the
first five phases were measured.

## 1. The density hypothesis is dead

`voi_mean` is the KL between the policy with messages and the same policy in
silence — the direct measure of whether a message changes behaviour.

| | d = 0.25 | d = 0.20 |
|---|---|---|
| `voi_mean` (nats) | 7.7e-4 | 7.4e-4 |
| `selection_advantage` | 1.011 | 1.016 |
| `comm_rate` | 0.243 | 0.244 |

Identical. The prediction in `FINDINGS-phase6.md` was that a materially
higher `voi_mean` at d=0.20 would show the VoI signal was real and merely
density-gated. It is not higher. Communication carries no measurable
information at either density, so the scheduler selects at chance (1.0),
exactly as it must when every candidate message is equally worthless.

## 2. The arm differences were evaluation noise

`eval_eps_per_env=2` at `num_envs=64` is 128 evaluation episodes, worth about
±4pp of binomial noise on success rate. The observed swing between
*consecutive* evaluations of one converged run is larger still:

```
phase6_d20 a16_none_s1, last five evals:   0.812  0.688  0.656  0.578  0.469
phase6_d20 a16_learned_s0, last five:      0.750  0.594  0.859  0.750  0.797
```

Both ablation scripts reported `[r for r in history if "eval_success" in r][-1]`
— one draw from that distribution, per seed. Re-aggregating the **same runs**
over the last 10 evaluations (~1,280 episodes per seed, no retraining):

| comparison (success) | final eval only | last 10 evals |
|---|---|---|
| learned − none | +0.053 | **−0.035** |
| learned − full | +0.128 | +0.025 |
| learned − random | +0.106 | **−0.002** |
| learned − distance | +0.153 (p=0.026) | +0.098 (p=0.025) |

| comparison (collisions) | final eval only | last 10 evals |
|---|---|---|
| full − none | +0.021 | **+0.000** |
| learned − random | +0.013 | +0.006 |

Every apparent effect except "learned beats `distance`" dissolves. Beating
`distance` survives because `distance` is genuinely a bad heuristic — always
messaging your nearest neighbours is worse than messaging at random — which
is a real but minor finding.

**With adequate evaluation, `none`, `full`, `random` and `learned` are the
same policy to within noise, and `none` is nominally the best of them.**

## 3. Phase 5's headline was inflated

The same re-aggregation applied to Phase 5's eight seeds:

| eval window | none | full | delta | p | d |
|---|---|---|---|---|---|
| final only (as published) | 0.1009 | 0.0760 | **−0.0248** | 0.0003 | −2.51 |
| last 5 | 0.0931 | 0.0827 | −0.0104 | 0.0056 | −1.66 |
| last 10 | 0.0902 | 0.0861 | −0.0041 | 0.0365 | −1.16 |

The effect keeps the same sign and stays nominally significant, but shrinks
by 6× as the measurement gets more precise. Monotone shrinkage under
increasing precision is the signature of an estimate inflated by noise, not
of a stable effect. Success moves the other way and becomes *more* clearly
negative (−0.041, p=0.011 at the last-10 window).

**The honest Phase 5 claim is a ~4% collision reduction bought with a ~4pp
success loss, not 25%.** `docs/FINDINGS-phase5-final.md` has been corrected
a second time.

## 4. Why there is nothing to communicate

`comm_mode="none"` sets `ctx = torch.zeros_like(h)` (`models/comm_policy.py`).
Two consequences, both of which were in every run from Phase 4 onward:

1. `IntentComm` receives no gradient during BC — it enters PPO at random
   initialisation.
2. The policy head is trained on `cat([h, zeros])`, so it has learned a
   function that is correct precisely when the context is zero.

PPO then starts feeding that policy the output of an untrained attention
module. The cheapest repair available to gradient descent is to drive the
context half of the policy weights toward zero — to learn to ignore
messages. `voi_mean` ~ 7e-4 is the measurement of that having happened.

But the deeper problem is upstream. BC on `a16h20` reaches **98.4% action
accuracy with communication switched off**. A 9×9 window plus the global
cost-to-go field determines the prioritised-planner expert's next move almost
everywhere. Communication has at most 1.6% of decisions available to improve.

A mechanism cannot beat a ceiling that low, and no scheduler can allocate
information that does not exist.

## 5. What changes

Measurement, fixed and pinned:

- `eval.stats.eval_window(history, k)` averages the last *k* evaluations;
  both ablation scripts now use it with `--eval-window 10` by default.
- `eval_eps_per_env` 2 → 8 (≈512 episodes per evaluation; negligible cost
  against 20M agent-steps).
- Four regression tests, including one asserting that windowing at least
  halves the sampling spread of a noisy run.

Next experiment — `scripts/comm_headroom.py`, minutes rather than GPU-hours.
It runs BC with the message pathway off and on and reports what fraction of
the residual error communication closes. The premise is testable in
supervised learning, where there is no RL variance to hide in, and it should
have been tested there first.

Regimes worth probing, in order of how much local information they remove:
smaller FOV (`make_demos.py --fov 5`), which is the one lever that directly
attacks the 98.4% ceiling; then the cost-to-go channel, which hands every
agent a globally-computed shortest-path field and is arguably the single
biggest reason the local decision is trivial.

**A regime that shows no headroom under BC cannot support a communication
result under PPO, and should not be given a 25-GPU-hour ablation.**

---

## 6. The shuffled control settles it: capacity, not information

The first headroom run looked like good news — `comm=full` closed 21% of BC's
residual error — and §4 above was briefly rewritten to say the premise was
alive. It is not. Adding the capacity control:

| demos | err none | err **shuffled** | err full | closed | survives shuffling |
|---|---|---|---|---|---|
| a16h20 (fov 9) | 0.0136 | **0.0112** | 0.0109 | 19.9% | **+2.3%** |
| a16h20_fov5 | 0.0155 | **0.0113** | 0.0114 | 26.4% | **−0.5%** |

`shuffled` permutes which agent's intent lands in each attention slot: the
graph, the attention module and every parameter stay in use, and only the
correspondence between sender and message is destroyed. It matches `full`
exactly — at fov=5 it is nominally *better*. The validation curves are
indistinguishable epoch by epoch:

```
epoch       0      2      4      6      8     10
full     .187   .121   .100   .095   .084   .083
shuffled .187   .121   .100   .095   .085   .083
```

**The entire BC gain is the extra usable capacity that `comm="none"` leaves
idle by zeroing the context. None of it is the content of the messages.**

This is the same conclusion `voi_mean` ~ 7e-4 reached in PPO, now reproduced
in supervised learning at a thousandth of the cost, with a control that
separates the two explanations. Both measurements agree.

### Shrinking the window did not create headroom

`fov=5` was meant to attack the 98.4% ceiling by removing local information.
It barely moved: BC error 0.0136 → 0.0155, and agents-on-goal was *unchanged*
(0.894 at fov 9, 0.897 at fov 5). A 9×9 window is not what makes the expert
predictable. The remaining suspect is the **cost-to-go channel**, a
globally-computed shortest-path field handed to every agent — which is a
strange thing to call partial observability.

### What this probe does not prove

BC measures whether peer intent helps predict *the expert's* action, and the
expert is a prioritised planner whose output is conflict-free by
construction — it never reacts to a peer, because conflicts were resolved
jointly before the first step. A decentralised learned policy does have
conflicts to resolve, so communication could in principle help it where it
cannot help predict the planner.

The honest statement is therefore narrower than "communication is useless":
**peer intent carries no information about the expert's action, and PPO fine-
tuning measures no information either.** Those are two independent
measurements pointing the same way, not a proof.

## 7. Two instrument failures found while probing

**`voi_mean` was never logged for `comm="full"`.** It was computed inside the
`sched is not None` branch, so only the scheduled arms produced it — the arm
with the cleanest version of the question ("every agent transmits; does
behaviour change?") silently reported nothing. It is now a diagnostic for
every communicating mode, with no loss term attached outside the scheduler.

**The critic warmup was destroying the warm start.** `critic_only` omitted the
policy loss but still back-propagated the value loss through the *shared*
encoder and GRU. On a checkpoint that passed its pre-flight at 0.925
agents-on-goal:

```
iter  0  critic_only=True   entropy=0.318  tentative_entropy=0.404  on_goal=0.594
iter  5  critic_only=True   entropy=0.622  tentative_entropy=1.078  on_goal=0.875
iter 10  critic_only=False  entropy=0.673  tentative_entropy=0.917  on_goal=0.229
iter 25                     eval_success=0.000, makespan=192 (every episode timed out)
```

The intent head had already doubled its entropy by iteration 5 — before a
single policy gradient was applied. Ten iterations of pure value loss rewrote
the trunk the behaviour-cloned policy depends on. The run recovered by
iteration 50 and reached 0.691 success, but it spent a quarter of its budget
repairing damage the warmup caused.

`_keep_only_value_grads(model)` now drops gradients outside the value head
during warmup, with a regression test. **Every PPO result in Phases 4-6 was
produced with the broken warmup**, which is one more reason the arms looked
alike: each of them was partly retrained from a damaged starting point.

---

## 8. The signal exists at initialisation and PPO deletes it

With both instruments repaired — VoI logged for every communicating mode, and
the value warmup no longer rewriting the shared trunk — a 2M-step run
warm-started from the `comm="full"` BC checkpoint:

```
iter   voi_mean   comm_rate   sel_adv   entropy
   0     0.0444       0.000     0.000     0.089   <- straight from BC
   5     0.0357       0.000     0.000     0.058
  10     0.0416       0.388     0.966     0.064
  15     0.0310       0.375     1.048     0.136
  20     0.0082       0.106     1.376     0.100   <- scheduler finds signal
  25     0.0025       0.087     1.421     0.119   <- budget undershoots badly
  35     0.0012       0.135     0.774     0.109
 121     0.0007       0.231     1.057     0.195   <- back to the Phase 6 value
```

Three things here had never been visible before.

**The BC checkpoint starts at `voi_mean` = 0.044 — sixty times the 7e-4 that
every run in Phases 4-6 reported.** Those runs warm-started from a
`comm="none"` checkpoint, so they began at zero and stayed there. The low
number was never a property of the environment; it was a property of the
starting point.

**`selection_advantage` reached 1.42.** The scheduler is not broken. Given
anything to rank, it ranks 40% better than chance. Every previous reading of
~1.01 was the scheduler correctly reporting that its input was constant.

**The budget overshoots.** The dual variable drove `comm_rate` from 0.388
past its 0.25 target down to 0.087, and the policy spent that window of
near-silence learning to do without messages. By the time the rate recovered
to 0.25 the policy was message-independent, and `voi_mean` never came back.

## 9. Why none of that rescues the thesis yet

It is tempting to read §8 as "the thesis was right and the warm start was the
bug". That reading does not survive contact with §6.

`policy_shift` is a KL divergence: it measures how much a message **changes**
behaviour, not whether the change is an **improvement**. A trained but
uninformative attention module perturbs the policy and scores a high KL. The
BC shuffle control already established that this channel is uninformative —
permuting who sent what left validation loss identical to three decimals.

Both observations are consistent, and together they explain the 1.42:

> The scheduler learned to predict which messages are *influential*. That is
> a property of the state, perfectly learnable, and says nothing about
> whether the influence helps. PPO then discarded an input that perturbs
> without helping, which is the correct thing to do.

So `voi_mean` = 0.044 decaying to 0.001 may be a policy shedding noise, not
losing information. **The instrument could not distinguish these, which makes
it the wrong instrument for the claim the project wants to make.**

`models/voi.py:value_of_information` is the fix: the advantage-weighted
log-ratio of the action actually taken, positive only when the message moves
probability toward actions the critic rates above average, and ~0 when the
shift is unrelated to action quality. It is now logged as `voi_value`
alongside `voi_mean` in every communicating run. `voi_mean` still decides
*what* to transmit; `voi_value` decides whether transmitting is worth
anything.

**The decisive comparison is `full` vs `shuffled`, not `full` vs `none`**, and
it has never been run in RL. `shuffled` is now an arm of the Phase 6 ablation.

## 10. A result that does not depend on the communication question

The same run, at 2M agent-steps against Phase 6's 20M:

| | Phase 6 d=0.20, best arm, 20M steps | this run, 2M steps |
|---|---|---|
| success | 0.650 | **0.727** |
| agents-on-goal | 0.913 | **0.971** |
| makespan | ~120 | **106** |

Same architecture, same environment, a tenth of the compute. The difference
is the warmup fix plus a warm start whose attention module was trained. The
earlier probe, run before the fix, collapsed from 0.925 agents-on-goal to
0.229 by iteration 10 and spent a quarter of its budget recovering; this one
never collapsed (`train_on_goal`: 0.562 → 0.887 → 0.926 → 1.000).

**Every PPO number in Phases 4-6 was produced with the broken warmup and a
comm-less warm start.** Whatever the communication question turns out to be,
those baselines are too weak to publish and need re-running.

---

## 11. Settled: the messages carry influence, not value

5 seeds, 4M steps, `--eval-every 10 --eval-window 8` (24 evaluations per run,
8 averaged). Seed standard deviation on success fell from ~0.10 to ~0.014,
which is what finally made these differences resolvable.

| arm | success | collisions | on-goal |
|---|---|---|---|
| none | **0.7621** (sd 0.011) | **0.1760** | 0.9752 |
| full | 0.7296 (sd 0.017) | 0.2093 | 0.9722 |
| shuffled | 0.7336 (sd 0.014) | 0.1672 | 0.9720 |

```
full vs shuffled    success     -0.0040   p=0.694   d=-0.26
                    collisions  +0.0421   p=0.040   d=+1.57
full vs none        success     -0.0325   p=0.009   d=-2.31
                    collisions  +0.0333   p=0.080   d=+1.29
shuffled vs none    success     -0.0285   p=0.008   d=-2.27
                    collisions  -0.0088   p=0.552   d=-0.39
```

**Real messages are indistinguishable from permuted ones on success, and
significantly *worse* on collisions.** Both communicating arms lose the same
~3pp of success against silence (p=0.009 and p=0.008, d≈-2.3). The cost comes
from carrying the pathway at all, not from what the messages say.

### The new instrument separated the hypotheses

On the `full` arm, trained iterations only, 24 readings per seed:

| seed | voi_value (second half) | voi_mean (second half) |
|---|---|---|
| s0 | −0.0026 | 0.0209 |
| s1 | +0.0023 | 0.0167 |
| s2 | −0.0034 | 0.0318 |
| s3 | +0.0007 | 0.0279 |
| s4 | −0.0015 | 0.0245 |

`voi_mean` ≈ 0.024: the messages substantially change behaviour.
`voi_value` ≈ −0.001, sign varying by seed: those changes are **uncorrelated
with action quality**.

That is the precise signature §9 predicted — influence without value — and
`policy_shift` alone could never have shown it. Every earlier reading of
`voi_mean` was measuring the wrong quantity, which is why a scheduler could
reach 1.42× chance on a channel worth nothing.

### The n=3 signal was noise, with the sign reversed

The previous control run gave full vs shuffled collisions at **d=−1.76**
(full better). At 5 seeds with 6× the evaluation it is **d=+1.57** (full
worse), p=0.040. Two "large effects" in opposite directions from the same
comparison. It was flagged at the time as a hint rather than a finding; that
was the right call, and it is the cleanest illustration in this project of
what effect sizes mean at n=3.

### The claim, at the width the evidence supports

In this benchmark — 16 agents, 32×32, d=0.20, peer goals hidden, a
prioritised space-time A* expert — **learned intent communication carries no
value and costs about 3pp of success.** Four independent measurements agree:

1. BC shuffle control: `full` ≡ `shuffled` to three decimals, at two FOVs
2. RL shuffle control: `full` ≡ `shuffled` on success, worse on collisions
3. `voi_value` ≈ 0 while `voi_mean` ≈ 0.024
4. Both communicating arms significantly below no-comm

This does not generalise to MAPF as a field. It is a statement about a
benchmark whose expert is 98.6% predictable from a local window plus a
global cost-to-go field — and identifying *that* as the binding constraint is
the contribution.

## 12. The no-communication baseline

| | Phase 6, best arm, 20M steps | Phase 7 `none`, 4M steps |
|---|---|---|
| success | 0.650 | **0.762** |
| agents-on-goal | 0.913 | **0.975** |
| collisions | ~0.10 (single eval) | 0.176 (8 evals) |
| seed sd (success) | 0.109 | **0.011** |

A tenth of the seed variance and 11pp more success at a fifth of the compute.
None of it came from communication: it came from the value warmup no longer
destroying the warm start, a warm start whose attention module was trained,
and an evaluation that measures the trained policy rather than one noisy draw
plus a pre-training row.

---

## 13. Removing the global cost-to-go field

Sections 6 and 9 named the cost-to-go channel as the remaining suspect: a
globally computed, obstacle-aware shortest-path map handed to a nominally
local observation, and the most likely reason the expert is 98.6% predictable
without communication. `obs_costmap="none"` zeroes channels 4 and 5 and the
cost at the current cell while keeping the straight-line bearing and distance
in the vector block, so the agent still knows **where** its goal is but not
**how to route around obstacles** to reach it. The channel count is unchanged,
so the network and its parameter count are identical across levels.

| observation | err none | err shuffled | err full | closed | vs shuf |
|---|---|---|---|---|---|
| 9x9 + field | 0.0136 | 0.0112 | 0.0109 | 19.9% | +2.3% |
| 5x5 + field | 0.0155 | 0.0113 | 0.0114 | 26.4% | −0.5% |
| **9x9, no field** | **0.0384** | **0.0372** | **0.0373** | **2.9%** | **−0.2%** |

**The headroom hypothesis was right and the communication hypothesis was
still wrong.** Removing the field nearly tripled the residual error, from
0.0136 to 0.0384, and closed-loop success fell from 0.599 to 0.224. The
expert is no longer locally predictable. Communication nonetheless closes
only 2.9% of the enlarged error, and the shuffled control matches full
communication to within 0.0001 — so even that 2.9% is capacity rather than
content.

This is the experiment the earlier sections could only speculate about, and
it removes the most obvious objection to the negative result: that the
benchmark was simply too easy for messages to matter. Made harder in exactly
the way the hypothesis predicted, it still shows nothing.

### An honest caveat

At this level the behaviour-cloned policy is weak in absolute terms: 0.224
success against 0.599 with the field. Collisions also *fall*, from 2.37 to
1.27 per agent-step, which is the signature of agents that move less and
traverse shorter distances rather than agents that coordinate better. It is
therefore possible that removing the field substituted one bottleneck for
another — that basic navigation, not coordination, now dominates the residual
error — and that an intermediate regime, with a coarse or noisy heuristic
field, would be the one where coordination binds. We have not tested that,
and we do not claim this result rules it out.

What the three rows together do establish is narrower and still useful: across
a 3x range of task difficulty, two observation window sizes and two levels of
global information, the shuffled control matches full communication every
time. Nothing we have varied has produced a regime in which the content of a
message matters.

---

## 14. Scaling to 32 agents

`a32h20`: 32 agents on 40x40 at density 0.20, peer goals hidden, 938,592
expert transitions, no planning failures. Density, horizon and visibility all
match the 16-agent regime, so agent count is the only deliberate variable.
Five seeds per arm, 4M agent-steps, warm-started from one shared `comm="full"`
behaviour-cloning checkpoint.

| arm | success | collisions | on-goal |
|---|---|---|---|
| none | **0.5740** (sd 0.045) | **0.8718** | 0.9530 |
| full | 0.3872 (sd 0.072) | 1.1296 | 0.9392 |
| shuffled | 0.3957 (sd 0.029) | 1.1110 | 0.9405 |

```
full vs shuffled    success     -0.0085   p=0.8156   d=-0.16
                    collisions  +0.0186   p=0.8480   d=+0.13
full vs none        success     -0.1867   p=0.0019   d=-3.13
                    collisions  +0.2578   p=0.0064   d=+2.49
shuffled vs none    success     -0.1783   p=0.0002   d=-4.75
                    collisions  +0.2392   p=0.0296   d=+1.84
```

The control is again indistinguishable from real communication on both
metrics, now at p > 0.8 with effect sizes below 0.2. Both communicating arms
lose 18 points of success and gain 27% more collisions against silence, at
p < 0.01.

`voi_mean` averages 0.019 and `voi_value` averages -0.0012 with a
seed-dependent sign -- the same influence-without-value signature as at 16
agents. `peers_in_range` rises from 3.84 to 5.16, so there are materially
more neighbours to talk to and the channel still carries nothing.

### The penalty is larger here, but the reason is not isolated

At 16 agents the communication penalty was 3.3 points of success; here it is
18.7. It is tempting to call that a scaling effect. It is not safe to.

A budget fixed in **agent-steps** does not hold gradient updates fixed across
agent counts. At 64 environments and a rollout of 32, one iteration consumes
`64 x 32 x N` agent-steps, so 4M steps buys 121 PPO iterations at 16 agents
and only 61 at 32. The 32-agent policies received half the updates, which
also explains their lower absolute success (0.574 against 0.762).

Every arm within this table received the same 61 iterations, so the
comparisons above are sound. What cannot be separated from this run alone is
whether the larger penalty reflects agent count or reflects communication
hurting more in an under-trained policy. Resolving it needs 8M agent-steps at
32 agents, which restores 122 iterations and matches the 16-agent run on
updates rather than on samples.

**The claim this run supports:** at 32 agents, real messages are
indistinguishable from permuted ones, and communication costs success. **The
claim it does not yet support:** that the cost of communication grows with
agent count.

---

## 15. Iteration-matched 32 agents, and a metric that lies

Re-running at 8M agent-steps restores 122 PPO iterations, matching the
16-agent run on gradient updates rather than on samples.

| arm | success | **agents-on-goal** | collisions |
|---|---|---|---|
| none | 0.4679 (sd 0.067) | **0.9487** (sd 0.0083) | 0.3218 |
| full | 0.4287 (sd 0.042) | **0.9473** (sd 0.0031) | 0.3126 |
| shuffled | 0.4340 (sd 0.025) | **0.9479** (sd 0.0023) | 0.2832 |

```
full vs shuffled    success  -0.0053  p=0.8172  d=-0.15
full vs none        success  -0.0392  p=0.3058  d=-0.70
shuffled vs none    success  -0.0339  p=0.3359  d=-0.67
```

### The 18.7-point penalty was an artefact

Section 14 measured an 18.7-point success penalty for communication at 32
agents and flagged that it might be an under-training effect rather than a
scaling effect. It was. At matched iterations the penalty is **3.9 points and
not significant** (p=0.31), statistically indistinguishable from the 3.3
points measured at 16 agents.

The inflated figure came from the evaluation window landing on different
parts of a non-monotonic curve: the 4M run had only six post-training
evaluations and averaged all of them, including iterations 10 and 20 where
the `none` arm peaks; the 8M run averages the last eight, all after that
peak. **There is no evidence that the cost of communication grows with agent
count.** The claim is withdrawn.

### Success rate amplifies agents-on-goal by an order of magnitude

Success requires every one of N agents to be on its goal, so it compounds the
per-agent arrival rate. Regressing success on agents-on-goal across all 195
evaluations at each scale:

| | on-goal (sd) | success (sd) | d(success)/d(on-goal) |
|---|---|---|---|
| 16 agents | 0.9717 (0.0089) | 0.7342 (0.0932) | 10.0 |
| 32 agents | 0.9463 (0.0130) | 0.4485 (0.1372) | 8.8 |

A one-point move in agents-on-goal shifts success by nine or ten points. The
dramatic trajectory of the `none` arm -- 0.906 at iteration 10, 0.094 at
iteration 50, 0.562 at iteration 120 -- corresponds to an agents-on-goal
range of 0.970 to 0.893 to 0.955. The underlying movement is real but modest;
success magnifies it tenfold.

**Agents-on-goal is the statistic to report as N grows.** On it, the three
arms at 32 agents are 0.9487, 0.9473 and 0.9479: a spread of 0.0014, with the
two communicating arms differing by 0.0006. Every conclusion survives the
change of metric, and the seed variance falls by a factor of eight.

One further observation: all five `none` seeds peak at iteration 10 and
settle slightly below that peak (on-goal 0.970 falling to roughly 0.955).
More training is mildly harmful here even on the stable metric, which is
worth understanding but does not affect the comparison between arms, since
every arm receives identical treatment.

### What the two 32-agent runs jointly establish

Real messages are indistinguishable from permuted messages at 32 agents, on
success (p > 0.8), on agents-on-goal (a difference of 0.0006) and on
collisions. This holds at both compute budgets and under both evaluation
windows. It is the fourth independent confirmation of the capacity
explanation, and the first at a different scale.

---

## 16. A difficulty dial: the intermediate observation level

Section 13 tested the two extremes of global information and found the
shuffle control matching full communication at both. The caveat recorded
there was that removing the cost-to-go field entirely may have substituted
one bottleneck for another -- that basic navigation, rather than
coordination, dominates the residual error at that level -- and that some
intermediate regime would be where coordination binds. `obs_costmap="coarse"`
builds that rung.

The field is quantised into bands of eight BFS steps and the downhill channel
is zeroed. The agent still learns roughly how far it is from its goal, but
not which adjacent cell is strictly closer, which is the single feature that
most directly determines the expert's next move. The straight-line bearing is
retained at every level, and the channel count never changes.

| level | distinct cost values | downhill marks | BC error (comm off) |
|---|---|---|---|
| field | 65 | present | 0.0136 |
| **coarse** | **9** | **absent** | **0.0279** |
| none | 1 | absent | 0.0384 |

If coarse lands between 0.0136 and 0.0384 the dial works, and the question
becomes whether `vs shuf` is positive anywhere along it. A positive reading
at the middle rung would mean coordination binds at an intermediate level of
global information and that the two extremes were each limited by something
else -- the most interesting outcome still available to this project. A third
null would mean the channel carries nothing across the whole range, which is
the strongest form of the result.

Note that the coarse level is a harder test of the hypothesis than it may
appear. Dropping the downhill channel removes the explicit marking of
improving neighbours while leaving the agent enough information to navigate,
so a message that says "I intend to enter the cell you are heading for" has
more to contribute here than at either extreme.

### Result: a third null, at the midpoint

| level | cost values | err none | err shuf | err full | closed | vs shuf |
|---|---|---|---|---|---|---|
| field | 65 | 0.0136 | 0.0112 | 0.0109 | 19.9% | +2.3% |
| **coarse** | **9** | **0.0279** | **0.0276** | **0.0275** | **1.7%** | **+0.6%** |
| none | 1 | 0.0384 | 0.0372 | 0.0373 | 2.9% | −0.2% |

The dial behaves exactly as designed: error rises monotonically, 0.0136 to
0.0279 to 0.0384, so the intermediate level really is intermediate. And
communication carries nothing there either. The share of error it closes is
**lower** at the midpoint than at either end, and six tenths of a percent of
that survives the shuffle.

This closes the last hypothesis the project had left. The caveat recorded in
Section 13 -- that removing the field entirely may have substituted a
navigation bottleneck for a coordination one, and that an intermediate regime
might be where coordination binds -- has now been tested rather than merely
acknowledged. It is not where coordination binds. Nothing we have varied has
produced a regime in which the content of a message matters:

- two observation window sizes (9x9, 5x5)
- three levels of global heuristic information (full, banded, absent)
- two agent counts (16, 32)
- two training budgets at 32 agents
- supervised learning and reinforcement learning
- influence measured directly, and value measured directly

Six independent routes to the same answer.

## 17. 64-agent demonstrations

`a64h20`: 1,377,088 transitions over 224 episodes, 64 agents on 64x64 at
density 0.20, peer goals hidden, no planning failures. Generation takes 137
seconds and the single shard is 108 MB. Note the action distribution is 52.9%
stay, against 46.0% at 16 agents -- agents spend more of the episode waiting
as the fleet grows, which is itself worth remembering when reading any
per-step metric at this scale.

## 18. 64 agents (13 of 15 runs)

64 agents on 64x64 at density 0.20, 16M agent-steps, 32 environments, which
gives 244 PPO iterations. The run was stopped by hand partway through the
fourth `shuffled` seed, so the arms have 5, 5 and 3 seeds respectively.
Agents-on-goal is the primary metric here; success is near-useless at this
scale, since 0.94 raised to the 64th power is essentially zero and the
amplification documented in Section 15 is correspondingly worse.

| arm | agents-on-goal | success | collisions | n |
|---|---|---|---|---|
| none | **0.9436** (sd 0.0056) | 0.2184 | 0.3218 | 5 |
| full | 0.9337 (sd 0.0104) | 0.1829 | 0.3359 | 5 |
| shuffled | 0.9251 (sd 0.0303) | 0.1771 | 0.4032 | 3 |

```
full vs shuffled    on-goal  +0.0086  p=0.678  d=+0.38
full vs none        on-goal  -0.0099  p=0.109  d=-1.19
shuffled vs none    on-goal  -0.0185  p=0.403  d=-0.85
```

`voi_mean` is 0.0325 and `voi_value` is -0.0015: the influence-without-value
signature again, now at the largest scale tested.

The pattern is consistent with every earlier scale -- full and shuffled
indistinguishable, both at or below silence -- but **nothing here reaches
significance**, and the reason is the three-seed `shuffled` arm. One of its
three seeds lands at 0.8907 against 0.9369 and 0.9478, giving a standard
deviation of 0.0303, five times that of the `none` arm. This is the same
small-sample behaviour that reversed the sign of a large effect in Section
11, and these numbers should not be reported as a finding until the two
missing seeds are run.

### The agent counts are not a clean series

Worth recording against any future scaling claim. Holding obstacle density
at 0.20 while changing both agent count and map size does not hold agent
density constant:

| preset | agents | free cells | agents per free cell | peers in range |
|---|---|---|---|---|
| a16 | 16 | 819 | 0.0195 | 3.84 |
| a32 | 32 | 1280 | 0.0250 | 5.16 |
| a64 | 64 | 3277 | 0.0195 | 4.59 |

a16 and a64 share an agent density; a32 is the densest of the three. The
neighbour counts differ for a further reason, namely that a radius-8
communication disc is frequently clipped by the boundary of a 32x32 map and
rarely by that of a 64x64 one. So the three points vary agent count, agent
density and boundary effects together. For the comparison between arms this
is harmless, since every arm within a scale sees an identical environment,
but no trend across scales should be read from these three points.

### Complete: 64 agents, five seeds per arm

| arm | agents-on-goal | success | collisions |
|---|---|---|---|
| none | **0.9436** (sd 0.0056) | 0.2184 | 0.3218 |
| full | 0.9337 (sd 0.0104) | 0.1829 | 0.3359 |
| shuffled | 0.9290 (sd 0.0229) | 0.1915 | 0.3803 |

```
full vs shuffled    on-goal  +0.0047  p=0.689  d=+0.27
full vs none        on-goal  -0.0099  p=0.109  d=-1.19
shuffled vs none    on-goal  -0.0146  p=0.230  d=-0.88
```

**The control result holds at a third scale**: real messages and permuted
messages are indistinguishable at 64 agents, p = 0.689, as they were at 16
(p = 0.694 on success) and 32 (p = 0.817). That is the claim this project
exists to make, and it is now confirmed at every scale tested.

The secondary claim is weaker and should be stated as such. Communication
costs about one point of agents-on-goal against silence, in the same
direction as every other scale, but p = 0.109 at 64 agents and p = 0.306 at
32. **Only the 16-agent comparison against silence reaches significance.** The
`shuffled` arm remains the noisiest at this scale (sd 0.0229 against 0.0056
for `none`, driven by one seed at 0.8907), and five seeds are not enough to
resolve a one-point difference against that spread.

So the honest summary across the three scales is: the content of a message
never matters, at any scale; whether carrying the pathway costs performance
is established at 16 agents and merely suggested at 32 and 64.

---

## 19. Structured maps: rooms work, mazes cannot be demonstrated

Random grids at density 0.20 leave agents plenty of room to route around one
another, which may be why intent never mattered: a conflict that can be
sidestepped does not need to be negotiated. Corridors and doorways remove
that option, so structured maps are the natural next regime.

Both generators already existed. Sizes were chosen by measuring the largest
connected component so that **agent density is held fixed**, which the agent
count series of Section 18 failed to do:

| map | size | free cells | agents per free cell |
|---|---|---|---|
| random, d=0.20 | 32x32 | 813 | 0.0197 |
| room, 6-cell rooms | 32x32 | 824 | 0.0194 |
| maze | 41x41 | 799 | 0.0200 |

Within three percent of each other, so map structure is the only variable.

### Rooms: generated cleanly

`a16room`: 612,768 transitions, 864 episodes, **no planning failures**, 20.6
seconds. Doorways are one cell wide and each 6x6 room has a single exit per
wall, so traffic concentrates at the doors.

### Mazes: the expert cannot solve them

Demonstration generation never wrote a shard. Prioritised planning fails on
essentially every maze instance, and the failure is structural rather than a
matter of congestion:

```
maze 41x41, prioritised planning success over 6 scenarios
   2 agents: 3/6      6 agents: 0/6     12 agents: 0/6
   4 agents: 1/6      8 agents: 0/6     16 agents: 0/6
```

**Two agents fail half the time.** The cause is visible in the degree
distribution of free cells:

| map | degree 1 | degree 2 | degree 3 | degree 4 | share with degree <= 2 |
|---|---|---|---|---|---|
| maze | 43 | 715 | 41 | 0 | **94.9%** |
| room | 0 | 82 | 282 | 460 | 10.0% |

Only 41 of 799 maze cells are junctions. Prioritised planning reserves each
agent's goal cell permanently, so an agent whose goal sits in a corridor --
which is to say almost any agent -- permanently severs that corridor for
everyone planning after it. No priority ordering recovers from this, and
raising the restart count does not help, because the obstruction is a
property of the goal assignment rather than of the ordering.

This is worth stating plainly, because it is a limitation of the paradigm and
not of our implementation. **The regime in which communication ought to
matter most is the regime in which a prioritised-planning expert cannot
produce demonstrations at all** [SUPERSEDED -- see 19.1: true at 16 agents,
false at 12 and below, and the 16-agent figure was measured under a horizon
too short for the map], and imitation-bootstrapped methods in the
PRIMAL line therefore cannot be evaluated there without a complete solver
such as conflict-based search. A decentralised learned policy could in
principle handle a maze, since it can back out of a corridor and yield, but
there is no expert to clone from.

Rooms are the structured regime that remains testable, and they are the
sensible next measurement: doorway bottlenecks concentrate exactly the
head-on conflicts that an intent message is supposed to resolve.

### 19.1 CORRECTION: most of that failure was the horizon, not the paradigm

The reading above is wrong, and the error is mine. `scripts/make_demos.py`
passed `max_steps=192` for every preset, and `experts/generate.py` hands that
straight to the planner as `max_t`. Maze shortest paths on 41x41 run to **432
steps** -- a corridor cannot be cut across, so distances are several times
those of a random grid with the same free-cell count. **Most maze instances
were unsolvable by construction, for any planner.** The table in section 19
measured a horizon, not prioritised planning.

Re-measured, same generator, same seeds, 6 scenarios per cell:

| agents | longest shortest-path | solved @ `max_t=192` | solved @ sized horizon |
|---|---|---|---|
| 2 | 312 | 3/6 | **6/6** |
| 4 | 351 | 0/6 | **6/6** |
| 8 | 419 | 0/6 | **4/6** |
| 16 | 432 | 0/6 | 0/6 |

The `max_t=192` column reproduces section 19 to within sampling noise, which
is what makes the diagnosis safe. With a horizon sized from the instance
(`2 x longest + 8N + 16`), prioritised planning solves **every** maze instance
at 2 and 4 agents.

Pushing on the upper end, 3 scenarios, 20 restarts:

| agents | solved | time |
|---|---|---|
| 12 | **3/3** | 1s |
| 16 | 0/3 | 45s |

So the boundary sits between 12 and 16 agents, and only there does the
structural argument of section 19 -- permanently reserved goals severing
degree-2 corridors -- actually bite. The degree distribution is still correct;
the conclusion drawn from it was reached about three agent counts too early.

**What stands and what does not:**

- *Not true*: "mazes cannot be demonstrated", "a limitation of the paradigm
  and not of our implementation", "there is no expert to clone from". At
  12 agents and below there is.
- *Still true*: prioritised planning fails mazes at 16 agents, where the
  original study's agent count sat, and raising restarts does not fix it.
- *New*: mazes are testable at 12 agents. Fixed presets `a12maze`, `a12room`
  and `a12h20` hold agent count constant across map kinds, which is the
  comparison the structure study needs.

CBS (`experts/cbs.py`) was built to remove this blocker and is not what
removed it. It solves the 2-agent maze in 0.05s and times out at 4 agents
within 8s: long horizons make the constraint tree expensive, so it is the
weaker option here. It stays for what it is good at -- it is complete and
optimal for sum-of-costs, it solves corridor instances prioritised planning
cannot (`tests/test_cbs.py`), and it is the fallback when an ordering fails.

**The lesson generalises past mazes.** A shared default sized for one regime
became a silent constraint in another, and the failure it produced was read as
a property of the algorithm. This is the fourth time in this project that a
measurement artefact was mistaken for a result -- after final-eval reporting
(section 2), iteration-0 contamination, and the density mismatch between two
ablation scripts. The pattern each time: a number that looked like a finding,
and a default nobody had re-derived for the new regime.

### Rooms: the eighth null

| demos | err none | err shuf | err full | closed | vs shuf |
|---|---|---|---|---|---|
| a16room | 0.0155 | **0.0115** | 0.0119 | 23.2% | **-2.3%** |

Doorway bottlenecks concentrate head-on conflicts, agent density matches the
random grid to within three percent, and the result is the same: the share of
error communication closes is entirely capacity, and this time the shuffled
control is nominally **better** than real messages.

Collecting every `vs shuf` reading the project has produced:

| regime | err (comm off) | closed | vs shuf |
|---|---|---|---|
| random, 9x9 window | 0.0136 | 19.9% | +2.3% |
| random, 5x5 window | 0.0155 | 26.4% | -0.5% |
| coarse heuristic field | 0.0279 | 1.7% | +0.6% |
| no heuristic field | 0.0384 | 2.9% | -0.2% |
| rooms | 0.0155 | 23.2% | -2.3% |

The readings scatter between -2.3% and +2.3% with a mean of -0.02% and no
consistent sign. That is what a quantity with no signal looks like measured
five times. The share of error that communication appears to close varies
from 1.7% to 26.4% across regimes -- and none of it survives destroying the
correspondence between sender and message.

## 20. QMIX: a second learning rule that cannot reach the task

The communication null rests entirely on MAPPO. A value-based learner was
added to test whether the result survives a different learning rule. It does
not get far enough to be asked. Three runs, in order.

### 20.1 Two evaluation protocols, conflated

`train/ppo_comm.py::_eval` **samples** actions. The QMIX evaluator was written
with `greedy=True`, on the argument that argmax is the greedy action of a
Q-network and so the same code path serves both. True, and beside the point:
it is a different measurement, and the pre-flight compared one against the
other and blamed the loader.

`scripts/probe_warmstart.py` factorises the two switches on
`runs/bccomm_a16h20/best.pt` (recorded 0.9212):

| comm | actions | on-goal | success | coll/step | makespan |
|---|---|---|---|---|---|
| none | sampled | 0.9556 | 0.7812 | 2.09 | 119 |
| none | **greedy** | **0.5771** | **0.0078** | 6.22 | 191 |
| full | sampled | 0.9645 | 0.8333 | 1.92 | 110 |
| full | greedy | 0.5718 | 0.0156 | 6.42 | 190 |
| shuffled | sampled | 0.9634 | 0.8281 | 1.99 | 111 |
| shuffled | greedy | 0.5728 | 0.0156 | 6.41 | 190 |
| -- | random | 0.0010 | 0.0000 | 3.04 | 192 |

Action rule: **-0.378** on-goal, and success falls 100-fold. Comm mode:
**+0.009**. A deterministic policy cannot break the symmetry between two
agents contesting a cell, so it livelocks to the step cap. Any QMIX-vs-MAPPO
number must hold the action rule fixed or state which rule each used; for a
Q-network there is no sampled variant, so the honest comparison is MAPPO
reported under both.

Incidentally, full vs shuffled under sampling: 0.9645 vs 0.9634. Ninth
independent null.

### 20.2 The warm start cannot survive the change of scale

`runs/qmix_full`, 16 agents, 4M steps, warm-started from the BC checkpoint.

| iter | q_tot | on-goal |
|---|---|---|
| 10 | **+4.36** | 0.5623 |
| 20 | +0.34 | 0.0000 |
| 40 | -7.26 | 0.0005 |
| 487 | -56.35 | 0.0452 |

The warm start survives exactly as long as `q_tot` sits at BC-logit scale and
dies as it crosses into return scale. The BC head emits logits, O(1); Q-values
here are O(-410). The first few dozen TD updates rewrite the head to fix the
scale and take the argmax ordering -- the only thing the warm start carried --
with it. A behaviour-cloned policy head is not a Q-function initialiser.

It never arrived either: true value about -410, reached -56 in 4M steps. The
whole run was spent on scale.

Underneath is the asymmetry that makes this environment hard for value-based
methods. With `step_penalty=-0.3`, N=16 and a 192-step horizon the discounted
team value is about -410, while the best and worst action for one agent differ
by 0.3 to 2.0. The regression target is dominated roughly **1400:1** by a
"time remaining" term carrying no information about what to do. MAPPO never
meets this: GAE subtracts the baseline and PPO optimises the advantage, so the
term cancels. A value-based learner must represent the absolute return *and*
resolve differences three orders of magnitude smaller inside it.

`team_reward` now defaults to the mean rather than the sum, which brings the
target to about -26 and the ratio to 85:1. A uniform rescaling: it changes the
conditioning of the regression, not the problem.

### 20.3 The degenerate fixed point

`runs/qmix_n4_scratch`, 4 agents, 16x16, d=0.15, no warm start, 1M steps.
Final: **on-goal 0.0000, coll/step 0.0000, makespan 96.0 (the cap)**, with
`td_err` down to 0.08 and `loss` ~0.1. Not unstable, not diverging -- fitting
confidently. The policy has stopped acting, and 0.0000 is below the random
floor, because a frozen agent never stumbles onto a goal.

It is a correct solution to the MDP the learner can see. Off-goal, per step:

    STAY : -0.5
    MOVE : -0.3 - 2.0 * p(blocked)
    => STAY is better once p(blocked) > 10%

Ten percent is a low bar for an untrained policy among obstacles, and the
`goal_reward` of 5.0 is reachable only by a long correct sequence that
epsilon-greedy will not find. So Q-learning converges to freezing.

MAPPO escapes only because BC puts it inside the good basin and on-policy
steps keep it there. QMIX has no such basin and cannot borrow MAPPO's (20.2).

### 20.4 Status

QMIX as it stands **cannot test the communication hypothesis**. A learner at
0.00 on-goal has no performance for communication to help or hurt; four arms
would agree, and that agreement would mean nothing. Running the four-arm
ablation in this state would manufacture a null.

Untried, in order of cost:

1. **Potential-based shaping** (`--shaping-coef`, ~10 on a 16x16 map). Now
   exposed. Policy-invariant by construction (Ng et al. 1999), so it changes
   what the learner can find, not what it should find, and all four arms get
   it -- the within-QMIX comparison stays internally valid. Must be reported,
   since the MAPPO runs had shaping off.
2. **Calibrate the Q head before exploring**: regress Q onto Monte-Carlo
   returns of the frozen BC policy, then start Q-learning. Fixes 20.2 at the
   root by getting the scale right while preserving the ordering. Not built.
3. Report the negative result as a scope limit: the communication study rests
   on one learning rule, and here is the measured reason why.

### 20.5 Shaping, and the close-out

Option 1 was tried. `runs/qmix_n4_shaped`, same config, `--shaping-coef 10`,
1M steps. It breaks the freeze attractor: on-goal rises from 0.000 at iter 320
to 0.031 at iter 487, still climbing, with collisions near zero. The
degenerate fixed point of 20.3 is escaped.

It is still failure. The late-phase slope extrapolates to roughly 8M further
steps to reach 0.5, and `success_rate` is 0.0000 throughout -- not one
instance solved.

The action-rule diagnostic, on `runs/qmix_n4_shaped/best.pt`, with a random
floor matched to this config at last:

| actions | on-goal | success | coll/step |
|---|---|---|---|
| sampled | 0.0358 | 0.0000 | 0.060 |
| greedy | 0.0391 | 0.0000 | 0.006 |
| greedy+0.05 | 0.0495 | 0.0000 | 0.041 |
| greedy+0.1 | 0.0469 | 0.0000 | 0.075 |
| greedy+0.2 | 0.0534 | 0.0000 | 0.139 |
| random | 0.0046 | 0.0000 | 0.668 |

Action noise buys +0.010 to +0.014. Compare the BC policy, where the action
rule was worth -0.378 (20.1). This policy is **not livelocked** -- it has not
learned the task. It sits 8-11x above the random floor and 20x below MAPPO's
0.9752.

**Conclusion.** QMIX is not a usable second learning rule for this study, and
the reason is measured rather than asserted: the reward has a freeze attractor
that epsilon-greedy falls into (20.3), potential-based shaping escapes it but
leaves a learner far too weak to carry an ablation (20.5), and the behaviour
cloning that rescues MAPPO cannot transfer to a Q-head because the scales
differ by three orders of magnitude (20.2). The communication result therefore
rests on one learning rule. That is a limitation of the study, and this
section is its evidence. Running a four-arm QMIX ablation on a learner at 0.03
on-goal would manufacture a null, not measure one.

Option 2 (regress Q onto Monte-Carlo returns of the frozen BC policy before
Q-learning) remains the one untried repair with a principled case. It is not
cheap and nothing above depends on it.
