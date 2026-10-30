# Phase 4 findings — a reward exploit, found by PPO

Date: 2026-10-11.

## The failure

The first MAPPO run, warm-started from behaviour cloning, collapsed:

| iteration | entropy | train return | eval success |
|---|---|---|---|
| 0 | 0.123 | — | **70.3%** |
| 10 | 0.123 | −11 | — |
| 20 | 0.198 | −3 | — |
| 28 | **0.754** | **+512** | **5.5%** |

The diagnostic signature is the contradiction: **episode return rising while
evaluation success collapses**. That combination always means the agent found
something the reward function pays for that the task does not want. Rising
entropy confirmed it — PPO was *broadening* the policy, not sharpening a good
one.

## The mechanism

`goal_reward` was paid on every transition *into* an agent's goal cell, not
just the first. So:

```
step off goal:  leave_goal_penalty -1.0  +  step_penalty -0.3  =  -1.3
step back on:   goal_reward        +5.0  +  step_penalty -0.3  =  +4.7
                                                        net    =  +3.4 / cycle
```

Repeatable until the step limit. Crucially, an episode ends only when **all**
agents are home, so a single agent that arrives early can farm this
indefinitely while its teammates are still travelling. Measured in isolation
the loop paid **+13.40 per cycle** once the team bonus was also re-triggered.

This was a Phase 1 design error that no amount of unit testing on the
environment would have caught, because the environment behaved exactly as
specified. It took an optimiser actively looking for the maximum.

## The fix

`MAPFConfig.one_time_goal_reward` (default `True`) pays the arrival bonus only
on an agent's first arrival per episode, tracked by a `reached` mask that
resets with the episode. Oscillating now costs 1.6 per cycle instead of
earning 3.4.

Also added `shaping_coef`, optional potential-based reward shaping on the
cost-to-go field, `F = coef * (gamma * Phi(s') - Phi(s))` with
`Phi = -distance`. Potential-based shaping is policy-invariant (Ng et al.
1999), so any cycle returning to the same state sums to exactly zero — it
cannot be farmed by construction. Disabled by default.

Three regression tests now guard this:

* `test_goal_reward_cannot_be_farmed_by_oscillating` — two agents, one parked
  and oscillating while the other never arrives, so the episode cannot end.
  A single agent cannot express this test: arriving terminates the episode.
* `test_first_arrival_is_rewarded_but_a_second_is_not`
* `test_potential_shaping_sums_to_zero_around_a_loop`

## After the fix

Identical configuration, 119k agent-steps, CPU:

| metric | before | after |
|---|---|---|
| eval success | 5.5% | **66.4%** |
| eval agents-on-goal | 67.8% | **96.4%** |
| collisions / step | 0.650 | 0.667 |
| entropy | 0.754 (diverging) | 0.219 (stable) |
| value loss | 25.3 (rising) | **6.1** (converging) |

Training is stable and the value function is converging. Success has not yet
*improved* over the warm start (66.4% vs 66.4%), which is expected at 119k
steps out of a planned 5M — but **collisions fell from 1.745 to 0.667, a 2.6x
reduction, with success held flat.** That is the trend the project needs:
MAPPO is buying collision avoidance that behaviour cloning could not.

## Lesson for the rest of the project

Every reward term added from here gets checked for cycles before training.
The question to ask of any new term: *is there a sequence of actions that
returns the environment to an equivalent state with positive accumulated
reward?* If yes, it is farmable. Potential-based shaping is the safe way to
add dense signal, because it answers "no" by construction.

This also belongs in the paper's methods section. Reward exploits are
under-reported in MARL work, and documenting one found and fixed is more
credible than presenting a reward function as if it were obvious.
