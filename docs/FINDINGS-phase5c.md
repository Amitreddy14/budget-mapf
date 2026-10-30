# Phase 5 — the first valid communication ablation

Date: 2026-10-12. 16 agents, 32x32, density 0.25, peer goals hidden,
20M agent-steps per arm, 3 seeds, identical architecture (924,267 parameters
either way) warm-started from one shared checkpoint.

## Result

| metric | no comm | full comm | delta |
|---|---|---|---|
| success | 0.677 ±0.029 | 0.703 ±0.022 | +0.026 |
| agents-on-goal | 0.934 ±0.006 | 0.929 ±0.005 | −0.004 |
| **collisions/step** | **0.092 ±0.012** | **0.093 ±0.012** | **+0.001** |

**GATE: FAILED.** Collisions are identical. The +2.6pp success difference is
within noise: pooled SE ≈ 0.021 over 3 seeds, t ≈ 1.2, nowhere near
significant.

This is the first run where the comparison was valid — warm start verified at
0.896–0.903 against a checkpoint recording 0.905, both arms reaching 93%
agents-on-goal, observability adopted automatically from the checkpoint.

## The confound that must be resolved first

`tentative_entropy`, comm=full seed 0, against a maximum of ln(5) = 1.609:

| iteration | 0 | 75 | 155 | 275 | 595 |
|---|---|---|---|---|---|
| tentative entropy | **0.240** | 0.627 | 1.025 | 1.221 | **1.208** |

Behaviour cloning trained the tentative head to predict the expert action —
sharp and informative. During PPO **nothing supervised it**, so it drifted to
near-uniform within roughly 200 iterations. For most of training the broadcast
intents carried almost no information.

So the result cannot currently distinguish:

1. PPO found messages useless and let them decay, or
2. messages decayed because nothing kept them meaningful, and *were therefore*
   useless.

`CommPPOConfig.tentative_coef` (default 0.3) now adds an auxiliary
cross-entropy keeping the tentative head predictive of the action the agent
actually takes — what a receiver needs to know. **This has not yet been
validated at scale**; a CPU smoke run was too small to tell. Watch
`tentative_entropy` and raise the coefficient if it passes ~0.8.

## What the broader picture already shows

| policy | collisions/step |
|---|---|
| scripted greedy + noise | 2.61 |
| **scripted intent oracle** | **0.08** |
| learned MAPPO, no comms | **0.092** |
| learned MAPPO, full comms | 0.093 |

Explicit intent-sharing transforms a *scripted* policy — a 33x collision
reduction. But a *learned* policy reaches the same 0.09 with no communication
at all, and adding communication buys nothing further.

The natural reading: **reinforcement learning discovers implicit coordination
conventions that achieve what explicit intent-broadcast achieves.** Agents
signal through motion rather than messages. If that survives the fixes below,
it is a clean and genuinely interesting finding, and a better-supported paper
than a marginal positive result would have been.

## Before concluding

Two cheap experiments, both a few GPU-hours:

1. **Re-run the ablation with `--tentative-coef 0.3`**, confirming
   `tentative_entropy` stays low. This closes the confound. A reviewer will
   ask whether the messages were actually informative; we need to answer yes.
2. **Scale to 32 agents** (`a32h` dataset is ready). Congestion is the most
   plausible regime for communication to matter, and it is the last untested
   lever. If it fails there too, the negative result is strong.

Only after both should the paper's framing be settled.
