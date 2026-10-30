# Running on your machine (RTX 4060)

Everything so far was developed and smoke-tested on CPU. Real training runs
locally on the GPU.

## One-time setup

```powershell
cd E:\MARL2.0\mapf_v2
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

That last line must print `True`. The cloud sandbox has no GPU, so CUDA has
never been exercised — if anything breaks, it will break here first.

## Verify the build

```powershell
python -m pytest tests\ -q            # 30 tests: env, expert, model
python scripts\bench_env.py           # env throughput
python scripts\eval_scripted.py       # scripted baselines
```

## Regenerate demonstrations (optional — already committed)

```powershell
python scripts\make_demos.py a16      # 618k samples, ~30s
python scripts\make_demos.py a32      # larger, for the scale study
```

## Behaviour cloning

```powershell
python scripts\run_bc.py --demos data\demos\a16 --epochs 20 --batch-size 256 ^
    --seq-len 16 --eval-every 4 --out runs\bc_a16
```

On CPU an epoch took ~107 s at batch 48. The 4060 should manage batch 256 at a
few seconds per epoch, so 20 epochs is minutes, not hours. Watch
`env_sample_on_goal` rather than `acc` — accuracy sits at 98% almost
immediately because the task is easy to imitate and tells you little.

## Evaluate a checkpoint

```powershell
python scripts\eval_checkpoint.py runs\bc_a16\best.pt --eps-per-env 8
```

## Hyperparameters worth sweeping first

| flag | default | why it matters |
|---|---|---|
| `--parked-weight` | 0.1 | 46% of samples are "parked on goal, stay". At 1.0 the policy collapses to STAY; too low and agents wander off their goal after arriving. The single most important knob. |
| `--seq-len` | 16 | BPTT window for the GRU. Longer helps memory, costs time. |
| `--lr` | 3e-4 | OneCycle schedule, 10% warmup. |
| `--no-gru` | off | Feed-forward ablation. Worth running — if it matches the GRU, the recurrence is not earning its place and Phase 5 gets cheaper. |

## Notes

- `pool_seed=999` gives evaluation a held-out map pool. Never evaluate on the
  training pool.
- Always report **sampled** actions. Argmax deadlocks (2.6% vs 67.7% at 16
  agents) because deterministic agents contest the same cell forever.
- `runs/bc_a16_cpu_smoke/` holds the 4-epoch CPU reference model, for
  comparison only. Retrain properly before using anything as a baseline.

## MAPPO (Phase 4)

```powershell
# warm-started from the behaviour-cloned policy
python scripts\run_ppo.py a16 --bc runs\bc_a16\best.pt --steps 5000000 ^
    --num-envs 64 --rollout 32 --minibatches 4 --eval-every 25 --out runs\ppo_a16
```

Memory: activations scale with `rollout x (num_envs * n_agents / minibatches)`.
At `--num-envs 64 --rollout 32` with 16 agents and 4 minibatches that is 256
agents x 32 steps per backward pass, roughly 250 MB of activations — safe on
8 GB. If you hit OOM, raise `--minibatches` before lowering `--num-envs`;
it costs nothing in sample efficiency.

### What to watch

| signal | healthy | trouble |
|---|---|---|
| `entropy` | slowly falling | **rising** = policy diverging, usually a reward exploit |
| `train_return` vs `eval_success` | move together | **return up, success down** = reward exploit (see FINDINGS-phase4.md) |
| `vf_loss` | falling then flat | rising = critic diverging; lower `--lr` |
| `approx_kl` | 0.005–0.02 | >0.05 = updates too large |
| `clipfrac` | 0.05–0.2 | >0.3 = lower `--lr` or raise `--minibatches` |

`--value-warmup 10` trains the critic alone for the first 10 iterations. The
BC checkpoint has no critic, so without this the first policy updates chase
meaningless advantages and can wreck the warm start within a few hundred steps.

`--bc-kl 0.1` anchors the policy to the BC reference with a KL penalty that
decays to zero. Off by default; try it if the warm start degrades early.

## Phase 5 — communication

The gate is an ablation, not a single run. Both arms use the *same*
architecture with `comm_mode` flipped, so parameter count is identical
(924,267 either way) and any gap is attributable to communication rather than
capacity.

```powershell
# single arm
python scripts\run_comm_ppo.py a16 --comm full --radius 8 --steps 20000000 ^
    --num-envs 64 --rollout 32 --out runs\comm_a16_full

# the gate: both arms, three seeds, with a pass/fail verdict
python scripts\phase5_ablation.py a16 --steps 20000000 --seeds 0 1 2 --num-envs 64
```

### Decision rule

Communication passes if it **lowers collisions per step** and **does not lose
agents-on-goal**. Success rate is reported but not decisive — it is hostage to
the single worst agent, which is exactly how the Phase 2 intent oracle looked
like a regression while reducing collisions 33x.

### Extra signals to watch

| signal | meaning |
|---|---|
| `tentative_entropy` | entropy of the pass-1 intent head. Max is ln(5) = 1.61. If this collapses toward 0, every agent broadcasts the same intent and messages carry no information — the comm benefit will vanish. |
| `peers_in_range` | mean audible peers per agent. Tune `--radius` so this sits around 3–6; near 0 means nobody can talk, near N means the range limit is doing nothing. |

## Revised regime from Phase 5 onward

Peer goals are hidden (`observe_peer_goals=False`) and density rises to 0.25.
Datasets `a16h` / `a32h` / `a64h` are generated with that setting baked in —
the BC trainer reads it from `meta.json`, so the environment it evaluates in
matches the data it trained on. Mixing a visible-goals checkpoint with a
hidden-goals environment silently degrades results; the metadata prevents it.

```powershell
# 1. BC on the COMMUNICATION architecture. Use run_bc_comm.py, not run_bc.py:
#    run_bc.py trains a different policy head, and warm-starting PPO from it
#    silently transfers only the encoder. See docs/FINDINGS-phase5b.md.
python scripts\run_bc_comm.py --demos data\demos\a16h --comm none ^
    --epochs 30 --batch-size 16 --patience 4 --out runs\bccomm_a16h

# 2. The Phase 5 gate.
python scripts\phase5_ablation.py a16 --warm-start runs\bccomm_a16h\best.pt ^
    --steps 20000000 --seeds 0 1 2 --num-envs 64
```

`--batch-size` for `run_bc_comm.py` counts **episodes**, not agents; each
carries all 16 agents, so 16 episodes is ~4,096 agent-samples per step.

`--comm none` for the BC stage is deliberate: it gives both ablation arms an
identical, communication-agnostic starting point, so any later difference is
attributable to what RL learned rather than to a different initialisation.

### Read the BC log carefully

Use `best.pt`, never `last.pt`. Validation loss keeps improving long after the
policy has got worse in the environment — measured at 52 points of success
lost between epoch 3 and epoch 19 while val_loss fell every epoch. The trainer
prints which epoch won and flags rows where val_loss improved but the
environment did not.

### The gate can now return INCONCLUSIVE

If either arm finishes below 50% agents-on-goal, the script refuses to compare
them and exits with status 2. A frozen policy scores zero collisions, so
"fewer collisions" between two failed runs is meaningless — an earlier version
reported PASS on exactly that.


## Warm starts verify themselves now

PPO prints per-module transfer coverage and **aborts** if `encoder`, `gru` or
`policy` is incomplete. It then evaluates the loaded model and compares against
the performance the checkpoint recorded for itself:

```
    policy         4/4   tensors
  warm-start check: agents-on-goal 0.849 (checkpoint recorded 0.823)
```

If those two numbers diverge by more than 0.25 the run aborts. Do not disable
this. A silently partial load cost six GPU-hours and produced six runs of
noise that looked like a real negative result.

## Keeping broadcast intent informative

`--tentative-coef` (default 0.3) supervises the pass-1 intent head during PPO.
Without it, `tentative_entropy` climbs from ~0.24 to ~1.21 (max ln 5 = 1.609)
within ~200 iterations: the messages decay into noise, and any communication
ablation run that way is uninterpretable.

**Watch `tentative_entropy` every run.** If it passes ~0.8, raise the
coefficient to 0.5 or 1.0 and re-run. `tentative_loss` is logged alongside it;
it should fall, not rise.

## Ablations report significance, not direction

`scripts/phase5_ablation.py` runs Welch's t-test on the collision difference
and requires p < 0.05. When it falls short it prints Cohen's d and the number
of seeds needed for 80% power, so the next run is already specified.

The t-distribution is implemented directly in `eval/stats.py`, not approximated.
At 3–8 seeds the degrees of freedom are often below 5, where a normal
approximation reports p = 0.038 for what is really p = 0.159.

## Phase 6 — budgeted scheduling

```powershell
# Quick check that the learned arm behaves before committing hours
python scripts\phase6_ablation.py a16 --warm-start runs\bccomm_a16h\best.pt ^
    --arms none full random learned --seeds 0 1 --steps 5000000 --num-envs 64 ^
    --out runs\phase6_smoke

# The gate: five arms, five seeds
python scripts\phase6_ablation.py a16 --warm-start runs\bccomm_a16h\best.pt ^
    --budget 0.25 --steps 20000000 --seeds 0 1 2 3 4 --num-envs 64
```

25 runs at roughly 10 minutes each, so about four hours unattended. The
`learned` arm is slower than the others: computing counterfactual value of
information costs an extra forward pass, subsampled to every second minibatch
(`--voi-coef 0`, or raising `voi_every`, turns it off).

### The metric that tells you whether the scheduler works

`selection_advantage` = share of total value-of-information captured by the
selected senders, divided by the comm rate.

| value | meaning |
|---|---|
| **1.0** | chance — selecting agents at random captures value in proportion to how many speak |
| **> 1.0** | the scheduler is choosing agents whose messages matter |
| **< 1.0** | actively worse than random; something is wrong |

Watch it alongside `comm_rate` (should settle near `--budget`) and `lambda`
(the dual variable; rises while over budget, returns to zero once satisfied —
zero at the end is correct, not a failure).

### Arms

| arm | what it isolates |
|---|---|
| `none` / `full` | the Phase 5 trade-off endpoints |
| `random` | **the baseline that matters**: budget-matched coin flips. Beating `none` only shows messages help; beating `random` shows the *scheduling* helps |
| `distance` | a hand-designed heuristic — speak when peers are near. A learned scheduler that cannot beat this is not earning its complexity |
| `learned` | Gumbel gating under a Lagrangian budget, trained on counterfactual VoI |

### Allocation

`--allocation rate` (default) uses per-agent gates with the budget enforced on
average, so agents can all speak in a crisis and stay silent in open space.
`--allocation topk` sends exactly k every step — stricter, but a crowded
doorway and an empty corridor get the same bandwidth. Worth reporting both.

## Density matters, and the regime now comes from the checkpoint

Communication measurably helps at **d=0.20** (collisions −25%, p<0.001, 8 seeds)
and not at **d=0.25** (no effect, and `voi_mean` ~9e-4 confirms messages barely
change the policy). Both datasets exist, matched:

| dataset | density | regime |
|---|---|---|
| `a16h20` | 0.20 | communication has value |
| `a16h` | 0.25 | communication has no value |

Ablation scripts read `n_agents`, density, `max_steps` and `observe_peer_goals`
from the warm-start checkpoint rather than a hard-coded preset, and a mismatch
between the training environment and the checkpoint **raises**. Two scripts
once carried different densities for the same preset name, which made two
different experiments look like a failed replication.

```powershell
# Phase 6 in the regime where communication actually carries information
python scripts\run_bc_comm.py --demos data\demos\a16h20 --comm none ^
    --epochs 30 --batch-size 16 --patience 4 --out runs\bccomm_a16h20

python scripts\phase6_ablation.py a16 --warm-start runs\bccomm_a16h20\best.pt ^
    --budget 0.25 --steps 20000000 --seeds 0 1 2 3 4 --num-envs 64 ^
    --out runs\phase6_d20
```

### Read `voi_mean` before anything else

It is the KL between the policy with messages and the policy in silence.

| value | meaning |
|---|---|
| ~1e-3 | messages barely matter; no scheduler can help, and a failed gate says nothing about scheduling |
| materially higher | there is information to allocate, and `selection_advantage` becomes the question |

## Phase 6b — probe the premise before buying another ablation

Phase 6 failed at both densities with `voi_mean` ~ 7e-4, and re-aggregating
the old runs over 10 evaluations instead of 1 dissolved every arm difference
(`FINDINGS-phase6b.md`). BC already reaches 98.4% action accuracy with
communication off, so there is almost nothing left for a message to fix.

Measure that ceiling directly — minutes, not GPU-hours. BC epochs take 5-12s.

```powershell
# 1. A regime with less local information (the 9x9 window is the ceiling).
python scripts\make_demos.py a16h20 --fov 5 --out data\demos\a16h20_fov5

# 2. Message pathway off vs on, on both regimes.
python scripts\comm_headroom.py --demos data\demos\a16h20 data\demos\a16h20_fov5
```

Read the `closed` column: the share of BC's residual error that switching
the message pathway on removes. Below 5% means intent does not predict the
expert in that regime and no scheduler will change it. Above 20% is worth a
PPO ablation.

If `fov=5` does not open headroom, the next lever is the cost-to-go channel:
it hands every agent a globally-computed shortest-path field, which is most
of why the local decision is already trivial.

### Re-reading the runs you already paid for

No retraining needed — the histories hold every evaluation:

```powershell
python scripts\phase6_ablation.py a16 --warm-start runs\bccomm_a16h20\best.pt ^
    --eval-window 10 ...      # new default; --eval-window 1 restores the old behaviour
```
