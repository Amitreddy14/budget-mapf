# Phase 5 (second attempt) — a silent partial warm start

Date: 2026-10-12.

## What the validity gate caught

The ablation returned **INCONCLUSIVE**, exactly as designed:

```
eval_on_goal     none 0.007 ±0.008    full 0.001 ±0.001
eval_coll        none 0.328 ±0.200    full 0.114 ±0.128
[INVALID] both arms must reach agents-on-goal >= 0.50
GATE: INCONCLUSIVE -- training did not succeed in both arms
```

Note that on the naive criteria this would have reported **PASS** again:
communication "reduced" collisions by 0.214. Both arms were simply too broken
to move. The floor earned its place on its first real outing.

## The bug

Behaviour cloning worked well — best epoch 4, **82.0% success / 93.7%
agents-on-goal** with peer goals hidden. But every PPO run reported
`eval_on_goal = 0.000` at **iteration 0, before a single policy update**, with
`entropy = 1.425` (ln 5 = 1.609, i.e. near-uniform random). The Phase 4 run
that worked started at `entropy = 0.184`.

`scripts/run_bc.py` trains `ActorCritic`, whose policy head is one `nn.Linear`
named `policy.weight`. `CommActorCritic`'s head is an `nn.Sequential`, named
`policy.0.weight` / `policy.2.weight`. The names do not match, so:

| module | transferred |
|---|---|
| encoder | 12/12 |
| gru | 4/4 |
| **policy** | **0/4** |
| tentative | 0/2 |
| comm | 0/14 |

PPO began from the cloned **encoder** with a **randomly initialised policy
head** — a uniformly random policy that happens to have good features. That is
indistinguishable from training from scratch, which we had already measured at
~0.2% agents-on-goal. Six 20M-step runs, roughly six GPU-hours, produced
nothing.

`load_partial` printed `warm start: 19 tensors` and said nothing else.

## Fixes

**1. Root cause — clone the architecture you intend to fine-tune.**
`train/bc_comm.py` trains `CommActorCritic` directly, so every tensor
transfers. It also supervises the pass-1 `tentative` head on the expert
action, which gives "intent" a concrete meaning instead of leaving it an
arbitrary embedding. Verified on CPU: 82.3% agents-on-goal after 3 epochs on
one shard.

**2. Loud loader.** `load_partial` now reports per-module coverage and
**raises** when any of `encoder`, `gru`, `policy` is incomplete, naming the
right script to use instead.

**3. Pre-flight check.** Before a single gradient step, PPO evaluates the
warm-started model and compares against the performance the checkpoint
recorded for itself. A gap over 0.25 aborts the run. This is the general
guard: it catches architecture mismatch, observability mismatch, and anything
else that makes a warm start not behave like its checkpoint — in ~30 seconds
rather than six hours.

```
warm start from runs/bccomm_smoke/best.pt
    comm          14/14  tensors
    encoder       12/12  tensors
    gru            4/4   tensors
    policy         4/4   tensors
    tentative      2/2   tensors
    value          4/4   tensors
  warm-start check: agents-on-goal 0.849 (checkpoint recorded 0.823)
```

**4. Checkpoints carry their observability.** `observe_peer_goals` is saved
with the weights, and PPO adopts it, overriding its own config with a printed
notice. A policy cloned without peer goals but evaluated in an environment
that shows them degrades silently, and nothing in the logs would explain why.

**5. Regression tests** (`tests/test_warmstart.py`): loading an `ActorCritic`
checkpoint into a `CommActorCritic` must raise; a matching checkpoint must
transfer every tensor and every value; a `comm=none` checkpoint must load into
all three comm modes, which is what lets both ablation arms start identical.

## Lesson

Two guards were worth more than any amount of care:

1. **A validity floor on comparisons.** Without it this run reports PASS.
2. **Verify a warm start reproduces its checkpoint before training on it.**
   Checkpoint loading fails partially and silently by design — `load_state_dict`
   with mismatched names is a no-op per tensor, not an error.

Both now run automatically.
