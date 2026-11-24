# Reproducing the results

Every claim in `README.md`, mapped to the command that produces it and the
number to expect. Commands are PowerShell (`^` continues a line); on bash use
`\`. Timings are for an RTX 4060 (8GB) unless stated.

Read `RUNNING.md` first if the environment is not set up.

---

## 0. Verify the build  —  1 minute

```powershell
python scripts\check_setup.py
python -m pytest tests\ -q -m "not slow"
python -m pytest tests\ -q -m slow
```

Expect 117 tests. The suite imports torch at module scope deliberately: on a
machine without torch it must **error**, not silently collect zero tests.

---

## 1. Demonstrations  —  30s to 3 min each

Everything downstream warm-starts from behaviour cloning, so build these
first. `make_demos.py` presets fix agent count, map size and density together;
the ablation scripts then read the environment back out of the checkpoint, so
a preset mismatch raises rather than quietly running a different experiment.

```powershell
python scripts\make_demos.py a16h20                                    # 16 agents, 32x32, d=0.20
python scripts\make_demos.py a16h20 --fov 5  --out data\demos\a16h20_fov5
python scripts\make_demos.py a16h20 --obs-costmap coarse --out data\demos\a16h20_coarse
python scripts\make_demos.py a16h20 --obs-costmap none   --out data\demos\a16h20_nocm
python scripts\make_demos.py a16room                                   # 32x32 rooms, room_size 6
python scripts\make_demos.py a32h20                                    # scale study
python scripts\make_demos.py a64h20                                    # needs ~16GB RAM
```

`a64h20` was killed by the OOM killer in a 3GB container and took 137s on a
16GB machine. Generate it locally.

**Mazes need their own horizon.** The `a12maze` preset carries
`max_steps=768` and 20 restarts; `a16maze` exists but no planner solves it.
A horizon sized for a random grid (192) makes most maze instances unsolvable
for any planner, which is what the original "mazes cannot be demonstrated"
reading actually measured -- see `FINDINGS-phase6b.md` §19.1.

```powershell
python scripts\make_demos.py a12maze        # 12 agents, 41x41 maze
python scripts\make_demos.py a12room        # same agent count, rooms
python scripts\make_demos.py a12h20         # same agent count, random grid
```

Those three hold agent count fixed across map kinds, which is the comparison
the structure study needs.

| agents | prioritised planning on 41x41 mazes |
|---|---|
| 2, 4 | 6/6 |
| 8 | 4/6 |
| 12 | 3/3 (20 restarts) |
| 16 | 0/3 |

```powershell
python scripts\bench_planner.py --kinds maze --agents 2 4 8 --scenarios 6
```

CBS (`experts/cbs.py`) is complete and optimal for sum-of-costs and solves
corridor instances prioritised planning cannot, but it is the weaker option
on mazes: 0.05s at 2 agents, timing out at 4.

---

## 2. The communication ceiling  —  minutes, no GPU-hours

The cheapest form of the whole question. Behaviour cloning with the message
pathway off, on, and shuffled, on each regime.

```powershell
python scripts\comm_headroom.py --demos data\demos\a16h20 data\demos\a16h20_fov5 ^
    data\demos\a16h20_coarse data\demos\a16h20_nocm data\demos\a16room
```

| regime | err (comm off) | closed | vs shuf |
|---|---|---|---|
| random 9x9 | 0.0136 | 19.9% | +2.3% |
| random 5x5 | 0.0155 | 26.4% | −0.5% |
| coarse heuristic field | 0.0279 | 1.7% | +0.6% |
| no heuristic field | 0.0384 | 2.9% | −0.2% |
| rooms | 0.0155 | 23.2% | −2.3% |

`closed` is the share of residual error the message pathway removes; `vs shuf`
is how much of that survives scrambling the addressing. **Mean `vs shuf`
= −0.02%, no consistent sign.** The `closed` column is capacity, not
information — which is exactly what the shuffled control exists to show.

---

## 3. Reinforcement learning  —  the headline  —  hours per arm

```powershell
# warm start
python scripts\run_bc_comm.py --demos data\demos\a16h20 --comm none ^
    --epochs 30 --batch-size 16 --patience 4 --out runs\bccomm_a16h20

# three arms, five seeds
python scripts\phase6_ablation.py a16 --warm-start runs\bccomm_a16h20\best.pt ^
    --arms none full shuffled --budget 0.25 --steps 20000000 ^
    --seeds 0 1 2 3 4 --num-envs 64 --out runs\phase6_d20
```

| | success | collisions | on-goal |
|---|---|---|---|
| **none** | **0.7621** (sd 0.011) | 0.1760 | 0.9752 |

| comparison | 16 agents | 32 agents | 64 agents |
|---|---|---|---|
| full vs shuffled | p = 0.694 | p = 0.817 | p = 0.689 |

Repeat with `a32`/`a64` presets and the matching checkpoints for the scale
rows.

### Two measurement traps, both of which fired here

**Report a window, never the final evaluation.** Consecutive evaluations of
one converged run differ by 0.20–0.30 on success. Reporting the last one is
one draw from that distribution per seed, and it invented effects that did not
exist — Phase 5's headline shrank from −25% to −4% under the same data.
`--eval-window 10` is the default; `--eval-window 1` restores the old
behaviour if you want to see it.

**Exclude iteration 0.** It evaluates the warm start under each arm's comm
mode before any training. At 2M steps only four evaluations exist, so that one
row inflated collisions 2.6x (1.96 vs 0.26) and favoured `none`. `eval_window`
drops it automatically.

Re-aggregation needs no retraining: the histories hold every evaluation.

---

## 4. Influence is not value  —  logged during any communicating run

Two quantities, deliberately separate:

```
policy_shift         = KL(pi_comm || pi_silent)                  influence only
value_of_information = A(s,a) * [log pi_comm(a|s) - log pi_silent(a|s)]
```

Expect `voi_mean` ~0.024 nats with `voi_value` ~−0.001: messages measurably
change behaviour, and the change is worth nothing. Reporting only the first is
how a channel like this is made to look useful.

Both are logged every `--voi-every` iterations. A structural test cannot catch
a misplaced logging branch — this one was wrong twice, once never firing and
once firing only during value warmup — so `tests/test_comm.py` carries a slow
integration test that trains and asserts the column appears.

---

## 5. Evaluation protocol  —  seconds

```powershell
python scripts\probe_warmstart.py runs\bccomm_a16h20\best.pt ^
    --epsilons 0,0.05,0.1,0.2
```

| comm | actions | on-goal | success | coll/step |
|---|---|---|---|---|
| none | sampled | 0.9556 | 0.7812 | 2.09 |
| none | **greedy** | **0.5771** | **0.0078** | 6.22 |
| -- | random | 0.0010 | 0.0000 | 3.04 |

**The action rule is worth −0.378 on-goal and collapses success 100-fold.** A
deterministic policy cannot break the symmetry between two agents contesting a
cell, so it livelocks to the step cap. Any cross-method comparison must hold
the rule fixed or state which rule each side used — for a Q-network argmax
*is* the policy, so there is no sampled variant and the honest comparison
reports the policy-gradient side under both.

The same script gives a random-policy floor, without which a low number cannot
be read at all.

---

## 6. QMIX  —  reproduces a negative result

```powershell
python scripts\run_qmix.py --comm none --agents 4 --height 16 --width 16 ^
    --density 0.15 --max-steps 96 --steps 1000000 --out runs\qmix_n4_scratch

python scripts\run_qmix.py --comm none --agents 4 --height 16 --width 16 ^
    --density 0.15 --max-steps 96 --steps 1000000 --shaping-coef 10 ^
    --out runs\qmix_n4_shaped
```

| run | final on-goal | coll/step | note |
|---|---|---|---|
| unshaped | 0.0000 | 0.0000 | below the random floor of 0.0046 |
| shaped | 0.0312 | ~0.00 | still climbing at cutoff |

The unshaped run converges to standing still, which is a correct solution to
the MDP it can see: off-goal, `STAY` costs −0.5 while `MOVE` costs
−0.3 − 2.0·p(blocked), so `STAY` wins once p(blocked) > 10%. Shaping
(policy-invariant, Ng et al. 1999) escapes that attractor but leaves a learner
far too weak to carry an ablation.

**Do not run the four-arm QMIX ablation.** Four arms at 0.03 on-goal would
agree with each other, and that agreement would be manufactured — the exact
failure the shuffled control exists to prevent. `FINDINGS-phase6b.md` §20 has
the full account.
