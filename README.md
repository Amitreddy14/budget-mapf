# BUDGET-MAPF

Does learned inter-agent communication help decentralised Multi-Agent Path
Finding?

**Measured answer: no.** Across nine independent measurements — five
behaviour-cloning regimes and reinforcement learning at 16, 32 and 64 agents —
a learned intent-communication channel is indistinguishable from one carrying
*deliberately scrambled* messages. The best policy in this repository does not
communicate at all.

This is a negative result, and the repository is organised around making it
checkable rather than around the architecture that produced it.

## The control that carries the claim

Most communication studies compare "with messages" against "no messages". That
comparison confounds two things: the information the messages carry, and the
extra parameters, depth and gradient paths the communication module adds.

The **shuffled** arm separates them. It runs the identical module — same
attention, same parameter count, same compute — but permutes *which agent's
intent lands in each attention slot*. The channel still exists and still
carries real intents; they are simply addressed to the wrong agents.

```
full      agent i reads  intent(1) .. intent(N)        correct addressing
shuffled  agent i reads  intent(p(1)) .. intent(p(N))  same tensors, permuted
none      agent i reads  zeros                         no channel
```

`full − shuffled` isolates the value of the information. `full − none`
measures the module. The gap between the two is the finding.

| comparison | 16 agents | 32 agents | 64 agents |
|---|---|---|---|
| full vs shuffled | p = 0.694 | p = 0.817 | p = 0.689 |

Across five behaviour-cloning regimes (random 9x9 and 5x5 fields, coarse
heuristic, no heuristic, rooms), the mean effect of correct addressing is
**−0.02%**, with no consistent sign.

## Headline numbers

| | value |
|---|---|
| Best policy — no-comm MAPPO, 16 agents, 4M steps | **0.7621** success, **0.9752** agents-on-goal |
| `voi_mean` (influence: KL of comm vs silent policy) | ~0.024 nats |
| `voi_value` (whether that influence helps) | ~−0.001 |
| Evaluation action rule: sampled → greedy | **−0.378** on-goal, 100x success collapse |
| QMIX as a second learning rule | fails; see `docs/FINDINGS-phase6b.md` §20 |

Messages measurably change behaviour and that change is worth nothing. The
two quantities are separated deliberately: `policy_shift` measures influence,
`value_of_information` measures whether the shift improves the action taken.
Reporting only the first is how a channel like this looks useful.

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python scripts/check_setup.py
```

## Verify

```bash
python -m pytest tests/ -q -m "not slow"    # 117 tests
python -m pytest tests/ -q -m slow          # trainer smoke runs, CPU
```

The test suite imports torch at module scope by design: on a machine without
torch the suite **errors** rather than silently collecting zero tests.

## Reproduce

See `docs/REPRODUCE.md` — every claim above mapped to the command that
produces it and the number to expect.

## Layout

```
mapf/      environment: maps, BFS cost-to-go fields, vectorised env, scripted policies
experts/   prioritised-planning expert + demonstration generation
models/    CNN+GRU encoder, intent communication, scheduler, QMIX mixer
train/     behaviour cloning, MAPPO, QMIX
eval/      metrics, episode harness, statistics
scripts/   entry points (see docs/REPRODUCE.md)
docs/      findings, in chronological order; phase6b is the current state
```

## Reading the findings

`docs/FINDINGS-phase6b.md` is the live document and supersedes the earlier
phases, several of whose headline numbers it corrects. It is written as a
record of what was measured and what was wrong with earlier measurements,
including:

- §2 why "final evaluation only" reporting invented effects that did not exist
- §3 how Phase 5's headline shrank from −25% to −4% under the same data
- §19 / §19.1 mazes: first read as undemonstrable, then corrected. Most of
  that failure was a planning horizon sized for random grids (192) applied to
  a map with 432-step shortest paths. Prioritised planning solves mazes at 12
  agents and fails at 16
- §20 QMIX: a reward-function freeze attractor, and why a behaviour-cloned
  policy head cannot initialise a Q-function

## Known limitations

- The communication result rests on **one learning rule** (MAPPO). QMIX was
  built and run; `docs/FINDINGS-phase6b.md` §20 is the measured reason it
  cannot carry an ablation here.
- The communication study was run on random grids and rooms. Mazes are
  demonstrable at 12 agents (§19.1) but were not part of the measurements
  above; at 16 agents, where the study sits, no expert solves them.
- Episode truncation is treated as termination (`done = solved | timeout`).
  The env auto-resets inside `step()` and does not expose the pre-reset
  observation, so bootstrapping through the time limit would use the *next*
  episode's state. Affects roughly one transition in 32.
