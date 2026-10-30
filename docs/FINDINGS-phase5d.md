# Phase 5 — supervising intent changes the answer

Date: 2026-10-12. Same setup as `FINDINGS-phase5c.md` (16 agents, 32x32,
d=0.25, peer goals hidden, 20M steps, 3 seeds), with `--tentative-coef 0.3`.

## The fix held

`tentative_entropy` across training, maximum ln(5) = 1.609:

| | start | mid | end |
|---|---|---|---|
| without supervision | 0.240 | 1.025 | **1.208** |
| with `tentative_coef=0.3` | ~0.18 | ~0.10 | **~0.09** |

Messages stayed informative for the whole run instead of decaying into noise
within ~200 iterations.

## And it changed the result

| metric | no comm | full comm | delta |
|---|---|---|---|
| success | 0.750 ±0.026 | 0.740 ±0.019 | −0.010 |
| agents-on-goal | 0.939 ±0.001 | 0.937 ±0.003 | −0.002 |
| **collisions/step** | **0.096 ±0.017** | **0.075 ±0.004** | **−0.021 (22%)** |

Compare the unsupervised run, where the same comparison gave +0.001. The only
difference between the two experiments is whether the broadcast intent was
kept meaningful.

**That is itself a finding**: communication provides no benefit unless
something maintains the informativeness of what is communicated. Message
content is not kept meaningful by the task reward alone.

## But it is not yet significant

Welch's test on the collision difference:

```
t = 2.08   df = 2.3   Cohen's d = 1.70   p = 0.159
seeds per arm needed for 80% power: ~6
```

The effect is large (d = 1.70) but the no-comm arm scatters widely across
seeds (0.081, 0.093, 0.115) while the comm arm is tight (0.072, 0.073, 0.080).
Three seeds cannot separate them. Success and agents-on-goal show no benefit
at all — both marginally negative, both within noise.

**Do not report this as a positive result yet.** Run 8 seeds per arm.

## Two bugs in the verdict machinery, both fixed

**1. The gate checked direction, not significance.** It returned `PASSED` on a
difference with p = 0.159. `scripts/phase5_ablation.py` now runs Welch's test
and requires p < 0.05, and prints the effect size and the seeds needed for 80%
power when it falls short.

**2. The p-value itself was wrong.** The first implementation fell back to a
normal approximation when scipy was absent, reporting **p = 0.038** where the
true Student-t value at df = 2.3 is **0.159**. At the sample sizes this project
uses — 3 to 8 seeds, df often below 5 — the normal approximation is badly
anti-conservative. `eval/stats.py` now implements the t-distribution directly
via the regularised incomplete beta function, validated against published
critical values at df = 1, 2, 3, 5, 10, 30, 100 (`tests/test_stats.py`).

A wrong p-value is worse than no p-value: it converts "we don't know yet" into
a claim.

## Next

1. **8 seeds per arm** at 16 agents to resolve significance.
2. **32 agents** (`a32h` ready) — the last untested lever.

Only then is the framing decision answerable.
