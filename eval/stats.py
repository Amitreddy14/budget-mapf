"""Statistics for ablation verdicts.

A direction check ("collisions went down") is not a result. With three seeds
and a noisy arm, a 22% improvement can carry p ~ 0.17 — the kind of number
that looks convincing in a table and evaporates on replication. Every
comparison this project reports goes through a test.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence


def _betacf(a: float, b: float, x: float, iters: int = 200) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, iters + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-12:
            break
    return h


def regularised_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_sf(t: float, df: float) -> float:
    """Upper-tail probability of Student's t.

    Implemented directly rather than falling back to a normal approximation.
    At the sample sizes used here (3-8 seeds, df often below 5) the normal
    approximation is badly anti-conservative: for t=2.08, df=2.3 it reports
    p=0.038 where the true value is ~0.17 -- the difference between claiming a
    result and not having one.
    """
    if not math.isfinite(t):
        return 0.0
    if df <= 0:
        return float("nan")
    return 0.5 * regularised_incomplete_beta(df / 2.0, 0.5, df / (df + t * t))


def welch_ttest(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Welch's unequal-variance t-test, plus effect size and required n.

    Returns t, Welch-Satterthwaite df, two-tailed p, Cohen's d, and the seeds
    per arm needed for 80% power at the observed effect size.
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"t": float("nan"), "df": float("nan"), "p": float("nan"),
                "cohens_d": float("nan"), "n_for_80pc_power": float("nan")}

    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)

    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return {"t": float("inf"), "df": float("nan"), "p": 0.0,
                "cohens_d": float("inf"), "n_for_80pc_power": 2.0}
    t = (ma - mb) / se
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))

    p = student_t_sf(abs(t), df) * 2.0

    pooled = math.sqrt((va + vb) / 2)
    d = (ma - mb) / pooled if pooled > 0 else float("inf")
    n_need = 2 * ((1.96 + 0.84) / d) ** 2 if d != 0 else float("inf")

    return {"t": t, "df": df, "p": p, "cohens_d": d,
            "n_for_80pc_power": math.ceil(n_need) if math.isfinite(n_need) else n_need}


def summarise(name: str, a: Sequence[float], b: Sequence[float],
              lower_is_better: bool = False) -> str:
    """One formatted line comparing two arms."""
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    r = welch_ttest(a, b)
    delta = mb - ma
    better = (delta < 0) if lower_is_better else (delta > 0)
    sig = "significant" if r["p"] < 0.05 else f"NOT significant (n={len(a)})"
    return (f"{name:<16}{ma:>9.4f}{mb:>12.4f}{delta:>+11.4f}"
            f"{r['p']:>9.3f}  {'better' if better else 'worse ':<7} {sig}")


def eval_window(history, window=10):
    """Average the last `window` evaluation rows of a training history.

    A single evaluation of ~128 episodes carries roughly +/-4pp of binomial
    noise on success rate, and the measured swing between consecutive
    evaluations of one converged run is larger still (0.20-0.30 absolute).
    Reporting only the final evaluation therefore reports one draw from that
    distribution, and the between-seed spread it produces is dominated by
    measurement error rather than by the thing being compared.

    Phase 6 was decided on single-eval numbers. Re-aggregating the same runs
    over the last 10 evaluations moved `learned` vs `random` on success from
    +0.106 to -0.002: the entire apparent effect was evaluation noise.

    Averaging reaches back into slightly earlier (less converged) checkpoints,
    which biases every arm the same way; that is a far smaller distortion than
    the variance it removes.
    """
    rows = [r for r in history if r.get("eval_success") is not None]
    if not rows:
        raise ValueError("history contains no evaluation rows")

    # Iteration 0 is evaluated BEFORE any gradient step, so it measures the
    # warm-start checkpoint, not the trained policy -- and it measures it
    # under whatever comm mode the arm uses, which differs per arm. At 20M
    # steps there are 26 evaluations and one stale row is harmless noise. At
    # 2M steps there are 4, and that row carries a quarter of the weight:
    # it reported collisions of 1.96 against 0.26 for the trained policy,
    # inflating the collision column roughly 2.5x and handing the `none` arm
    # a success rate it earned before training began.
    if len(rows) > 1:
        rows = [r for r in rows if r.get("iter", 1) != 0]

    if len(rows) < window:
        print(f"  note: eval_window={window} but only {len(rows)} post-training "
              f"evaluations exist; averaging all of them. Shorter runs need a "
              f"smaller --eval-every to average meaningfully.")
    tail = rows[-window:]
    keys = [k for k in tail[-1] if k.startswith("eval_")]
    out = {k: float(sum(r[k] for r in tail) / len(tail)) for k in keys
           if all(k in r and r[k] is not None for r in tail)}
    # carry the non-eval diagnostics from the final row unchanged
    for k in ("comm_rate", "voi_mean", "selection_advantage", "lambda",
              "entropy", "tentative_entropy", "peers_in_range", "iter"):
        if k in tail[-1] and tail[-1][k] is not None:
            out[k] = float(sum(r[k] for r in tail if r.get(k) is not None)
                           / max(1, sum(1 for r in tail if r.get(k) is not None)))
    out["eval_window"] = len(tail)
    return out
