"""Does correct addressing help, once seed noise is averaged out?

`scripts/comm_headroom.py` trains each arm once. On the a12 regimes that gave
a closed-loop gap between `full` and `shuffled` on mazes of about 5pp
(on-goal 0.559 vs 0.512 over the last three epochs) while the offline
next-action error showed almost nothing (+1.4%). One seed per arm cannot
separate those. This repeats the probe across seeds and tests the gap.

DECISION RULE, fixed before any result exists
---------------------------------------------
Primary comparison: `full` minus `shuffled`, closed-loop agents-on-goal, mean
of the last --window epochs per run, Welch's t across seeds, two-sided,
alpha = 0.05, on the demo set named first on the command line.

The offline error `1 - best_acc` must point the same way. A closed-loop gap
that the offline metric does not corroborate is reported as unconfirmed, not
as a finding.

Every other row (shuffled vs none, full vs none, the other regimes) is
exploratory. With nine comparisons printed, roughly one will cross p < 0.05
by chance; none of them is a result on its own.

History here says why this is written down first: Phase 5's -25% collisions
shrank to -4% under the same data, and a 3-seed d = -1.76 became d = +1.57
at 5 seeds.

RESUMABLE
---------
A run is complete when its history.json has --epochs rows. Re-running skips
complete runs, so this can be started in chunks, interrupted, and restarted.

    python scripts/headroom_seeds.py --demos data/demos/a12maze --seeds 0 1 2 3 4
    python scripts/headroom_seeds.py --demos data/demos/a12maze --report-only
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.stats import welch_ttest

ARMS = ("none", "shuffled", "full")


def run_dir(out, tag, mode, seed):
    return os.path.join(out, f"{tag}_{mode}_s{seed}")


def is_complete(path, epochs):
    p = os.path.join(path, "history.json")
    if not os.path.exists(p):
        return False
    try:
        with open(p) as fh:
            return len(json.load(fh)) >= epochs
    except (OSError, ValueError):
        return False


def window_mean(history, key, window):
    """Mean of `key` over the last `window` epochs that report it.

    BC histories are per-epoch with no iteration-0 row, so unlike
    eval.stats.eval_window nothing is excluded: epoch 1 is already trained.
    """
    vals = [r[key] for r in history if r.get(key) is not None]
    if not vals:
        raise ValueError(f"no rows report {key!r}")
    tail = vals[-window:]
    return sum(tail) / len(tail)


def summarise_run(history, window):
    accs = [r["acc"] for r in history if r.get("acc") is not None]
    return {
        "on_goal": window_mean(history, "env_on_goal", window),
        "success": window_mean(history, "env_success", window),
        "coll": window_mean(history, "env_coll", window),
        "err": 1.0 - max(accs),          # best validation accuracy, as comm_headroom.py
        "epochs": len(history),
    }


def collect(out, tag, seeds, window):
    """{arm: [per-seed summaries]} from whatever runs are complete."""
    got = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for s in seeds:
            p = os.path.join(run_dir(out, tag, arm, s), "history.json")
            if not os.path.exists(p):
                continue
            with open(p) as fh:
                got[arm].append(summarise_run(json.load(fh), window))
    return got


def _mean_sd(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else float("nan")
    return m, sd


def compare(got, a_arm, b_arm, key):
    """b minus a, with Welch's test across seeds."""
    a = [r[key] for r in got[a_arm]]
    b = [r[key] for r in got[b_arm]]
    if len(a) < 2 or len(b) < 2:
        return None
    r = welch_ttest(b, a)                       # positive t == b higher
    return {"delta": sum(b) / len(b) - sum(a) / len(a), "p": r["p"],
            "d": r["cohens_d"], "n_a": len(a), "n_b": len(b),
            "n_for_80pc_power": r["n_for_80pc_power"]}


def verdict(got, min_seeds=5):
    """Apply the pre-registered rule to full vs shuffled."""
    on = compare(got, "shuffled", "full", "on_goal")
    er = compare(got, "shuffled", "full", "err")
    if on is None or er is None:
        return "INCOMPLETE -- need at least 2 seeds per arm to test anything"
    if min(on["n_a"], on["n_b"]) < min_seeds:
        return (f"UNDERPOWERED -- {min(on['n_a'], on['n_b'])} seeds per arm, rule "
                f"needs {min_seeds}. Do not read the p-value as a result.")
    if on["p"] >= 0.05:
        return (f"NULL -- full vs shuffled on-goal {on['delta']:+.4f}, "
                f"p={on['p']:.3f}. Correct addressing is not detectably better.")
    if on["delta"] > 0 and er["delta"] < 0:
        return (f"SIGNAL -- on-goal {on['delta']:+.4f} (p={on['p']:.3f}) and "
                f"offline error {er['delta']:+.4f} agree in sign.")
    if on["delta"] > 0:
        return (f"UNCONFIRMED -- on-goal {on['delta']:+.4f} (p={on['p']:.3f}) but "
                f"offline error moves {er['delta']:+.4f}, the wrong way or not at all. "
                f"The rule requires both.")
    return (f"REVERSED -- shuffled beats full on-goal by {-on['delta']:.4f} "
            f"(p={on['p']:.3f}).")


def report(got, tag, window):
    print(f"\n=== {tag}: last {window} epochs per run, Welch across seeds ===\n")
    print(f"{'arm':<10}{'n':>3}{'on-goal':>16}{'success':>16}{'coll':>14}{'offline err':>16}")
    print("-" * 75)
    for arm in ARMS:
        rs = got[arm]
        if not rs:
            print(f"{arm:<10}{0:>3}   (no completed runs)")
            continue
        cells = []
        for k in ("on_goal", "success", "coll", "err"):
            m, sd = _mean_sd([r[k] for r in rs])
            cells.append(f"{m:.4f}+-{sd:.4f}" if not math.isnan(sd) else f"{m:.4f}")
        print(f"{arm:<10}{len(rs):>3}{cells[0]:>16}{cells[1]:>16}{cells[2]:>14}{cells[3]:>16}")

    print(f"\n{'comparison (b - a)':<24}{'metric':<10}{'delta':>9}{'p':>8}{'d':>7}{'n/arm':>7}")
    print("-" * 65)
    primary = ("shuffled", "full")
    for a, b in (("shuffled", "full"), ("none", "shuffled"), ("none", "full")):
        for key in ("on_goal", "err"):
            c = compare(got, a, b, key)
            if c is None:
                continue
            star = "  <- PRIMARY" if (a, b) == primary and key == "on_goal" else ""
            print(f"{b + ' - ' + a:<24}{key:<10}{c['delta']:>+9.4f}{c['p']:>8.3f}"
                  f"{c['d']:>+7.2f}{c['n_a']:>4}/{c['n_b']:<2}{star}")
    print(f"\nVERDICT (pre-registered rule): {verdict(got)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--window", type=int, default=4,
                    help="average the last N epochs per run; single epochs of "
                         "one converged BC run swing by 0.2-0.3 on-goal")
    ap.add_argument("--eval-eps", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--radius", type=float, default=8.0)
    ap.add_argument("--out", default="runs/headroom_seeds")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    if not a.report_only:
        import torch
        from train.bc_comm import CommBCConfig, train_comm_bc
        device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    for root in a.demos:
        tag = os.path.basename(root.rstrip("/\\"))
        if not a.report_only:
            todo = [(s, m) for s in a.seeds for m in ARMS
                    if not is_complete(run_dir(a.out, tag, m, s), a.epochs)]
            print(f"\n{tag}: {len(todo)} of {len(a.seeds) * len(ARMS)} runs to do "
                  f"({len(a.seeds) * len(ARMS) - len(todo)} already complete)")
            for i, (s, m) in enumerate(todo, 1):
                print(f"\n=== [{i}/{len(todo)}] {tag} | comm={m} | seed={s} ===", flush=True)
                # patience > epochs disables early stopping so every arm gets
                # the same budget, as in comm_headroom.py.
                cfg = CommBCConfig(
                    demo_root=root, comm_mode=m, comm_radius=a.radius,
                    epochs=a.epochs, batch_size=a.batch_size,
                    patience=a.epochs + 1, eval_eps_per_env=a.eval_eps,
                    seed=s, out_dir=run_dir(a.out, tag, m, s), device=device)
                train_comm_bc(cfg)

        got = collect(a.out, tag, a.seeds, a.window)
        report(got, tag, a.window)
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, f"{tag}_summary.json"), "w") as fh:
            json.dump(got, fh, indent=2)


if __name__ == "__main__":
    main()
