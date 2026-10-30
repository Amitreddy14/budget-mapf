"""Phase 6 gate: can a BUDGETED scheduler beat both Phase 5 arms at once?

Phase 5 measured a trade-off at 16 agents, 8 seeds:

    no comm     collisions 0.1009   success 0.7402
    full comm   collisions 0.0760   success 0.7012   (p<0.001, d=2.51)

Full communication removes a quarter of the collisions and costs ~3.9pp of
task success, because agents told everyone's intentions become deferential
everywhere rather than where it matters. The hypothesis under test:

    SELECTIVE communication keeps the collision benefit without the
    throughput cost.

Target:  collisions <= 0.080  AND  success >= 0.740  with B < N senders.

Arms (all identical architecture, identical parameter count, one shared
warm-start checkpoint; only the scheduling policy differs):

  none      silent
  full      everyone transmits -- an unbudgeted upper bound on bandwidth
  random    budget-matched coin flips -- isolates "learned anything at all"
  distance  budget-matched hand heuristic -- speak when peers are close
  learned   Gumbel gating under a Lagrangian budget, trained on counterfactual
            value of information

`random` is the baseline that matters most. Beating `none` only shows messages
help; beating `random` at the same budget shows the *scheduling* is what helps.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from eval.stats import summarise, welch_ttest, eval_window
from train.ppo_comm import CommPPOConfig, train_comm_ppo

# Fallback only. preset_from_checkpoint() overrides every field from the
# warm-start checkpoint and prints which source it used; these values apply
# solely to a checkpoint that carries no map_spec. A stale entry here is how
# phase5 and phase6 once ran the same preset name at different densities.
PRESETS = {
    "a16": dict(n_agents=16, height=32, width=32, density=0.25),
    "a32": dict(n_agents=32, height=40, width=40, density=0.25),
    "a64": dict(n_agents=64, height=64, width=64, density=0.25),
}
# "shuffled" is the capacity control: every agent transmits, but which
# agent's intent lands in each attention slot is permuted. Same graph, same
# module, same parameters in use -- only the sender/message correspondence is
# destroyed. In behaviour cloning it matched "full" to three decimals, which
# is what proved the BC gain was capacity rather than information. It had
# never been run in RL; the comparison that matters is full vs shuffled, not
# full vs none.
ARMS = ("none", "full", "shuffled", "random", "distance", "learned")


def preset_from_checkpoint(path, fallback):
    """Read the training regime from the warm-start checkpoint.

    Hard-coding a preset in each script is how phase5 and phase6 ended up
    running the same named config at different densities. The checkpoint
    records what it was trained on; trust that instead.
    """
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=False)
    spec, n = blob.get("map_spec"), blob.get("n_agents")
    del blob
    if spec and n:
        # Carry the whole spec, not just its dimensions. Keeping only
        # height/width/density silently rebuilt every map as "random", so a
        # checkpoint trained on rooms or a maze would have been fine-tuned on
        # open grids with nothing in the logs to say so.
        return dict(n_agents=n, spec=dict(spec)), True
    return dict(n_agents=fallback["n_agents"],
                spec={"kind": "random", "height": fallback["height"],
                      "width": fallback["width"],
                      "density": fallback["density"]}), False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("preset", choices=list(PRESETS), nargs="?", default="a16")
    ap.add_argument("--warm-start", required=True)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--budget", type=float, default=0.25)
    ap.add_argument("--allocation", choices=["rate", "topk"], default="rate")
    ap.add_argument("--voi-coef", type=float, default=0.5)
    ap.add_argument("--tentative-coef", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=20_000_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--out", default="runs/phase6")
    ap.add_argument("--eval-every", type=int, default=25,
                    help="iterations between evaluations. --eval-window can "
                         "only average what exists: a short run at the default "
                         "25 produces 4 evaluations, too few to average.")
    ap.add_argument("--eval-window", type=int, default=10,
                    help="average the last K evaluations per seed "
                         "(1 = the old single-eval behaviour)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    p, from_ckpt = preset_from_checkpoint(a.warm_start, PRESETS[a.preset])
    print(f"regime: {p}  (" + ("from checkpoint" if from_ckpt
          else "preset fallback -- checkpoint carried no map_spec") + ")")
    results = {}
    for arm in a.arms:
        per_seed = []
        for seed in a.seeds:
            print(f"\n=== {arm} seed={seed} ===")
            cfg = CommPPOConfig(
                n_agents=p["n_agents"],
                map_spec=p["spec"],
                comm_mode=arm, warm_start=a.warm_start,
                budget_frac=a.budget, allocation=a.allocation,
                voi_coef=a.voi_coef if arm == "learned" else 0.0,
                tentative_coef=a.tentative_coef,
                num_envs=a.num_envs, total_agent_steps=a.steps,
                eval_every=a.eval_every,
                seed=seed, device=a.device,
                out_dir=os.path.join(a.out, f"{a.preset}_{arm}_s{seed}"),
            )
            out = train_comm_ppo(cfg)
            final = eval_window(out["history"], a.eval_window)
            per_seed.append(final)
            print(f"  -> success={final['eval_success']:.3f} "
                  f"on_goal={final['eval_on_goal']:.3f} coll={final['eval_coll']:.3f}")
        results[arm] = per_seed

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    def col(arm, key):
        return [r[key] for r in results[arm]]

    print(f"\n{'arm':<10}{'success':>10}{'on-goal':>10}{'collisions':>12}"
          f"{'comm rate':>11}")
    print("-" * 55)
    for arm in a.arms:
        # "shuffled" transmits from every agent exactly like "full"; only the
        # sender/message correspondence is permuted. It has no scheduler, so
        # it logs no comm_rate, and the old fallback printed 0.000 for it.
        rate = np.mean(col(arm, "comm_rate")) if "comm_rate" in results[arm][0] \
            else (1.0 if arm in ("full", "shuffled") else 0.0)
        print(f"{arm:<10}{np.mean(col(arm, 'eval_success')):>10.4f}"
              f"{np.mean(col(arm, 'eval_on_goal')):>10.4f}"
              f"{np.mean(col(arm, 'eval_coll')):>12.4f}{rate:>11.3f}")

    if "learned" in a.arms:
        print(f"\n{'comparison':<22}{'baseline':>10}{'learned':>10}{'delta':>10}"
              f"{'p':>9}  verdict")
        print("-" * 86)
        checks = []
        for other in [x for x in ("none", "full", "random", "distance") if x in a.arms]:
            print(f"\n  vs {other}:")
            for key, lower in (("eval_coll", True), ("eval_success", False)):
                label = "collisions" if lower else "success"
                print("  " + summarise(label, col(other, key), col("learned", key),
                                       lower_is_better=lower))

        # The gate: beat full-comm on success without giving back its collisions.
        if "full" in a.arms and "none" in a.arms:
            lc, ls = np.mean(col("learned", "eval_coll")), np.mean(col("learned", "eval_success"))
            fc = np.mean(col("full", "eval_coll"))
            ns = np.mean(col("none", "eval_success"))
            checks = [
                (f"collisions near full-comm (<= {fc + 0.005:.3f})", lc <= fc + 0.005),
                (f"success near no-comm (>= {ns - 0.005:.3f})", ls >= ns - 0.005),
            ]
            if "random" in a.arms:
                r = welch_ttest(col("random", "eval_coll"), col("learned", "eval_coll"))
                checks.append((f"beats budget-matched random (p={r['p']:.3f})",
                               lc < np.mean(col("random", "eval_coll")) and r["p"] < 0.05))
            print()
            for name, ok in checks:
                print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
            print("\nGATE: " + ("PASSED" if all(v for _, v in checks) else "FAILED"))
