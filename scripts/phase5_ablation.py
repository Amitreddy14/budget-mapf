"""Phase 5 gate: does intent communication beat no communication?

Runs the identical architecture with messaging off and on, across seeds. The
only difference between arms is `comm_mode`, so parameter count and capacity
are held fixed by construction.

Decision rule, from docs/FINDINGS-phase2.md: communication must lower
collisions per step AND not lose agents-on-goal. Success rate is secondary --
it is hostage to the single worst agent.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from eval.stats import eval_window
from train.ppo_comm import CommPPOConfig, train_comm_ppo

PRESETS = {
    "a16": dict(n_agents=16, height=32, width=32, density=0.20),
    "a32": dict(n_agents=32, height=40, width=40, density=0.20),
}


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
        return dict(n_agents=n, height=spec["height"], width=spec["width"],
                    density=spec["density"]), True
    return fallback, False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("preset", choices=list(PRESETS), nargs="?", default="a16")
    ap.add_argument("--steps", type=int, default=5_000_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--warm-start", required=True,
                    help="BC checkpoint. NOT optional: from scratch, 20M steps "
                         "reaches ~0.2%% agents-on-goal, versus 99%% warm-started.")
    ap.add_argument("--min-on-goal", type=float, default=0.50,
                    help="validity floor -- below this the comparison is meaningless")
    ap.add_argument("--tentative-coef", type=float, default=0.3)
    ap.add_argument("--out", default="runs/phase5")
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
    for mode in ("none", "full"):
        per_seed = []
        for seed in a.seeds:
            print(f"\n=== comm={mode} seed={seed} ===")
            cfg = CommPPOConfig(
                n_agents=p["n_agents"],
                map_spec={"kind": "random", "height": p["height"],
                          "width": p["width"], "density": p["density"]},
                comm_mode=mode, warm_start=a.warm_start, num_envs=a.num_envs,
                total_agent_steps=a.steps, seed=seed, device=a.device,
                eval_every=a.eval_every,
                tentative_coef=a.tentative_coef,
                out_dir=os.path.join(a.out, f"{a.preset}_{mode}_s{seed}"),
            )
            out = train_comm_ppo(cfg)
            final = eval_window(out["history"], a.eval_window)
            per_seed.append(final)
            print(f"  -> success={final['eval_success']:.3f} "
                  f"on_goal={final['eval_on_goal']:.3f} coll={final['eval_coll']:.3f}")
        results[mode] = per_seed

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    def agg(mode, key):
        vals = [r[key] for r in results[mode]]
        return np.mean(vals), np.std(vals)

    from eval.stats import summarise, welch_ttest, eval_window

    def seeds(mode, key):
        return [r[key] for r in results[mode]]

    print(f"\n{'metric':<16}{'no comm':>9}{'full comm':>12}{'delta':>11}"
          f"{'p':>9}  verdict")
    print("-" * 86)
    print(summarise("success", seeds("none", "eval_success"),
                    seeds("full", "eval_success")))
    print(summarise("agents-on-goal", seeds("none", "eval_on_goal"),
                    seeds("full", "eval_on_goal")))
    print(summarise("collisions", seeds("none", "eval_coll"),
                    seeds("full", "eval_coll"), lower_is_better=True))
    if "eval_makespan" in results["none"][0]:
        # Longer episodes with communication would confirm the caution
        # mechanism: agents that hear intentions yield rather than contest.
        print(summarise("makespan", seeds("none", "eval_makespan"),
                        seeds("full", "eval_makespan"), lower_is_better=True))

    coll = welch_ttest(seeds("none", "eval_coll"), seeds("full", "eval_coll"))
    print(f"\n  collisions: t={coll['t']:.2f}, df={coll['df']:.1f}, "
          f"Cohen's d={coll['cohens_d']:.2f}")
    if coll["p"] >= 0.05:
        print(f"  seeds per arm needed for 80% power at this effect size: "
              f"~{coll['n_for_80pc_power']}")

    # VALIDITY GATE, checked before any comparison.
    #
    # "fewer collisions, agents-on-goal no worse" is trivially satisfiable by
    # a policy that does nothing: frozen agents never collide. An earlier
    # version of this script reported PASS on two arms that both reached ~1%
    # agents-on-goal. A comparison between two broken policies is not a
    # result, so require both arms to actually work first.
    print()
    none_goal, _ = agg("none", "eval_on_goal")
    full_goal, _ = agg("full", "eval_on_goal")
    floor = a.min_on_goal
    if min(none_goal, full_goal) < floor:
        print(f"  [INVALID] both arms must reach agents-on-goal >= {floor:.2f} "
              f"(none={none_goal:.3f}, full={full_goal:.3f})")
        print("            A frozen policy scores zero collisions. Comparing two")
        print("            failed runs says nothing. Check the warm start and")
        print("            whether training converged at all.")
        print("\nGATE: INCONCLUSIVE -- training did not succeed in both arms")
        sys.exit(2)

    none_coll, _ = agg("none", "eval_coll")
    full_coll, _ = agg("full", "eval_coll")
    # Direction alone is not evidence. A 22% collision reduction over three
    # seeds carried p ~ 0.17 in the first run that reached this point, which
    # is exactly the kind of number that looks convincing in a table and
    # disappears on replication.
    coll_sig = coll["p"] < 0.05
    checks = [
        (f"both arms trained (on-goal >= {floor:.2f})", True),
        ("collisions lower with communication", full_coll < none_coll),
        (f"collision difference is significant (p={coll['p']:.3f})", coll_sig),
        ("agents-on-goal not worse", full_goal >= none_goal - 0.01),
    ]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    # A binary verdict cannot express a trade-off, and this comparison is one:
    # communication reliably removes collisions while costing task completion.
    # Reporting that as a plain FAIL throws away the actual finding.
    coll_better = full_coll < none_coll and coll_sig
    goal_worse = full_goal < none_goal - 0.005
    if coll_better and goal_worse:
        print("\nGATE: TRADE-OFF")
        print("  Communication significantly reduces collisions while costing")
        print("  task completion. This is the same signature as the scripted")
        print("  intent+priority oracle in Phase 2 (33x fewer collisions, 28pp")
        print("  lower success): agents that know peers' intentions yield more")
        print("  readily, which removes conflicts and slows progress.")
        print("  -> A SELECTIVE scheduler is the natural response: keep the")
        print("     collision benefit, pay the caution cost only where it buys")
        print("     something. That is Phase 6.")
    else:
        print("\nGATE: " + ("PASSED" if all(v for _, v in checks) else "FAILED"))
