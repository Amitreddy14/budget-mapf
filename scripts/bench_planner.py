"""Prioritised planning vs CBS, by map kind and agent count.

docs/FINDINGS-phase6b.md section 19 records that prioritised planning cannot
demonstrate mazes: 3/6 at two agents, 0/6 at six and beyond. The stated remedy
was "a complete solver such as conflict-based search". This measures whether
that remedy actually reaches useful agent counts, rather than assuming it.

CBS is complete but worst-case exponential in the number of conflicts, so a
failure here has three different meanings and the table keeps them apart:
`infeasible` (no solution exists), `node_limit` / `time_limit` (the budget
ran out, and a bigger budget might succeed).

    python scripts/bench_planner.py
    python scripts/bench_planner.py --kinds maze --agents 2 4 8 --scenarios 12
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from mapf.maps import make_map
from mapf.heuristic import backward_bfs_batch
from experts.prioritized import prioritized_plan
from experts.cbs import CBSStats, cbs_plan

SPECS = {
    "random": {"kind": "random", "height": 32, "width": 32, "density": 0.20},
    "room":   {"kind": "room", "height": 32, "width": 32, "room_size": 6},
    "maze":   {"kind": "maze", "height": 41, "width": 41},
}


def sample_instance(spec, n_agents, rng):
    obs = make_map(spec, rng)
    free = np.argwhere(~obs)
    if len(free) < 2 * n_agents:
        return None
    pick = rng.choice(len(free), size=2 * n_agents, replace=False)
    starts, goals = free[pick[:n_agents]], free[pick[n_agents:]]
    ctg = backward_bfs_batch(obs[None], goals[None])[0]
    if any(ctg[i][tuple(starts[i])] < 0 for i in range(n_agents)):
        return None                      # disconnected map, not a planner failure
    return obs, starts, goals, ctg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["random", "room", "maze"])
    ap.add_argument("--agents", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--scenarios", type=int, default=6)
    ap.add_argument("--max-t", type=int, default=0,
                    help="0 = size it from the instance. A fixed horizon "
                         "chosen for random grids turns solvable mazes into "
                         "failures: maze shortest paths here exceed 300.")
    ap.add_argument("--node-limit", type=int, default=10_000)
    ap.add_argument("--time-limit", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"{args.scenarios} scenarios per cell | CBS budget: "
          f"{args.node_limit} nodes / {args.time_limit}s\n")
    header = (f"{'map':<8}{'agents':>7}{'prioritised':>13}{'CBS':>9}"
              f"{'CBS s/solve':>13}  why CBS failed")
    print(header); print("-" * len(header))

    for kind in args.kinds:
        for n in args.agents:
            rng = np.random.default_rng(args.seed)
            pp_ok = cbs_ok = tried = 0
            cbs_time = 0.0
            reasons = {}
            for _ in range(args.scenarios):
                inst = sample_instance(SPECS[kind], n, rng)
                if inst is None:
                    continue
                obs, starts, goals, ctg = inst
                tried += 1
                # Horizon from the instance: the longest shortest-path, with
                # room to detour and to wait for every other agent.
                longest = max(int(ctg[i][tuple(starts[i])]) for i in range(n))
                max_t = args.max_t or int(longest * 2 + 8 * n + 16)

                if prioritized_plan(obs, starts, goals, ctg, max_t,
                                    np.random.default_rng(0)) is not None:
                    pp_ok += 1

                st = CBSStats()
                t0 = time.perf_counter()
                paths = cbs_plan(obs, starts, goals, ctg, max_t,
                                 node_limit=args.node_limit,
                                 time_limit=args.time_limit, stats=st)
                dt = time.perf_counter() - t0
                if paths is not None:
                    cbs_ok += 1
                    cbs_time += dt
                else:
                    reasons[st.reason] = reasons.get(st.reason, 0) + 1

            if not tried:
                continue
            per = f"{cbs_time / cbs_ok:>12.2f}s" if cbs_ok else f"{'--':>13}"
            why = ", ".join(f"{k} x{v}" for k, v in sorted(reasons.items())) or "--"
            print(f"{kind:<8}{n:>7}{pp_ok:>8}/{tried:<4}{cbs_ok:>5}/{tried:<3}{per}  {why}")


if __name__ == "__main__":
    main()
