"""Measure prioritised-planning success rate, cost and speed."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from experts.prioritized import prioritized_plan
from mapf.env import MAPFConfig, VecMAPFEnv


def evaluate(n_agents, size, density, instances=100, restarts=6, seed=0):
    cfg = MAPFConfig(n_agents=n_agents, max_steps=256, map_pool_size=64,
                     map_spec={"kind": "random", "height": size, "width": size,
                               "density": density})
    env = VecMAPFEnv(cfg, num_envs=1, seed=seed)
    rng = np.random.default_rng(seed)

    solved, makespan, soc, lb, times = 0, [], [], [], []
    for _ in range(instances):
        t0 = time.perf_counter()
        paths = prioritized_plan(env.obstacles[0], env.pos[0], env.goals[0],
                                 env.dist[0], cfg.max_steps, rng, restarts)
        times.append(time.perf_counter() - t0)
        if paths is not None:
            solved += 1
            makespan.append(max(len(p) - 1 for p in paths))
            soc.append(sum(len(p) - 1 for p in paths))
            # Lower bound: every agent on its own shortest path, ignoring others.
            lb.append(sum(int(env.dist[0, i][tuple(env.pos[0, i])]) for i in range(n_agents)))
        env._reset_envs(np.array([0]))

    ratio = np.mean([s / max(l, 1) for s, l in zip(soc, lb)]) if soc else float("nan")
    return (solved / instances, np.mean(makespan) if makespan else float("nan"),
            ratio, np.mean(times) * 1000, np.percentile(times, 95) * 1000)


if __name__ == "__main__":
    print(f"{'config':<30}{'solved':>8}{'makespan':>10}{'SoC/LB':>9}{'ms/inst':>10}{'p95 ms':>9}")
    print("-" * 76)
    for n, s, d in [(4, 20, 0.10), (8, 20, 0.15), (8, 32, 0.20),
                    (16, 32, 0.20), (32, 40, 0.20), (64, 64, 0.20)]:
        sr, mk, ratio, ms, p95 = evaluate(n, s, d)
        print(f"{n:2d} agents {s}x{s} d={d:<12.2f}{sr:>7.0%}{mk:>10.1f}{ratio:>9.3f}{ms:>10.1f}{p95:>9.1f}")
