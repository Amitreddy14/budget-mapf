"""Does the expert's action depend on information OUTSIDE the agent's FOV?

This is the decisive question for the whole project. Communication can only
help where an agent's correct action depends on state it cannot see. If every
conflict-driven decision involves an agent already inside the field of view,
the FOV is sufficient and messages are redundant -- no scheduler, however
clever, can beat that.

Method. For each expert decision we compute the set of actions that greedily
descend the agent's own cost-to-go field (own goal + static map only, other
agents ignored). When the expert picks something outside that set, it is
deviating in order to accommodate other agents. We then ask how far away the
nearest other agent is: inside the FOV (locally observable) or beyond it
(requires non-local information).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from experts.prioritized import prioritized_plan
from mapf.env import ACTION_DELTA, MAPFConfig, VecMAPFEnv
from experts.space_time_astar import MOVES


def greedy_action_set(obstacles, cost, cell):
    """All actions that strictly descend the cost-to-go field (plus ties)."""
    H, W = obstacles.shape
    best, acts = None, []
    for a, (dr, dc) in enumerate(MOVES):
        r, c = cell[0] + dr, cell[1] + dc
        if not (0 <= r < H and 0 <= c < W) or obstacles[r, c]:
            continue
        d = int(cost[r, c])
        if d < 0:
            continue
        if best is None or d < best:
            best, acts = d, [a]
        elif d == best:
            acts.append(a)
    return set(acts)


def study(map_spec, n_agents, fov, instances=60, seed=0, max_steps=256):
    cfg = MAPFConfig(n_agents=n_agents, max_steps=max_steps, fov=fov,
                     map_pool_size=32, map_spec=map_spec)
    env = VecMAPFEnv(cfg, num_envs=1, seed=seed)
    rng = np.random.default_rng(seed)
    radius = fov // 2

    decisions = deviations = 0
    dev_local = dev_nonlocal = 0
    nearest_hist = []

    solved = 0
    for _ in range(instances):
        obstacles = env.obstacles[0].copy()
        goals = env.goals[0].copy()
        cost = env.dist[0].copy()
        paths = prioritized_plan(obstacles, env.pos[0], goals, cost,
                                 max_steps, rng, restarts=6)
        if paths is None:
            env._reset_envs(np.array([0])); continue
        solved += 1

        horizon = max(len(p) for p in paths)
        for t in range(horizon - 1):
            # Where everyone is at time t.
            now = np.array([p[min(t, len(p) - 1)] for p in paths])
            for i, p in enumerate(paths):
                if t >= len(p) - 1:
                    continue
                cell = p[t]
                if cell == tuple(goals[i]):
                    continue
                delta = (p[t + 1][0] - cell[0], p[t + 1][1] - cell[1])
                a_expert = MOVES.index(delta)

                decisions += 1
                if a_expert in greedy_action_set(obstacles, cost[i], cell):
                    continue
                deviations += 1

                others = np.delete(now, i, axis=0)
                cheb = np.abs(others - np.array(cell)).max(axis=1)
                nearest = int(cheb.min()) if len(cheb) else 99
                nearest_hist.append(nearest)
                if nearest <= radius:
                    dev_local += 1
                else:
                    dev_nonlocal += 1

        env._reset_envs(np.array([0]))

    if decisions == 0:
        return None
    return dict(
        solved=solved / instances,
        dev_rate=deviations / decisions,
        nonlocal_of_dev=dev_nonlocal / max(deviations, 1),
        nonlocal_of_all=dev_nonlocal / decisions,
        median_nearest=float(np.median(nearest_hist)) if nearest_hist else float("nan"),
    )


if __name__ == "__main__":
    print("Expert decisions that deviate from pure greedy descent, and whether the")
    print("agent responsible is visible within the FOV.\n")
    print(f"{'map':<16}{'agents':>7}{'fov':>5}{'deviate':>9}{'of those,':>11}{'non-local':>11}{'med dist':>10}")
    print(f"{'':<16}{'':>7}{'':>5}{'% of all':>9}{'non-local':>11}{'% of all':>11}{'to blame':>10}")
    print("-" * 70)
    specs = [
        ("random d=0.15", {"kind": "random", "height": 32, "width": 32, "density": 0.15}),
        ("random d=0.25", {"kind": "random", "height": 32, "width": 32, "density": 0.25}),
        ("room  rs=8",    {"kind": "room", "height": 33, "width": 33, "room_size": 8}),
    ]
    last = None
    for name, spec in specs:
        for n in (8, 16, 32):
            for fov in (5, 9, 15):
                r = study(spec, n, fov)
                if r is None:
                    continue
                if last and last != (name, n):
                    print()
                print(f"{name:<16}{n:>7}{fov:>5}{r['dev_rate']:>9.1%}"
                      f"{r['nonlocal_of_dev']:>11.1%}{r['nonlocal_of_all']:>11.1%}"
                      f"{r['median_nearest']:>10.0f}")
                last = (name, n)
