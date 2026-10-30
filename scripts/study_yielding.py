"""How often must the expert WAIT (rather than detour) to resolve a conflict?

The fraction of off-goal STAY actions in an optimal-ish plan measures how much
of the coordination in a map family is temporal (someone yields) rather than
spatial (someone goes around). Temporal coordination is what inter-agent
communication can plausibly help with; spatial avoidance needs no negotiation.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from experts.prioritized import plans_to_action_matrix, prioritized_plan
from mapf.env import MAPFConfig, VecMAPFEnv


def study(map_spec, n_agents, instances=60, seed=0, max_steps=256):
    cfg = MAPFConfig(n_agents=n_agents, max_steps=max_steps, map_pool_size=32,
                     map_spec=map_spec)
    try:
        env = VecMAPFEnv(cfg, num_envs=1, seed=seed)
    except ValueError:
        return None
    rng = np.random.default_rng(seed)

    solved = 0
    off_goal_stay = moves = 0
    detour = []
    for _ in range(instances):
        paths = prioritized_plan(env.obstacles[0], env.pos[0], env.goals[0],
                                 env.dist[0], max_steps, rng, restarts=6)
        if paths is None:
            env._reset_envs(np.array([0])); continue
        solved += 1
        for i, p in enumerate(paths):
            goal = tuple(env.goals[0, i])
            shortest = int(env.dist[0, i][tuple(env.pos[0, i])])
            taken = len(p) - 1
            detour.append(taken - shortest)
            for t in range(len(p) - 1):
                at_goal = p[t] == goal
                waited = p[t + 1] == p[t]
                if at_goal:
                    continue
                if waited:
                    off_goal_stay += 1
                else:
                    moves += 1
        env._reset_envs(np.array([0]))

    total = off_goal_stay + moves
    return dict(
        solved=solved / instances,
        wait_frac=off_goal_stay / max(total, 1),
        extra_steps=float(np.mean(detour)) if detour else float("nan"),
    )


if __name__ == "__main__":
    rows = []
    for name, spec in [
        ("random d=0.15", {"kind": "random", "height": 32, "width": 32, "density": 0.15}),
        ("random d=0.25", {"kind": "random", "height": 32, "width": 32, "density": 0.25}),
        ("random d=0.35", {"kind": "random", "height": 32, "width": 32, "density": 0.35}),
        ("room  rs=8",    {"kind": "room", "height": 33, "width": 33, "room_size": 8}),
        ("room  rs=5",    {"kind": "room", "height": 35, "width": 35, "room_size": 5}),
        ("maze",          {"kind": "maze", "height": 33, "width": 33}),
    ]:
        for n in (8, 16, 32):
            r = study(spec, n)
            if r is None:
                continue
            rows.append((name, n, r))

    print(f"{'map':<16}{'agents':>7}{'solved':>9}{'wait %':>9}{'extra steps':>13}")
    print("-" * 56)
    last = None
    for name, n, r in rows:
        if last and last != name:
            print()
        print(f"{name:<16}{n:>7}{r['solved']:>8.0%}{r['wait_frac']:>9.1%}{r['extra_steps']:>13.2f}")
        last = name
