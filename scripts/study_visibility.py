"""How much does a REACTIVE policy gain from seeing further?

This upper-bounds the value of communication. Messages are, at best, a way of
learning about agents you cannot see. If a reactive policy performs the same
with a radius-2 view as with full global visibility, then there is nothing for
any communication scheme to contribute and the project premise fails.

Run this before committing to the communication architecture.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from mapf.env import MAPFConfig, VecMAPFEnv
from mapf.policies import congestion_aware_actions


def evaluate(cfg, radius, episodes=300, num_envs=64, seed=0, eps=0.05):
    env = VecMAPFEnv(cfg, num_envs=num_envs, seed=seed, pool_seed=999)
    rng = np.random.default_rng(seed)
    solved, on_goal, steps = [], [], []
    while len(solved) < episodes:
        a = congestion_aware_actions(env, rng, radius=radius, epsilon=eps)
        _, _, _, done, info = env.step(a)
        if done.any():
            d = np.flatnonzero(done)
            solved.extend(info["solved"][d].tolist())
            on_goal.extend((info["agents_on_goal"][d] / cfg.n_agents).tolist())
            steps.extend(info["makespan"][d].tolist())
    k = episodes
    return np.mean(solved[:k]), np.mean(on_goal[:k]), np.mean(steps[:k])


if __name__ == "__main__":
    configs = [
        ("16 agents 32x32 d=0.20", 16, {"kind": "random", "height": 32, "width": 32, "density": 0.20}),
        ("32 agents 40x40 d=0.20", 32, {"kind": "random", "height": 40, "width": 40, "density": 0.20}),
        ("32 agents 32x32 d=0.25", 32, {"kind": "random", "height": 32, "width": 32, "density": 0.25}),
        ("32 agents room rs=8",    32, {"kind": "room", "height": 33, "width": 33, "room_size": 8}),
    ]
    radii = [1, 2, 4, 8, 16, 64]
    print("Reactive congestion-aware policy, success rate by visibility radius")
    print("(radius 4 ~ the 9x9 FOV; radius 64 = full global visibility)\n")
    header = "".join(f"{('r=' + str(r)):>9}" for r in radii)
    print(f"{'config':<26}{header}{'gain':>9}")
    print("-" * (26 + 9 * len(radii) + 9))
    for name, n, spec in configs:
        cfg = MAPFConfig(n_agents=n, max_steps=192, map_pool_size=64, map_spec=spec)
        row, goals = [], []
        for r in radii:
            s, g, _ = evaluate(cfg, r)
            row.append(s); goals.append(g)
        gain = row[-1] - row[2]          # full visibility vs FOV-sized radius
        print(f"{name:<26}" + "".join(f"{v:>8.1%}" + " " for v in row) + f"{gain:>+8.1%}")
        print(f"{'  (agents on goal)':<26}" + "".join(f"{v:>8.1%}" + " " for v in goals))
