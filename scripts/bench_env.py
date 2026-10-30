"""Throughput and sanity benchmark for the vectorised MAPF environment."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from mapf.env import MAPFConfig, N_ACTIONS, VecMAPFEnv


def bench(num_envs, n_agents, size, steps=200, density=0.15, label=""):
    cfg = MAPFConfig(n_agents=n_agents, max_steps=128,
                     map_spec={"kind": "random", "height": size, "width": size,
                               "density": density})
    env = VecMAPFEnv(cfg, num_envs=num_envs, seed=0)
    rng = np.random.default_rng(0)
    acts = rng.integers(0, N_ACTIONS, (steps, num_envs, n_agents))

    env.step(acts[0])                       # warm up
    t0 = time.perf_counter()
    on_goal, collisions = [], []
    for s in range(steps):
        _, _, _, _, info = env.step(acts[s])
        on_goal.append(info["agents_on_goal"].mean())
        collisions.append(info["collisions"].mean())
    dt = time.perf_counter() - t0

    env_sps = num_envs * steps / dt
    agent_sps = env_sps * n_agents
    print(f"{label:28s} env-steps/s {env_sps:10,.0f}   agent-steps/s {agent_sps:11,.0f}   "
          f"on-goal {np.mean(on_goal)/n_agents:5.1%}   collisions/step {np.mean(collisions):.2f}")
    return env_sps


if __name__ == "__main__":
    print("Random-policy benchmark (correctness baseline: a random policy should"
          " almost never solve an instance)\n")
    bench(64,  8,  20, label="64 envs,  8 agents, 20x20")
    bench(128, 8,  20, label="128 envs, 8 agents, 20x20")
    bench(256, 8,  20, label="256 envs, 8 agents, 20x20")
    bench(128, 16, 32, label="128 envs, 16 agents, 32x32")
    bench(64,  32, 40, label="64 envs,  32 agents, 40x40")
