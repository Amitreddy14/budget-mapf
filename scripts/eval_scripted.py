"""Scripted baselines, measured with the unbiased episode harness.

Every environment contributes the same number of episodes, so slow failures
are represented in their true proportion. See eval/metrics.py for why the
naive harness over-reports success by ~20 points.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from eval.metrics import run_episodes
from mapf.env import MAPFConfig, VecMAPFEnv
from mapf.policies import greedy_actions, random_actions

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_intent import intent_priority_actions


def evaluate(policy, cfg, eps=0.0, eps_per_env=4, num_envs=64, seed=0):
    env = VecMAPFEnv(cfg, num_envs=num_envs, seed=seed, pool_seed=999)
    rng = np.random.default_rng(seed)
    env._priority = rng.permutation(cfg.n_agents)[None, :].repeat(num_envs, 0)

    def act(e):
        if policy == "random":
            return random_actions(e, rng)
        if policy == "greedy":
            return greedy_actions(e, rng, epsilon=eps)
        return intent_priority_actions(e, rng, epsilon=eps)

    def on_end(finished):
        for b in finished:
            env._priority[b] = rng.permutation(cfg.n_agents)

    return run_episodes(env, act, episodes_per_env=eps_per_env, on_episode_end=on_end)


POLICIES = [
    ("random",            "random", 0.00),
    ("greedy",            "greedy", 0.00),
    ("greedy + noise",    "greedy", 0.10),
    ("intent + priority", "intent", 0.00),
    ("intent + pri+noise","intent", 0.05),
]

if __name__ == "__main__":
    print(f"{'config':<24}{'policy':<20}{'success':>9}{'on-goal':>9}{'coll/step':>11}{'eps':>6}")
    print("-" * 80)
    for n, size, d in [(8, 20, 0.15), (16, 32, 0.20), (32, 40, 0.20)]:
        cfg = MAPFConfig(n_agents=n, max_steps=160, map_pool_size=48,
                         map_spec={"kind": "random", "height": size, "width": size,
                                   "density": d})
        tag = f"{n} agents {size}x{size}"
        for name, pol, e in POLICIES:
            r = evaluate(pol, cfg, eps=e)
            print(f"{tag:<24}{name:<20}{r['success_rate']:>8.1%}"
                  f"{r['agents_on_goal']:>9.1%}{r['collisions_per_step']:>11.2f}"
                  f"{r['episodes']:>6}")
        print()
