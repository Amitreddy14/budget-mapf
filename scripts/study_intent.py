"""Is the bottleneck SEEING FURTHER, or KNOWING INTENT?

Position is observable; intention is not, at any range. Two adjacent agents
can see each other perfectly and still both step into the same cell, because
neither knows what the other is about to do. No increase in FOV ever fixes
that -- but one message does.

This isolates the two effects with scripted policies:

  greedy            : no coordination at all
  greedy + noise    : breaks ties by luck
  intent + priority : every agent broadcasts its intended action; conflicts are
                      resolved by a shared priority convention. This is a
                      FULL-COMMUNICATION ORACLE over intent, with no learning.

The gap between `greedy+noise` and `intent+priority` is the headroom that
intention-sharing can buy -- i.e. the ceiling for the whole project.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from mapf.env import ACTION_DELTA, MAPFConfig, N_ACTIONS, STAY, VecMAPFEnv
from mapf.policies import greedy_actions


def ranked_actions(env):
    """Each agent's actions sorted by its own cost-to-go (best first)."""
    B, N = env.num_envs, env.n_agents
    far = env.height * env.width + 1
    cand = env.pos[:, :, None, :] + ACTION_DELTA[None, None, :, :]
    r = np.clip(cand[..., 0], 0, env.height - 1)
    c = np.clip(cand[..., 1], 0, env.width - 1)
    b = np.arange(B)[:, None, None]
    n = np.arange(N)[None, :, None]
    cost = env.dist[b, n, r, c].astype(np.int64)
    in_bounds = (
        (cand[..., 0] >= 0) & (cand[..., 0] < env.height)
        & (cand[..., 1] >= 0) & (cand[..., 1] < env.width)
    )
    legal = in_bounds & ~env.obstacles[b, r, c] & (cost >= 0)
    score = np.where(legal, cost, far) * 2
    score[..., STAY] += 1
    return score.argsort(-1), legal


def intent_priority_actions(env, rng, epsilon=0.0):
    """Broadcast intended actions; resolve conflicts by a shared priority order.

    Agents declare their preferred move. Where two declarations collide (same
    target cell, or a head-on swap), the lower-priority agent yields and takes
    its next-best legal action. Priorities are reshuffled per episode so no
    agent is permanently starved.
    """
    B, N = env.num_envs, env.n_agents
    order, legal = ranked_actions(env)
    choice = np.zeros((B, N), dtype=np.int64)
    rank = np.zeros((B, N), dtype=np.int64)          # which preference each agent is on

    prio = env._priority                              # (B, N) lower = wins
    cur = env.pos

    for _ in range(4):                                # a few yield rounds
        actions = order[np.arange(B)[:, None], np.arange(N)[None, :], rank]
        target = cur + ACTION_DELTA[actions]

        cell = target[..., 0] * env.width + target[..., 1]
        same = (cell[:, :, None] == cell[:, None, :]) & ~np.eye(N, dtype=bool)[None]
        swap = (
            (target[:, :, None, :] == cur[:, None, :, :]).all(-1)
            & (cur[:, :, None, :] == target[:, None, :, :]).all(-1)
        ) & ~np.eye(N, dtype=bool)[None]
        clash = same | swap

        # An agent yields if it clashes with anyone of higher priority.
        outranked = prio[:, :, None] > prio[:, None, :]
        must_yield = (clash & outranked).any(-1)
        if not must_yield.any():
            break
        rank = np.where(must_yield & (rank < N_ACTIONS - 1), rank + 1, rank)

    actions = order[np.arange(B)[:, None], np.arange(N)[None, :], rank]
    actions = np.where(legal.any(-1), actions, STAY)
    if epsilon > 0:
        explore = rng.random((B, N)) < epsilon
        actions = np.where(explore, rng.integers(0, N_ACTIONS, (B, N)), actions)
    return actions


def evaluate(policy, cfg, episodes=200, num_envs=64, seed=0, eps=0.0):
    env = VecMAPFEnv(cfg, num_envs=num_envs, seed=seed, pool_seed=999)
    rng = np.random.default_rng(seed)
    env._priority = rng.permutation(cfg.n_agents)[None, :].repeat(num_envs, 0)
    solved, on_goal, coll = [], [], []
    while len(solved) < episodes:
        if policy == "greedy":
            a = greedy_actions(env, rng, epsilon=eps)
        else:
            a = intent_priority_actions(env, rng, epsilon=eps)
        _, _, _, done, info = env.step(a)
        coll.append(info["collisions"].mean())
        if done.any():
            d = np.flatnonzero(done)
            solved.extend(info["solved"][d].tolist())
            on_goal.extend((info["agents_on_goal"][d] / cfg.n_agents).tolist())
            env._priority[d] = rng.permutation(cfg.n_agents)
    k = episodes
    return np.mean(solved[:k]), np.mean(on_goal[:k]), np.mean(coll)


if __name__ == "__main__":
    print(f"{'config':<24}{'policy':<20}{'success':>9}{'on-goal':>9}{'coll/step':>11}")
    print("-" * 74)
    for n, size, d in [(8, 20, 0.15), (16, 32, 0.20), (32, 32, 0.25), (32, 40, 0.20)]:
        cfg = MAPFConfig(n_agents=n, max_steps=160, map_pool_size=48,
                         map_spec={"kind": "random", "height": size, "width": size,
                                   "density": d})
        tag = f"{n} agents {size}x{size} d={d}"
        base = None
        for pol, e, name in [("greedy", 0.0, "greedy"),
                             ("greedy", 0.10, "greedy + noise"),
                             ("intent", 0.0, "intent + priority"),
                             ("intent", 0.05, "intent + pri + noise")]:
            s, g, c = evaluate(pol, cfg, eps=e)
            if name == "greedy + noise":
                base = s
            mark = f"   ({s - base:+.1%})" if base is not None and "intent" in name else ""
            print(f"{tag:<24}{name:<20}{s:>8.1%}{g:>9.1%}{c:>11.2f}{mark}")
        print()
