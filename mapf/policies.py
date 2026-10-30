"""Scripted reference policies.

These are not learners. They exist to (a) prove the environment is solvable
and correctly wired, (b) act as the trivial lower bounds in the results table,
and (c) measure how much a *reactive* agent gains from seeing further -- which
upper-bounds what communication could ever contribute.
"""
from __future__ import annotations

import numpy as np

from .env import ACTION_DELTA, N_ACTIONS, STAY, VecMAPFEnv


def greedy_actions(env: VecMAPFEnv, rng: np.random.Generator,
                   epsilon: float = 0.0) -> np.ndarray:
    """One-step-greedy descent on each agent's own cost-to-go field.

    Completely uncoordinated: agents ignore each other entirely.
    """
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
    cost = np.where(legal, cost, far) * 2
    cost[..., STAY] += 1                       # prefer moving on ties

    actions = np.where(legal.any(-1), cost.argmin(-1), STAY)
    if epsilon > 0:
        explore = rng.random((B, N)) < epsilon
        actions = np.where(explore, rng.integers(0, N_ACTIONS, (B, N)), actions)
    return actions


def congestion_aware_actions(
    env: VecMAPFEnv,
    rng: np.random.Generator,
    radius: int = 4,
    congestion_weight: float = 3.0,
    epsilon: float = 0.05,
) -> np.ndarray:
    """Greedy descent that also steers around *visible* other agents.

    ``radius`` is the Chebyshev visibility range, in cells. Sweeping it from 1
    to the whole map measures how much a reactive policy gains from non-local
    information -- the headroom any communication scheme is competing for.

    Each candidate cell is scored by the agent's own cost-to-go plus a penalty
    for visible agents sitting on or beside it, so a larger radius lets an
    agent give a distant crowd a wide berth rather than blundering into it.
    """
    B, N = env.num_envs, env.n_agents
    pos = env.pos
    far = env.height * env.width + 1

    # Who can see whom.
    pair = np.abs(pos[:, :, None, :] - pos[:, None, :, :]).max(-1)      # (B,N,N)
    visible = (pair <= radius) & ~np.eye(N, dtype=bool)[None]

    cand = pos[:, :, None, :] + ACTION_DELTA[None, None, :, :]          # (B,N,5,2)
    r = np.clip(cand[..., 0], 0, env.height - 1)
    c = np.clip(cand[..., 1], 0, env.width - 1)
    b = np.arange(B)[:, None, None]
    n = np.arange(N)[None, :, None]

    cost = env.dist[b, n, r, c].astype(np.float64)
    in_bounds = (
        (cand[..., 0] >= 0) & (cand[..., 0] < env.height)
        & (cand[..., 1] >= 0) & (cand[..., 1] < env.width)
    )
    legal = in_bounds & ~env.obstacles[b, r, c] & (cost >= 0)

    # Crowding around each candidate cell, counting only visible agents.
    d_cand = np.abs(cand[:, :, :, None, :] - pos[:, None, None, :, :]).max(-1)  # (B,N,5,N)
    vis = visible[:, :, None, :]
    occupied = ((d_cand == 0) & vis).any(-1)
    crowding = ((d_cand <= 1) & vis).sum(-1).astype(np.float64)

    score = np.where(legal, cost, far) * 2.0
    score += congestion_weight * crowding
    score += np.where(occupied, 50.0, 0.0)          # never walk onto a seen agent
    score[..., STAY] += 1.0                          # prefer moving on ties

    actions = np.where(legal.any(-1), score.argmin(-1), STAY)
    if epsilon > 0:
        explore = rng.random((B, N)) < epsilon
        actions = np.where(explore, rng.integers(0, N_ACTIONS, (B, N)), actions)
    return actions


def random_actions(env: VecMAPFEnv, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, N_ACTIONS, (env.num_envs, env.n_agents))
