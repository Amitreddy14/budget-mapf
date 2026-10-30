"""Backward BFS cost-to-go fields.

Every agent gets a full distance-to-its-own-goal field over the grid. This is
the single most informative input a learned MAPF policy can receive: it turns
"navigate to a goal you cannot see" into "follow the local gradient", leaving
the network free to spend capacity on the part that is actually hard --
negotiating with other agents.

Implemented as frontier expansion over whole arrays, so cost is O(diameter)
numpy operations rather than O(cells) Python iterations.
"""
from __future__ import annotations

import numpy as np

UNREACHABLE = np.int32(-1)


def backward_bfs_batch(obstacles: np.ndarray, goals: np.ndarray) -> np.ndarray:
    """4-connected shortest-path distance from each goal to every free cell.

    Args:
        obstacles: ``(B, H, W)`` bool, True == blocked.
        goals: ``(B, N, 2)`` int, goal (row, col) per agent.

    Returns:
        ``(B, N, H, W)`` int32 distances; ``-1`` where unreachable or blocked.
    """
    obstacles = np.asarray(obstacles)
    goals = np.asarray(goals)
    batch, height, width = obstacles.shape
    n_agents = goals.shape[1]

    free = ~obstacles[:, None, :, :]                       # (B, 1, H, W)
    free = np.broadcast_to(free, (batch, n_agents, height, width))

    dist = np.full((batch, n_agents, height, width), UNREACHABLE, dtype=np.int32)
    frontier = np.zeros((batch, n_agents, height, width), dtype=bool)

    b_idx = np.arange(batch)[:, None]
    n_idx = np.arange(n_agents)[None, :]
    frontier[b_idx, n_idx, goals[..., 0], goals[..., 1]] = True
    dist[b_idx, n_idx, goals[..., 0], goals[..., 1]] = 0

    step = 0
    max_steps = height * width
    while frontier.any() and step < max_steps:
        step += 1
        nxt = np.zeros_like(frontier)
        nxt[:, :, 1:, :] |= frontier[:, :, :-1, :]
        nxt[:, :, :-1, :] |= frontier[:, :, 1:, :]
        nxt[:, :, :, 1:] |= frontier[:, :, :, :-1]
        nxt[:, :, :, :-1] |= frontier[:, :, :, 1:]
        nxt &= free & (dist < 0)
        dist[nxt] = step
        frontier = nxt
    return dist


def backward_bfs(obstacles: np.ndarray, goal) -> np.ndarray:
    """Single-map, single-goal convenience wrapper. Returns ``(H, W)``."""
    goal_arr = np.asarray(goal, dtype=np.int64).reshape(1, 1, 2)
    return backward_bfs_batch(obstacles[None], goal_arr)[0, 0]
