"""Prioritised planning: a fast, near-complete classical MAPF solver.

Agents are planned one at a time in priority order; each avoids the
space-time footprint of everyone before it. Prioritised planning is
incomplete -- a bad ordering can make a later agent unsolvable -- so we retry
with reshuffled orderings. In practice a handful of restarts solves the vast
majority of random instances, which is all we need to bootstrap learning.

We deliberately do not use optimal CBS here: demonstrations only need to be
*good and plentiful*, and prioritised planning is orders of magnitude cheaper.
CBS can be added later purely as an optimality reference for the paper.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .space_time_astar import (Reservations, path_to_actions,
                               space_time_astar)

STAY = 0


def prioritized_plan(
    obstacles: np.ndarray,
    starts: np.ndarray,
    goals: np.ndarray,
    cost_to_go: np.ndarray,
    max_t: int,
    rng: np.random.Generator,
    restarts: int = 6,
) -> Optional[List[List[tuple]]]:
    """Plan conflict-free paths for every agent.

    Args:
        obstacles: ``(H, W)`` bool.
        starts, goals: ``(N, 2)`` int.
        cost_to_go: ``(N, H, W)`` int, each agent's backward-BFS field.
        max_t: planning horizon.

    Returns paths indexed by agent, or ``None`` if every ordering failed.
    """
    n_agents = len(starts)

    for attempt in range(restarts):
        if attempt == 0:
            # Hardest-first: agents with the longest journeys get right of way.
            order = np.argsort([-cost_to_go[i][tuple(starts[i])] for i in range(n_agents)])
        else:
            order = rng.permutation(n_agents)

        res = Reservations()
        paths: List[Optional[list]] = [None] * n_agents
        ok = True
        for i in order:
            path = space_time_astar(
                obstacles, cost_to_go[i], tuple(starts[i]), tuple(goals[i]), res, max_t
            )
            if path is None:
                ok = False
                break
            res.add_path(path)
            paths[i] = path
        if ok:
            return paths  # type: ignore[return-value]

    return None


def plans_to_action_matrix(paths: List[List[tuple]], horizon: Optional[int] = None) -> np.ndarray:
    """Stack per-agent paths into an ``(T, N)`` action matrix padded with STAY."""
    per_agent = [path_to_actions(p) for p in paths]
    T = horizon if horizon is not None else max((len(a) for a in per_agent), default=0)
    out = np.full((T, len(paths)), STAY, dtype=np.int64)
    for i, acts in enumerate(per_agent):
        k = min(len(acts), T)
        out[:k, i] = acts[:k]
    return out
