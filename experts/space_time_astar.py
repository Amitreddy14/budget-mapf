"""Single-agent space-time A* against a reservation table.

This is the inner solver for prioritised planning. It searches over
``(row, col, time)`` states, honouring vertex and edge reservations already
laid down by higher-priority agents.

The heuristic is the agent's own backward-BFS cost-to-go field, which the
environment already computes. It ignores other agents, so it never
overestimates the true remaining cost -- admissible and consistent, which
makes the search tight.
"""
from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# Must match ACTION_DELTA order in mapf.env.
MOVES: Tuple[Tuple[int, int], ...] = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))

Cell = Tuple[int, int]
INF = 1 << 30


class Reservations:
    """Space-time occupancy claimed by already-planned agents."""

    __slots__ = ("vertex", "edge", "permanent", "last_use")

    def __init__(self) -> None:
        self.vertex: Set[Tuple[int, int, int]] = set()            # (t, r, c)
        self.edge: Set[Tuple[int, int, int, int, int]] = set()    # (t, r1, c1, r2, c2)
        self.permanent: Dict[Cell, int] = {}                      # cell -> parked from t
        self.last_use: Dict[Cell, int] = {}                       # cell -> last reserved t

    def vertex_blocked(self, t: int, r: int, c: int) -> bool:
        if (t, r, c) in self.vertex:
            return True
        parked = self.permanent.get((r, c))
        return parked is not None and t >= parked

    def edge_blocked(self, t: int, r1: int, c1: int, r2: int, c2: int) -> bool:
        """True if moving (r1,c1)->(r2,c2) over [t, t+1] swaps with a reserved move."""
        return (t, r2, c2, r1, c1) in self.edge

    def add_path(self, path: Sequence[Cell]) -> None:
        """Claim a path, including the agent parking on its final cell forever."""
        for t, (r, c) in enumerate(path):
            self.vertex.add((t, r, c))
            if t > self.last_use.get((r, c), -1):
                self.last_use[(r, c)] = t
        for t in range(len(path) - 1):
            (r1, c1), (r2, c2) = path[t], path[t + 1]
            self.edge.add((t, r1, c1, r2, c2))
        goal = path[-1]
        arrive = len(path) - 1
        self.permanent[goal] = min(self.permanent.get(goal, INF), arrive)


def space_time_astar(
    obstacles: np.ndarray,
    cost_to_go: np.ndarray,
    start: Cell,
    goal: Cell,
    res: Reservations,
    max_t: int,
) -> Optional[List[Cell]]:
    """Shortest conflict-free path from ``start`` to ``goal``.

    Returns the cell sequence for ``t = 0 .. T``, where the agent arrives at
    ``goal`` at ``T`` and can remain there indefinitely. ``None`` if no such
    path exists within ``max_t``.
    """
    height, width = obstacles.shape
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))

    # Someone else has already parked on our goal -- hopeless under this ordering.
    if goal in res.permanent:
        return None
    if cost_to_go[start] < 0:
        return None
    goal_free_after = res.last_use.get(goal, -1)

    came: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    seen: Set[Tuple[int, int, int]] = {(start[0], start[1], 0)}
    h0 = int(cost_to_go[start])
    heap: List[Tuple[int, int, int, int, int]] = [(h0, h0, 0, start[0], start[1])]

    while heap:
        _, _, t, r, c = heapq.heappop(heap)

        if (r, c) == goal and t > goal_free_after:
            path = [(r, c)]
            key = (r, c, t)
            while key in came:
                pr, pc, pt = came[key]
                path.append((pr, pc))
                key = (pr, pc, pt)
            path.reverse()
            return path

        if t >= max_t:
            continue

        nt = t + 1
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if obstacles[nr, nc]:
                continue
            h = int(cost_to_go[nr, nc])
            if h < 0:
                continue
            key = (nr, nc, nt)
            if key in seen:
                continue
            if res.vertex_blocked(nt, nr, nc):
                continue
            if res.edge_blocked(t, r, c, nr, nc):
                continue
            seen.add(key)
            came[key] = (r, c, t)
            heapq.heappush(heap, (nt + h, h, nt, nr, nc))

    return None


def path_to_actions(path: Sequence[Cell]) -> List[int]:
    """Convert a cell sequence into environment action indices."""
    actions = []
    for t in range(len(path) - 1):
        delta = (path[t + 1][0] - path[t][0], path[t + 1][1] - path[t][1])
        actions.append(MOVES.index(delta))
    return actions
