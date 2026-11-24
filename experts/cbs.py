"""Conflict-based search: a complete MAPF solver.

Prioritised planning cannot demonstrate mazes. It reserves each agent's goal
cell permanently, so an agent whose goal sits in a corridor severs that
corridor for everyone planned after it -- and 94.9% of maze free cells have
degree <= 2. Two agents fail half the time, and no priority ordering recovers,
because the obstruction is a property of the goal assignment rather than of
the ordering. See docs/FINDINGS-phase6b.md section 19.

CBS has no priority ordering to get wrong. It plans each agent independently,
finds the first conflict between the resulting paths, and branches on it: one
child forbids agent A from that cell at that time, the other forbids agent B.
Both children are explored, so no single commitment can make the instance
unsolvable. It is complete and optimal for sum-of-costs (Sharon et al., 2015).

The price is worst-case exponential time in the number of conflicts, which is
why `cbs_plan` takes a node and time budget and returns None rather than
running forever. Whether that budget suffices in a maze is an empirical
question -- `scripts/bench_planner.py` measures it rather than assuming.

The low-level search is `space_time_astar` unchanged. `ConstraintSet` presents
the same interface as `Reservations`, so the solver that prioritised planning
already relies on is the one CBS uses too, rather than a second copy that can
drift from it.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .space_time_astar import Cell, path_to_actions, space_time_astar

INF = 1 << 30


class ConstraintSet:
    """One agent's CBS constraints, shaped like `Reservations`.

    The duck-typing is deliberate: `space_time_astar` asks a reservation
    object four questions, and a constraint set can answer all four. Reusing
    it means the low-level search is the same tested code in both solvers.

    The one subtlety is `last_use`. In prioritised planning it records the
    last time another agent passes through a cell, so our agent may not settle
    there earlier. A CBS vertex constraint on the goal means exactly the same
    thing -- the agent may not be parked there at that time -- so goal
    constraints are stored there and the low level enforces them unchanged.
    """

    __slots__ = ("vertex", "edge", "permanent", "last_use")

    def __init__(self) -> None:
        self.vertex: Set[Tuple[int, int, int]] = set()           # (t, r, c)
        self.edge: Set[Tuple[int, int, int, int, int]] = set()   # (t, r1, c1, r2, c2)
        self.permanent: Dict[Cell, int] = {}                     # always empty
        self.last_use: Dict[Cell, int] = {}                      # goal -> latest forbidden t

    def vertex_blocked(self, t: int, r: int, c: int) -> bool:
        return (t, r, c) in self.vertex

    def edge_blocked(self, t: int, r1: int, c1: int, r2: int, c2: int) -> bool:
        # A CBS edge constraint forbids this exact move, unlike a reservation
        # table, which forbids the move that would swap with a reserved one.
        return (t, r1, c1, r2, c2) in self.edge

    def add_vertex(self, t: int, cell: Cell, goal: Optional[Cell] = None) -> None:
        self.vertex.add((t, cell[0], cell[1]))
        if goal is not None and tuple(cell) == tuple(goal):
            # Forbidding the goal at t also forbids settling there at or
            # before t; the low level reads that from last_use.
            if t > self.last_use.get(tuple(goal), -1):
                self.last_use[tuple(goal)] = t

    def add_edge(self, t: int, a: Cell, b: Cell) -> None:
        self.edge.add((t, a[0], a[1], b[0], b[1]))

    def copy(self) -> "ConstraintSet":
        out = ConstraintSet()
        out.vertex = set(self.vertex)
        out.edge = set(self.edge)
        out.last_use = dict(self.last_use)
        return out


def at(path: Sequence[Cell], t: int) -> Cell:
    """Where an agent is at time t, parked on its goal once it has arrived."""
    return path[t] if t < len(path) else path[-1]


def find_first_conflict(paths: Sequence[Sequence[Cell]]):
    """Earliest conflict between any pair, or None.

    Paths are compared past their own ends, because an agent sitting on its
    goal still occupies a cell. That is the whole reason mazes are hard: in a
    corridor, a parked agent is a wall.

    Returns ("vertex", i, j, t, cell) or ("edge", i, j, t, (a_from, a_to)).
    """
    n = len(paths)
    horizon = max((len(p) for p in paths), default=0)
    for t in range(horizon):
        for i in range(n):
            pi, pi_next = at(paths[i], t), at(paths[i], t + 1)
            for j in range(i + 1, n):
                pj, pj_next = at(paths[j], t), at(paths[j], t + 1)
                if pi == pj:
                    return ("vertex", i, j, t, pi)
                if pi == pj_next and pi_next == pj:
                    return ("edge", i, j, t, (pi, pi_next))
    return None


def count_conflicts(paths: Sequence[Sequence[Cell]]) -> int:
    """Total conflicting pairs-times. Used only to order the open list."""
    n = len(paths)
    horizon = max((len(p) for p in paths), default=0)
    total = 0
    for t in range(horizon):
        for i in range(n):
            pi, pi_next = at(paths[i], t), at(paths[i], t + 1)
            for j in range(i + 1, n):
                pj, pj_next = at(paths[j], t), at(paths[j], t + 1)
                if pi == pj or (pi == pj_next and pi_next == pj):
                    total += 1
    return total


def sum_of_costs(paths: Sequence[Sequence[Cell]]) -> int:
    return sum(len(p) - 1 for p in paths)


@dataclass(order=True)
class _Node:
    cost: int
    conflicts: int
    seq: int
    constraints: List[ConstraintSet] = field(compare=False, default_factory=list)
    paths: List[List[Cell]] = field(compare=False, default_factory=list)


@dataclass
class CBSStats:
    expanded: int = 0
    generated: int = 0
    seconds: float = 0.0
    # "solved" | "node_limit" | "time_limit" | "unreachable" | "horizon"
    reason: str = ""


def cbs_plan(
    obstacles: np.ndarray,
    starts: np.ndarray,
    goals: np.ndarray,
    cost_to_go: np.ndarray,
    max_t: int,
    node_limit: int = 10_000,
    time_limit: float = 10.0,
    stats: Optional[CBSStats] = None,
) -> Optional[List[List[Cell]]]:
    """Conflict-free paths for every agent, or None within the budget.

    None means one of four different things, and they are not
    interchangeable:

        unreachable  no path exists even ignoring every other agent
        horizon      a path exists but is longer than max_t
        node_limit   the search budget ran out
        time_limit   the clock ran out

    Only the first is a property of the instance. The rest are properties of
    the budget, and a benchmark that reports them all as "failed" is measuring
    itself. `horizon` matters more than it sounds: maze shortest paths here
    run past 300 steps, so a horizon chosen for random grids silently turns
    solvable mazes into failures.
    """
    st = stats if stats is not None else CBSStats()
    t_start = time.perf_counter()
    n = len(starts)

    constraints = [ConstraintSet() for _ in range(n)]
    root_paths: List[List[Cell]] = []
    for i in range(n):
        p = space_time_astar(obstacles, cost_to_go[i], tuple(starts[i]),
                             tuple(goals[i]), constraints[i], max_t)
        if p is None:
            # Separate the two causes: the goal may be walled off, or simply
            # further away than max_t allows.
            st.reason = ("unreachable" if cost_to_go[i][tuple(starts[i])] < 0
                         else "horizon")
            st.seconds = time.perf_counter() - t_start
            return None
        root_paths.append(p)

    seq = 0
    root = _Node(sum_of_costs(root_paths), count_conflicts(root_paths), seq,
                 constraints, root_paths)
    open_list: List[_Node] = [root]
    st.generated = 1

    while open_list:
        if time.perf_counter() - t_start > time_limit:
            st.reason = "time_limit"
            break
        if st.expanded >= node_limit:
            st.reason = "node_limit"
            break

        node = heapq.heappop(open_list)
        st.expanded += 1

        conflict = find_first_conflict(node.paths)
        if conflict is None:
            st.reason = "solved"
            st.seconds = time.perf_counter() - t_start
            return node.paths

        kind, i, j, t, what = conflict
        for agent in (i, j):
            child_constraints = [c.copy() for c in node.constraints]
            cs = child_constraints[agent]
            if kind == "vertex":
                cs.add_vertex(t, what, goal=tuple(goals[agent]))
            else:
                a_from, a_to = what
                # The move to forbid is this agent's own, which for the second
                # agent is the reverse of the pair reported by the detector.
                if agent == i:
                    cs.add_edge(t, a_from, a_to)
                else:
                    cs.add_edge(t, a_to, a_from)

            replanned = space_time_astar(
                obstacles, cost_to_go[agent], tuple(starts[agent]),
                tuple(goals[agent]), cs, max_t)
            if replanned is None:
                continue                    # this branch is dead, the sibling may not be

            child_paths = list(node.paths)
            child_paths[agent] = replanned
            seq += 1
            st.generated += 1
            heapq.heappush(open_list, _Node(
                sum_of_costs(child_paths), count_conflicts(child_paths), seq,
                child_constraints, child_paths))

    if not st.reason:
        st.reason = "exhausted"         # open list emptied: no solution exists

    st.seconds = time.perf_counter() - t_start
    return None


def plans_to_action_matrix(paths: List[List[Cell]],
                           horizon: Optional[int] = None) -> np.ndarray:
    """Stack per-agent paths into a (T, N) action matrix padded with STAY."""
    per_agent = [path_to_actions(p) for p in paths]
    T = horizon if horizon is not None else max((len(a) for a in per_agent), default=0)
    out = np.zeros((T, len(paths)), dtype=np.int64)
    for i, acts in enumerate(per_agent):
        k = min(len(acts), T)
        out[:k, i] = acts[:k]
    return out
