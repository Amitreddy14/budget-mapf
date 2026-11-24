"""Conflict-based search.

The claim CBS exists to support is narrow and testable: prioritised planning
fails on instances that are solvable, because it commits to a priority order
and parks agents on their goals forever. These tests build the smallest
instance with that property and check that CBS solves what prioritised
planning cannot.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experts.cbs import (CBSStats, ConstraintSet, at, cbs_plan,
                         find_first_conflict, sum_of_costs)
from experts.prioritized import prioritized_plan
from mapf.heuristic import backward_bfs_batch


def _fields(obs, goals):
    """(N, H, W) cost-to-go, one field per agent, from the batched helper."""
    return backward_bfs_batch(obs[None], np.asarray(goals)[None])[0]


def _corridor():
    """A 1-wide corridor of 5 cells with a single alcove at its midpoint.

        . . . . .
        # # . # #

    Two agents must exchange ends. The only way past each other is for one to
    wait in the alcove, which requires the other to be *late* -- and a
    prioritised planner gives the first agent the shortest path by
    construction, then parks it on the second agent's start.
    """
    obs = np.ones((2, 5), dtype=bool)
    obs[0, :] = False
    obs[1, 2] = False
    starts = np.array([[0, 0], [0, 4]])
    goals = np.array([[0, 4], [0, 0]])
    return obs, starts, goals, _fields(obs, goals)


def _assert_valid(paths, starts, goals, obs):
    for i, p in enumerate(paths):
        assert tuple(p[0]) == tuple(starts[i]), f"agent {i} starts elsewhere"
        assert tuple(p[-1]) == tuple(goals[i]), f"agent {i} does not reach its goal"
        for (r, c) in p:
            assert not obs[r, c], f"agent {i} walks through a wall"
        for t in range(len(p) - 1):
            (r1, c1), (r2, c2) = p[t], p[t + 1]
            assert abs(r1 - r2) + abs(c1 - c2) <= 1, f"agent {i} teleports at t={t}"
    assert find_first_conflict(paths) is None, "paths conflict"


# --------------------------------------------------------------- conflicts

def test_conflict_detection_sees_a_parked_agent():
    """An agent sitting on its goal is still an obstacle. In a corridor it is
    a wall, which is the whole reason mazes defeat prioritised planning."""
    short = [(0, 0), (0, 1)]              # arrives at t=1, parks there
    passer = [(0, 3), (0, 2), (0, 1)]     # walks onto that cell at t=2
    c = find_first_conflict([short, passer])
    assert c is not None and c[0] == "vertex" and c[3] == 2


def test_conflict_detection_sees_a_swap():
    a = [(0, 0), (0, 1)]
    b = [(0, 1), (0, 0)]
    c = find_first_conflict([a, b])
    assert c is not None and c[0] == "edge" and c[3] == 0


def test_no_conflict_when_following():
    a = [(0, 0), (0, 1), (0, 2)]
    b = [(0, 1), (0, 2), (0, 3)]
    assert find_first_conflict([a, b]) is None


def test_at_parks_past_the_end():
    p = [(0, 0), (0, 1)]
    assert at(p, 0) == (0, 0) and at(p, 1) == (0, 1)
    assert at(p, 9) == (0, 1)


# ------------------------------------------------------------- constraints

def test_goal_constraint_forbids_settling_early():
    """A vertex constraint on the goal must also stop the agent parking there
    before that time, or CBS cannot resolve a conflict with a parked agent."""
    cs = ConstraintSet()
    cs.add_vertex(5, (2, 3), goal=(2, 3))
    assert cs.last_use[(2, 3)] == 5
    cs.add_vertex(2, (2, 3), goal=(2, 3))
    assert cs.last_use[(2, 3)] == 5, "an earlier constraint must not lower it"


def test_edge_constraint_is_directional():
    """Unlike a reservation table, a CBS edge constraint forbids one specific
    move, not the move that would swap with it."""
    cs = ConstraintSet()
    cs.add_edge(3, (0, 0), (0, 1))
    assert cs.edge_blocked(3, 0, 0, 0, 1)
    assert not cs.edge_blocked(3, 0, 1, 0, 0)
    assert not cs.edge_blocked(4, 0, 0, 0, 1)


def test_copy_does_not_alias():
    cs = ConstraintSet()
    cs.add_vertex(1, (0, 0))
    other = cs.copy()
    other.add_vertex(2, (1, 1))
    assert len(cs.vertex) == 1 and len(other.vertex) == 2


# ------------------------------------------------------------------ solver

def test_single_agent_takes_the_shortest_path():
    obs = np.zeros((5, 5), dtype=bool)
    starts, goals = np.array([[0, 0]]), np.array([[0, 4]])
    paths = cbs_plan(obs, starts, goals, _fields(obs, goals), max_t=32)
    assert paths is not None
    assert sum_of_costs(paths) == 4


def test_prioritised_planning_fails_the_corridor():
    """The premise. If this ever passes, the corridor stopped being a
    counterexample and the CBS test below proves nothing."""
    obs, starts, goals, ctg = _corridor()
    rng = np.random.default_rng(0)
    assert prioritized_plan(obs, starts, goals, ctg, max_t=32, rng=rng,
                            restarts=16) is None


def test_cbs_solves_the_corridor():
    """And the payoff: the same instance, solved."""
    obs, starts, goals, ctg = _corridor()
    stats = CBSStats()
    paths = cbs_plan(obs, starts, goals, ctg, max_t=32, stats=stats)
    assert paths is not None, f"CBS failed: {stats.reason}"
    assert stats.reason == "solved"
    _assert_valid(paths, starts, goals, obs)
    # One agent must have used the alcove; there is no other way past.
    assert any((1, 2) in [tuple(c) for c in p] for p in paths)


def test_budget_exhaustion_is_distinguishable_from_infeasible():
    """None has three meanings and they are not interchangeable."""
    obs, starts, goals, ctg = _corridor()
    stats = CBSStats()
    assert cbs_plan(obs, starts, goals, ctg, max_t=32,
                    node_limit=1, stats=stats) is None
    assert stats.reason == "node_limit"

    # A goal walled off from its start is genuinely infeasible.
    obs2 = np.zeros((3, 3), dtype=bool)
    obs2[1, :] = True
    s2, g2 = np.array([[0, 0]]), np.array([[2, 2]])
    stats2 = CBSStats()
    assert cbs_plan(obs2, s2, g2, _fields(obs2, g2), max_t=16,
                    stats=stats2) is None
    assert stats2.reason == "unreachable"

    # A reachable goal beyond the horizon is a budget failure, not a property
    # of the instance -- maze shortest paths here exceed 300 steps.
    obs3 = np.zeros((1, 20), dtype=bool)
    s3, g3 = np.array([[0, 0]]), np.array([[0, 19]])
    stats3 = CBSStats()
    assert cbs_plan(obs3, s3, g3, _fields(obs3, g3), max_t=5,
                    stats=stats3) is None
    assert stats3.reason == "horizon"


@pytest.mark.parametrize("seed", range(12))
def test_random_instances_are_solved_validly(seed):
    """Whatever CBS returns must be a legal, conflict-free, goal-reaching
    plan. Failing to solve inside the budget is allowed; lying is not."""
    rng = np.random.default_rng(seed)
    h = w = 8
    obs = rng.random((h, w)) < 0.15
    free = np.argwhere(~obs)
    if len(free) < 8:
        pytest.skip("degenerate map")
    pick = rng.choice(len(free), size=6, replace=False)
    starts, goals = free[pick[:3]], free[pick[3:]]
    ctg = _fields(obs, goals)
    if any(ctg[i][tuple(starts[i])] < 0 for i in range(3)):
        pytest.skip("goal unreachable from start")

    stats = CBSStats()
    paths = cbs_plan(obs, starts, goals, ctg, max_t=64,
                     node_limit=2_000, time_limit=5.0, stats=stats)
    if paths is None:
        assert stats.reason in ("node_limit", "time_limit", "infeasible")
        return
    _assert_valid(paths, starts, goals, obs)


def test_cbs_is_never_worse_than_prioritised_when_both_solve():
    """CBS is optimal for sum-of-costs; prioritised planning is not. Equal is
    expected on easy instances, worse would mean the search is broken."""
    rng = np.random.default_rng(7)
    compared = 0
    for seed in range(25):
        r = np.random.default_rng(seed)
        obs = r.random((9, 9)) < 0.12
        free = np.argwhere(~obs)
        if len(free) < 10:
            continue
        pick = r.choice(len(free), size=8, replace=False)
        starts, goals = free[pick[:4]], free[pick[4:]]
        ctg = _fields(obs, goals)
        if any(ctg[i][tuple(starts[i])] < 0 for i in range(4)):
            continue
        pp = prioritized_plan(obs, starts, goals, ctg, max_t=64, rng=rng)
        cb = cbs_plan(obs, starts, goals, ctg, max_t=64,
                      node_limit=3_000, time_limit=5.0)
        if pp is None or cb is None:
            continue
        compared += 1
        assert sum_of_costs(cb) <= sum_of_costs(pp), f"seed {seed}"
    assert compared >= 5, f"only {compared} comparable instances"
