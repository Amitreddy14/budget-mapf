"""Expert planner tests.

The strongest check here is cross-validation: the planner and the environment
implement MAPF conflict rules independently, so replaying a plan through the
env must produce exactly the planned positions with zero collisions. If either
side has a bug, they disagree.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from experts.prioritized import plans_to_action_matrix, prioritized_plan
from experts.space_time_astar import (Reservations, path_to_actions,
                                      space_time_astar)
from mapf.env import MAPFConfig, VecMAPFEnv


def _plan_env(env, b, rng, restarts=6):
    return prioritized_plan(
        env.obstacles[b], env.pos[b], env.goals[b], env.dist[b],
        max_t=env.cfg.max_steps, rng=rng, restarts=restarts,
    )


# ------------------------------------------------------------ unit behaviour

def test_astar_finds_straight_line_on_empty_grid():
    obstacles = np.zeros((7, 7), dtype=bool)
    from mapf.heuristic import backward_bfs
    h = backward_bfs(obstacles, (3, 6))
    path = space_time_astar(obstacles, h, (3, 0), (3, 6), Reservations(), 32)
    assert path is not None
    assert len(path) == 7, f"expected 6 moves, got {len(path) - 1}"
    assert path[0] == (3, 0) and path[-1] == (3, 6)


def test_astar_waits_to_avoid_a_reservation():
    """A corridor blocked at one timestep forces a wait, not a detour."""
    obstacles = np.ones((3, 5), dtype=bool)
    obstacles[1, :] = False                      # single horizontal corridor
    from mapf.heuristic import backward_bfs
    h = backward_bfs(obstacles, (1, 4))

    res = Reservations()
    res.vertex.add((1, 1, 1))                    # cell (1,1) taken at t=1
    path = space_time_astar(obstacles, h, (1, 0), (1, 4), res, 32)
    assert path is not None
    assert path[1] == (1, 0), "agent should have waited at its start"
    assert path[-1] == (1, 4)


def test_astar_respects_edge_swap_reservation():
    obstacles = np.ones((3, 4), dtype=bool)
    obstacles[1, :] = False
    from mapf.heuristic import backward_bfs
    h = backward_bfs(obstacles, (1, 3))
    res = Reservations()
    res.edge.add((0, 1, 2, 1, 1))                # someone moves (1,2)->(1,1) at t=0
    path = space_time_astar(obstacles, h, (1, 1), (1, 3), res, 32)
    # Moving (1,1)->(1,2) at t=0 would be a head-on swap, so it must wait.
    assert path is None or path[1] != (1, 2)


def test_astar_rejects_a_permanently_parked_goal():
    obstacles = np.zeros((5, 5), dtype=bool)
    from mapf.heuristic import backward_bfs
    h = backward_bfs(obstacles, (4, 4))
    res = Reservations()
    res.add_path([(4, 4)])                       # another agent already parked there
    assert space_time_astar(obstacles, h, (0, 0), (4, 4), res, 32) is None


def test_path_to_actions_roundtrip():
    path = [(2, 2), (2, 3), (1, 3), (1, 3), (2, 3)]
    assert path_to_actions(path) == [4, 1, 0, 2]   # RIGHT, UP, STAY, DOWN


# ------------------------------------------------- cross-validation with env

@pytest.mark.parametrize("n_agents,size,density", [(4, 16, 0.10), (8, 20, 0.15), (16, 32, 0.20)])
def test_expert_plans_replay_without_collision(n_agents, size, density):
    cfg = MAPFConfig(n_agents=n_agents, max_steps=128, map_pool_size=16,
                     map_spec={"kind": "random", "height": size, "width": size,
                               "density": density})
    env = VecMAPFEnv(cfg, num_envs=1, seed=3)
    rng = np.random.default_rng(3)

    checked = 0
    for _ in range(12):
        paths = _plan_env(env, 0, rng)
        if paths is None:
            env._reset_envs(np.array([0]))
            continue
        checked += 1

        expected = [np.array(p) for p in paths]
        actions = plans_to_action_matrix(paths)
        for t in range(len(actions)):
            _, _, _, done, info = env.step(actions[t][None])
            assert info["collisions"][0] == 0, f"collision at t={t} replaying an expert plan"
            if done[0]:
                assert info["solved"][0], "plan finished without solving the instance"
                break
            # Positions must match the plan exactly.
            for i, p in enumerate(expected):
                want = p[min(t + 1, len(p) - 1)]
                assert tuple(env.pos[0, i]) == tuple(want), (
                    f"agent {i} diverged from plan at t={t}")
        else:
            raise AssertionError("plan exhausted without the episode terminating")

    assert checked >= 8, f"planner only solved {checked}/12 instances"
