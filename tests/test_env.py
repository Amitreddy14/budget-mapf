"""Correctness tests for the vectorised MAPF environment."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from mapf.env import (ACTION_DELTA, DOWN, LEFT, MAPFConfig, N_ACTIONS, RIGHT,
                      STAY, UP, VecMAPFEnv)
from mapf.maps import make_map


def _env(**kw):
    num_envs = kw.pop("num_envs", 16)
    return VecMAPFEnv(MAPFConfig(**kw), num_envs=num_envs, seed=0)


# ----------------------------------------------------------------- invariants

def test_agents_never_overlap_or_enter_walls():
    env = _env(n_agents=8, map_spec={"kind": "random", "height": 16, "width": 16, "density": 0.2})
    rng = np.random.default_rng(0)
    for _ in range(300):
        env.step(rng.integers(0, N_ACTIONS, size=(env.num_envs, env.n_agents)))
        b = np.arange(env.num_envs)[:, None]
        assert not env.obstacles[b, env.pos[..., 0], env.pos[..., 1]].any(), "agent inside a wall"
        cells = env.pos[..., 0] * env.width + env.pos[..., 1]
        for row in cells:
            assert len(np.unique(row)) == len(row), "two agents share a cell"
        assert (env.pos >= 0).all() and (env.pos[..., 0] < env.height).all()
        assert (env.pos[..., 1] < env.width).all()


def test_goals_are_always_reachable():
    env = _env(n_agents=6, map_spec={"kind": "maze", "height": 21, "width": 21})
    b = np.arange(env.num_envs)[:, None]
    n = np.arange(env.n_agents)[None, :]
    d = env.dist[b, n, env.pos[..., 0], env.pos[..., 1]]
    assert (d >= 0).all(), "sampled a start that cannot reach its goal"


# ------------------------------------------------------------ conflict rules

def test_vertex_conflict_both_agents_blocked():
    """Two agents stepping into the same empty cell must both be refused."""
    env = _env(n_agents=2, num_envs=1,
               map_spec={"kind": "empty", "height": 5, "width": 5})
    env.obstacles[:] = False
    env.pos[0] = np.array([[2, 1], [2, 3]])      # left and right of (2,2)
    env.goals[0] = np.array([[0, 0], [4, 4]])
    before = env.pos[0].copy()
    env.step(np.array([[RIGHT, LEFT]]))
    assert np.array_equal(env.pos[0], before), "vertex conflict was not refused"
    assert env.last_blocked[0].all()


def test_edge_swap_conflict_refused():
    """Adjacent agents may not exchange cells."""
    env = _env(n_agents=2, num_envs=1,
               map_spec={"kind": "empty", "height": 5, "width": 5})
    env.obstacles[:] = False
    env.pos[0] = np.array([[2, 2], [2, 3]])
    env.goals[0] = np.array([[0, 0], [4, 4]])
    before = env.pos[0].copy()
    env.step(np.array([[RIGHT, LEFT]]))
    assert np.array_equal(env.pos[0], before), "edge/swap conflict was not refused"
    assert env.last_blocked[0].all()


def test_rotation_cycle_is_allowed():
    """Three agents rotating around a free cycle conflict with nobody."""
    env = _env(n_agents=3, num_envs=1,
               map_spec={"kind": "empty", "height": 5, "width": 5})
    env.obstacles[:] = False
    #  A B      target:  . A
    #  C .               C B
    env.pos[0] = np.array([[1, 1], [1, 2], [2, 1]])
    env.goals[0] = np.array([[4, 4], [4, 3], [4, 2]])
    env.step(np.array([[RIGHT, DOWN, UP]]))
    expected = np.array([[1, 2], [2, 2], [1, 1]])
    assert np.array_equal(env.pos[0], expected), f"rotation blocked, got {env.pos[0]}"
    assert not env.last_blocked[0].any()


def test_chain_follow_is_allowed():
    """A follows B into the cell B vacates -- legal, must not be reverted."""
    env = _env(n_agents=2, num_envs=1,
               map_spec={"kind": "empty", "height": 5, "width": 5})
    env.obstacles[:] = False
    env.pos[0] = np.array([[2, 1], [2, 2]])
    env.goals[0] = np.array([[0, 0], [4, 4]])
    env.step(np.array([[RIGHT, RIGHT]]))
    assert np.array_equal(env.pos[0], np.array([[2, 2], [2, 3]]))
    assert not env.last_blocked[0].any()


def test_wall_collision_blocked_and_penalised():
    env = _env(n_agents=1, num_envs=1,
               map_spec={"kind": "empty", "height": 5, "width": 5})
    env.obstacles[:] = False
    env.obstacles[0, 1, 2] = True
    env.pos[0] = np.array([[2, 2]])
    env.goals[0] = np.array([[4, 4]])
    _, _, reward, _, _ = env.step(np.array([[UP]]))
    assert np.array_equal(env.pos[0], np.array([[2, 2]]))
    assert reward[0, 0] < env.cfg.collision_penalty / 2


# -------------------------------------------------------------------- rewards

def test_reaching_goal_is_rewarded_and_resting_is_free():
    env = _env(n_agents=1, num_envs=1,
               map_spec={"kind": "empty", "height": 5, "width": 5})
    env.obstacles[:] = False
    env.pos[0] = np.array([[2, 2]])
    env.goals[0] = np.array([[2, 3]])
    env.dist[:] = 0
    _, _, reward, done, _ = env.step(np.array([[RIGHT]]))
    # goal reward + team bonus (single agent => team solved), minus the step cost
    assert reward[0, 0] > env.cfg.goal_reward
    assert done[0], "single agent on its goal should finish the episode"


# --------------------------------------------------------------- observations

def test_observation_shapes_and_ranges():
    env = _env(n_agents=4, fov=9,
               map_spec={"kind": "random", "height": 16, "width": 16, "density": 0.2})
    grid, vec = env.reset()
    assert grid.shape == (env.num_envs, 4, 6, 9, 9)
    assert vec.shape == (env.num_envs, 4, 9)
    assert grid.dtype == np.float32 and vec.dtype == np.float32
    assert np.isfinite(grid).all() and np.isfinite(vec).all()
    assert grid.min() >= 0.0 and grid.max() <= 1.0


def test_agent_never_sees_itself_in_the_other_agents_channel():
    env = _env(n_agents=6, fov=7,
               map_spec={"kind": "empty", "height": 12, "width": 12})
    grid, _ = env.reset()
    centre = env.cfg.fov // 2
    assert (grid[:, :, 1, centre, centre] == 0).all()


def test_own_goal_channel_excluded_from_other_goals():
    env = _env(n_agents=4, fov=11,
               map_spec={"kind": "empty", "height": 9, "width": 9})
    grid, _ = env.reset()
    overlap = grid[:, :, 2] * grid[:, :, 3]     # other-goals AND own-goal
    assert overlap.sum() == 0


def test_cost_to_go_decreases_toward_goal():
    """The downhill channel must be non-empty whenever the agent is off-goal."""
    env = _env(n_agents=4, fov=5,
               map_spec={"kind": "empty", "height": 12, "width": 12})
    grid, vec = env.reset()
    off_goal = vec[..., 4] == 0
    downhill_any = grid[:, :, 5].reshape(env.num_envs, env.n_agents, -1).sum(-1) > 0
    assert downhill_any[off_goal].all(), "no downhill cell visible for an off-goal agent"


# ----------------------------------------------------------- reproducibility

def test_same_seed_same_trajectory():
    def rollout(seed):
        env = VecMAPFEnv(MAPFConfig(n_agents=4), num_envs=4, seed=seed)
        rng = np.random.default_rng(123)
        total = 0.0
        for _ in range(50):
            _, _, r, _, _ = env.step(rng.integers(0, N_ACTIONS, (4, 4)))
            total += r.sum()
        return total, env.pos.copy()

    a_r, a_p = rollout(7)
    b_r, b_p = rollout(7)
    c_r, _ = rollout(8)
    assert a_r == b_r and np.array_equal(a_p, b_p)
    assert a_r != c_r, "different seeds produced identical returns"


def test_episode_terminates_at_max_steps():
    env = _env(n_agents=8, max_steps=10,
               map_spec={"kind": "random", "height": 16, "width": 16, "density": 0.2})
    rng = np.random.default_rng(0)
    seen_done = False
    for _ in range(10):
        _, _, _, done, _ = env.step(rng.integers(0, N_ACTIONS, (env.num_envs, 8)))
        seen_done |= done.any()
    assert seen_done and (env.t < 10).all(), "timeout reset did not fire"


# ------------------------------------------------- reward exploit regression

def test_goal_reward_cannot_be_farmed_by_oscillating():
    """Stepping off a goal and back must never be net-profitable.

    Paying `goal_reward` on every entry makes this loop worth +3.4 per cycle,
    and PPO will find it: training return rises while success collapses.
    """
    # Two agents: #0 sits on its goal and oscillates, #1 never arrives, so the
    # episode never terminates and #0 can farm for as long as it likes. A
    # single agent cannot show this -- arriving ends the episode immediately.
    env = _env(n_agents=2, num_envs=1, max_steps=1000,
               map_spec={"kind": "empty", "height": 7, "width": 7})
    env.obstacles[:] = False
    env.pos[0] = np.array([[3, 3], [0, 0]])
    env.goals[0] = np.array([[3, 3], [6, 6]])
    env.dist[:] = 0
    env.reached[:] = np.array([[True, False]])

    total = 0.0
    for i in range(10):
        a = LEFT if i % 2 == 0 else RIGHT
        _, _, r, done, _ = env.step(np.array([[a, STAY]]))
        assert not done[0], "episode ended -- agent 1 should never reach its goal"
        total += float(r[0, 0])
    assert total < 0, f"oscillating on/off the goal earned {total:+.2f} -- farmable"


def test_first_arrival_is_rewarded_but_a_second_is_not():
    env = _env(n_agents=1, num_envs=1, max_steps=1000,
               map_spec={"kind": "empty", "height": 5, "width": 5})
    env.obstacles[:] = False
    env.pos[0] = np.array([[2, 1]])
    env.goals[0] = np.array([[2, 2]])
    env.dist[:] = 0

    _, _, r_first, _, _ = env.step(np.array([[RIGHT]]))      # arrive
    env.step(np.array([[LEFT]]))                             # leave
    _, _, r_again, _, _ = env.step(np.array([[RIGHT]]))      # arrive again

    assert float(r_first[0, 0]) > 4.0, "first arrival should pay goal_reward"
    assert float(r_again[0, 0]) < 1.0, "re-arrival must not pay goal_reward again"


def test_potential_shaping_sums_to_zero_around_a_loop():
    """Potential-based shaping is policy-invariant: any cycle nets ~zero."""
    cfg = MAPFConfig(n_agents=1, max_steps=1000, shaping_coef=1.0, shaping_gamma=1.0,
                     step_penalty=0.0, stay_penalty=0.0, collision_penalty=0.0,
                     goal_reward=0.0, leave_goal_penalty=0.0, team_bonus=0.0,
                     map_spec={"kind": "empty", "height": 7, "width": 7})
    env = VecMAPFEnv(cfg, num_envs=1, seed=0)
    env.obstacles[:] = False
    env.pos[0] = np.array([[3, 1]])
    env.goals[0] = np.array([[0, 0]])
    env.reached[:] = True
    from mapf.heuristic import backward_bfs_batch
    env.dist[:] = backward_bfs_batch(env.obstacles, env.goals)

    total = 0.0
    for a in (RIGHT, DOWN, LEFT, UP):          # a closed square loop
        _, _, r, _, _ = env.step(np.array([[a]]))
        total += float(r[0, 0])
    assert abs(total) < 1e-4, f"shaping is not potential-based: loop netted {total:+.4f}"


# ------------------------------------------- observability and reset staggering

def test_peer_goal_channel_can_be_hidden_without_changing_shape():
    """Hiding peer goals must zero the channel, not remove it.

    Keeping the tensor shape means the ablation does not change model
    capacity, so any performance difference is attributable to the
    information, not to a smaller network.
    """
    from mapf.env import CH_OTHER_GOALS
    spec = {"kind": "empty", "height": 12, "width": 12}
    seen = VecMAPFEnv(MAPFConfig(n_agents=6, fov=11, map_spec=spec,
                                 observe_peer_goals=True), num_envs=8, seed=1)
    hidden = VecMAPFEnv(MAPFConfig(n_agents=6, fov=11, map_spec=spec,
                                   observe_peer_goals=False), num_envs=8, seed=1)
    gs, _ = seen.observe()
    gh, _ = hidden.observe()
    assert gs.shape == gh.shape, "hiding peer goals changed the observation shape"
    assert gh[:, :, CH_OTHER_GOALS].sum() == 0, "peer-goal channel is not empty"
    assert gs[:, :, CH_OTHER_GOALS].sum() > 0, "nothing was hidden -- bad test fixture"
    # Every other channel must be untouched.
    for ch in range(gs.shape[2]):
        if ch != CH_OTHER_GOALS:
            assert np.allclose(gs[:, :, ch], gh[:, :, ch]), f"channel {ch} changed too"


def test_resets_are_staggered_across_environments():
    """Synchronised timeouts make every rollout batch correlated."""
    cfg = MAPFConfig(n_agents=4, max_steps=64, stagger_resets=True,
                     map_spec={"kind": "random", "height": 16, "width": 16, "density": 0.1})
    env = VecMAPFEnv(cfg, num_envs=32, seed=0)
    assert len(np.unique(env.t)) > 8, f"episode clocks barely vary: {np.unique(env.t)}"
    assert env.t.max() < cfg.max_steps

    same = VecMAPFEnv(MAPFConfig(n_agents=4, max_steps=64, stagger_resets=False,
                                 map_spec=cfg.map_spec), num_envs=32, seed=0)
    assert (same.t == 0).all(), "staggering should be off when disabled"


# --- obs_costmap ablation ---------------------------------------------------
# The cost-to-go channels are a globally computed, obstacle-aware shortest-path
# map handed to a nominally local observation. Behaviour cloning on them
# predicts the expert 98.6% of the time with no communication, which leaves
# almost no decision headroom. "none" removes them while keeping the
# straight-line bearing, so the task stays solvable.

def _costmap_env(level, **kw):
    from mapf.env import MAPFConfig, VecMAPFEnv
    cfg = MAPFConfig(n_agents=4, fov=9, max_steps=16, map_pool_size=4,
                     map_spec={"kind": "random", "height": 16, "width": 16,
                               "density": 0.15},
                     obs_costmap=level, **kw)
    e = VecMAPFEnv(cfg, num_envs=2, seed=0); e.reset()
    return e


def test_costmap_none_zeroes_the_global_field():
    import numpy as np
    g, v = _costmap_env("none").observe()
    assert np.all(g[:, :, 4] == 0.0), "cost-to-go channel must be zeroed"
    assert np.all(g[:, :, 5] == 0.0), "downhill channel must be zeroed"
    assert np.all(v[:, :, 3] == 0.0), "cost at the current cell must be zeroed"


def test_costmap_none_keeps_the_bearing_to_goal():
    """Otherwise the agent cannot know where its goal is and the task breaks."""
    import numpy as np
    _, v_field = _costmap_env("field").observe()
    _, v_none = _costmap_env("none").observe()
    assert np.allclose(v_field[:, :, :3], v_none[:, :, :3])
    assert np.abs(v_none[:, :, :3]).sum() > 0


def test_costmap_level_does_not_change_the_channel_count():
    """Parameter count must stay identical across levels, as for comm arms."""
    gf, vf = _costmap_env("field").observe()
    gn, vn = _costmap_env("none").observe()
    assert gf.shape == gn.shape and vf.shape == vn.shape


def test_costmap_field_is_the_default():
    from mapf.env import MAPFConfig
    assert MAPFConfig().obs_costmap == "field"


def test_costmap_coarse_sits_between_the_two_extremes():
    """A middle rung: distance bands, but no per-cell downhill marks."""
    import numpy as np
    counts = {}
    for level in ("field", "coarse", "none"):
        g, v = _costmap_env(level).observe()
        counts[level] = (len(np.unique(g[:, :, 4])), g[:, :, 5].sum())
    assert counts["field"][0] > counts["coarse"][0] > counts["none"][0] == 1
    assert counts["field"][1] > 0                      # downhill present
    assert counts["coarse"][1] == 0 == counts["none"][1]   # and removed above


def test_costmap_coarse_quantises_to_whole_bands():
    import numpy as np
    from mapf.env import MAPFConfig
    e = _costmap_env("coarse")
    g, _ = e.observe()
    scale = e.height + e.width
    steps = g[:, :, 4] * scale / MAPFConfig().costmap_quantum
    assert np.allclose(steps, np.round(steps), atol=1e-4), "values must land on band edges"


def test_costmap_coarse_keeps_the_bearing():
    import numpy as np
    _, v_field = _costmap_env("field").observe()
    _, v_coarse = _costmap_env("coarse").observe()
    assert np.allclose(v_field[:, :, :3], v_coarse[:, :, :3])
