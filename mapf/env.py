"""Vectorised partially-observable Multi-Agent Path Finding environment.

Design notes
------------
* ``B`` independent environments are stepped together as numpy array ops.
  Everything -- movement, conflict resolution, reward, observation -- is
  batched; there is no per-agent Python loop in the hot path.
* Agents are genuinely partially observable: each sees an ``F x F`` window
  centred on itself. This is what makes inter-agent communication meaningful,
  and is precisely what the v1 project was missing.
* Conflicts follow standard MAPF semantics: vertex conflicts (two agents
  entering one cell) and edge/swap conflicts (two agents exchanging cells)
  are both illegal. Rotations around a cycle are permitted.
* Maps are drawn from a pre-generated pool so that resets stay cheap. Use a
  distinct ``pool_seed`` for evaluation to get held-out maps.

Performance
-----------
Quantities that are constant within an episode (padded obstacle plane, padded
cost-to-go fields, goal plane) are cached at reset. Only the agent-occupancy
plane is rebuilt each step. Because agent positions are unique by construction
we can scatter with plain fancy indexing instead of ``np.add.at``, which is
roughly two orders of magnitude faster.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .heuristic import backward_bfs_batch
from .maps import largest_component_mask, make_map, sample_scenario

# Action indices.
STAY, UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4
N_ACTIONS = 5
ACTION_DELTA = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)
ACTION_NAMES = ("STAY", "UP", "DOWN", "LEFT", "RIGHT")

N_OBS_CHANNELS = 6
N_VEC_FEATURES = 9

# Observation channel order, for readability elsewhere.
CH_OBSTACLE, CH_AGENTS, CH_OTHER_GOALS, CH_OWN_GOAL, CH_COST, CH_DOWNHILL = range(6)


@dataclass
class MAPFConfig:
    """Environment configuration."""

    n_agents: int = 8
    map_spec: dict = field(default_factory=lambda: {
        "kind": "random", "height": 20, "width": 20, "density": 0.15
    })
    fov: int = 9                      # must be odd
    max_steps: int = 128
    map_pool_size: int = 128          # distinct maps cycled through at reset

    # Whether an agent can see OTHER agents' goal cells (observation channel 2).
    #
    # Leaving this True hands agents their neighbours' intentions for free:
    # measured at 16 agents / FOV 9, an agent sees at least one peer's goal
    # 69% of the time, and knowing where a neighbour is headed IS its intent.
    # That leaves nothing for a message to carry, and is also unrealistic --
    # a real robot cannot observe another robot's destination, it has to be
    # told. Setting this False zeroes the channel (keeping the tensor shape,
    # so model capacity is unchanged) and makes intent genuinely unobservable.
    observe_peer_goals: bool = True

    # "field"  -- channels 4/5 carry the backward-BFS cost-to-go and its
    #             downhill direction, and vec carries the cost at the current
    #             cell. This is a globally computed, obstacle-aware
    #             shortest-path map handed to a nominally local observation,
    #             and behaviour cloning on it predicts the expert's action
    #             98.6% of the time with no communication at all.
    # "coarse" -- the field is quantised to bands of `costmap_quantum` BFS
    #             steps and the downhill channel is zeroed. The agent learns
    #             roughly how far away its goal is but not which adjacent cell
    #             is strictly closer, which is the single feature that most
    #             directly determines the expert's next move. This is the
    #             middle rung between the two extremes below.
    # "none"   -- those three features are zeroed. The agent keeps the
    #             straight-line bearing and distance to its goal from the
    #             vector block, so the task stays solvable, but it must route
    #             around obstacles from what it can actually see.
    # The channel COUNT is unchanged so the network, and therefore the
    # parameter count, is identical across levels.
    obs_costmap: str = "field"        # "field" | "coarse" | "none"
    costmap_quantum: int = 8          # band width, in BFS steps, for "coarse"

    # Stagger initial episode lengths so environments do not all time out on
    # the same step. Synchronised resets make every rollout batch highly
    # correlated and leave most iterations with no completed episodes at all.
    stagger_resets: bool = True

    # Reward shaping.
    step_penalty: float = -0.3        # cost of any move off-goal
    stay_penalty: float = -0.5        # cost of idling off-goal
    collision_penalty: float = -2.0   # blocked by wall or lost a conflict
    goal_reward: float = 5.0          # FIRST arrival at own goal (see below)
    leave_goal_penalty: float = -1.0  # abandoning own goal
    team_bonus: float = 10.0          # every agent on goal simultaneously

    # Pay `goal_reward` only on an agent's first arrival in an episode.
    #
    # This is not optional in practice. Paying it on every entry creates a
    # farmable loop: step off the goal (-1.0 leave, -0.3 step) and back on
    # (+5.0 - 0.3) nets +3.4 per cycle, repeatable until the step limit.
    # Because an episode ends only when *all* agents are home, one agent can
    # farm this indefinitely while others are still travelling. PPO found and
    # exploited it within 30 iterations: episode return rose to +512 while
    # evaluation success collapsed from 70% to 6%. Leave this True unless you
    # are deliberately reproducing that failure.
    one_time_goal_reward: bool = True

    # Potential-based reward shaping on the cost-to-go field (Ng et al. 1999):
    #   F = coef * (gamma * Phi(s') - Phi(s)),  Phi = -distance_to_goal
    # Policy-invariant by construction, so it cannot be farmed -- any cycle
    # returning to the same state sums to zero. 0.0 disables it.
    shaping_coef: float = 0.0
    shaping_gamma: float = 0.99

    def __post_init__(self) -> None:
        if self.fov % 2 == 0:
            raise ValueError(f"fov must be odd, got {self.fov}")


class VecMAPFEnv:
    """``num_envs`` parallel MAPF instances sharing one grid shape."""

    def __init__(self, cfg: MAPFConfig, num_envs: int = 32, seed: int = 0,
                 pool_seed: Optional[int] = None):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.rng = np.random.default_rng(seed)
        self.n_agents = cfg.n_agents
        self.pad = cfg.fov // 2

        self._build_pool(seed if pool_seed is None else pool_seed)
        self.height, self.width = self._pool_grids.shape[1:]
        hp, wp = self.height + 2 * self.pad, self.width + 2 * self.pad

        B, N = self.num_envs, self.n_agents
        self.obstacles = np.zeros((B, self.height, self.width), dtype=bool)
        self.pos = np.zeros((B, N, 2), dtype=np.int64)
        self.goals = np.zeros((B, N, 2), dtype=np.int64)
        self.dist = np.zeros((B, N, self.height, self.width), dtype=np.int32)
        self.t = np.zeros(B, dtype=np.int64)
        self.last_blocked = np.zeros((B, N), dtype=bool)
        self.reached = np.zeros((B, N), dtype=bool)   # first arrival, per episode

        # Episode-constant caches (rebuilt only on reset).
        self._obst_pad = np.ones((B, hp, wp), dtype=np.float32)
        self._goal_pad = np.zeros((B, hp, wp), dtype=np.float32)
        self._dist_pad = np.ones((B, N, hp, wp), dtype=np.float32)

        # Reusable index helpers.
        self._b1 = np.arange(B)[:, None]
        self._n1 = np.arange(N)[None, :]
        self._b4 = np.arange(B)[:, None, None, None]
        self._n4 = np.arange(N)[None, :, None, None]
        self._offs = np.arange(cfg.fov)
        self._eye = np.eye(N, dtype=bool)[None]
        self._occ = np.zeros((B, self.height, self.width), dtype=np.float32)

        self._ep_return = np.zeros(B, dtype=np.float64)
        self.last_episode_stats: Dict[str, float] = {}

        self._reset_envs(np.arange(B))
        if cfg.stagger_resets and self.num_envs > 1:
            self.t = self.rng.integers(0, cfg.max_steps, size=self.num_envs)

    # ------------------------------------------------------------- map pool

    def _build_pool(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        grids: List[np.ndarray] = []
        regions: List[np.ndarray] = []
        need = 2 * self.cfg.n_agents
        attempts = 0
        while len(grids) < self.cfg.map_pool_size and attempts < self.cfg.map_pool_size * 20:
            attempts += 1
            grid = make_map(self.cfg.map_spec, rng)
            region = largest_component_mask(grid)
            if region.sum() >= need:
                grids.append(grid)
                regions.append(region)
        if not grids:
            raise ValueError("could not generate any map with enough free space")
        self._pool_grids = np.stack(grids)
        self._pool_regions = np.stack(regions)
        self._pool_cells = [np.argwhere(r) for r in self._pool_regions]

    # ------------------------------------------------------------------ setup

    def _reset_envs(self, idx: np.ndarray) -> None:
        """Regenerate map, starts and goals for the given environment indices."""
        idx = np.atleast_1d(idx)
        pad, N = self.pad, self.n_agents
        picks = self.rng.integers(0, len(self._pool_grids), size=len(idx))

        for b, p in zip(idx, picks):
            cells = self._pool_cells[p]
            sel = self.rng.choice(len(cells), size=2 * N, replace=False)
            self.obstacles[b] = self._pool_grids[p]
            self.pos[b] = cells[sel[:N]]
            self.goals[b] = cells[sel[N:]]

        self.dist[idx] = backward_bfs_batch(self.obstacles[idx], self.goals[idx])
        self.t[idx] = 0
        self.last_blocked[idx] = False
        self.reached[idx] = False
        self._ep_return[idx] = 0.0

        # Refresh episode-constant caches for these environments. A single
        # advanced index followed by slices keeps the batch axis in place.
        H, W = self.height, self.width
        rs, cs = slice(pad, pad + H), slice(pad, pad + W)

        self._obst_pad[idx] = 1.0
        self._obst_pad[idx, rs, cs] = self.obstacles[idx].astype(np.float32)

        goal_plane = np.zeros((len(idx), H, W), dtype=np.float32)
        goal_plane[np.arange(len(idx))[:, None], self.goals[idx][..., 0],
                   self.goals[idx][..., 1]] = 1.0
        self._goal_pad[idx] = 0.0
        self._goal_pad[idx, rs, cs] = goal_plane

        far = np.float32(H + W)
        dist_f = np.where(self.dist[idx] < 0, far, self.dist[idx]).astype(np.float32) / far
        self._dist_pad[idx] = 1.0
        self._dist_pad[idx, :, rs, cs] = dist_f

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self._reset_envs(np.arange(self.num_envs))
        return self.observe()

    # ------------------------------------------------------------ observation

    def _windows(self, padded: np.ndarray, per_agent: bool) -> np.ndarray:
        """Extract ``F x F`` windows centred on each agent -> ``(B, N, F, F)``."""
        rows = self.pos[..., 0][..., None, None] + self._offs[None, None, :, None]
        cols = self.pos[..., 1][..., None, None] + self._offs[None, None, None, :]
        if per_agent:
            return padded[self._b4, self._n4, rows, cols]
        return padded[self._b4, rows, cols]

    def observe(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns ``(grid_obs, vec_obs)``.

        ``grid_obs``: ``(B, N, C, F, F)`` float32
        ``vec_obs`` : ``(B, N, V)``       float32
        """
        pad, fov = self.pad, self.cfg.fov
        B, N, H, W = self.num_envs, self.n_agents, self.height, self.width

        obst_fov = self._windows(self._obst_pad, False)

        # Agent occupancy: positions are unique, so plain scatter is safe.
        occ = self._occ
        occ.fill(0.0)
        occ[self._b1, self.pos[..., 0], self.pos[..., 1]] = 1.0
        occ_pad = np.zeros_like(self._obst_pad)
        occ_pad[:, pad:pad + H, pad:pad + W] = occ
        others_fov = self._windows(occ_pad, False)
        others_fov[:, :, pad, pad] = 0.0

        # Own goal is a single cell: place it directly rather than building a plane.
        rel = self.goals - self.pos
        in_fov = (np.abs(rel) <= pad).all(-1)
        own_goal_fov = np.zeros((B, N, fov, fov), dtype=np.float32)
        ridx = np.clip(rel[..., 0] + pad, 0, fov - 1)
        cidx = np.clip(rel[..., 1] + pad, 0, fov - 1)
        own_goal_fov[self._b1, self._n1, ridx, cidx] = in_fov.astype(np.float32)

        all_goals_fov = self._windows(self._goal_pad, False)
        other_goals_fov = np.clip(all_goals_fov - own_goal_fov, 0.0, None)
        if not self.cfg.observe_peer_goals:
            other_goals_fov = np.zeros_like(other_goals_fov)

        cost_fov = self._windows(self._dist_pad, True)
        own_cost = self._dist_pad[self._b1, self._n1,
                                  self.pos[..., 0] + pad, self.pos[..., 1] + pad]
        downhill_fov = (cost_fov < own_cost[..., None, None]).astype(np.float32)
        downhill_fov *= (1.0 - obst_fov)

        if self.cfg.obs_costmap == "none":
            cost_fov = np.zeros_like(cost_fov)
            downhill_fov = np.zeros_like(downhill_fov)
            own_cost = np.zeros_like(own_cost)
        elif self.cfg.obs_costmap == "coarse":
            # The field is stored normalised by (H + W); recover BFS steps,
            # floor to the band, and renormalise. Within a band the agent
            # cannot tell which neighbour is closer to the goal.
            scale = np.float32(self.height + self.width)
            q = np.float32(self.cfg.costmap_quantum)
            cost_fov = np.floor(cost_fov * scale / q) * q / scale
            own_cost = np.floor(own_cost * scale / q) * q / scale
            downhill_fov = np.zeros_like(downhill_fov)

        grid_obs = np.stack(
            [obst_fov, others_fov, other_goals_fov, own_goal_fov, cost_fov, downhill_fov],
            axis=2,
        )

        far = np.float32(H + W)
        delta = rel.astype(np.float32)
        at_goal = (rel == 0).all(-1).astype(np.float32)
        n_nbrs = others_fov.reshape(B, N, -1).sum(-1) / max(N - 1, 1)
        unreachable = (self.dist[self._b1, self._n1,
                                 self.pos[..., 0], self.pos[..., 1]] < 0).astype(np.float32)
        time_frac = np.broadcast_to((self.t / self.cfg.max_steps)[:, None], (B, N)).astype(np.float32)

        vec_obs = np.stack([
            delta[..., 0] / H,
            delta[..., 1] / W,
            np.abs(delta).sum(-1) / far,
            own_cost,
            at_goal,
            n_nbrs,
            time_frac,
            self.last_blocked.astype(np.float32),
            unreachable,
        ], axis=-1)

        return grid_obs, vec_obs

    def comm_graph(self, radius: float) -> Tuple[np.ndarray, np.ndarray]:
        """Who can hear whom, and from which direction.

        Communication is range-limited: an agent can only receive from peers
        within ``radius`` cells (Chebyshev). Returns

            adjacency: ``(B, N, N)`` bool, True where j can send to i.
                       The diagonal is False -- an agent is not its own peer.
            rel_pos:   ``(B, N, N, 2)`` float32, j's offset from i, scaled to
                       roughly [-1, 1] by the radius.

        The relative offset matters as much as the message: "someone intends
        to move left" means something different depending on whether they are
        in front of you or behind you.
        """
        delta = (self.pos[:, None, :, :] - self.pos[:, :, None, :]).astype(np.float32)
        cheb = np.abs(delta).max(-1)
        adjacency = (cheb <= radius) & ~np.eye(self.n_agents, dtype=bool)[None]
        return adjacency, delta / max(float(radius), 1.0)

    def global_state(self) -> np.ndarray:
        """Centralised state for the critic / mixer: ``(B, N * 5)``."""
        H, W = np.float32(self.height), np.float32(self.width)
        at_goal = (self.pos == self.goals).all(-1).astype(np.float32)
        feats = np.stack([
            self.pos[..., 0] / H, self.pos[..., 1] / W,
            self.goals[..., 0] / H, self.goals[..., 1] / W,
            at_goal,
        ], axis=-1)
        return feats.reshape(self.num_envs, -1).astype(np.float32)

    # ------------------------------------------------------------- transition

    def _resolve(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply actions under MAPF conflict rules.

        Returns ``(new_pos, blocked)`` where ``blocked`` marks every agent that
        was prevented from executing its chosen move.
        """
        W = self.width
        cur = self.pos
        proposed = cur + ACTION_DELTA[actions]

        # Static blocking: grid bounds and walls.
        in_bounds = (
            (proposed[..., 0] >= 0) & (proposed[..., 0] < self.height)
            & (proposed[..., 1] >= 0) & (proposed[..., 1] < self.width)
        )
        safe = np.where(in_bounds[..., None], proposed, cur)
        hits_wall = self.obstacles[self._b1, safe[..., 0], safe[..., 1]]
        blocked = (~in_bounds) | hits_wall
        proposed = np.where(blocked[..., None], cur, safe)

        # Edge/swap conflicts need checking only once: a reverted agent has
        # proposed == cur, and a swap requires both parties to move, so
        # reverting can never create a new swap.
        moved = (proposed != cur).any(-1)
        if moved.any():
            swap = (
                (proposed[:, :, None, :] == cur[:, None, :, :]).all(-1)
                & (cur[:, :, None, :] == proposed[:, None, :, :]).all(-1)
            )
            swap &= ~self._eye_for(cur.shape[1])
            revert = swap.any(-1) & moved
            if revert.any():
                proposed = np.where(revert[..., None], cur, proposed)
                blocked |= revert

        # Vertex conflicts must iterate: reverting an agent can block another.
        for _ in range(self.n_agents + 1):
            moved = (proposed != cur).any(-1)
            if not moved.any():
                break
            cell = proposed[..., 0] * W + proposed[..., 1]
            order = np.argsort(cell, axis=1, kind="stable")
            srt = np.take_along_axis(cell, order, axis=1)
            eq = srt[:, 1:] == srt[:, :-1]
            if not eq.any():
                break
            dup = np.zeros_like(srt, dtype=bool)
            dup[:, 1:] |= eq
            dup[:, :-1] |= eq
            vertex = np.zeros_like(dup)
            np.put_along_axis(vertex, order, dup, axis=1)

            revert = vertex & moved
            if not revert.any():
                break
            proposed = np.where(revert[..., None], cur, proposed)
            blocked |= revert

        return proposed, blocked

    def _eye_for(self, n: int) -> np.ndarray:
        if self._eye.shape[1] != n:
            self._eye = np.eye(n, dtype=bool)[None]
        return self._eye

    def step(self, actions: np.ndarray):
        """Advance every environment by one timestep.

        Finished environments are reset automatically; ``info['final']`` holds
        their terminal statistics.
        """
        cfg = self.cfg
        actions = np.asarray(actions, dtype=np.int64).reshape(self.num_envs, self.n_agents)

        at_goal_before = (self.pos == self.goals).all(-1)
        old_pos = self.pos
        new_pos, blocked = self._resolve(actions)
        at_goal_after = (new_pos == self.goals).all(-1)

        idled = actions == STAY
        reward = np.where(idled, cfg.stay_penalty, cfg.step_penalty).astype(np.float32)
        reward[at_goal_after & idled] = 0.0                    # resting on goal is free
        reward += cfg.collision_penalty * blocked

        arrival = at_goal_after & ~at_goal_before
        if cfg.one_time_goal_reward:
            arrival &= ~self.reached
        reward += cfg.goal_reward * arrival
        reward += cfg.leave_goal_penalty * (at_goal_before & ~at_goal_after)
        self.reached |= at_goal_after

        if cfg.shaping_coef != 0.0:
            far = np.float32(self.height + self.width)
            phi_old = -self.dist[self._b1, self._n1, old_pos[..., 0], old_pos[..., 1]] / far
            phi_new = -self.dist[self._b1, self._n1, new_pos[..., 0], new_pos[..., 1]] / far
            reward += cfg.shaping_coef * (cfg.shaping_gamma * phi_new - phi_old)

        self.pos = new_pos
        self.last_blocked = blocked
        self.t += 1

        solved = at_goal_after.all(-1)
        reward += cfg.team_bonus * solved[:, None]
        timeout = self.t >= cfg.max_steps
        done = solved | timeout

        self._ep_return += reward.sum(-1)
        info: Dict[str, np.ndarray] = {
            "solved": solved,
            "timeout": timeout,
            "collisions": blocked.sum(-1),
            "agents_on_goal": at_goal_after.sum(-1),
            "makespan": self.t.copy(),
        }

        if done.any():
            d = np.flatnonzero(done)
            self.last_episode_stats = {
                "success_rate": float(solved[d].mean()),
                "agents_on_goal_frac": float(at_goal_after[d].sum(-1).mean() / self.n_agents),
                "makespan": float(self.t[d].mean()),
                "episode_return": float(self._ep_return[d].mean()),
            }
            info["final"] = dict(self.last_episode_stats)
            self._reset_envs(d)

        grid_obs, vec_obs = self.observe()
        return grid_obs, vec_obs, reward, done, info

    # ------------------------------------------------------------------ utils

    @property
    def obs_shape(self) -> Tuple[int, int, int]:
        return (N_OBS_CHANNELS, self.cfg.fov, self.cfg.fov)

    @property
    def vec_shape(self) -> int:
        return N_VEC_FEATURES

    @property
    def state_shape(self) -> int:
        return self.n_agents * 5

    def ascii(self, b: int = 0) -> str:
        """Render environment ``b`` as text. Lowercase letter = agent on goal."""
        grid = [["#" if self.obstacles[b, r, c] else "." for c in range(self.width)]
                for r in range(self.height)]
        for i in range(self.n_agents):
            gr, gc = self.goals[b, i]
            if grid[gr][gc] == ".":
                grid[gr][gc] = str(i % 10)
        for i in range(self.n_agents):
            r, c = self.pos[b, i]
            on_goal = bool((self.pos[b, i] == self.goals[b, i]).all())
            grid[r][c] = chr(ord("a" if on_goal else "A") + i % 26)
        return "\n".join("".join(row) for row in grid)
