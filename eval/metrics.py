"""Unbiased episode collection.

The obvious harness -- step until `episodes` have completed, then take the
first `episodes` records -- is **biased toward success**. Solved episodes
terminate as soon as the last agent arrives, while failures run to the step
limit and complete last, so truncating the list preferentially discards
failures. Measured inflation on a 16-agent config: 81.2% vs a true 58.2%.

The fix: every environment contributes exactly `episodes_per_env` episodes.
Each env's quota is independent of how long its episodes take, so slow and
fast episodes are represented in their true proportion.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from mapf.env import VecMAPFEnv


def run_episodes(
    env: VecMAPFEnv,
    act_fn: Callable[[VecMAPFEnv], np.ndarray],
    episodes_per_env: int = 4,
    on_episode_end: Optional[Callable[[np.ndarray], None]] = None,
    max_steps_guard: int = 100_000,
) -> Dict[str, float]:
    """Roll out until every environment has finished its quota of episodes."""
    counts = np.zeros(env.num_envs, dtype=np.int64)
    solved: List[float] = []
    on_goal: List[float] = []
    makespan: List[float] = []
    collisions: List[float] = []

    steps = 0
    while counts.min() < episodes_per_env and steps < max_steps_guard:
        steps += 1
        _, _, _, done, info = env.step(act_fn(env))
        collisions.append(float(info["collisions"].mean()))
        if done.any():
            finished = np.flatnonzero(done)
            keep = finished[counts[finished] < episodes_per_env]
            if len(keep):
                solved.extend(info["solved"][keep].tolist())
                on_goal.extend((info["agents_on_goal"][keep] / env.n_agents).tolist())
                makespan.extend(info["makespan"][keep].tolist())
            counts[finished] += 1
            if on_episode_end is not None:
                on_episode_end(finished)

    return {
        "success_rate": float(np.mean(solved)) if solved else 0.0,
        "agents_on_goal": float(np.mean(on_goal)) if on_goal else 0.0,
        "makespan": float(np.mean(makespan)) if makespan else float("nan"),
        "collisions_per_step": float(np.mean(collisions)) if collisions else float("nan"),
        "episodes": len(solved),
    }
