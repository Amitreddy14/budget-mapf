"""Generate behaviour-cloning demonstrations by replaying expert plans.

Trajectories are stored as sequences, not shuffled transitions, so the same
dataset supports both a feed-forward and a recurrent policy. Shards are
written as compressed ``.npz``.

Storage layout per shard (E = episodes, T = horizon, N = agents):
    grid     (E, T, N, C, F, F)  float16   local FOV observations
    vec      (E, T, N, V)        float16   per-agent scalar features
    state    (E, T, S)           float16   centralised state, for the critic
    action   (E, T, N)           int8      expert action
    step_ok  (E, T)              bool      timestep is inside its episode
    at_goal  (E, T, N)           bool      agent already parked on its goal
    pos      (E, T, N, 2)        int16     agent grid positions

``pos`` is what lets a communication-aware learner rebuild the range-limited
peer graph offline. Without it, behaviour cloning cannot pretrain anything
that depends on who could hear whom.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from mapf.env import MAPFConfig, N_ACTIONS, VecMAPFEnv

from .prioritized import plans_to_action_matrix, prioritized_plan


def _plan_batch(env: VecMAPFEnv, rng: np.random.Generator, restarts: int):
    """Plan every environment in the batch from its current state."""
    plans: List[Optional[list]] = []
    for b in range(env.num_envs):
        plans.append(prioritized_plan(
            env.obstacles[b], env.pos[b], env.goals[b], env.dist[b],
            max_t=env.cfg.max_steps, rng=rng, restarts=restarts,
        ))
    return plans


def generate(
    cfg: MAPFConfig,
    target_samples: int,
    out_dir: str,
    seed: int = 0,
    num_envs: int = 32,
    restarts: int = 6,
    episodes_per_shard: int = 256,
    verbose: bool = True,
) -> Dict:
    """Collect ``target_samples`` agent-level expert decisions.

    One agent acting once is one sample, so an episode of ``T`` steps with
    ``N`` agents contributes ``T * N``.
    """
    os.makedirs(out_dir, exist_ok=True)
    env = VecMAPFEnv(cfg, num_envs=num_envs, seed=seed)
    rng = np.random.default_rng(seed)
    N = cfg.n_agents

    shard_id, total_samples, total_eps, plan_failures = 0, 0, 0, 0
    action_counts = np.zeros(N_ACTIONS, dtype=np.int64)
    buf: Dict[str, list] = {k: [] for k in
                            ("grid", "vec", "state", "action", "step_ok",
                             "at_goal", "pos")}
    t_start = time.perf_counter()

    def flush():
        nonlocal shard_id, buf
        if not buf["grid"]:
            return
        horizon = max(x.shape[0] for x in buf["grid"])
        out = {}
        for key, dtype in (("grid", np.float16), ("vec", np.float16),
                           ("state", np.float16), ("action", np.int8),
                           ("step_ok", bool), ("at_goal", bool),
                           ("pos", np.int16)):
            padded = []
            for arr in buf[key]:
                pad = horizon - arr.shape[0]
                if pad:
                    arr = np.concatenate(
                        [arr, np.zeros((pad,) + arr.shape[1:], dtype=arr.dtype)], axis=0)
                padded.append(arr.astype(dtype))
            out[key] = np.stack(padded)
        path = os.path.join(out_dir, f"shard_{shard_id:04d}.npz")
        np.savez_compressed(path, **out)
        if verbose:
            mb = os.path.getsize(path) / 1e6
            print(f"  wrote {os.path.basename(path)}  "
                  f"{out['grid'].shape[0]} episodes x T={horizon}  {mb:.1f} MB")
        shard_id += 1
        buf = {k: [] for k in buf}

    while total_samples < target_samples:
        env.reset()
        plans = _plan_batch(env, rng, restarts)
        solvable = np.array([p is not None for p in plans])
        plan_failures += int((~solvable).sum())
        if not solvable.any():
            continue

        horizon = max(max(len(p) for p in pl) for pl in plans if pl is not None)
        actions = np.zeros((horizon, env.num_envs, N), dtype=np.int64)
        for b, pl in enumerate(plans):
            if pl is not None:
                actions[:, b, :] = plans_to_action_matrix(pl, horizon=horizon)

        active = solvable.copy()
        rec: Dict[str, list] = {k: [] for k in buf}
        for t in range(horizon):
            grid, vec = env.observe()
            rec["grid"].append(grid)
            rec["vec"].append(vec)
            rec["state"].append(env.global_state())
            rec["action"].append(actions[t])
            rec["step_ok"].append(active.copy())
            rec["at_goal"].append((env.pos == env.goals).all(-1))
            rec["pos"].append(env.pos.copy())
            _, _, _, done, _ = env.step(actions[t])
            active &= ~done
            if not active.any():
                break

        # (T, B, ...) -> per-episode (T, ...) for the environments that solved.
        step_ok = np.stack(rec["step_ok"])                       # (T, B)
        for b in np.flatnonzero(solvable):
            length = int(step_ok[:, b].sum())
            if length == 0:
                continue
            buf["grid"].append(np.stack([x[b] for x in rec["grid"]])[:length])
            buf["vec"].append(np.stack([x[b] for x in rec["vec"]])[:length])
            buf["state"].append(np.stack([x[b] for x in rec["state"]])[:length])
            acts = np.stack([x[b] for x in rec["action"]])[:length]
            buf["action"].append(acts)
            buf["step_ok"].append(np.ones(length, dtype=bool))
            buf["at_goal"].append(np.stack([x[b] for x in rec["at_goal"]])[:length])
            buf["pos"].append(np.stack([x[b] for x in rec["pos"]])[:length])

            action_counts += np.bincount(acts.ravel(), minlength=N_ACTIONS)
            total_samples += length * N
            total_eps += 1

        if len(buf["grid"]) >= episodes_per_shard:
            flush()
        if verbose:
            print(f"  {total_samples:>9,} / {target_samples:,} samples "
                  f"({total_eps} episodes)", end="\r")

    flush()
    elapsed = time.perf_counter() - t_start

    meta = {
        "samples": int(total_samples),
        "episodes": int(total_eps),
        "shards": shard_id,
        "n_agents": N,
        "fov": cfg.fov,
        "map_spec": cfg.map_spec,
        "max_steps": cfg.max_steps,
        "observe_peer_goals": cfg.observe_peer_goals,
        "obs_costmap": cfg.obs_costmap,
        "seed": seed,
        "plan_failures": int(plan_failures),
        "action_distribution": {
            name: round(float(c) / max(action_counts.sum(), 1), 4)
            for name, c in zip(("STAY", "UP", "DOWN", "LEFT", "RIGHT"), action_counts)
        },
        "seconds": round(elapsed, 1),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    if verbose:
        print(f"\n  done in {elapsed:.1f}s -> {out_dir}")
    return meta
