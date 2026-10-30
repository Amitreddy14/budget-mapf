"""Run a trained policy in the environment and collect metrics."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from eval.metrics import run_episodes
from mapf.env import MAPFConfig, VecMAPFEnv


@torch.no_grad()
def evaluate_policy(
    model,
    cfg: MAPFConfig,
    device,
    episodes_per_env: int = 4,
    num_envs: int = 64,
    seed: int = 0,
    pool_seed: int = 999,
    greedy: bool = False,
) -> Dict[str, float]:
    """Roll the policy out on held-out maps, using the unbiased harness.

    ``greedy=False`` samples from the policy. For MAPF that usually beats
    argmax: two deterministic agents can insist on the same cell forever,
    whereas sampling breaks the tie. Both are reported at evaluation time.
    """
    was_training = model.training
    model.eval()
    env = VecMAPFEnv(cfg, num_envs=num_envs, seed=seed, pool_seed=pool_seed)
    N = cfg.n_agents
    M = num_envs * N
    state = {"hidden": model.initial_state(M, device)}

    def act(e: VecMAPFEnv) -> np.ndarray:
        grid, vec = e.observe()
        g = torch.as_tensor(grid.reshape(M, *grid.shape[2:]), device=device)
        v = torch.as_tensor(vec.reshape(M, -1), device=device)
        action, _, state["hidden"] = model.act(g, v, state["hidden"], greedy=greedy)
        return action.view(num_envs, N).cpu().numpy()

    def on_end(finished: np.ndarray) -> None:
        reset = torch.zeros(M, dtype=torch.bool, device=device)
        for b in finished:
            reset[b * N:(b + 1) * N] = True
        state["hidden"] = state["hidden"] * (~reset).to(state["hidden"].dtype).view(1, M, 1)

    out = run_episodes(env, act, episodes_per_env=episodes_per_env, on_episode_end=on_end)
    if was_training:
        model.train()
    return out


@torch.no_grad()
def evaluate_comm_policy(
    model,
    cfg: MAPFConfig,
    device,
    radius: float = 8.0,
    episodes_per_env: int = 4,
    num_envs: int = 64,
    seed: int = 0,
    pool_seed: int = 999,
    greedy: bool = False,
) -> Dict[str, float]:
    """Evaluate a CommActorCritic, driving the range-limited peer graph."""
    was_training = model.training
    model.eval()
    env = VecMAPFEnv(cfg, num_envs=num_envs, seed=seed, pool_seed=pool_seed)
    N, B = cfg.n_agents, num_envs
    st = {"h": model.initial_state(B * N, device), "peers": []}

    def act(e: VecMAPFEnv) -> np.ndarray:
        grid, vec = e.observe()
        adj, rel = e.comm_graph(radius)
        action, _, _, st["h"], _ = model.act(
            torch.as_tensor(grid, device=device),
            torch.as_tensor(vec, device=device),
            torch.as_tensor(adj, device=device),
            torch.as_tensor(rel, device=device),
            hidden=st["h"], greedy=greedy)
        st["peers"].append(float(adj.sum(-1).mean()))
        return action.cpu().numpy()

    def on_end(finished: np.ndarray) -> None:
        reset = torch.zeros(B * N, dtype=torch.bool, device=device)
        for b in finished:
            reset[b * N:(b + 1) * N] = True
        st["h"] = st["h"] * (~reset).to(st["h"].dtype).view(1, B * N, 1)

    out = run_episodes(env, act, episodes_per_env=episodes_per_env,
                       on_episode_end=on_end)
    out["peers_in_range"] = float(np.mean(st["peers"])) if st["peers"] else 0.0
    if was_training:
        model.train()
    return out
