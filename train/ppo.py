"""Recurrent MAPPO with CTDE and behaviour-cloning warm start.

Design notes
------------
* **CTDE.** The actor sees only its own FOV and recurrent state. The critic
  additionally receives the global state (`ModelConfig.state_dim > 0`). Extra
  information shapes learning but never reaches execution.
* **Recurrence and minibatching.** With a GRU, minibatches must keep time
  contiguous, so we split the *agent* axis and always run the full rollout
  length. The GRU state at the start of the rollout is stored per agent, and
  `resets` zeroes it at episode boundaries inside the window.
* **Warm start.** PPO from scratch on MAPF is slow; starting from the
  behaviour-cloned policy skips the phase where agents learn what a goal is.
  The BC checkpoint has no critic, so the value head starts untrained -- hence
  `value_warmup_iters`, which trains the critic alone before letting policy
  gradients flow. Without it the first updates chase meaningless advantages
  and can destroy the warm-started policy in a few hundred steps.

Known simplification
--------------------
Timeouts are treated as terminal rather than bootstrapped. Correct handling
needs the pre-reset observation, which the vectorised env discards on
auto-reset. The bias discourages long episodes, which happens to align with
the MAPF objective, but it is a bias. Revisit if value loss plateaus high.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from eval.rollout import evaluate_policy
from mapf.env import MAPFConfig, VecMAPFEnv
from models.encoder import ActorCritic, ModelConfig, count_parameters


@dataclass
class PPOConfig:
    # Environment
    n_agents: int = 16
    map_spec: dict = field(default_factory=lambda: {
        "kind": "random", "height": 32, "width": 32, "density": 0.20})
    max_steps: int = 128
    fov: int = 9
    num_envs: int = 32
    map_pool_size: int = 256

    # Rollout / optimisation
    rollout: int = 32
    total_agent_steps: int = 5_000_000
    ppo_epochs: int = 4
    minibatches: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    value_warmup_iters: int = 10

    # Warm start
    bc_checkpoint: Optional[str] = None
    bc_kl_coef: float = 0.0            # >0 anchors the policy to BC early on
    bc_kl_decay_iters: int = 500

    # Bookkeeping
    eval_every: int = 25
    eval_eps_per_env: int = 2
    log_every: int = 5
    out_dir: str = "runs/ppo"
    seed: int = 0
    device: str = "cuda"


def _load_bc_weights(model: ActorCritic, path: str, device) -> int:
    """Copy BC weights into the actor, skipping shape-mismatched tensors.

    The BC model has no centralised critic, so its value head will not match.
    We report how many tensors transferred so a silent no-op is visible.
    """
    blob = torch.load(path, map_location=device, weights_only=False)
    src = blob["model"]
    own = model.state_dict()
    copied = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
    own.update(copied)
    model.load_state_dict(own)
    return len(copied)


class Rollout:
    """Fixed-size storage for one PPO iteration."""

    def __init__(self, T: int, M: int, obs_shape, vec_dim: int, state_dim: int, device):
        self.grid = torch.zeros(T, M, *obs_shape, device=device)
        self.vec = torch.zeros(T, M, vec_dim, device=device)
        self.state = torch.zeros(T, M, state_dim, device=device)
        self.action = torch.zeros(T, M, dtype=torch.long, device=device)
        self.logp = torch.zeros(T, M, device=device)
        self.value = torch.zeros(T, M, device=device)
        self.reward = torch.zeros(T, M, device=device)
        self.done = torch.zeros(T, M, device=device)
        self.resets = torch.zeros(T, M, dtype=torch.bool, device=device)


def compute_gae(roll: Rollout, last_value: torch.Tensor, gamma: float, lam: float):
    T = roll.reward.shape[0]
    adv = torch.zeros_like(roll.reward)
    running = torch.zeros_like(last_value)
    next_value = last_value
    for t in reversed(range(T)):
        nonterminal = 1.0 - roll.done[t]
        delta = roll.reward[t] + gamma * next_value * nonterminal - roll.value[t]
        running = delta + gamma * lam * nonterminal * running
        adv[t] = running
        next_value = roll.value[t]
    return adv, adv + roll.value


def train_ppo(cfg: PPOConfig) -> Dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available())
                          else "cpu")
    os.makedirs(cfg.out_dir, exist_ok=True)

    env_cfg = MAPFConfig(n_agents=cfg.n_agents, fov=cfg.fov, max_steps=cfg.max_steps,
                         map_pool_size=cfg.map_pool_size, map_spec=cfg.map_spec)
    env = VecMAPFEnv(env_cfg, num_envs=cfg.num_envs, seed=cfg.seed)
    B, N = cfg.num_envs, cfg.n_agents
    M, T = B * N, cfg.rollout

    mcfg = ModelConfig(in_channels=env.obs_shape[0], fov=cfg.fov,
                       vec_dim=env.vec_shape, state_dim=env.state_shape)
    model = ActorCritic(mcfg).to(device)
    if cfg.bc_checkpoint:
        n = _load_bc_weights(model, cfg.bc_checkpoint, device)
        print(f"warm start: {n} tensors copied from {cfg.bc_checkpoint}")
    print(f"model: {count_parameters(model):,} parameters | device={device}")

    bc_ref = None
    if cfg.bc_kl_coef > 0 and cfg.bc_checkpoint:
        bc_ref = ActorCritic(mcfg).to(device)
        _load_bc_weights(bc_ref, cfg.bc_checkpoint, device)
        bc_ref.eval()
        for p in bc_ref.parameters():
            p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, eps=1e-5)
    roll = Rollout(T, M, env.obs_shape, env.vec_shape, env.state_shape, device)

    hidden = model.initial_state(M, device)
    prev_done = torch.zeros(M, dtype=torch.bool, device=device)
    grid_np, vec_np = env.observe()

    iters = max(1, cfg.total_agent_steps // (T * M))
    mb_size = max(1, M // cfg.minibatches)
    history, best, steps_done = [], -1.0, 0
    t_start = time.perf_counter()

    for it in range(iters):
        # ------------------------------------------------------------ collect
        model.eval()
        init_hidden = hidden.detach().clone()
        ep_returns, ep_solved, ep_on_goal = [], [], []
        with torch.no_grad():
            for t in range(T):
                g = torch.as_tensor(grid_np.reshape(M, *env.obs_shape), device=device)
                v = torch.as_tensor(vec_np.reshape(M, -1), device=device)
                s = torch.as_tensor(env.global_state(), device=device)
                s = s.unsqueeze(1).expand(B, N, -1).reshape(M, -1)

                roll.grid[t], roll.vec[t], roll.state[t] = g, v, s
                roll.resets[t] = prev_done

                hidden = hidden * (~prev_done).to(hidden.dtype).view(1, M, 1)
                action, logp, value, hidden = model.act_with_logp(g, v, hidden, state=s)
                roll.action[t], roll.logp[t], roll.value[t] = action, logp, value

                grid_np, vec_np, reward, done, info = env.step(
                    action.view(B, N).cpu().numpy())
                roll.reward[t] = torch.as_tensor(reward.reshape(M), device=device)
                done_agents = np.repeat(done, N)
                roll.done[t] = torch.as_tensor(done_agents.astype(np.float32), device=device)
                prev_done = torch.as_tensor(done_agents, device=device)

                if "final" in info:
                    ep_returns.append(info["final"]["episode_return"])
                    ep_solved.append(info["final"]["success_rate"])
                    ep_on_goal.append(info["final"]["agents_on_goal_frac"])

            g = torch.as_tensor(grid_np.reshape(M, *env.obs_shape), device=device)
            v = torch.as_tensor(vec_np.reshape(M, -1), device=device)
            s = torch.as_tensor(env.global_state(), device=device)
            s = s.unsqueeze(1).expand(B, N, -1).reshape(M, -1)
            h_boot = hidden * (~prev_done).to(hidden.dtype).view(1, M, 1)
            _, last_value, _ = model.act(g, v, h_boot, state=s)

        adv, returns = compute_gae(roll, last_value, cfg.gamma, cfg.gae_lambda)
        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
        steps_done += T * M

        # ------------------------------------------------------------- update
        model.train()
        critic_only = it < cfg.value_warmup_iters
        kl_coef = cfg.bc_kl_coef * max(0.0, 1.0 - it / max(cfg.bc_kl_decay_iters, 1))
        stats = {"pg": [], "vf": [], "ent": [], "kl": [], "clipfrac": [], "bc_kl": []}

        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(M, device=device)
            for start in range(0, M, mb_size):
                idx = perm[start:start + mb_size]
                logits, values, _ = model(
                    roll.grid[:, idx], roll.vec[:, idx],
                    hidden=init_hidden[:, idx], resets=roll.resets[:, idx],
                    state=roll.state[:, idx])
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(roll.action[:, idx])
                ratio = (new_logp - roll.logp[:, idx]).exp()

                a = adv_norm[:, idx]
                pg = -torch.min(ratio * a,
                                ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                vf = F.mse_loss(values, returns[:, idx])
                ent = dist.entropy().mean()

                loss = cfg.value_coef * vf
                if not critic_only:
                    loss = loss + pg - cfg.entropy_coef * ent
                    if bc_ref is not None and kl_coef > 0:
                        with torch.no_grad():
                            ref_logits, _, _ = bc_ref(
                                roll.grid[:, idx], roll.vec[:, idx],
                                hidden=init_hidden[:, idx] * 0,
                                resets=roll.resets[:, idx])
                        bc_kl = F.kl_div(F.log_softmax(logits, -1),
                                         F.log_softmax(ref_logits, -1),
                                         log_target=True, reduction="batchmean")
                        loss = loss + kl_coef * bc_kl
                        stats["bc_kl"].append(bc_kl.item())

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()

                with torch.no_grad():
                    stats["pg"].append(pg.item())
                    stats["vf"].append(vf.item())
                    stats["ent"].append(ent.item())
                    stats["kl"].append((roll.logp[:, idx] - new_logp).mean().item())
                    stats["clipfrac"].append(
                        ((ratio - 1).abs() > cfg.clip).float().mean().item())

        hidden = hidden.detach()

        # -------------------------------------------------------------- log
        if it % cfg.log_every == 0 or it == iters - 1:
            row = {
                "iter": it, "agent_steps": steps_done,
                "critic_only": critic_only,
                "pg_loss": float(np.mean(stats["pg"])) if stats["pg"] else 0.0,
                "vf_loss": float(np.mean(stats["vf"])),
                "entropy": float(np.mean(stats["ent"])),
                "approx_kl": float(np.mean(stats["kl"])),
                "clipfrac": float(np.mean(stats["clipfrac"])),
                "train_return": float(np.mean(ep_returns)) if ep_returns else float("nan"),
                "train_success": float(np.mean(ep_solved)) if ep_solved else float("nan"),
                "train_on_goal": float(np.mean(ep_on_goal)) if ep_on_goal else float("nan"),
                "sps": int(steps_done / (time.perf_counter() - t_start)),
            }
            if it % cfg.eval_every == 0 or it == iters - 1:
                r = evaluate_policy(model, env_cfg, device,
                                    episodes_per_env=cfg.eval_eps_per_env)
                row.update({"eval_success": r["success_rate"],
                            "eval_on_goal": r["agents_on_goal"],
                            "eval_coll": r["collisions_per_step"]})
                if r["agents_on_goal"] > best:
                    best = r["agents_on_goal"]
                    torch.save({"model": model.state_dict(), "model_cfg": asdict(mcfg),
                                "ppo_cfg": asdict(cfg), "iter": it},
                               os.path.join(cfg.out_dir, "best.pt"))
            history.append(row)
            print("  " + "  ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in row.items()))
            with open(os.path.join(cfg.out_dir, "history.json"), "w") as fh:
                json.dump(history, fh, indent=2)

    torch.save({"model": model.state_dict(), "model_cfg": asdict(mcfg),
                "ppo_cfg": asdict(cfg)}, os.path.join(cfg.out_dir, "last.pt"))
    return {"history": history, "best_on_goal": best}
