"""MAPPO with intent communication.

Differs from `train/ppo.py` in two structural ways:

* **Shapes.** Communication couples agents inside an environment, so rollouts
  are stored as ``(T, B, N, ...)`` rather than flattening to ``M = B * N``.
* **Minibatching splits environments, not agents.** All N agents of an
  environment must stay together, or the attention step sees a partial graph
  and the gradient is wrong. This is the one place where the obvious
  refactor of `ppo.py` would silently corrupt training.

`comm_mode` ("none" | "full" | "learned") selects the ablation at runtime on
a single architecture, so the no-communication baseline has exactly the same
parameter count. See `models/comm_policy.py` for why that matters.
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

from eval.metrics import run_episodes
from mapf.env import MAPFConfig, VecMAPFEnv
from models.comm_policy import CommActorCritic, CommConfig
from models.encoder import ModelConfig
from models.voi import (policy_shift, scheduler_loss, sender_value,
                        voi_summary, value_of_information)


@dataclass
class CommPPOConfig:
    # Environment
    n_agents: int = 16
    map_spec: dict = field(default_factory=lambda: {
        "kind": "random", "height": 32, "width": 32, "density": 0.20})
    max_steps: int = 128
    fov: int = 9
    num_envs: int = 32
    map_pool_size: int = 256
    observe_peer_goals: bool = True   # must match the warm-start dataset
    obs_costmap: str = "field"        # must match the warm-start dataset
    allow_env_mismatch: bool = False  # training env may differ from the checkpoint

    # Communication
    comm_mode: str = "full"
    comm_radius: float = 8.0
    msg_dim: int = 64
    n_heads: int = 4

    # Optimisation
    rollout: int = 32
    total_agent_steps: int = 5_000_000
    ppo_epochs: int = 4
    minibatches: int = 4            # splits the ENV axis
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    value_warmup_iters: int = 10

    # Keep broadcast intent informative.
    #
    # Behaviour cloning trains the tentative head to predict the expert action
    # (entropy ~0.24). During PPO nothing supervises it, and it drifts to
    # ~1.21 against a maximum of ln(5)=1.609 -- i.e. the messages decay into
    # near-uniform noise within ~200 iterations. A communication ablation run
    # that way cannot distinguish "messages are useless" from "messages were
    # allowed to become meaningless". This term keeps the tentative head
    # predictive of the action the agent actually takes, which is exactly what
    # a receiver needs to know. Watch `tentative_entropy` to confirm it holds.
    tentative_coef: float = 0.3

    # Scheduling (comm_mode="learned").
    budget_frac: float = 0.25     # share of agents permitted to transmit
    allocation: str = "rate"      # rate (adaptive, Lagrangian) | topk (strict)
    voi_coef: float = 0.5         # weight on the value-of-information loss
    voi_every: int = 2            # compute VoI every k minibatches (it costs a forward)
    lambda_lr: float = 0.05

    warm_start: Optional[str] = None
    eval_every: int = 25
    # 2 eps/env (128 episodes at num_envs=64) gives ~4pp of binomial noise on
    # success and swings of 0.20-0.30 between consecutive evaluations of one
    # converged run -- enough to invent effects that are not there. 8 raises
    # each evaluation to ~512 episodes at negligible cost against 20M steps.
    eval_eps_per_env: int = 8
    log_every: int = 5
    out_dir: str = "runs/ppo_comm"
    seed: int = 0
    device: str = "cuda"


# Modules whose weights must transfer for a warm start to mean anything.
# The encoder alone is not enough: with a random policy head the model is a
# uniformly random policy that happens to have good features, which trains no
# better than from scratch.

def _keep_only_value_grads(model: torch.nn.Module) -> int:
    """Drop gradients on everything but the value head. Returns #dropped.

    The value head reads the shared encoder/GRU trunk, so a "critic only"
    warmup that merely omits the policy loss still back-propagates the value
    loss into that trunk and rewrites the representation a behaviour-cloned
    policy depends on.
    """
    dropped = 0
    for name, prm in model.named_parameters():
        if not name.startswith("value") and prm.grad is not None:
            prm.grad = None
            dropped += 1
    return dropped


CRITICAL_MODULES = ("encoder", "gru", "policy")


def load_partial(model: torch.nn.Module, path: str, device) -> Dict:
    """Copy tensors that match by name and shape, and report per-module coverage.

    A silent partial load is the most expensive failure mode in this codebase.
    A `bc.py` checkpoint transfers its encoder into `CommActorCritic` but not
    its policy head (`policy.weight` vs `policy.0.weight`), leaving the head at
    random initialisation. The loader used to report "19 tensors" and nothing
    else; 20M steps across six runs produced ~0% agents-on-goal before anyone
    noticed. This version refuses to be quiet about it.
    """
    blob = torch.load(path, map_location=device, weights_only=False)
    src = blob["model"]
    own = model.state_dict()
    copied = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
    own.update(copied)
    model.load_state_dict(own)

    coverage, missing = {}, []
    for name in sorted({k.split(".")[0] for k in model.state_dict()}):
        keys = [k for k in model.state_dict() if k.split(".")[0] == name]
        got = sum(1 for k in keys if k in copied)
        coverage[name] = (got, len(keys))
        if name in CRITICAL_MODULES and got < len(keys):
            missing.append(name)

    print(f"warm start from {path}")
    for name, (got, total) in coverage.items():
        flag = "  <-- INCOMPLETE" if (name in CRITICAL_MODULES and got < total) else ""
        print(f"    {name:<12} {got:>3}/{total:<3} tensors{flag}")

    if missing:
        raise RuntimeError(
            f"warm start did not populate critical module(s): {', '.join(missing)}.\n"
            f"  The checkpoint architecture does not match this model. Training "
            f"from here is equivalent to training from scratch.\n"
            f"  For CommActorCritic, produce the checkpoint with "
            f"scripts/run_bc_comm.py, not scripts/run_bc.py.")

    return {"copied": len(copied), "coverage": coverage,
            "reported_on_goal": blob.get("env_on_goal"),
            "observe_peer_goals": blob.get("observe_peer_goals"),
            "obs_costmap": blob.get("obs_costmap"),
            "n_agents": blob.get("n_agents")}


class CommRollout:
    def __init__(self, T, B, N, obs_shape, vec_dim, state_dim, device):
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=device)
        self.grid = z(T, B, N, *obs_shape)
        self.vec = z(T, B, N, vec_dim)
        self.state = z(T, B, N, state_dim)
        self.adj = z(T, B, N, N, dt=torch.bool)
        self.rel = z(T, B, N, N, 2)
        self.action = z(T, B, N, dt=torch.long)
        self.logp = z(T, B, N)
        self.value = z(T, B, N)
        self.reward = z(T, B, N)
        self.done = z(T, B, N)
        self.resets = z(T, B, N, dt=torch.bool)


def compute_gae(roll, last_value, gamma, lam):
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


def _eval(model, env_cfg, comm_cfg, device, eps_per_env, seed=0, num_envs=32):
    """Evaluate with the unbiased harness, driving the comm policy."""
    model.eval()
    env = VecMAPFEnv(env_cfg, num_envs=num_envs, seed=seed, pool_seed=999)
    N, B = env_cfg.n_agents, num_envs
    st = {"h": model.initial_state(B * N, device), "sent": []}

    def act(e):
        grid, vec = e.observe()
        adj_np, rel_np = e.comm_graph(comm_cfg.radius)
        with torch.no_grad():
            action, _, _, st["h"], _ = model.act(
                torch.as_tensor(grid, device=device),
                torch.as_tensor(vec, device=device),
                torch.as_tensor(adj_np, device=device),
                torch.as_tensor(rel_np, device=device),
                hidden=st["h"])
        st["sent"].append(float(adj_np.sum(-1).mean()))
        return action.cpu().numpy()

    def on_end(finished):
        reset = torch.zeros(B * N, dtype=torch.bool, device=device)
        for b in finished:
            reset[b * N:(b + 1) * N] = True
        st["h"] = st["h"] * (~reset).to(st["h"].dtype).view(1, B * N, 1)

    out = run_episodes(env, act, episodes_per_env=eps_per_env, on_episode_end=on_end)
    out["peers_in_range"] = float(np.mean(st["sent"])) if st["sent"] else 0.0
    model.train()
    return out


def train_comm_ppo(cfg: CommPPOConfig) -> Dict:
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available())
                          else "cpu")
    os.makedirs(cfg.out_dir, exist_ok=True)

    # The checkpoint records the observability it was trained under. Adopt it,
    # because a policy cloned without peer goals evaluated in an environment
    # that shows them (or the reverse) silently degrades, and nothing in the
    # logs would say why.
    observe_peer_goals = cfg.observe_peer_goals
    obs_costmap = cfg.obs_costmap
    max_steps = cfg.max_steps
    if cfg.warm_start:
        meta = torch.load(cfg.warm_start, map_location="cpu", weights_only=False)
        ck_obs = meta.get("observe_peer_goals")
        if ck_obs is not None and ck_obs != observe_peer_goals:
            print(f"adopting observe_peer_goals={ck_obs} from the warm-start "
                  f"checkpoint (config said {observe_peer_goals})")
            observe_peer_goals = ck_obs

        ck_cm = meta.get("obs_costmap")
        if ck_cm is not None and ck_cm != obs_costmap:
            print(f"adopting obs_costmap={ck_cm!r} from the warm-start "
                  f"checkpoint (config said {obs_costmap!r})")
            obs_costmap = ck_cm

        # Any difference between the environment a checkpoint was trained in
        # and the one it is fine-tuned in silently changes what the numbers
        # mean. Two ablation scripts once carried different densities for the
        # same preset (0.20 vs 0.25), which made their results look like a
        # failed replication when they were simply different experiments.
        ck_map = meta.get("map_spec")
        if ck_map:
            diffs = {k: (ck_map.get(k), cfg.map_spec.get(k))
                     for k in set(ck_map) | set(cfg.map_spec)
                     if ck_map.get(k) != cfg.map_spec.get(k)}
            if diffs:
                print("  WARNING: environment differs from the warm-start checkpoint:")
                for k, (ck, now) in sorted(diffs.items()):
                    print(f"    {k}: checkpoint={ck!r}  this run={now!r}")
                print("    Results will not be comparable to runs that match the "
                      "checkpoint. Pass --allow-env-mismatch only if deliberate.")
                if not cfg.allow_env_mismatch:
                    raise RuntimeError(
                        "environment does not match the warm-start checkpoint; "
                        "fix the preset or pass allow_env_mismatch=True")
        ck_agents = meta.get("n_agents")
        if ck_agents is not None and ck_agents != cfg.n_agents:
            print(f"  WARNING: checkpoint trained with {ck_agents} agents, "
                  f"this run uses {cfg.n_agents}")

        # Episode length changes how many instances time out, which moves
        # success rate without any change in policy quality. The demos were
        # generated at 192 while training defaulted to 128.
        ck_steps = meta.get("max_steps")
        if ck_steps is not None and ck_steps != max_steps:
            print(f"adopting max_steps={ck_steps} from the warm-start checkpoint "
                  f"(config said {max_steps})")
            max_steps = ck_steps
        del meta

    env_cfg = MAPFConfig(n_agents=cfg.n_agents, fov=cfg.fov, max_steps=max_steps,
                         map_pool_size=cfg.map_pool_size, map_spec=cfg.map_spec,
                         observe_peer_goals=observe_peer_goals,
                         obs_costmap=obs_costmap)
    env = VecMAPFEnv(env_cfg, num_envs=cfg.num_envs, seed=cfg.seed)
    B, N, T = cfg.num_envs, cfg.n_agents, cfg.rollout
    M = B * N

    mcfg = ModelConfig(in_channels=env.obs_shape[0], fov=cfg.fov,
                       vec_dim=env.vec_shape, state_dim=env.state_shape)
    ccfg = CommConfig(mode=cfg.comm_mode, radius=cfg.comm_radius,
                      msg_dim=cfg.msg_dim, n_heads=cfg.n_heads,
                      budget_frac=cfg.budget_frac, allocation=cfg.allocation,
                      lambda_lr=cfg.lambda_lr)
    model = CommActorCritic(mcfg, ccfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} parameters | comm={cfg.comm_mode} "
          f"r={cfg.comm_radius} | device={device}")

    if cfg.warm_start:
        info = load_partial(model, cfg.warm_start, device)

        # A checkpoint trained with comm="none" has `ctx = zeros` throughout,
        # so IntentComm received no gradient and the policy head learned a
        # function of zeros. Fine-tuning a communicating arm from it hands the
        # policy an untrained attention module, and the cheapest repair
        # available to gradient descent is to learn to ignore it. Every Phase 6
        # run was warm-started this way and every one reported voi_mean ~ 7e-4.
        ck_mode = (torch.load(cfg.warm_start, map_location="cpu", weights_only=False)
                   .get("comm_cfg", {}) or {}).get("mode")
        if ck_mode == "none" and cfg.comm_mode not in ("none",):
            print(f"\n  WARNING: checkpoint was trained with comm='none' but this "
                  f"run is comm='{cfg.comm_mode}'.\n"
                  f"  The attention module starts from random initialisation and "
                  f"the policy head\n  has never seen a non-zero context. Expect "
                  f"voi_mean to collapse toward zero.\n"
                  f"  Warm-start from a comm='full' checkpoint to fine-tune a "
                  f"policy that already\n  reads messages. See "
                  f"docs/FINDINGS-phase6b.md.\n")

        # Pre-flight: a warm start that does not reproduce the checkpoint's own
        # recorded performance is broken, and it is far cheaper to find out now
        # than after 20M steps. Checked before a single gradient is taken.
        probe = _eval(model, env_cfg, ccfg, device, eps_per_env=2)
        expected = info.get("reported_on_goal")
        print(f"  warm-start check: agents-on-goal {probe['agents_on_goal']:.3f}"
              + (f" (checkpoint recorded {expected:.3f})" if expected else ""))
        if expected is not None and probe["agents_on_goal"] < expected - 0.25:
            raise RuntimeError(
                f"warm start is not behaving like its checkpoint: "
                f"{probe['agents_on_goal']:.3f} here vs {expected:.3f} recorded.\n"
                f"  Usually an architecture or observability mismatch -- check "
                f"that the demo dataset's observe_peer_goals matches this run.")
        if expected is None and probe["agents_on_goal"] < 0.10:
            print("  WARNING: warm-started policy reaches <10% agents-on-goal. "
                  "Training will likely behave as if from scratch.")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, eps=1e-5)
    roll = CommRollout(T, B, N, env.obs_shape, env.vec_shape, env.state_shape, device)

    hidden = model.initial_state(M, device)
    prev_done = torch.zeros(B, N, dtype=torch.bool, device=device)
    grid_np, vec_np = env.observe()

    iters = max(1, cfg.total_agent_steps // (T * M))
    env_mb = max(1, B // cfg.minibatches)
    history, best, steps = [], -1.0, 0
    t0_all = time.perf_counter()

    for it in range(iters):
        model.eval()
        init_hidden = hidden.detach().clone()
        ep_solved, ep_on_goal = [], []

        with torch.no_grad():
            for t in range(T):
                adj_np, rel_np = env.comm_graph(cfg.comm_radius)
                g = torch.as_tensor(grid_np, device=device)
                v = torch.as_tensor(vec_np, device=device)
                s = torch.as_tensor(env.global_state(), device=device)
                s = s.unsqueeze(1).expand(B, N, -1)

                roll.grid[t], roll.vec[t], roll.state[t] = g, v, s
                roll.adj[t] = torch.as_tensor(adj_np, device=device)
                roll.rel[t] = torch.as_tensor(rel_np, device=device)
                roll.resets[t] = prev_done

                hidden = hidden * (~prev_done.reshape(M)).to(hidden.dtype).view(1, M, 1)
                action, logp, value, hidden, _ = model.act(
                    g, v, roll.adj[t], roll.rel[t], hidden=hidden, state=s)
                roll.action[t], roll.logp[t], roll.value[t] = action, logp, value

                grid_np, vec_np, reward, done, info = env.step(action.cpu().numpy())
                roll.reward[t] = torch.as_tensor(reward, device=device)
                d = torch.as_tensor(done, device=device)[:, None].expand(B, N)
                roll.done[t] = d.to(torch.float32)
                prev_done = d
                if "final" in info:
                    ep_solved.append(info["final"]["success_rate"])
                    ep_on_goal.append(info["final"]["agents_on_goal_frac"])

            adj_np, rel_np = env.comm_graph(cfg.comm_radius)
            s = torch.as_tensor(env.global_state(), device=device).unsqueeze(1).expand(B, N, -1)
            h_boot = hidden * (~prev_done.reshape(M)).to(hidden.dtype).view(1, M, 1)
            _, _, last_value, _, _ = model.act(
                torch.as_tensor(grid_np, device=device),
                torch.as_tensor(vec_np, device=device),
                torch.as_tensor(adj_np, device=device),
                torch.as_tensor(rel_np, device=device), hidden=h_boot, state=s)

        adv, returns = compute_gae(roll, last_value, cfg.gamma, cfg.gae_lambda)
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        steps += T * M

        model.train()
        critic_only = it < cfg.value_warmup_iters
        stats = {k: [] for k in ("pg", "vf", "ent", "tent_ent", "tent_loss",
                                 "kl", "clipfrac", "voi_loss", "voi_mean",
                                 "voi_value",
                                 "voi_captured", "comm_rate")}
        mb_counter = 0
        h4 = init_hidden.view(-1, B, N, mcfg.hidden)

        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(B, device=device)
            for start in range(0, B, env_mb):
                e = perm[start:start + env_mb]
                mb = len(e)
                h0 = h4[:, e].reshape(-1, mb * N, mcfg.hidden)
                logits, values, _, tent, attn, sched = model(
                    roll.grid[:, e], roll.vec[:, e], roll.adj[:, e], roll.rel[:, e],
                    hidden=h0, resets=roll.resets[:, e].reshape(T, mb * N),
                    state=roll.state[:, e])


                def _voi_value(silent_logits):
                    """Advantage-weighted: did the message help, or just shift?"""
                    act = roll.action[:, e].unsqueeze(-1)
                    lw = torch.log_softmax(logits.detach(), -1).gather(-1, act)
                    lo = torch.log_softmax(silent_logits, -1).gather(-1, act)
                    return float(value_of_information(
                        lw.squeeze(-1), lo.squeeze(-1), a).mean())

                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(roll.action[:, e])
                ratio = (new_logp - roll.logp[:, e]).exp()
                a = adv_n[:, e]
                pg = -torch.min(ratio * a,
                                ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                vf = F.mse_loss(values, returns[:, e])
                ent = dist.entropy().mean()

                # Auxiliary: broadcast intent should predict this agent's own
                # action, so a message stays informative to whoever hears it.
                tent_loss = F.cross_entropy(
                    tent.reshape(-1, tent.shape[-1]),
                    roll.action[:, e].reshape(-1))

                loss = cfg.value_coef * vf
                if not critic_only:
                    loss = loss + pg - cfg.entropy_coef * ent
                    if cfg.tentative_coef > 0:
                        loss = loss + cfg.tentative_coef * tent_loss

                    if sched is not None:
                        # The budget is enforced by a dual variable rather than
                        # a hand-tuned penalty traded off against task reward.
                        loss = loss + model.scheduler.budget_penalty(sched["send"])
                        stats["comm_rate"].append(float(sched["rate"]))

                        mb_counter += 1
                        if cfg.voi_coef > 0 and mb_counter % cfg.voi_every == 0:
                            # Counterfactual: what would these agents have done
                            # in silence? The shift is the value the messages
                            # actually delivered, and it is a dense signal --
                            # unlike task reward, which told v1's gate nothing.
                            with torch.no_grad():
                                silent, _, _, _, _, _ = model(
                                    roll.grid[:, e], roll.vec[:, e],
                                    roll.adj[:, e], roll.rel[:, e],
                                    hidden=h0, resets=roll.resets[:, e].reshape(T, mb * N),
                                    state=roll.state[:, e],
                                    send_mask=torch.zeros_like(sched["send"]))
                                voi = policy_shift(logits.detach(), silent)
                                val = sender_value(voi.reshape(-1, N), attn.detach())
                            sl = scheduler_loss(
                                sched["logits"].reshape(-1, N), val)
                            loss = loss + cfg.voi_coef * sl
                            with torch.no_grad():
                                stats["voi_loss"].append(sl.item())
                                summ = voi_summary(voi.reshape(-1, N),
                                                   sched["send"].reshape(-1, N))
                                stats["voi_mean"].append(summ["voi_mean"])
                                stats["voi_captured"].append(summ["voi_captured"])
                                stats["voi_value"].append(_voi_value(silent))

                # VoI is the instrument this project is built on, but it was
                # computed only inside the scheduler branch, so a comm="full"
                # run logged nothing -- the arm where "do messages change
                # behaviour?" has the cleanest answer, since every agent
                # transmits. The first repair attached this to `critic_only`
                # as an `elif`, which measured it during warmup and never
                # again: two readings per run, both before training. It is a
                # separate branch, guarded on the absence of a scheduler so it
                # cannot double-count. No loss term is added here, only the
                # number.
                if sched is None and ccfg.mode != "none" and cfg.voi_every > 0:
                    mb_counter += 1
                    if mb_counter % cfg.voi_every == 0:
                        with torch.no_grad():
                            silent, _, _, _, _, _ = model(
                                roll.grid[:, e], roll.vec[:, e],
                                roll.adj[:, e], roll.rel[:, e],
                                hidden=h0,
                                resets=roll.resets[:, e].reshape(T, mb * N),
                                state=roll.state[:, e],
                                send_mask=torch.zeros(T, mb, N, dtype=torch.bool,
                                                      device=device))
                            stats["voi_mean"].append(float(
                                policy_shift(logits.detach(), silent).mean()))
                            stats["voi_value"].append(_voi_value(silent))

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if critic_only:
                    # "critic only" excluded the policy loss but still sent the
                    # value loss through the SHARED encoder and GRU, so ten
                    # warmup iterations rewrote the trunk a behaviour-cloned
                    # policy depends on. Observed on a warm start that checked
                    # out at 0.925 agents-on-goal: by iteration 5 -- before any
                    # policy gradient was applied -- tentative_entropy had gone
                    # 0.404 -> 1.078 and on-goal was collapsing. Warm up the
                    # value head alone.
                    _keep_only_value_grads(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()

                with torch.no_grad():
                    stats["pg"].append(pg.item()); stats["vf"].append(vf.item())
                    stats["ent"].append(ent.item())
                    # Watch for intent collapse: if the tentative head becomes
                    # deterministic or constant, messages carry no information.
                    stats["tent_ent"].append(
                        torch.distributions.Categorical(logits=tent).entropy().mean().item())
                    stats["tent_loss"].append(tent_loss.item())
                    stats["kl"].append((roll.logp[:, e] - new_logp).mean().item())
                    stats["clipfrac"].append(
                        ((ratio - 1).abs() > cfg.clip).float().mean().item())

        hidden = hidden.detach()
        if stats["comm_rate"]:
            # Dual ascent once per iteration, on the realised rate.
            model.scheduler.dual_step(float(np.mean(stats["comm_rate"])))

        if it % cfg.log_every == 0 or it == iters - 1:
            row = {"iter": it, "agent_steps": steps, "critic_only": critic_only,
                   "pg_loss": float(np.mean(stats["pg"])),
                   "vf_loss": float(np.mean(stats["vf"])),
                   "entropy": float(np.mean(stats["ent"])),
                   "tentative_entropy": float(np.mean(stats["tent_ent"])),
                   "tentative_loss": float(np.mean(stats["tent_loss"])),
                   "approx_kl": float(np.mean(stats["kl"])),
                   "clipfrac": float(np.mean(stats["clipfrac"])),
                   "train_success": float(np.mean(ep_solved)) if ep_solved else float("nan"),
                   "train_on_goal": float(np.mean(ep_on_goal)) if ep_on_goal else float("nan"),
                   "sps": int(steps / (time.perf_counter() - t0_all))}
            for k, label in (("comm_rate", "comm_rate"), ("voi_loss", "voi_loss"),
                             ("voi_mean", "voi_mean"),
                             ("voi_value", "voi_value"),
                             ("voi_captured", "voi_captured")):
                if stats[k]:
                    row[label] = float(np.mean(stats[k]))
            if cfg.comm_mode == "learned":
                row["lambda"] = float(model.scheduler.lam)
                # The headline diagnostic. Selecting agents at random captures
                # a share of total value equal to the comm rate, so this ratio
                # is 1.0 for chance. Below 1.0 means the scheduler is actively
                # choosing worse than random and something is wrong.
                if stats["voi_captured"] and stats["comm_rate"]:
                    rate = float(np.mean(stats["comm_rate"]))
                    if rate > 1e-6:
                        row["selection_advantage"] = (
                            float(np.mean(stats["voi_captured"])) / rate)
            if it % cfg.eval_every == 0 or it == iters - 1:
                r = _eval(model, env_cfg, ccfg, device, cfg.eval_eps_per_env)
                row.update({"eval_success": r["success_rate"],
                            "eval_on_goal": r["agents_on_goal"],
                            "eval_coll": r["collisions_per_step"],
                            "eval_makespan": r["makespan"],
                            "peers_in_range": r["peers_in_range"]})
                if r["agents_on_goal"] > best:
                    best = r["agents_on_goal"]
                    torch.save({"model": model.state_dict(), "model_cfg": asdict(mcfg),
                                "comm_cfg": asdict(ccfg), "ppo_cfg": asdict(cfg),
                                "iter": it}, os.path.join(cfg.out_dir, "best.pt"))
            history.append(row)
            print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in row.items()))
            with open(os.path.join(cfg.out_dir, "history.json"), "w") as fh:
                json.dump(history, fh, indent=2)

    torch.save({"model": model.state_dict(), "model_cfg": asdict(mcfg),
                "comm_cfg": asdict(ccfg), "ppo_cfg": asdict(cfg)},
               os.path.join(cfg.out_dir, "last.pt"))
    return {"history": history, "best_on_goal": best, "params": n_params}
