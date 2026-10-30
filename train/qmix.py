"""QMIX training for the communication ablation.

The agent network is `CommActorCritic` unchanged, with its policy head read
as action values (see models/qmix.py for why). Only the learning rule
differs from `train/ppo_comm.py`, which is what makes this a test of whether
the negative communication result is specific to policy gradients.

Replay. The network is recurrent, so a transition cannot be replayed on its
own. Blocks of `rollout` consecutive steps are stored together with the GRU
state at the start of the block, and replayed from that stored state -- the
approach R2D2 uses. Storing whole episodes would be more faithful and far
more memory, at 192 steps per episode.

A caveat to carry into any comparison with the MAPPO numbers: those were
measured on freshly collected on-policy rollouts, whereas the counterfactual
quantities here are computed on replayed blocks drawn from a buffer that is
up to `buffer_blocks` iterations old. The arms are affected identically, so
comparisons between them are sound, but absolute values of influence and
value are not directly comparable across the two backbones.
"""
from __future__ import annotations

import copy
import json
import os
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from eval.metrics import run_episodes
from mapf.env import MAPFConfig, VecMAPFEnv
from models.comm_policy import CommActorCritic, CommConfig
from models.encoder import ModelConfig, N_ACTIONS
from models.qmix import (MixerConfig, QMixer, q_influence,
                         q_value_of_information)
from train.ppo_comm import load_partial


@dataclass
class QMIXConfig:
    # Environment
    n_agents: int = 16
    map_spec: dict = field(default_factory=lambda: {
        "kind": "random", "height": 32, "width": 32, "density": 0.20})
    max_steps: int = 192
    fov: int = 9
    num_envs: int = 16
    map_pool_size: int = 256
    observe_peer_goals: bool = False
    obs_costmap: str = "field"
    # Potential-based shaping on the cost-to-go field, Phi = -dist/(H+W).
    # Off (0.0) for the policy-gradient study, which never needed it: BC put
    # the policy inside the good basin and PPO's small on-policy steps kept it
    # there. A value-based learner starting from scratch has no such basin,
    # and without shaping it finds the degenerate fixed point instead --
    # see docs/FINDINGS-phase6b.md section 20.
    #
    # Ng et al. (1999): potential-based shaping leaves the optimal policy
    # unchanged, so turning it on changes what the learner can find, not what
    # it should find. The env divides Phi by (H+W), so the coefficient is in
    # reward units per map-diameter of progress; around 10 puts one step of
    # progress on the same scale as one step of penalty on a 16x16 map.
    # Any cross-backbone number must state whether shaping was on.
    shaping_coef: float = 0.0

    # Communication: the same four arms as the policy-gradient study
    comm_mode: str = "full"          # none | full | shuffled | learned
    comm_radius: float = 8.0
    msg_dim: int = 64
    n_heads: int = 4
    budget_frac: float = 0.25
    allocation: str = "rate"

    # Replay
    rollout: int = 32                # steps per stored block
    buffer_blocks: int = 64
    batch_blocks: int = 4
    min_blocks: int = 8              # collect this many before learning

    # Optimisation
    total_agent_steps: int = 4_000_000
    updates_per_block: int = 2
    lr: float = 5e-4
    gamma: float = 0.99
    # How the per-agent rewards become the one scalar QMIX factorises.
    #
    # "sum" is the literal team return and was the first run's setting. It is
    # a bad choice here. With step_penalty=-0.3, N=16 and a 192-step horizon,
    # the discounted team value is about -410, while the difference between
    # the best and worst action for one agent is 0.3 to 2.0. The regression
    # target is therefore dominated, roughly 1400:1, by a "time remaining"
    # term that carries no information about which action to take. A
    # policy-gradient learner never meets this problem because GAE subtracts
    # the baseline and it optimises the advantage; a value-based learner has
    # to represent the absolute return accurately AND resolve differences
    # three orders of magnitude smaller inside it.
    #
    # "mean" divides by N and brings the target to about -26, an 85:1 ratio.
    # It is a uniform rescaling, so the ordering of joint actions is
    # untouched -- it changes the conditioning of the regression, not the
    # problem being solved.
    team_reward: str = "mean"        # mean | sum
    reward_scale: float = 1.0        # further uniform scaling, applied after
    grad_clip: float = 10.0
    target_update: int = 50          # iterations between hard target copies
    double_q: bool = True
    mixer_embed: int = 32

    # Exploration
    eps_start: float = 0.5
    eps_end: float = 0.05
    eps_decay_frac: float = 0.4      # fraction of training spent decaying

    warm_start: Optional[str] = None
    allow_env_mismatch: bool = False
    eval_every: int = 10
    eval_eps_per_env: int = 8
    # Action noise at evaluation. 0.0 is the QMIX policy (pure argmax) and is
    # the number to report. A small value is the diagnostic for a run that
    # looks stuck: deterministic policies livelock in this domain, so if
    # eval_epsilon=0.05 scores far higher, the ceiling is the action rule and
    # not the learner. See scripts/probe_warmstart.py.
    eval_epsilon: float = 0.0
    log_every: int = 5
    out_dir: str = "runs/qmix"
    seed: int = 0
    device: str = "cuda"


class Block:
    """One stored chunk of experience plus the recurrent state it began in."""

    __slots__ = ("grid", "vec", "adj", "rel", "state", "action", "reward",
                 "done", "resets", "h0", "next_grid", "next_vec", "next_adj",
                 "next_rel", "next_state")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _epsilon(cfg: QMIXConfig, frac: float) -> float:
    span = max(cfg.eps_decay_frac, 1e-6)
    t = min(frac / span, 1.0)
    return cfg.eps_start + t * (cfg.eps_end - cfg.eps_start)


def _masked_q(model, grid, vec, adj, rel, hidden, resets, send_mask=None):
    """Action values with illegal moves already masked by the network."""
    q, _, h, _, _, sched = model(grid, vec, adj, rel, hidden=hidden,
                                 resets=resets, send_mask=send_mask)
    return q, h, sched


def _evaluate(model, env_cfg, ccfg, device, eps_per_env, radius, num_envs=32,
              greedy=True, epsilon=0.0, comm_mode=None):
    """Evaluate the agent network, with the action rule stated explicitly.

    `greedy=True` is the QMIX policy: argmax of the masked head, which for a
    Q-network is the greedy action. It is NOT what `train/ppo_comm.py::_eval`
    does -- that one samples -- and the difference is large in this domain.
    Measured on `runs/bccomm_a16h20/best.pt` (scripts/probe_warmstart.py):

        comm=none, sampled : 0.9556 on-goal, 0.781 success, 2.08 coll/step
        comm=none, greedy  : 0.5771 on-goal, 0.008 success, 6.22 coll/step

    A deterministic policy has no way to break the symmetry between two agents
    contesting the same cell, so it livelocks and burns the step budget. Any
    comparison of a QMIX number against a MAPPO number must either hold this
    rule fixed or say which rule each used. `epsilon` adds uniform noise to the
    greedy rule, which is the diagnostic for a run that looks stuck.

    The recurrent state must be zeroed for the environments that finished, or
    an episode inherits the hidden state of its predecessor; `run_episodes`
    reports which ones through `on_episode_end`.
    """
    model.eval()
    prev_mode = model.comm_cfg.mode
    if comm_mode is not None:
        model.comm_cfg.mode = comm_mode
    env = VecMAPFEnv(env_cfg, num_envs=num_envs, seed=0, pool_seed=999)
    N, B = env_cfg.n_agents, num_envs
    st = {"h": model.initial_state(B * N, device), "peers": []}

    def act(e):
        grid, vec = e.observe()
        adj_np, rel_np = e.comm_graph(radius)
        with torch.no_grad():
            action, _, _, st["h"], _ = model.act(
                torch.as_tensor(grid, device=device),
                torch.as_tensor(vec, device=device),
                torch.as_tensor(adj_np, device=device),
                torch.as_tensor(rel_np, device=device),
                hidden=st["h"], greedy=greedy)
        if epsilon > 0.0:
            explore = torch.rand(action.shape, device=action.device) < epsilon
            action = torch.where(
                explore, torch.randint_like(action, N_ACTIONS), action)
        st["peers"].append(float(adj_np.sum(-1).mean()))
        return action.cpu().numpy()

    def on_end(finished):
        reset = torch.zeros(B * N, dtype=torch.bool, device=device)
        for b in finished:
            reset[b * N:(b + 1) * N] = True
        st["h"] = st["h"] * (~reset).to(st["h"].dtype).view(1, B * N, 1)

    try:
        out = run_episodes(env, act, episodes_per_env=eps_per_env,
                           on_episode_end=on_end)
    finally:
        model.comm_cfg.mode = prev_mode
    out["peers_in_range"] = float(np.mean(st["peers"])) if st["peers"] else 0.0
    model.train()
    return out


def train_qmix(cfg: QMIXConfig) -> Dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available())
                          else "cpu")
    os.makedirs(cfg.out_dir, exist_ok=True)

    observe_peer_goals, obs_costmap, max_steps = (
        cfg.observe_peer_goals, cfg.obs_costmap, cfg.max_steps)
    if cfg.warm_start:
        meta = torch.load(cfg.warm_start, map_location="cpu", weights_only=False)
        for name, cur in (("observe_peer_goals", observe_peer_goals),
                          ("obs_costmap", obs_costmap), ("max_steps", max_steps)):
            got = meta.get(name)
            if got is not None and got != cur:
                print(f"adopting {name}={got!r} from the warm-start checkpoint "
                      f"(config said {cur!r})")
        observe_peer_goals = meta.get("observe_peer_goals", observe_peer_goals)
        obs_costmap = meta.get("obs_costmap", obs_costmap)
        max_steps = meta.get("max_steps", max_steps)
        ck_comm = (meta.get("comm_cfg", {}) or {}).get("mode")
        ck_map = meta.get("map_spec")
        if ck_map and not cfg.allow_env_mismatch:
            diffs = {k: (ck_map.get(k), cfg.map_spec.get(k))
                     for k in set(ck_map) | set(cfg.map_spec)
                     if ck_map.get(k) != cfg.map_spec.get(k)}
            if diffs:
                raise RuntimeError(
                    f"environment does not match the warm-start checkpoint: {diffs}. "
                    f"Pass allow_env_mismatch=True only if that is deliberate.")
        del meta

    env_cfg = MAPFConfig(n_agents=cfg.n_agents, fov=cfg.fov, max_steps=max_steps,
                         map_pool_size=cfg.map_pool_size, map_spec=cfg.map_spec,
                         observe_peer_goals=observe_peer_goals,
                         obs_costmap=obs_costmap,
                         shaping_coef=cfg.shaping_coef)
    env = VecMAPFEnv(env_cfg, num_envs=cfg.num_envs, seed=cfg.seed)
    B, N, T = cfg.num_envs, cfg.n_agents, cfg.rollout
    M = B * N
    state_dim = N * 5

    grid_np, vec_np = env.reset()
    mcfg = ModelConfig(in_channels=grid_np.shape[2], fov=cfg.fov,
                       vec_dim=vec_np.shape[-1], state_dim=state_dim)
    ccfg = CommConfig(mode=cfg.comm_mode, radius=cfg.comm_radius,
                      msg_dim=cfg.msg_dim, n_heads=cfg.n_heads,
                      budget_frac=cfg.budget_frac, allocation=cfg.allocation)
    model = CommActorCritic(mcfg, ccfg).to(device)
    mixer = QMixer(MixerConfig(n_agents=N, state_dim=state_dim,
                               embed=cfg.mixer_embed)).to(device)
    print(f"agent net {sum(p.numel() for p in model.parameters()):,} params | "
          f"mixer {sum(p.numel() for p in mixer.parameters()):,} | "
          f"comm={cfg.comm_mode} | device={device}")

    if cfg.warm_start:
        info = load_partial(model, cfg.warm_start, device)

        if ck_comm == "none" and cfg.comm_mode != "none":
            print("\n  WARNING: checkpoint was trained with comm='none' but this "
                  f"run is comm='{cfg.comm_mode}'.\n"
                  "  The attention module starts from random initialisation and "
                  "the Q head has\n  never seen a non-zero context. Expect "
                  "q_influence to collapse toward zero.\n")

        # Pre-flight in two parts, because a warm start can fail for two
        # unrelated reasons and only one of them means the load is broken.
        #
        # (1) Integrity. Replay the checkpoint under its OWN protocol -- its
        #     comm mode, sampled actions -- which is how its recorded number
        #     was produced. Anything else compares two different measurements
        #     and blames the loader for the difference. This is the check that
        #     raises.
        expected = info.get("reported_on_goal")
        probe = _evaluate(model, env_cfg, ccfg, device, 2, cfg.comm_radius,
                          greedy=False, comm_mode=ck_comm or cfg.comm_mode)
        print(f"  warm-start integrity ({ck_comm or cfg.comm_mode}, sampled): "
              f"agents-on-goal {probe['agents_on_goal']:.3f}"
              + (f" (checkpoint recorded {expected:.3f})" if expected else ""))
        if expected is not None and probe["agents_on_goal"] < expected - 0.25:
            raise RuntimeError(
                f"warm start is not behaving like its checkpoint: "
                f"{probe['agents_on_goal']:.3f} here vs {expected:.3f} recorded "
                f"under the checkpoint's own protocol.\n"
                f"  An architecture or observability mismatch -- check that the "
                f"demo dataset's observe_peer_goals matches this run.")
        if expected is None and probe["agents_on_goal"] < 0.10:
            print("  WARNING: warm-started policy reaches <10% agents-on-goal. "
                  "Training will likely behave as if from scratch.")

        # (2) Starting point. The same weights under THIS run's protocol --
        #     its comm mode, greedy actions, which is the QMIX policy. The gap
        #     between the two lines is not a defect: a policy trained against a
        #     sampling objective loses most of its performance when its argmax
        #     is taken instead (0.956 -> 0.577 on the a16h20 checkpoint). It is
        #     reported so the first eval_on_goal is not mistaken for a
        #     collapse, and so the figure QMIX has to beat is on the record.
        start = _evaluate(model, env_cfg, ccfg, device, 2, cfg.comm_radius,
                          greedy=True)
        print(f"  QMIX starting point ({cfg.comm_mode}, greedy):  "
              f"agents-on-goal {start['agents_on_goal']:.3f}  "
              f"success {start['success_rate']:.3f}  "
              f"coll/step {start['collisions_per_step']:.2f}")
        print(f"  greedy costs {start['agents_on_goal'] - probe['agents_on_goal']:+.3f} "
              f"on-goal before a single gradient step.\n")

    target_model = copy.deepcopy(model)
    target_mixer = copy.deepcopy(mixer)
    for p in list(target_model.parameters()) + list(target_mixer.parameters()):
        p.requires_grad_(False)

    opt = torch.optim.Adam(list(model.parameters()) + list(mixer.parameters()),
                           lr=cfg.lr)

    buffer: deque = deque(maxlen=cfg.buffer_blocks)
    hidden = model.initial_state(M, device)
    prev_done = torch.zeros(B, N, dtype=torch.bool, device=device)

    steps_per_iter = B * N * T
    iters = max(1, cfg.total_agent_steps // steps_per_iter)
    history: List[Dict] = []
    best = -1.0

    for it in range(iters):
        eps = _epsilon(cfg, it / max(iters - 1, 1))

        # ---------------------------------------------------------- collect
        blk = {k: [] for k in ("grid", "vec", "adj", "rel", "state", "action",
                               "reward", "done", "resets")}
        h0 = hidden.detach().clone()
        with torch.no_grad():
            for _ in range(T):
                adj_np, rel_np = env.comm_graph(cfg.comm_radius)
                g = torch.as_tensor(grid_np, device=device)[None]
                v = torch.as_tensor(vec_np, device=device)[None]
                adj = torch.as_tensor(adj_np, device=device)[None]
                rel = torch.as_tensor(rel_np, device=device)[None]
                s = torch.as_tensor(env.global_state(), device=device)

                resets = prev_done[None]
                q, hidden, _ = _masked_q(model, g, v, adj, rel, hidden, resets)
                greedy = q[0].argmax(-1)
                explore = torch.rand(B, N, device=device) < eps
                rand_a = torch.randint(0, N_ACTIONS, (B, N), device=device)
                action = torch.where(explore, rand_a, greedy)

                for k, val in (("grid", g[0]), ("vec", v[0]), ("adj", adj[0]),
                               ("rel", rel[0]), ("state", s),
                               ("action", action), ("resets", prev_done)):
                    blk[k].append(val.cpu())

                grid_np, vec_np, reward, done, info = env.step(action.cpu().numpy())
                # QMIX factorises a single team value, so the per-agent
                # rewards become one scalar per environment. See
                # QMIXConfig.team_reward for why the default is the mean.
                team = (reward.mean(-1) if cfg.team_reward == "mean"
                        else reward.sum(-1))
                blk["reward"].append(
                    torch.as_tensor(team * cfg.reward_scale))
                d = torch.as_tensor(done, device=device)
                blk["done"].append(d.cpu())
                prev_done = d[:, None].expand(B, N)
                hidden = hidden * (~prev_done.reshape(M)).to(hidden.dtype).view(1, M, 1)

            # bootstrap observation for the final transition of the block
            adj_np, rel_np = env.comm_graph(cfg.comm_radius)
            nxt = dict(
                grid=torch.as_tensor(grid_np).clone(),
                vec=torch.as_tensor(vec_np).clone(),
                adj=torch.as_tensor(adj_np).clone(),
                rel=torch.as_tensor(rel_np).clone(),
                state=torch.as_tensor(env.global_state()).clone())

        buffer.append(Block(
            h0=h0.cpu(), next_grid=nxt["grid"], next_vec=nxt["vec"],
            next_adj=nxt["adj"], next_rel=nxt["rel"], next_state=nxt["state"],
            **{k: torch.stack(v) for k, v in blk.items()}))

        if len(buffer) < cfg.min_blocks:
            continue

        # ----------------------------------------------------------- learn
        stats = {k: [] for k in ("loss", "q_tot", "td_err", "influence", "value")}
        for u in range(cfg.updates_per_block):
            batch = random.sample(list(buffer), min(cfg.batch_blocks, len(buffer)))
            loss_total = 0.0
            for b in batch:
                g = b.grid.to(device); v = b.vec.to(device)
                adj = b.adj.to(device); rel = b.rel.to(device)
                act = b.action.to(device); rew = b.reward.to(device)
                done = b.done.to(device).float(); res = b.resets.to(device)
                st = b.state.to(device); h0b = b.h0.to(device)

                q, _, _ = _masked_q(model, g, v, adj, rel, h0b, res)
                chosen = q.gather(-1, act.unsqueeze(-1)).squeeze(-1)   # (T,B,N)
                q_tot = mixer(chosen.reshape(T * B, N), st.reshape(T * B, -1))
                q_tot = q_tot.view(T, B)

                with torch.no_grad():
                    # shift by one step: the target for step t uses step t+1
                    g2 = torch.cat([g[1:], b.next_grid.to(device)[None]], 0)
                    v2 = torch.cat([v[1:], b.next_vec.to(device)[None]], 0)
                    a2 = torch.cat([adj[1:], b.next_adj.to(device)[None]], 0)
                    r2 = torch.cat([rel[1:], b.next_rel.to(device)[None]], 0)
                    s2 = torch.cat([st[1:], b.next_state.to(device)[None]], 0)
                    res2 = torch.cat([res[1:], done[-1:, :, None].bool()
                                      .expand(1, B, N)], 0)

                    q_tgt, _, _ = _masked_q(target_model, g2, v2, a2, r2, h0b, res2)
                    if cfg.double_q:
                        q_on, _, _ = _masked_q(model, g2, v2, a2, r2, h0b, res2)
                        a_star = q_on.argmax(-1, keepdim=True)
                    else:
                        a_star = q_tgt.argmax(-1, keepdim=True)
                    nxt_chosen = q_tgt.gather(-1, a_star).squeeze(-1)
                    nxt_tot = target_mixer(nxt_chosen.reshape(T * B, N),
                                           s2.reshape(T * B, -1)).view(T, B)
                    target = rew + cfg.gamma * (1.0 - done) * nxt_tot

                td = q_tot - target
                loss = td.pow(2).mean()
                loss_total = loss_total + loss

                with torch.no_grad():
                    stats["q_tot"].append(float(q_tot.mean()))
                    stats["td_err"].append(float(td.abs().mean()))
                    if ccfg.mode != "none":
                        q_sil, _, _ = _masked_q(
                            model, g, v, adj, rel, h0b, res,
                            send_mask=torch.zeros(T, B, N, dtype=torch.bool,
                                                  device=device))
                        stats["influence"].append(float(q_influence(q.detach(), q_sil).mean()))
                        stats["value"].append(float(
                            q_value_of_information(q.detach(), q_sil, act).mean()))

            loss_total = loss_total / len(batch)
            opt.zero_grad(set_to_none=True)
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(mixer.parameters()), cfg.grad_clip)
            opt.step()
            stats["loss"].append(float(loss_total.detach()))

        if it % cfg.target_update == 0:
            target_model.load_state_dict(model.state_dict())
            target_mixer.load_state_dict(mixer.state_dict())

        # ------------------------------------------------------------- log
        if it % cfg.log_every == 0 or it == iters - 1:
            row = {"iter": it, "agent_steps": (it + 1) * steps_per_iter,
                   "epsilon": eps, "buffer": len(buffer)}
            for k, vals in stats.items():
                if vals:
                    row[k if k != "influence" else "q_influence"] = float(np.mean(vals))
            if stats["value"]:
                row["q_voi"] = float(np.mean(stats["value"]))
                row.pop("value", None)
            if it % cfg.eval_every == 0 or it == iters - 1:
                model.eval()
                r = _evaluate(model, env_cfg, ccfg, device,
                              cfg.eval_eps_per_env, cfg.comm_radius,
                              epsilon=cfg.eval_epsilon)
                model.train()
                row.update(eval_success=r["success_rate"],
                           eval_on_goal=r["agents_on_goal"],
                           eval_coll=r["collisions_per_step"],
                           eval_makespan=r.get("makespan", 0.0))
                if r["agents_on_goal"] > best:
                    best = r["agents_on_goal"]
                    torch.save({"model": model.state_dict(),
                                "mixer": mixer.state_dict(),
                                "model_cfg": asdict(mcfg), "comm_cfg": asdict(ccfg),
                                "qmix_cfg": asdict(cfg),
                                "observe_peer_goals": observe_peer_goals,
                                "obs_costmap": obs_costmap, "max_steps": max_steps,
                                "n_agents": N, "map_spec": cfg.map_spec,
                                "fov": cfg.fov, "env_on_goal": best},
                               os.path.join(cfg.out_dir, "best.pt"))
            history.append(row)
            print("  " + "  ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in row.items()))
            with open(os.path.join(cfg.out_dir, "history.json"), "w") as fh:
                json.dump(history, fh, indent=2)

    return {"history": history, "best_on_goal": best}
