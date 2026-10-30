"""Behaviour cloning for the communication architecture.

Why this exists instead of reusing `train/bc.py`
------------------------------------------------
`bc.py` trains `ActorCritic`, whose policy head is a single `nn.Linear`
(`policy.weight`). `CommActorCritic`'s head is an `nn.Sequential`
(`policy.0`, `policy.2`). The names do not match, so warm-starting PPO from a
`bc.py` checkpoint silently transferred **only the encoder** and left the
policy head at random initialisation. Training then began from a uniformly
random policy with good features -- indistinguishable from training from
scratch, which reaches ~0.2% agents-on-goal. Six hours of GPU time produced
nothing, and the loader reported "19 tensors" without complaint.

The fix is structural: clone the architecture you intend to fine-tune. This
module trains `CommActorCritic` directly, so every tensor transfers.

Both heads are supervised on the expert action:

* **tentative** -- what the agent would do from its own observation alone.
* **policy**    -- what it does after attending to peer intents.

That gives "intent" a concrete meaning rather than leaving it an arbitrary
embedding, and it is the semantics the whole communication design rests on.
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

from eval.rollout import evaluate_comm_policy
from mapf.env import MAPFConfig
from models.comm_policy import CommActorCritic, CommConfig
from models.encoder import ModelConfig


@dataclass
class CommBCConfig:
    demo_root: str = "data/demos/a16h"
    comm_mode: str = "none"        # see note in train_comm_bc()
    comm_radius: float = 8.0
    seq_len: int = 16
    batch_size: int = 16           # EPISODES, not agents (each carries N agents)
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    parked_weight: float = 0.1
    tentative_coef: float = 0.5    # weight on the pass-1 auxiliary loss
    max_shards: Optional[int] = None
    eval_every: int = 1
    eval_eps_per_env: int = 3
    patience: int = 4
    out_dir: str = "runs/bc_comm"
    seed: int = 0
    device: str = "cuda"


def _comm_graph_from_pos(pos: torch.Tensor, radius: float):
    """Build the peer graph from stored positions. ``pos``: (L, B, N, 2)."""
    delta = (pos[:, :, None, :, :] - pos[:, :, :, None, :]).float()
    cheb = delta.abs().amax(-1)
    n = pos.shape[2]
    eye = torch.eye(n, dtype=torch.bool, device=pos.device)
    adj = (cheb <= radius) & ~eye
    return adj, delta / max(float(radius), 1.0)


def train_comm_bc(cfg: CommBCConfig) -> Dict:
    from train.dataset import DemoDataset

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available())
                          else "cpu")
    os.makedirs(cfg.out_dir, exist_ok=True)

    data = DemoDataset(cfg.demo_root, seq_len=cfg.seq_len,
                       parked_weight=cfg.parked_weight, max_shards=cfg.max_shards,
                       seed=cfg.seed, group_agents=True)
    print(data)

    env_cfg = MAPFConfig(
        n_agents=data.meta.get("n_agents", data.n_agents),
        fov=data.obs_shape[1],
        max_steps=data.meta.get("max_steps", 128),
        map_pool_size=64,
        observe_peer_goals=data.meta.get("observe_peer_goals", True),
        obs_costmap=data.meta.get("obs_costmap", "field"),
        map_spec=data.meta.get("map_spec", {"kind": "random", "height": 32,
                                            "width": 32, "density": 0.20}),
    )
    mcfg = ModelConfig(in_channels=data.obs_shape[0], fov=data.obs_shape[1],
                       vec_dim=data.vec_dim, state_dim=env_cfg.n_agents * 5)
    ccfg = CommConfig(mode=cfg.comm_mode, radius=cfg.comm_radius)
    model = CommActorCritic(mcfg, ccfg).to(device)
    print(f"model: {sum(p.numel() for p in model.parameters()):,} parameters "
          f"| comm={cfg.comm_mode} | device={device}")

    steps_per_epoch = max(1, len(data.train_index) // cfg.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch, pct_start=0.1)

    def losses(batch):
        grid = torch.as_tensor(batch["grid"], device=device)
        vec = torch.as_tensor(batch["vec"], device=device)
        pos = torch.as_tensor(batch["pos"], device=device)
        action = torch.as_tensor(batch["action"], device=device)
        weight = torch.as_tensor(batch["weight"], device=device)
        adj, rel = _comm_graph_from_pos(pos, cfg.comm_radius)
        logits, _, _, tent, _, _ = model(grid, vec, adj, rel)

        def wce(x):
            ce = F.cross_entropy(x.reshape(-1, x.shape[-1]), action.reshape(-1),
                                 reduction="none").view_as(action)
            return (ce * weight).sum() / weight.sum().clamp(min=1.0)

        return wce(logits), wce(tent), logits, action, weight, batch

    history, best, best_epoch, stale = [], -1.0, -1, 0
    for epoch in range(cfg.epochs):
        model.train()
        t0, tr = time.perf_counter(), []
        for batch in data.batches(cfg.batch_size, "train"):
            main, tentative, *_ = losses(batch)
            loss = main + cfg.tentative_coef * tentative
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            tr.append(loss.item())

        model.eval()
        vl, acc_main, acc_tent = [], [], []
        with torch.no_grad():
            for batch in data.batches(cfg.batch_size, "val", shuffle=False):
                main, tentative, logits, action, weight, _ = losses(batch)
                vl.append((main + cfg.tentative_coef * tentative).item())
                valid = weight > 0
                acc_main.append(((logits.argmax(-1) == action) & valid).sum().item()
                                / max(valid.sum().item(), 1))
        row = {"epoch": epoch, "train_loss": float(np.mean(tr)),
               "val_loss": float(np.mean(vl)) if vl else float("nan"),
               "acc": float(np.mean(acc_main)) if acc_main else float("nan"),
               "seconds": round(time.perf_counter() - t0, 1)}

        improved = False
        if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            r = evaluate_comm_policy(model, env_cfg, device, radius=cfg.comm_radius,
                                     episodes_per_env=cfg.eval_eps_per_env)
            row.update({"env_success": r["success_rate"],
                        "env_on_goal": r["agents_on_goal"],
                        "env_coll": r["collisions_per_step"]})
            if r["agents_on_goal"] > best + 1e-4:
                best, best_epoch, improved = r["agents_on_goal"], epoch, True
                torch.save({"model": model.state_dict(), "model_cfg": asdict(mcfg),
                            "comm_cfg": asdict(ccfg), "bc_cfg": asdict(cfg),
                            "epoch": epoch, "env_on_goal": best,
                            "env_success": r["success_rate"],
                            "observe_peer_goals": env_cfg.observe_peer_goals,
                            "obs_costmap": env_cfg.obs_costmap,
                            "n_agents": env_cfg.n_agents,
                            "map_spec": env_cfg.map_spec,
                            "max_steps": env_cfg.max_steps,
                            "fov": env_cfg.fov},
                           os.path.join(cfg.out_dir, "best.pt"))
            stale = 0 if improved else stale + 1

        history.append(row)
        print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in row.items()))
        with open(os.path.join(cfg.out_dir, "history.json"), "w") as fh:
            json.dump(history, fh, indent=2)

        if stale >= cfg.patience:
            print(f"  early stop: {stale} epochs without environment improvement")
            break

    print(f"\n  best: epoch {best_epoch}, agents-on-goal {best:.3f} "
          f"-> {os.path.join(cfg.out_dir, 'best.pt')}")
    return {"history": history, "best_on_goal": best, "best_epoch": best_epoch}
