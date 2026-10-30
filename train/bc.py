"""Behaviour cloning from expert demonstrations.

Weighted cross-entropy over sub-sequences, with the trivial parked-on-goal
samples downweighted. Metrics are reported *per behaviour class*, not just
overall: a policy can look 95% accurate while being useless, because half the
dataset is "stay on your goal".

Model selection runs on the ENVIRONMENT, not on validation loss
---------------------------------------------------------------
Measured on a 16-agent run, validation loss fell monotonically while closed-
loop success collapsed:

    epoch   val_loss   accuracy   env success
      3      0.0804     98.24%       80.9%
     19      0.0543     98.75%       28.9%

Validation loss scores single-step accuracy on the *expert's* state
distribution. Closed-loop performance depends on the policy's *own*
distribution, where small errors compound into states the expert never
visited. Fitting the expert more tightly makes the policy more brittle
off-distribution, so the usual early-stopping signal points the wrong way.

Hence: evaluate in the environment every epoch, select on
`env_sample_on_goal`, and stop on patience against that metric.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from eval.rollout import evaluate_policy
from mapf.env import MAPFConfig
from models.encoder import ActorCritic, ModelConfig, count_parameters
from train.dataset import DemoDataset


@dataclass
class BCConfig:
    demo_root: str = "data/demos/a8"
    seq_len: int = 16
    batch_size: int = 64
    epochs: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    parked_weight: float = 0.1
    entropy_bonus: float = 0.0
    max_shards: Optional[int] = None
    eval_every: int = 1                # evaluate in-env every epoch by default
    eval_episodes_per_env: int = 4
    patience: int = 4                  # epochs without env improvement before stopping
    out_dir: str = "runs/bc"
    seed: int = 0
    device: str = "cuda"
    use_gru: bool = True


def _metrics(logits, action, at_goal, valid) -> Dict[str, float]:
    """Accuracy broken down by behaviour class."""
    pred = logits.argmax(-1)
    correct = (pred == action) & valid
    parked = at_goal & (action == 0) & valid
    yielding = (~at_goal) & (action == 0) & valid
    moving = (action != 0) & valid

    def acc(mask):
        n = mask.sum()
        return float((correct & mask).sum() / n) if n > 0 else float("nan")

    return {
        "acc": float(correct.sum() / valid.sum()) if valid.sum() > 0 else 0.0,
        "acc_moving": acc(moving),
        "acc_parked": acc(parked),
        "acc_yield": acc(yielding),
        "pred_stay_frac": float((pred[valid] == 0).float().mean()),
        "n_yield": int(yielding.sum()),
    }


def train_bc(cfg: BCConfig) -> Dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu"
                          else "cpu")
    os.makedirs(cfg.out_dir, exist_ok=True)

    data = DemoDataset(cfg.demo_root, seq_len=cfg.seq_len,
                       parked_weight=cfg.parked_weight,
                       max_shards=cfg.max_shards, seed=cfg.seed)
    print(data)

    mcfg = ModelConfig(in_channels=data.obs_shape[0], fov=data.obs_shape[1],
                       vec_dim=data.vec_dim, use_gru=cfg.use_gru)
    model = ActorCritic(mcfg).to(device)
    print(f"model: {count_parameters(model):,} parameters, device={device}")

    steps_per_epoch = max(1, len(data.train_index) // cfg.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch, pct_start=0.1)

    env_cfg = MAPFConfig(
        n_agents=data.meta.get("n_agents", data.n_agents),
        fov=data.obs_shape[1],
        max_steps=data.meta.get("max_steps", 128),
        map_pool_size=64,
        map_spec=data.meta.get("map_spec", {"kind": "random", "height": 20,
                                            "width": 20, "density": 0.15}),
        observe_peer_goals=data.meta.get("observe_peer_goals", True),
    )

    history, best, best_epoch, stale = [], -1.0, -1, 0
    for epoch in range(cfg.epochs):
        model.train()
        t0, losses, agg = time.perf_counter(), [], []
        for step, batch in enumerate(data.batches(cfg.batch_size, "train")):
            grid = torch.as_tensor(batch["grid"], device=device)
            vec = torch.as_tensor(batch["vec"], device=device)
            action = torch.as_tensor(batch["action"], device=device)
            weight = torch.as_tensor(batch["weight"], device=device)

            logits, _, _ = model(grid, vec)
            ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), action.reshape(-1), reduction="none"
            ).view_as(action)
            loss = (ce * weight).sum() / weight.sum().clamp(min=1.0)

            if cfg.entropy_bonus > 0:
                logp = F.log_softmax(logits, dim=-1)
                ent = -(logp.exp() * logp).sum(-1)
                loss = loss - cfg.entropy_bonus * (ent * (weight > 0)).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            losses.append(loss.item())

        # ----------------------------------------------------------- validate
        model.eval()
        val_losses, chunks = [], []
        with torch.no_grad():
            for batch in data.batches(cfg.batch_size, "val", shuffle=False):
                grid = torch.as_tensor(batch["grid"], device=device)
                vec = torch.as_tensor(batch["vec"], device=device)
                action = torch.as_tensor(batch["action"], device=device)
                weight = torch.as_tensor(batch["weight"], device=device)
                at_goal = torch.as_tensor(batch["at_goal"], device=device)
                logits, _, _ = model(grid, vec)
                ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                     action.reshape(-1), reduction="none").view_as(action)
                val_losses.append(float((ce * weight).sum() / weight.sum().clamp(min=1.0)))
                chunks.append(_metrics(logits, action, at_goal, weight > 0))

        m = {k: float(np.nanmean([c[k] for c in chunks])) for k in chunks[0]} if chunks else {}
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "val_loss": float(np.mean(val_losses)) if val_losses else float("nan"),
               **m, "seconds": round(time.perf_counter() - t0, 1)}

        improved = False
        if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            r = evaluate_policy(model, env_cfg, device,
                                episodes_per_env=cfg.eval_episodes_per_env)
            row["env_sample_success"] = r["success_rate"]
            row["env_sample_on_goal"] = r["agents_on_goal"]
            row["env_sample_coll"] = r["collisions_per_step"]
            score = r["agents_on_goal"]
            if score > best + 1e-4:
                best, best_epoch, improved = score, epoch, True
                torch.save({"model": model.state_dict(), "model_cfg": asdict(mcfg),
                            "bc_cfg": asdict(cfg), "epoch": epoch},
                           os.path.join(cfg.out_dir, "best.pt"))
            stale = 0 if improved else stale + 1

            # Flag the divergence explicitly -- it is counter-intuitive enough
            # that a silent log invites selecting the wrong checkpoint.
            if len(history) and "val_loss" in history[-1]:
                if row["val_loss"] < history[-1]["val_loss"] and not improved:
                    row["warn"] = "val_loss improved but env did not"

        history.append(row)
        print("  " + "  ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in row.items()))

        with open(os.path.join(cfg.out_dir, "history.json"), "w") as fh:
            json.dump(history, fh, indent=2)

        if stale >= cfg.patience:
            print(f"  early stop: {stale} epochs without environment improvement "
                  f"(best was epoch {best_epoch}, on_goal={best:.3f})")
            break

    torch.save({"model": model.state_dict(), "model_cfg": asdict(mcfg),
                "bc_cfg": asdict(cfg)}, os.path.join(cfg.out_dir, "last.pt"))
    print(f"\n  best checkpoint: epoch {best_epoch}, agents-on-goal {best:.3f} "
          f"-> {os.path.join(cfg.out_dir, 'best.pt')}")
    print("  use best.pt, not last.pt -- see this module's docstring.")
    return {"history": history, "best_on_goal": best, "best_epoch": best_epoch}
