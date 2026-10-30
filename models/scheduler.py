"""Budgeted communication scheduling.

The problem this solves, measured
---------------------------------
Phase 5 established that full communication is a *trade-off*: at 16 agents it
removes 25% of collisions (p < 0.001, d = 2.51) but costs ~3.9pp of task
success. The mechanism, matching the Phase 2 scripted oracle, is that agents
told their neighbours' intentions become deferential — they yield instead of
contesting, which eliminates conflicts and slows progress.

If that is right, full communication makes agents cautious *everywhere*
rather than *where it matters*. A scheduler that transmits selectively should
keep the collision benefit and drop the throughput cost:

    no comm     collisions 0.1009   success 0.7402
    full comm   collisions 0.0760   success 0.7012
    target      collisions <= 0.080  success >= 0.740   with B < N senders

Allocation modes
----------------
* ``topk``  — exactly ``k`` senders every timestep. Strict and simple, but
  rigid: a crowded doorway and an empty corridor get the same budget.
* ``rate``  — per-agent independent gates with a Lagrangian dual enforcing the
  budget *on average*. Agents can all speak during a crisis and stay silent in
  open space. This is the point of adaptive scheduling, so it is the default.

Both use a straight-through estimator: the forward pass sends a hard 0/1 mask
(so the budget is real, not a soft approximation), while the backward pass
sees a differentiable relaxation.

Baselines (``random`` and ``distance``) share the interface so the ablation
compares scheduling policies, not implementations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SchedulerConfig:
    mode: str = "learned"        # learned | random | distance | all | none
    allocation: str = "rate"     # rate | topk
    budget_frac: float = 0.25    # fraction of agents permitted to transmit
    tau: float = 1.0             # relaxation temperature
    lambda_lr: float = 0.05      # dual ascent step on the budget multiplier
    lambda_init: float = 0.0
    hidden: int = 256


class BudgetedScheduler(nn.Module):
    """Decides which agents transmit, under a hard or average budget."""

    def __init__(self, cfg: SchedulerConfig):
        super().__init__()
        self.cfg = cfg
        self.scorer = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.hidden // 2), nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden // 2, 1),
        )
        # The dual variable is state, not a parameter: it is updated by dual
        # ascent on the constraint violation, not by the policy gradient.
        self.register_buffer("lam", torch.tensor(float(cfg.lambda_init)))

    # ------------------------------------------------------------------ score

    def score(self, h: torch.Tensor) -> torch.Tensor:
        """Transmit priority per agent. ``h``: (..., N, H) -> (..., N)."""
        return self.scorer(h).squeeze(-1)

    # ------------------------------------------------------------- selection

    def _gumbel_like(self, x: torch.Tensor) -> torch.Tensor:
        u = torch.rand_like(x).clamp_(1e-9, 1 - 1e-9)
        return -torch.log(-torch.log(u))

    def _logistic_like(self, x: torch.Tensor) -> torch.Tensor:
        u = torch.rand_like(x).clamp_(1e-9, 1 - 1e-9)
        return torch.log(u) - torch.log1p(-u)

    def _select_topk(self, logits: torch.Tensor, training: bool) -> torch.Tensor:
        """Exactly k senders per row, differentiably."""
        n = logits.shape[-1]
        k = max(1, int(round(self.cfg.budget_frac * n)))
        perturbed = logits + self._gumbel_like(logits) if training else logits
        idx = perturbed.topk(k, dim=-1).indices
        hard = torch.zeros_like(logits).scatter(-1, idx, 1.0)
        # Relaxation normalised to sum to k, so its scale matches the hard mask.
        soft = torch.softmax(perturbed / self.cfg.tau, dim=-1) * k
        # Parentheses matter: (hard + soft) - soft.detach() loses the exact
        # 0/1 forward value to floating-point cancellation, and then the
        # 'hard' budget is only approximately hard.
        return hard + (soft - soft.detach())

    def _select_rate(self, logits: torch.Tensor, training: bool) -> torch.Tensor:
        """Independent gates; the budget is enforced on average by the dual."""
        z = torch.sigmoid((logits + self._logistic_like(logits)) / self.cfg.tau) \
            if training else torch.sigmoid(logits)
        hard = (z > 0.5).to(z.dtype)
        return hard + (z - z.detach())

    # ------------------------------------------------------------------ main

    def forward(
        self,
        h: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns ``(send_mask, info)``; ``send_mask`` is ``(..., N)``.

        ``adjacency`` (..., N, N) is used by the distance baseline and to
        report how many of the selected senders actually had an audience.
        """
        cfg = self.cfg
        n = h.shape[-2]

        if cfg.mode == "all":
            send = torch.ones(h.shape[:-1], device=h.device, dtype=h.dtype)
            logits = torch.zeros_like(send)
        elif cfg.mode == "none":
            send = torch.zeros(h.shape[:-1], device=h.device, dtype=h.dtype)
            logits = torch.zeros_like(send)
        elif cfg.mode == "random":
            # Budget-matched coin flips: isolates whether a learned scheduler
            # has learned anything beyond respecting the budget.
            logits = torch.zeros(h.shape[:-1], device=h.device, dtype=h.dtype)
            noise = torch.rand_like(logits)
            if cfg.allocation == "topk":
                k = max(1, int(round(cfg.budget_frac * n)))
                idx = noise.topk(k, dim=-1).indices
                send = torch.zeros_like(noise).scatter(-1, idx, 1.0)
            else:
                send = (noise < cfg.budget_frac).to(h.dtype)
        elif cfg.mode == "distance":
            # Hand-designed heuristic: speak when someone is close enough to
            # care. A learned scheduler that cannot beat this is not earning
            # its complexity.
            if adjacency is None:
                raise ValueError("distance scheduler needs an adjacency matrix")
            logits = adjacency.sum(-1).to(h.dtype)
            send = self._select_topk(logits, training=False)
        elif cfg.mode == "learned":
            logits = self.score(h)
            send = (self._select_topk(logits, training) if cfg.allocation == "topk"
                    else self._select_rate(logits, training))
        else:
            raise ValueError(f"unknown scheduler mode {cfg.mode!r}")

        rate = send.detach().mean()
        info = {
            "logits": logits,
            "rate": rate,
            "budget": torch.tensor(cfg.budget_frac, device=h.device),
            "violation": rate - cfg.budget_frac,
            "lam": self.lam.detach().clone(),
        }
        if adjacency is not None:
            heard = (adjacency.any(-2).to(send.dtype) * send).detach()
            info["useful_rate"] = heard.mean()
        return send, info

    # ------------------------------------------------------------- budgeting

    def budget_penalty(self, send: torch.Tensor) -> torch.Tensor:
        """Lagrangian term to add to the loss. Zero for `topk` (exact by design)."""
        if self.cfg.mode != "learned" or self.cfg.allocation == "topk":
            return send.sum() * 0.0
        return self.lam * (send.mean() - self.cfg.budget_frac)

    @torch.no_grad()
    def dual_step(self, realised_rate: float) -> float:
        """Dual ascent on the multiplier; projected to stay non-negative.

        Raising `lam` when the realised rate exceeds the budget makes talking
        more expensive until the constraint is met, without hand-tuning a
        penalty coefficient against the task reward.
        """
        if self.cfg.allocation == "topk":
            return float(self.lam)
        self.lam += self.cfg.lambda_lr * (realised_rate - self.cfg.budget_frac)
        self.lam.clamp_(min=0.0)
        return float(self.lam)
