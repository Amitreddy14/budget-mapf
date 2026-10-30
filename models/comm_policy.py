"""Actor-critic with intent communication.

Experimental-design note
------------------------
`comm_mode` switches communication off, on, or learned **at runtime, on one
architecture**. The no-communication baseline is therefore the *same network
with the message context zeroed*, not a smaller network.

This matters. If the no-comm baseline were a separate, narrower model, any
gap could be explained by capacity rather than communication, and the central
claim would be unfalsifiable. Same parameters, same capacity, one bit of
difference.

Shapes. Communication needs the ``(B, N)`` structure that the flat
``M = B * N`` layout of `ActorCritic` destroys, so this model works in
``(T, B, N, ...)``. Time folds into the batch axis for the attention step,
since communication is independent across timesteps -- no Python loop over T.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.comm import IntentComm
from models.encoder import (N_ACTIONS, NEG_INF, AgentEncoder, ModelConfig,
                            legal_action_mask)
from models.scheduler import BudgetedScheduler, SchedulerConfig


@dataclass
class CommConfig:
    mode: str = "full"          # "none" | "full" | "shuffled" (control)
                            # | "learned" | "random" | "distance"
    radius: float = 8.0         # Chebyshev communication range, in cells
    msg_dim: int = 64
    n_heads: int = 4
    use_rel_pos: bool = True
    dropout: float = 0.0
    # Scheduling, used when mode is not "none" or "full".
    budget_frac: float = 0.25
    allocation: str = "rate"    # rate | topk
    tau: float = 1.0
    lambda_lr: float = 0.05


class CommActorCritic(nn.Module):
    """Decentralised actor with intent messaging; centralised critic."""

    def __init__(self, cfg: ModelConfig, comm: CommConfig):
        super().__init__()
        self.cfg, self.comm_cfg = cfg, comm
        self.encoder = AgentEncoder(cfg)
        self.gru = nn.GRU(cfg.hidden, cfg.hidden, num_layers=cfg.gru_layers) \
            if cfg.use_gru else None

        # Pass 1: what the agent would do knowing only its own observation.
        # This doubles as the message content and is supervised during BC, so
        # "intent" means something concrete rather than an arbitrary embedding.
        self.tentative = nn.Linear(cfg.hidden, N_ACTIONS)

        self.comm = IntentComm(cfg.hidden, comm.msg_dim, comm.n_heads,
                               comm.use_rel_pos, comm.dropout)

        # Always constructed, so every comm mode has the same parameter count
        # and the ablation isolates the scheduling policy rather than capacity.
        self.scheduler = BudgetedScheduler(SchedulerConfig(
            mode=comm.mode, allocation=comm.allocation,
            budget_frac=comm.budget_frac, tau=comm.tau,
            lambda_lr=comm.lambda_lr, hidden=cfg.hidden))

        # Pass 2: decide again, now with the peer context.
        self.policy = nn.Sequential(
            nn.Linear(cfg.hidden * 2, cfg.hidden), nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden, N_ACTIONS),
        )
        self.value = nn.Sequential(
            nn.Linear(cfg.hidden + cfg.state_dim, cfg.hidden), nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden, 1),
        )
        self.apply(self._init)
        for head in (self.tentative, self.policy[-1]):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=2 ** 0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def initial_state(self, m: int, device) -> torch.Tensor:
        return torch.zeros(self.cfg.gru_layers, m, self.cfg.hidden, device=device)

    def forward(
        self,
        grid: torch.Tensor,          # (T, B, N, C, F, F)
        vec: torch.Tensor,           # (T, B, N, V)
        adjacency: torch.Tensor,     # (T, B, N, N) bool
        rel_pos: Optional[torch.Tensor] = None,   # (T, B, N, N, 2)
        hidden: Optional[torch.Tensor] = None,    # (L, B*N, H)
        resets: Optional[torch.Tensor] = None,    # (T, B*N) bool
        state: Optional[torch.Tensor] = None,     # (T, B, N, state_dim)
        send_mask: Optional[torch.Tensor] = None, # (T, B, N)
    ):
        T, B, N = grid.shape[:3]
        M = B * N

        feats = self.encoder(grid.reshape(T * M, *grid.shape[3:]),
                             vec.reshape(T * M, -1)).view(T, M, -1)

        if self.gru is None:
            core = feats
            hidden = hidden if hidden is not None else self.initial_state(M, grid.device)
        else:
            if hidden is None:
                hidden = self.initial_state(M, grid.device)
            if resets is None:
                core, hidden = self.gru(feats, hidden)
            else:
                outs = []
                for t in range(T):
                    hidden = hidden * (~resets[t]).to(hidden.dtype).view(1, M, 1)
                    out, hidden = self.gru(feats[t:t + 1], hidden)
                    outs.append(out)
                core = torch.cat(outs, dim=0)

        h = core.view(T, B, N, -1)
        mask = legal_action_mask(grid.reshape(T * M, *grid.shape[3:])).view(T, B, N, N_ACTIONS)

        tentative_logits = self.tentative(h).masked_fill(~mask, NEG_INF)

        sched_info = None
        if self.comm_cfg.mode == "none":
            ctx = torch.zeros_like(h)
            attn = None
        else:
            flat = T * B
            intent = torch.softmax(tentative_logits, dim=-1)

            if self.comm_cfg.mode == "shuffled":
                # Capacity control. With mode="none" the context is zeroed, so
                # the attention module gets no gradient and the policy head
                # learns a function of zeros -- a comm=full model therefore has
                # strictly more usable capacity, and any gain it shows could be
                # capacity rather than information.
                #
                # Shuffling permutes which agent's intent lands in each slot.
                # The graph, the attention module and every parameter stay in
                # use; only the correspondence between sender and message is
                # destroyed. shuffled ~ full means capacity. shuffled ~ none
                # means the content of the messages is doing the work.
                perm = torch.randperm(N, device=intent.device)
                intent = intent[:, :, perm, :]

            adj_flat = adjacency.reshape(flat, N, N)

            if send_mask is not None:
                sm = send_mask.reshape(flat, N)
            elif self.comm_cfg.mode in ("full", "shuffled"):
                sm = None
            else:
                sm, sched_info = self.scheduler(
                    h.reshape(flat, N, -1), adj_flat, training=self.training)

            ctx, attn = self.comm(
                h.reshape(flat, N, -1),
                intent.reshape(flat, N, N_ACTIONS),
                adj_flat,
                None if rel_pos is None else rel_pos.reshape(flat, N, N, 2),
                send_mask=sm,
            )
            ctx = ctx.view(T, B, N, -1)
            if sched_info is not None:
                sched_info["send"] = sm.view(T, B, N)

        logits = self.policy(torch.cat([h, ctx], dim=-1)).masked_fill(~mask, NEG_INF)

        value_in = h if self.cfg.state_dim == 0 else torch.cat([
            h, torch.zeros(T, B, N, self.cfg.state_dim, device=h.device, dtype=h.dtype)
            if state is None else state], dim=-1)
        value = self.value(value_in).squeeze(-1)

        return logits, value, hidden, tentative_logits, attn, sched_info

    @torch.no_grad()
    def act(self, grid, vec, adjacency, rel_pos=None, hidden=None, state=None,
            send_mask=None, greedy: bool = False):
        """Single timestep. ``grid``: ``(B, N, C, F, F)``."""
        def lift(x):
            return None if x is None else x.unsqueeze(0)
        logits, value, hidden, tent, attn, _ = self.forward(
            grid.unsqueeze(0), vec.unsqueeze(0), adjacency.unsqueeze(0),
            lift(rel_pos), hidden, None, lift(state), lift(send_mask))
        logits, value = logits[0], value[0]
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(-1) if greedy else dist.sample()
        return action, dist.log_prob(action), value, hidden, attn
