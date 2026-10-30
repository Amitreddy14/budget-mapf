"""Intent communication between agents.

Why intent, not observations
----------------------------
Phase 2 measured this directly (`docs/FINDINGS-phase2.md`). Essentially 100%
of expert conflict decisions involve an agent at Chebyshev distance 1 -- one
the agent can already see. Widening the field of view buys nothing. What an
agent cannot observe at *any* range is what its neighbour is about to *do*.
Two agents can see each other perfectly and still step into the same cell.

An intent-broadcast oracle cut collisions 33x at 16 agents, so the signal is
real. This module learns to use it.

Mechanism
---------
Two passes per timestep:

1. Each agent forms a tentative action distribution from its own observation.
2. It broadcasts that intent; receivers attend over the intents of peers in
   range and revise their decision.

    h_i ──► tentative logits ──► intent m_i = f([h_i, softmax(logits)])
                                     │
                     ┌───────────────┴──────────────┐
                     ▼                              ▼
             peers within radius R          relative offsets (j - i)
                     └──────────► multi-head attention ◄──────┘
                                     │
                              context c_i
                                     │
            final logits = policy([h_i, c_i])

``send_mask`` gates which agents actually transmit. Phase 5 uses all-ones
(full communication) or all-zeros (none); Phase 6 replaces it with the learned
budgeted scheduler, so nothing downstream has to change.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

N_ACTIONS = 5
NEG_INF = -1e9


class IntentComm(nn.Module):
    """Range-limited multi-head attention over broadcast intents."""

    def __init__(self, hidden: int, msg_dim: int = 64, n_heads: int = 4,
                 use_rel_pos: bool = True, dropout: float = 0.0):
        super().__init__()
        if msg_dim % n_heads:
            raise ValueError(f"msg_dim {msg_dim} must divide by n_heads {n_heads}")
        self.hidden, self.msg_dim, self.n_heads = hidden, msg_dim, n_heads
        self.head_dim = msg_dim // n_heads
        self.use_rel_pos = use_rel_pos

        rel = 2 if use_rel_pos else 0
        self.msg_encoder = nn.Sequential(
            nn.Linear(hidden + N_ACTIONS, msg_dim), nn.ReLU(inplace=True),
            nn.Linear(msg_dim, msg_dim),
        )
        self.q = nn.Linear(hidden, msg_dim)
        self.k = nn.Linear(msg_dim + rel, msg_dim)
        self.v = nn.Linear(msg_dim + rel, msg_dim)
        self.out = nn.Linear(msg_dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        h: torch.Tensor,
        intent: torch.Tensor,
        adjacency: torch.Tensor,
        rel_pos: Optional[torch.Tensor] = None,
        send_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aggregate peer intents into a per-agent context vector.

        Args:
            h:         ``(B, N, H)`` agent hidden states.
            intent:    ``(B, N, 5)`` tentative action probabilities.
            adjacency: ``(B, N, N)`` bool, True where j may send to i.
            rel_pos:   ``(B, N, N, 2)`` j's offset from i.
            send_mask: ``(B, N)`` float/bool, 1 where agent j transmits.
                       Differentiable values are respected, so a
                       straight-through scheduler can pass gradients here.

        Returns ``(context, attention)`` with context ``(B, N, H)`` and
        attention ``(B, heads, N, N)`` for analysis.
        """
        B, N, _ = h.shape
        msg = self.msg_encoder(torch.cat([h, intent], dim=-1))        # (B, N, D)

        # Broadcast each sender's message to every receiver.
        kv = msg.unsqueeze(1).expand(B, N, N, self.msg_dim)
        if self.use_rel_pos:
            if rel_pos is None:
                rel_pos = torch.zeros(B, N, N, 2, device=h.device, dtype=h.dtype)
            kv = torch.cat([kv, rel_pos], dim=-1)

        q = self.q(h).view(B, N, self.n_heads, self.head_dim)
        k = self.k(kv).view(B, N, N, self.n_heads, self.head_dim)
        v = self.v(kv).view(B, N, N, self.n_heads, self.head_dim)

        scores = torch.einsum("bihd,bijhd->bhij", q, k) / math.sqrt(self.head_dim)

        mask = adjacency
        if send_mask is not None:
            mask = mask & (send_mask > 0).unsqueeze(1)
        scores = scores.masked_fill(~mask.unsqueeze(1), NEG_INF)

        # An agent with no audible peers gets a uniform row from softmax over
        # all -inf, which is meaningless. Detect and zero those rows instead.
        has_peers = mask.any(-1)                                       # (B, N)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.where(has_peers.unsqueeze(1).unsqueeze(-1), attn,
                           torch.zeros_like(attn))

        if send_mask is not None:
            # Let gradients reach a differentiable scheduler: weight each
            # sender's contribution by its (possibly soft) transmit value.
            attn = attn * send_mask.to(attn.dtype)[:, None, None, :]

        ctx = torch.einsum("bhij,bijhd->bihd", self.drop(attn), v)
        ctx = self.out(ctx.reshape(B, N, self.msg_dim))
        return self.norm(ctx), attn
