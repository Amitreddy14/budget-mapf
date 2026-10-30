"""Per-agent observation encoder and actor-critic heads.

Architecture
------------
    FOV grid (6 x 9 x 9) ──► 4 valid-padding convs ──► 128
                        └──► centre 3x3 skip ───────►  54
    vector features ────────────────────────────────►   9
                                                       ───
                                                       191 ──► MLP ──► GRU(256)
                                                                        │
                                                            ┌───────────┴───────────┐
                                                         policy (5)            value (1)

Two deliberate choices:

* **No pooling.** A 9x9 window is small and the decision hinges on exactly
  which adjacent cell is free. Max-pooling blurs precisely the information
  that matters, so the convs use valid padding to walk the window down to
  1x1 in steps of 2 (9 -> 7 -> 5 -> 3 -> 1 at the default fov).
* **Centre 3x3 skip connection.** Collapsing to 1x1 forces neighbour
  occupancy through many channels. Feeding the agent's immediate
  neighbourhood straight to the MLP makes "is the cell I want to enter free?"
  trivially available.

Illegal actions are masked from the policy logits using the obstacle channel,
so the network never has to spend capacity learning the map boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

N_ACTIONS = 5
CH_OBSTACLE = 0
# (dr, dc) per action, matching mapf.env.ACTION_DELTA.
_DELTA = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
NEG_INF = -1e9


@dataclass
class ModelConfig:
    in_channels: int = 6
    fov: int = 9
    vec_dim: int = 9
    hidden: int = 256
    gru_layers: int = 1
    use_gru: bool = True
    mask_illegal: bool = True
    state_dim: int = 0          # >0 enables the centralised critic (CTDE)


def legal_action_mask(grid: torch.Tensor) -> torch.Tensor:
    """Derive a legal-action mask from the obstacle channel.

    ``grid``: ``(M, C, F, F)``. Returns ``(M, 5)`` bool, True == permitted.
    STAY is always legal. Out-of-bounds cells are encoded as obstacles by the
    environment's padding, so bounds need no special handling.
    """
    centre = grid.shape[-1] // 2
    obstacles = grid[:, CH_OBSTACLE]
    mask = torch.ones(grid.shape[0], N_ACTIONS, dtype=torch.bool, device=grid.device)
    for a, (dr, dc) in enumerate(_DELTA):
        if a == 0:
            continue
        mask[:, a] = obstacles[:, centre + dr, centre + dc] < 0.5
    return mask


def _conv_widths(fov: int):
    """Channel widths for the valid-padding conv stack that walks `fov` -> 1.

    fov=9 reproduces the original [32, 64, 64, 128] exactly; smaller windows
    drop leading layers, larger ones repeat the 32-channel entry. The stack
    always ends at 128 so the projection head is independent of `fov`.
    """
    if fov < 3 or fov % 2 == 0:
        raise ValueError(f"fov must be odd and at least 3, got {fov}")
    n = (fov - 1) // 2
    base = [32, 64, 64, 128]
    return base[-n:] if n <= len(base) else [32] * (n - len(base)) + base


class AgentEncoder(nn.Module):
    """Maps one agent's observation to a hidden state."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg.in_channels
        # The stack was written as four hard-coded convs for fov=9. Any other
        # window size then died inside conv2d with "kernel size can't be
        # greater than actual input size", which is how fov=5 first failed.
        # Each valid 3x3 conv removes 2, so walking F -> 1 takes (F-1)//2 of
        # them, and the stack always ends at 128 channels so `proj` is
        # unchanged.
        widths = _conv_widths(cfg.fov)
        layers = []
        prev = c
        for w in widths:
            layers += [nn.Conv2d(prev, w, 3), nn.ReLU(inplace=True)]
            prev = w
        self.conv = nn.Sequential(*layers)
        skip_dim = c * 9                                      # centre 3x3 patch
        self.proj = nn.Sequential(
            nn.Linear(128 + skip_dim + cfg.vec_dim, cfg.hidden), nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden, cfg.hidden), nn.ReLU(inplace=True),
        )

    def forward(self, grid: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        centre = grid.shape[-1] // 2
        patch = grid[:, :, centre - 1:centre + 2, centre - 1:centre + 2]
        feats = torch.cat([
            self.conv(grid).flatten(1),
            patch.flatten(1),
            vec,
        ], dim=-1)
        return self.proj(feats)


class ActorCritic(nn.Module):
    """Recurrent policy and centralised-input value head.

    Shapes. The caller flattens environments and agents into one axis ``M``.
    Sequence calls take ``(T, M, ...)`` and carry a GRU state of ``(L, M, H)``.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = AgentEncoder(cfg)
        if cfg.use_gru:
            self.gru = nn.GRU(cfg.hidden, cfg.hidden, num_layers=cfg.gru_layers)
        else:
            self.gru = None
        self.policy = nn.Linear(cfg.hidden, N_ACTIONS)
        # Centralised critic: the value head additionally sees the global
        # state, which the actor never does. This is the "CT" of CTDE -- the
        # extra information shapes learning but never reaches execution.
        self.value = nn.Sequential(
            nn.Linear(cfg.hidden + cfg.state_dim, cfg.hidden), nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden, 1),
        )
        self.apply(self._init)
        # Small final policy weights keep the initial distribution near uniform.
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.zeros_(self.policy.bias)

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
        grid: torch.Tensor,
        vec: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        resets: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the network over a sequence.

        Args:
            grid: ``(T, M, C, F, F)``
            vec:  ``(T, M, V)``
            hidden: ``(L, M, H)`` or None for zeros.
            resets: ``(T, M)`` bool -- True zeroes the GRU state before that step.
            state: ``(T, M, state_dim)`` centralised state for the critic.

        Returns ``(logits, value, hidden)`` with logits ``(T, M, 5)``.
        """
        T, M = grid.shape[:2]
        feats = self.encoder(grid.reshape(T * M, *grid.shape[2:]),
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

        logits = self.policy(core)
        if self.cfg.mask_illegal:
            mask = legal_action_mask(grid.reshape(T * M, *grid.shape[2:])).view(T, M, N_ACTIONS)
            logits = logits.masked_fill(~mask, NEG_INF)

        if self.cfg.state_dim > 0:
            if state is None:
                state = torch.zeros(T, M, self.cfg.state_dim, device=grid.device,
                                    dtype=core.dtype)
            value_in = torch.cat([core, state], dim=-1)
        else:
            value_in = core
        return logits, self.value(value_in).squeeze(-1), hidden

    @torch.no_grad()
    def act(self, grid, vec, hidden=None, greedy: bool = False, state=None):
        """Single-timestep convenience wrapper. ``grid``: ``(M, C, F, F)``."""
        logits, value, hidden = self.forward(
            grid.unsqueeze(0), vec.unsqueeze(0), hidden,
            state=None if state is None else state.unsqueeze(0))
        logits, value = logits[0], value[0]
        if greedy:
            action = logits.argmax(-1)
        else:
            action = torch.distributions.Categorical(logits=logits).sample()
        return action, value, hidden

    @torch.no_grad()
    def act_with_logp(self, grid, vec, hidden=None, state=None):
        """Sample an action and return its log-probability, for PPO rollouts."""
        logits, value, hidden = self.forward(
            grid.unsqueeze(0), vec.unsqueeze(0), hidden,
            state=None if state is None else state.unsqueeze(0))
        dist = torch.distributions.Categorical(logits=logits[0])
        action = dist.sample()
        return action, dist.log_prob(action), value[0], hidden


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
