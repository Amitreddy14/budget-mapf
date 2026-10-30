"""QMIX mixing network, for testing whether the communication result is
specific to the policy-gradient learner.

Every negative result in this project was produced with MAPPO. A reviewer's
first question about a negative result is whether it is a property of the
problem or of the algorithm, so the second backbone has to differ in the
learning rule and in nothing else.

That is why there is no agent network in this file. The per-agent Q-function
is `models.comm_policy.CommActorCritic` used unchanged, with its policy head
read as action values rather than logits. The encoder, the recurrent core,
the intent head, the attention pathway, the scheduler and the shuffled
control are therefore bit-for-bit the same modules the MAPPO experiments
used, with the same parameter count, and the arms mean exactly what they
meant before.

The mixer below is the only new component.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MixerConfig:
    n_agents: int = 16
    state_dim: int = 80        # env.global_state() is (B, N * 5)
    embed: int = 32
    hyper_hidden: int = 64


class QMixer(nn.Module):
    """Monotonic mixing of per-agent action values into a joint value.

    QMIX factorises the joint action-value as a mixing network whose weights
    are produced by hypernetworks conditioned on the global state, under the
    constraint that the mixture is monotone in every agent's value:

        d Q_tot / d Q_i  >=  0   for all i

    Monotonicity is what makes decentralised execution sound: the argmax of
    the joint value is then the vector of per-agent argmaxes, so an agent can
    act on its own Q-values alone. It is enforced by taking the absolute
    value of the hypernetwork outputs that act as weights. The biases are
    unconstrained, which is why `hyper_b2` may be a two-layer network while
    the weight hypernetworks must not introduce a sign.
    """

    def __init__(self, cfg: MixerConfig):
        super().__init__()
        self.cfg = cfg
        n, s, e, h = cfg.n_agents, cfg.state_dim, cfg.embed, cfg.hyper_hidden
        self.hyper_w1 = nn.Sequential(nn.Linear(s, h), nn.ReLU(inplace=True),
                                      nn.Linear(h, n * e))
        self.hyper_w2 = nn.Sequential(nn.Linear(s, h), nn.ReLU(inplace=True),
                                      nn.Linear(h, e))
        self.hyper_b1 = nn.Linear(s, e)
        self.hyper_b2 = nn.Sequential(nn.Linear(s, e), nn.ReLU(inplace=True),
                                      nn.Linear(e, 1))

    def forward(self, agent_q: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """``agent_q``: ``(B, N)`` chosen-action values. ``state``: ``(B, S)``.

        Returns ``(B,)``.
        """
        if agent_q.shape[-1] != self.cfg.n_agents:
            raise ValueError(f"expected {self.cfg.n_agents} agents, "
                             f"got {agent_q.shape[-1]}")
        if state.shape[-1] != self.cfg.state_dim:
            raise ValueError(f"expected state_dim {self.cfg.state_dim}, "
                             f"got {state.shape[-1]}")
        B, n, e = agent_q.shape[0], self.cfg.n_agents, self.cfg.embed

        w1 = self.hyper_w1(state).abs().view(B, n, e)
        b1 = self.hyper_b1(state).view(B, 1, e)
        hidden = F.elu(torch.bmm(agent_q.view(B, 1, n), w1) + b1)

        w2 = self.hyper_w2(state).abs().view(B, e, 1)
        b2 = self.hyper_b2(state).view(B, 1, 1)
        return (torch.bmm(hidden, w2) + b2).view(B)


def greedy_from_q(q: torch.Tensor) -> torch.Tensor:
    """Per-agent greedy action from masked Q-values. ``q``: ``(..., A)``."""
    return q.argmax(dim=-1)


def q_influence(q_comm: torch.Tensor, q_silent: torch.Tensor) -> torch.Tensor:
    """How far hearing neighbours moves the action values. ``(..., A)`` in.

    The policy-gradient analogue is `models.voi.policy_shift`, a KL between
    two distributions. A Q-network has no distribution, so influence is the
    mean absolute change in action value, in the units of the value function.
    Like the KL, it is blind to whether the change is an improvement.
    """
    return (q_comm - q_silent).abs().mean(dim=-1)


def q_value_of_information(q_comm: torch.Tensor, q_silent: torch.Tensor,
                           action: torch.Tensor) -> torch.Tensor:
    """Does hearing neighbours raise the value of the action actually taken?

    The counterpart of `models.voi.value_of_information`. Positive only when
    the message moves the chosen action's value up relative to the mean value
    of the alternatives, so a uniform shift in all action values -- which
    changes no decision -- scores zero.

    ``q_comm``, ``q_silent``: ``(..., A)``; ``action``: ``(...)``.
    """
    idx = action.unsqueeze(-1)
    adv_comm = q_comm.gather(-1, idx).squeeze(-1) - q_comm.mean(dim=-1)
    adv_silent = q_silent.gather(-1, idx).squeeze(-1) - q_silent.mean(dim=-1)
    return adv_comm - adv_silent
