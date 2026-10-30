"""Counterfactual value of information.

The v1 project's communication gate never learned, for one concrete reason:
its only training signal was the task reward, which is far too sparse and
delayed to say whether *this* message, at *this* step, was worth sending.

This module supplies a dense signal instead. For each receiver we measure how
much its policy actually changes when the messages arrive:

    VoI_i = KL( pi_i(. | obs, messages)  ||  pi_i(. | obs, silence) )

A receiver whose action is unchanged learned nothing; one whose distribution
shifts was genuinely informed. Credit is then assigned to senders through the
attention weights the receiver used:

    VoI(j -> i) = attention[i, j] * VoI_i
    value(j)    = sum_i VoI(j -> i)

The scheduler is trained to rank agents by that value. Cost is one extra
forward pass per update — not one per candidate sender, which would be
quadratic and unaffordable.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def policy_shift(logits_with: torch.Tensor,
                 logits_without: torch.Tensor) -> torch.Tensor:
    """KL(with || without) per agent. Logits ``(..., A)`` -> ``(...)``.

    Illegal actions carry logits of -1e9 rather than -inf, so their
    probability underflows to exactly zero while the log-ratio stays finite;
    the product contributes 0 instead of NaN. Using -inf here would poison the
    whole batch, which is why the mask value matters.
    """
    lp = F.log_softmax(logits_with, dim=-1)
    lq = F.log_softmax(logits_without, dim=-1)
    return (lp.exp() * (lp - lq)).sum(-1)


def sender_value(voi_receiver: torch.Tensor,
                 attention: torch.Tensor) -> torch.Tensor:
    """Attribute receiver-side VoI back to senders.

    Args:
        voi_receiver: ``(M, N)`` -- KL shift per receiver.
        attention: ``(M, heads, N, N)`` indexed ``[receiver, sender]``.

    Returns ``(M, N)`` value per sender.
    """
    attn = attention.mean(1)                                   # (M, N, N)
    return torch.einsum("mij,mi->mj", attn, voi_receiver)


def scheduler_loss(scores: torch.Tensor,
                   values: torch.Tensor,
                   temperature: float = 1.0,
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Train transmit scores to rank agents by measured value.

    Matching distributions rather than regressing raw numbers keeps this
    scale-free: VoI magnitudes drift by orders of magnitude over training,
    but the *ordering* of who is worth hearing is what the scheduler needs.
    """
    if mask is not None:
        neg = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~mask, neg)
        values = values.masked_fill(~mask, neg)
    target = F.softmax(values.detach() / temperature, dim=-1)
    logp = F.log_softmax(scores, dim=-1)
    return F.kl_div(logp, target, reduction="batchmean")


def voi_summary(voi_receiver: torch.Tensor, send: torch.Tensor) -> dict:
    """Diagnostics: is the scheduler selecting the agents that matter?

    ``precision`` is the share of realised VoI captured by the agents actually
    chosen to transmit. At the budget fraction it equals random selection;
    above it, the scheduler is doing something.
    """
    with torch.no_grad():
        total = voi_receiver.sum().clamp(min=1e-9)
        return {
            "voi_mean": voi_receiver.mean().item(),
            "voi_max": voi_receiver.max().item(),
            "send_rate": send.mean().item(),
            "voi_captured": (voi_receiver * send).sum().div(total).item(),
        }


def value_of_information(logp_with, logp_without, advantages):
    """Does hearing the message move probability toward *better* actions?

    `policy_shift` is a KL: it measures how much a message changes behaviour,
    not whether the change is an improvement. Any trained-but-uninformative
    attention module perturbs the policy and therefore scores a high KL. The
    BC shuffle control showed exactly that -- permuting which agent sent which
    intent left validation loss unchanged to three decimals, so the channel
    carries no information -- while the same checkpoint entered PPO with
    `policy_shift` = 0.044 nats and a scheduler that reached 1.42x chance at
    predicting it.

    Both facts are consistent: the scheduler learned to predict which messages
    are *influential*, which is a property of the state and perfectly
    learnable, and says nothing about whether the influence helps.

    This is the advantage-weighted log-ratio of the action actually taken. It
    is positive only when the message moves mass toward actions the critic
    rates above average, and ~0 when the shift is unrelated to action quality.
    Use `policy_shift` to decide what to transmit; use this to decide whether
    transmitting is worth anything at all.
    """
    return advantages * (logp_with - logp_without)
