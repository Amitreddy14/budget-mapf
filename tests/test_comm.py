"""Tests for intent communication.

These are the tests that keep the central claim falsifiable. If messages are
secretly a no-op, or the no-comm baseline is secretly a different-capacity
model, or range limits are not enforced, then any measured "communication
benefit" is an artefact.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mapf.env import MAPFConfig, VecMAPFEnv
from models.comm import IntentComm
from models.comm_policy import CommActorCritic, CommConfig
from models.encoder import ModelConfig


def _model(mode="full", **kw):
    return CommActorCritic(ModelConfig(state_dim=20), CommConfig(mode=mode, **kw))


def _inputs(T=2, B=3, N=4, device="cpu"):
    g = torch.rand(T, B, N, 6, 9, 9, device=device)
    v = torch.rand(T, B, N, 9, device=device)
    adj = (~torch.eye(N, dtype=torch.bool, device=device)).expand(T, B, N, N).clone()
    rel = torch.rand(T, B, N, N, 2, device=device) * 2 - 1
    st = torch.rand(T, B, N, 20, device=device)
    return g, v, adj, rel, st


# ----------------------------------------------------- experimental validity

def test_no_comm_and_full_comm_have_identical_capacity():
    """The ablation must differ by communication alone, not model size."""
    a = sum(p.numel() for p in _model("none").parameters())
    b = sum(p.numel() for p in _model("full").parameters())
    assert a == b, f"no-comm has {a:,} params, full-comm {b:,} -- not a fair ablation"


def test_communication_actually_changes_the_policy():
    """If messages were a no-op, every reported benefit would be noise."""
    torch.manual_seed(0)
    m = _model("full")
    g, v, adj, rel, st = _inputs()
    with torch.no_grad():
        on, _, _, _, _, _ = m(g, v, adj, rel, state=st)
        m.comm_cfg.mode = "none"
        off, _, _, _, _, _ = m(g, v, adj, rel, state=st)
    assert not torch.allclose(on, off, atol=1e-5), "messages do not affect the policy"


def test_tentative_head_is_independent_of_messages():
    """Pass 1 is the agent's own intent; it must not already contain peer info."""
    torch.manual_seed(0)
    m = _model("full")
    g, v, adj, rel, st = _inputs()
    with torch.no_grad():
        _, _, _, t_all, _, _ = m(g, v, adj, rel, state=st)
        _, _, _, t_none, _, _ = m(g, v, torch.zeros_like(adj), rel, state=st)
    assert torch.allclose(t_all, t_none, atol=1e-6), \
        "tentative intent leaks peer information -- it is not a pass-1 quantity"


# -------------------------------------------------------------- range limits

def test_out_of_range_peers_cannot_be_heard():
    torch.manual_seed(0)
    m = _model("full")
    T, B, N = 1, 1, 3
    g, v, _, rel, st = _inputs(T, B, N)
    # Only agent 1 may talk to agent 0; agent 2 is out of range of everyone.
    adj = torch.zeros(T, B, N, N, dtype=torch.bool)
    adj[0, 0, 0, 1] = True

    with torch.no_grad():
        base, _, _, _, _, _ = m(g, v, adj, rel, state=st)
        g2 = g.clone()
        g2[0, 0, 2] = torch.rand(6, 9, 9)        # change the unheard agent
        moved, _, _, _, _, _ = m(g2, v, adj, rel, state=st)
    assert torch.allclose(base[0, 0, 0], moved[0, 0, 0], atol=1e-5), \
        "agent 0 was influenced by an out-of-range peer"


def test_isolated_agent_produces_finite_output():
    """Softmax over an all-masked row must not yield NaN or a uniform fiction."""
    m = _model("full")
    T, B, N = 1, 1, 3
    g, v, _, rel, st = _inputs(T, B, N)
    adj = torch.zeros(T, B, N, N, dtype=torch.bool)   # nobody can hear anybody
    logits, value, _, _, attn, _ = m(g, v, adj, rel, state=st)
    assert torch.isfinite(logits[logits > -1e8]).all()
    assert torch.isfinite(value).all()
    assert float(attn.sum()) == pytest.approx(0.0, abs=1e-6), \
        "isolated agents received attention mass from nowhere"


# ------------------------------------------------------- scheduler interface

def test_send_mask_silences_an_agent():
    torch.manual_seed(0)
    m = _model("full")
    T, B, N = 1, 1, 4
    g, v, adj, rel, st = _inputs(T, B, N)
    send = torch.ones(T, B, N)
    send[0, 0, 2] = 0.0                                # agent 2 stays silent

    with torch.no_grad():
        base, _, _, _, _, _ = m(g, v, adj, rel, state=st, send_mask=send)
        g2 = g.clone()
        g2[0, 0, 2] = torch.rand(6, 9, 9)              # change the silent agent
        moved, _, _, _, _, _ = m(g2, v, adj, rel, state=st, send_mask=send)
    others = [0, 1, 3]
    assert torch.allclose(base[0, 0, others], moved[0, 0, others], atol=1e-5), \
        "a silenced agent still influenced its peers"


def test_gradients_flow_through_send_mask():
    """Phase 6 needs this: the scheduler learns through the transmit decision."""
    m = _model("full")
    g, v, adj, rel, st = _inputs(T=1, B=2, N=4)
    send = torch.full((1, 2, 4), 0.5, requires_grad=True)
    logits, _, _, _, _, _ = m(g, v, adj, rel, state=st, send_mask=send)
    logits.masked_fill(logits < -1e8, 0.0).sum().backward()
    assert send.grad is not None and float(send.grad.abs().sum()) > 0, \
        "no gradient reaches send_mask -- a learned scheduler cannot train"


def test_attention_is_a_distribution_over_audible_peers():
    m = IntentComm(hidden=32, msg_dim=16, n_heads=2)
    B, N = 2, 5
    h = torch.rand(B, N, 32)
    intent = torch.softmax(torch.rand(B, N, 5), -1)
    adj = (~torch.eye(N, dtype=torch.bool)).expand(B, N, N).clone()
    _, attn = m(h, intent, adj, torch.rand(B, N, N, 2))
    sums = attn.sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    diag = attn[:, :, torch.arange(N), torch.arange(N)]
    assert float(diag.abs().max()) == pytest.approx(0.0, abs=1e-6), \
        "an agent is attending to its own message"


# --------------------------------------------------------- env integration

def test_env_comm_graph_respects_radius_and_is_symmetric():
    cfg = MAPFConfig(n_agents=8, map_spec={"kind": "empty", "height": 20, "width": 20})
    env = VecMAPFEnv(cfg, num_envs=4, seed=0)
    for radius in (1, 3, 8):
        adj, rel = env.comm_graph(radius)
        assert adj.shape == (4, 8, 8) and rel.shape == (4, 8, 8, 2)
        assert not adj[:, np.arange(8), np.arange(8)].any(), "self-edge present"
        assert np.array_equal(adj, adj.transpose(0, 2, 1)), "graph is not symmetric"
        cheb = np.abs(env.pos[:, None, :, :] - env.pos[:, :, None, :]).max(-1)
        expected = (cheb <= radius) & ~np.eye(8, dtype=bool)[None]
        assert np.array_equal(adj, expected)


def test_comm_graph_relative_positions_point_the_right_way():
    cfg = MAPFConfig(n_agents=2, map_spec={"kind": "empty", "height": 10, "width": 10})
    env = VecMAPFEnv(cfg, num_envs=1, seed=0)
    env.pos[0] = np.array([[2, 2], [2, 5]])          # agent 1 is 3 cells right
    _, rel = env.comm_graph(8)
    assert rel[0, 0, 1, 1] > 0, "peer to the right should have positive column offset"
    assert rel[0, 1, 0, 1] < 0, "the reverse offset should be negative"


# --- value_of_information vs policy_shift ----------------------------------
# policy_shift (KL) measures how much a message changes behaviour. It scores
# a trained-but-useless channel just as highly as a useful one, which is how
# selection_advantage reached 1.42 on a channel the BC shuffle control had
# already shown to be worthless. value_of_information weights the shift by
# advantage, so it is positive only when the message moves mass toward
# actions the critic rates above average.

def test_value_of_information_is_positive_when_the_message_helps():
    import torch
    from models.voi import value_of_information
    adv = torch.tensor([1.0, 1.0, 1.0])
    # Hearing the message raised the log-prob of the action taken, and the
    # action taken was better than average.
    v = value_of_information(torch.tensor([-0.5, -0.4, -0.3]),
                             torch.tensor([-1.0, -1.0, -1.0]), adv)
    assert v.mean() > 0


def test_value_of_information_is_negative_when_it_hurts():
    import torch
    from models.voi import value_of_information
    adv = torch.tensor([1.0, 1.0])
    v = value_of_information(torch.tensor([-1.5, -1.6]),
                             torch.tensor([-1.0, -1.0]), adv)
    assert v.mean() < 0


def test_value_of_information_is_zero_when_the_shift_ignores_advantage():
    """A large shift uncorrelated with action quality must not score."""
    import torch
    from models.voi import value_of_information
    torch.manual_seed(0)
    shift = torch.randn(4096)              # big, random shifts
    adv = torch.randn(4096)                # independent of the shift
    v = value_of_information(shift, torch.zeros(4096), adv)
    assert abs(float(v.mean())) < 0.05


def test_policy_shift_cannot_tell_those_cases_apart():
    """The flaw this metric exists to fix, pinned as a test."""
    import torch
    from models.voi import policy_shift
    base = torch.tensor([[[2.0, 0.0, 0.0, 0.0, 0.0]]])
    helpful = torch.tensor([[[3.0, 0.0, 0.0, 0.0, 0.0]]])
    harmful = torch.tensor([[[1.0, 1.0, 0.0, 0.0, 0.0]]])
    # Both differ from base; KL alone gives no sign of which is an improvement.
    assert float(policy_shift(helpful, base).mean()) > 0
    assert float(policy_shift(harmful, base).mean()) > 0


# --- VoI must be measured on TRAINED iterations ----------------------------
# The diagnostic was first attached to `critic_only` as an `elif`, so it ran
# only during value warmup: a 60-iteration run logged exactly two readings,
# both from before the policy had been trained. A structural test is not
# enough here -- the bug was in which branch the code sat in, and only
# running it catches that.

import pytest


@pytest.mark.slow
def test_voi_is_logged_after_the_value_warmup():
    from train.ppo_comm import CommPPOConfig, train_comm_ppo

    cfg = CommPPOConfig(
        n_agents=4,
        map_spec={"kind": "random", "height": 12, "width": 12, "density": 0.15},
        max_steps=24, num_envs=4, map_pool_size=8, rollout=8,
        total_agent_steps=4 * 4 * 8 * 14, value_warmup_iters=2,
        comm_mode="full", voi_every=1, log_every=1, eval_every=10_000,
        warm_start=None, device="cpu", out_dir="runs/_test_voi",
    )
    hist = train_comm_ppo(cfg)["history"]

    trained = [r for r in hist
               if not r.get("critic_only") and r.get("voi_mean") is not None]
    assert trained, ("voi_mean was never logged on a trained iteration -- "
                     "the diagnostic is in the wrong branch again")
    assert all("voi_value" in r for r in trained), "voi_value must accompany it"

    warmup_only = [r for r in hist if r.get("critic_only")
                   and r.get("voi_mean") is not None]
    assert len(trained) > len(warmup_only), (
        f"only {len(trained)} trained readings against {len(warmup_only)} "
        f"warmup ones -- VoI is still gated on the warmup branch")
