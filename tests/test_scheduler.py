"""Tests for budgeted scheduling and the value-of-information signal.

The claim Phase 6 makes is that *selective* communication keeps the collision
benefit without the throughput cost. That claim is only meaningful if the
budget genuinely binds, the baselines are budget-matched, and the learned
scheduler can actually receive gradient. Each is tested here.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

torch = pytest.importorskip("torch")

from models.comm_policy import CommActorCritic, CommConfig
from models.encoder import ModelConfig
from models.scheduler import BudgetedScheduler, SchedulerConfig
from models.voi import (policy_shift, scheduler_loss, sender_value,
                        voi_summary)


def _sched(**kw):
    return BudgetedScheduler(SchedulerConfig(hidden=32, **kw))


def _h(b=4, n=8, d=32):
    return torch.randn(b, n, d)


# ----------------------------------------------------------------- the budget

def test_topk_sends_exactly_the_budgeted_number():
    n, frac = 12, 0.25
    s = _sched(allocation="topk", budget_frac=frac)
    send, _ = s(_h(n=n), training=True)
    expected = max(1, round(frac * n))
    counts = send.sum(-1)
    assert torch.allclose(counts, torch.full_like(counts, float(expected))), \
        f"budget not binding: sent {counts.tolist()}, expected {expected}"


def test_topk_budget_binds_at_evaluation_too():
    s = _sched(allocation="topk", budget_frac=0.5)
    send, _ = s(_h(n=10), training=False)
    assert (send.sum(-1) == 5).all()


def test_rate_mode_converges_to_the_budget_under_dual_ascent():
    """The Lagrangian must pull an over-talkative scheduler down to budget."""
    torch.manual_seed(0)
    target = 0.25
    s = _sched(allocation="rate", budget_frac=target, lambda_lr=0.5)
    # Start strongly biased toward transmitting.
    with torch.no_grad():
        s.scorer[-1].bias.fill_(3.0)

    opt = torch.optim.SGD(s.parameters(), lr=0.5)
    rates, peak_lam = [], 0.0
    for _ in range(300):
        h = _h(b=16, n=8)
        send, info = s(h, training=True)
        loss = s.budget_penalty(send)
        opt.zero_grad(); loss.backward(); opt.step()
        peak_lam = max(peak_lam, s.dual_step(float(info["rate"])))
        rates.append(float(info["rate"]))

    send, info = s(_h(b=64, n=8), training=False)
    assert rates[0] > target + 0.2, "fixture failed: scheduler did not start over budget"
    assert float(info["rate"]) < target + 0.15, \
        f"dual ascent failed to enforce the budget: rate={float(info['rate']):.3f}"
    # The multiplier is projected to be non-negative and legitimately returns
    # to zero once the constraint stops binding, so check that it rose while
    # the scheduler was over budget rather than where it finished.
    assert peak_lam > 0, "multiplier never rose despite starting over budget"


def test_straight_through_is_hard_forward_and_differentiable_backward():
    for allocation in ("topk", "rate"):
        s = _sched(allocation=allocation)
        h = _h().requires_grad_(True)
        send, _ = s(h, training=True)
        assert torch.all((send == 0) | (send == 1)), \
            f"{allocation}: forward mask is not hard 0/1 -- the budget is fictional"
        send.sum().backward()
        assert h.grad is not None and h.grad.abs().sum() > 0, \
            f"{allocation}: no gradient reaches the scorer"


# -------------------------------------------------------------- the baselines

def test_random_baseline_matches_the_budget():
    """Budget-matched randomness is what isolates 'learned anything at all'."""
    torch.manual_seed(0)
    s = _sched(mode="random", allocation="topk", budget_frac=0.25)
    send, _ = s(_h(b=64, n=16), training=True)
    assert (send.sum(-1) == 4).all()

    s = _sched(mode="random", allocation="rate", budget_frac=0.25)
    send, info = s(_h(b=256, n=16), training=True)
    assert abs(float(info["rate"]) - 0.25) < 0.05


def test_all_and_none_modes_are_degenerate_as_expected():
    assert _sched(mode="all")(_h(), training=True)[0].min() == 1.0
    assert _sched(mode="none")(_h(), training=True)[0].max() == 0.0


def test_distance_baseline_prefers_crowded_agents():
    s = _sched(mode="distance", budget_frac=0.25)
    n = 8
    adj = torch.zeros(1, n, n, dtype=torch.bool)
    adj[0, 0, 1:6] = True               # agent 0 has many neighbours
    adj[0, 7, 6] = True                 # agent 7 has one
    send, _ = s(_h(b=1, n=n), adj, training=False)
    assert send[0, 0] == 1.0, "crowded agent was not selected"


def test_every_scheduler_mode_has_the_same_parameter_count():
    counts = {}
    for mode in ("none", "full", "learned", "random", "distance"):
        m = CommActorCritic(ModelConfig(state_dim=40), CommConfig(mode=mode))
        counts[mode] = sum(p.numel() for p in m.parameters())
    assert len(set(counts.values())) == 1, f"capacity differs across arms: {counts}"


# --------------------------------------------------------------------- VoI

def test_identical_policies_have_zero_value_of_information():
    x = torch.randn(3, 5, 5)
    assert policy_shift(x, x).abs().max() < 1e-6


def test_value_of_information_is_positive_when_the_policy_moves():
    a = torch.tensor([[[2.0, 0.0, 0.0, 0.0, 0.0]]])
    b = torch.tensor([[[0.0, 2.0, 0.0, 0.0, 0.0]]])
    assert float(policy_shift(a, b)) > 0.5


def test_masked_actions_do_not_produce_nan():
    """Illegal actions use -1e9, not -inf, precisely so this stays finite."""
    a = torch.tensor([[[1.0, -1e9, 0.5, -1e9, 0.0]]])
    b = torch.tensor([[[0.0, -1e9, 1.5, -1e9, 0.2]]])
    out = policy_shift(a, b)
    assert torch.isfinite(out).all(), "masked logits produced NaN"


def test_sender_value_follows_attention():
    """A sender nobody attends to earns no credit."""
    m, n, heads = 1, 3, 2
    voi = torch.tensor([[1.0, 0.0, 0.0]])            # only receiver 0 was informed
    attn = torch.zeros(m, heads, n, n)
    attn[:, :, 0, 1] = 1.0                           # receiver 0 attended only to 1
    val = sender_value(voi, attn)
    assert val[0, 1] > 0.9, "the attended sender got no credit"
    assert val[0, 0] < 1e-6 and val[0, 2] < 1e-6, "unattended senders got credit"


def test_scheduler_loss_rewards_correct_ranking():
    values = torch.tensor([[3.0, 0.0, 0.0, 0.0]])
    good = torch.tensor([[5.0, 0.0, 0.0, 0.0]])      # agrees with the ranking
    bad = torch.tensor([[0.0, 0.0, 0.0, 5.0]])       # ranks the wrong agent first
    assert scheduler_loss(good, values) < scheduler_loss(bad, values)


def test_voi_captured_measures_selection_quality():
    """At the budget fraction this equals chance; above it means something."""
    voi = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    perfect = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    wrong = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    assert voi_summary(voi, perfect)["voi_captured"] == pytest.approx(1.0)
    assert voi_summary(voi, wrong)["voi_captured"] == pytest.approx(0.0)


def test_learned_scheduler_runs_inside_the_policy():
    m = CommActorCritic(ModelConfig(state_dim=40),
                        CommConfig(mode="learned", budget_frac=0.25))
    T, B, N = 2, 3, 8
    grid, vec = torch.rand(T, B, N, 6, 9, 9), torch.rand(T, B, N, 9)
    adj = (~torch.eye(N, dtype=torch.bool)).expand(T, B, N, N).clone()
    logits, value, _, tent, attn, sched = m(grid, vec, adj, state=torch.rand(T, B, N, 40))
    assert logits.shape == (T, B, N, 5)
    assert sched is not None and "send" in sched
    assert sched["send"].shape == (T, B, N)
    assert torch.all((sched["send"] == 0) | (sched["send"] == 1))
