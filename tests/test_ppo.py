"""Tests for the PPO machinery -- mostly GAE, which is easy to get subtly wrong."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models.encoder import ActorCritic, ModelConfig
from train.ppo import PPOConfig, Rollout, compute_gae, train_ppo


def _roll(rewards, values, dones, device="cpu"):
    T = len(rewards)
    r = Rollout(T, 1, (6, 9, 9), 9, 4, torch.device(device))
    r.reward[:, 0] = torch.tensor(rewards, dtype=torch.float32)
    r.value[:, 0] = torch.tensor(values, dtype=torch.float32)
    r.done[:, 0] = torch.tensor(dones, dtype=torch.float32)
    return r


def test_gae_matches_hand_computation():
    gamma, lam = 0.99, 0.95
    rewards, values, dones = [1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]
    last = torch.tensor([0.5])
    adv, ret = compute_gae(_roll(rewards, values, dones), last, gamma, lam)

    # Reference implementation, written independently of the one under test.
    exp, running, nv = [0.0] * 3, 0.0, 0.5
    for t in reversed(range(3)):
        nt = 1.0 - dones[t]
        delta = rewards[t] + gamma * nv * nt - values[t]
        running = delta + gamma * lam * nt * running
        exp[t] = running
        nv = values[t]
    assert np.allclose(adv[:, 0].numpy(), exp, atol=1e-5)
    assert np.allclose(ret[:, 0].numpy(), np.array(exp) + np.array(values), atol=1e-5)


def test_gae_stops_at_episode_boundary():
    """A terminal at t must block bootstrapping from the next episode."""
    gamma, lam = 0.99, 0.95
    # Huge reward after the boundary must not leak backwards.
    adv_a, _ = compute_gae(_roll([0.0, 0.0, 100.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
                           torch.tensor([0.0]), gamma, lam)
    adv_b, _ = compute_gae(_roll([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
                           torch.tensor([0.0]), gamma, lam)
    assert np.allclose(adv_a[0, 0].item(), adv_b[0, 0].item(), atol=1e-6), \
        "reward from the next episode leaked across the done flag"
    assert adv_a[2, 0].item() == pytest.approx(100.0, abs=1e-4)


def test_gae_terminal_value_is_not_bootstrapped():
    adv, _ = compute_gae(_roll([1.0], [0.0], [1.0]), torch.tensor([999.0]), 0.99, 0.95)
    assert adv[0, 0].item() == pytest.approx(1.0, abs=1e-5), \
        "bootstrapped through a terminal state"


def test_centralised_critic_actually_uses_global_state():
    m = ActorCritic(ModelConfig(state_dim=20))
    grid, vec = torch.rand(2, 3, 6, 9, 9), torch.rand(2, 3, 9)
    _, v1, _ = m(grid, vec, state=torch.zeros(2, 3, 20))
    _, v2, _ = m(grid, vec, state=torch.ones(2, 3, 20))
    assert not torch.allclose(v1, v2), "value head ignores the centralised state"


def test_actor_does_not_see_global_state():
    m = ActorCritic(ModelConfig(state_dim=20))
    grid, vec = torch.rand(2, 3, 6, 9, 9), torch.rand(2, 3, 9)
    l1, _, _ = m(grid, vec, state=torch.zeros(2, 3, 20))
    l2, _, _ = m(grid, vec, state=torch.ones(2, 3, 20))
    assert torch.allclose(l1, l2), "CTDE violated -- the policy sees global state"


def test_act_with_logp_is_consistent():
    m = ActorCritic(ModelConfig(state_dim=12))
    grid, vec, state = torch.rand(5, 6, 9, 9), torch.rand(5, 9), torch.rand(5, 12)
    action, logp, value, _ = m.act_with_logp(grid, vec, state=state)
    logits, _, _ = m(grid.unsqueeze(0), vec.unsqueeze(0), state=state.unsqueeze(0))
    expected = torch.distributions.Categorical(logits=logits[0]).log_prob(action)
    assert torch.allclose(logp, expected, atol=1e-5)
    assert value.shape == (5,)


@pytest.mark.slow
def test_ppo_runs_end_to_end_and_improves_value_loss():
    cfg = PPOConfig(n_agents=4, num_envs=4, rollout=8, minibatches=2, ppo_epochs=2,
                    total_agent_steps=8 * 16 * 6, value_warmup_iters=2,
                    map_spec={"kind": "random", "height": 12, "width": 12, "density": 0.1},
                    max_steps=24, map_pool_size=8, eval_every=10_000,
                    log_every=1, device="cpu", out_dir="/tmp/ppo_test")
    out = train_ppo(cfg)
    assert len(out["history"]) >= 3
    assert np.isfinite(out["history"][-1]["vf_loss"]), "value loss went non-finite"
    assert np.isfinite(out["history"][-1]["entropy"])
