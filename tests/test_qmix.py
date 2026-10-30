"""QMIX mixer and Q-value counterfactuals.

Monotonicity is the property the whole factorisation rests on: if it fails,
an agent's own argmax is no longer the joint argmax and decentralised
execution is unsound. It is cheap to check by autograd and expensive to
notice any other way, so it is checked directly rather than assumed from the
abs() calls.
"""
import pytest
import torch

from models.qmix import (MixerConfig, QMixer, greedy_from_q, q_influence,
                         q_value_of_information)


def _mixer(n=4, s=20, **kw):
    return QMixer(MixerConfig(n_agents=n, state_dim=s, **kw))


def test_mixer_output_shape():
    m = _mixer()
    out = m(torch.randn(7, 4), torch.randn(7, 20))
    assert out.shape == (7,)


def test_mixer_is_monotone_in_every_agent_value():
    """d Q_tot / d Q_i >= 0 for every agent, over many random states."""
    torch.manual_seed(0)
    m = _mixer(n=5, s=25)
    q = torch.randn(64, 5, requires_grad=True)
    total = m(q, torch.randn(64, 25)).sum()
    grad, = torch.autograd.grad(total, q)
    assert (grad >= -1e-6).all(), f"min gradient {grad.min().item()}"


def test_mixer_monotone_after_a_gradient_step():
    """abs() must keep holding once the hypernetworks have moved."""
    torch.manual_seed(0)
    m = _mixer(n=4, s=20)
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    for _ in range(20):
        loss = (m(torch.randn(32, 4), torch.randn(32, 20)) - 3.0).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    q = torch.randn(64, 4, requires_grad=True)
    grad, = torch.autograd.grad(m(q, torch.randn(64, 20)).sum(), q)
    assert (grad >= -1e-6).all(), f"min gradient {grad.min().item()}"


def test_mixer_rejects_wrong_shapes():
    m = _mixer(n=4, s=20)
    with pytest.raises(ValueError):
        m(torch.randn(3, 5), torch.randn(3, 20))
    with pytest.raises(ValueError):
        m(torch.randn(3, 4), torch.randn(3, 11))


def test_greedy_is_the_argmax():
    q = torch.tensor([[[0.1, 0.9, 0.2]], [[2.0, -1.0, 0.0]]])
    assert greedy_from_q(q).flatten().tolist() == [1, 0]


# --- the two counterfactual measures ---------------------------------------

def test_influence_is_zero_when_nothing_changes():
    q = torch.randn(8, 5)
    assert torch.allclose(q_influence(q, q), torch.zeros(8), atol=1e-7)


def test_influence_ignores_whether_the_change_helps():
    """The flaw this pairs with value_of_information to expose."""
    base = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
    better = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0]])
    worse = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
    assert float(q_influence(better, base)) > 0
    assert float(q_influence(worse, base)) > 0


def test_value_is_positive_when_the_chosen_action_gains():
    base = torch.tensor([[1.0, 1.0, 1.0]])
    comm = torch.tensor([[2.0, 1.0, 1.0]])
    act = torch.tensor([0])
    assert float(q_value_of_information(comm, base, act)) > 0


def test_value_is_negative_when_the_chosen_action_loses():
    base = torch.tensor([[1.0, 1.0, 1.0]])
    comm = torch.tensor([[0.0, 1.0, 1.0]])
    act = torch.tensor([0])
    assert float(q_value_of_information(comm, base, act)) < 0


def test_value_is_zero_under_a_uniform_shift():
    """A constant added to every action value changes no decision."""
    base = torch.randn(16, 5)
    comm = base + 3.7
    act = torch.randint(0, 5, (16,))
    assert torch.allclose(q_value_of_information(comm, base, act),
                          torch.zeros(16), atol=1e-5)


def test_value_is_near_zero_when_the_shift_ignores_action_quality():
    torch.manual_seed(0)
    base = torch.randn(4096, 5)
    comm = base + torch.randn(4096, 5)      # noise unrelated to the action
    act = torch.randint(0, 5, (4096,))
    assert abs(float(q_value_of_information(comm, base, act).mean())) < 0.08


# --- the trainer -----------------------------------------------------------
# A smoke run is the only way to catch shape and device errors in a recurrent
# off-policy loop; the pieces type-check individually and still fail when
# wired together.

@pytest.mark.slow
def test_qmix_trains_without_error_and_logs_both_measures():
    from train.qmix import QMIXConfig, train_qmix
    cfg = QMIXConfig(
        n_agents=4,
        map_spec={"kind": "random", "height": 12, "width": 12, "density": 0.15},
        max_steps=24, num_envs=4, map_pool_size=8, rollout=8,
        total_agent_steps=4 * 4 * 8 * 14, buffer_blocks=6, batch_blocks=2,
        min_blocks=2, updates_per_block=1, target_update=3,
        comm_mode="full", eval_every=6, eval_eps_per_env=1, log_every=1,
        warm_start=None, device="cpu", out_dir="runs/_test_qmix")
    hist = train_qmix(cfg)["history"]
    assert hist, "no iterations logged"
    assert any("loss" in r for r in hist), "no learning happened"
    assert any("q_influence" in r for r in hist), "influence never measured"
    assert any("q_voi" in r for r in hist), "value never measured"
    assert any("eval_on_goal" in r for r in hist), "never evaluated"


@pytest.mark.slow
def test_qmix_none_arm_skips_the_counterfactual():
    """With no communication there is nothing to measure against silence."""
    from train.qmix import QMIXConfig, train_qmix
    cfg = QMIXConfig(
        n_agents=4,
        map_spec={"kind": "random", "height": 12, "width": 12, "density": 0.15},
        max_steps=24, num_envs=4, map_pool_size=8, rollout=8,
        total_agent_steps=4 * 4 * 8 * 10, buffer_blocks=6, batch_blocks=2,
        min_blocks=2, updates_per_block=1, target_update=3,
        comm_mode="none", eval_every=100, log_every=1,
        warm_start=None, device="cpu", out_dir="runs/_test_qmix_none")
    hist = train_qmix(cfg)["history"]
    assert hist and not any("q_influence" in r for r in hist)


# The warm-start pre-flight raised on a sound checkpoint because it changed
# both the comm mode and the action rule, then blamed the comm mode. These pin
# the evaluator's two switches so that cannot recur silently.

@pytest.mark.slow
def test_evaluator_action_rule_changes_the_result():
    """Greedy and sampled are different measurements, not two names for one."""
    import torch
    from train.qmix import _evaluate
    from mapf.env import MAPFConfig, VecMAPFEnv
    from models.comm_policy import CommActorCritic, CommConfig
    from models.encoder import ModelConfig

    env_cfg = MAPFConfig(n_agents=4, fov=9, max_steps=32, map_pool_size=8,
                         map_spec={"kind": "random", "height": 12,
                                   "width": 12, "density": 0.15})
    probe = VecMAPFEnv(env_cfg, num_envs=2, seed=0)
    g, v = probe.reset()
    mcfg = ModelConfig(in_channels=g.shape[2], fov=9, vec_dim=v.shape[-1],
                       state_dim=20)
    ccfg = CommConfig(mode="full")
    model = CommActorCritic(mcfg, ccfg)
    dev = torch.device("cpu")

    torch.manual_seed(0)
    a = _evaluate(model, env_cfg, ccfg, dev, 1, 5, num_envs=4, greedy=True)
    torch.manual_seed(0)
    b = _evaluate(model, env_cfg, ccfg, dev, 1, 5, num_envs=4, greedy=True)
    assert a["agents_on_goal"] == b["agents_on_goal"], (
        "greedy evaluation must be deterministic")

    torch.manual_seed(0)
    c = _evaluate(model, env_cfg, ccfg, dev, 1, 5, num_envs=4, greedy=False)
    assert c["collisions_per_step"] != a["collisions_per_step"], (
        "sampled and greedy returned identical rollouts; the action rule is "
        "not reaching model.act")


@pytest.mark.slow
def test_evaluator_restores_the_comm_mode():
    """A probe under another mode must not leave the model switched over."""
    import torch
    from train.qmix import _evaluate
    from mapf.env import MAPFConfig, VecMAPFEnv
    from models.comm_policy import CommActorCritic, CommConfig
    from models.encoder import ModelConfig

    env_cfg = MAPFConfig(n_agents=4, fov=9, max_steps=16, map_pool_size=4,
                         map_spec={"kind": "random", "height": 12,
                                   "width": 12, "density": 0.15})
    probe = VecMAPFEnv(env_cfg, num_envs=2, seed=0)
    g, v = probe.reset()
    mcfg = ModelConfig(in_channels=g.shape[2], fov=9, vec_dim=v.shape[-1],
                       state_dim=20)
    model = CommActorCritic(mcfg, CommConfig(mode="full"))
    _evaluate(model, env_cfg, CommConfig(mode="full"), torch.device("cpu"),
              1, 5, num_envs=2, comm_mode="none")
    assert model.comm_cfg.mode == "full"
