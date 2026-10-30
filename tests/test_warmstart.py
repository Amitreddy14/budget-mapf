"""Warm-start integrity tests.

A silently partial warm start is the most expensive bug this project has hit:
a `bc.py` checkpoint transferred its encoder into `CommActorCritic` but not its
policy head, so six 20M-step runs trained from an effectively random policy and
produced ~0% agents-on-goal. The loader reported "19 tensors" and nothing else.

These tests make that class of failure loud.
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

torch = pytest.importorskip("torch")

from models.comm_policy import CommActorCritic, CommConfig
from models.encoder import ActorCritic, ModelConfig
from train.ppo_comm import CRITICAL_MODULES, load_partial


def _save(model, path, **extra):
    torch.save({"model": model.state_dict(), **extra}, path)
    return path


def test_mismatched_checkpoint_raises_instead_of_loading_quietly():
    """The exact bug: an ActorCritic checkpoint into a CommActorCritic."""
    mcfg = ModelConfig(state_dim=80)
    with tempfile.TemporaryDirectory() as d:
        path = _save(ActorCritic(ModelConfig(state_dim=0)), os.path.join(d, "bc.pt"))
        target = CommActorCritic(mcfg, CommConfig(mode="full"))
        with pytest.raises(RuntimeError, match="critical module"):
            load_partial(target, path, torch.device("cpu"))


def test_matching_checkpoint_transfers_every_tensor():
    mcfg = ModelConfig(state_dim=80)
    ccfg = CommConfig(mode="none")
    with tempfile.TemporaryDirectory() as d:
        source = CommActorCritic(mcfg, ccfg)
        path = _save(source, os.path.join(d, "bc_comm.pt"), env_on_goal=0.9)
        target = CommActorCritic(mcfg, CommConfig(mode="full"))
        info = load_partial(target, path, torch.device("cpu"))

        assert info["copied"] == len(source.state_dict()), "not every tensor transferred"
        for name in CRITICAL_MODULES:
            got, total = info["coverage"][name]
            assert got == total, f"{name} only {got}/{total}"
        for k, v in source.state_dict().items():
            assert torch.allclose(v, target.state_dict()[k]), f"{k} differs after load"


def test_comm_mode_does_not_block_weight_transfer():
    """A checkpoint trained with comm off must load into a comm-on model.

    This is what lets both ablation arms start from one identical checkpoint.
    """
    mcfg = ModelConfig(state_dim=80)
    with tempfile.TemporaryDirectory() as d:
        path = _save(CommActorCritic(mcfg, CommConfig(mode="none")),
                     os.path.join(d, "c.pt"), env_on_goal=0.9)
        for mode in ("none", "full", "learned"):
            target = CommActorCritic(mcfg, CommConfig(mode=mode))
            info = load_partial(target, path, torch.device("cpu"))
            assert info["copied"] == len(target.state_dict()), f"mode={mode} incomplete"


def test_loader_reports_the_checkpoints_recorded_performance():
    """The pre-flight check needs this to compare against."""
    mcfg = ModelConfig(state_dim=80)
    with tempfile.TemporaryDirectory() as d:
        path = _save(CommActorCritic(mcfg, CommConfig(mode="none")),
                     os.path.join(d, "c.pt"), env_on_goal=0.937,
                     observe_peer_goals=False)
        info = load_partial(CommActorCritic(mcfg, CommConfig(mode="full")),
                            path, torch.device("cpu"))
        assert info["reported_on_goal"] == pytest.approx(0.937)
        assert info["observe_peer_goals"] is False


def test_environment_mismatch_is_refused():
    """Two ablation scripts once ran the same preset at different densities."""
    import numpy as np
    from train.ppo_comm import CommPPOConfig, train_comm_ppo
    mcfg = ModelConfig(state_dim=40)
    with tempfile.TemporaryDirectory() as d:
        path = _save(CommActorCritic(mcfg, CommConfig(mode="none")),
                     os.path.join(d, "c.pt"), env_on_goal=0.9,
                     observe_peer_goals=False, n_agents=8,
                     map_spec={"kind": "random", "height": 16, "width": 16,
                               "density": 0.25},
                     max_steps=192)
        cfg = CommPPOConfig(
            n_agents=8, warm_start=path, device="cpu",
            map_spec={"kind": "random", "height": 16, "width": 16,
                      "density": 0.20},          # <-- the mismatch
            num_envs=2, rollout=4, total_agent_steps=64, map_pool_size=4,
            out_dir=os.path.join(d, "run"))
        with pytest.raises(RuntimeError, match="does not match the warm-start"):
            train_comm_ppo(cfg)


# --- critic-only warmup must not touch the shared trunk ---------------------
# `critic_only` omitted the policy loss but still sent the value loss through
# the shared encoder and GRU. On a warm start that checked out at 0.925
# agents-on-goal, tentative_entropy went 0.404 -> 1.078 by iteration 5 --
# before any policy gradient was applied -- and on-goal collapsed to 0.23.

def test_warmup_drops_every_gradient_outside_the_value_head():
    import torch
    from models.comm_policy import CommActorCritic, CommConfig
    from models.encoder import ModelConfig
    from train.ppo_comm import _keep_only_value_grads

    T, B, N, fov = 2, 2, 3, 9
    mcfg = ModelConfig(in_channels=6, fov=fov, vec_dim=9)
    model = CommActorCritic(mcfg, CommConfig(mode="full"))
    logits, value, *_ = model(torch.zeros(T, B, N, 6, fov, fov),
                              torch.zeros(T, B, N, 9),
                              torch.ones(T, B, N, N, dtype=torch.bool))
    (value.sum() + logits.sum()).backward()

    assert any(p.grad is not None for n, p in model.named_parameters()
               if n.startswith("encoder")), "setup: encoder should have grads"

    _keep_only_value_grads(model)

    for name, prm in model.named_parameters():
        if name.startswith("value"):
            continue
        assert prm.grad is None, f"{name} kept a gradient during value warmup"
    assert any(p.grad is not None for n, p in model.named_parameters()
               if n.startswith("value")), "value head must still train"
