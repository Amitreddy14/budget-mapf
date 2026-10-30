"""Shape and behaviour tests for the actor-critic network."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models.encoder import (ActorCritic, ModelConfig, count_parameters,
                            legal_action_mask)


def _model(**kw):
    return ActorCritic(ModelConfig(**kw))


def test_forward_shapes():
    m = _model()
    T, M = 7, 12
    grid = torch.rand(T, M, 6, 9, 9)
    vec = torch.rand(T, M, 9)
    logits, value, hidden = m(grid, vec)
    assert logits.shape == (T, M, 5)
    assert value.shape == (T, M)
    assert hidden.shape == (1, M, 256)


def test_parameter_count_is_small_enough_for_an_8gb_gpu():
    n = count_parameters(_model())
    assert n < 2_000_000, f"{n:,} parameters is heavier than planned"


def test_legal_mask_blocks_walls_and_always_allows_stay():
    grid = torch.zeros(3, 6, 9, 9)
    centre = 4
    grid[0, 0, centre - 1, centre] = 1.0      # wall above  -> UP illegal
    grid[1, 0, centre, centre + 1] = 1.0      # wall right  -> RIGHT illegal
    mask = legal_action_mask(grid)
    assert mask[:, 0].all(), "STAY must always be legal"
    assert not mask[0, 1] and mask[0, 2] and mask[0, 3] and mask[0, 4]
    assert not mask[1, 4] and mask[1, 1]
    assert mask[2].all(), "open cell should permit every action"


def test_illegal_actions_get_no_probability():
    m = _model()
    grid = torch.zeros(1, 4, 6, 9, 9)
    grid[0, :, 0, 3, 4] = 1.0                  # wall above every agent
    vec = torch.zeros(1, 4, 9)
    logits, _, _ = m(grid, vec)
    probs = torch.softmax(logits[0], dim=-1)
    assert probs[:, 1].max() < 1e-6, "policy can still choose to walk into a wall"
    assert torch.allclose(probs.sum(-1), torch.ones(4), atol=1e-5)


def test_gru_state_resets_zero_memory():
    m = _model()
    T, M = 4, 5
    grid = torch.rand(T, M, 6, 9, 9)
    vec = torch.rand(T, M, 9)
    resets = torch.zeros(T, M, dtype=torch.bool)
    resets[2, :] = True
    out_a, _, _ = m(grid, vec, resets=resets)
    # With a reset at t=2, steps 2..3 must not depend on steps 0..1.
    grid2 = grid.clone(); grid2[:2] = torch.rand(2, M, 6, 9, 9)
    out_b, _, _ = m(grid2, vec, resets=resets)
    assert torch.allclose(out_a[2:], out_b[2:], atol=1e-5), "state leaked across a reset"
    assert not torch.allclose(out_a[:2], out_b[:2]), "early steps should differ"


def test_recurrence_actually_uses_history():
    m = _model()
    M = 3
    grid = torch.rand(6, M, 6, 9, 9)
    vec = torch.rand(6, M, 9)
    full, _, _ = m(grid, vec)
    last_only, _, _ = m(grid[5:], vec[5:])
    assert not torch.allclose(full[5], last_only[0], atol=1e-4), \
        "output is independent of history -- the GRU is doing nothing"


def test_act_returns_valid_actions():
    m = _model()
    grid = torch.rand(9, 6, 9, 9)
    vec = torch.rand(9, 9)
    action, value, hidden = m.act(grid, vec, greedy=True)
    assert action.shape == (9,) and value.shape == (9,)
    assert int(action.min()) >= 0 and int(action.max()) <= 4


def test_feedforward_variant_runs():
    m = _model(use_gru=False)
    logits, value, _ = m(torch.rand(3, 4, 6, 9, 9), torch.rand(3, 4, 9))
    assert logits.shape == (3, 4, 5)


# --- fov-aware conv stack ---------------------------------------------------
# The stack was four hard-coded 3x3 valid convs, which only works at fov=9;
# fov=5 died inside conv2d with "kernel size can't be greater than actual
# input size". Each conv removes 2, so walking F -> 1 takes (F-1)//2 of them.

def test_conv_widths_reproduce_the_original_stack_at_fov_9():
    from models.encoder import _conv_widths
    assert _conv_widths(9) == [32, 64, 64, 128]


def test_conv_widths_scale_with_fov_and_always_end_at_128():
    from models.encoder import _conv_widths
    for fov in (3, 5, 7, 9, 11, 13):
        w = _conv_widths(fov)
        assert len(w) == (fov - 1) // 2
        assert w[-1] == 128


def test_conv_widths_reject_even_or_tiny_windows():
    import pytest
    from models.encoder import _conv_widths
    for bad in (2, 4, 8, 1, 0, -1):
        with pytest.raises(ValueError):
            _conv_widths(bad)


def test_encoder_runs_at_every_supported_fov():
    import torch
    from models.encoder import AgentEncoder, ModelConfig
    for fov in (3, 5, 7, 9, 11):
        cfg = ModelConfig(in_channels=6, fov=fov, vec_dim=9)
        enc = AgentEncoder(cfg)
        out = enc(torch.zeros(2, 6, fov, fov), torch.zeros(2, 9))
        assert out.shape == (2, cfg.hidden)


def test_shuffled_comm_keeps_the_pathway_but_destroys_correspondence():
    """The capacity control must still produce a non-zero context."""
    import torch
    from models.comm_policy import CommActorCritic, CommConfig
    from models.encoder import ModelConfig
    T, B, N, fov = 2, 3, 4, 9
    mcfg = ModelConfig(in_channels=6, fov=fov, vec_dim=9)
    grid = torch.zeros(T, B, N, 6, fov, fov)
    vec = torch.zeros(T, B, N, 9)
    adj = torch.ones(T, B, N, N, dtype=torch.bool)
    adj &= ~torch.eye(N, dtype=torch.bool)

    outs = {}
    for mode in ("none", "full", "shuffled"):
        torch.manual_seed(0)
        model = CommActorCritic(mcfg, CommConfig(mode=mode))
        torch.manual_seed(0)
        outs[mode] = model(grid, vec, adj)[0]

    # Same parameter count in every arm, as the ablation requires.
    counts = set()
    for mode in ("none", "full", "shuffled"):
        torch.manual_seed(0)
        m = CommActorCritic(mcfg, CommConfig(mode=mode))
        counts.add(sum(p.numel() for p in m.parameters()))
    assert len(counts) == 1

    # Shuffled must go through the comm pathway, not the zeroed one.
    assert not torch.allclose(outs["shuffled"], outs["none"])
