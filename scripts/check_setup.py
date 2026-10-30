"""Verify the local GPU setup before committing to a long training run.

Checks the things that fail silently: the numpy<->torch bridge, real GPU
throughput for *this* model, and that an end-to-end rollout works on CUDA.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from mapf.env import MAPFConfig, VecMAPFEnv
from models.encoder import ActorCritic, ModelConfig, count_parameters

OK, BAD = "  [ok]  ", "  [!!]  "


def main() -> int:
    fails = 0
    print(f"torch {torch.__version__} | numpy {np.__version__}")

    # 1. The numpy bridge. A disabled bridge still "works" via slow copies,
    #    so test the zero-copy path explicitly.
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    try:
        t = torch.from_numpy(a)
        t[0, 0] = 99.0
        shared = a[0, 0] == 99.0
        back = torch.as_tensor(a).numpy()
        print(f"{OK if shared else BAD}numpy bridge: from_numpy shares memory={shared}, "
              f"roundtrip ok={np.array_equal(back, a)}")
        fails += 0 if shared else 1
    except Exception as e:
        print(f"{BAD}numpy bridge broken: {e}")
        return 1

    if not torch.cuda.is_available():
        print(f"{BAD}CUDA not available -- training will be very slow")
        return 1
    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"{OK}{name}, {total:.1f} GB")

    # 2. Throughput on the real model, CPU vs GPU.
    model_cfg = ModelConfig()
    print(f"{OK}model: {count_parameters(ActorCritic(model_cfg)):,} parameters")

    for device, label in ((torch.device("cpu"), "cpu"), (dev, "cuda")):
        m = ActorCritic(model_cfg).to(device)
        grid = torch.rand(16, 256, 6, 9, 9, device=device)
        vec = torch.rand(16, 256, 9, device=device)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        for _ in range(2):                                  # warm up
            logits, value, _ = m(grid, vec)
            (logits.mean() + value.mean()).backward()
            opt.step(); opt.zero_grad()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            logits, value, _ = m(grid, vec)
            (logits.mean() + value.mean()).backward()
            opt.step(); opt.zero_grad()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 5
        print(f"{OK}{label:<5} fwd+bwd on (16 x 256) agent-steps: {dt*1000:7.1f} ms "
              f"-> {16*256/dt:,.0f} agent-steps/s")

    # 3. End-to-end: policy driving the environment on GPU.
    cfg = MAPFConfig(n_agents=16, map_spec={"kind": "random", "height": 32,
                                            "width": 32, "density": 0.20})
    env = VecMAPFEnv(cfg, num_envs=32, seed=0)
    m = ActorCritic(ModelConfig()).to(dev)
    M = 32 * 16
    hidden = m.initial_state(M, dev)
    t0 = time.perf_counter()
    for _ in range(50):
        grid, vec = env.observe()
        g = torch.as_tensor(grid.reshape(M, *grid.shape[2:]), device=dev)
        v = torch.as_tensor(vec.reshape(M, -1), device=dev)
        action, _, hidden = m.act(g, v, hidden)
        env.step(action.view(32, 16).cpu().numpy())
    dt = time.perf_counter() - t0
    print(f"{OK}end-to-end rollout: {50*32/dt:,.0f} env-steps/s "
          f"({50*M/dt:,.0f} agent-steps/s)")

    free = torch.cuda.memory_allocated() / 1e9
    print(f"{OK}GPU memory in use after rollout: {free:.2f} GB")
    print("\nall good -- ready for Phase 4" if fails == 0 else f"\n{fails} problem(s)")
    return fails


if __name__ == "__main__":
    sys.exit(main())
