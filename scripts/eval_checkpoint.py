"""Evaluate a trained checkpoint across agent counts and map families.

The policy is fully local (per-agent FOV), so it transfers across agent
counts and map sizes without retraining -- worth measuring, because
generalisation is a result in its own right.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from eval.rollout import evaluate_policy
from mapf.env import MAPFConfig
from models.encoder import ActorCritic, ModelConfig

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--eps-per-env", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    blob = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    model = ActorCritic(ModelConfig(**blob["model_cfg"])).to(a.device)
    model.load_state_dict(blob["model"]); model.eval()

    grid = [
        ("8a 20x20 d.15",  8,  {"kind": "random", "height": 20, "width": 20, "density": 0.15}),
        ("16a 32x32 d.20", 16, {"kind": "random", "height": 32, "width": 32, "density": 0.20}),
        ("32a 40x40 d.20", 32, {"kind": "random", "height": 40, "width": 40, "density": 0.20}),
        ("16a room rs8",   16, {"kind": "room", "height": 33, "width": 33, "room_size": 8}),
    ]
    print(f"{'config':<18}{'mode':<9}{'success':>9}{'on-goal':>9}{'coll/step':>11}")
    print("-" * 56)
    for name, n, spec in grid:
        cfg = MAPFConfig(n_agents=n, max_steps=160, map_pool_size=48, map_spec=spec)
        for mode, greedy in (("sample", False), ("greedy", True)):
            r = evaluate_policy(model, cfg, a.device, episodes_per_env=a.eps_per_env, greedy=greedy)
            print(f"{name:<18}{mode:<9}{r['success_rate']:>8.1%}"
                  f"{r['agents_on_goal']:>9.1%}{r['collisions_per_step']:>11.2f}")
        print()
