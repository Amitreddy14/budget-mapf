"""Train recurrent MAPPO, optionally warm-started from behaviour cloning."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from train.ppo import PPOConfig, train_ppo

PRESETS = {
    "a8":  dict(n_agents=8,  height=20, width=20, density=0.15),
    "a16": dict(n_agents=16, height=32, width=32, density=0.20),
    "a32": dict(n_agents=32, height=40, width=40, density=0.20),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("preset", choices=list(PRESETS), nargs="?", default="a16")
    ap.add_argument("--bc", default=None, help="BC checkpoint to warm start from")
    ap.add_argument("--steps", type=int, default=5_000_000)
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--bc-kl", type=float, default=0.0)
    ap.add_argument("--value-warmup", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    p = PRESETS[a.preset]
    cfg = PPOConfig(
        n_agents=p["n_agents"],
        map_spec={"kind": "random", "height": p["height"], "width": p["width"],
                  "density": p["density"]},
        num_envs=a.num_envs, rollout=a.rollout, minibatches=a.minibatches,
        total_agent_steps=a.steps, lr=a.lr, entropy_coef=a.entropy,
        bc_checkpoint=a.bc, bc_kl_coef=a.bc_kl, value_warmup_iters=a.value_warmup,
        eval_every=a.eval_every, seed=a.seed, device=a.device,
        out_dir=a.out or f"runs/ppo_{a.preset}",
    )
    train_ppo(cfg)
