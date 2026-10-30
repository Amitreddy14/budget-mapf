"""Train MAPPO with (or without) intent communication."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from train.ppo_comm import CommPPOConfig, train_comm_ppo

PRESETS = {
    "a8":  dict(n_agents=8,  height=20, width=20, density=0.15),
    "a16": dict(n_agents=16, height=32, width=32, density=0.20),
    "a32": dict(n_agents=32, height=40, width=40, density=0.20),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("preset", choices=list(PRESETS), nargs="?", default="a16")
    ap.add_argument("--comm", choices=["none", "full", "learned", "random", "distance"],
                    default="full")
    ap.add_argument("--budget", type=float, default=0.25,
                    help="fraction of agents permitted to transmit")
    ap.add_argument("--allocation", choices=["rate", "topk"], default="rate")
    ap.add_argument("--voi-coef", type=float, default=0.5)
    ap.add_argument("--radius", type=float, default=8.0)
    ap.add_argument("--warm-start", default=None)
    ap.add_argument("--steps", type=int, default=5_000_000)
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--value-warmup", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--tentative-coef", type=float, default=0.3,
                    help="keeps broadcast intent informative; 0 reproduces the decay")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    p = PRESETS[a.preset]
    cfg = CommPPOConfig(
        n_agents=p["n_agents"],
        map_spec={"kind": "random", "height": p["height"], "width": p["width"],
                  "density": p["density"]},
        comm_mode=a.comm, comm_radius=a.radius, warm_start=a.warm_start,
        budget_frac=a.budget, allocation=a.allocation, voi_coef=a.voi_coef,
        num_envs=a.num_envs, rollout=a.rollout, minibatches=a.minibatches,
        total_agent_steps=a.steps, lr=a.lr, value_warmup_iters=a.value_warmup,
        tentative_coef=a.tentative_coef,
        eval_every=a.eval_every, seed=a.seed, device=a.device,
        out_dir=a.out or f"runs/comm_{a.preset}_{a.comm}_s{a.seed}",
    )
    train_comm_ppo(cfg)
