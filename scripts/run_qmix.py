"""QMIX training run: the second learning rule for the communication ablation.

    python scripts/run_qmix.py --warm-start runs/bccomm_a16h20/best.pt --comm full
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from train.qmix import QMIXConfig, train_qmix

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm-start", default=None)
    ap.add_argument("--comm", choices=["none", "full", "shuffled", "learned"],
                    default="full")
    ap.add_argument("--agents", type=int, default=16)
    ap.add_argument("--steps", type=int, default=4_000_000)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--buffer-blocks", type=int, default=64)
    ap.add_argument("--batch-blocks", type=int, default=4)
    ap.add_argument("--updates-per-block", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--target-update", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-epsilon", type=float, default=0.0,
                    help="action noise at evaluation; 0.0 is the QMIX policy. "
                         "Set ~0.05 only as a diagnostic (see "
                         "scripts/probe_warmstart.py).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--height", type=int, default=32)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--density", type=float, default=0.20)
    ap.add_argument("--max-steps", type=int, default=192)
    ap.add_argument("--eps-start", type=float, default=0.5)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--eps-decay-frac", type=float, default=0.4)
    ap.add_argument("--team-reward", choices=["mean", "sum"], default="mean")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--shaping-coef", type=float, default=0.0,
                    help="potential-based shaping on cost-to-go; ~10 on a "
                         "16x16 map. Policy-invariant, but report it.")
    ap.add_argument("--out", default="runs/qmix")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    cfg = QMIXConfig(
        n_agents=a.agents, comm_mode=a.comm, warm_start=a.warm_start,
        total_agent_steps=a.steps, num_envs=a.num_envs, rollout=a.rollout,
        buffer_blocks=a.buffer_blocks, batch_blocks=a.batch_blocks,
        updates_per_block=a.updates_per_block, lr=a.lr,
        target_update=a.target_update, eval_every=a.eval_every,
        eval_epsilon=a.eval_epsilon, gamma=a.gamma,
        team_reward=a.team_reward, max_steps=a.max_steps,
        shaping_coef=a.shaping_coef,
        eps_start=a.eps_start, eps_end=a.eps_end,
        eps_decay_frac=a.eps_decay_frac,
        map_spec={"kind": "random", "height": a.height,
                  "width": a.width, "density": a.density},
        seed=a.seed, out_dir=a.out, device=a.device)
    out = train_qmix(cfg)
    print(f"\n  best agents-on-goal {out['best_on_goal']:.3f} -> {a.out}/best.pt")
