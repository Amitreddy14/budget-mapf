"""Behaviour cloning on the communication architecture (Phase 5 onward).

Produces a checkpoint whose every tensor transfers into the PPO model, which
`scripts/run_bc.py` does not -- see train/bc_comm.py for what that cost.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from train.bc_comm import CommBCConfig, train_comm_bc

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="data/demos/a16h")
    ap.add_argument("--comm", choices=["none", "full", "shuffled"], default="none",
                    help="'none' gives both ablation arms an identical, "
                         "communication-agnostic starting point")
    ap.add_argument("--radius", type=float, default=8.0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16, help="EPISODES per batch")
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--parked-weight", type=float, default=0.1)
    ap.add_argument("--tentative-coef", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    train_comm_bc(CommBCConfig(
        demo_root=a.demos, comm_mode=a.comm, comm_radius=a.radius,
        epochs=a.epochs, batch_size=a.batch_size, seq_len=a.seq_len, lr=a.lr,
        parked_weight=a.parked_weight, tentative_coef=a.tentative_coef,
        patience=a.patience, max_shards=a.max_shards, device=a.device,
        out_dir=a.out or f"runs/bccomm_{os.path.basename(a.demos)}",
    ))
