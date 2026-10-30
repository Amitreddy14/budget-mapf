"""Train a behaviour-cloning policy on expert demonstrations."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from train.bc import BCConfig, train_bc

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="data/demos/a8")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--parked-weight", type=float, default=0.1)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--eval-eps-per-env", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--no-gru", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    cfg = BCConfig(
        demo_root=a.demos, epochs=a.epochs, batch_size=a.batch_size,
        seq_len=a.seq_len, lr=a.lr, parked_weight=a.parked_weight,
        max_shards=a.max_shards, eval_episodes_per_env=a.eval_eps_per_env,
        eval_every=a.eval_every, patience=a.patience,
        use_gru=not a.no_gru, device=a.device,
        out_dir=a.out or f"runs/bc_{os.path.basename(a.demos)}",
    )
    train_bc(cfg)
