"""Does peer intent help predict the expert's action at all?

Twenty-five GPU-hours of PPO ablation across two densities produced
`voi_mean` ~ 7e-4 nats every time: the policy behaves the same whether or
not it hears its neighbours. Before spending another twenty-five, measure
the premise directly, in supervised learning, where there is no RL variance
to hide behind.

Behaviour cloning on the a16h20 demonstrations reaches 98.4% action accuracy
with communication switched off. The expert is a prioritised space-time A*
planner, and a 9x9 window plus a global cost-to-go field apparently
determines its next move almost everywhere. That leaves communication at
most 1.6% of decisions to improve -- which is the same statement as
`voi_mean` ~ 0, arrived at for a thousandth of the cost.

This probe runs BC twice per demonstration set, comm off and comm on, and
reports what fraction of the residual error the message pathway closes. A
regime where that fraction is near zero cannot support a communication
result no matter which scheduler is bolted on; a regime where it is large
is worth the PPO budget.

  python scripts/comm_headroom.py --demos data/demos/a16h20 data/demos/a16h20_fov5

Note that `comm=none` zeroes the context vector, so the message pathway
receives no gradient and the policy head learns a function of zeros. That
is also why warm-starting a communicating PPO run from a comm=none
checkpoint hands it a randomly initialised attention module -- the cheapest
repair for which is to learn to ignore it.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from train.bc_comm import CommBCConfig, train_comm_bc


def best_row(history, key="env_on_goal"):
    rows = [r for r in history if r.get(key) is not None]
    return max(rows, key=lambda r: r[key]) if rows else history[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", nargs="+", default=["data/demos/a16h20"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    # Early stopping fires at a different epoch per arm (comm=none stopped at
    # 9, comm=full at 10 on the first run), so the arms get unequal budgets
    # and the accuracies are not comparable. Off by default here.
    ap.add_argument("--patience", type=int, default=0,
                    help="0 disables early stopping so every arm trains for "
                         "exactly --epochs")
    ap.add_argument("--modes", nargs="+", default=["none", "full", "shuffled"],
                    choices=["none", "full", "shuffled"])
    ap.add_argument("--radius", type=float, default=8.0)
    ap.add_argument("--out", default="runs/headroom")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    table = {}
    for root in a.demos:
        tag = os.path.basename(root.rstrip("/\\"))
        for mode in a.modes:
            print(f"\n=== {tag} | comm={mode} ===")
            cfg = CommBCConfig(
                demo_root=root, comm_mode=mode, comm_radius=a.radius,
                epochs=a.epochs, batch_size=a.batch_size,
                patience=a.patience or a.epochs + 1,
                out_dir=os.path.join(a.out, f"{tag}_{mode}"), device=a.device,
            )
            out = train_comm_bc(cfg)
            hist = out["history"] if isinstance(out, dict) else out
            row = best_row(hist)
            # The headroom question is purely supervised: can peer intent
            # predict the expert's action? Take the best validation accuracy
            # any epoch reached, not the accuracy of the epoch the env-metric
            # early stop happened to select.
            accs = [r["acc"] for r in hist if r.get("acc") is not None]
            losses = [r["val_loss"] for r in hist if r.get("val_loss") is not None]
            table[(tag, mode)] = {
                "best_acc": max(accs), "best_val_loss": min(losses),
                "acc": row.get("acc"), "val_loss": row.get("val_loss"),
                "env_success": row.get("env_success"),
                "env_on_goal": row.get("env_on_goal"),
                "env_coll": row.get("env_coll"), "epoch": row.get("epoch"),
            }

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "headroom.json"), "w") as fh:
        json.dump({f"{k[0]}|{k[1]}": v for k, v in table.items()}, fh, indent=2)

    print(f"\n{'demos':<22}{'comm':>6}{'best acc':>10}{'err':>9}{'val_loss':>10}"
          f"{'success':>9}{'on-goal':>9}{'coll':>8}")
    print("-" * 84)
    for (tag, mode), v in table.items():
        print(f"{tag:<22}{mode:>6}{v['best_acc']:>10.4f}{1 - v['best_acc']:>9.4f}"
              f"{v['best_val_loss']:>10.4f}{v['env_success']:>9.4f}"
              f"{v['env_on_goal']:>9.4f}{v['env_coll']:>8.3f}")

    print(f"\n{'demos':<22}{'err none':>10}{'err shuf':>10}{'err full':>10}"
          f"{'closed':>9}{'vs shuf':>9}  verdict")
    print("-" * 88)
    for root in a.demos:
        tag = os.path.basename(root.rstrip("/\\"))
        if (tag, "none") not in table or (tag, "full") not in table:
            continue
        en = 1 - table[(tag, "none")]["best_acc"]
        ef = 1 - table[(tag, "full")]["best_acc"]
        closed = (en - ef) / en if en > 0 else 0.0
        es = 1 - table[(tag, "shuffled")]["best_acc"] if (tag, "shuffled") in table else None
        # Of the error full closes, how much survives destroying the message
        # content? That residual is the part attributable to information
        # rather than to the extra capacity comm=full brings into use.
        info = ((es - ef) / en) if (es is not None and en > 0) else None

        if en < 0.005:
            verdict = "SATURATED -- no room for communication to matter"
        elif closed < 0.05:
            verdict = "no gain -- intent does not predict the expert here"
        elif info is not None and info < 0.25 * closed:
            verdict = "CAPACITY, not information -- shuffling costs almost nothing"
        elif closed < 0.20:
            verdict = "marginal"
        else:
            verdict = "REAL HEADROOM -- worth a PPO ablation"
        print(f"{tag:<22}{en:>10.4f}"
              f"{(f'{es:.4f}' if es is not None else '-'):>10}{ef:>10.4f}"
              f"{closed:>8.1%}{(f'{info:+.1%}' if info is not None else '-'):>9}  {verdict}")

    print("\n'closed' is the share of BC's residual error that switching messages"
          "\non removes. 'vs shuf' is how much of that survives shuffling who sent"
          "\nwhat -- the part that is information rather than extra capacity in use."
          "\n\nA regime earns a 25-GPU-hour ablation only if 'vs shuf' is clearly"
          " positive.")


if __name__ == "__main__":
    main()
