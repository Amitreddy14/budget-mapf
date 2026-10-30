"""Build expert demonstration datasets for behaviour cloning."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experts.generate import generate
from mapf.env import MAPFConfig

# Presets suffixed "h" hide peer goals (observe_peer_goals=False), making a
# neighbour's intent genuinely unobservable. With peer goals visible an agent
# can already see where 69% of its neighbours are headed, which leaves a
# communication channel nothing to carry. See docs/FINDINGS-phase5.md.
PRESETS = {
    "debug":  dict(n_agents=8,  size=20, density=0.15, samples=20_000),
    "a8":     dict(n_agents=8,  size=20, density=0.15, samples=300_000),
    "a16":    dict(n_agents=16, size=32, density=0.20, samples=600_000),
    "a32":    dict(n_agents=32, size=40, density=0.20, samples=800_000),
    # Harder regime: intent hidden, higher congestion.
    # d=0.20 is the regime where communication measurably helps (Phase 5);
    # d=0.25 is where it does not (Phase 6). Both are needed, matched.
    "a16h20": dict(n_agents=16, size=32, density=0.20, samples=600_000, hide=True),
    "a16h":   dict(n_agents=16, size=32, density=0.25, samples=600_000, hide=True),
    "a32h":   dict(n_agents=32, size=40, density=0.25, samples=900_000, hide=True),
    # Scaling series at the density the 16-agent result was measured at, so
    # agent count is the only variable. a32h/a64h sit at 0.25 and are not
    # comparable to a16h20.
    "a32h20": dict(n_agents=32, size=40, density=0.20, samples=900_000, hide=True),
    "a64h20": dict(n_agents=64, size=64, density=0.20, samples=1_200_000, hide=True),
    "a64h":   dict(n_agents=64, size=64, density=0.25, samples=1_200_000, hide=True),
    # Structured maps at the SAME agent density as a16h20 (0.0197 agents per
    # free cell), so map structure is the only variable. Sizes were chosen by
    # measuring the largest connected component: room 32x32 with 6-cell rooms
    # gives 824 free cells and maze 41x41 gives 799, against 813 for the
    # random grid. Corridors and doorways force head-on negotiation that an
    # open grid lets agents route around.
    "a16room": dict(n_agents=16, size=32, density=0.20, samples=600_000, hide=True,
                    kind="room", room_size=6),
    "a16maze": dict(n_agents=16, size=41, density=0.20, samples=600_000, hide=True,
                    kind="maze"),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("preset", choices=list(PRESETS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--map-kind", choices=["random", "room", "maze"], default="random",
                    help="'room' is 8x8 rooms joined by one-cell doors and "
                         "'maze' is a recursive-backtracker maze of one-cell "
                         "corridors. Both force head-on negotiation in a way "
                         "open random grids do not, since an agent cannot "
                         "simply route around a neighbour.")
    ap.add_argument("--room-size", type=int, default=8)
    ap.add_argument("--obs-costmap", choices=["field", "coarse", "none"], default="field",
                    help="'coarse' quantises the global cost-to-go field into "
                         "bands and drops the downhill channel; 'none' zeroes "
                         "them entirely. The straight-line bearing to the goal "
                         "is kept in every case, so the task stays solvable.")
    ap.add_argument("--fov", type=int, default=9,
                    help="observation window. Smaller windows remove the local\n                         information that currently makes the expert action\n                         98%% predictable without any communication.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--hide-peer-goals", action="store_true",
                    help="force-hide peer goals regardless of preset")
    args = ap.parse_args()

    p = PRESETS[args.preset]
    out = args.out or f"data/demos/{args.preset}"
    hide = args.hide_peer_goals or p.get("hide", False)

    # A recursive-backtracker maze lives on the odd lattice, so an even side
    # length leaves a dead border column and row.
    kind = args.map_kind if args.map_kind != "random" else p.get("kind", "random")
    size = p["size"] | 1 if kind == "maze" else p["size"]
    room_size = p.get("room_size", args.room_size)
    if kind == "random":
        map_spec = {"kind": "random", "height": size, "width": size,
                    "density": p["density"]}
    elif kind == "room":
        map_spec = {"kind": "room", "height": size, "width": size,
                    "room_size": room_size}
    else:
        map_spec = {"kind": "maze", "height": size, "width": size}
    cfg = MAPFConfig(
        n_agents=p["n_agents"], max_steps=192, map_pool_size=512, fov=args.fov,
        obs_costmap=args.obs_costmap,
        observe_peer_goals=not hide,
        map_spec=map_spec,
    )
    print(f"generating '{args.preset}': {p['n_agents']} agents, "
          f"{size}x{size} {kind}, fov={args.fov}, "
          f"costmap={args.obs_costmap}, "
          f"peer_goals={'hidden' if hide else 'visible'}")
    meta = generate(cfg, args.samples or p["samples"], out,
                    seed=args.seed, num_envs=args.num_envs)
    print(json.dumps(meta, indent=2))
