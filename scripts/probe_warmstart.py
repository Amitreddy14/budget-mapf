"""Factorial probe of a warm-start checkpoint: comm mode x action rule.

The QMIX pre-flight refused a warm start that the PPO pre-flight accepts, and
the two evaluators differ in two ways at once: QMIX runs the checkpoint under
its own comm mode and selects actions greedily, while the recorded number came
from the checkpoint's mode with sampled actions. One of those explains the gap,
or both do, and guessing is how a measurement error becomes a finding.

Greedy action selection is not a free choice for a Q-network -- argmax *is* the
policy -- so if the action rule turns out to carry the drop, every QMIX number
is on a different footing from the MAPPO baseline and the comparison has to say
so.

    python scripts/probe_warmstart.py runs/bccomm_a16h20/best.pt
"""
import argparse, os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from mapf.env import MAPFConfig, VecMAPFEnv
from models.comm_policy import CommActorCritic, CommConfig, ModelConfig
from models.encoder import N_ACTIONS
from eval.metrics import run_episodes


def rollout(model, env_cfg, radius, device, greedy, eps_per_env, num_envs=32,
            epsilon=0.0):
    model.eval()
    env = VecMAPFEnv(env_cfg, num_envs=num_envs, seed=0, pool_seed=999)
    N, B = env_cfg.n_agents, num_envs
    st = {"h": model.initial_state(B * N, device)}

    def act(e):
        grid, vec = e.observe()
        adj_np, rel_np = e.comm_graph(radius)
        with torch.no_grad():
            action, _, _, st["h"], _ = model.act(
                torch.as_tensor(grid, device=device),
                torch.as_tensor(vec, device=device),
                torch.as_tensor(adj_np, device=device),
                torch.as_tensor(rel_np, device=device),
                hidden=st["h"], greedy=greedy)
        if epsilon > 0.0:
            explore = torch.rand(action.shape, device=action.device) < epsilon
            action = torch.where(
                explore, torch.randint_like(action, N_ACTIONS), action)
        return action.cpu().numpy()

    def on_end(finished):
        reset = torch.zeros(B * N, dtype=torch.bool, device=device)
        for b in finished:
            reset[b * N:(b + 1) * N] = True
        st["h"] = st["h"] * (~reset).to(st["h"].dtype).view(1, B * N, 1)

    return run_episodes(env, act, episodes_per_env=eps_per_env,
                        on_episode_end=on_end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--modes", default="none,full,shuffled")
    ap.add_argument("--eps-per-env", type=int, default=4)
    ap.add_argument("--epsilons", default="0",
                    help="comma-separated action noise for the greedy rows. "
                         "A greedy policy that scores far better with noise "
                         "is livelocked, not ignorant.")
    ap.add_argument("--seeds", type=int, default=3,
                    help="repeats; sampled rollouts are stochastic")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ck_comm = (blob.get("comm_cfg", {}) or {}).get("mode")
    recorded = blob.get("env_on_goal")
    spec = blob.get("map_spec")
    n_agents = blob.get("n_agents")
    radius = (blob.get("comm_cfg", {}) or {}).get("radius", 5)

    print(f"checkpoint {args.checkpoint}")
    print(f"  trained with comm={ck_comm!r}, recorded agents-on-goal "
          f"{recorded if recorded is None else round(recorded, 4)}")
    print(f"  n_agents={n_agents}  map_spec={spec}\n")

    env_cfg = MAPFConfig(n_agents=n_agents, fov=blob.get("fov", 9),
                         max_steps=blob.get("max_steps", 256),
                         map_pool_size=64, map_spec=spec,
                         observe_peer_goals=blob.get("observe_peer_goals", True),
                         obs_costmap=blob.get("obs_costmap", "field"))

    probe_env = VecMAPFEnv(env_cfg, num_envs=2, seed=0)
    grid_np, vec_np = probe_env.reset()
    mcfg = ModelConfig(in_channels=grid_np.shape[2], fov=env_cfg.fov,
                       vec_dim=vec_np.shape[-1], state_dim=n_agents * 5)
    ccfg = CommConfig(**(blob.get("comm_cfg") or {}))
    model = CommActorCritic(mcfg, ccfg).to(device)
    own = model.state_dict()
    src = blob["model"]
    copied = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
    assert len(copied) == len(own), (
        f"partial load: {len(copied)}/{len(own)} tensors -- this probe is only "
        f"meaningful on a complete load")
    own.update(copied)
    model.load_state_dict(own)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print(f"{'comm':<10}{'actions':<10}{'on-goal':>10}{'success':>10}"
          f"{'coll/step':>12}{'makespan':>10}")
    print("-" * 62)
    eps_list = [float(e) for e in args.epsilons.split(",") if e.strip() != ""]
    rules = [("sampled", False, 0.0)] + [
        ("greedy" if e == 0.0 else f"greedy+{e:g}", True, e) for e in eps_list]

    rows = {}
    for mode, (label, greedy, eps) in itertools.product(modes, rules):
        model.comm_cfg.mode = mode
        # only a noiseless greedy rollout is deterministic
        reps = 1 if (greedy and eps == 0.0) else args.seeds
        vals = []
        for s in range(reps):
            torch.manual_seed(s); np.random.seed(s)
            vals.append(rollout(model, env_cfg, radius, device, greedy,
                                args.eps_per_env, epsilon=eps))
        agg = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
        rows[(mode, label)] = agg
        print(f"{mode:<10}{label:<10}"
              f"{agg['agents_on_goal']:>10.4f}{agg['success_rate']:>10.4f}"
              f"{agg['collisions_per_step']:>12.4f}{agg['makespan']:>10.1f}")

    # A floor: what a uniformly random policy scores. Without it there is no
    # way to read a low number -- 0.045 on-goal is either a learner that
    # barely works or one that is worse than chance, and those are different
    # findings.
    vals = []
    for s_ in range(args.seeds):
        rng = np.random.default_rng(s_)
        e = VecMAPFEnv(env_cfg, num_envs=32, seed=0, pool_seed=999)
        vals.append(run_episodes(
            e, lambda env_: rng.integers(0, N_ACTIONS, size=(32, n_agents)),
            episodes_per_env=args.eps_per_env))
    rnd = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
    print(f"{'--':<10}{'random':<10}{rnd['agents_on_goal']:>10.4f}"
          f"{rnd['success_rate']:>10.4f}{rnd['collisions_per_step']:>12.4f}"
          f"{rnd['makespan']:>10.1f}")

    print()
    base = rows.get((ck_comm, "sampled"))
    if base is not None:
        if recorded is not None:
            print(f"reproduction check ({ck_comm}, sampled): "
                  f"{base['agents_on_goal']:.4f} vs {recorded:.4f} recorded "
                  f"-> {'OK' if abs(base['agents_on_goal'] - recorded) < 0.1 else 'MISMATCH'}")
        g = rows.get((ck_comm, "greedy"))
        if g is not None:
            print(f"cost of greedy actions   : "
                  f"{g['agents_on_goal'] - base['agents_on_goal']:+.4f} on-goal")
            for label, _, e in rules:
                if e > 0.0 and (ck_comm, label) in rows:
                    print(f"  action noise {e:<4g}        : "
                          f"{rows[(ck_comm, label)]['agents_on_goal'] - g['agents_on_goal']:+.4f}"
                          f" on-goal vs noiseless greedy")
        for m in modes:
            if m != ck_comm and (m, "sampled") in rows:
                print(f"cost of comm={m:<9}: "
                      f"{rows[(m, 'sampled')]['agents_on_goal'] - base['agents_on_goal']:+.4f}"
                      f" on-goal (sampled)")


if __name__ == "__main__":
    main()
