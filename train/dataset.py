"""Loading expert demonstrations for behaviour cloning.

Shards hold whole episodes, so the loader hands back *sub-sequences* -- the
GRU needs contiguous time. One training sample is one agent's slice of one
episode: ``(L, C, F, F)`` grids plus the matching actions.

Sample weighting is the important part. Roughly 46% of expert decisions are
"I am parked on my goal, stay put", which is both trivial and the majority
class. Trained unweighted, the policy collapses onto STAY and every agent
freezes. `parked_weight` downweights those samples; the rare off-goal STAY
(a genuine yield) keeps full weight because it is the coordination signal.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np


class DemoDataset:
    """In-memory demonstration store with sub-sequence sampling."""

    def __init__(
        self,
        root: str,
        seq_len: int = 16,
        parked_weight: float = 0.1,
        val_fraction: float = 0.05,
        max_shards: Optional[int] = None,
        seed: int = 0,
        group_agents: bool = False,
    ):
        # group_agents=True returns every agent of an episode together, shaped
        # (L, B, N, ...) instead of (L, B, ...). Communication couples agents
        # within an environment, so a comm-aware learner cannot be trained on
        # independently sampled agents -- it would see a partial peer graph.
        self.group_agents = group_agents
        self.root = root
        self.seq_len = seq_len
        self.parked_weight = parked_weight
        self.rng = np.random.default_rng(seed)

        meta_path = os.path.join(root, "meta.json")
        self.meta: Dict = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

        files = sorted(glob.glob(os.path.join(root, "shard_*.npz")))
        if not files:
            raise FileNotFoundError(f"no shards in {root}")
        if max_shards:
            files = files[:max_shards]

        # "pos" is only present in shards written after agent positions were
        # added to the format; it is required for group_agents (the peer graph
        # is rebuilt from it) and ignored otherwise.
        keys = ["grid", "vec", "action", "step_ok", "at_goal"]
        self.shards: List[Dict[str, np.ndarray]] = []
        for path in files:
            with np.load(path) as z:
                have = set(z.files)
                if group_agents and "pos" not in have:
                    raise ValueError(
                        f"{path} has no 'pos' array, so the peer graph cannot be "
                        f"rebuilt. Regenerate with scripts/make_demos.py.")
                self.shards.append({k: z[k] for k in keys + (["pos"] if "pos" in have else [])})

        self.n_agents = self.shards[0]["action"].shape[2]
        self.obs_shape = self.shards[0]["grid"].shape[3:]
        self.vec_dim = self.shards[0]["vec"].shape[3]

        index = self._build_index()
        self.rng.shuffle(index)
        n_val = max(1, int(len(index) * val_fraction))
        self.val_index = index[:n_val]
        self.train_index = index[n_val:]

    def _build_index(self) -> np.ndarray:
        """Every valid (shard, episode, agent, start) sub-sequence origin.

        With ``group_agents`` the agent column is a placeholder: one row per
        (episode, start) covering all N agents at once.
        """
        rows = []
        agents = [0] if self.group_agents else range(self.n_agents)
        for s, shard in enumerate(self.shards):
            step_ok = shard["step_ok"]                   # (E, T)
            E, T = step_ok.shape
            lengths = step_ok.sum(1)
            for e in range(E):
                usable = int(lengths[e]) - self.seq_len + 1
                if usable <= 0:
                    # Short episode: one sample, padded by the mask.
                    if lengths[e] > 0:
                        for n in agents:
                            rows.append((s, e, n, 0))
                    continue
                # Stride by half a window to limit redundancy.
                for t0 in range(0, usable, max(1, self.seq_len // 2)):
                    for n in agents:
                        rows.append((s, e, n, t0))
        return np.array(rows, dtype=np.int64)

    # ------------------------------------------------------------------ batch

    def _gather_grouped(self, rows: np.ndarray) -> Dict[str, np.ndarray]:
        """All N agents of each episode, shaped (L, B, N, ...)."""
        L, B, N = self.seq_len, len(rows), self.n_agents
        grid = np.zeros((L, B, N) + self.obs_shape, dtype=np.float32)
        vec = np.zeros((L, B, N, self.vec_dim), dtype=np.float32)
        pos = np.zeros((L, B, N, 2), dtype=np.int64)
        action = np.zeros((L, B, N), dtype=np.int64)
        weight = np.zeros((L, B, N), dtype=np.float32)
        at_goal = np.zeros((L, B, N), dtype=bool)

        for i, (s, e, _, t0) in enumerate(rows):
            sh = self.shards[s]
            valid = int(sh["step_ok"][e].sum())
            hi = min(t0 + L, valid)
            k = hi - t0
            if k <= 0:
                continue
            grid[:k, i] = sh["grid"][e, t0:hi].astype(np.float32)
            vec[:k, i] = sh["vec"][e, t0:hi].astype(np.float32)
            pos[:k, i] = sh["pos"][e, t0:hi].astype(np.int64)
            a = sh["action"][e, t0:hi].astype(np.int64)
            g = sh["at_goal"][e, t0:hi]
            action[:k, i], at_goal[:k, i] = a, g
            w = np.ones((k, N), dtype=np.float32)
            w[g & (a == 0)] = self.parked_weight
            weight[:k, i] = w

        return dict(grid=grid, vec=vec, pos=pos, action=action,
                    weight=weight, at_goal=at_goal)

    def _gather(self, rows: np.ndarray) -> Dict[str, np.ndarray]:
        L, B = self.seq_len, len(rows)
        grid = np.zeros((L, B) + self.obs_shape, dtype=np.float32)
        vec = np.zeros((L, B, self.vec_dim), dtype=np.float32)
        action = np.zeros((L, B), dtype=np.int64)
        weight = np.zeros((L, B), dtype=np.float32)
        at_goal = np.zeros((L, B), dtype=bool)

        for i, (s, e, n, t0) in enumerate(rows):
            sh = self.shards[s]
            valid = int(sh["step_ok"][e].sum())
            hi = min(t0 + L, valid)
            k = hi - t0
            if k <= 0:
                continue
            grid[:k, i] = sh["grid"][e, t0:hi, n].astype(np.float32)
            vec[:k, i] = sh["vec"][e, t0:hi, n].astype(np.float32)
            a = sh["action"][e, t0:hi, n].astype(np.int64)
            g = sh["at_goal"][e, t0:hi, n]
            action[:k, i] = a
            at_goal[:k, i] = g
            w = np.ones(k, dtype=np.float32)
            w[g & (a == 0)] = self.parked_weight     # trivial parked-on-goal STAY
            weight[:k, i] = w

        return dict(grid=grid, vec=vec, action=action, weight=weight, at_goal=at_goal)

    def batches(self, batch_size: int, split: str = "train",
                shuffle: bool = True) -> Iterator[Dict[str, np.ndarray]]:
        index = self.train_index if split == "train" else self.val_index
        order = self.rng.permutation(len(index)) if shuffle else np.arange(len(index))
        gather = self._gather_grouped if self.group_agents else self._gather
        for i in range(0, len(order) - batch_size + 1, batch_size):
            yield gather(index[order[i:i + batch_size]])

    def __repr__(self) -> str:
        return (f"DemoDataset({self.root}, {len(self.shards)} shards, "
                f"{len(self.train_index):,} train / {len(self.val_index):,} val "
                f"sub-sequences of L={self.seq_len}, N={self.n_agents})")
