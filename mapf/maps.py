"""Grid map loading, generation and scenario sampling for MAPF.

Convention: obstacle maps are boolean arrays where True == blocked.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

# MovingAI terrain characters that an agent may traverse.
_PASSABLE = frozenset(".GS")


def load_movingai(path: str) -> np.ndarray:
    """Parse a MovingAI benchmark ``.map`` file into a boolean obstacle grid."""
    with open(path, "r") as fh:
        lines = fh.read().splitlines()

    height = width = None
    body_start = 0
    for i, line in enumerate(lines[:16]):
        token = line.strip().lower()
        if token.startswith("height"):
            height = int(token.split()[1])
        elif token.startswith("width"):
            width = int(token.split()[1])
        elif token == "map":
            body_start = i + 1
            break
    if height is None or width is None:
        raise ValueError(f"{path}: malformed MovingAI header (missing height/width)")

    grid = np.ones((height, width), dtype=bool)
    for r, line in enumerate(lines[body_start:body_start + height]):
        for c in range(width):
            ch = line[c] if c < len(line) else "@"
            grid[r, c] = ch not in _PASSABLE
    return grid


def random_map(height: int, width: int, density: float, rng: np.random.Generator) -> np.ndarray:
    """Uniform random obstacle placement, as in MovingAI ``random-*`` maps."""
    return rng.random((height, width)) < density


def room_map(
    height: int,
    width: int,
    room_size: int = 8,
    door_width: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Rooms separated by walls with a door in each wall segment."""
    rng = rng or np.random.default_rng()
    grid = np.zeros((height, width), dtype=bool)
    for r in range(room_size, height, room_size + 1):
        grid[r, :] = True
    for c in range(room_size, width, room_size + 1):
        grid[:, c] = True

    # Punch doors so the free space stays connected.
    for r in range(room_size, height, room_size + 1):
        for c0 in range(0, width, room_size + 1):
            seg_end = min(c0 + room_size, width)
            if seg_end > c0:
                d = rng.integers(c0, seg_end)
                grid[r, d:d + door_width] = False
    for c in range(room_size, width, room_size + 1):
        for r0 in range(0, height, room_size + 1):
            seg_end = min(r0 + room_size, height)
            if seg_end > r0:
                d = rng.integers(r0, seg_end)
                grid[d:d + door_width, c] = False
    return grid


def maze_map(height: int, width: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Recursive-backtracker maze on the odd-indexed lattice."""
    rng = rng or np.random.default_rng()
    grid = np.ones((height, width), dtype=bool)
    start = (1, 1)
    grid[start] = False
    stack = [start]
    while stack:
        r, c = stack[-1]
        nbrs = []
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nr, nc = r + dr, c + dc
            if 0 < nr < height - 1 and 0 < nc < width - 1 and grid[nr, nc]:
                nbrs.append((nr, nc, dr, dc))
        if not nbrs:
            stack.pop()
            continue
        nr, nc, dr, dc = nbrs[rng.integers(len(nbrs))]
        grid[r + dr // 2, c + dc // 2] = False
        grid[nr, nc] = False
        stack.append((nr, nc))
    return grid


def connected_components(obstacles: np.ndarray) -> np.ndarray:
    """Label 4-connected free-space components. Obstacles get label -1.

    Uses iterative numpy label propagation -- no scipy dependency.
    """
    height, width = obstacles.shape
    free = ~obstacles
    labels = np.where(free, np.arange(height * width).reshape(height, width), -1).astype(np.int64)
    while True:
        prev = labels
        cand = labels.copy()
        # Propagate the maximum label to 4-connected free neighbours.
        cand[1:, :] = np.maximum(cand[1:, :], np.where(free[:-1, :], labels[:-1, :], -1))
        cand[:-1, :] = np.maximum(cand[:-1, :], np.where(free[1:, :], labels[1:, :], -1))
        cand[:, 1:] = np.maximum(cand[:, 1:], np.where(free[:, :-1], labels[:, :-1], -1))
        cand[:, :-1] = np.maximum(cand[:, :-1], np.where(free[:, 1:], labels[:, 1:], -1))
        labels = np.where(free, cand, -1)
        if np.array_equal(labels, prev):
            return labels


def largest_component_mask(obstacles: np.ndarray) -> np.ndarray:
    """Boolean mask of the largest 4-connected free region."""
    labels = connected_components(obstacles)
    free_labels = labels[labels >= 0]
    if free_labels.size == 0:
        raise ValueError("map has no free cells")
    values, counts = np.unique(free_labels, return_counts=True)
    return labels == values[counts.argmax()]


def sample_scenario(
    obstacles: np.ndarray,
    n_agents: int,
    rng: np.random.Generator,
    region: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample distinct starts and goals inside one connected free region.

    Returns ``(starts, goals)`` each shaped ``(n_agents, 2)`` as (row, col).
    Sampling within a single component guarantees every goal is reachable.
    """
    if region is None:
        region = largest_component_mask(obstacles)
    cells = np.argwhere(region)
    if len(cells) < 2 * n_agents:
        raise ValueError(
            f"region has {len(cells)} free cells, need >= {2 * n_agents} for {n_agents} agents"
        )
    start_idx = rng.choice(len(cells), size=n_agents, replace=False)
    goal_idx = rng.choice(len(cells), size=n_agents, replace=False)
    return cells[start_idx].copy(), cells[goal_idx].copy()


def make_map(spec: dict, rng: np.random.Generator) -> np.ndarray:
    """Build an obstacle grid from a config dict.

    Examples::

        {"kind": "random", "height": 32, "width": 32, "density": 0.2}
        {"kind": "room",   "height": 32, "width": 32, "room_size": 8}
        {"kind": "maze",   "height": 33, "width": 33}
        {"kind": "file",   "path": "data/maps/random-32-32-20.map"}
    """
    kind = spec.get("kind", "random")
    if kind == "file":
        return load_movingai(spec["path"])
    if kind == "random":
        return random_map(spec["height"], spec["width"], spec.get("density", 0.2), rng)
    if kind == "room":
        return room_map(spec["height"], spec["width"], spec.get("room_size", 8), rng=rng)
    if kind == "maze":
        return maze_map(spec["height"], spec["width"], rng=rng)
    if kind == "empty":
        return np.zeros((spec["height"], spec["width"]), dtype=bool)
    raise ValueError(f"unknown map kind: {kind!r}")
