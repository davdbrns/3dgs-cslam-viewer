"""Discovering and loading C-SLAM run folders (manifest.json, trajectories.npz, loops.npz, maps)."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RUN_MARKERS = ("manifest.json", "trajectories.npz")


@dataclass
class Agent:
    id: int
    kf_ids: np.ndarray  # (K,) frame ids of the keyframes
    c2w: np.ndarray     # (K, 4, 4) optimized camera-to-world, OpenCV camera convention
    gt_kf_c2w: np.ndarray | None = None  # (K, 4, 4) ground truth at the same keyframes
    gt_c2w: np.ndarray | None = None     # (F, 4, 4) ground truth at every frame, only for drawing


@dataclass
class Loops:
    src_pos: np.ndarray   # (L, 3)
    tgt_pos: np.ndarray   # (L, 3)
    accepted: np.ndarray  # (L,) bool
    intra: np.ndarray     # (L,) bool, True when both ends belong to the same agent


@dataclass
class Run:
    name: str
    map_paths: list[Path]
    agents: list[Agent] = field(default_factory=list)
    loops: Loops | None = None


def discover(paths: list[Path]) -> list[Run]:
    """Turns dropped/passed files and folders into runs. A folder with a run marker is one run;
    PLY files outside any run become their own map-only run."""
    roots: list[Path] = []
    loose_maps: list[Path] = []
    for p in paths:
        p = p.expanduser().resolve()
        if p.is_dir():
            found = sorted({m.parent for name in RUN_MARKERS for m in p.rglob(name)})
            roots += found
            if not found:
                loose_maps += sorted(p.rglob("*.ply"))
        elif p.name in RUN_MARKERS or p.name == "loops.npz":
            roots.append(p.parent)
        elif p.suffix.lower() == ".ply":
            loose_maps.append(p)
        else:
            raise ValueError(f"Unsupported file: {p.name}")

    runs = [load_run(r) for r in dict.fromkeys(roots)]
    claimed = {m for r in runs for m in r.map_paths}
    runs += [Run(m.stem, [m]) for m in dict.fromkeys(loose_maps) if m not in claimed]
    return runs


def load_run(root: Path) -> Run:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if "maps" in manifest:
        maps = [root / m for m in manifest["maps"] if (root / m).exists()]
    else:
        # The _yup copies live in a different world frame than the poses.
        maps = [m for m in sorted(root.glob("*.ply")) + sorted(root.glob("maps/*.ply"))
                if "_yup" not in m.stem]

    agents = []
    traj_path = root / "trajectories.npz"
    if traj_path.exists():
        z = np.load(traj_path)
        for a in z["agent_ids"].tolist():
            key = f"agent{a}_optimized_kf_c2w"
            if key not in z:
                key = f"agent{a}_estimated_kf_c2w"
            gt_kf = z[f"agent{a}_gt_kf_c2w"] if f"agent{a}_gt_kf_c2w" in z else None
            gt_all = z[f"agent{a}_gt_c2w"] if f"agent{a}_gt_c2w" in z else None
            agents.append(Agent(int(a), z[f"agent{a}_kf_ids"], z[key].astype(np.float64), gt_kf, gt_all))

    loops = None
    loops_path = root / "loops.npz"
    if loops_path.exists():
        z = np.load(loops_path)
        if len(z["src_pos"]):
            loops = Loops(z["src_pos"], z["tgt_pos"], z["accepted"].astype(bool), z["intra"].astype(bool))

    return Run(root.name, maps, agents, loops)


def gt_pairs(agent: Agent) -> tuple[np.ndarray, np.ndarray]:
    """Estimated and ground-truth keyframe positions, skipping keyframes without valid ground truth."""
    valid = np.isfinite(agent.gt_kf_c2w).all(axis=(1, 2))
    return agent.c2w[valid, :3, 3], agent.gt_kf_c2w[valid, :3, 3]


def align_gt(agents: list[Agent]) -> np.ndarray:
    """One rigid transform (Umeyama, no scale) taking ground truth onto the estimate, fitted over
    all agents jointly because they share one world frame. Returns a 4x4 matrix."""
    est, gt = (np.concatenate(x) for x in zip(*(gt_pairs(a) for a in agents)))
    mu_e, mu_g = est.mean(axis=0), gt.mean(axis=0)
    U, _, Vt = np.linalg.svd((est - mu_e).T @ (gt - mu_g))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    T = np.eye(4)
    T[:3, :3] = U @ D @ Vt
    T[:3, 3] = mu_e - T[:3, :3] @ mu_g
    return T


def ate_rmse(agent: Agent, gt_to_est: np.ndarray) -> float:
    est, gt = gt_pairs(agent)
    gt = gt @ gt_to_est[:3, :3].T + gt_to_est[:3, 3]
    return float(np.sqrt(np.mean(np.sum((est - gt) ** 2, axis=1))))


def estimate_up(agents: list[Agent], snap_deg: float = 20.0) -> np.ndarray | None:
    """Gravity-up guess: cameras are mostly held upright, so their mean up vector (-Y in OpenCV)
    points against gravity. Snaps to a world axis when close, since SLAM worlds are usually
    gravity-aligned and the mean is biased by camera pitch."""
    if not agents:
        return None
    ups = np.concatenate([-a.c2w[:, :3, 1] for a in agents])
    mean = ups.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm < 0.3:
        return None
    mean /= norm
    axis = int(np.argmax(np.abs(mean)))
    if abs(mean[axis]) > np.cos(np.deg2rad(snap_deg)):
        snapped = np.zeros(3)
        snapped[axis] = np.sign(mean[axis])
        return snapped
    return mean
