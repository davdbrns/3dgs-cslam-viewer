"""Loading 3DGS PLY files into the arrays viser's splat renderer expects."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814

_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}
_REQUIRED = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


@dataclass
class Splats:
    centers: np.ndarray      # (N, 3) float32
    covariances: np.ndarray  # (N, 3, 3) float32
    rgbs: np.ndarray         # (N, 3) float32 in [0, 1]
    opacities: np.ndarray    # (N, 1) float32 in [0, 1]
    radii: np.ndarray        # (N,) float32, 2 sigma along the largest axis
    total: int               # Gaussians in the file before pruning
    bounds: np.ndarray       # (2, 3) robust min/max, ignores floaters


def read_ply_vertices(path: Path) -> np.ndarray:
    """Memory-maps the vertex element of a binary little-endian PLY."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path.name} is not a PLY file")
        fields, elements, count, in_vertex, fmt = [], [], 0, False, None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path.name}: missing end_header")
            tokens = line.decode("ascii", "replace").split()
            if not tokens:
                continue
            if tokens[0] == "format":
                fmt = tokens[1]
            elif tokens[0] == "element":
                elements.append(tokens[1])
                in_vertex = tokens[1] == "vertex"
                count = int(tokens[2]) if in_vertex else count
            elif tokens[0] == "property" and in_vertex:
                if tokens[1] == "list":
                    raise ValueError(f"{path.name}: list properties are not supported")
                fields.append((tokens[2], "<" + _PLY_TYPES[tokens[1]]))
            elif tokens[0] == "end_header":
                break
        offset = f.tell()
    if not elements or elements[0] != "vertex":
        raise ValueError(f"{path.name}: expected the vertex element to come first")
    if fmt != "binary_little_endian":
        raise ValueError(f"{path.name}: only binary_little_endian PLY is supported (got {fmt})")
    return np.memmap(path, dtype=np.dtype(fields), mode="r", offset=offset, shape=(count,))


def load_splats(path: Path, max_splats: int, min_opacity: float) -> Splats:
    v = read_ply_vertices(path)
    missing = [k for k in _REQUIRED if k not in v.dtype.names]
    if missing:
        raise ValueError(f"{path.name} is not a 3DGS map (missing {', '.join(missing[:3])})")

    opacity = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float32)))
    keep = np.flatnonzero(opacity >= min_opacity)
    if len(keep) > max_splats:
        # Rank by opacity times projected area, so pruning removes what contributes least to the image.
        log_scale_sum = (np.asarray(v["scale_0"][keep], np.float64)
                         + v["scale_1"][keep] + v["scale_2"][keep])
        score = np.log(opacity[keep]) + log_scale_sum * (2.0 / 3.0)
        keep = np.sort(keep[np.argpartition(-score, max_splats)[:max_splats]])

    sub = v[keep]
    centers = np.stack([sub["x"], sub["y"], sub["z"]], axis=1).astype(np.float32)
    scales = np.exp(np.stack([sub["scale_0"], sub["scale_1"], sub["scale_2"]], axis=1)).astype(np.float32)
    quats = np.stack([sub["rot_0"], sub["rot_1"], sub["rot_2"], sub["rot_3"]], axis=1).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True) + 1e-12
    rs = quat_to_matrix(quats) * scales[:, None, :]
    covariances = rs @ rs.transpose(0, 2, 1)
    f_dc = np.stack([sub["f_dc_0"], sub["f_dc_1"], sub["f_dc_2"]], axis=1).astype(np.float32)
    rgbs = np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0)

    if len(centers):
        bounds = np.percentile(centers, [1, 99], axis=0)
    else:
        bounds = np.zeros((2, 3))
    radii = 2.0 * scales.max(axis=1)
    return Splats(centers, covariances, rgbs, opacity[keep, None], radii, len(v), bounds)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """(N, 4) wxyz unit quaternions to (N, 3, 3) rotation matrices."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=1).reshape(-1, 3, 3)
