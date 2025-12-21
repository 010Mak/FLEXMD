from __future__ import annotations

import datetime
import io
import json
import math
import re
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utilities.radii import vdw_radius

try:
    from skimage.measure import marching_cubes
    _HAVE_MARCHING_CUBES = True
except Exception:
    marching_cubes = None
    _HAVE_MARCHING_CUBES = False


DEFAULT_HANDHELD_TARGET_MAX_MM = 85.0
DEFAULT_VOXEL_SIZE_MM = 0.25
DEFAULT_GAP_MM = 0.25
MAX_TOTAL_VOXELS = 12_000_000


DEFAULT_COLOR_BY_ELEMENT: Dict[str, Dict[str, str]] = {
    "H":  {"name": "white",   "hex": "#FFFFFF"},
    "C":  {"name": "gray",    "hex": "#4A4A4A"},
    "N":  {"name": "blue",    "hex": "#1E88E5"},
    "O":  {"name": "red",     "hex": "#E53935"},
    "F":  {"name": "green",   "hex": "#43A047"},
    "Cl": {"name": "green",   "hex": "#43A047"},
    "Br": {"name": "brown",   "hex": "#6D4C41"},
    "I":  {"name": "purple",  "hex": "#8E24AA"},
    "S":  {"name": "yellow",  "hex": "#FDD835"},
    "P":  {"name": "orange",  "hex": "#FB8C00"},
    "B":  {"name": "salmon",  "hex": "#FF8A65"},
    "Si": {"name": "tan",     "hex": "#A1887F"},
    "Na": {"name": "purple",  "hex": "#8E24AA"},
    "K":  {"name": "purple",  "hex": "#8E24AA"},
    "Li": {"name": "purple",  "hex": "#8E24AA"},
    "Mg": {"name": "green",   "hex": "#43A047"},
    "Ca": {"name": "green",   "hex": "#43A047"},
    "Fe": {"name": "orange",  "hex": "#FB8C00"},
}


def _norm_elem(e: str) -> str:
    if not e:
        return "C"
    e = str(e)
    return (e[0].upper() + e[1:].lower()) if len(e) > 1 else e.upper()


def _sanitize_label(label: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(label).strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "part"


def _safe_float3(pos: Any) -> Tuple[float, float, float]:
    return (float(pos[0]), float(pos[1]), float(pos[2]))


def _bbox_extent_A(positions_A: np.ndarray, radii_A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    mins = positions_A - radii_A[:, None]
    maxs = positions_A + radii_A[:, None]
    mn = np.min(mins, axis=0)
    mx = np.max(maxs, axis=0)
    extent = float(np.max(mx - mn))
    if not math.isfinite(extent) or extent <= 1e-9:
        extent = 1.0
    return mn, mx, extent


def _choose_scale_mm_per_A(
    *,
    size: str,
    target_max_mm: Optional[float],
    scale_mm_per_A: Optional[float],
    extent_A: float,
) -> float:
    if scale_mm_per_A is not None:
        s = float(scale_mm_per_A)
        if not math.isfinite(s) or s <= 0:
            raise ValueError("scale_mm_per_A must be > 0")
        return s

    if (size or "").lower() == "handheld":
        t = float(target_max_mm) if target_max_mm is not None else DEFAULT_HANDHELD_TARGET_MAX_MM
        s = t / max(1e-9, float(extent_A))
        s = max(3.0, min(12.0, s))
        return float(s)

    t = float(target_max_mm) if target_max_mm is not None else DEFAULT_HANDHELD_TARGET_MAX_MM
    return float(t / max(1e-9, float(extent_A)))

def _adjust_voxel_size(span_mm: np.ndarray, voxel_size_mm: float) -> Tuple[float, Tuple[int, int, int], bool]:
    vox = float(voxel_size_mm)
    if vox <= 0 or not math.isfinite(vox):
        vox = DEFAULT_VOXEL_SIZE_MM

    adjusted = False

    for _ in range(12):
        dims = tuple(int(max(1, math.ceil(float(span) / vox))) for span in span_mm)
        total = int(dims[0] * dims[1] * dims[2])

        if total <= MAX_TOTAL_VOXELS:
            return vox, dims, adjusted

        factor = (total / MAX_TOTAL_VOXELS) ** (1.0 / 3.0)
        vox *= float(factor) * 1.03
        adjusted = True

    dims = tuple(int(max(1, math.ceil(float(span) / vox))) for span in span_mm)
    total = int(dims[0] * dims[1] * dims[2])
    raise RuntimeError(f"voxel grid too large even after adjustment ({total} voxels)")

def _range_for_center(
    *,
    center: float,
    radius: float,
    origin: float,
    vox: float,
    n: int,
) -> Tuple[int, int]:
    lo = int(math.ceil(((center - radius - origin) / vox) - 0.5))
    hi = int(math.floor(((center + radius - origin) / vox) - 0.5))
    lo = max(0, lo)
    hi = min(n - 1, hi)
    return lo, hi


def _fill_inf(arr: np.ndarray, fill: float) -> None:
    m = ~np.isfinite(arr)
    if np.any(m):
        arr[m] = np.float32(fill)


def _compute_best_second_winner_atom(
    *,
    positions_mm: np.ndarray,
    radii_mm: np.ndarray,
    origin_mm: np.ndarray,
    voxel_size_mm: float,
    dims: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = dims
    vox = float(voxel_size_mm)

    best = np.full((nx, ny, nz), np.inf, dtype=np.float32)
    second = np.full((nx, ny, nz), np.inf, dtype=np.float32)
    winner = np.full((nx, ny, nz), -1, dtype=np.int32)

    shell_pad = 2.5 * vox

    for i, (c, r) in enumerate(zip(positions_mm, radii_mm)):
        r = float(r)
        if not math.isfinite(r) or r <= 0:
            continue

        cx, cy, cz = float(c[0]), float(c[1]), float(c[2])
        r_pad = r + shell_pad
        r_pad2 = r_pad * r_pad

        ix0, ix1 = _range_for_center(center=cx, radius=r_pad, origin=float(origin_mm[0]), vox=vox, n=nx)
        iy0, iy1 = _range_for_center(center=cy, radius=r_pad, origin=float(origin_mm[1]), vox=vox, n=ny)
        iz0, iz1 = _range_for_center(center=cz, radius=r_pad, origin=float(origin_mm[2]), vox=vox, n=nz)
        if ix1 < ix0 or iy1 < iy0 or iz1 < iz0:
            continue

        xs = float(origin_mm[0]) + (np.arange(ix0, ix1 + 1, dtype=np.float32) + 0.5) * np.float32(vox)
        ys = float(origin_mm[1]) + (np.arange(iy0, iy1 + 1, dtype=np.float32) + 0.5) * np.float32(vox)
        zs = float(origin_mm[2]) + (np.arange(iz0, iz1 + 1, dtype=np.float32) + 0.5) * np.float32(vox)

        dx2 = (xs - np.float32(cx)) ** 2
        dy2 = (ys - np.float32(cy)) ** 2
        dz2 = (zs - np.float32(cz)) ** 2

        d2 = dx2[:, None, None] + dy2[None, :, None] + dz2[None, None, :]
        near = d2 <= np.float32(r_pad2)
        if not np.any(near):
            continue

        d = np.sqrt(d2, dtype=np.float32) - np.float32(r)

        sub_best = best[ix0:ix1 + 1, iy0:iy1 + 1, iz0:iz1 + 1]
        sub_second = second[ix0:ix1 + 1, iy0:iy1 + 1, iz0:iz1 + 1]
        sub_win = winner[ix0:ix1 + 1, iy0:iy1 + 1, iz0:iz1 + 1]

        m1 = near & (d < sub_best)
        if np.any(m1):
            sub_second[m1] = sub_best[m1]
            sub_best[m1] = d[m1]
            sub_win[m1] = np.int32(i)

        m2 = near & (~m1) & (d < sub_second)
        if np.any(m2):
            sub_second[m2] = d[m2]

    span = np.array([nx, ny, nz], dtype=np.float32) * np.float32(vox)
    fill = float(np.max(span) + 10.0 * vox)
    _fill_inf(best, fill)
    _fill_inf(second, fill)

    return best, second, winner


def _update_min_field_for_atoms(
    field: np.ndarray,
    *,
    atom_indices: List[int],
    positions_mm: np.ndarray,
    radii_mm: np.ndarray,
    origin_mm: np.ndarray,
    voxel_size_mm: float,
    dims: Tuple[int, int, int],
) -> None:
    nx, ny, nz = dims
    vox = float(voxel_size_mm)
    shell_pad = 2.5 * vox

    for i in atom_indices:
        c = positions_mm[i]
        r = float(radii_mm[i])
        if not math.isfinite(r) or r <= 0:
            continue

        cx, cy, cz = float(c[0]), float(c[1]), float(c[2])
        r_pad = r + shell_pad
        r_pad2 = r_pad * r_pad

        ix0, ix1 = _range_for_center(center=cx, radius=r_pad, origin=float(origin_mm[0]), vox=vox, n=nx)
        iy0, iy1 = _range_for_center(center=cy, radius=r_pad, origin=float(origin_mm[1]), vox=vox, n=ny)
        iz0, iz1 = _range_for_center(center=cz, radius=r_pad, origin=float(origin_mm[2]), vox=vox, n=nz)
        if ix1 < ix0 or iy1 < iy0 or iz1 < iz0:
            continue

        xs = float(origin_mm[0]) + (np.arange(ix0, ix1 + 1, dtype=np.float32) + 0.5) * np.float32(vox)
        ys = float(origin_mm[1]) + (np.arange(iy0, iy1 + 1, dtype=np.float32) + 0.5) * np.float32(vox)
        zs = float(origin_mm[2]) + (np.arange(iz0, iz1 + 1, dtype=np.float32) + 0.5) * np.float32(vox)

        dx2 = (xs - np.float32(cx)) ** 2
        dy2 = (ys - np.float32(cy)) ** 2
        dz2 = (zs - np.float32(cz)) ** 2

        d2 = dx2[:, None, None] + dy2[None, :, None] + dz2[None, None, :]
        near = d2 <= np.float32(r_pad2)
        if not np.any(near):
            continue

        d = np.sqrt(d2, dtype=np.float32) - np.float32(r)

        sub = field[ix0:ix1 + 1, iy0:iy1 + 1, iz0:iz1 + 1]
        upd = near & (d < sub)
        if np.any(upd):
            sub[upd] = d[upd]


def _mesh_from_sdf(
    sdf: np.ndarray,
    *,
    voxel_size_mm: float,
    origin_mm: np.ndarray,
    level: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if not _HAVE_MARCHING_CUBES or marching_cubes is None:
        raise RuntimeError(
            "Smooth export requires scikit-image. Install with: "
            "conda install -c conda-forge scikit-image  OR  pip install scikit-image"
        )

    vox = float(voxel_size_mm)
    verts, faces, _normals, _values = marching_cubes(
        sdf.astype(np.float32, copy=False),
        level=float(level),
        spacing=(vox, vox, vox),
        allow_degenerate=False,
    )

    offset = origin_mm.astype(np.float32) + np.float32(0.5 * vox)
    verts_mm = verts.astype(np.float32) + offset[None, :]
    return verts_mm, faces.astype(np.int32)


def _write_binary_stl_mesh(
    *,
    verts_mm: np.ndarray,
    faces: np.ndarray,
    name: str,
) -> Tuple[bytes, int]:
    import struct

    faces = np.asarray(faces, dtype=np.int32)
    n_tris = int(faces.shape[0])

    bio = io.BytesIO()
    header = (name[:80]).encode("ascii", "ignore")[:80]
    header = header + b" " * (80 - len(header))
    bio.write(header)
    bio.write(struct.pack("<I", n_tris))

    for f in faces:
        a = verts_mm[int(f[0])]
        b = verts_mm[int(f[1])]
        c = verts_mm[int(f[2])]

        ux, uy, uz = (b - a).tolist()
        vx, vy, vz = (c - a).tolist()
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        if norm > 1e-12 and math.isfinite(norm):
            nx /= norm
            ny /= norm
            nz /= norm
        else:
            nx = ny = nz = 0.0

        bio.write(struct.pack("<fff", float(nx), float(ny), float(nz)))
        bio.write(struct.pack("<fff", float(a[0]), float(a[1]), float(a[2])))
        bio.write(struct.pack("<fff", float(b[0]), float(b[1]), float(b[2])))
        bio.write(struct.pack("<fff", float(c[0]), float(c[1]), float(c[2])))
        bio.write(struct.pack("<H", 0))

    return bio.getvalue(), n_tris


def build_spacefill_kit_zip(
    *,
    atoms_json: List[Dict[str, Any]],
    identity: Optional[Dict[str, Any]] = None,
    size: str = "handheld",
    target_max_mm: Optional[float] = None,
    scale_mm_per_A: Optional[float] = None,
    voxel_size_mm: Optional[float] = None,
    gap_mm: Optional[float] = None,
    split_mode: str = "element",
    label_map: Optional[Dict[str, str]] = None,
    color_map: Optional[Dict[str, Any]] = None,
    center: bool = True,
) -> Tuple[bytes, Dict[str, Any]]:
    if not isinstance(atoms_json, list) or not atoms_json:
        raise ValueError("atoms_json must be a non-empty list")

    if not _HAVE_MARCHING_CUBES:
        raise RuntimeError(
            "Smooth export requires scikit-image. Install with: "
            "conda install -c conda-forge scikit-image  OR  pip install scikit-image"
        )

    label_map = dict(label_map or {})
    color_map = dict(color_map or {})

    elements = [_norm_elem(a.get("element", "C")) for a in atoms_json]
    posA = np.array([_safe_float3(a.get("position", (0, 0, 0))) for a in atoms_json], dtype=np.float32)
    radiiA = np.array([float(vdw_radius(e)) for e in elements], dtype=np.float32)

    _, _, extentA = _bbox_extent_A(posA, radiiA)
    scale = _choose_scale_mm_per_A(size=size, target_max_mm=target_max_mm, scale_mm_per_A=scale_mm_per_A, extent_A=extentA)

    pos_mm = posA * np.float32(scale)
    radii_mm = radiiA * np.float32(scale)

    mins_mm = pos_mm - radii_mm[:, None]
    maxs_mm = pos_mm + radii_mm[:, None]
    mn_mm = np.min(mins_mm, axis=0).astype(np.float32)
    mx_mm = np.max(maxs_mm, axis=0).astype(np.float32)

    vox_req = float(voxel_size_mm) if voxel_size_mm is not None else DEFAULT_VOXEL_SIZE_MM
    vox_req = max(0.05, float(vox_req))

    vox = vox_req
    vox_adjusted_any = False
    dims = (1, 1, 1)
    mn_pad = mn_mm.copy()
    mx_pad = mx_mm.copy()

    for _ in range(3):
        pad = np.float32(2.0 * vox)
        mn_pad = mn_mm - pad
        mx_pad = mx_mm + pad
        span = (mx_pad - mn_pad).astype(np.float32)
        vox2, dims2, adj = _adjust_voxel_size(span_mm=span, voxel_size_mm=vox)
        vox_adjusted_any = vox_adjusted_any or adj
        vox = vox2
        dims = dims2

    origin = mn_pad.astype(np.float32)
    if center:
        ctr = (mn_pad + mx_pad) * 0.5
        pos_mm = pos_mm - ctr
        mn_pad = mn_pad - ctr
        mx_pad = mx_pad - ctr
        origin = origin - ctr.astype(np.float32)

    best, second, winner_atom = _compute_best_second_winner_atom(
        positions_mm=pos_mm,
        radii_mm=radii_mm,
        origin_mm=origin,
        voxel_size_mm=vox,
        dims=dims,
    )
    union_mask = best <= np.float32(0.0)

    full_verts, full_faces = _mesh_from_sdf(best, voxel_size_mm=vox, origin_mm=origin, level=0.0)
    full_stl, full_tris = _write_binary_stl_mesh(verts_mm=full_verts, faces=full_faces, name="full_reference")

    stls: Dict[str, bytes] = {"full_reference.stl": full_stl}
    tri_counts: Dict[str, int] = {"full_reference.stl": int(full_tris)}

    split_mode = (split_mode or "element").lower().strip()
    if split_mode not in {"none", "element", "atom"}:
        raise ValueError("split_mode must be one of: none, element, atom")

    if split_mode == "none":
        label_names = ["all"]
        atom_label_ids = np.zeros((len(elements),), dtype=np.int32)
    elif split_mode == "atom":
        label_names = [f"{elements[i]}_{i+1}" for i in range(len(elements))]
        atom_label_ids = np.arange(len(elements), dtype=np.int32)
    else:
        atom_labels = [str(label_map.get(e, e)) for e in elements]
        label_names = sorted(set(atom_labels))
        label_to_id = {nm: i for i, nm in enumerate(label_names)}
        atom_label_ids = np.array([label_to_id[nm] for nm in atom_labels], dtype=np.int32)

    winner_lbl = np.full_like(winner_atom, -1, dtype=np.int32)
    valid = winner_atom >= 0
    if np.any(valid):
        winner_lbl[valid] = atom_label_ids[winner_atom[valid]]

    resolved_colors: Dict[str, Dict[str, str]] = {}
    for lname in label_names:
        c = color_map.get(lname)
        if isinstance(c, dict):
            resolved_colors[lname] = {
                "name": str(c.get("name") or "gray"),
                "hex": str(c.get("hex") or "#888888"),
            }
        elif isinstance(c, str):
            resolved_colors[lname] = {"name": c, "hex": "#888888"}
        else:
            base_el = _norm_elem(lname.split("_", 1)[0]) if "_" in lname else _norm_elem(lname)
            resolved_colors[lname] = dict(DEFAULT_COLOR_BY_ELEMENT.get(base_el, {"name": "gray", "hex": "#888888"}))

    parts: List[Dict[str, Any]] = []

    if split_mode != "none":
        gap = float(gap_mm) if gap_mm is not None else DEFAULT_GAP_MM
        gap = max(0.0, float(gap))

        counts = [int(np.sum(union_mask & (winner_lbl == lid))) for lid in range(len(label_names))]
        base_lid = int(np.argmax(np.array(counts, dtype=np.int64))) if counts else None

        if split_mode == "atom":
            fill = float(np.max(np.abs(best)) + 10.0 * float(vox))
            nx, ny, nz = dims
            vox_f = float(vox)

            for atom_i, lname in enumerate(label_names):
                delta = gap if (base_lid is not None and atom_i != base_lid) else 0.0

                part_sdf = np.full_like(best, np.float32(fill), dtype=np.float32)

                c = pos_mm[atom_i]
                r = float(radii_mm[atom_i])
                shell_pad = 2.5 * vox_f
                r_pad = r + shell_pad
                r_pad2 = r_pad * r_pad

                ix0, ix1 = _range_for_center(center=float(c[0]), radius=r_pad, origin=float(origin[0]), vox=vox_f, n=nx)
                iy0, iy1 = _range_for_center(center=float(c[1]), radius=r_pad, origin=float(origin[1]), vox=vox_f, n=ny)
                iz0, iz1 = _range_for_center(center=float(c[2]), radius=r_pad, origin=float(origin[2]), vox=vox_f, n=nz)
                if ix1 < ix0 or iy1 < iy0 or iz1 < iz0:
                    continue

                xs = float(origin[0]) + (np.arange(ix0, ix1 + 1, dtype=np.float32) + 0.5) * np.float32(vox_f)
                ys = float(origin[1]) + (np.arange(iy0, iy1 + 1, dtype=np.float32) + 0.5) * np.float32(vox_f)
                zs = float(origin[2]) + (np.arange(iz0, iz1 + 1, dtype=np.float32) + 0.5) * np.float32(vox_f)

                dx2 = (xs - np.float32(c[0])) ** 2
                dy2 = (ys - np.float32(c[1])) ** 2
                dz2 = (zs - np.float32(c[2])) ** 2
                d2 = dx2[:, None, None] + dy2[None, :, None] + dz2[None, None, :]

                near = d2 <= np.float32(r_pad2)
                if not np.any(near):
                    continue

                d_in = np.sqrt(d2, dtype=np.float32) - np.float32(r)

                sub_best = best[ix0:ix1 + 1, iy0:iy1 + 1, iz0:iz1 + 1]
                sub_second = second[ix0:ix1 + 1, iy0:iy1 + 1, iz0:iz1 + 1]
                sub_win = winner_atom[ix0:ix1 + 1, iy0:iy1 + 1, iz0:iz1 + 1]

                min_other = np.where(sub_win == np.int32(atom_i), sub_second, sub_best)

                region = d_in - min_other + np.float32(delta)
                sub_part = np.maximum(sub_best, region)

                sub_field = part_sdf[ix0:ix1 + 1, iy0:iy1 + 1, iz0:iz1 + 1]
                sub_field[near] = sub_part[near]

                verts, faces = _mesh_from_sdf(part_sdf, voxel_size_mm=vox_f, origin_mm=origin, level=0.0)

                safe = _sanitize_label(lname)
                fname = f"part_{safe}.stl"
                stl_bytes, tris = _write_binary_stl_mesh(verts_mm=verts, faces=faces, name=f"part_{safe}")

                stls[fname] = stl_bytes
                tri_counts[fname] = int(tris)

                parts.append(
                    {
                        "label": lname,
                        "filename": fname,
                        "recommended_color": resolved_colors.get(lname),
                        "triangle_count": int(tris),
                        "is_base": bool(base_lid is not None and atom_i == base_lid),
                        "clearance_mm": float(delta),
                    }
                )

        else:
            all_indices = list(range(len(elements)))
            span = np.array(dims, dtype=np.float32) * np.float32(vox)
            fill2 = float(np.max(span) + 10.0 * float(vox))

            for lid, lname in enumerate(label_names):
                delta = gap if (base_lid is not None and lid != base_lid) else 0.0

                in_atoms = [i for i in all_indices if atom_label_ids[i] == lid]
                out_atoms = [i for i in all_indices if atom_label_ids[i] != lid]
                if not in_atoms:
                    continue

                d_in = np.full_like(best, np.inf, dtype=np.float32)
                _update_min_field_for_atoms(
                    d_in,
                    atom_indices=in_atoms,
                    positions_mm=pos_mm,
                    radii_mm=radii_mm,
                    origin_mm=origin,
                    voxel_size_mm=vox,
                    dims=dims,
                )

                d_out = np.full_like(best, np.inf, dtype=np.float32)
                if out_atoms:
                    _update_min_field_for_atoms(
                        d_out,
                        atom_indices=out_atoms,
                        positions_mm=pos_mm,
                        radii_mm=radii_mm,
                        origin_mm=origin,
                        voxel_size_mm=vox,
                        dims=dims,
                    )

                _fill_inf(d_in, fill2)
                _fill_inf(d_out, fill2)

                region = (d_in - d_out) + np.float32(delta)
                part_sdf = np.maximum(best, region)

                verts, faces = _mesh_from_sdf(part_sdf, voxel_size_mm=vox, origin_mm=origin, level=0.0)

                safe = _sanitize_label(lname)
                fname = f"part_{safe}.stl"
                stl_bytes, tris = _write_binary_stl_mesh(verts_mm=verts, faces=faces, name=f"part_{safe}")

                stls[fname] = stl_bytes
                tri_counts[fname] = int(tris)

                parts.append(
                    {
                        "label": lname,
                        "filename": fname,
                        "recommended_color": resolved_colors.get(lname),
                        "triangle_count": int(tris),
                        "is_base": bool(base_lid is not None and lid == base_lid),
                        "clearance_mm": float(delta),
                    }
                )

        parts.sort(key=lambda p: (not p.get("is_base", False), str(p.get("label", ""))))

    guide_lines: List[str] = []
    guide_lines.append("FLEXMD Space-Filling Print Kit (smooth seams)")
    guide_lines.append("")
    guide_lines.append("How to use:")
    guide_lines.append("1) Print each part STL in the recommended color.")
    guide_lines.append("2) Assemble starting with the base part (largest).")
    guide_lines.append("3) If tight: reduce gap_mm, or lightly sand seam faces.")
    guide_lines.append("")
    guide_lines.append(f"split_mode={split_mode}")
    guide_lines.append(f"voxel_size_mm_used={vox:.3f}")
    if vox_adjusted_any:
        guide_lines.append("NOTE: voxel_size_mm was auto-increased to cap memory.")
    guide_lines.append("")

    if parts:
        guide_lines.append("Parts:")
        for p in parts:
            col = p.get("recommended_color") or {}
            guide_lines.append(f"- {p['filename']} (label={p['label']}, color={col.get('name','')}, {col.get('hex','')})")
    else:
        guide_lines.append("Parts: (none)")

    assembly_txt = "\n".join(guide_lines).encode("utf-8")

    now = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    bbox_mm = {
        "min": [float(mn_pad[0]), float(mn_pad[1]), float(mn_pad[2])],
        "max": [float(mx_pad[0]), float(mx_pad[1]), float(mx_pad[2])],
        "max_dim": float(np.max(mx_pad - mn_pad)),
    }

    manifest: Dict[str, Any] = {
        "schema": "flexmd.spacefill_kit.v3",
        "generated_utc": now,
        "size_preset": str(size),
        "units": {"input_length": "angstrom", "output_length": "mm"},
        "scaling": {
            "target_max_mm": float(target_max_mm) if target_max_mm is not None else (DEFAULT_HANDHELD_TARGET_MAX_MM if (str(size).lower() == "handheld") else None),
            "scale_mm_per_A": float(scale),
        },
        "voxel": {
            "voxel_size_mm_requested": float(vox_req),
            "voxel_size_mm_used": float(vox),
            "voxel_size_was_adjusted": bool(vox_adjusted_any),
            "dims": [int(dims[0]), int(dims[1]), int(dims[2])],
            "max_total_voxels": int(MAX_TOTAL_VOXELS),
            "samples": "voxel_centers",
        },
        "geometry": {
            "spacefill": True,
            "surface_method": "sdf + marching_cubes",
            "seams": "distance_based (d_in - d_out)",
            "vdw_radii_source": "utilities.radii.vdw_radius",
            "bbox_mm": bbox_mm,
        },
        "splitting": {
            "split_mode": split_mode,
            "gap_mm": float(gap_mm) if gap_mm is not None else float(DEFAULT_GAP_MM),
            "label_map": label_map,
            "labels": label_names,
            "recommended_colors": resolved_colors,
        },
        "files": {
            "full_reference": "full_reference.stl",
            "parts": parts,
            "assembly_guide": "assembly.txt",
            "input_atoms": "atoms.json",
            "manifest": "manifest.json",
        },
        "stats": {
            "atoms": int(len(atoms_json)),
            "stl_triangles": tri_counts,
        },
    }
    if identity is not None:
        manifest["identity"] = identity

    base = "spacefill_kit"
    if identity and isinstance(identity.get("name"), str) and identity["name"].strip():
        base = _sanitize_label(identity["name"])[:40] or base
    zip_name = f"{base}.zip"

    zbio = io.BytesIO()
    with zipfile.ZipFile(zbio, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
        z.writestr("assembly.txt", assembly_txt)
        z.writestr("atoms.json", json.dumps({"atoms": atoms_json}, indent=2).encode("utf-8"))
        for fname, data in stls.items():
            z.writestr(fname, data)

    manifest.setdefault("package", {})
    manifest["package"].update({"format": "zip", "filename": zip_name})

    return zbio.getvalue(), manifest
