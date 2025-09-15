from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import math
import hashlib

import numpy as np

try:
    from utilities.radii import covalent_radius
except Exception:
    _FALLBACK = {
        "H": 0.31, 
    }
    def covalent_radius(sym: str) -> float:
        return _FALLBACK.get(sym.capitalize(), 0.75)


def build_bonds(
    symbols: Sequence[str],
    coords_ang: np.ndarray,
    distance_scale: float = 1.2,
) -> List[Tuple[int, int]]:
    n = len(symbols)
    bonds: List[Tuple[int, int]] = []
    for i in range(n - 1):
        ri = float(covalent_radius(symbols[i]))
        xi, yi, zi = map(float, coords_ang[i])
        for j in range(i + 1, n):
            rj = float(covalent_radius(symbols[j]))
            cutoff = distance_scale * (ri + rj)
            xj, yj, zj = map(float, coords_ang[j])
            dx, dy, dz = xi - xj, yi - yj, zi - zj
            dij = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dij <= cutoff:
                bonds.append((i, j))
    bonds.sort()
    return bonds

def _components(n_atoms: int, bonds: Sequence[Tuple[int, int]]) -> List[List[int]]:
    adj = [[] for _ in range(n_atoms)]
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)
    seen = [False] * n_atoms
    comps: List[List[int]] = []
    for s in range(n_atoms):
        if seen[s]:
            continue
        stack = [s]; seen[s] = True; comp = [s]
        while stack:
            v = stack.pop()
            for w in adj[v]:
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
                    comp.append(w)
        comps.append(sorted(comp))
    return comps


_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

def _sub(n: int) -> str:
    return "" if n == 1 else str(n).translate(_SUB)

def _hill_formula(symbols: Sequence[str]) -> str:
    counts: Dict[str, int] = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    parts: List[str] = []
    if "C" in counts:
        parts.append(f"C{_sub(counts.pop('C'))}")
        if "H" in counts:
            parts.append(f"H{_sub(counts.pop('H'))}")
    for el in sorted(counts):
        parts.append(f"{el}{_sub(counts[el])}")
    return "".join(parts)

def _identity_from_symbols(symbols: Sequence[str]) -> str:
    return _hill_formula(symbols)

def topology_hash(bonds: Sequence[Tuple[int, int]]) -> str:
    s = ",".join(f"{i}-{j}" for (i, j) in sorted(bonds))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def summarize_topology(
    symbols: Sequence[str],
    coords_ang: np.ndarray,
    distance_scale: float = 1.2,
) -> Dict[str, object]:
    coords_ang = np.asarray(coords_ang, dtype=float)
    if coords_ang.ndim != 2 or coords_ang.shape[1] != 3:
        raise ValueError("coords_ang must be shape (n_atoms, 3) in Å")

    bonds = build_bonds(symbols, coords_ang, distance_scale)
    comps = _components(len(symbols), bonds)

    comp_summaries = []
    for comp in comps:
        comp_symbols = [symbols[i] for i in comp]
        comp_summaries.append({
            "indices": comp,
            "formula": _hill_formula(comp_symbols),
            "identity": _identity_from_symbols(comp_symbols),
        })

    return {
        "bonds": bonds,
        "components": comp_summaries,
        "hash": topology_hash(bonds),
    }
