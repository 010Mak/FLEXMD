from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple, Set, Dict, Optional

try:
    from utilities.config import FORCEFIELD_DIR as _CFG_FF_DIR
    _HAS_CFG = True
except Exception:
    _HAS_CFG = False
    _CFG_FF_DIR = Path(".")

_PT: Set[str] = {
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
    "Fr","Ra","Ac","Th","Pa","U"
}

_ORG_OK: Set[str] = {"H","C","N","O","F","P","S","Cl","Br","I","B","Si"}

_IGNORE_TOKENS = {"ff", "ffield", "reax", "reaxff"}

def _default_ff_dir(ff_dir: Optional[Path]) -> Path:
    if ff_dir is None:
        return Path(_CFG_FF_DIR) if _HAS_CFG else Path(".")
    return Path(ff_dir)

def _iter_potential_files(ff_dir: Path) -> Iterable[Path]:
    if not ff_dir.exists():
        return []
    for p in ff_dir.iterdir():
        if not p.is_file():
            continue
        name = p.name.lower()
        if name.startswith("ffield") or "reax" in name or name.endswith(".ff"):
            yield p

def _parse_elements_from_name(filename: str) -> Set[str]:
    base = os.path.basename(filename)
    parts = re.split(r"[^A-Za-z]+", base)
    out: Set[str] = set()

    for part in parts:
        if not part:
            continue
        low = part.lower()
        if low in _IGNORE_TOKENS or low.startswith("reax"):
            continue

        i = 0
        L = len(part)
        while i < L:
            took_two = False
            if i + 1 < L:
                if part[i].isupper() and part[i + 1].islower():
                    cand2 = part[i] + part[i + 1]
                    if cand2 in _PT:
                        out.add(cand2)
                        i += 2
                        took_two = True
                        continue
            if not took_two:
                cand1 = part[i].upper()
                if cand1 in _PT:
                    out.add(cand1)
                i += 1
    return out

def scan_potentials_dir(ff_dir: Optional[Path] = None) -> List[Dict[str, object]]:
    d = _default_ff_dir(ff_dir)
    files = list(_iter_potential_files(d))
    rows: List[Dict[str, object]] = []
    for p in files:
        try:
            els = sorted(_parse_elements_from_name(p.name), key=lambda s: (s not in _ORG_OK, s))
            st = p.stat()
            rows.append({
                "path": str(p),
                "filename": p.name,
                "elements": els,
                "contains_metals": any((e not in _ORG_OK) for e in els),
                "size": int(st.st_size),
                "mtime": float(st.st_mtime),
            })
        except Exception:
            els = sorted(_parse_elements_from_name(p.name))
            rows.append({
                "path": str(p),
                "filename": p.name,
                "elements": els,
                "contains_metals": any((e not in _ORG_OK) for e in els),
                "size": None,
                "mtime": None,
            })
    rows.sort(key=lambda r: (r["contains_metals"], -(len(r["elements"])), r["filename"]))
    return rows

def _score_file_for_elements(want: Set[str], inferred: Set[str], metal_file: bool) -> int:
    if not want.issubset(inferred):
        return -10_000

    extras = len(inferred - want)
    score = len(want) * 100
    score -= extras * 5

    organic_only = all(e in _ORG_OK for e in want)
    if organic_only and metal_file:
        score -= 500

    if want <= {"C", "H", "O"}:
        score += 1

    return score

def candidates_for_elements(elements: Iterable[str], ff_dir: Optional[Path] = None) -> List[Tuple[str, List[str]]]:
    d = _default_ff_dir(ff_dir)
    want: Set[str] = {str(e).capitalize() for e in elements}
    rows = scan_potentials_dir(d)

    coverers: List[Tuple[int, str, bool]] = []
    for r in rows:
        inferred = set(r.get("elements", []))
        metal_file = bool(r.get("contains_metals", False))
        s = _score_file_for_elements(want, inferred, metal_file)
        if s > -10_000:
            coverers.append((s, r["path"], metal_file))

    if not coverers:
        return []

    coverers.sort(key=lambda t: (-t[0], t[1]))
    if all(e in _ORG_OK for e in want):
        ordered = [(s, p) for (s, p, m) in coverers if not m] or [(s, p) for (s, p, m) in coverers]
    else:
        ordered = [(s, p) for (s, p, m) in coverers]

    order = sorted(want)
    return [(p, order) for (s, p) in ordered]

def pick_ffield(elements: Iterable[str], ff_dir: Optional[Path] = None) -> Tuple[str, List[str]]:
    cands = candidates_for_elements(elements, ff_dir)
    if not cands:
        d = _default_ff_dir(ff_dir)
        want = sorted({str(e).capitalize() for e in elements})
        raise FileNotFoundError(f"No ReaxFF potential in {d} covers elements {want}")
    return cands[0]
