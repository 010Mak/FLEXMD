from __future__ import annotations
import re
from pathlib import Path
from typing import List, Set, Tuple, Optional, Union
from functools import lru_cache

from utilities.config import FORCEFIELD_DIR

_EL_PAT = re.compile(r"[A-Z][a-z]?")

def elements_from_name(name: str) -> List[str]:
    stem = Path(name).stem
    stem = re.sub(r"^(ffield|reaxff?)[._-]*", "", stem, flags=re.I)
    symbols: List[str] = []
    for frag in re.split(r"[^A-Za-z]", stem):
        symbols.extend(_EL_PAT.findall(frag))
    return symbols

@lru_cache(maxsize=None)
def _find_ffield_files(dir_path: str) -> Tuple[Path, ...]:
    p = Path(dir_path)
    return tuple(sorted(p.glob("ffield*")))

def pick_ffield(
    elements: Set[str],
    ff_dir: Optional[Union[str, Path]] = None,
) -> Tuple[str, List[str]]:
    want = {e.capitalize() for e in elements}
    dir_path = Path(ff_dir or FORCEFIELD_DIR).resolve()
    files = _find_ffield_files(str(dir_path))

    best: Optional[Path] = None
    best_order: List[str] = []
    best_extra = float("inf")

    for ff in files:
        order = elements_from_name(ff.name)
        have = set(order)
        if not want.issubset(have):
            continue
        extra = len(have - want)
        if extra < best_extra or (extra == best_extra and (best is None or ff.name < best.name)):
            best, best_order, best_extra = ff, order, extra

    if best is None:
        raise FileNotFoundError(f"no reaxff file in '{dir_path}' covers elements {sorted(want)}")

    return str(best), best_order

def clear_cache() -> None:
    _find_ffield_files.cache_clear()
