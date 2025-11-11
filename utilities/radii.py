from __future__ import annotations
from functools import lru_cache
from typing import Optional

from mendeleev import element as _me
from periodictable import elements as _pt

_PM_TO_ANG = 0.01

def _norm_radius(val: Optional[float]) -> Optional[float]:
    if val is None:
        return None
    v = float(val)
    if v <= 0:
        return None
    return v * _PM_TO_ANG if v > 10.0 else v

@lru_cache(maxsize=None)
def _m(symbol: str):
    return _me(symbol.capitalize())

@lru_cache(maxsize=None)
def _p(symbol: str):
    return getattr(_pt, symbol.capitalize())

def _pick_radius(*vals: Optional[float]) -> float:
    for v in vals:
        nv = _norm_radius(v)
        if nv is not None:
            return float(nv)
    raise ValueError("no usable radius value found")

def covalent_radius(symbol: str) -> float:
    m = _m(symbol)
    p = _p(symbol)
    return _pick_radius(
        getattr(m, "covalent_radius_pyykko", None),
        getattr(m, "covalent_radius_cordero", None),
        getattr(p, "covalent_radius", None),
    )

def vdw_radius(symbol: str) -> float:
    m = _m(symbol)
    p = _p(symbol)
    try:
        return _pick_radius(
            getattr(m, "vdw_radius", None),
            getattr(p, "vdw_radius", None),
        )
    except ValueError:
        return 1.2 * covalent_radius(symbol)

def get_atomic_mass(symbol: str) -> float:
    m = _m(symbol)
    p = _p(symbol)
    for candidate in (
        getattr(m, "atomic_weight", None),
        getattr(p, "mass", None),
    ):
        if candidate is not None and float(candidate) > 0:
            return float(candidate)
    raise ValueError(f"no atomic mass for {symbol}")
