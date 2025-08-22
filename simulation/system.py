from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
from utilities.radii import get_atomic_mass

class Atom:
    def __init__(self, element: str, position: List[float], properties: Dict[str, Any] = None):
        self.element = element
        self.position = np.asarray(position, dtype=float)
        self.properties = properties or {}

class Bond:
    def __init__(self, atom1: int, atom2: int):
        self.atom1 = int(atom1)
        self.atom2 = int(atom2)

class System:
    def __init__(self):
        self.atoms: List[Atom] = []
        self.bonds: List[Bond] = []
        self.positions: np.ndarray
        self.velocities: np.ndarray
        self.masses: np.ndarray

    @classmethod
    def from_json(cls, atoms_json: List[Dict[str, Any]], *, bonds: List[List[int]] | None = None) -> "System":
        sys = cls()
        for entry in atoms_json:
            el = entry["element"]
            pos = entry["position"]
            props = entry.get("properties", {})
            sys.atoms.append(Atom(el, pos, props))

        sys.positions = np.vstack([a.position for a in sys.atoms])
        n = len(sys.atoms)
        sys.velocities = np.zeros((n, 3), dtype=float)

        masses = []
        for a in sys.atoms:
            m = a.properties.get("mass", None)
            if m is None:
                m = get_atomic_mass(a.element)
            masses.append(float(m))
        sys.masses = np.asarray(masses, dtype=float)

        sys.bonds = []
        if bonds:
            for i, j in bonds:
                if not (0 <= i < n and 0 <= j < n):
                    raise ValueError(f"bond index out of range: {i}-{j}")
                sys.bonds.append(Bond(i, j))
        return sys
