from __future__ import annotations
from typing import Type
from simulation.plugin_interface import ForceCalculator
from simulation.plugin_manager import PLUGINS

_DEFAULT_CAP_ORDER = ["classical", "reaxff", "psi4", "custom"]

def decide(
    system,
    *,
    user_choice: str = "auto",
    threshold: int = 10,
) -> str:
    choice = user_choice.lower()
    n_atoms = len(system.atoms)

    candidates = {
        name: cls
        for name, cls in PLUGINS.items()
        if cls.is_available() and cls.MIN_ATOMS <= n_atoms <= cls.MAX_ATOMS
    }

    if choice != "auto":
        if choice not in candidates:
            raise RuntimeError(f"plugin '{choice}' not available for {n_atoms} atoms")
        return choice

    if n_atoms == 0:
        raise RuntimeError("no atoms to simulate")

    if n_atoms < 2:
        for cap in ("classical", "reaxff"):
            for name, cls in candidates.items():
                if cls.CAPABILITY == cap:
                    return name
        raise RuntimeError("no non-psi4 plugin available for single-atom job")

    if n_atoms <= threshold:
        for name, cls in candidates.items():
            if cls.CAPABILITY == "psi4":
                return name

    for cap in _DEFAULT_CAP_ORDER:
        for name, cls in candidates.items():
            if cls.CAPABILITY == cap:
                return name

    raise RuntimeError(f"no simulation plugin available for {n_atoms} atoms")
