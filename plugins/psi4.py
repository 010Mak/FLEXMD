
from __future__ import annotations
import os
import logging
from typing import Any, Optional
import numpy as np
import psi4
from psi4 import energy as psi_energy, gradient as psi_gradient, geometry as psi_geometry, set_memory, set_options
from simulation.plugin_interface import ForceCalculator
from simulation.system import System

_HARTREE2KJ = 2625.499638
_KJ2KCAL = 1.0 / 4.184
_HARTREE2KCAL = _HARTREE2KJ * _KJ2KCAL
_BOHR2ANG = 0.529177210903

_log = logging.getLogger("Psi4Plugin")
if not _log.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

class Psi4Plugin(ForceCalculator):
    NAME = "psi4"
    CAPABILITY = "psi4"
    MIN_ATOMS = 1
    MAX_ATOMS = 20

    @classmethod
    def is_available(cls) -> bool:
        try:
            import psi4
            return True
        except Exception:
            return False

    def __init__(
        self,
        method: str = "hf",
        basis: str = "sto-3g",
        scf_type: str | None = None,
        freeze_core: bool = False,
        memory: str | None = None,
        charge: Optional[int] = None,
        multiplicity: int = 1,
    ):
        self.method = method.lower()
        self.basis = basis.lower()
        self.user_charge = charge
        self.mult = int(multiplicity)

        if memory:
            set_memory(memory)
        opts: dict[str, Any] = {"basis": self.basis}
        if scf_type:
            opts["scf_type"] = scf_type
        if freeze_core:
            opts["freeze_core"] = True
        set_options(opts)

        psi4.core.set_output_file(os.devnull, False)
        _log.info("initialized psi4 method=%s basis=%s chg=%s mult=%d", self.method, self.basis, str(self.user_charge), self.mult)

    def _net_charge_from_system(self, system: System) -> int:
        tot = 0.0
        for a in system.atoms:
            props = a.properties or {}
            if "formal_charge" in props:
                tot += float(props.get("formal_charge", 0.0))
            elif "charge" in props:
                tot += float(props.get("charge", 0.0))
        return int(round(tot))

    def _mol_from_system(self, system: System):
        lines: list[str] = [f"{self.user_charge if self.user_charge is not None else self._net_charge_from_system(system)} {self.mult}"]
        for atom in system.atoms:
            x, y, z = atom.position.tolist()
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                raise ValueError(f"invalid coords for {atom.element}: {atom.position}")
            lines.append(f"{atom.element} {x:.12f} {y:.12f} {z:.12f}")
        lines.append("units angstrom")
        geom = "\n".join(lines)
        try:
            return psi_geometry(geom)
        except Exception as e:
            _log.error("psi4 geometry parse failed:\n%s", "molecule {\n" + geom + "\n}")
            raise ValueError(f"psi4 geometry parse error: {e}") from None

    def initialize(self, system: System) -> None:
        pass

    def compute_forces(self, system: System) -> np.ndarray:
        mol = self._mol_from_system(system)
        try:
            grad = psi_gradient(self.method, molecule=mol)
        except Exception as e:
            _log.error("psi4 gradient failed: %s", e)
            raise RuntimeError("psi4 force calculation error") from e
        g = np.array(grad.to_array()).reshape((len(system.atoms), 3))
        return -g * (_HARTREE2KCAL / _BOHR2ANG)

    def compute_energy(self, system: System) -> float:
        mol = self._mol_from_system(system)
        try:
            e_h = psi_energy(self.method, molecule=mol)
        except Exception as e:
            _log.error("psi4 energy failed: %s", e)
            raise RuntimeError("psi4 energy calculation error") from e
        return float(e_h) * _HARTREE2KCAL
