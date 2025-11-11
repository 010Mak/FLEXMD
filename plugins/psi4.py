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

_USE_MENDELEEV = False
try:
    from mendeleev import element as _md_element
    _USE_MENDELEEV = True
except Exception:
    _USE_MENDELEEV = False

_SYMBOL_TO_Z: dict[str, int] = {
    "H": 1,  "He": 2,
    "Li": 3, "Be": 4, "B": 5,  "C": 6,  "N": 7,  "O": 8,  "F": 9,  "Ne": 10,
    "Na": 11,"Mg": 12,"Al": 13,"Si": 14,"P": 15, "S": 16,"Cl": 17,"Ar": 18,
    "K": 19, "Ca": 20,"Sc": 21,"Ti": 22,"V": 23, "Cr": 24,"Mn": 25,"Fe": 26,
    "Co": 27,"Ni": 28,"Cu": 29,"Zn": 30,"Ga": 31,"Ge": 32,"As": 33,"Se": 34,
    "Br": 35,"Kr": 36,
}

def _canon_symbol(sym: str) -> str:
    s = str(sym).strip()
    if not s:
        raise ValueError("empty element symbol")
    return s[0].upper() + s[1:].lower()

def _atomic_number(symbol: str) -> int:
    sym = _canon_symbol(symbol)
    if _USE_MENDELEEV:
        return int(_md_element(sym).atomic_number)
    if sym not in _SYMBOL_TO_Z:
        raise ValueError(f"unsupported element '{symbol}' (install 'mendeleev' for full coverage)")
    return _SYMBOL_TO_Z[sym]

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
        self.mult = max(1, int(multiplicity))

        if memory:
            set_memory(memory)

        base_opts: dict[str, Any] = {"basis": self.basis}
        if scf_type:
            base_opts["scf_type"] = scf_type
        if freeze_core:
            base_opts["freeze_core"] = True
        set_options(base_opts)

        psi4.core.set_output_file(os.devnull, False)
        _log.info(
            "initialized psi4 method=%s basis=%s chg=%s mult=%d",
            self.method, self.basis, str(self.user_charge), self.mult
        )


    def _net_charge_from_system(self, system: System) -> int:
        tot = 0.0
        for a in system.atoms:
            props = a.properties or {}
            if "formal_charge" in props:
                tot += float(props.get("formal_charge", 0.0))
            elif "charge" in props:
                tot += float(props.get("charge", 0.0))
        return int(round(tot))

    def _electron_count(self, system: System, total_charge: int) -> int:
        Zsum = 0
        for a in system.atoms:
            Zsum += int(_atomic_number(str(a.element)))
        return Zsum - int(total_charge)

    def _choose_consistent_multiplicity(self, nelec: int, requested_mult: int) -> int:
        if (nelec + requested_mult) % 2 == 1:
            return requested_mult
        suggested = 2 if (nelec % 2 == 1) else 1
        _log.warning(
            "Psi4Plugin: overriding multiplicity %d -> %d to match %d electrons",
            requested_mult, suggested, nelec
        )
        return suggested

    def _reference_for_method(self, method: str, multiplicity: int) -> str:
        m = method.lower()
        open_shell = (multiplicity != 1)

        post_hf = {
            "mp2", "lmp2", "df-mp2", "mp2p5", "mp3",
            "ccd", "ccsd", "ccsd(t)", "ccsdt", "omp2", "omp3"
        }

        if m in {"hf", "scf"} or m.endswith("-hf"):
            return "uhf" if open_shell else "rhf"
        if m in post_hf:
            return "uhf" if open_shell else "rhf"

        return "uks" if open_shell else "rks"

    def _apply_scf_options(self, reference: str) -> None:
        set_options({
            "reference": reference,
            "guess": "sad",
            "e_convergence": 1e-8,
            "d_convergence": 1e-8,
            "ints_tolerance": 1e-12,
            "scf_type": "df",
        })

    def _mol_from_system(self, system: System):
        total_charge = (
            int(self.user_charge) if self.user_charge is not None else self._net_charge_from_system(system)
        )

        nelec = self._electron_count(system, total_charge)
        mult = self._choose_consistent_multiplicity(nelec, self.mult)

        lines: list[str] = [f"{total_charge} {mult}", "units angstrom"]
        for atom in system.atoms:
            x, y, z = atom.position.tolist()
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                raise ValueError(f"invalid coords for {atom.element}: {atom.position}")
            lines.append(f"{atom.element} {x:.12f} {y:.12f} {z:.12f}")

        geom = "\n".join(lines) + "\n"
        _log.info(
            "psi4 geometry: using body (no molecule{} wrapper); "
            "spin resolve: nelec=%d charge=%d user_mult=%d -> mult=%d",
            nelec, total_charge, self.mult, mult
        )
        try:
            return psi_geometry(geom), total_charge, mult
        except Exception as e:
            _log.error("psi4 geometry parse failed:\n%s", geom)
            raise ValueError(f"psi4 geometry parse error: {e}") from None


    def initialize(self, system: System) -> None:
        pass

    def compute_forces(self, system: System) -> np.ndarray:
        mol, chg, mult = self._mol_from_system(system)
        ref = self._reference_for_method(self.method, mult)
        self._apply_scf_options(ref)
        _log.info("psi4 reference=%s (forces)", ref)

        try:
            grad = psi_gradient(self.method, molecule=mol)
        except Exception as e:
            _log.error("psi4 gradient failed: %s", e)
            raise RuntimeError("psi4 force calculation error") from e

        g = np.array(grad.to_array()).reshape((len(system.atoms), 3))
        return -g * (_HARTREE2KCAL / _BOHR2ANG)

    def compute_energy(self, system: System) -> float:
        mol, chg, mult = self._mol_from_system(system)
        ref = self._reference_for_method(self.method, mult)
        self._apply_scf_options(ref)
        _log.info("psi4 reference=%s (energy)", ref)

        try:
            e_h = psi_energy(self.method, molecule=mol)
        except Exception as e:
            _log.error("psi4 energy failed: %s", e)
            raise RuntimeError("psi4 energy calculation error") from e
        return float(e_h) * _HARTREE2KCAL
