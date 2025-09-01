from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from simulation.plugin_interface import ForceCalculator
from simulation.system import System

try:
    # LAMMPS Python module
    from lammps import lammps
except Exception as _e:  # pragma: no cover
    lammps = None  # type: ignore

_log = logging.getLogger(__name__)


# ---- simple periodic table for masses (g/mol) ----
_ATOMIC_MASS = {
    "H": 1.008, "He": 4.002602, "Li": 6.94, "Be": 9.0121831, "B": 10.81, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998403163, "Ne": 20.1797,
    "Na": 22.98976928, "Mg": 24.305, "Al": 26.9815385, "Si": 28.085, "P": 30.973761998,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.0983, "Ca": 40.078,
    "Fe": 55.845, "Ni": 58.6934, "Cu": 63.546, "Zn": 65.38, "Br": 79.904, "I": 126.90447,
}

# Small safety margins for box sizing (Å)
_BOX_PAD_DEFAULT = 12.0
_BOX_MIN_EXTENT = 30.0  # never let a dimension collapse below this (Å)


class ReaxFFPlugin(ForceCalculator):
    """
    LAMMPS ReaxFF force/energy provider.

    - Units: 'real' (Å, kcal/mol, fs). Returned forces: kcal/mol/Å
    - Robust to changing atom counts between calls (auto-rebuild).
    - Sanitizes non-finite positions using last-good or clamped values.
    - Optional auto-resize of the simulation box to current coordinates.
    """

    NAME = "reaxff"
    CAPABILITY = "reactive"
    MIN_ATOMS = 1
    MAX_ATOMS = int(1e8)

    @classmethod
    def is_available(cls) -> bool:
        if lammps is None:
            return False
        try:
            lm = lammps()
            ok = lm.has_style("pair", "reaxff") or lm.has_style("pair", "reax/c")
            ok = ok and (
                lm.has_style("fix", "qeq/shielded")
                or lm.has_style("fix", "qeq/reaxff")
                or lm.has_style("fix", "acks2/reaxff")
            )
            lm.close()
            return bool(ok)
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        timestep_ps: float = 0.00025,
        qeq_fix: Optional[str] = None,  # "shielded" | "qeq/reaxff" | "acks2/reaxff"
        potential_file: Optional[str] = None,
        auto_resize_box: bool = False,
        box_pad_A: float = _BOX_PAD_DEFAULT,
        **kwargs: Any,
    ):
        # config
        self.dt_ps = float(timestep_ps)  # LAMMPS "real" expects fs for timestep command, but we run 0 steps for forces
        self.qeq_fix_requested = qeq_fix  # take as-is if provided; else auto-pick
        self.potential_file = potential_file or os.path.join(
            os.getcwd(), "potentials", "ffield_FeOC_H_2015.ff"
        )
        self.auto_resize_box = bool(auto_resize_box)
        self.box_pad_A = float(box_pad_A)

        # state
        self._lmp: Optional[lammps] = None
        self._pair_style: Optional[str] = None  # "reaxff" or "reax/c"
        self._elem_types: List[str] = []        # ordered unique elements
        self._type_index: Dict[str, int] = {}   # element -> LAMMPS type index (1-based)
        self._natoms_ctx: int = 0               # atoms in current LAMMPS instance
        self._last_pos: Optional[np.ndarray] = None  # last-good (N,3) Å
        self._last_energy_kcal: Optional[float] = None

    def set_timestep_ps(self, dt_ps: float) -> None:
        self.dt_ps = float(dt_ps)
        if self._lmp is not None:
            # We generally "run 0", but set anyway for completeness
            self._lmp.command(f"timestep {self.dt_ps * 1000.0:.6f}")

    def initialize(self, system: System) -> None:
        """
        Build / rebuild the entire LAMMPS simulation for the given System.
        This is called on first use and whenever atom count changes.
        """
        if lammps is None:
            raise RuntimeError("LAMMPS Python module not available")

        # Destroy any old instance
        if self._lmp is not None:
            try:
                self._lmp.close()
            except Exception:
                pass
            self._lmp = None

        # Prepare element typing
        symbols = [a.element.capitalize() for a in system.atoms]
        uniq: List[str] = []
        seen: set[str] = set()
        for s in symbols:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        if not uniq:
            raise ValueError("reaxff: system has no atoms")

        self._elem_types = uniq
        self._type_index = {e: i + 1 for i, e in enumerate(self._elem_types)}
        self._natoms_ctx = len(symbols)

        # Choose pair style
        lm = lammps()
        self._pair_style = "reaxff" if lm.has_style("pair", "reaxff") else "reax/c"
        if not (lm.has_style("pair", "reaxff") or lm.has_style("pair", "reax/c")):
            lm.close()
            raise RuntimeError("LAMMPS missing reaxff or reax/c pair style")
        self._lmp = lm

        # ---- LAMMPS input deck (minimal) ----
        lm.command("units real")
        lm.command("atom_style charge")
        lm.command("boundary f f f")
        lm.command("neigh_modify delay 0 every 1 check yes")
        lm.command("neighbor 2.0 bin")
        lm.command(f"timestep {self.dt_ps * 1000.0:.6f}")  # fs

        # Create box sized to current positions
        pos = np.asarray(system.positions, dtype=np.float64, order="C")
        pos = self._sanitize_positions(system, pos)
        xlo, xhi, ylo, yhi, zlo, zhi = self._compute_box(pos)
        lm.command(
            f"region simbox block {xlo:.6f} {xhi:.6f} {ylo:.6f} {yhi:.6f} {zlo:.6f} {zhi:.6f} units box"
        )
        lm.command(f"create_box {len(self._elem_types)} simbox")

        # Masses (not strictly required for run 0, but harmless and tidy)
        for i, elem in enumerate(self._elem_types, start=1):
            mass = _ATOMIC_MASS.get(elem, 12.0)
            lm.command(f"mass {i} {mass:.6f}")

        # Create atoms at current coordinates with proper type
        for idx, (elem, p) in enumerate(zip(symbols, pos), start=1):
            t = self._type_index[elem]
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            lm.command(f"create_atoms {t} single {x:.8f} {y:.8f} {z:.8f} units box")

        # Zero initial charges (QEq will adjust)
        lm.command("set type * charge 0.0")

        # Pair style + coeffs
        if self._pair_style == "reax/c":
            lm.command("pair_style reax/c NULL")
        else:
            lm.command("pair_style reaxff")
        mapping = " ".join(self._elem_types)
        lm.command(f"pair_coeff * * {self.potential_file} {mapping}")

        # QEq choice
        qeq_fix_cmd = self._pick_qeq_fix(lm)
        if qeq_fix_cmd:
            lm.command(qeq_fix_cmd)

        # Thermo so we can read PE/press if desired
        lm.command("thermo 1")
        lm.command("thermo_style custom step temp pe press")

        self._last_pos = pos.copy()
        self._last_energy_kcal = None

        _log.info(
            "reaxff dt_ps=%.6f (fs=%.3f) ready: %d atoms, pair=%s, ff=%s",
            self.dt_ps, self.dt_ps * 1000.0, self._natoms_ctx, self._pair_style, self.potential_file
        )

    # ------------------------------------------------------------------ #
    # Core compute API
    # ------------------------------------------------------------------ #
    def compute_forces(self, system: System) -> np.ndarray:
        """
        Return forces (N,3) in kcal/mol/Å for the given configuration.
        """
        if self._lmp is None or self._natoms_ctx != len(system.atoms):
            self.initialize(system)

        lm = self._lmp
        assert lm is not None

        # Sanitize positions and keep last-good snapshot
        pos = np.asarray(system.positions, dtype=np.float64, order="C")
        pos = self._sanitize_positions(system, pos)

        # Resize box if needed
        if self.auto_resize_box:
            self._resize_box_if_needed(lm, pos)

        # Scatter positions to LAMMPS
        self._scatter_positions(lm, pos)

        # Force evaluation; run 0 steps is sufficient
        lm.command("run 0")

        # Gather forces
        forces = self._gather_forces(lm, n=len(system.atoms))

        # Cache last-good for next recovery
        self._last_pos = pos.copy()

        return forces

    def compute_energy(self, system: System) -> float:
        """
        Return potential energy (kcal/mol) for the current configuration.
        Side-effect: will perform a 'run 0' if necessary to ensure thermo is up-to-date.
        """
        if self._lmp is None or self._natoms_ctx != len(system.atoms):
            self.initialize(system)

        lm = self._lmp
        assert lm is not None

        # Ensure positions are in sync and finite (match compute_forces path)
        pos = np.asarray(system.positions, dtype=np.float64, order="C")
        pos = self._sanitize_positions(system, pos)
        if self.auto_resize_box:
            self._resize_box_if_needed(lm, pos)
        self._scatter_positions(lm, pos)

        # Compute
        lm.command("run 0")
        try:
            pe = float(lm.get_thermo("pe"))
        except Exception:
            # Some LAMMPS builds expose PE via compute
            pe = float(lm.extract_compute("thermo_pe", 0, 0))  # type: ignore
        self._last_energy_kcal = pe
        self._last_pos = pos.copy()
        return pe

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _pick_qeq_fix(self, lm: lammps) -> Optional[str]:
        """
        Choose and build a QEq fix command compatible with the active pair style.
        Priority: user request -> available styles.
        """
        # Requested explicitly?
        requested = (self.qeq_fix_requested or "").strip().lower()

        # Determine compatible candidates
        pair_tag = "reax/c" if self._pair_style == "reax/c" else "reaxff"

        # Map of friendly names to full fix commands
        candidates: List[Tuple[str, str]] = [
            ("qeq/shielded", f"fix reaxQEq all qeq/shielded 1 0.0 10.0 1e-6 {pair_tag}"),
            ("qeq/reaxff", "fix reaxQEq all qeq/reaxff 1 0.0 10.0 1e-6"),
            ("acks2/reaxff", "fix reaxQEq all acks2/reaxff 1 0.0 10.0 1e-6"),
        ]

        # If user requested one, try to honor it first (provided the style exists)
        if requested:
            for name, cmd in candidates:
                if requested == name and lm.has_style("fix", name.split()[0]):
                    return cmd

        # Otherwise auto-pick the first available one
        for name, cmd in candidates:
            if lm.has_style("fix", name.split()[0]):
                return cmd

        _log.warning("reaxff: no QEq fix styles available; continuing without explicit QEq fix")
        return None

    def _sanitize_positions(self, system: System, pos: np.ndarray) -> np.ndarray:
        """
        Ensure finite (N,3) Å positions. If non-finite values are present,
        prefer last-good; else clamp NaN/Inf to conservative finite values.
        """
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"positions must be (N,3); got {pos.shape}")

        if np.all(np.isfinite(pos)):
            return pos

        _log.warning("reaxff: non-finite positions detected; recovering")
        if self._last_pos is not None and self._last_pos.shape == pos.shape and np.all(np.isfinite(self._last_pos)):
            clean = self._last_pos.copy()
        else:
            # Clamp to modest extents to avoid exploding the neighbor bins
            clean = np.nan_to_num(pos, nan=0.0, posinf=1e3, neginf=-1e3)

        # propagate back so callers see finite coordinates
        system.positions = clean.tolist()
        return clean

    def _compute_box(self, pos: np.ndarray) -> Tuple[float, float, float, float, float, float]:
        mins = np.nanmin(pos, axis=0)
        maxs = np.nanmax(pos, axis=0)
        pad = float(self.box_pad_A)
        xlo = float(mins[0] - pad)
        xhi = float(maxs[0] + pad)
        ylo = float(mins[1] - pad)
        yhi = float(maxs[1] + pad)
        zlo = float(mins[2] - pad)
        zhi = float(maxs[2] + pad)

        # enforce minimum extents to avoid degenerate boxes
        if xhi - xlo < _BOX_MIN_EXTENT:
            mid = 0.5 * (xlo + xhi)
            xlo, xhi = mid - 0.5 * _BOX_MIN_EXTENT, mid + 0.5 * _BOX_MIN_EXTENT
        if yhi - ylo < _BOX_MIN_EXTENT:
            mid = 0.5 * (ylo + yhi)
            ylo, yhi = mid - 0.5 * _BOX_MIN_EXTENT, mid + 0.5 * _BOX_MIN_EXTENT
        if zhi - zlo < _BOX_MIN_EXTENT:
            mid = 0.5 * (zlo + zhi)
            zlo, zhi = mid - 0.5 * _BOX_MIN_EXTENT, mid + 0.5 * _BOX_MIN_EXTENT

        return xlo, xhi, ylo, yhi, zlo, zhi

    def _resize_box_if_needed(self, lm: lammps, pos: np.ndarray) -> None:
        xlo, xhi, ylo, yhi, zlo, zhi = self._compute_box(pos)
        # change_box with 'units box' and explicit final edges; remap no (we're setting coords after anyway)
        lm.command(
            f"change_box all x final {xlo:.6f} {xhi:.6f} y final {ylo:.6f} {yhi:.6f} "
            f"z final {zlo:.6f} {zhi:.6f} units box"
        )

    def _scatter_positions(self, lm: lammps, pos: np.ndarray) -> None:
        """
        Push (N,3) positions to LAMMPS.
        Prefer scatter_atoms if available; fall back to per-atom set.
        """
        n = pos.shape[0]
        try:
            # Newer LAMMPS python interfaces accept a flattened array
            lm.scatter_atoms("x", 1, 3, pos.astype(float).ravel(order="C"))
        except Exception:
            # Slow but robust fallback
            for i in range(n):
                x, y, z = float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])
                lm.command(f"set atom {i+1} x {x:.8f} y {y:.8f} z {z:.8f}")

    def _gather_forces(self, lm: lammps, n: int) -> np.ndarray:
        """
        Collect forces from LAMMPS as (N,3) in kcal/mol/Å (units real).
        """
        try:
            raw = lm.gather_atoms("f", 1, 3)
            f = np.array(raw, dtype=float).reshape(n, 3)
            if not np.all(np.isfinite(f)):
                _log.warning("reaxff: non-finite forces; zeroing problematic entries")
                f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
            return f
        except Exception:
            # Fallback via extract_atom pointer-of-pointers
            fa = lm.extract_atom("f", 3)
            f = np.zeros((n, 3), dtype=float)
            for i in range(n):
                f[i, 0] = float(fa[i][0])
                f[i, 1] = float(fa[i][1])
                f[i, 2] = float(fa[i][2])
            if not np.all(np.isfinite(f)):
                _log.warning("reaxff: non-finite forces via extract_atom; clamping")
                f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
            return f
