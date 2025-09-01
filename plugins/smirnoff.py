from __future__ import annotations

import logging
from typing import Any, List, Optional

import numpy as np
import openmm
import openmm.unit as unit
from openff.toolkit.topology import Molecule as OFFMolecule
from openff.toolkit.topology import Topology as OFFTopology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.toolkit.utils.exceptions import ChargeMethodUnavailableError

from simulation.plugin_interface import ForceCalculator
from simulation.system import System
from utilities.radii import covalent_radius

# Unit conversions
_KJNM_TO_KCALA = 0.0239005736     # kJ/mol/nm -> kcal/mol/Å
_KJ_TO_KCAL = 0.239005736         # kJ/mol -> kcal/mol

_log = logging.getLogger(__name__)


class SMIRNOFFPlugin(ForceCalculator):
    """
    OpenFF/SMIRNOFF (OpenMM) force/energy provider.

    Robustness:
    - Rebuilds OpenMM System/Context automatically when the atom count changes
      between calls (avoids "wrong number of positions" exceptions).
    - Optional partial charges (e.g., 'zeros') when requested.
    - RDKit bond perception with distance-based fallback.
    """

    NAME = "smirnoff"
    CAPABILITY = "classical"
    MIN_ATOMS = 0
    MAX_ATOMS = int(1e12)

    @classmethod
    def is_available(cls) -> bool:
        try:
            import openmm  # noqa: F401
            import openff.toolkit  # noqa: F401
            import rdkit  # noqa: F401
            return True
        except Exception:
            return False

    def __init__(
        self,
        ff_xml: str = "openff-2.0.0.offxml",
        timestep_ps: float = 0.002,
        partial_charge_method: Optional[str] = None,
        charge: Optional[int] = None,
        fallback_connectivity: bool = True,
        distance_scale: float = 1.2,
    ):
        try:
            self.ff = ForceField(ff_xml)
        except Exception as e:
            raise RuntimeError(f"could not load smirnoff xml '{ff_xml}': {e}")

        # Integrator / timestep (engine may call set_timestep_ps later)
        self.dt = float(timestep_ps) * unit.picoseconds
        self.integrator = openmm.VerletIntegrator(self.dt)

        # Options
        self.partial_charge_method = partial_charge_method
        self.user_charge = int(charge) if charge is not None else None
        self.fallback_connectivity = bool(fallback_connectivity)
        self.distance_scale = float(distance_scale)

        # State
        self.context: Optional[openmm.Context] = None
        self.system_omm: Optional[openmm.System] = None
        self.off_mol: Optional[OFFMolecule] = None
        self.off_top: Optional[OFFTopology] = None
        self._n_ctx: Optional[int] = None

    # -------------------------
    # Engine hooks
    # -------------------------

    def set_timestep_ps(self, dt_ps: float) -> None:
        self.dt = float(dt_ps) * unit.picoseconds
        self.integrator = openmm.VerletIntegrator(self.dt)
        if self.system_omm is not None:
            # Re-create context with the new integrator
            self.context = openmm.Context(self.system_omm, self.integrator)

    def initialize(self, system: System) -> None:
        """
        Build RDKit -> OFFTopology -> OpenMM System and Context for 'system'.
        """
        # 1) Build RDKit molecule (explicit bonds preferred, then perception)
        rd_mol = self._rdkit_from_system(system)

        # 2) OFF molecule
        try:
            offmol = OFFMolecule.from_rdkit(rd_mol, allow_undefined_stereo=True)
        except Exception as e:
            _log.error("OFFMolecule.from_rdkit failed: %s", e)
            # Fallback: empty container of the right size (rare path)
            from rdkit import Chem
            rw = Chem.RWMol()
            for a in system.atoms:
                rw.AddAtom(Chem.Atom(a.element.capitalize()))
            mol2 = rw.GetMol()
            offmol = OFFMolecule.from_rdkit(mol2, allow_undefined_stereo=True)

        # 3) Assign partial charges if requested
        charges_set = False
        if self.partial_charge_method:
            try:
                offmol.assign_partial_charges(self.partial_charge_method)
                charges_set = True
                _log.info("smirnoff charges assigned via %s", self.partial_charge_method)
            except ChargeMethodUnavailableError:
                _log.warning("charge method '%s' unavailable; falling back to toolkit defaults",
                             self.partial_charge_method)
            except Exception as e:
                _log.warning("partial charge assignment failed: %s", e)

        # 4) OFF topology
        off_top = OFFTopology.from_molecules([offmol])

        # 5) OpenMM System
        try:
            # Newer toolkit may accept 'charge_from_molecule'
            if charges_set:
                self.system_omm = self.ff.create_openmm_system(off_top, charge_from_molecule=offmol)
            else:
                self.system_omm = self.ff.create_openmm_system(off_top)
        except TypeError:
            # Older toolkit compatibility
            if charges_set:
                try:
                    self.system_omm = self.ff.create_openmm_system(off_top, charge_from_molecules=[offmol])
                except Exception:
                    self.system_omm = self.ff.create_openmm_system(off_top)
            else:
                self.system_omm = self.ff.create_openmm_system(off_top)

        # 6) Context
        self.context = openmm.Context(self.system_omm, self.integrator)
        self._n_ctx = int(self.system_omm.getNumParticles())
        self.off_mol = offmol
        self.off_top = off_top

    # -------------------------
    # Core computations
    # -------------------------

    def _ensure_context(self, system: System) -> None:
        """Rebuild the Context if missing or if atom count changed."""
        n_now = len(system.atoms)
        if (self.context is None) or (self.system_omm is None) or (self._n_ctx != n_now):
            self.initialize(system)

    def compute_forces(self, system: System) -> np.ndarray:
        """
        Returns forces in kcal/mol/Å for the current configuration.
        Robust to changes in atom count between calls.
        """
        self._ensure_context(system)

        coords_ang = np.asarray([atom.position for atom in system.atoms], dtype=float)
        try:
            self.context.setPositions((coords_ang * 0.1) * unit.nanometer)
        except Exception:
            # Recover from a stale Context (e.g., if atom count changed mid-stream)
            self.initialize(system)
            self.context.setPositions((coords_ang * 0.1) * unit.nanometer)

        state = self.context.getState(getForces=True)
        f_kj_nm = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )
        return np.asarray(f_kj_nm) * _KJNM_TO_KCALA

    def compute_energy(self, system: System) -> float:
        """Returns potential energy in kcal/mol."""
        self._ensure_context(system)
        state = self.context.getState(getEnergy=True)
        e_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        return float(e_kj) * _KJ_TO_KCAL

    # -------------------------
    # Helpers
    # -------------------------

    def _net_charge_from_system(self, system: System) -> int:
        """Sum formal_charge/charge properties if provided."""
        tot = 0.0
        for a in system.atoms:
            props: dict[str, Any] = a.properties or {}
            if "formal_charge" in props:
                tot += float(props.get("formal_charge", 0.0))
            elif "charge" in props:
                tot += float(props.get("charge", 0.0))
        return int(round(tot))

    def _rdkit_from_system(self, system: System):
        """
        Build an RDKit Mol with coordinates.
        Preference order:
          1) Use explicit bonds from System.bonds if provided.
          2) Use rdDetermineBonds (with net charge) to perceive bonds.
          3) Fallback to simple distance-based bonding.
        """
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
        from rdkit.Geometry import Point3D

        symbols = [a.element for a in system.atoms]

        # fresh molecule
        rw = Chem.RWMol()
        for s in symbols:
            rw.AddAtom(Chem.Atom(s.capitalize()))
        mol = rw.GetMol()

        # add conformer
        conf = Chem.Conformer(len(system.atoms))
        for i, pos in enumerate(system.positions):
            x, y, z = map(float, pos)
            conf.SetAtomPosition(i, Point3D(x, y, z))
        mol.AddConformer(conf, assignId=True)

        # 1) explicit bonds (if present)
        if getattr(system, "bonds", None):
            rw = Chem.RWMol(mol)
            for b in system.bonds:
                i = int(b.atom1); j = int(b.atom2)
                if rw.GetBondBetweenAtoms(i, j) is None:
                    rw.AddBond(i, j, Chem.BondType.SINGLE)
            mol = rw.GetMol()
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                try:
                    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS)
                except Exception:
                    pass
            return mol

        # 2) rdDetermineBonds (chemically smarter)
        netq = self._net_charge_from_system(system) if self.user_charge is None else self.user_charge
        try:
            rdDetermineBonds.DetermineBonds(mol, charge=int(netq))
            Chem.SanitizeMol(mol)
            return mol
        except Exception:
            # fall through to distance-based perception
            pass

        # 3) distance-based fallback
        if self.fallback_connectivity:
            return self._distance_connectivity(len(symbols), np.asarray(system.positions, dtype=float), symbols)
        else:
            return mol

    def _distance_connectivity(self, natoms: int, coords: np.ndarray, symbols: List[str]):
        """Simple covalent-radius based bond addition. Returns an RDKit Mol."""
        from rdkit import Chem
        from rdkit.Geometry import Point3D

        rw = Chem.RWMol()
        for s in symbols:
            rw.AddAtom(Chem.Atom(s.capitalize()))
        mol = rw.GetMol()

        # coordinates
        conf = Chem.Conformer(natoms)
        for i in range(natoms):
            x, y, z = map(float, coords[i])
            conf.SetAtomPosition(i, Point3D(x, y, z))
        mol.AddConformer(conf, assignId=True)

        # add bonds by distance threshold
        def thr(i: int, j: int) -> float:
            ri = covalent_radius(symbols[i])
            rj = covalent_radius(symbols[j])
            return float(self.distance_scale) * float(ri + rj)

        rw = Chem.RWMol(mol)
        for i in range(natoms - 1):
            for j in range(i + 1, natoms):
                # Euclidean distance in Å
                dij = float(np.linalg.norm(coords[i] - coords[j]))
                if dij <= thr(i, j):
                    if rw.GetBondBetweenAtoms(i, j) is None:
                        rw.AddBond(i, j, Chem.BondType.SINGLE)
        out = rw.GetMol()
        try:
            Chem.SanitizeMol(out)
        except Exception:
            try:
                Chem.SanitizeMol(out, sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS)
            except Exception:
                pass
        return out
