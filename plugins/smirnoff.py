
from __future__ import annotations
import logging
from typing import Any, List
import numpy as np
import openmm
import openmm.unit as unit
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.toolkit.topology import Molecule as OFFMolecule
from simulation.plugin_interface import ForceCalculator
from simulation.system import System
from utilities.radii import covalent_radius
from openff.toolkit.utils.exceptions import ChargeMethodUnavailableError

_KJNM_TO_KCALA = 0.0239005736
_KJ_TO_KCAL = 0.239005736

_log = logging.getLogger(__name__)

class SMIRNOFFPlugin(ForceCalculator):
    NAME = "smirnoff"
    CAPABILITY = "classical"
    MIN_ATOMS = 0
    MAX_ATOMS = int(1e12)

    @classmethod
    def is_available(cls) -> bool:
        try:
            import openmm
            import openff.toolkit
            import rdkit
            return True
        except Exception:
            return False

    def __init__(
        self,
        ff_xml: str = "openff-2.0.0.offxml",
        timestep_ps: float = 0.002,
        partial_charge_method: str | None = None,
        charge: int | None = None,
        fallback_connectivity: bool = True,
        distance_scale: float = 1.2,
    ):
        try:
            self.ff = ForceField(ff_xml)
        except Exception as e:
            raise RuntimeError(f"could not load smirnoff xml '{ff_xml}': {e}")
        self.dt = float(timestep_ps) * unit.picoseconds
        self.integrator = openmm.VerletIntegrator(self.dt)
        self.system_omm = None
        self.context = None
        self.charge_method = (partial_charge_method or "").lower() if partial_charge_method else None
        self.user_charge = None if charge is None else int(charge)
        self.fallback_connectivity = bool(fallback_connectivity)
        self.distance_scale = float(distance_scale)

    def set_timestep_ps(self, dt_ps: float) -> None:
        self.dt = float(dt_ps) * unit.picoseconds
        self.integrator = openmm.VerletIntegrator(self.dt)
        if self.system_omm is not None:
            self.context = openmm.Context(self.system_omm, self.integrator)

    def _net_charge_from_system(self, system: System) -> int:
        tot = 0.0
        for a in system.atoms:
            props = a.properties or {}
            if "formal_charge" in props:
                tot += float(props.get("formal_charge", 0.0))
            elif "charge" in props:
                tot += float(props.get("charge", 0.0))
        return int(round(tot))

    def _distance_connectivity(self, natoms: int, coords: np.ndarray, symbols: List[str]):
        from rdkit import Chem
        from rdkit.Geometry import Point3D

        rw = Chem.RWMol()
        for s in symbols:
            rw.AddAtom(Chem.Atom(s.capitalize()))
        mol = rw.GetMol()
        conf = Chem.Conformer(natoms)
        for i in range(natoms):
            x, y, z = map(float, coords[i])
            conf.SetAtomPosition(i, Point3D(x, y, z))
        mol.AddConformer(conf, assignId=True)

        rw = Chem.RWMol(mol)
        for i in range(natoms):
            ri = covalent_radius(symbols[i])
            for j in range(i + 1, natoms):
                rj = covalent_radius(symbols[j])
                cutoff = self.distance_scale * (ri + rj)
                if float(np.linalg.norm(coords[i] - coords[j])) <= cutoff:
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

    def _rdkit_from_system(self, system: System):
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
        from rdkit.Geometry import Point3D

        symbols = [a.element for a in system.atoms]
        rw = Chem.RWMol()
        for s in symbols:
            rw.AddAtom(Chem.Atom(s.capitalize()))
        mol = rw.GetMol()

        conf = Chem.Conformer(len(system.atoms))
        for i, pos in enumerate(system.positions):
            x, y, z = map(float, pos)
            conf.SetAtomPosition(i, Point3D(x, y, z))
        mol.AddConformer(conf, assignId=True)

        if system.bonds:
            rw = Chem.RWMol(mol)
            for b in system.bonds:
                if rw.GetBondBetweenAtoms(int(b.atom1), int(b.atom2)) is None:
                    rw.AddBond(int(b.atom1), int(b.atom2), Chem.BondType.SINGLE)
            mol = rw.GetMol()
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                try:
                    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS)
                except Exception:
                    pass
            return mol

        netq = self._net_charge_from_system(system) if self.user_charge is None else self.user_charge
        try:
            rdDetermineBonds.DetermineBonds(mol, charge=int(netq))
            Chem.SanitizeMol(mol)
            return mol
        except Exception:
            pass

        coords = np.asarray(system.positions, dtype=float)
        return self._distance_connectivity(len(system.atoms), coords, symbols)

    def initialize(self, system: System) -> None:
        rdmol = self._rdkit_from_system(system)
        offmol = OFFMolecule.from_rdkit(rdmol, allow_undefined_stereo=True)

        coords_ang = np.asarray([atom.position for atom in system.atoms], dtype=float)
        if not offmol.conformers:
            offmol.add_conformer((coords_ang * 0.1) * unit.nanometer)

        charges_set = False
        if self.charge_method:
            try:
                if self.charge_method == "zeros":
                    offmol.assign_partial_charges("zeros")
                else:
                    offmol.assign_partial_charges(self.charge_method, use_conformers=offmol.conformers)
                charges_set = True
                _log.info("smirnoff charges assigned via %s", self.charge_method)
            except Exception as e:
                _log.warning("charge assignment via %s failed: %s; proceeding without preassigned charges", self.charge_method, e)

        topo = offmol.to_topology()
        try:
            if charges_set:
                self.system_omm = self.ff.create_openmm_system(topo, charge_from_molecules=[offmol])
            else:
                self.system_omm = self.ff.create_openmm_system(topo)
        except TypeError:
            if charges_set:
                self.system_omm = self.ff.create_openmm_system(topo, charge_from_molecule=offmol)
            else:
                self.system_omm = self.ff.create_openmm_system(topo)

        self.context = openmm.Context(self.system_omm, self.integrator)

    def compute_forces(self, system: System) -> np.ndarray:
        coords_ang = np.asarray([atom.position for atom in system.atoms], dtype=float)
        self.context.setPositions((coords_ang * 0.1) * unit.nanometer)
        state = self.context.getState(getForces=True)
        f_kj_nm = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )
        return np.asarray(f_kj_nm) * _KJNM_TO_KCALA

    def compute_energy(self, system: System) -> float:
        state = self.context.getState(getEnergy=True)
        e_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        return float(e_kj) * _KJ_TO_KCAL

        
