from __future__ import annotations

import logging
from typing import List, Tuple, Any

import numpy as np
import openmm
import openmm.unit as unit
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.toolkit.topology import Molecule as OFFMolecule
from openff.toolkit.utils.exceptions import ChargeMethodUnavailableError

from simulation.plugin_interface import ForceCalculator
from simulation.system import System
from utilities.radii import covalent_radius

_KJNM_TO_KCALA = 2.39005736
_KJ_TO_KCAL    = 0.239005736

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
        timestep_ps: float = 0.001,
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

        self.system_omm: openmm.System | None = None
        self.context: openmm.Context | None = None

        self.charge_method = (partial_charge_method or "").lower() if partial_charge_method else None
        self.user_charge = None if charge is None else int(charge)
        self.fallback_connectivity = bool(fallback_connectivity)
        self.distance_scale = float(distance_scale)

        self._rd_bonds: list[tuple[int,int]] = []

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

    def _rdkit_from_system(self, system: System) -> Tuple[Any, str, int, int]:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
        from rdkit.Geometry import Point3D

        symbols = [a.element for a in system.atoms]
        coords_ang = np.asarray([a.position for a in system.atoms], dtype=float)

        rw = Chem.RWMol()
        for s in symbols:
            rw.AddAtom(Chem.Atom(s.capitalize()))
        mol = rw.GetMol()

        conf = Chem.Conformer(len(symbols))
        for i, pos in enumerate(coords_ang):
            x, y, z = map(float, pos)
            conf.SetAtomPosition(i, Point3D(x, y, z))
        mol.AddConformer(conf, assignId=True)

        if system.bonds:
            rw = Chem.RWMol(mol)
            for b in system.bonds:
                i = int(b.atom1)
                j = int(b.atom2)
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
            mode = "explicit"
            n_bonds = mol.GetNumBonds()
            netq = self._net_charge_from_system(system) if self.user_charge is None else self.user_charge
            _log.info("smirnoff connectivity: picked=%s bonds=%d charge=%s", mode, n_bonds, netq)
            return mol, mode, n_bonds, netq

        netq = self._net_charge_from_system(system) if self.user_charge is None else self.user_charge
        try:
            rdDetermineBonds.DetermineBonds(mol, charge=int(netq))
            Chem.SanitizeMol(mol)
            mode = "determine_bonds"
            n_bonds = mol.GetNumBonds()
            _log.info("smirnoff connectivity: picked=%s bonds=%d charge=%s", mode, n_bonds, netq)
            return mol, mode, n_bonds, netq
        except Exception:
            pass

        if self.fallback_connectivity:
            mol = self._distance_connectivity(len(symbols), coords_ang, symbols)
            mode = "distance"
            n_bonds = mol.GetNumBonds()
            _log.info("smirnoff connectivity: picked=%s bonds=%d charge=%s", mode, n_bonds, netq)
            return mol, mode, n_bonds, netq

        _log.warning("smirnoff connectivity: no bonds could be determined; proceeding atoms-only")
        return mol, "none", 0, netq

    @staticmethod
    def _get_harmonic_bond_force(system: openmm.System) -> openmm.HarmonicBondForce | None:
        for f in system.getForces():
            if isinstance(f, openmm.HarmonicBondForce):
                return f
        return None

    @staticmethod
    def _remove_all_constraints(system: openmm.System) -> int:
        n0 = system.getNumConstraints()
        for idx in reversed(range(n0)):
            system.removeConstraint(idx)
        return n0

    def _inject_harmonic_bonds_for_rdkit(
        self, system: openmm.System, rdmol, coords_ang: np.ndarray
    ) -> int:
        hb = self._get_harmonic_bond_force(system)
        if hb is None:
            hb = openmm.HarmonicBondForce()
            system.addForce(hb)

        existing_pairs = set()
        for n in range(hb.getNumBonds()):
            i, j, _, _ = hb.getBondParameters(n)
            existing_pairs.add((min(int(i), int(j)), max(int(i), int(j))))

        coords_nm = coords_ang * 0.1
        added = 0
        rd_pairs: list[tuple[int,int]] = []
        for b in rdmol.GetBonds():
            ai = b.GetBeginAtom()
            aj = b.GetEndAtom()
            i = int(ai.GetIdx()); j = int(aj.GetIdx())
            rd_pairs.append((i, j))
            key = (min(i, j), max(i, j))
            if key in existing_pairs:
                continue

            dij_nm = float(np.linalg.norm(coords_nm[i] - coords_nm[j]))
            r0_nm = dij_nm if (np.isfinite(dij_nm) and dij_nm >= 1e-6) else 0.101

            si = ai.GetSymbol(); sj = aj.GetSymbol()
            kcal_per_A2 = 300.0 if ("H" in (si, sj)) else 200.0
            k_val = (kcal_per_A2 * unit.kilocalories_per_mole / (unit.angstrom ** 2)
                    ).value_in_unit(unit.kilojoule_per_mole / (unit.nanometer ** 2))

            hb.addBond(i, j, r0_nm, k_val)
            added += 1

        _log.info(
            "Injected %d harmonic bonds (total=%d); constraints now=%d",
            added, hb.getNumBonds(), system.getNumConstraints()
        )
        self._rd_bonds = rd_pairs
        return added

    def initialize(self, system: System) -> None:
        rdmol, mode, n_bonds_rd, netq = self._rdkit_from_system(system)

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
            except (ChargeMethodUnavailableError, Exception) as e:
                _log.warning(
                    "charge assignment via %s failed: %s; proceeding without preassigned charges",
                    self.charge_method, e
                )

        topo = offmol.to_topology()

        tried = []
        sys_omm: openmm.System | None = None
        if charges_set:
            tried.append({"charge_from_molecules": [offmol]})
            tried.append({"charge_from_molecule": offmol})
        tried.append({})

        last_err: Exception | None = None
        for kw in tried:
            try:
                sys_omm = self.ff.create_openmm_system(topo, **kw)
                break
            except Exception as e:
                last_err = e
                continue

        if sys_omm is None:
            raise RuntimeError(f"failed to create OpenMM system from OpenFF: {last_err}")

        removed = self._remove_all_constraints(sys_omm)
        if removed > 0:
            _log.info("Removed %d OpenMM constraints to match external integration.", removed)

        self._inject_harmonic_bonds_for_rdkit(sys_omm, rdmol, coords_ang)

        _log.info(
            "OpenMM system ready: Harmonic bonds=%s, constraints=%s",
            next((f.getNumBonds() for f in sys_omm.getForces() if isinstance(f, openmm.HarmonicBondForce)), 0),
            sys_omm.getNumConstraints()
        )

        self.system_omm = sys_omm
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

    def render_hints(self) -> dict[str, Any]:
        return {"bonds": [[int(i), int(j)] for (i, j) in self._rd_bonds]}
