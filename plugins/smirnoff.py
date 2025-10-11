from __future__ import annotations

import logging
from typing import List, Tuple, Any

import numpy as np
import openmm
import openmm.unit as unit
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.toolkit.topology import Topology as OFFTopology, Molecule as OFFMolecule

from simulation.plugin_interface import ForceCalculator
from simulation.system import System
from utilities.radii import covalent_radius

_KJ_TO_KCAL = 0.2390057361376673
_NM_PER_A = 0.1
_A_PER_NM = 10.0
_KJNM_TO_KCALA = _KJ_TO_KCAL / _A_PER_NM

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
        forbid_hh_bonds: bool = True,
        prune_after_determine: bool = True,
    ):
        try:
            self.ff = ForceField(ff_xml)
        except Exception as e:
            raise RuntimeError(f"could not load smirnoff xml '{ff_xml}': {e}")

        self.dt_ps: float = float(timestep_ps)
        self.integrator: openmm.Integrator | None = None
        self.context: openmm.Context | None = None
        self.system_omm: openmm.System | None = None

        self.charge_method = (partial_charge_method or "").lower() if partial_charge_method else None
        self.user_charge = None if charge is None else int(charge)
        self.fallback_connectivity = bool(fallback_connectivity)
        self.distance_scale = float(distance_scale)
        self.forbid_hh_bonds = bool(forbid_hh_bonds)
        self.prune_after_determine = bool(prune_after_determine)

        self._rd_bonds: list[tuple[int, int]] = []
        self._last_elements: Tuple[str, ...] | None = None
        self._sys_to_off: np.ndarray | None = None
        self._off_to_sys: np.ndarray | None = None

    def _dispose_context(self):
        try:
            if self.context is not None:
                del self.context
        except Exception:
            pass
        self.context = None
        try:
            if self.integrator is not None:
                del self.integrator
        except Exception:
            pass
        self.integrator = None

    def _create_context(self):
        if self.system_omm is None:
            raise RuntimeError("cannot create OpenMM Context before system is built")
        self._dispose_context()
        self.integrator = openmm.VerletIntegrator(self.dt_ps * unit.picoseconds)
        self.context = openmm.Context(self.system_omm, self.integrator)

    def set_timestep_ps(self, dt_ps: float) -> None:
        self.dt_ps = float(dt_ps)
        if self.system_omm is not None:
            self._create_context()

    def _net_charge_from_system(self, system: System) -> int:
        tot = 0.0
        for a in system.atoms:
            props = a.properties or {}
            if "formal_charge" in props:
                tot += float(props.get("formal_charge", 0.0))
            elif "charge" in props:
                tot += float(props.get("charge", 0.0))
        return int(round(tot))

    def _prune_bonds_by_distance_and_hh(
        self,
        bonds: List[Tuple[int, int]],
        coords_ang: np.ndarray,
        symbols: List[str],
    ) -> List[Tuple[int, int]]:
        keep: List[Tuple[int, int]] = []
        for (i, j) in bonds:
            si = symbols[i].capitalize()
            sj = symbols[j].capitalize()

            if self.forbid_hh_bonds and si == "H" and sj == "H":
                continue

            ri = covalent_radius(si)
            rj = covalent_radius(sj)
            cutoff = self.distance_scale * (ri + rj)
            d_ij = float(np.linalg.norm(coords_ang[i] - coords_ang[j]))
            if d_ij <= cutoff:
                keep.append((i, j))
        return keep

    def _distance_connectivity(
        self, natoms: int, coords_ang: np.ndarray, symbols: List[str], scale_boost: float = 1.0
    ) -> List[Tuple[int, int]]:
        pairs: List[Tuple[int, int]] = []
        for i in range(natoms):
            si = symbols[i].capitalize()
            ri = covalent_radius(si)
            for j in range(i + 1, natoms):
                sj = symbols[j].capitalize()
                if self.forbid_hh_bonds and si == "H" and sj == "H":
                    continue
                rj = covalent_radius(sj)
                cutoff = (self.distance_scale * scale_boost) * (ri + rj)
                if float(np.linalg.norm(coords_ang[i] - coords_ang[j])) <= cutoff:
                    pairs.append((i, j))
        return pairs

    def _determine_bonds(self, system: System) -> tuple[List[Tuple[int, int]], str, int]:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
        from rdkit.Geometry import Point3D

        symbols = [a.element for a in system.atoms]
        coords_ang = np.asarray([a.position for a in system.atoms], dtype=float)
        natoms = len(symbols)

        rw = Chem.RWMol()
        for s in symbols:
            rw.AddAtom(Chem.Atom(s.capitalize()))
        mol = rw.GetMol()

        conf = Chem.Conformer(natoms)
        for i, pos in enumerate(coords_ang):
            x, y, z = map(float, pos)
            conf.SetAtomPosition(i, Point3D(x, y, z))
        mol.AddConformer(conf, assignId=True)

        netq = self._net_charge_from_system(system) if self.user_charge is None else self.user_charge

        if system.bonds:
            pairs = []
            for b in system.bonds:
                i = int(b.atom1)
                j = int(b.atom2)
                if 0 <= i < natoms and 0 <= j < natoms and i != j:
                    pairs.append((min(i, j), max(i, j)))
            pairs = list(sorted(set(pairs)))
            pairs = self._prune_bonds_by_distance_and_hh(pairs, coords_ang, symbols)
            _log.info("smirnoff connectivity: picked=%s bonds=%d charge=%s", "explicit+prune", len(pairs), netq)
            return pairs, "explicit+prune", netq

        try:
            rdDetermineBonds.DetermineBonds(mol, charge=int(netq))
            pairs = []
            for b in mol.GetBonds():
                i = int(b.GetBeginAtomIdx())
                j = int(b.GetEndAtomIdx())
                pairs.append((min(i, j), max(i, j)))
            pairs = list(sorted(set(pairs)))
            if self.prune_after_determine or self.forbid_hh_bonds:
                pairs = self._prune_bonds_by_distance_and_hh(pairs, coords_ang, symbols)
            mode = "determine_bonds+prune" if (self.prune_after_determine or self.forbid_hh_bonds) else "determine_bonds"
            _log.info("smirnoff connectivity: picked=%s bonds=%d charge=%s", mode, len(pairs), netq)
            return pairs, mode, netq
        except Exception:
            pass

        if self.fallback_connectivity:
            pairs = self._distance_connectivity(natoms, coords_ang, symbols, scale_boost=1.0)
            if not pairs and natoms > 1:
                pairs = self._distance_connectivity(natoms, coords_ang, symbols, scale_boost=1.25)
            _log.info("smirnoff connectivity: picked=%s bonds=%d charge=%s", "distance", len(pairs), netq)
            return pairs, "distance", netq

        _log.warning("smirnoff connectivity: no bonds could be determined; proceeding atoms-only")
        return [], "none", netq

    def _split_to_off_molecules_and_order(
        self,
        natoms: int,
        pairs: List[Tuple[int, int]],
        symbols: List[str],
        coords_ang: np.ndarray,
    ) -> tuple[List[OFFMolecule], List[int]]:
        from rdkit import Chem
        from rdkit.Geometry import Point3D

        rw = Chem.RWMol()
        for s in symbols:
            rw.AddAtom(Chem.Atom(s.capitalize()))
        for (i, j) in pairs:
            if i != j and rw.GetBondBetweenAtoms(i, j) is None:
                rw.AddBond(i, j, Chem.BondType.SINGLE)

        for idx in range(natoms):
            a = rw.GetAtomWithIdx(idx)
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(0)

        mol = rw.GetMol()
        mol.UpdatePropertyCache(strict=False)

        conf = Chem.Conformer(natoms)
        for i, pos in enumerate(coords_ang):
            x, y, z = map(float, pos)
            conf.SetAtomPosition(i, Point3D(x, y, z))
        mol.AddConformer(conf, assignId=True)

        idx_frags = Chem.GetMolFrags(mol, asMols=False)
        off_mols: List[OFFMolecule] = []
        off_order: List[int] = []

        for idx_tuple in idx_frags:
            amap = {orig: k for k, orig in enumerate(idx_tuple)}
            sub_rw = Chem.RWMol()
            for orig in idx_tuple:
                sub_rw.AddAtom(Chem.Atom(symbols[orig].capitalize()))
            for k in range(len(idx_tuple)):
                a = sub_rw.GetAtomWithIdx(k)
                a.SetNoImplicit(True)
                a.SetNumExplicitHs(0)
            for (i, j) in pairs:
                if i in amap and j in amap:
                    sub_rw.AddBond(amap[i], amap[j], Chem.BondType.SINGLE)
            sub = sub_rw.GetMol()
            sub.UpdatePropertyCache(strict=False)

            sub_conf = Chem.Conformer(len(idx_tuple))
            for k, orig in enumerate(idx_tuple):
                x, y, z = map(float, coords_ang[orig])
                sub_conf.SetAtomPosition(k, Point3D(x, y, z))
            sub.AddConformer(sub_conf, assignId=True)

            off_mol = OFFMolecule.from_rdkit(sub, allow_undefined_stereo=True)
            off_mols.append(off_mol)
            off_order.extend(list(idx_tuple))

        if not pairs and natoms > 0 and not off_mols:
            for i in range(natoms):
                srw = Chem.RWMol()
                srw.AddAtom(Chem.Atom(symbols[i].capitalize()))
                a = srw.GetAtomWithIdx(0)
                a.SetNoImplicit(True)
                a.SetNumExplicitHs(0)
                sm = srw.GetMol()
                sm.UpdatePropertyCache(strict=False)
                conf1 = Chem.Conformer(1)
                x, y, z = map(float, coords_ang[i])
                conf1.SetAtomPosition(0, Point3D(x, y, z))
                sm.AddConformer(conf1, assignId=True)
                off_mols.append(OFFMolecule.from_rdkit(sm, allow_undefined_stereo=True))
                off_order.append(i)

        return off_mols, off_order

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

    def _inject_harmonic_bonds(
        self,
        system: openmm.System,
        pairs_off: List[Tuple[int, int]],
        coords_ang_off: np.ndarray,
        symbols_off: List[str],
    ) -> int:
        if not pairs_off:
            self._rd_bonds = []
            return 0

        hb = self._get_harmonic_bond_force(system)
        if hb is None:
            hb = openmm.HarmonicBondForce()
            system.addForce(hb)

        existing = set()
        for n in range(hb.getNumBonds()):
            i, j, _, _ = hb.getBondParameters(n)
            existing.add((min(int(i), int(j)), max(int(i), int(j))))

        coords_nm = coords_ang_off * _NM_PER_A
        added = 0
        rd_pairs: list[tuple[int, int]] = []

        for (i, j) in pairs_off:
            key = (min(i, j), max(i, j))
            rd_pairs.append(key)
            if key in existing:
                continue

            dij_nm = float(np.linalg.norm(coords_nm[i] - coords_nm[j]))
            r0_nm = dij_nm if (np.isfinite(dij_nm) and dij_nm >= 1e-6) else 0.101

            si = symbols_off[i].capitalize()
            sj = symbols_off[j].capitalize()
            kcal_per_A2 = 300.0 if ("H" in (si, sj)) else 200.0
            k_val = (
                (kcal_per_A2 * unit.kilocalories_per_mole / (unit.angstrom ** 2))
                .value_in_unit(unit.kilojoule_per_mole / (unit.nanometer ** 2))
            )
            hb.addBond(i, j, r0_nm, k_val)
            added += 1

        _log.info(
            "Injected %d harmonic bonds (total=%d); constraints now=%d",
            added, hb.getNumBonds(), system.getNumConstraints()
        )
        self._rd_bonds = rd_pairs
        return added

    def initialize(self, system: System) -> None:
        symbols = [a.element for a in system.atoms]
        coords_ang_sys = np.asarray([atom.position for atom in system.atoms], dtype=float)
        natoms = len(symbols)

        pairs_sys, mode, netq = self._determine_bonds(system)

        off_mols, off_order = self._split_to_off_molecules_and_order(
            natoms, pairs_sys, symbols, coords_ang_sys
        )

        if not off_mols and natoms == 0:
            self.system_omm = openmm.System()
            self._create_context()
            self._last_elements = tuple()
            self._sys_to_off = np.array([], dtype=int)
            self._off_to_sys = np.array([], dtype=int)
            return

        _log.info("SMIRNOFF: building OFF topology from molecules (no OpenMM conversion)")
        off_top = OFFTopology.from_molecules(off_mols)

        try:
            sys_omm = self.ff.create_openmm_system(off_top)
        except Exception as e:
            raise RuntimeError(f"failed to create OpenMM system from OpenFF: {e}")

        removed = self._remove_all_constraints(sys_omm)
        if removed > 0:
            _log.info("Removed %d OpenMM constraints to match external integration.", removed)

        off_order = list(off_order) if len(off_order) == natoms else list(range(natoms))
        sys_to_off = np.empty(natoms, dtype=int)
        for off_idx, sys_idx in enumerate(off_order):
            sys_to_off[sys_idx] = off_idx
        off_to_sys = np.array(off_order, dtype=int)
        self._sys_to_off = sys_to_off
        self._off_to_sys = off_to_sys

        coords_ang_off = coords_ang_sys[off_to_sys]
        symbols_off = [symbols[i] for i in off_to_sys]

        pairs_off = [(sys_to_off[i], sys_to_off[j]) for (i, j) in pairs_sys] if pairs_sys else []
        self._inject_harmonic_bonds(sys_omm, pairs_off, coords_ang_off, symbols_off)

        _log.info(
            "OpenMM system ready: Harmonic bonds=%s, constraints=%s",
            next((f.getNumBonds() for f in sys_omm.getForces()
                  if isinstance(f, openmm.HarmonicBondForce)), 0),
            sys_omm.getNumConstraints()
        )

        self.system_omm = sys_omm
        self._create_context()
        self._last_elements = tuple(a.element for a in system.atoms)

    def _ensure_context_current(self, system: System) -> None:
        need_reinit = (
            self.context is None
            or self.system_omm is None
            or self.system_omm.getNumParticles() != len(system.atoms)
            or self._last_elements != tuple(a.element for a in system.atoms)
        )
        if need_reinit:
            self.initialize(system)

    def compute_forces(self, system: System) -> np.ndarray:
        self._ensure_context_current(system)
        if self.context is None or self._sys_to_off is None or self._off_to_sys is None:
            raise RuntimeError("SMIRNOFFPlugin not initialized")

        coords_ang_sys = np.asarray([atom.position for atom in system.atoms], dtype=float)
        coords_ang_off = coords_ang_sys[self._off_to_sys]
        self.context.setPositions(coords_ang_off * _NM_PER_A * unit.nanometer)

        vels = getattr(system, "velocities", None)
        if vels is not None:
            vels_nm_ps_off = np.asarray(vels, dtype=float)[self._off_to_sys] * _NM_PER_A
            self.context.setVelocities(vels_nm_ps_off * (unit.nanometer / unit.picosecond))

        state = self.context.getState(getForces=True)
        f_kj_per_mol_per_nm_off = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )
        f_kcal_per_mol_per_A_off = np.asarray(f_kj_per_mol_per_nm_off, dtype=float) * _KJNM_TO_KCALA

        f_sys = np.empty_like(f_kcal_per_mol_per_A_off)
        for off_idx, sys_idx in enumerate(self._off_to_sys):
            f_sys[sys_idx] = f_kcal_per_mol_per_A_off[off_idx]
        return f_sys

    def compute_energy(self, system: System) -> float:
        self._ensure_context_current(system)
        if self.context is None or self._off_to_sys is None:
            raise RuntimeError("SMIRNOFFPlugin not initialized")
        coords_ang_sys = np.asarray([atom.position for atom in system.atoms], dtype=float)
        coords_ang_off = coords_ang_sys[self._off_to_sys]
        self.context.setPositions(coords_ang_off * _NM_PER_A * unit.nanometer)
        state = self.context.getState(getEnergy=True)
        e_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        return float(e_kj) * _KJ_TO_KCAL

    def render_hints(self) -> dict[str, Any]:
        if self._off_to_sys is None:
            return {"bonds": []}
        bonds_sys = [[int(self._off_to_sys[i]), int(self._off_to_sys[j])] for (i, j) in self._rd_bonds]
        return {"bonds": bonds_sys}
