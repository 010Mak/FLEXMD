from __future__ import annotations

from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from simulation.system import System
from simulation.integrators import BaseIntegrator
from simulation.thermostats import Thermostat
from simulation.plugin_interface import ForceCalculator as _FCBase
from simulation.plugin_selector import decide as select_plugin
from simulation.plugin_manager import PLUGINS
from utilities.config import DEFAULT_PSI4_THRESHOLD, FORCEFIELD_DIR
from utilities.ffield_utils import candidates_for_elements
from simulation.thermo import attach_thermo

_ORGANIC = {"C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I", "B", "Si"}


class StepResult:
    def __init__(
        self,
        step: int,
        time: float,
        positions: np.ndarray,
        velocities: np.ndarray,
        energy: float,
        forces: np.ndarray,
        *,
        kinetic: Optional[float] = None,
        temperature: Optional[float] = None,
        total_energy: Optional[float] = None,
    ):
        self.step = int(step)
        self.time = float(time)
        self.positions = np.asarray(positions, dtype=float).copy()
        self.velocities = np.asarray(velocities, dtype=float).copy()
        self.energy = float(energy)
        self.forces = np.asarray(forces, dtype=float).copy()
        self.kinetic = kinetic
        self.temperature = temperature
        self.total_energy = total_energy

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "step": self.step,
            "time_ps": self.time,
            "positions": self.positions.tolist(),
            "velocities": self.velocities.tolist(),
            "energy": self.energy,
            "forces": self.forces.tolist(),
        }
        if self.kinetic is not None:
            out["kinetic_energy_kcal_per_mol"] = float(self.kinetic)
        if self.temperature is not None:
            out["temperature_K"] = float(self.temperature)
        if self.total_energy is not None:
            out["total_energy_kcal_per_mol"] = float(self.total_energy)
        return out


class EngineCore:
    def __init__(
        self,
        system: System,
        integrator: BaseIntegrator,
        plugin,
        thermostat: Optional[Thermostat] = None,
    ):
        self.system = system
        self.integrator = integrator
        self.plugin = plugin
        self.thermostat = thermostat
        self.time = 0.0

    @classmethod
    def from_config(
        cls,
        system: System,
        integrator: BaseIntegrator,
        system_kwargs: Dict[str, Any],
    ) -> "EngineCore":

        backend = system_kwargs.get("backend", "auto")
        threshold = system_kwargs.get("psi4_threshold", DEFAULT_PSI4_THRESHOLD)
        plugin_name = select_plugin(system, user_choice=backend, threshold=threshold)

        if plugin_name not in PLUGINS:
            raise RuntimeError(f"Plugin '{plugin_name}' not available")

        plugin_cls = PLUGINS[plugin_name]

        pkwargs = dict(system_kwargs.get("plugin_args", {}) or {})
        plugin = plugin_cls(**pkwargs)

        try:
            plugin.set_timestep_ps(integrator.dt)
        except Exception:
            pass

        thermostat = system_kwargs.get("thermostat")
        return cls(system, integrator, plugin, thermostat)

    def get_meta(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        meta["backend"] = getattr(self.plugin, "NAME", "unknown")
        for k in ("ffield_id", "ff_id", "ff_path", "qeq_mode", "version", "ff_version"):
            v = getattr(self.plugin, k, None)
            if v is not None:
                meta[k] = v
        return meta

    def _elements_sorted(self) -> List[str]:
        return sorted({str(a.element).capitalize() for a in self.system.atoms})

    def _sync_atoms_from_arrays(self) -> None:
        atoms = getattr(self.system, "atoms", None)
        if not atoms:
            return
        p = np.asarray(self.system.positions, dtype=float)
        if p.ndim != 2 or p.shape[1] != 3 or len(p) != len(atoms):
            return
        for i, a in enumerate(atoms):
            try:
                a.position[...] = p[i]
            except Exception:
                a.position = p[i].copy()

    def _initialize_plugin(self) -> None:
        self._sync_atoms_from_arrays()
        self.plugin.initialize(self.system)

    def _first_forces_energy(self) -> Tuple[np.ndarray, float]:
        self._sync_atoms_from_arrays()
        f = self.plugin.compute_forces(self.system)
        e = self.plugin.compute_energy(self.system)
        return np.asarray(f, float), float(e)

    def _try_reaxff_fallbacks(self, first_error: Exception) -> bool:
        elems = self._elements_sorted()
        cand_list = candidates_for_elements(elems, FORCEFIELD_DIR)
        for ff_path, _order in cand_list:
            try:
                try:
                    if hasattr(self.plugin, "close"):
                        self.plugin.close()
                except Exception:
                    pass

                pkwargs = {}
                for k, v in getattr(self.plugin, "__dict__", {}).items():
                    if not str(k).startswith("_"):
                        pkwargs[k] = v
                pkwargs["ff_path"] = ff_path

                plugin_cls = PLUGINS["reaxff"]
                new_plugin = plugin_cls(**pkwargs)
                try:
                    new_plugin.set_timestep_ps(self.integrator.dt)
                except Exception:
                    pass
                self._sync_atoms_from_arrays()
                new_plugin.initialize(self.system)
                self.plugin = new_plugin
                return True
            except Exception:
                continue
        return False

    def _try_smirnoff_fallback(self) -> bool:
        elems = set(self._elements_sorted())
        if not elems.issubset(_ORGANIC):
            return False
        if "smirnoff" not in PLUGINS:
            return False
        try:
            plugin_cls = PLUGINS["smirnoff"]
            new_plugin = plugin_cls()
            try:
                new_plugin.set_timestep_ps(self.integrator.dt)
            except Exception:
                pass
            self._sync_atoms_from_arrays()
            new_plugin.initialize(self.system)
            self.plugin = new_plugin
            return True
        except Exception:
            return False

    def _plugin_has_native_integrate(self) -> bool:
        integ = getattr(self.plugin, "integrate", None)
        if integ is None:
            return False
        try:
            return getattr(integ, "__func__", None) is not _FCBase.integrate
        except Exception:
            return True

    def run(
        self,
        n_steps: int,
        *,
        report_stride: int = 1,
        include_thermo: bool = False,
    ) -> List[StepResult]:
        if n_steps <= 0:
            return []

        try:
            self._initialize_plugin()
        except Exception as e:
            name = getattr(self.plugin, "NAME", "").lower()
            if name == "reaxff":
                if not self._try_reaxff_fallbacks(e) and not self._try_smirnoff_fallback():
                    raise
            else:
                raise

        if self._plugin_has_native_integrate():
            self._sync_atoms_from_arrays()
            frames = self.plugin.integrate(
                self.system,
                n_steps,
                self.integrator.dt,
                report_stride=max(1, int(report_stride)),
            )
            results: List[StepResult] = []
            t_ps = 0.0
            step_idx = 0
            for i, fr in enumerate(frames or []):
                pos = np.asarray(fr.get("positions", self.system.positions), float)
                vel = np.asarray(fr.get("velocities", getattr(self.system, "velocities", np.zeros_like(pos))), float)
                ene = float(fr.get("energy", 0.0))
                frc = np.asarray(fr.get("forces", np.zeros_like(pos)), float)
                t_ps = float(fr.get("time_ps", t_ps + self.integrator.dt * max(1, int(report_stride))))
                step_idx = int(fr.get("step", (i + 1) * max(1, int(report_stride))))

                res = StepResult(step=step_idx, time=t_ps, positions=pos, velocities=vel, energy=ene, forces=frc)

                if include_thermo:
                    masses_amu = getattr(self.system, "masses_amu", None)
                    if masses_amu is None:
                        masses_amu = getattr(self.system, "masses", None)
                    if masses_amu is None:
                        raise RuntimeError("system is missing masses (amu) needed for thermo.")
                    attach_thermo(
                        res,
                        velocities_A_per_ps=res.velocities,
                        masses_amu=masses_amu,
                        potential_energy_kcal_per_mol=res.energy,
                        remove_com=True,
                        dof_offset=0,
                    )

                results.append(res)
            return results

        try:
            forces, energy = self._first_forces_energy()
        except Exception as e:
            name = getattr(self.plugin, "NAME", "").lower()
            if name == "reaxff":
                if self._try_reaxff_fallbacks(e) or self._try_smirnoff_fallback():
                    forces, energy = self._first_forces_energy()
                else:
                    raise
            else:
                raise

        results: List[StepResult] = []
        stride = max(1, int(report_stride))

        for step in range(1, n_steps + 1):
            self.integrator.pre_force(self.system, forces)

            self._sync_atoms_from_arrays()

            if self.thermostat:
                self.thermostat.apply(self.system, step, self.integrator.dt)

            forces_new = np.asarray(self.plugin.compute_forces(self.system), float)
            energy = float(self.plugin.compute_energy(self.system))

            self.integrator.post_force(self.system, forces_new)
            self.time += self.integrator.dt

            if (step % stride) == 0:
                res = StepResult(
                    step=step,
                    time=self.time,
                    positions=self.system.positions,
                    velocities=self.system.velocities,
                    energy=energy,
                    forces=forces_new,
                )
                if include_thermo:
                    masses_amu = getattr(self.system, "masses_amu", None)
                    if masses_amu is None:
                        masses_amu = getattr(self.system, "masses", None)
                    if masses_amu is None:
                        raise RuntimeError("system is missing masses (amu) needed for thermo.")
                    attach_thermo(
                        res,
                        velocities_A_per_ps=res.velocities,
                        masses_amu=masses_amu,
                        potential_energy_kcal_per_mol=res.energy,
                        remove_com=True,
                        dof_offset=0,
                    )
                results.append(res)
            forces = forces_new
        return results
