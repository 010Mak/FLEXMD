from __future__ import annotations
from typing import Optional, List, Dict, Any
import numpy as np

from simulation.system import System
from simulation.integrators import BaseIntegrator
from simulation.thermostats import Thermostat
from simulation.plugin_interface import ForceCalculator
from simulation.plugin_selector import decide as select_plugin
from simulation.plugin_manager import PLUGINS
from utilities.config import DEFAULT_PSI4_THRESHOLD
from simulation.thermo import kinetic_and_temperature, attach_thermo

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
        self.step = step
        self.time = time
        self.positions = positions.copy()
        self.velocities = velocities.copy()
        self.energy = float(energy)
        self.forces = forces.copy()
        self.kinetic = kinetic
        self.temperature = temperature
        self.total_energy = total_energy

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "step": self.step,
            "time_ps": float(self.time),
            "positions": self.positions.tolist(),
            "velocities": self.velocities.tolist(),
            "energy": float(self.energy),
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
        plugin: ForceCalculator,
        thermostat: Optional[Thermostat] = None,
    ):
        self.system = system
        self.integrator = integrator
        self.plugin = plugin
        self.thermostat = thermostat
        self.time = 0.0
        self.plugin.initialize(self.system)

    @classmethod
    def from_config(cls, system: System, integrator: BaseIntegrator, system_kwargs: Dict[str, Any]) -> "EngineCore":
        backend = system_kwargs.get("backend", "auto")
        threshold = system_kwargs.get("psi4_threshold", DEFAULT_PSI4_THRESHOLD)
        plugin_name = select_plugin(system, user_choice=backend, threshold=threshold)
        plugin_cls = PLUGINS[plugin_name]

        pkwargs = dict(system_kwargs.get("plugin_args", {}) or {})
        _ = pkwargs.pop("timestep_ps", None)
        plugin = plugin_cls(**pkwargs)

        try:
            plugin.set_timestep_ps(integrator.dt)
        except Exception:
            pass

        thermostat = system_kwargs.get("thermostat")
        return cls(system, integrator, plugin, thermostat)

    def run(self, n_steps: int, *, report_stride: int = 1, include_thermo: bool = False) -> List[StepResult]:
        forces = self.plugin.compute_forces(self.system)
        energy = self.plugin.compute_energy(self.system)

        results: List[StepResult] = []
        for step in range(1, n_steps + 1):
            self.integrator.pre_force(self.system, forces)
            if self.thermostat:
                self.thermostat.apply(self.system, step, self.integrator.dt)

            forces_new = self.plugin.compute_forces(self.system)
            energy = self.plugin.compute_energy(self.system)

            self.integrator.post_force(self.system, forces_new)
            self.time += self.integrator.dt

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
