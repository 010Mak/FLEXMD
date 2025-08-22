from __future__ import annotations
import abc
import numpy as np
from simulation.system import System

class Thermostat(abc.ABC):
    def __init__(self, target_temp: float):
        self.target_temp = target_temp  

    @abc.abstractmethod
    def apply(self, system: System, step: int, dt: float) -> None:
        raise NotImplementedError("apply() must be implemented by subclass")

class BerendsenThermostat(Thermostat):
    def __init__(self, target_temp: float, tau: float):
        super().__init__(target_temp)
        self.tau = tau  

    def apply(self, system: System, step: int, dt: float) -> None:
        amu_to_kg = 1.66053906660e-27
        ang_to_m = 1e-10
        ps_to_s = 1e-12
        masses_si = system.masses * amu_to_kg
        v_si = system.velocities * (ang_to_m / ps_to_s)
        kin_si = 0.5 * np.sum(masses_si[:, None] * v_si**2)
        dof = 3 * len(system.atoms)
        kB = 1.380649e-23 
        T_inst = 2 * kin_si / (dof * kB)
        lam = np.sqrt(1.0 + (dt / self.tau) * (self.target_temp / T_inst - 1.0))
        system.velocities *= lam

class LangevinThermostat(Thermostat):
    def __init__(self, target_temp: float, friction: float):
        super().__init__(target_temp)
        self.friction = friction  

    def apply(self, system: System, step: int, dt: float) -> None:
        exp_fac = np.exp(-self.friction * dt)
        system.velocities *= exp_fac

        amu_to_kg = 1.66053906660e-27
        ang_to_m = 1e-10
        ps_to_s = 1e-12
        masses_si = system.masses * amu_to_kg
        kB = 1.380649e-23  
        var = (1.0 - exp_fac**2) * kB * self.target_temp / masses_si
        sigma_si = np.sqrt(var)
        rand = np.random.normal(size=system.velocities.shape)
        sigma_md = sigma_si * (ps_to_s / ang_to_m) 
        system.velocities += rand * sigma_md[:, None]

__all__ = ["Thermostat", "BerendsenThermostat", "LangevinThermostat"]
