from __future__ import annotations
import numpy as np
from simulation.system import System

_F2A = 418.4

class BaseIntegrator:
    def __init__(self, timestep: float):
        self.dt = float(timestep)

    def pre_force(self, system: System, forces: np.ndarray) -> None:
        raise NotImplementedError

    def post_force(self, system: System, forces: np.ndarray) -> None:
        raise NotImplementedError

class VerletIntegrator(BaseIntegrator):
    def pre_force(self, system: System, forces: np.ndarray) -> None:
        inv_m = 1.0 / system.masses[:, None]
        a = _F2A * forces * inv_m            
        half_dt = 0.5 * self.dt
        system.velocities += half_dt * a     
        system.positions  += self.dt * system.velocities  

    def post_force(self, system: System, forces: np.ndarray) -> None:
        inv_m = 1.0 / system.masses[:, None]
        a = _F2A * forces * inv_m          
        half_dt = 0.5 * self.dt
        system.velocities += half_dt * a    

class LeapfrogIntegrator(BaseIntegrator):
    def pre_force(self, system: System, forces: np.ndarray) -> None:
        inv_m = 1.0 / system.masses[:, None]
        a = _F2A * forces * inv_m
        system.velocities += self.dt * a
        system.positions  += self.dt * system.velocities

    def post_force(self, system: System, forces: np.ndarray) -> None:
        pass

class EulerIntegrator(BaseIntegrator):
    def pre_force(self, system: System, forces: np.ndarray) -> None:
        inv_m = 1.0 / system.masses[:, None]
        a = _F2A * forces * inv_m
        system.positions  += self.dt * system.velocities
        system.velocities += self.dt * a

    def post_force(self, system: System, forces: np.ndarray) -> None:
        pass
