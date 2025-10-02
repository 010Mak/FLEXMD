from __future__ import annotations
import numpy as np

KCALMOL_PER_A_TO_A_PS2_PER_AMU = 418.4

class BaseIntegrator:
    def __init__(self, dt_ps: float = None, *, timestep: float = None, dt: float = None, **_):
        chosen = dt_ps
        if chosen is None:
            chosen = timestep
        if chosen is None:
            chosen = dt
        if chosen is None:
            raise TypeError("Integrator requires a timestep in ps (use dt_ps=..., or timestep=..., or dt=...).")
        self.dt = float(chosen)

    def _accel(self, system, forces: np.ndarray) -> np.ndarray:
        f = np.asarray(forces, dtype=float)
        if f.ndim != 2 or f.shape[1] != 3:
            raise RuntimeError("Forces must have shape (N,3) in (kcal/mol)/Å.")

        m = getattr(system, "masses_amu", None)
        if m is None:
            m = getattr(system, "masses", None)
        if m is None:
            raise RuntimeError("System is missing masses (amu).")

        m = np.asarray(m, dtype=float)
        if m.ndim != 1 or m.shape[0] != f.shape[0]:
            raise RuntimeError(f"Mass array shape mismatch: masses={m.shape}, forces={f.shape}")

        invm = 1.0 / m
        a = (KCALMOL_PER_A_TO_A_PS2_PER_AMU * f) * invm[:, None]
        return a

    def pre_force(self, system, forces: np.ndarray) -> None:
        raise NotImplementedError

    def post_force(self, system, forces_new: np.ndarray) -> None:
        raise NotImplementedError


class VerletIntegrator(BaseIntegrator):
    def pre_force(self, system, forces: np.ndarray) -> None:
        a = self._accel(system, forces)
        system.velocities += 0.5 * self.dt * a
        system.positions  += self.dt * system.velocities

    def post_force(self, system, forces_new: np.ndarray) -> None:
        a_new = self._accel(system, forces_new)
        system.velocities += 0.5 * self.dt * a_new


VelocityVerlet = VerletIntegrator


class LeapfrogIntegrator(BaseIntegrator):
    def pre_force(self, system, forces: np.ndarray) -> None:
        a = self._accel(system, forces)
        system.velocities += self.dt * a
        system.positions  += self.dt * system.velocities

    def post_force(self, system, forces_new: np.ndarray) -> None:
        pass

class EulerIntegrator(BaseIntegrator):
    def pre_force(self, system, forces: np.ndarray) -> None:
        a = self._accel(system, forces)
        system.velocities += self.dt * a
        system.positions  += self.dt * system.velocities

    def post_force(self, system, forces_new: np.ndarray) -> None:
        pass

__all__ = [
    "BaseIntegrator",
    "VerletIntegrator",
    "LeapfrogIntegrator",
    "EulerIntegrator",
    "VelocityVerlet",
    "KCALMOL_PER_A_TO_A_PS2_PER_AMU",
]
