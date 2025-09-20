from __future__ import annotations
import numpy as np

K_BOLTZMANN = 0.00198720425864083

VELVAR_CONV = 418.4

class Thermostat:
    def apply(self, system, step: int, dt_ps: float) -> None:
        pass

class LangevinThermostat(Thermostat):
    def __init__(self,
                 temperature_K: float = 300.0,
                 gamma_ps: float = None,
                 *,
                 friction_ps: float = None,
                 friction_coeff: float = None,
                 gamma: float = None,
                 **_ignore):
        self.temperature_K = float(temperature_K)

        g = gamma_ps
        if g is None: g = friction_ps
        if g is None: g = friction_coeff
        if g is None: g = gamma
        if g is None: g = 1.0
        self.gamma_ps = float(g)

    def apply(self, system, step: int, dt_ps: float) -> None:
        if self.gamma_ps <= 0.0 or self.temperature_K <= 0.0:
            return

        m = getattr(system, "masses_amu", None)
        if m is None:
            m = getattr(system, "masses", None)
        if m is None:
            raise RuntimeError("LangevinThermostat: system missing masses (amu).")

        v = np.asarray(system.velocities, dtype=float)
        m = np.asarray(m, dtype=float)

        if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] != m.shape[0]:
            raise RuntimeError("LangevinThermostat: velocity/mass shape mismatch.")

        dt = float(dt_ps)
        c = float(np.exp(-self.gamma_ps * dt))

        sigma2 = (1.0 - c * c) * VELVAR_CONV * K_BOLTZMANN * self.temperature_K / m
        sigma = np.sqrt(sigma2)

        noise = np.random.normal(0.0, 1.0, size=v.shape)
        v[:] = c * v + sigma[:, None] * noise
        system.velocities = v


class BerendsenThermostat(Thermostat):
    def __init__(self, temperature_K: float = 300.0, tau_ps: float = None, *, time_constant_ps: float = None, tau: float = None, **_ignore):
        self.temperature_K = float(temperature_K)
        t = tau_ps
        if t is None: t = time_constant_ps
        if t is None: t = tau
        if t is None: t = 1.0
        self.tau_ps = float(t)

    def apply(self, system, step: int, dt_ps: float) -> None:
        m = getattr(system, "masses_amu", None)
        if m is None:
            m = getattr(system, "masses", None)
        if m is None:
            raise RuntimeError("BerendsenThermostat: system missing masses (amu).")

        v = np.asarray(system.velocities, dtype=float)
        m = np.asarray(m, dtype=float)

        if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] != m.shape[0]:
            raise RuntimeError("BerendsenThermostat: velocity/mass shape mismatch.")

        dof = 3 * v.shape[0]
        K_num = 0.5 * float(np.sum(m[:, None] * (v * v))) / VELVAR_CONV
        if K_num <= 0.0 or dof <= 0:
            return
        T_inst = (2.0 * K_num) / (dof * K_BOLTZMANN)

        lam_sq = 1.0 + (dt_ps / max(1e-12, self.tau_ps)) * (self.temperature_K / max(1e-12, T_inst) - 1.0)
        lam_sq = max(lam_sq, 0.0)
        lam = np.sqrt(lam_sq)
        system.velocities *= lam


Langevin = LangevinThermostat
Berendsen = BerendsenThermostat

__all__ = [
    "Thermostat",
    "LangevinThermostat",
    "BerendsenThermostat",
    "Langevin",
    "Berendsen",
    "K_BOLTZMANN",
    "VELVAR_CONV",
]
