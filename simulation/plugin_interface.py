from __future__ import annotations
import abc
from typing import ClassVar, List, Dict, Any
import numpy as np

from simulation.system import System

class ForceCalculator(abc.ABC):
    NAME: ClassVar[str]
    CAPABILITY: ClassVar[str]
    MIN_ATOMS: ClassVar[int] = 0
    MAX_ATOMS: ClassVar[int] = int(1e12)

    @classmethod
    @abc.abstractmethod
    def is_available(cls) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def initialize(self, system: System) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def compute_forces(self, system: System) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def compute_energy(self, system: System) -> float:
        raise NotImplementedError

    def set_timestep_ps(self, dt_ps: float) -> None:
        if hasattr(self, "timestep_ps"):
            try:
                setattr(self, "timestep_ps", float(dt_ps))
            except Exception:
                pass

    def integrate(self, system: System, n_steps: int, dt_ps: float, report_stride: int = 1) -> List[Dict[str, Any]]:
        raise NotImplementedError("integrate not implemented for this plugin")
