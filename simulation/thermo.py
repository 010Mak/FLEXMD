from __future__ import annotations
import numpy as np

NA = 6.02214076e23
R_J_PER_MOL_K = 8.314462618
J_PER_KCAL = 4184.0
AMU_TO_KG = 1.66053906660e-27
ANGSTROM_TO_M = 1.0e-10
PS_TO_S = 1.0e-12

AMU_A2_PS2_TO_KCAL_PER_MOL = (
    AMU_TO_KG * (ANGSTROM_TO_M / PS_TO_S) ** 2 * NA / J_PER_KCAL
)

def kinetic_and_temperature(
    velocities_A_per_ps: np.ndarray,
    masses_amu: np.ndarray,
    *,
    remove_com: bool = True,
    dof_offset: int = 0
) -> tuple[float, float, int]:
    v = np.asarray(velocities_A_per_ps, dtype=float)
    m = np.asarray(masses_amu, dtype=float)
    assert v.ndim == 2 and v.shape[1] == 3, "velocities must be (N,3)"
    assert m.ndim == 1 and m.shape[0] == v.shape[0], "masses must be (N,)"

    N = v.shape[0]
    removed = 0
    if remove_com and N > 0:
        vcom = (m[:, None] * v).sum(axis=0) / m.sum()
        v = v - vcom
        removed = 3

    mv2 = (m[:, None] * v * v).sum()
    K_kcalmol = 0.5 * AMU_A2_PS2_TO_KCAL_PER_MOL * mv2

    dof = max(1, 3 * N - removed - int(dof_offset))
    T_K = (2.0 * (K_kcalmol * J_PER_KCAL)) / (dof * R_J_PER_MOL_K)

    return float(K_kcalmol), float(T_K), int(dof)

def attach_thermo(
    result: object,
    *,
    velocities_A_per_ps,
    masses_amu,
    potential_energy_kcal_per_mol: float,
    remove_com: bool = True,
    dof_offset: int = 0
) -> object:
    K_kcalmol, T_K, _ = kinetic_and_temperature(
        velocities_A_per_ps, masses_amu, remove_com=remove_com, dof_offset=dof_offset
    )
    setattr(result, "kinetic", K_kcalmol)
    setattr(result, "temperature", T_K)
    setattr(result, "total_energy", float(potential_energy_kcal_per_mol + K_kcalmol))
    return result
