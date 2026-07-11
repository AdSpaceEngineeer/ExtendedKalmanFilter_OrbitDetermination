from __future__ import annotations

import numpy as np

MU_EARTH_KM3_S2 = 398600.4418


def two_body_derivative(_time_s: float, state_km_kms: np.ndarray) -> np.ndarray:
    """Return two-body Cartesian dynamics for [r, v] in ECI coordinates."""
    position = state_km_kms[:3]
    velocity = state_km_kms[3:]
    radius = np.linalg.norm(position)
    acceleration = -MU_EARTH_KM3_S2 * position / radius**3
    return np.hstack((velocity, acceleration))


def propagate_state(state_km_kms: np.ndarray, dt_s: float) -> np.ndarray:
    """Propagate a Cartesian state by dt seconds with fixed-step RK4."""
    if abs(dt_s) < 1e-12:
        return state_km_kms.copy()

    step_count = max(1, int(np.ceil(abs(dt_s) / 30.0)))
    step_s = dt_s / step_count
    state = state_km_kms.copy()
    for _ in range(step_count):
        k1 = two_body_derivative(0.0, state)
        k2 = two_body_derivative(0.0, state + 0.5 * step_s * k1)
        k3 = two_body_derivative(0.0, state + 0.5 * step_s * k2)
        k4 = two_body_derivative(0.0, state + step_s * k3)
        state = state + (step_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return state


def finite_difference_state_transition(state_km_kms: np.ndarray, dt_s: float, step: float = 1e-5) -> np.ndarray:
    """Numerically approximate the state-transition matrix over dt seconds."""
    size = state_km_kms.size
    transition = np.zeros((size, size))
    for index in range(size):
        perturbation = np.zeros(size)
        perturbation[index] = step
        plus = propagate_state(state_km_kms + perturbation, dt_s)
        minus = propagate_state(state_km_kms - perturbation, dt_s)
        transition[:, index] = (plus - minus) / (2.0 * step)
    return transition
