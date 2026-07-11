from __future__ import annotations

import numpy as np


def range_range_rate(state_km_kms: np.ndarray, station_state_km_kms: np.ndarray) -> np.ndarray:
    """Measure range and range-rate from a ground-station ECI state."""
    relative_position = state_km_kms[:3] - station_state_km_kms[:3]
    relative_velocity = state_km_kms[3:] - station_state_km_kms[3:]
    range_km = np.linalg.norm(relative_position)
    range_rate_km_s = float(np.dot(relative_position, relative_velocity) / range_km)
    return np.array([range_km, range_rate_km_s])


def stacked_range_range_rate(state_km_kms: np.ndarray, station_states_km_kms: np.ndarray) -> np.ndarray:
    """Stack range/range-rate measurements from one or more stations."""
    stations = np.atleast_2d(station_states_km_kms)
    return np.hstack([range_range_rate(state_km_kms, station) for station in stations])


def finite_difference_measurement_jacobian(
    state_km_kms: np.ndarray,
    station_state_km_kms: np.ndarray,
    step: float = 1e-5,
) -> np.ndarray:
    """Numerically approximate the range/range-rate measurement Jacobian."""
    base_measurement = stacked_range_range_rate(state_km_kms, station_state_km_kms)
    measurement_size = base_measurement.size
    state_size = state_km_kms.size
    jacobian = np.zeros((measurement_size, state_size))
    for index in range(state_size):
        perturbation = np.zeros(state_size)
        perturbation[index] = step
        plus = stacked_range_range_rate(state_km_kms + perturbation, station_state_km_kms)
        minus = stacked_range_range_rate(state_km_kms - perturbation, station_state_km_kms)
        jacobian[:, index] = (plus - minus) / (2.0 * step)
    return jacobian
