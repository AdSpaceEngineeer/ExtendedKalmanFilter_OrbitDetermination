from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamics import propagate_state
from .filter import ExtendedKalmanFilter
from .measurements import stacked_range_range_rate


@dataclass
class DemoResult:
    times_s: np.ndarray
    truth_states: np.ndarray
    estimated_states: np.ndarray
    covariances: np.ndarray
    measurements: np.ndarray
    residuals: np.ndarray


def station_states_eci(_time_s: float) -> np.ndarray:
    """Three simple ECI ground-station states for a compact geometry demo."""
    return np.array(
        [
            [6378.137, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-3189.0685, 5523.6287, 0.0, 0.0, 0.0, 0.0],
            [-3189.0685, -2761.8143, 4783.6010, 0.0, 0.0, 0.0],
        ]
    )


def simulate_truth(initial_state: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    truth = [initial_state]
    for previous, current in zip(times_s[:-1], times_s[1:]):
        truth.append(propagate_state(truth[-1], current - previous))
    return np.vstack(truth)


def run_demo(seed: int = 7) -> DemoResult:
    rng = np.random.default_rng(seed)
    times_s = np.linspace(0.0, 5400.0, 90)
    truth_initial = np.array([7000.0, 0.0, 80.0, 0.0, 7.52, 0.35])
    truth_states = simulate_truth(truth_initial, times_s)

    single_station_sigma = np.array([0.02, 0.00002])
    measurement_sigma = np.tile(single_station_sigma, 3)
    measurements = []
    for time_s, state in zip(times_s, truth_states):
        clean = stacked_range_range_rate(state, station_states_eci(time_s))
        measurements.append(clean + rng.normal(0.0, measurement_sigma))
    measurements = np.vstack(measurements)

    initial_error = np.array([1.5, -1.2, 0.8, 0.0008, -0.0012, 0.0007])
    ekf = ExtendedKalmanFilter(
        state=truth_initial + initial_error,
        covariance=np.diag([4.0, 4.0, 4.0, 0.0004, 0.0004, 0.0004]),
        process_noise=np.diag([1e-8, 1e-8, 1e-8, 1e-11, 1e-11, 1e-11]),
        measurement_noise=np.diag(measurement_sigma**2),
    )

    estimates = []
    covariances = []
    residuals = []
    previous_time = times_s[0]
    for time_s, measurement in zip(times_s, measurements):
        ekf.predict(time_s - previous_time)
        residual = ekf.update(measurement, station_states_eci(time_s))
        estimates.append(ekf.state.copy())
        covariances.append(ekf.covariance.copy())
        residuals.append(residual)
        previous_time = time_s

    return DemoResult(
        times_s=times_s,
        truth_states=truth_states,
        estimated_states=np.vstack(estimates),
        covariances=np.stack(covariances),
        measurements=measurements,
        residuals=np.vstack(residuals),
    )
