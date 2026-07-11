from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamics import finite_difference_state_transition, propagate_state
from .measurements import finite_difference_measurement_jacobian, stacked_range_range_rate


@dataclass
class ExtendedKalmanFilter:
    """Minimal EKF for Cartesian orbit determination."""

    state: np.ndarray
    covariance: np.ndarray
    process_noise: np.ndarray
    measurement_noise: np.ndarray

    def predict(self, dt_s: float) -> None:
        transition = finite_difference_state_transition(self.state, dt_s)
        self.state = propagate_state(self.state, dt_s)
        self.covariance = transition @ self.covariance @ transition.T + self.process_noise
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def update(self, measurement: np.ndarray, station_state: np.ndarray) -> np.ndarray:
        expected = stacked_range_range_rate(self.state, station_state)
        jacobian = finite_difference_measurement_jacobian(self.state, station_state)
        innovation = measurement - expected
        innovation_covariance = jacobian @ self.covariance @ jacobian.T + self.measurement_noise
        kalman_gain = self.covariance @ jacobian.T @ np.linalg.inv(innovation_covariance)

        self.state = self.state + kalman_gain @ innovation
        identity = np.eye(self.state.size)
        joseph = identity - kalman_gain @ jacobian
        self.covariance = joseph @ self.covariance @ joseph.T + kalman_gain @ self.measurement_noise @ kalman_gain.T
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        return innovation
