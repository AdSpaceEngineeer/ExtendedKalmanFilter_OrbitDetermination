from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .simulation import DemoResult


def save_plots(result: DemoResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "orbit": output_dir / "orbit_estimate.svg",
        "position_error": output_dir / "position_error.svg",
        "residuals": output_dir / "residuals.svg",
    }

    _plot_orbit(result, paths["orbit"])
    _plot_position_error(result, paths["position_error"])
    _plot_residuals(result, paths["residuals"])
    return paths


def _plot_orbit(result: DemoResult, output_path: Path) -> None:
    fig = plt.figure(figsize=(8.5, 6))
    axis = fig.add_subplot(111, projection="3d")
    axis.plot(*result.truth_states[:, :3].T, label="Truth", linewidth=2.2)
    axis.plot(*result.estimated_states[:, :3].T, label="EKF estimate", linewidth=1.8, linestyle="--")
    axis.scatter(*result.truth_states[0, :3], label="Start", s=30)
    axis.set_title("True vs EKF Estimated Orbit")
    axis.set_xlabel("ECI x (km)")
    axis.set_ylabel("ECI y (km)")
    axis.set_zlabel("ECI z (km)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_position_error(result: DemoResult, output_path: Path) -> None:
    time_min = result.times_s / 60.0
    position_error = np.linalg.norm(result.estimated_states[:, :3] - result.truth_states[:, :3], axis=1)
    sigma_3 = 3.0 * np.sqrt(np.trace(result.covariances[:, :3, :3], axis1=1, axis2=2))

    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.plot(time_min, position_error, label="Position error", linewidth=2.2)
    axis.plot(time_min, sigma_3, label="3-sigma covariance bound", linewidth=1.8, linestyle="--")
    axis.set_title("EKF Position Error and Covariance Bound")
    axis.set_xlabel("Time (min)")
    axis.set_ylabel("Position error (km)")
    axis.grid(True, alpha=0.35)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_residuals(result: DemoResult, output_path: Path) -> None:
    time_min = result.times_s / 60.0
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6), sharex=True)
    range_residual_rms = np.sqrt(np.mean(result.residuals[:, 0::2] ** 2, axis=1))
    range_rate_residual_rms = np.sqrt(np.mean(result.residuals[:, 1::2] ** 2, axis=1))
    axes[0].plot(time_min, range_residual_rms, linewidth=1.8)
    axes[0].set_title("Measurement Residuals")
    axes[0].set_ylabel("Range RMS (km)")
    axes[0].grid(True, alpha=0.35)

    axes[1].plot(time_min, range_rate_residual_rms * 1000.0, linewidth=1.8)
    axes[1].set_xlabel("Time (min)")
    axes[1].set_ylabel("Range-rate RMS (m/s)")
    axes[1].grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
