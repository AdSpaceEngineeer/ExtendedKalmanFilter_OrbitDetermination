from pathlib import Path

import numpy as np

from ekf_od.plotting import save_plots
from ekf_od.simulation import run_demo


def main() -> None:
    result = run_demo()
    output_dir = Path(__file__).resolve().parents[1] / "docs" / "plots"
    paths = save_plots(result, output_dir)

    position_error = np.linalg.norm(result.estimated_states[:, :3] - result.truth_states[:, :3], axis=1)
    velocity_error = np.linalg.norm(result.estimated_states[:, 3:] - result.truth_states[:, 3:], axis=1)
    print(f"First posterior position error: {position_error[0]:.3f} km")
    print(f"Final position error: {position_error[-1]:.3f} km")
    print(f"First posterior velocity error: {velocity_error[0]:.6f} km/s")
    print(f"Final velocity error: {velocity_error[-1]:.6f} km/s")
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
