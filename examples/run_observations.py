from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ekf_od.angles import estimate_gauss_states, results_to_dataframe
from ekf_od.data import load_observations


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    observation_path = repo_root / "Observations.xlsx"
    output_dir = repo_root / "docs" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    observations = load_observations(observation_path)
    results = estimate_gauss_states(observations)
    states = results_to_dataframe(results)
    csv_path = repo_root / "docs" / "gauss_iod_states.csv"
    states.to_csv(csv_path, index=False)

    plot_path = output_dir / "gauss_iod_observation_states.svg"
    _plot_states(states, plot_path)

    first = states.iloc[0]
    print(f"Loaded observations: {len(observations)}")
    print(f"Gauss IOD triplets processed: {len(states)}")
    print(
        "First middle-observation state: "
        f"r=[{first.rx_km:.2f}, {first.ry_km:.2f}, {first.rz_km:.2f}] km, "
        f"v=[{first.vx_km_s:.5f}, {first.vy_km_s:.5f}, {first.vz_km_s:.5f}] km/s"
    )
    print(f"State CSV: {csv_path}")
    print(f"Plot: {plot_path}")


def _plot_states(states, output_path: Path) -> None:
    fig = plt.figure(figsize=(8.5, 6))
    axis = fig.add_subplot(111, projection="3d")
    axis.plot(states.rx_km, states.ry_km, states.rz_km, marker="o", linewidth=1.8)
    axis.set_title("Gauss IOD States from Observations.xlsx")
    axis.set_xlabel("ECI x (km)")
    axis.set_ylabel("ECI y (km)")
    axis.set_zlabel("ECI z (km)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
