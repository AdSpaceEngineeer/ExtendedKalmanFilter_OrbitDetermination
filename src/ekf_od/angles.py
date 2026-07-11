from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

MU_EARTH_KM3_S2 = 398600.0
EARTH_RADIUS_KM = 6378.0
EARTH_ECCENTRICITY = 0.08182
OBSERVATORY_LATITUDE_DEG = 40.1164
OBSERVATORY_HEIGHT_KM = 233.0 / 1000.0


@dataclass
class GaussIODResult:
    triplet_index: int
    time_mjd: float
    position_km: np.ndarray
    velocity_km_s: np.ndarray
    range_km: float
    station_position_km: np.ndarray
    line_of_sight: np.ndarray


def estimate_gauss_states(observations: pd.DataFrame) -> list[GaussIODResult]:
    """Estimate middle-observation states from optical angle triplets."""
    data = _prepare_triplets(observations)
    results: list[GaussIODResult] = []
    for triplet_index, start in enumerate(range(0, len(data), 3)):
        triplet = data.iloc[start : start + 3]
        result = gauss_iod_triplet(
            right_ascension_deg=triplet["right_ascension_deg"].to_numpy(float),
            declination_deg=triplet["declination_deg"].to_numpy(float),
            local_sidereal_deg=triplet["local_sidereal_deg"].to_numpy(float),
            time_mjd=triplet["time_mjd"].to_numpy(float),
            triplet_index=triplet_index,
        )
        results.append(result)
    return results


def gauss_iod_triplet(
    right_ascension_deg: np.ndarray,
    declination_deg: np.ndarray,
    local_sidereal_deg: np.ndarray,
    time_mjd: np.ndarray,
    triplet_index: int = 0,
) -> GaussIODResult:
    """Gauss angles-only preliminary orbit determination for one 3-observation arc."""
    station = station_position_eci(local_sidereal_deg[1])
    alpha1, alpha2, alpha3 = np.deg2rad(right_ascension_deg)
    delta1, delta2, delta3 = np.deg2rad(declination_deg)
    tau1 = (time_mjd[0] - time_mjd[1]) * 24.0 * 60.0 * 60.0
    tau3 = (time_mjd[2] - time_mjd[1]) * 24.0 * 60.0 * 60.0
    tau = (time_mjd[2] - time_mjd[0]) * 24.0 * 60.0 * 60.0

    los1 = line_of_sight(alpha1, delta1)
    los2 = line_of_sight(alpha2, delta2)
    los3 = line_of_sight(alpha3, delta3)

    cross_products = [np.cross(los2, los3), np.cross(los1, los3), np.cross(los1, los2)]
    d0 = float(np.dot(los1, cross_products[0]))
    if abs(d0) < 1e-12:
        raise ValueError(f"Singular Gauss geometry in triplet {triplet_index}")

    # The original script assumes the observatory position is effectively
    # constant over each short three-observation arc.
    d = np.array([np.dot(station, cross_product) for _ in range(3) for cross_product in cross_products])

    a_coeff = (1.0 / d0) * (-d[1] * tau3 / tau + d[4] + d[7] * tau1 / tau)
    b_coeff = (1.0 / (6.0 * d0)) * (
        d[1] * (tau3**2 - tau**2) * tau3 / tau
        + d[7] * (tau**2 - tau1**2) * tau1 / tau
    )
    station_projection = float(np.dot(station, los2))
    station_radius_sq = float(np.dot(station, station))

    polynomial = [
        1.0,
        0.0,
        -(a_coeff**2 + 2.0 * a_coeff * station_projection + station_radius_sq),
        0.0,
        0.0,
        -2.0 * MU_EARTH_KM3_S2 * b_coeff * (a_coeff + station_projection),
        0.0,
        0.0,
        -(MU_EARTH_KM3_S2**2) * b_coeff**2,
    ]
    positive_roots = [root.real for root in np.roots(polynomial) if np.isreal(root) and root.real > 0]
    if not positive_roots:
        raise ValueError(f"No positive real Gauss root in triplet {triplet_index}")
    r2_norm = max(positive_roots)

    c1 = (tau3 / tau) * (1.0 + (MU_EARTH_KM3_S2 * (tau**2 - tau3**2)) / (6.0 * r2_norm**3))
    c3 = -(tau1 / tau) * (1.0 + (MU_EARTH_KM3_S2 * (tau**2 - tau1**2)) / (6.0 * r2_norm**3))
    rho1 = (1.0 / d0) * (-d[0] + d[3] / c1 - c3 * d[6] / c1)
    rho2 = (1.0 / d0) * (-c1 * d[1] + d[4] - c3 * d[7])
    rho3 = (1.0 / d0) * (-c1 * d[2] / c3 + d[5] / c3 - d[8])

    r1_vec = station + rho1 * los1
    r2_vec = station + rho2 * los2
    r3_vec = station + rho3 * los3

    f1 = 1.0 - 0.5 * MU_EARTH_KM3_S2 * tau1**2 / r2_norm**3
    f3 = 1.0 - 0.5 * MU_EARTH_KM3_S2 * tau3**2 / r2_norm**3
    denominator = tau - (1.0 / 6.0) * MU_EARTH_KM3_S2 * tau**3 / r2_norm**3
    v2_vec = (f1 * r3_vec - f3 * r1_vec) / denominator

    return GaussIODResult(
        triplet_index=triplet_index,
        time_mjd=float(time_mjd[1]),
        position_km=r2_vec,
        velocity_km_s=v2_vec,
        range_km=float(rho2),
        station_position_km=station,
        line_of_sight=los2,
    )


def station_position_eci(local_sidereal_deg: float) -> np.ndarray:
    latitude_rad = math.radians(OBSERVATORY_LATITUDE_DEG)
    sidereal_rad = math.radians(local_sidereal_deg)
    denominator = math.sqrt(1.0 - EARTH_ECCENTRICITY**2 * math.sin(latitude_rad) ** 2)
    equatorial = (EARTH_RADIUS_KM / denominator + OBSERVATORY_HEIGHT_KM) * math.cos(latitude_rad)
    polar = (
        EARTH_RADIUS_KM * (1.0 - EARTH_ECCENTRICITY**2) / denominator + OBSERVATORY_HEIGHT_KM
    ) * math.sin(latitude_rad)
    return np.array([equatorial * math.cos(sidereal_rad), equatorial * math.sin(sidereal_rad), polar])


def line_of_sight(right_ascension_rad: float, declination_rad: float) -> np.ndarray:
    return np.array(
        [
            math.cos(declination_rad) * math.cos(right_ascension_rad),
            math.cos(declination_rad) * math.sin(right_ascension_rad),
            math.sin(declination_rad),
        ]
    )


def results_to_dataframe(results: list[GaussIODResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "triplet": result.triplet_index,
                "time_mjd": result.time_mjd,
                "rx_km": result.position_km[0],
                "ry_km": result.position_km[1],
                "rz_km": result.position_km[2],
                "vx_km_s": result.velocity_km_s[0],
                "vy_km_s": result.velocity_km_s[1],
                "vz_km_s": result.velocity_km_s[2],
                "range_km": result.range_km,
            }
        )
    return pd.DataFrame(rows)


def _prepare_triplets(observations: pd.DataFrame) -> pd.DataFrame:
    data = observations.reset_index(drop=True)
    if len(data) % 3 != 0 and len(data) > 1 and (len(data) - 1) % 3 == 0:
        data = data.iloc[1:].reset_index(drop=True)
    usable_count = (len(data) // 3) * 3
    if usable_count == 0:
        raise ValueError("Need at least three observations for Gauss IOD")
    return data.iloc[:usable_count].reset_index(drop=True)
