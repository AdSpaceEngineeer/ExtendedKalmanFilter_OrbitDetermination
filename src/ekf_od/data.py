from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "right_ascension_deg": {"right_ascension_deg", "right ascension deg", "ra_deg", "ra"},
    "declination_deg": {"declination_deg", "declination deg", "dec_deg", "declination"},
    "local_sidereal_deg": {"local sidereal", "local_sidereal", "local_sidereal_deg", "lst_deg"},
    "time_mjd": {"time_mjd", "time mjd", "mjd"},
}


def load_observations(path: str | Path) -> pd.DataFrame:
    """Load optical observations from the original workbook or a CSV export."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path)

    normalized = {_normalize_column(column): column for column in raw.columns}
    selected: dict[str, pd.Series] = {}
    for canonical, aliases in REQUIRED_COLUMNS.items():
        match = next((normalized[name] for name in aliases if name in normalized), None)
        if match is None:
            raise ValueError(
                f"Missing required observation column for {canonical!r}. "
                f"Available columns: {list(raw.columns)!r}"
            )
        selected[canonical] = pd.to_numeric(raw[match], errors="coerce")

    observations = pd.DataFrame(selected).dropna().reset_index(drop=True)
    if observations.empty:
        raise ValueError(f"No numeric observations found in {path}")
    return observations


def _normalize_column(column: object) -> str:
    return str(column).strip().lower().replace("_", " ")
