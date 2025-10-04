"""Utilities for loading and preprocessing exoplanet candidate datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

# Columns used as model features. These were selected to cover orbital,
# planetary, and stellar properties that are common across the TESS, Kepler,
# and K2 catalogues.
FEATURE_COLUMNS: List[str] = [
    "pl_orbper",
    "pl_orbsmax",
    "pl_rade",
    "pl_bmasse",
    "pl_eqt",
    "pl_insol",
    "st_teff",
    "st_rad",
    "st_mass",
    "st_met",
    "st_logg",
]

LABEL_COLUMN = "label"
DEFAULT_TARGET_CANDIDATES: Sequence[str] = (
    "tfopwg_disp",
    "disposition",
    "koi_disposition",
    "koi_pdisposition",
    "label",
)


def _resolve_data_dir(data_dir: Optional[Path] = None) -> Path:
    """Return the directory that stores the raw CSV files."""

    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "data"
    return data_dir


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame using UTF-8 encoding."""

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path, comment="#", skipinitialspace=True)


def _normalise_target(df: pd.DataFrame, candidate_names: Iterable[str]) -> pd.DataFrame:
    """Rename the first existing column from *candidate_names* to `label`."""

    for name in candidate_names:
        if name in df.columns:
            return df.rename(columns={name: LABEL_COLUMN})
    raise ValueError(
        "None of the target columns were found in the dataset. Expected one of: "
        + ", ".join(candidate_names)
    )


def _coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Ensure the requested columns exist and are numeric."""

    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _fill_numeric_gaps(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Fill missing numeric values with the column median (fallback to 0)."""

    for column in columns:
        if df[column].notna().any():
            median = df[column].median()
        else:
            median = 0.0
        if pd.isna(median):
            median = 0.0
        df[column] = df[column].fillna(median)
    return df


def prepare_dataframe(
    df: pd.DataFrame,
    target_candidates: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return a cleaned frame with unified label column and numeric features."""

    candidates = target_candidates or DEFAULT_TARGET_CANDIDATES
    df = _normalise_target(df, candidates)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].replace("", pd.NA)
    df = df.dropna(subset=[LABEL_COLUMN])

    df = _coerce_numeric_columns(df, FEATURE_COLUMNS)
    selection = FEATURE_COLUMNS + [LABEL_COLUMN]
    df = df[selection]
    df = _fill_numeric_gaps(df, FEATURE_COLUMNS)
    return df




def prepare_features_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature-only frame suitable for inference."""

    df = df.copy()
    df = _coerce_numeric_columns(df, FEATURE_COLUMNS)
    df = df[FEATURE_COLUMNS]
    df = _fill_numeric_gaps(df, FEATURE_COLUMNS)
    return df

def load_datasets(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load, align, and merge the TESS, Kepler, and K2 catalogues."""

    data_dir = _resolve_data_dir(data_dir)

    datasets_config = [
        ("TOI_2025.10.04_09.55.27.csv", ("tfopwg_disp",)),
        ("cumulative_2025.10.04_09.55.00.csv", ("disposition", "koi_disposition")),
        ("k2pandc_2025.10.04_09.55.15.csv", ("disposition", "koi_disposition", "koi_pdisposition")),
    ]

    combined_frames: List[pd.DataFrame] = []

    for filename, target_cols in datasets_config:
        dataset_path = data_dir / filename
        df = _read_csv(dataset_path)
        df = prepare_dataframe(df, target_cols)
        combined_frames.append(df)

    combined_df = pd.concat(combined_frames, ignore_index=True, sort=False)
    return combined_df




def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _derive_year_column(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    for column in candidates:
        if column in df.columns:
            series = df[column]
            if pd.api.types.is_numeric_dtype(series):  # type: ignore[attr-defined]
                numeric = pd.to_numeric(series, errors="coerce")
                if numeric.notna().any():
                    return numeric.round().astype('Int64')
            converted = pd.to_datetime(series, errors="coerce")
            if converted.notna().any():
                return converted.dt.year.astype('Int64')
    return pd.Series(pd.NA, index=df.index, dtype='Int64')


STAR_MAP_DATASETS = [
    {
        "filename": "TOI_2025.10.04_09.55.27.csv",
        "label_candidates": ("tfopwg_disp",),
        "ra_candidates": ("ra",),
        "dec_candidates": ("dec",),
        "radius_candidates": ("pl_rade",),
        "year_candidates": ("toi_created", "rowupdate"),
        "name_candidates": ("toi", "tid"),
        "dataset": "TESS TOI",
    },
    {
        "filename": "cumulative_2025.10.04_09.55.00.csv",
        "label_candidates": ("disposition", "koi_disposition"),
        "ra_candidates": ("ra",),
        "dec_candidates": ("dec",),
        "radius_candidates": ("pl_rade", "koi_prad"),
        "year_candidates": ("rowupdate", "koi_time0bk"),
        "name_candidates": ("kepler_name", "kepoi_name", "kepid"),
        "dataset": "Kepler Cumulative",
    },
    {
        "filename": "k2pandc_2025.10.04_09.55.15.csv",
        "label_candidates": ("disposition", "koi_disposition", "koi_pdisposition"),
        "ra_candidates": ("ra",),
        "dec_candidates": ("dec",),
        "radius_candidates": ("pl_rade",),
        "year_candidates": ("disc_year", "pl_pubdate", "rowupdate"),
        "name_candidates": ("pl_name", "hostname"),
        "dataset": "K2",
    },
]


def load_star_map_data(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Return RA/Dec data suitable for star map visualisation."""

    data_dir = _resolve_data_dir(data_dir)
    frames: List[pd.DataFrame] = []

    for config in STAR_MAP_DATASETS:
        dataset_path = data_dir / config["filename"]
        if not dataset_path.exists():
            continue
        df = _read_csv(dataset_path)

        try:
            df = _normalise_target(df, config["label_candidates"])
        except ValueError:
            df[LABEL_COLUMN] = pd.NA

        ra_col = _first_existing_column(df, config["ra_candidates"])
        dec_col = _first_existing_column(df, config["dec_candidates"])
        radius_col = _first_existing_column(df, config["radius_candidates"])
        name_col = _first_existing_column(df, config["name_candidates"])

        if ra_col is None or dec_col is None:
            continue

        subset = pd.DataFrame({
            "dataset": config["dataset"],
            "ra": pd.to_numeric(df[ra_col], errors="coerce"),
            "dec": pd.to_numeric(df[dec_col], errors="coerce"),
            "planet_radius": pd.to_numeric(df[radius_col], errors="coerce") if radius_col else pd.Series(pd.NA, index=df.index),
            LABEL_COLUMN: df.get(LABEL_COLUMN, pd.NA),
        })

        subset["identifier"] = df[name_col] if name_col else pd.NA
        subset["discovery_year"] = _derive_year_column(df, config["year_candidates"])

        subset = subset.dropna(subset=["ra", "dec"])
        frames.append(subset)

    if not frames:
        return pd.DataFrame(columns=["dataset", "ra", "dec", "planet_radius", LABEL_COLUMN, "identifier", "discovery_year"])

    return pd.concat(frames, ignore_index=True, sort=False)

def split_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) tuple ready for model training."""

    features = df[FEATURE_COLUMNS]
    target = df[LABEL_COLUMN]
    return features, target
