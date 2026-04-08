"""Spatial cross-validation utilities for MVP-B.

This module centralizes all spatial grouping logic used by training and
validation code. In MVP-B, spatial validation is not an optional experiment:
it is the default evaluation mode.

Design goals:
- deterministic block-based spatial grouping;
- explicit and validated coordinate handling;
- clean integration with sklearn GroupKFold workflows;
- lightweight diagnostics for group and fold structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


# -----------------------------------------------------------------------------
# Default coordinate settings
# -----------------------------------------------------------------------------

DEFAULT_LAT_COL: str = "latitude"
DEFAULT_LON_COL: str = "longitude"
DEFAULT_BLOCK_SIZE_DEG: float = 0.02
DEFAULT_N_SPLITS: int = 5


# -----------------------------------------------------------------------------
# Specifications and diagnostics
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SpatialCVSpec:
    """Specification of a block-based spatial CV setup.

    Attributes:
        lat_col: Latitude column name.
        lon_col: Longitude column name.
        block_size_deg: Spatial block size in decimal degrees.
        n_splits: Number of GroupKFold splits.
        lat_origin: Optional origin used for stable block indexing.
        lon_origin: Optional origin used for stable block indexing.
        group_column: Name of the produced group column.
    """

    lat_col: str = DEFAULT_LAT_COL
    lon_col: str = DEFAULT_LON_COL
    block_size_deg: float = DEFAULT_BLOCK_SIZE_DEG
    n_splits: int = DEFAULT_N_SPLITS
    lat_origin: float | None = None
    lon_origin: float | None = None
    group_column: str = "spatial_group"

    def to_dict(self) -> Dict[str, object]:
        return {
            "lat_col": self.lat_col,
            "lon_col": self.lon_col,
            "block_size_deg": self.block_size_deg,
            "n_splits": self.n_splits,
            "lat_origin": self.lat_origin,
            "lon_origin": self.lon_origin,
            "group_column": self.group_column,
        }


@dataclass(frozen=True)
class SpatialFoldSpec:
    """Metadata describing one spatial CV fold."""

    fold_index: int
    n_train_rows: int
    n_test_rows: int
    n_train_groups: int
    n_test_groups: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "fold_index": self.fold_index,
            "n_train_rows": self.n_train_rows,
            "n_test_rows": self.n_test_rows,
            "n_train_groups": self.n_train_groups,
            "n_test_groups": self.n_test_groups,
        }


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------


def validate_spatial_cv_spec(spec: SpatialCVSpec) -> None:
    """Validate a spatial CV specification.

    Raises:
        ValueError: if the specification is invalid.
    """
    if not spec.lat_col:
        raise ValueError("lat_col must be a non-empty string.")
    if not spec.lon_col:
        raise ValueError("lon_col must be a non-empty string.")
    if spec.block_size_deg <= 0:
        raise ValueError("block_size_deg must be > 0.")
    if spec.n_splits < 2:
        raise ValueError("n_splits must be >= 2.")
    if not spec.group_column:
        raise ValueError("group_column must be a non-empty string.")



def validate_coordinate_columns(
    df: pd.DataFrame,
    lat_col: str = DEFAULT_LAT_COL,
    lon_col: str = DEFAULT_LON_COL,
    allow_nulls: bool = False,
) -> None:
    """Validate presence and basic integrity of coordinate columns.

    Raises:
        KeyError: if required columns are missing.
        ValueError: if coordinates contain nulls or non-numeric values.
    """
    missing = [col for col in (lat_col, lon_col) if col not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(f"Missing required coordinate columns: {missing_str}")

    for col in (lat_col, lon_col):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Coordinate column {col!r} must be numeric.")

    if not allow_nulls:
        null_cols = [col for col in (lat_col, lon_col) if df[col].isna().any()]
        if null_cols:
            null_str = ", ".join(null_cols)
            raise ValueError(f"Coordinate columns must not contain nulls: {null_str}")


# -----------------------------------------------------------------------------
# Block indexing
# -----------------------------------------------------------------------------


def infer_block_origin(
    df: pd.DataFrame,
    lat_col: str = DEFAULT_LAT_COL,
    lon_col: str = DEFAULT_LON_COL,
) -> Tuple[float, float]:
    """Infer a deterministic block origin from the minimum coordinates.

    This produces stable block IDs for a given dataset when the same origin is
    reused across runs.
    """
    validate_coordinate_columns(df, lat_col=lat_col, lon_col=lon_col, allow_nulls=False)
    return float(df[lat_col].min()), float(df[lon_col].min())



def compute_block_indices(
    values: Sequence[float] | pd.Series | np.ndarray,
    block_size_deg: float,
    origin: float | None = None,
) -> np.ndarray:
    """Convert coordinate values into integer block indices.

    The formula is:
        floor((value - origin) / block_size_deg)

    If origin is not provided, 0.0 is used.
    """
    if block_size_deg <= 0:
        raise ValueError("block_size_deg must be > 0.")

    values_array = np.asarray(values, dtype=float)
    if np.isnan(values_array).any():
        raise ValueError("Coordinate values must not contain NaN.")

    effective_origin = 0.0 if origin is None else float(origin)
    return np.floor((values_array - effective_origin) / block_size_deg).astype(int)



def make_spatial_group_labels(
    df: pd.DataFrame,
    block_size_deg: float = DEFAULT_BLOCK_SIZE_DEG,
    lat_col: str = DEFAULT_LAT_COL,
    lon_col: str = DEFAULT_LON_COL,
    lat_origin: float | None = None,
    lon_origin: float | None = None,
) -> pd.Series:
    """Create deterministic block-based spatial group labels.

    Group labels are strings of the form:
        "latBlock_lonBlock"

    This keeps them easy to inspect in CSV outputs and diagnostics.
    """
    validate_coordinate_columns(df, lat_col=lat_col, lon_col=lon_col, allow_nulls=False)

    lat_idx = compute_block_indices(
        values=df[lat_col].to_numpy(),
        block_size_deg=block_size_deg,
        origin=lat_origin,
    )
    lon_idx = compute_block_indices(
        values=df[lon_col].to_numpy(),
        block_size_deg=block_size_deg,
        origin=lon_origin,
    )

    labels = pd.Series(
        [f"{lat_block}_{lon_block}" for lat_block, lon_block in zip(lat_idx, lon_idx)],
        index=df.index,
        name="spatial_group",
    )
    return labels



def add_spatial_groups(
    df: pd.DataFrame,
    spec: SpatialCVSpec,
    copy: bool = True,
) -> pd.DataFrame:
    """Attach spatial group labels to a dataframe using a SpatialCVSpec."""
    validate_spatial_cv_spec(spec)
    group_labels = make_spatial_group_labels(
        df=df,
        block_size_deg=spec.block_size_deg,
        lat_col=spec.lat_col,
        lon_col=spec.lon_col,
        lat_origin=spec.lat_origin,
        lon_origin=spec.lon_origin,
    )

    out = df.copy() if copy else df
    out[spec.group_column] = group_labels.values
    return out



def get_spatial_groups(df: pd.DataFrame, spec: SpatialCVSpec) -> pd.Series:
    """Return spatial group labels as a Series aligned to the dataframe index."""
    validate_spatial_cv_spec(spec)
    groups = make_spatial_group_labels(
        df=df,
        block_size_deg=spec.block_size_deg,
        lat_col=spec.lat_col,
        lon_col=spec.lon_col,
        lat_origin=spec.lat_origin,
        lon_origin=spec.lon_origin,
    )
    groups.name = spec.group_column
    return groups


# -----------------------------------------------------------------------------
# GroupKFold helpers
# -----------------------------------------------------------------------------


def build_group_kfold(n_splits: int = DEFAULT_N_SPLITS) -> GroupKFold:
    """Build a GroupKFold splitter."""
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2.")
    return GroupKFold(n_splits=n_splits)



def iter_spatial_splits(
    X: pd.DataFrame | np.ndarray,
    groups: Sequence[object] | pd.Series | np.ndarray,
    n_splits: int = DEFAULT_N_SPLITS,
) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """Yield `(fold_index, train_idx, test_idx)` for GroupKFold splits."""
    groups_array = np.asarray(groups)
    n_unique_groups = np.unique(groups_array).size
    if n_unique_groups < n_splits:
        raise ValueError(
            f"Not enough unique spatial groups for GroupKFold: "
            f"n_unique_groups={n_unique_groups}, n_splits={n_splits}."
        )

    splitter = build_group_kfold(n_splits=n_splits)
    dummy_y = np.zeros(len(groups_array), dtype=float)

    for fold_index, (train_idx, test_idx) in enumerate(
        splitter.split(X=X, y=dummy_y, groups=groups_array)
    ):
        yield fold_index, train_idx, test_idx



def build_spatial_fold_specs(
    X: pd.DataFrame | np.ndarray,
    groups: Sequence[object] | pd.Series | np.ndarray,
    n_splits: int = DEFAULT_N_SPLITS,
) -> List[SpatialFoldSpec]:
    """Return fold diagnostics for a spatial CV setup."""
    groups_array = np.asarray(groups)
    specs: List[SpatialFoldSpec] = []

    for fold_index, train_idx, test_idx in iter_spatial_splits(X=X, groups=groups_array, n_splits=n_splits):
        train_groups = np.unique(groups_array[train_idx])
        test_groups = np.unique(groups_array[test_idx])
        specs.append(
            SpatialFoldSpec(
                fold_index=fold_index,
                n_train_rows=int(len(train_idx)),
                n_test_rows=int(len(test_idx)),
                n_train_groups=int(len(train_groups)),
                n_test_groups=int(len(test_groups)),
            )
        )

    return specs


# -----------------------------------------------------------------------------
# Diagnostics and summaries
# -----------------------------------------------------------------------------


def summarize_spatial_groups(
    df: pd.DataFrame,
    groups: Sequence[object] | pd.Series | np.ndarray,
    lat_col: str = DEFAULT_LAT_COL,
    lon_col: str = DEFAULT_LON_COL,
) -> pd.DataFrame:
    """Summarize row counts and coordinate spans for each spatial group."""
    validate_coordinate_columns(df, lat_col=lat_col, lon_col=lon_col, allow_nulls=False)

    groups_series = pd.Series(groups, index=df.index, name="spatial_group")
    work_df = df[[lat_col, lon_col]].copy()
    work_df["spatial_group"] = groups_series

    summary = (
        work_df.groupby("spatial_group", dropna=False)
        .agg(
            n_rows=(lat_col, "size"),
            lat_min=(lat_col, "min"),
            lat_max=(lat_col, "max"),
            lon_min=(lon_col, "min"),
            lon_max=(lon_col, "max"),
        )
        .reset_index()
        .sort_values(["n_rows", "spatial_group"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return summary



def summarize_spatial_cv_setup(
    X: pd.DataFrame,
    spec: SpatialCVSpec,
) -> Dict[str, object]:
    """Return a compact summary of the spatial CV configuration and groups."""
    groups = get_spatial_groups(X, spec)
    unique_groups = groups.nunique(dropna=False)
    fold_specs = build_spatial_fold_specs(X=X, groups=groups, n_splits=spec.n_splits)

    return {
        "spec": spec.to_dict(),
        "n_rows": int(len(X)),
        "n_unique_groups": int(unique_groups),
        "n_splits": int(spec.n_splits),
        "folds": [fold.to_dict() for fold in fold_specs],
    }



def fold_specs_to_frame(fold_specs: Sequence[SpatialFoldSpec]) -> pd.DataFrame:
    """Convert fold specs to a DataFrame for logging or CSV export."""
    return pd.DataFrame([fold.to_dict() for fold in fold_specs])


# -----------------------------------------------------------------------------
# Convenience helpers
# -----------------------------------------------------------------------------


def make_default_spatial_spec(
    block_size_deg: float = DEFAULT_BLOCK_SIZE_DEG,
    n_splits: int = DEFAULT_N_SPLITS,
    lat_col: str = DEFAULT_LAT_COL,
    lon_col: str = DEFAULT_LON_COL,
) -> SpatialCVSpec:
    """Build a default SpatialCVSpec with explicit overrides."""
    spec = SpatialCVSpec(
        lat_col=lat_col,
        lon_col=lon_col,
        block_size_deg=block_size_deg,
        n_splits=n_splits,
    )
    validate_spatial_cv_spec(spec)
    return spec



def resolve_spatial_spec(
    spec: SpatialCVSpec | None = None,
    *,
    block_size_deg: float = DEFAULT_BLOCK_SIZE_DEG,
    n_splits: int = DEFAULT_N_SPLITS,
    lat_col: str = DEFAULT_LAT_COL,
    lon_col: str = DEFAULT_LON_COL,
    lat_origin: float | None = None,
    lon_origin: float | None = None,
    group_column: str = "spatial_group",
) -> SpatialCVSpec:
    """Return a validated SpatialCVSpec from either an explicit spec or kwargs."""
    resolved = spec or SpatialCVSpec(
        lat_col=lat_col,
        lon_col=lon_col,
        block_size_deg=block_size_deg,
        n_splits=n_splits,
        lat_origin=lat_origin,
        lon_origin=lon_origin,
        group_column=group_column,
    )
    validate_spatial_cv_spec(resolved)
    return resolved


__all__ = [
    "DEFAULT_LAT_COL",
    "DEFAULT_LON_COL",
    "DEFAULT_BLOCK_SIZE_DEG",
    "DEFAULT_N_SPLITS",
    "SpatialCVSpec",
    "SpatialFoldSpec",
    "validate_spatial_cv_spec",
    "validate_coordinate_columns",
    "infer_block_origin",
    "compute_block_indices",
    "make_spatial_group_labels",
    "add_spatial_groups",
    "get_spatial_groups",
    "build_group_kfold",
    "iter_spatial_splits",
    "build_spatial_fold_specs",
    "summarize_spatial_groups",
    "summarize_spatial_cv_setup",
    "fold_specs_to_frame",
    "make_default_spatial_spec",
    "resolve_spatial_spec",
]
