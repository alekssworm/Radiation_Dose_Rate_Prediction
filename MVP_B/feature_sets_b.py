"""Official feature-set registry for MVP-B.

This module defines the controlled shortlist of feature sets allowed in MVP-B.
It is intentionally conservative: MVP-B is not an open-ended exploration stage,
but a reproducible engineering stage built on top of MVP-A findings.

Design goals:
- keep feature-set definitions explicit and versionable;
- make mode resolution deterministic (`env_mode` vs `nuclide_mode`);
- provide a stable API for training, evaluation, and inference code;
- fail fast if the registry becomes internally inconsistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple


# -----------------------------------------------------------------------------
# Core column names
# -----------------------------------------------------------------------------

TARGET_COLUMN: str = "target_dose_rate"
SECONDARY_TARGET_COLUMN: str = "target_dose_rate_0_1m"
ID_COLUMNS: Tuple[str, ...] = ("Code",)
COORD_COLUMNS: Tuple[str, ...] = ("latitude", "longitude")

EXCLUDED_COLUMNS: Tuple[str, ...] = (
    *ID_COLUMNS,
    *COORD_COLUMNS,
    SECONDARY_TARGET_COLUMN,
)


# -----------------------------------------------------------------------------
# Base feature blocks
# -----------------------------------------------------------------------------

ENV_BASE_FEATURES: Tuple[str, ...] = (
    "organic_carbon_b0",
    "organic_carbon_b10",
    "clay_fraction_0_30",
    "clay_fraction_30_60",
    "sand_fraction_b0",
    "sand_fraction_b10",
    "bulk_density_b0",
    "bulk_density_b10",
    "soil_pH_b0",
    "soil_pH_b10",
    "elevation_m",
    "slope_deg_final",
    "twi_scaled",
)

CONTAMINATION_FEATURES: Tuple[str, ...] = (
    "cs137_kBq_m2",
    "sr90_kBq_m2",
    "ratio_cs_sr",
)

CONTAMINATION_NO_RATIO_FEATURES: Tuple[str, ...] = (
    "cs137_kBq_m2",
    "sr90_kBq_m2",
)

NATURAL_NUCLIDE_FEATURES: Tuple[str, ...] = (
    "k40_Bq_kg",
    "ra226_Bq_kg",
    "th232_Bq_kg",
)

FULL_NUCLIDE_FEATURES: Tuple[str, ...] = (
    *CONTAMINATION_FEATURES,
    *NATURAL_NUCLIDE_FEATURES,
)

FULL_NUCLIDE_NO_RATIO_FEATURES: Tuple[str, ...] = (
    *CONTAMINATION_NO_RATIO_FEATURES,
    *NATURAL_NUCLIDE_FEATURES,
)

NUCLIDE_FEATURE_UNIVERSE: Tuple[str, ...] = (
    *FULL_NUCLIDE_FEATURES,
)


# -----------------------------------------------------------------------------
# Modes
# -----------------------------------------------------------------------------

ENV_MODE: str = "env_mode"
NUCLIDE_MODE: str = "nuclide_mode"


# -----------------------------------------------------------------------------
# Feature-set specification
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureSetSpec:
    """Immutable specification of an allowed feature set.

    Attributes:
        name: Stable registry key used across training/evaluation artifacts.
        features: Ordered feature list expected by downstream pipelines.
        mode: Mode required to run this feature set.
        description: Human-readable explanation for reports and metadata.
    """

    name: str
    features: Tuple[str, ...]
    mode: str
    description: str

    @property
    def n_features(self) -> int:
        return len(self.features)

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "features": list(self.features),
            "mode": self.mode,
            "description": self.description,
            "n_features": self.n_features,
        }


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

FEATURE_SET_REGISTRY: Mapping[str, FeatureSetSpec] = {
    "env_only": FeatureSetSpec(
        name="env_only",
        features=ENV_BASE_FEATURES,
        mode=ENV_MODE,
        description="Environment-only reference feature set: soil + terrain.",
    ),
    "env_plus_cs137_only": FeatureSetSpec(
        name="env_plus_cs137_only",
        features=(*ENV_BASE_FEATURES, "cs137_kBq_m2"),
        mode=NUCLIDE_MODE,
        description="Environment features plus Cs-137 contamination signal only.",
    ),
    "env_plus_cs137_sr90": FeatureSetSpec(
        name="env_plus_cs137_sr90",
        features=(*ENV_BASE_FEATURES, "cs137_kBq_m2", "sr90_kBq_m2"),
        mode=NUCLIDE_MODE,
        description="Environment features plus Cs-137 and Sr-90 contamination signals.",
    ),
    "env_plus_contamination_only": FeatureSetSpec(
        name="env_plus_contamination_only",
        features=(*ENV_BASE_FEATURES, *CONTAMINATION_FEATURES),
        mode=NUCLIDE_MODE,
        description="Environment features plus contamination-related nuclides only.",
    ),
    "env_plus_full_nuclide": FeatureSetSpec(
        name="env_plus_full_nuclide",
        features=(*ENV_BASE_FEATURES, *FULL_NUCLIDE_FEATURES),
        mode=NUCLIDE_MODE,
        description="Environment features plus contamination and natural nuclides.",
    ),
    "env_plus_no_ratio": FeatureSetSpec(
        name="env_plus_no_ratio",
        features=(*ENV_BASE_FEATURES, *FULL_NUCLIDE_NO_RATIO_FEATURES),
        mode=NUCLIDE_MODE,
        description="Environment features plus Cs-137, Sr-90, and natural nuclides without Cs/Sr ratio.",
    ),
}


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def list_feature_sets() -> List[str]:
    """Return all registered feature-set names in insertion order."""
    return list(FEATURE_SET_REGISTRY.keys())



def get_feature_set_spec(name: str) -> FeatureSetSpec:
    """Return the immutable spec for a registered feature set.

    Raises:
        KeyError: if the feature set name is unknown.
    """
    try:
        return FEATURE_SET_REGISTRY[name]
    except KeyError as exc:
        allowed = ", ".join(list_feature_sets())
        raise KeyError(f"Unknown feature set: {name!r}. Allowed values: {allowed}") from exc



def get_feature_columns(name: str) -> List[str]:
    """Return feature columns for a registered feature set as a new list."""
    return list(get_feature_set_spec(name).features)



def get_feature_mode(name: str) -> str:
    """Return required mode for a registered feature set."""
    return get_feature_set_spec(name).mode



def get_feature_registry_dict() -> Dict[str, List[str]]:
    """Return the registry as a plain dict[str, list[str]] for serialization."""
    return {name: list(spec.features) for name, spec in FEATURE_SET_REGISTRY.items()}



def get_specs_as_dict() -> Dict[str, Dict[str, object]]:
    """Return fully serializable feature-set specs."""
    return {name: spec.to_dict() for name, spec in FEATURE_SET_REGISTRY.items()}



def resolve_feature_set(name_or_features: str | Iterable[str]) -> FeatureSetSpec:
    """Resolve either a registered name or an explicit feature iterable.

    This is useful when training code wants to support named feature sets while
    still allowing internal comparisons against explicit lists.

    For explicit iterables, the mode is inferred automatically.
    """
    if isinstance(name_or_features, str):
        return get_feature_set_spec(name_or_features)

    features = tuple(name_or_features)
    mode = infer_mode_from_features(features)
    return FeatureSetSpec(
        name="custom",
        features=features,
        mode=mode,
        description="Custom explicit feature set.",
    )



def infer_mode_from_features(features: Iterable[str]) -> str:
    """Infer model mode from a feature iterable.

    Any feature set containing one or more nuclide features is treated as
    `nuclide_mode`. Otherwise it is treated as `env_mode`.
    """
    feature_set = set(features)
    if feature_set.intersection(NUCLIDE_FEATURE_UNIVERSE):
        return NUCLIDE_MODE
    return ENV_MODE



def is_env_only_feature_set(name: str) -> bool:
    """Return True if the named feature set belongs to env-only mode."""
    return get_feature_mode(name) == ENV_MODE



def is_nuclide_feature_set(name: str) -> bool:
    """Return True if the named feature set belongs to nuclide mode."""
    return get_feature_mode(name) == NUCLIDE_MODE



def required_columns_for_feature_set(name: str, include_target: bool = True) -> List[str]:
    """Return required columns for training/evaluation for a given feature set.

    By default, this includes the target column because most training and
    evaluation workflows require it.
    """
    columns = get_feature_columns(name)
    if include_target:
        return [TARGET_COLUMN, *columns]
    return columns



def required_mode_columns(mode: str, include_target: bool = False) -> List[str]:
    """Return the superset of required columns for a mode.

    For `nuclide_mode`, this returns the full nuclide-capable feature universe.
    For `env_mode`, this returns the environment-only base features.
    """
    if mode == ENV_MODE:
        columns = list(ENV_BASE_FEATURES)
    elif mode == NUCLIDE_MODE:
        columns = list((*ENV_BASE_FEATURES, *FULL_NUCLIDE_FEATURES))
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Expected {ENV_MODE!r} or {NUCLIDE_MODE!r}.")

    if include_target:
        return [TARGET_COLUMN, *columns]
    return columns


# -----------------------------------------------------------------------------
# Internal validation
# -----------------------------------------------------------------------------


def _assert_unique_feature_order(features: Tuple[str, ...], feature_set_name: str) -> None:
    duplicates = [col for idx, col in enumerate(features) if col in features[:idx]]
    if duplicates:
        dup_str = ", ".join(duplicates)
        raise ValueError(f"Feature set {feature_set_name!r} contains duplicate columns: {dup_str}")



def _assert_no_excluded_columns(features: Tuple[str, ...], feature_set_name: str) -> None:
    forbidden = set(features).intersection(EXCLUDED_COLUMNS)
    if forbidden:
        forbidden_str = ", ".join(sorted(forbidden))
        raise ValueError(
            f"Feature set {feature_set_name!r} contains excluded columns: {forbidden_str}"
        )



def _assert_mode_consistency(spec: FeatureSetSpec) -> None:
    inferred_mode = infer_mode_from_features(spec.features)
    if spec.mode != inferred_mode:
        raise ValueError(
            f"Feature set {spec.name!r} has inconsistent mode. "
            f"Declared={spec.mode!r}, inferred={inferred_mode!r}."
        )



def validate_feature_registry() -> None:
    """Validate registry integrity.

    This should run at import time because registry corruption is a developer
    error and should fail fast.
    """
    if not FEATURE_SET_REGISTRY:
        raise ValueError("FEATURE_SET_REGISTRY must not be empty.")

    for name, spec in FEATURE_SET_REGISTRY.items():
        if spec.name != name:
            raise ValueError(
                f"Registry key/spec mismatch: key={name!r}, spec.name={spec.name!r}"
            )
        if not spec.features:
            raise ValueError(f"Feature set {name!r} must contain at least one feature.")

        _assert_unique_feature_order(spec.features, name)
        _assert_no_excluded_columns(spec.features, name)
        _assert_mode_consistency(spec)


validate_feature_registry()


__all__ = [
    "TARGET_COLUMN",
    "SECONDARY_TARGET_COLUMN",
    "ID_COLUMNS",
    "COORD_COLUMNS",
    "EXCLUDED_COLUMNS",
    "ENV_BASE_FEATURES",
    "CONTAMINATION_FEATURES",
    "CONTAMINATION_NO_RATIO_FEATURES",
    "NATURAL_NUCLIDE_FEATURES",
    "FULL_NUCLIDE_FEATURES",
    "FULL_NUCLIDE_NO_RATIO_FEATURES",
    "NUCLIDE_FEATURE_UNIVERSE",
    "ENV_MODE",
    "NUCLIDE_MODE",
    "FeatureSetSpec",
    "FEATURE_SET_REGISTRY",
    "list_feature_sets",
    "get_feature_set_spec",
    "get_feature_columns",
    "get_feature_mode",
    "get_feature_registry_dict",
    "get_specs_as_dict",
    "resolve_feature_set",
    "infer_mode_from_features",
    "is_env_only_feature_set",
    "is_nuclide_feature_set",
    "required_columns_for_feature_set",
    "required_mode_columns",
    "validate_feature_registry",
]
