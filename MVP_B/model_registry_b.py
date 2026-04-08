"""Official model registry for MVP-B.

This module defines the controlled shortlist of model families allowed in MVP-B,
along with their default configurations, compact tuning spaces, and factory
functions for estimator construction.

MVP-B is a controlled engineering stage. The model registry is intentionally
small and explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Global defaults
# -----------------------------------------------------------------------------

RANDOM_STATE: int = 42
N_JOBS: int = -1

DEFAULT_RIDGE_ALPHAS: Tuple[float, ...] = tuple(np.logspace(-3, 3, 13))

DEFAULT_RF_PARAMS: Dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 8,
    "min_samples_leaf": 10,
    "max_features": 0.5,
    "random_state": RANDOM_STATE,
    "n_jobs": N_JOBS,
}

DEFAULT_ET_PARAMS: Dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 6,
    "min_samples_leaf": 2,
    "max_features": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": N_JOBS,
}

RF_SEARCH_SPACE: Mapping[str, Tuple[Any, ...]] = {
    "n_estimators": (500,),
    "max_depth": (8, 12, None),
    "min_samples_leaf": (1, 3, 5, 10),
    "max_features": ("sqrt", 0.5),
}

ET_SEARCH_SPACE: Mapping[str, Tuple[Any, ...]] = {
    "n_estimators": (500,),
    "max_depth": (6, 8, 12, None),
    "min_samples_leaf": (1, 2, 3, 5, 10),
    "max_features": (1.0, "sqrt", 0.5),
}

RIDGE_SEARCH_SPACE: Mapping[str, Tuple[Any, ...]] = {
    "alphas": (DEFAULT_RIDGE_ALPHAS,),
}


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """Immutable specification of an allowed MVP-B model family.

    Attributes:
        name: Stable model family key.
        estimator_family: High-level estimator family name.
        description: Human-readable explanation.
        requires_scaling: Whether the numeric pipeline requires scaling.
        supports_feature_importance: Whether the fitted model exposes feature importances.
        default_params: Default parameter set used for the reference configuration.
        search_space: Controlled tuning space for candidate generation.
    """

    name: str
    estimator_family: str
    description: str
    requires_scaling: bool
    supports_feature_importance: bool
    default_params: Mapping[str, Any]
    search_space: Mapping[str, Tuple[Any, ...]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "estimator_family": self.estimator_family,
            "description": self.description,
            "requires_scaling": self.requires_scaling,
            "supports_feature_importance": self.supports_feature_importance,
            "default_params": _to_serializable_dict(self.default_params),
            "search_space": _to_serializable_search_space(self.search_space),
        }


@dataclass(frozen=True)
class ModelCandidate:
    """A concrete candidate configuration derived from the registry."""

    model_name: str
    params: Mapping[str, Any]
    candidate_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "params": _to_serializable_dict(self.params),
            "candidate_id": self.candidate_id,
        }


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

MODEL_REGISTRY: Mapping[str, ModelSpec] = {
    "ridge_cv": ModelSpec(
        name="ridge_cv",
        estimator_family="RidgeCV",
        description="Linear reference model with mandatory scaling and internal alpha selection.",
        requires_scaling=True,
        supports_feature_importance=False,
        default_params={"alphas": DEFAULT_RIDGE_ALPHAS},
        search_space=RIDGE_SEARCH_SPACE,
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        estimator_family="RandomForestRegressor",
        description="Tree ensemble baseline with controlled regularization for spatial robustness.",
        requires_scaling=False,
        supports_feature_importance=True,
        default_params=DEFAULT_RF_PARAMS,
        search_space=RF_SEARCH_SPACE,
    ),
    "extra_trees": ModelSpec(
        name="extra_trees",
        estimator_family="ExtraTreesRegressor",
        description="High-variance tree ensemble baseline with controlled regularization.",
        requires_scaling=False,
        supports_feature_importance=True,
        default_params=DEFAULT_ET_PARAMS,
        search_space=ET_SEARCH_SPACE,
    ),
}


# -----------------------------------------------------------------------------
# Public API: registry access
# -----------------------------------------------------------------------------


def list_model_names() -> List[str]:
    """Return all registered model family names in insertion order."""
    return list(MODEL_REGISTRY.keys())



def get_model_spec(name: str) -> ModelSpec:
    """Return a registered model specification.

    Raises:
        KeyError: if the model family is unknown.
    """
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        allowed = ", ".join(list_model_names())
        raise KeyError(f"Unknown model: {name!r}. Allowed values: {allowed}") from exc



def get_default_params(name: str) -> Dict[str, Any]:
    """Return a copy of the default params for a model family."""
    return dict(get_model_spec(name).default_params)



def get_search_space(name: str) -> Dict[str, Tuple[Any, ...]]:
    """Return a copy of the controlled search space for a model family."""
    return {k: tuple(v) for k, v in get_model_spec(name).search_space.items()}



def get_registry_as_dict() -> Dict[str, Dict[str, Any]]:
    """Return the full model registry as a serializable dictionary."""
    return {name: spec.to_dict() for name, spec in MODEL_REGISTRY.items()}



def supports_feature_importance(name: str) -> bool:
    """Return whether the given model family exposes feature importance."""
    return get_model_spec(name).supports_feature_importance



def requires_scaling(name: str) -> bool:
    """Return whether the given model family requires scaling."""
    return get_model_spec(name).requires_scaling


# -----------------------------------------------------------------------------
# Public API: model construction
# -----------------------------------------------------------------------------


def build_model(name: str, params: Mapping[str, Any] | None = None) -> Pipeline:
    """Build a fresh sklearn Pipeline for a registered model family.

    All MVP-B models use a `SimpleImputer(strategy="median")` first.
    Ridge additionally uses `StandardScaler` before the estimator.
    """
    merged_params = get_default_params(name)
    if params:
        merged_params.update(params)

    if name == "ridge_cv":
        return _build_ridge_cv_pipeline(merged_params)
    if name == "random_forest":
        return _build_random_forest_pipeline(merged_params)
    if name == "extra_trees":
        return _build_extra_trees_pipeline(merged_params)

    raise ValueError(f"Unsupported model family: {name!r}")



def build_default_model(name: str) -> Pipeline:
    """Build a pipeline from the registry default configuration."""
    return build_model(name=name, params=None)



def build_reference_models() -> Dict[str, Pipeline]:
    """Build all default registry models."""
    return {name: build_default_model(name) for name in list_model_names()}


# -----------------------------------------------------------------------------
# Public API: candidate generation
# -----------------------------------------------------------------------------


def build_candidate_grid(
    model_names: Iterable[str] | None = None,
    include_defaults: bool = True,
) -> List[ModelCandidate]:
    """Expand the controlled tuning spaces into concrete candidate configs.

    RidgeCV usually contributes one candidate because alpha selection happens
    internally. Tree models expand across compact, explicitly controlled grids.
    """
    selected_models = list(model_names) if model_names is not None else list_model_names()
    candidates: List[ModelCandidate] = []

    for model_name in selected_models:
        spec = get_model_spec(model_name)
        grid_candidates = _expand_search_space(model_name, spec.search_space)

        if include_defaults:
            default_candidate = ModelCandidate(
                model_name=model_name,
                params=dict(spec.default_params),
                candidate_id=_make_candidate_id(model_name, spec.default_params),
            )
            candidates.append(default_candidate)

        for params in grid_candidates:
            if include_defaults and _params_equal(params, spec.default_params):
                continue
            candidates.append(
                ModelCandidate(
                    model_name=model_name,
                    params=params,
                    candidate_id=_make_candidate_id(model_name, params),
                )
            )

    return candidates



def build_default_candidates(model_names: Iterable[str] | None = None) -> List[ModelCandidate]:
    """Return one default candidate per selected model family."""
    selected_models = list(model_names) if model_names is not None else list_model_names()
    return [
        ModelCandidate(
            model_name=name,
            params=get_default_params(name),
            candidate_id=_make_candidate_id(name, get_default_params(name)),
        )
        for name in selected_models
    ]


# -----------------------------------------------------------------------------
# Public API: estimator extraction
# -----------------------------------------------------------------------------


def extract_final_estimator(model: Pipeline) -> RegressorMixin:
    """Return the final sklearn estimator from a fitted or unfitted pipeline."""
    estimator = model.named_steps.get("model")
    if estimator is None:
        raise ValueError("Expected pipeline step named 'model' was not found.")
    return estimator


# -----------------------------------------------------------------------------
# Pipeline builders
# -----------------------------------------------------------------------------


def _build_ridge_cv_pipeline(params: Mapping[str, Any]) -> Pipeline:
    alphas = tuple(params.get("alphas", DEFAULT_RIDGE_ALPHAS))
    estimator = RidgeCV(alphas=alphas)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )



def _build_random_forest_pipeline(params: Mapping[str, Any]) -> Pipeline:
    estimator = RandomForestRegressor(**params)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )



def _build_extra_trees_pipeline(params: Mapping[str, Any]) -> Pipeline:
    estimator = ExtraTreesRegressor(**params)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _expand_search_space(model_name: str, search_space: Mapping[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    if not search_space:
        return []

    keys = list(search_space.keys())
    values_product = product(*(search_space[key] for key in keys))
    candidates: List[Dict[str, Any]] = []

    for value_tuple in values_product:
        params = {key: value for key, value in zip(keys, value_tuple)}

        if model_name in {"random_forest", "extra_trees"}:
            params["random_state"] = RANDOM_STATE
            params["n_jobs"] = N_JOBS

        candidates.append(params)

    return candidates



def _make_candidate_id(model_name: str, params: Mapping[str, Any]) -> str:
    parts = [model_name]
    for key in sorted(params.keys()):
        value = params[key]
        value_str = _candidate_value_to_str(value)
        parts.append(f"{key}={value_str}")
    return "__".join(parts)



def _candidate_value_to_str(value: Any) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) > 5:
            return f"seq{len(value)}"
        return "-".join(str(v) for v in value)
    if value is None:
        return "none"
    return str(value).replace(" ", "")



def _params_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return _normalize_params(left) == _normalize_params(right)



def _normalize_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            normalized[key] = tuple(value.tolist())
        elif isinstance(value, list):
            normalized[key] = tuple(value)
        else:
            normalized[key] = value
    return normalized



def _to_serializable_dict(params: Mapping[str, Any]) -> Dict[str, Any]:
    serializable: Dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, tuple):
            serializable[key] = list(value)
        else:
            serializable[key] = value
    return serializable



def _to_serializable_search_space(search_space: Mapping[str, Tuple[Any, ...]]) -> Dict[str, Any]:
    return {key: list(values) for key, values in search_space.items()}


# -----------------------------------------------------------------------------
# Internal validation
# -----------------------------------------------------------------------------


def validate_model_registry() -> None:
    """Fail fast if the model registry is inconsistent."""
    if not MODEL_REGISTRY:
        raise ValueError("MODEL_REGISTRY must not be empty.")

    for name, spec in MODEL_REGISTRY.items():
        if spec.name != name:
            raise ValueError(
                f"Registry key/spec mismatch: key={name!r}, spec.name={spec.name!r}"
            )
        if not spec.estimator_family:
            raise ValueError(f"Model spec {name!r} must define estimator_family.")
        if not spec.default_params:
            raise ValueError(f"Model spec {name!r} must define default_params.")

        for grid_key, grid_values in spec.search_space.items():
            if not grid_values:
                raise ValueError(
                    f"Model spec {name!r} has empty candidate list for search-space key {grid_key!r}."
                )

        if spec.name == "ridge_cv" and "alphas" not in spec.default_params:
            raise ValueError("ridge_cv must define 'alphas' in default_params.")

        if spec.name in {"random_forest", "extra_trees"}:
            for required_key in ("random_state", "n_jobs"):
                if required_key not in spec.default_params:
                    raise ValueError(
                        f"{name!r} default_params must include {required_key!r}."
                    )


validate_model_registry()


__all__ = [
    "RANDOM_STATE",
    "N_JOBS",
    "DEFAULT_RIDGE_ALPHAS",
    "DEFAULT_RF_PARAMS",
    "DEFAULT_ET_PARAMS",
    "RF_SEARCH_SPACE",
    "ET_SEARCH_SPACE",
    "RIDGE_SEARCH_SPACE",
    "ModelSpec",
    "ModelCandidate",
    "MODEL_REGISTRY",
    "list_model_names",
    "get_model_spec",
    "get_default_params",
    "get_search_space",
    "get_registry_as_dict",
    "supports_feature_importance",
    "requires_scaling",
    "build_model",
    "build_default_model",
    "build_reference_models",
    "build_candidate_grid",
    "build_default_candidates",
    "extract_final_estimator",
    "validate_model_registry",
]
