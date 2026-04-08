"""Run schema-validated inference for MVP-B artifacts.

This script applies a selected MVP-B model to a new tabular CSV.

Main responsibilities:
- load metadata and model artifacts;
- resolve prediction mode (`env_mode` or `nuclide_mode`);
- validate input schema against required feature columns;
- optionally flag out-of-domain rows if feature ranges are available;
- produce CSV-out predictions with warning columns.

Notes:
- missing required features are treated as errors;
- nulls in required features are treated as errors;
- outside-range values are warnings only when feature ranges are available;
- coordinates are not required for prediction unless they are present for context.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd


DEFAULT_PREDICTION_COLUMN = "predicted_target_dose_rate"
DEFAULT_MODE_COLUMN = "model_mode"
DEFAULT_VERSION_COLUMN = "model_version"
DEFAULT_FEATURE_SET_COLUMN = "feature_set"
DEFAULT_MODEL_NAME_COLUMN = "model_name"
DEFAULT_WARNING_COUNT_COLUMN = "warning_count"
DEFAULT_WARNING_COLUMNS_COLUMN = "warning_columns"
DEFAULT_WARNING_TYPES_COLUMN = "warning_types"
DEFAULT_ROW_ID_FALLBACK = "source_row_index"

ENV_MODE = "env_mode"
NUCLIDE_MODE = "nuclide_mode"


@dataclass(frozen=True)
class PredictionConfig:
    input_csv: str
    output_csv: str
    metadata_path: str
    artifact_path: str | None
    mode: str | None
    row_id_col: str | None
    fail_on_extra_columns: bool
    include_all_input_columns: bool
    prediction_column_name: str | None


@dataclass(frozen=True)
class ValidationSummary:
    mode: str
    feature_set: str
    model_name: str
    required_features: Tuple[str, ...]
    unexpected_columns: Tuple[str, ...]
    warning_rows: int
    warning_columns_present: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "feature_set": self.feature_set,
            "model_name": self.model_name,
            "required_features": list(self.required_features),
            "unexpected_columns": list(self.unexpected_columns),
            "warning_rows": self.warning_rows,
            "warning_columns_present": self.warning_columns_present,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run schema-validated inference for MVP-B.")
    parser.add_argument("--input-csv", required=True, type=str, help="Path to input CSV.")
    parser.add_argument("--output-csv", required=True, type=str, help="Path to output CSV.")
    parser.add_argument(
        "--metadata-path",
        required=True,
        type=str,
        help="Path to mvp_b_metadata.json.",
    )
    parser.add_argument(
        "--artifact-path",
        default=None,
        type=str,
        help="Optional explicit artifact path. If omitted, the path is resolved from metadata and mode.",
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=[ENV_MODE, NUCLIDE_MODE],
        help="Explicit mode for prediction. Required unless artifact-path uniquely determines the mode.",
    )
    parser.add_argument(
        "--row-id-col",
        default=None,
        type=str,
        help="Optional input column to preserve as a row identifier.",
    )
    parser.add_argument(
        "--fail-on-extra-columns",
        action="store_true",
        help="Raise an error if the input contains columns outside the required feature set.",
    )
    parser.add_argument(
        "--features-only-output",
        action="store_true",
        help="Write a compact output instead of copying all input columns.",
    )
    parser.add_argument(
        "--prediction-column-name",
        default=None,
        type=str,
        help="Override output prediction column name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PredictionConfig(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        metadata_path=args.metadata_path,
        artifact_path=args.artifact_path,
        mode=args.mode,
        row_id_col=args.row_id_col,
        fail_on_extra_columns=bool(args.fail_on_extra_columns),
        include_all_input_columns=not bool(args.features_only_output),
        prediction_column_name=args.prediction_column_name,
    )

    metadata = load_metadata(Path(config.metadata_path))
    mode = resolve_prediction_mode(config=config, metadata=metadata)
    artifact_path = resolve_artifact_path(config=config, metadata=metadata, mode=mode)
    artifact_payload = load_artifact(artifact_path)

    validate_artifact_against_metadata(artifact_payload=artifact_payload, metadata=metadata, mode=mode)

    df = load_input_dataframe(Path(config.input_csv))
    validation = validate_input_dataframe(
        df=df,
        metadata=metadata,
        artifact_payload=artifact_payload,
        mode=mode,
        row_id_col=config.row_id_col,
        fail_on_extra_columns=config.fail_on_extra_columns,
    )

    output_df = run_prediction(
        df=df,
        metadata=metadata,
        artifact_payload=artifact_payload,
        mode=mode,
        row_id_col=config.row_id_col,
        include_all_input_columns=config.include_all_input_columns,
        prediction_column_name=config.prediction_column_name,
    )

    output_path = Path(config.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"[OK] Mode: {validation.mode}")
    print(f"[OK] Feature set: {validation.feature_set}")
    print(f"[OK] Model: {validation.model_name}")
    print(f"[OK] Rows predicted: {len(output_df)}")
    print(f"[OK] Warning rows: {validation.warning_rows}")
    if validation.unexpected_columns:
        unexpected = ", ".join(validation.unexpected_columns)
        print(f"[WARN] Unexpected input columns were ignored: {unexpected}")
    print(f"[OK] Output saved to: {output_path}")


def load_metadata(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file was not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Metadata file does not contain a JSON object: {path}")
    return payload


def load_artifact(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Artifact file was not found: {path}")
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact is expected to be a dict payload: {path}")
    if "pipeline" not in payload:
        raise ValueError(f"Artifact payload is missing 'pipeline': {path}")
    return payload


def load_input_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV was not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {path}")
    return df.reset_index(drop=True)


def resolve_prediction_mode(config: PredictionConfig, metadata: Mapping[str, Any]) -> str:
    if config.mode is not None:
        return config.mode

    if config.artifact_path:
        artifact_path_value = str(Path(config.artifact_path))
        mode_to_artifact = metadata.get("mode_to_artifact", {})
        matching_modes = [
            mode
            for mode, artifact_ref in mode_to_artifact.items()
            if _same_path_string(artifact_path_value, str(artifact_ref))
        ]
        if len(matching_modes) == 1:
            return matching_modes[0]

    raise ValueError(
        "Prediction mode could not be resolved automatically. "
        "Pass --mode env_mode or --mode nuclide_mode."
    )


def resolve_artifact_path(config: PredictionConfig, metadata: Mapping[str, Any], mode: str) -> Path:
    if config.artifact_path is not None:
        return Path(config.artifact_path)

    mode_to_artifact = metadata.get("mode_to_artifact", {})
    artifact_ref = mode_to_artifact.get(mode)
    if not artifact_ref:
        raise KeyError(f"No artifact path is registered for mode {mode!r} in metadata.")

    metadata_path = Path(config.metadata_path).resolve()
    metadata_dir = metadata_path.parent
    artifact_path = Path(str(artifact_ref))
    if artifact_path.is_absolute():
        return artifact_path
    return (metadata_dir / artifact_path).resolve()


def validate_artifact_against_metadata(
    artifact_payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    mode: str,
) -> None:
    artifact_mode = str(artifact_payload.get("mode"))
    if artifact_mode != mode:
        raise ValueError(
            f"Artifact mode mismatch: artifact={artifact_mode!r}, requested={mode!r}."
        )

    metadata_required = tuple(metadata.get("required_features_by_mode", {}).get(mode, []))
    artifact_required = tuple(artifact_payload.get("feature_columns", []))
    if metadata_required and artifact_required and metadata_required != artifact_required:
        raise ValueError(
            "Artifact/metadata feature mismatch for mode "
            f"{mode!r}: metadata={metadata_required}, artifact={artifact_required}"
        )


def validate_input_dataframe(
    df: pd.DataFrame,
    metadata: Mapping[str, Any],
    artifact_payload: Mapping[str, Any],
    mode: str,
    row_id_col: str | None,
    fail_on_extra_columns: bool,
) -> ValidationSummary:
    feature_set = resolve_feature_set_name(metadata=metadata, artifact_payload=artifact_payload, mode=mode)
    model_name = str(artifact_payload.get("model_name", "unknown_model"))
    required_features = tuple(resolve_required_features(metadata=metadata, artifact_payload=artifact_payload, mode=mode))

    missing_features = [col for col in required_features if col not in df.columns]
    if missing_features:
        missing_str = ", ".join(missing_features)
        raise KeyError(f"Input CSV is missing required feature columns: {missing_str}")

    if row_id_col is not None and row_id_col not in df.columns:
        raise KeyError(f"Requested row-id column is missing from input CSV: {row_id_col!r}")

    null_features = [col for col in required_features if df[col].isna().any()]
    if null_features:
        null_str = ", ".join(null_features)
        raise ValueError(f"Null values are not allowed in required features: {null_str}")

    allowed_columns = set(required_features)
    if row_id_col is not None:
        allowed_columns.add(row_id_col)

    unexpected_columns = tuple(col for col in df.columns if col not in allowed_columns)
    if unexpected_columns and fail_on_extra_columns:
        unexpected_str = ", ".join(unexpected_columns)
        raise ValueError(
            f"Input CSV contains unexpected columns outside the required schema: {unexpected_str}"
        )

    warning_columns_present = bool(resolve_feature_ranges(metadata=metadata, artifact_payload=artifact_payload, mode=mode))
    warning_rows = count_warning_rows(
        df=df,
        feature_ranges=resolve_feature_ranges(metadata=metadata, artifact_payload=artifact_payload, mode=mode),
        required_features=required_features,
    )

    return ValidationSummary(
        mode=mode,
        feature_set=feature_set,
        model_name=model_name,
        required_features=required_features,
        unexpected_columns=unexpected_columns,
        warning_rows=warning_rows,
        warning_columns_present=warning_columns_present,
    )


def run_prediction(
    df: pd.DataFrame,
    metadata: Mapping[str, Any],
    artifact_payload: Mapping[str, Any],
    mode: str,
    row_id_col: str | None,
    include_all_input_columns: bool,
    prediction_column_name: str | None,
) -> pd.DataFrame:
    required_features = resolve_required_features(metadata=metadata, artifact_payload=artifact_payload, mode=mode)
    model_version = str(metadata.get("artifact_version") or artifact_payload.get("artifact_version") or "unknown")
    feature_set = resolve_feature_set_name(metadata=metadata, artifact_payload=artifact_payload, mode=mode)
    model_name = str(artifact_payload.get("model_name", "unknown_model"))
    pred_col = (
        prediction_column_name
        or str(metadata.get("prediction_column_name") or artifact_payload.get("prediction_column_name") or DEFAULT_PREDICTION_COLUMN)
    )

    feature_ranges = resolve_feature_ranges(metadata=metadata, artifact_payload=artifact_payload, mode=mode)
    warning_frame = build_warning_frame(
        df=df,
        required_features=required_features,
        feature_ranges=feature_ranges,
    )

    X = df[required_features].copy()
    pipeline = artifact_payload["pipeline"]
    predictions = pipeline.predict(X)

    if include_all_input_columns:
        out_df = df.copy()
    else:
        out_df = pd.DataFrame(index=df.index)

    if row_id_col and row_id_col not in out_df.columns:
        out_df[row_id_col] = df[row_id_col]
    elif row_id_col is None and DEFAULT_ROW_ID_FALLBACK not in out_df.columns:
        out_df[DEFAULT_ROW_ID_FALLBACK] = np.arange(len(df), dtype=int)

    out_df[pred_col] = np.asarray(predictions, dtype=float)
    out_df[DEFAULT_MODE_COLUMN] = mode
    out_df[DEFAULT_VERSION_COLUMN] = model_version
    out_df[DEFAULT_FEATURE_SET_COLUMN] = feature_set
    out_df[DEFAULT_MODEL_NAME_COLUMN] = model_name
    out_df[DEFAULT_WARNING_COUNT_COLUMN] = warning_frame[DEFAULT_WARNING_COUNT_COLUMN].astype(int)
    out_df[DEFAULT_WARNING_COLUMNS_COLUMN] = warning_frame[DEFAULT_WARNING_COLUMNS_COLUMN]
    out_df[DEFAULT_WARNING_TYPES_COLUMN] = warning_frame[DEFAULT_WARNING_TYPES_COLUMN]
    return out_df


def resolve_required_features(
    metadata: Mapping[str, Any],
    artifact_payload: Mapping[str, Any],
    mode: str,
) -> List[str]:
    metadata_required = metadata.get("required_features_by_mode", {}).get(mode)
    if metadata_required:
        return list(metadata_required)

    artifact_required = artifact_payload.get("feature_columns")
    if artifact_required:
        return list(artifact_required)

    raise KeyError(f"Required features are not available for mode {mode!r}.")


def resolve_feature_set_name(
    metadata: Mapping[str, Any],
    artifact_payload: Mapping[str, Any],
    mode: str,
) -> str:
    feature_set = metadata.get("selected_feature_set_by_mode", {}).get(mode)
    if feature_set:
        return str(feature_set)
    artifact_feature_set = artifact_payload.get("feature_set")
    if artifact_feature_set:
        return str(artifact_feature_set)
    return "unknown_feature_set"


def resolve_feature_ranges(
    metadata: Mapping[str, Any],
    artifact_payload: Mapping[str, Any],
    mode: str,
) -> Dict[str, Dict[str, float]]:
    metadata_ranges = metadata.get("feature_ranges_by_mode", {}).get(mode)
    if isinstance(metadata_ranges, dict):
        return _normalize_feature_ranges(metadata_ranges)

    artifact_ranges = artifact_payload.get("feature_ranges")
    if isinstance(artifact_ranges, dict):
        return _normalize_feature_ranges(artifact_ranges)

    return {}


def count_warning_rows(
    df: pd.DataFrame,
    feature_ranges: Mapping[str, Mapping[str, float]],
    required_features: Sequence[str],
) -> int:
    if not feature_ranges:
        return 0
    warning_frame = build_warning_frame(df=df, required_features=required_features, feature_ranges=feature_ranges)
    return int((warning_frame[DEFAULT_WARNING_COUNT_COLUMN] > 0).sum())


def build_warning_frame(
    df: pd.DataFrame,
    required_features: Sequence[str],
    feature_ranges: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    warning_counts: List[int] = []
    warning_columns_list: List[str] = []
    warning_types_list: List[str] = []

    for row_idx in range(len(df)):
        row_warning_columns: List[str] = []
        row_warning_types: List[str] = []

        if feature_ranges:
            row = df.iloc[row_idx]
            for feature in required_features:
                feature_range = feature_ranges.get(feature)
                if not feature_range:
                    continue

                value = row[feature]
                min_value = feature_range.get("min")
                max_value = feature_range.get("max")

                if min_value is not None and value < min_value:
                    row_warning_columns.append(feature)
                    row_warning_types.append(f"below_train_range:{feature}")
                elif max_value is not None and value > max_value:
                    row_warning_columns.append(feature)
                    row_warning_types.append(f"above_train_range:{feature}")

        warning_counts.append(len(row_warning_types))
        warning_columns_list.append(";".join(row_warning_columns))
        warning_types_list.append(";".join(row_warning_types))

    return pd.DataFrame(
        {
            DEFAULT_WARNING_COUNT_COLUMN: warning_counts,
            DEFAULT_WARNING_COLUMNS_COLUMN: warning_columns_list,
            DEFAULT_WARNING_TYPES_COLUMN: warning_types_list,
        },
        index=df.index,
    )


def _normalize_feature_ranges(raw_ranges: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    normalized: Dict[str, Dict[str, float]] = {}
    for feature, bounds in raw_ranges.items():
        if not isinstance(bounds, Mapping):
            continue
        entry: Dict[str, float] = {}
        if "min" in bounds and bounds["min"] is not None:
            entry["min"] = float(bounds["min"])
        if "max" in bounds and bounds["max"] is not None:
            entry["max"] = float(bounds["max"])
        if entry:
            normalized[str(feature)] = entry
    return normalized


def _same_path_string(left: str, right: str) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except Exception:
        return str(left) == str(right)


if __name__ == "__main__":
    main()
