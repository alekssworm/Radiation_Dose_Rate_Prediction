"""Train MVP-B models under spatial cross-validation.

This script is the main training entry point for MVP-B.

Core responsibilities:
- load mode-specific training datasets;
- evaluate a fixed shortlist of feature sets and model families;
- use spatial GroupKFold as the default quality criterion;
- select the best env reference model and the best primary model;
- refit selected models on full data;
- save artifacts, summaries, metadata, and compact diagnostics.

Important assumptions:
- input CSVs already come from the cleaned tabular pipeline;
- spatial evaluation requires `latitude` and `longitude` columns to be present;
- coordinates are used only for spatial grouping, never as model features.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from feature_sets_b import (
    COORD_COLUMNS,
    ENV_MODE,
    EXCLUDED_COLUMNS,
    NUCLIDE_MODE,
    TARGET_COLUMN,
    get_feature_columns,
    get_feature_mode,
    get_feature_set_spec,
    get_specs_as_dict,
    list_feature_sets,
)
from model_registry_b import (
    ModelCandidate,
    build_candidate_grid,
    build_default_candidates,
    build_model,
    extract_final_estimator,
    get_registry_as_dict,
    list_model_names,
    supports_feature_importance,
)
from spatial_cv import (
    SpatialCVSpec,
    build_spatial_fold_specs,
    fold_specs_to_frame,
    get_spatial_groups,
    infer_block_origin,
    iter_spatial_splits,
    resolve_spatial_spec,
    summarize_spatial_groups,
    validate_coordinate_columns,
)


PREDICTION_COLUMN = "predicted_target_dose_rate"
DEFAULT_RUN_PREFIX = "mvp_b_train"
PRIMARY_ROLE = "primary_candidate"
ENV_ROLE = "env_reference"
DEFAULT_GROUP_COLUMN = "spatial_group"
DEFAULT_OUTPUT_DIR = Path("mvp_b") / "outputs" / "training_runs"
DEFAULT_ARTIFACTS_DIRNAME = "artifacts"
DEFAULT_EVAL_DIRNAME = "evaluation"
DEFAULT_PREDICTIONS_DIRNAME = "predictions"
DEFAULT_REPORT_NAME = "model_passport.md"


@dataclass(frozen=True)
class RunConfig:
    env_dataset: str | None
    nuclide_dataset: str | None
    output_dir: str
    run_name: str
    block_size_deg: float
    n_splits: int
    lat_col: str
    lon_col: str
    target_col: str
    models: List[str]
    feature_sets: List[str]
    defaults_only: bool
    overwrite: bool
    save_group_summaries: bool
    save_oof_predictions: bool
    verbose_folds: bool
    quiet: bool


@dataclass(frozen=True)
class SelectedConfig:
    role: str
    mode: str
    feature_set: str
    model_name: str
    candidate_id: str
    params: Dict[str, Any]
    n_rows: int
    n_features: int
    cv_test_r2_mean: float
    cv_test_r2_std: float
    cv_train_r2_mean: float
    gap_r2: float
    cv_test_rmse_mean: float
    cv_test_mae_mean: float

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["params_json"] = json.dumps(self.params, ensure_ascii=False, sort_keys=True)
        return payload


@dataclass
class ProgressTracker:
    total_candidates: int
    total_feature_sets: int
    total_expected_fits: int
    completed_candidates: int = 0
    completed_fits: int = 0

    def mark_fit(self) -> None:
        self.completed_fits += 1

    def mark_candidate(self) -> None:
        self.completed_candidates += 1

    @property
    def fit_progress_pct(self) -> float:
        if self.total_expected_fits <= 0:
            return 0.0
        return 100.0 * self.completed_fits / self.total_expected_fits


@dataclass(frozen=True)
class Logger:
    quiet: bool = False

    def log(self, message: str) -> None:
        if not self.quiet:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MVP-B models with spatial CV.")
    parser.add_argument("--env-dataset", type=str, default=None, help="Path to env-capable training CSV.")
    parser.add_argument(
        "--nuclide-dataset",
        type=str,
        default=None,
        help="Path to nuclide-capable training CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where run outputs will be saved.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional explicit run directory name. Defaults to timestamp-based name.",
    )
    parser.add_argument("--block-size-deg", type=float, default=0.02, help="Spatial block size in degrees.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of GroupKFold splits.")
    parser.add_argument("--lat-col", type=str, default=COORD_COLUMNS[0], help="Latitude column name.")
    parser.add_argument("--lon-col", type=str, default=COORD_COLUMNS[1], help="Longitude column name.")
    parser.add_argument("--target-col", type=str, default=TARGET_COLUMN, help="Target column name.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list_model_names(),
        choices=list_model_names(),
        help="Model families to evaluate.",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=list_feature_sets(),
        choices=list_feature_sets(),
        help="Registered feature sets to evaluate.",
    )
    parser.add_argument(
        "--defaults-only",
        action="store_true",
        help="Evaluate one default candidate per model family instead of the full controlled grid.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing run directory.",
    )
    parser.add_argument(
        "--no-group-summaries",
        action="store_true",
        help="Disable saving spatial group summary CSVs.",
    )
    parser.add_argument(
        "--no-oof-predictions",
        action="store_true",
        help="Disable saving out-of-fold predictions for selected models.",
    )
    parser.add_argument(
        "--verbose-folds",
        action="store_true",
        help="Print every spatial fold during training.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig(
        env_dataset=args.env_dataset,
        nuclide_dataset=args.nuclide_dataset,
        output_dir=args.output_dir,
        run_name=args.run_name or make_default_run_name(),
        block_size_deg=args.block_size_deg,
        n_splits=args.n_splits,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
        target_col=args.target_col,
        models=list(args.models),
        feature_sets=list(args.feature_sets),
        defaults_only=bool(args.defaults_only),
        overwrite=bool(args.overwrite),
        save_group_summaries=not bool(args.no_group_summaries),
        save_oof_predictions=not bool(args.no_oof_predictions),
        verbose_folds=bool(args.verbose_folds),
        quiet=bool(args.quiet),
    )
    logger = Logger(quiet=config.quiet)

    logger.log("Starting MVP-B training run.")
    logger.log(f"Run name: {config.run_name}")
    logger.log(f"Output root: {config.output_dir}")
    logger.log(
        f"Settings: block_size_deg={config.block_size_deg}, n_splits={config.n_splits}, "
        f"defaults_only={config.defaults_only}"
    )
    logger.log(f"Model shortlist: {', '.join(config.models)}")
    logger.log(f"Feature-set shortlist: {', '.join(config.feature_sets)}")

    run_dir = prepare_run_dir(Path(config.output_dir), config.run_name, overwrite=config.overwrite)
    artifacts_dir = ensure_dir(run_dir / DEFAULT_ARTIFACTS_DIRNAME)
    evaluation_dir = ensure_dir(run_dir / DEFAULT_EVAL_DIRNAME)
    predictions_dir = ensure_dir(run_dir / DEFAULT_PREDICTIONS_DIRNAME)
    logger.log(f"Run directory: {run_dir}")

    dataset_by_mode = load_mode_datasets(config=config, logger=logger)
    spatial_spec_by_mode = build_mode_spatial_specs(dataset_by_mode=dataset_by_mode, config=config, logger=logger)
    feature_ranges_by_mode = compute_feature_ranges_by_mode(dataset_by_mode=dataset_by_mode, config=config)

    all_summary_rows: List[Dict[str, Any]] = []
    all_fold_rows: List[Dict[str, Any]] = []
    group_summary_paths: Dict[str, str] = {}

    if config.save_group_summaries:
        logger.log("Saving spatial group summaries.")
        for mode, df in dataset_by_mode.items():
            group_summary_paths[mode] = save_group_summary(
                df=df,
                spec=spatial_spec_by_mode[mode],
                out_path=evaluation_dir / f"spatial_groups_{mode}.csv",
            )
            logger.log(f"Saved spatial groups for {mode}: {group_summary_paths[mode]}")

    candidates = (
        build_default_candidates(config.models)
        if config.defaults_only
        else build_candidate_grid(model_names=config.models, include_defaults=True)
    )
    feature_sets_by_mode = split_feature_sets_by_mode(config.feature_sets)

    total_feature_sets = sum(len(names) for names in feature_sets_by_mode.values())
    total_candidates = len(candidates) * total_feature_sets
    total_expected_fits = total_candidates * config.n_splits
    progress = ProgressTracker(
        total_candidates=total_candidates,
        total_feature_sets=total_feature_sets,
        total_expected_fits=total_expected_fits,
    )
    logger.log(
        f"Planned workload: {len(candidates)} candidates per feature set, "
        f"{total_feature_sets} feature sets, about {total_expected_fits} CV fits."
    )

    for mode, feature_set_names in feature_sets_by_mode.items():
        df = dataset_by_mode[mode]
        spatial_spec = spatial_spec_by_mode[mode]
        logger.log(
            f"Mode {mode}: rows={len(df)}, feature_sets={len(feature_set_names)}, "
            f"unique_spatial_groups={get_spatial_groups(df, spatial_spec).nunique()}"
        )

        for feature_set_idx, feature_set_name in enumerate(feature_set_names, start=1):
            feature_cols = get_feature_columns(feature_set_name)
            logger.log(
                f"Starting feature set [{feature_set_idx}/{len(feature_set_names)}] for {mode}: "
                f"{feature_set_name} ({len(feature_cols)} features)"
            )
            evaluation = evaluate_feature_set_candidates(
                df=df,
                feature_set_name=feature_set_name,
                candidates=candidates,
                spatial_spec=spatial_spec,
                target_col=config.target_col,
                progress=progress,
                logger=logger,
                verbose_folds=config.verbose_folds,
            )
            all_summary_rows.extend(evaluation["summary_rows"])
            all_fold_rows.extend(evaluation["fold_rows"])
            logger.log(f"Completed feature set {feature_set_name}.")

    summary_df = pd.DataFrame(all_summary_rows)
    fold_df = pd.DataFrame(all_fold_rows)

    if summary_df.empty:
        raise RuntimeError("No training results were produced. Check datasets, feature sets, and models.")

    summary_df = sort_summary_frame(summary_df)
    fold_df = sort_fold_frame(fold_df)

    summary_path = evaluation_dir / "baseline_v3_spatial_results_summary.csv"
    folds_path = evaluation_dir / "baseline_v3_per_fold_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    fold_df.to_csv(folds_path, index=False)
    logger.log(f"Saved summary CSV: {summary_path}")
    logger.log(f"Saved fold metrics CSV: {folds_path}")

    best_configs = select_best_configs(summary_df)
    best_configs_df = pd.DataFrame([cfg.to_dict() for cfg in best_configs.values()])
    best_configs_path = evaluation_dir / "baseline_v3_best_configs.csv"
    best_configs_df.to_csv(best_configs_path, index=False)
    logger.log(f"Saved selected configs: {best_configs_path}")

    fold_specs_path = save_fold_specs(
        dataset_by_mode=dataset_by_mode,
        spatial_spec_by_mode=spatial_spec_by_mode,
        out_dir=evaluation_dir,
    )
    logger.log(f"Saved fold specs: {fold_specs_path}")

    saved_artifact_paths: Dict[str, str] = {}
    feature_importance_paths: Dict[str, str] = {}
    oof_prediction_paths: Dict[str, str] = {}

    for mode, selected in best_configs.items():
        logger.log(
            f"Refitting selected {mode} model: feature_set={selected.feature_set}, "
            f"model={selected.model_name}, candidate_id={selected.candidate_id}"
        )
        df = dataset_by_mode[mode]
        spatial_spec = spatial_spec_by_mode[mode]

        fitted_payload = fit_selected_model(
            df=df,
            selected=selected,
            spatial_spec=spatial_spec,
            target_col=config.target_col,
            feature_ranges=feature_ranges_by_mode.get(mode, {}),
        )

        artifact_filename = "mvp_b_env_model.joblib" if mode == ENV_MODE else "mvp_b_primary_model.joblib"
        artifact_path = artifacts_dir / artifact_filename
        joblib.dump(fitted_payload, artifact_path)
        saved_artifact_paths[mode] = str(artifact_path)
        logger.log(f"Saved artifact for {mode}: {artifact_path}")

        feature_importance_path = save_feature_importance_if_available(
            fitted_payload=fitted_payload,
            out_dir=evaluation_dir,
        )
        if feature_importance_path is not None:
            feature_importance_paths[mode] = feature_importance_path
            logger.log(f"Saved feature importances for {mode}: {feature_importance_path}")

        if config.save_oof_predictions:
            logger.log(f"Collecting OOF predictions for {mode}.")
            oof_df = collect_oof_predictions(
                df=df,
                selected=selected,
                spatial_spec=spatial_spec,
                target_col=config.target_col,
            )
            oof_path = predictions_dir / f"oof_predictions_{selected.role}.csv"
            oof_df.to_csv(oof_path, index=False)
            oof_prediction_paths[mode] = str(oof_path)
            logger.log(f"Saved OOF predictions for {mode}: {oof_path}")

    metadata = build_metadata(
        config=config,
        spatial_spec_by_mode=spatial_spec_by_mode,
        dataset_by_mode=dataset_by_mode,
        best_configs=best_configs,
        saved_artifact_paths=saved_artifact_paths,
        summary_path=summary_path,
        folds_path=folds_path,
        best_configs_path=best_configs_path,
        fold_specs_path=fold_specs_path,
        group_summary_paths=group_summary_paths,
        feature_importance_paths=feature_importance_paths,
        oof_prediction_paths=oof_prediction_paths,
        feature_ranges_by_mode=feature_ranges_by_mode,
    )
    metadata_path = artifacts_dir / "mvp_b_metadata.json"
    write_json(metadata_path, metadata)
    logger.log(f"Saved metadata: {metadata_path}")

    report_path = evaluation_dir / DEFAULT_REPORT_NAME
    write_text(report_path, render_model_passport(best_configs=best_configs, metadata=metadata))
    logger.log(f"Saved model passport: {report_path}")

    logger.log("Selected working configurations:")
    for mode, selected in best_configs.items():
        logger.log(
            f"  - {mode}: {selected.model_name} | {selected.feature_set} | "
            f"cv_test_r2_mean={selected.cv_test_r2_mean:.6f} | gap_r2={selected.gap_r2:.6f}"
        )

    logger.log("Training completed successfully.")
    print(f"[OK] Training run saved to: {run_dir}", flush=True)
    print(f"[OK] Summary: {summary_path}", flush=True)
    print(f"[OK] Best configs: {best_configs_path}", flush=True)
    print(f"[OK] Metadata: {metadata_path}", flush=True)


def make_default_run_name() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{DEFAULT_RUN_PREFIX}_{timestamp}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_run_dir(base_output_dir: Path, run_name: str, overwrite: bool = False) -> Path:
    run_dir = base_output_dir / run_name
    if run_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Use --overwrite or choose another --run-name."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_mode_datasets(config: RunConfig, logger: Logger) -> Dict[str, pd.DataFrame]:
    dataset_by_mode: Dict[str, pd.DataFrame] = {}

    required_modes = {get_feature_mode(name) for name in config.feature_sets}
    logger.log(f"Required modes from feature sets: {', '.join(sorted(required_modes))}")

    if ENV_MODE in required_modes:
        if not config.env_dataset:
            raise ValueError("env_mode feature sets were requested, but --env-dataset was not provided.")
        logger.log(f"Loading env dataset: {config.env_dataset}")
        dataset_by_mode[ENV_MODE] = load_training_dataframe(
            path=Path(config.env_dataset),
            lat_col=config.lat_col,
            lon_col=config.lon_col,
            target_col=config.target_col,
        )
        logger.log(f"Loaded env dataset with {len(dataset_by_mode[ENV_MODE])} rows.")

    if NUCLIDE_MODE in required_modes:
        if not config.nuclide_dataset:
            raise ValueError(
                "nuclide_mode feature sets were requested, but --nuclide-dataset was not provided."
            )
        logger.log(f"Loading nuclide dataset: {config.nuclide_dataset}")
        dataset_by_mode[NUCLIDE_MODE] = load_training_dataframe(
            path=Path(config.nuclide_dataset),
            lat_col=config.lat_col,
            lon_col=config.lon_col,
            target_col=config.target_col,
        )
        logger.log(f"Loaded nuclide dataset with {len(dataset_by_mode[NUCLIDE_MODE])} rows.")

    return dataset_by_mode


def load_training_dataframe(path: Path, lat_col: str, lon_col: str, target_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Training dataset was not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Training dataset is empty: {path}")

    validate_coordinate_columns(df, lat_col=lat_col, lon_col=lon_col, allow_nulls=False)
    if target_col not in df.columns:
        raise KeyError(f"Target column {target_col!r} is missing from dataset: {path}")
    if df[target_col].isna().any():
        raise ValueError(f"Target column {target_col!r} contains null values in dataset: {path}")

    return df.reset_index(drop=True)


def build_mode_spatial_specs(
    dataset_by_mode: Mapping[str, pd.DataFrame],
    config: RunConfig,
    logger: Logger,
) -> Dict[str, SpatialCVSpec]:
    spatial_specs: Dict[str, SpatialCVSpec] = {}
    for mode, df in dataset_by_mode.items():
        lat_origin, lon_origin = infer_block_origin(df, lat_col=config.lat_col, lon_col=config.lon_col)
        spatial_specs[mode] = resolve_spatial_spec(
            block_size_deg=config.block_size_deg,
            n_splits=config.n_splits,
            lat_col=config.lat_col,
            lon_col=config.lon_col,
            lat_origin=lat_origin,
            lon_origin=lon_origin,
            group_column=DEFAULT_GROUP_COLUMN,
        )
        logger.log(
            f"Spatial spec for {mode}: block_size_deg={config.block_size_deg}, n_splits={config.n_splits}, "
            f"lat_origin={lat_origin:.6f}, lon_origin={lon_origin:.6f}"
        )
    return spatial_specs


def compute_feature_ranges_by_mode(
    dataset_by_mode: Mapping[str, pd.DataFrame],
    config: RunConfig,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    feature_ranges_by_mode: Dict[str, Dict[str, Dict[str, float]]] = {}
    for mode, df in dataset_by_mode.items():
        ranges: Dict[str, Dict[str, float]] = {}
        candidate_feature_columns = [
            col for col in df.columns if col not in {config.target_col, config.lat_col, config.lon_col, *EXCLUDED_COLUMNS}
        ]
        for col in candidate_feature_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                ranges[col] = {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                }
        feature_ranges_by_mode[mode] = ranges
    return feature_ranges_by_mode


def split_feature_sets_by_mode(feature_set_names: Sequence[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {ENV_MODE: [], NUCLIDE_MODE: []}
    for name in feature_set_names:
        grouped[get_feature_mode(name)].append(name)
    return {mode: names for mode, names in grouped.items() if names}


def evaluate_feature_set_candidates(
    df: pd.DataFrame,
    feature_set_name: str,
    candidates: Sequence[ModelCandidate],
    spatial_spec: SpatialCVSpec,
    target_col: str,
    progress: ProgressTracker,
    logger: Logger,
    verbose_folds: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    feature_cols = get_feature_columns(feature_set_name)
    validate_dataset_for_feature_set(df, feature_set_name, target_col=target_col)

    groups = get_spatial_groups(df, spatial_spec)
    X = df[feature_cols].copy()
    y = df[target_col].to_numpy(dtype=float)

    summary_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []

    for candidate_index, candidate in enumerate(candidates, start=1):
        logger.log(
            f"Candidate [{candidate_index}/{len(candidates)}] for {feature_set_name}: "
            f"{candidate.model_name} | {candidate.candidate_id}"
        )
        evaluation = evaluate_candidate_cv(
            X=X,
            y=y,
            groups=groups,
            candidate=candidate,
            feature_set_name=feature_set_name,
            spatial_spec=spatial_spec,
            progress=progress,
            logger=logger,
            verbose_folds=verbose_folds,
        )
        summary_rows.append(evaluation["summary_row"])
        fold_rows.extend(evaluation["fold_rows"])
        progress.mark_candidate()
        logger.log(
            f"Done candidate {candidate.candidate_id} | "
            f"cv_test_r2_mean={evaluation['summary_row']['cv_test_r2_mean']:.6f} | "
            f"progress={progress.completed_candidates}/{progress.total_candidates} candidates, "
            f"{progress.completed_fits}/{progress.total_expected_fits} fits ({progress.fit_progress_pct:.1f}%)"
        )

    return {"summary_rows": summary_rows, "fold_rows": fold_rows}


def validate_dataset_for_feature_set(df: pd.DataFrame, feature_set_name: str, target_col: str) -> None:
    feature_cols = get_feature_columns(feature_set_name)
    required_cols = [target_col, *feature_cols]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(
            f"Dataset is missing required columns for feature set {feature_set_name!r}: {missing_str}"
        )

    if df[target_col].isna().any():
        raise ValueError(f"Target column {target_col!r} contains null values.")


def evaluate_candidate_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    candidate: ModelCandidate,
    feature_set_name: str,
    spatial_spec: SpatialCVSpec,
    progress: ProgressTracker,
    logger: Logger,
    verbose_folds: bool,
) -> Dict[str, Any]:
    fold_rows: List[Dict[str, Any]] = []

    for fold_index, train_idx, test_idx in iter_spatial_splits(X=X, groups=groups, n_splits=spatial_spec.n_splits):
        if verbose_folds:
            logger.log(
                f"  Fold [{fold_index + 1}/{spatial_spec.n_splits}] {feature_set_name} | {candidate.model_name} | "
                f"train_rows={len(train_idx)}, test_rows={len(test_idx)}"
            )

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        groups_train = groups.iloc[train_idx]
        groups_test = groups.iloc[test_idx]

        model = build_model(candidate.model_name, params=candidate.params)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        fold_rows.append(
            {
                "feature_set": feature_set_name,
                "mode": get_feature_mode(feature_set_name),
                "model_name": candidate.model_name,
                "candidate_id": candidate.candidate_id,
                "params_json": json.dumps(candidate.params, ensure_ascii=False, sort_keys=True),
                "fold_index": fold_index,
                "n_features": int(X.shape[1]),
                "n_train_rows": int(len(train_idx)),
                "n_test_rows": int(len(test_idx)),
                "n_train_groups": int(groups_train.nunique()),
                "n_test_groups": int(groups_test.nunique()),
                "train_r2": safe_r2(y_train, train_pred),
                "test_r2": safe_r2(y_test, test_pred),
                "train_rmse": rmse(y_train, train_pred),
                "test_rmse": rmse(y_test, test_pred),
                "train_mae": mae(y_train, train_pred),
                "test_mae": mae(y_test, test_pred),
                "block_size_deg": float(spatial_spec.block_size_deg),
                "n_splits": int(spatial_spec.n_splits),
            }
        )
        progress.mark_fit()

    fold_df = pd.DataFrame(fold_rows)
    summary_row = summarize_candidate_metrics(
        fold_df=fold_df,
        feature_set_name=feature_set_name,
        candidate=candidate,
        n_rows=len(X),
        n_features=X.shape[1],
        block_size_deg=spatial_spec.block_size_deg,
        n_splits=spatial_spec.n_splits,
    )
    return {"summary_row": summary_row, "fold_rows": fold_rows}


def summarize_candidate_metrics(
    fold_df: pd.DataFrame,
    feature_set_name: str,
    candidate: ModelCandidate,
    n_rows: int,
    n_features: int,
    block_size_deg: float,
    n_splits: int,
) -> Dict[str, Any]:
    row = {
        "feature_set": feature_set_name,
        "mode": get_feature_mode(feature_set_name),
        "model_name": candidate.model_name,
        "candidate_id": candidate.candidate_id,
        "params_json": json.dumps(candidate.params, ensure_ascii=False, sort_keys=True),
        "n_rows": int(n_rows),
        "n_features": int(n_features),
        "block_size_deg": float(block_size_deg),
        "n_splits": int(n_splits),
        "cv_train_r2_mean": float(fold_df["train_r2"].mean()),
        "cv_train_r2_std": float(fold_df["train_r2"].std(ddof=0)),
        "cv_test_r2_mean": float(fold_df["test_r2"].mean()),
        "cv_test_r2_std": float(fold_df["test_r2"].std(ddof=0)),
        "cv_train_rmse_mean": float(fold_df["train_rmse"].mean()),
        "cv_train_rmse_std": float(fold_df["train_rmse"].std(ddof=0)),
        "cv_test_rmse_mean": float(fold_df["test_rmse"].mean()),
        "cv_test_rmse_std": float(fold_df["test_rmse"].std(ddof=0)),
        "cv_train_mae_mean": float(fold_df["train_mae"].mean()),
        "cv_train_mae_std": float(fold_df["train_mae"].std(ddof=0)),
        "cv_test_mae_mean": float(fold_df["test_mae"].mean()),
        "cv_test_mae_std": float(fold_df["test_mae"].std(ddof=0)),
    }
    row["gap_r2"] = float(row["cv_train_r2_mean"] - row["cv_test_r2_mean"])
    return row


def sort_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(
            by=["mode", "cv_test_r2_mean", "gap_r2", "cv_test_r2_std", "cv_test_rmse_mean"],
            ascending=[True, False, True, True, True],
        )
        .reset_index(drop=True)
    )


def sort_fold_frame(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(by=["mode", "feature_set", "model_name", "candidate_id", "fold_index"])
        .reset_index(drop=True)
    )


def select_best_configs(summary_df: pd.DataFrame) -> Dict[str, SelectedConfig]:
    best_configs: Dict[str, SelectedConfig] = {}

    for mode in (ENV_MODE, NUCLIDE_MODE):
        mode_df = summary_df[summary_df["mode"] == mode].copy()
        if mode_df.empty:
            continue

        mode_df = mode_df.sort_values(
            by=["cv_test_r2_mean", "gap_r2", "cv_test_r2_std", "cv_test_rmse_mean"],
            ascending=[False, True, True, True],
        ).reset_index(drop=True)

        best_row = mode_df.iloc[0]
        role = ENV_ROLE if mode == ENV_MODE else PRIMARY_ROLE
        best_configs[mode] = SelectedConfig(
            role=role,
            mode=mode,
            feature_set=str(best_row["feature_set"]),
            model_name=str(best_row["model_name"]),
            candidate_id=str(best_row["candidate_id"]),
            params=json.loads(best_row["params_json"]),
            n_rows=int(best_row["n_rows"]),
            n_features=int(best_row["n_features"]),
            cv_test_r2_mean=float(best_row["cv_test_r2_mean"]),
            cv_test_r2_std=float(best_row["cv_test_r2_std"]),
            cv_train_r2_mean=float(best_row["cv_train_r2_mean"]),
            gap_r2=float(best_row["gap_r2"]),
            cv_test_rmse_mean=float(best_row["cv_test_rmse_mean"]),
            cv_test_mae_mean=float(best_row["cv_test_mae_mean"]),
        )

    if ENV_MODE not in best_configs:
        raise RuntimeError("No env_mode configuration was selected.")
    if NUCLIDE_MODE not in best_configs:
        raise RuntimeError("No nuclide_mode configuration was selected.")

    return best_configs


def save_fold_specs(
    dataset_by_mode: Mapping[str, pd.DataFrame],
    spatial_spec_by_mode: Mapping[str, SpatialCVSpec],
    out_dir: Path,
) -> str:
    frames: List[pd.DataFrame] = []
    for mode, df in dataset_by_mode.items():
        groups = get_spatial_groups(df, spatial_spec_by_mode[mode])
        fold_specs = build_spatial_fold_specs(X=df, groups=groups, n_splits=spatial_spec_by_mode[mode].n_splits)
        frame = fold_specs_to_frame(fold_specs)
        frame.insert(0, "mode", mode)
        frames.append(frame)

    out_path = out_dir / "spatial_fold_specs.csv"
    pd.concat(frames, axis=0, ignore_index=True).to_csv(out_path, index=False)
    return str(out_path)


def save_group_summary(df: pd.DataFrame, spec: SpatialCVSpec, out_path: Path) -> str:
    groups = get_spatial_groups(df, spec)
    summary_df = summarize_spatial_groups(df, groups=groups, lat_col=spec.lat_col, lon_col=spec.lon_col)
    summary_df.to_csv(out_path, index=False)
    return str(out_path)


def fit_selected_model(
    df: pd.DataFrame,
    selected: SelectedConfig,
    spatial_spec: SpatialCVSpec,
    target_col: str,
    feature_ranges: Mapping[str, Mapping[str, float]],
) -> Dict[str, Any]:
    feature_cols = get_feature_columns(selected.feature_set)
    X = df[feature_cols].copy()
    y = df[target_col].to_numpy(dtype=float)

    model = build_model(selected.model_name, params=selected.params)
    model.fit(X, y)

    payload = {
        "artifact_version": "1.0.0",
        "created_at_utc": utc_now_iso(),
        "role": selected.role,
        "mode": selected.mode,
        "target_name": target_col,
        "prediction_column_name": PREDICTION_COLUMN,
        "feature_set": selected.feature_set,
        "feature_columns": feature_cols,
        "model_name": selected.model_name,
        "model_params": selected.params,
        "excluded_columns": list(EXCLUDED_COLUMNS),
        "spatial_cv_spec": spatial_spec.to_dict(),
        "training_rows": int(len(df)),
        "feature_ranges": {k: dict(v) for k, v in feature_ranges.items()},
        "pipeline": model,
    }
    return payload


def save_feature_importance_if_available(fitted_payload: Mapping[str, Any], out_dir: Path) -> str | None:
    model_name = str(fitted_payload["model_name"])
    if not supports_feature_importance(model_name):
        return None

    pipeline = fitted_payload["pipeline"]
    estimator = extract_final_estimator(pipeline)
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return None

    feature_cols = list(fitted_payload["feature_columns"])
    frame = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": np.asarray(importances, dtype=float),
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    role = str(fitted_payload["role"])
    out_path = out_dir / f"feature_importance_{role}.csv"
    frame.to_csv(out_path, index=False)
    return str(out_path)


def collect_oof_predictions(
    df: pd.DataFrame,
    selected: SelectedConfig,
    spatial_spec: SpatialCVSpec,
    target_col: str,
) -> pd.DataFrame:
    feature_cols = get_feature_columns(selected.feature_set)
    X = df[feature_cols].copy()
    y = df[target_col].to_numpy(dtype=float)
    groups = get_spatial_groups(df, spatial_spec)

    rows: List[Dict[str, Any]] = []

    for fold_index, train_idx, test_idx in iter_spatial_splits(X=X, groups=groups, n_splits=spatial_spec.n_splits):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        model = build_model(selected.model_name, params=selected.params)
        model.fit(X_train, y_train)
        test_pred = model.predict(X_test)

        test_block = df.iloc[test_idx].copy()
        group_values = groups.iloc[test_idx].to_numpy()

        for local_idx, (row_index, true_value, pred_value, group_value) in enumerate(
            zip(test_block.index, y_test, test_pred, group_values)
        ):
            row_payload = {
                "source_row_index": int(row_index),
                "fold_index": int(fold_index),
                "model_role": selected.role,
                "mode": selected.mode,
                "feature_set": selected.feature_set,
                "model_name": selected.model_name,
                "candidate_id": selected.candidate_id,
                spatial_spec.group_column: group_value,
                target_col: float(true_value),
                PREDICTION_COLUMN: float(pred_value),
                "residual": float(true_value - pred_value),
                "abs_error": float(abs(true_value - pred_value)),
            }
            if spatial_spec.lat_col in test_block.columns:
                row_payload[spatial_spec.lat_col] = float(test_block.iloc[local_idx][spatial_spec.lat_col])
            if spatial_spec.lon_col in test_block.columns:
                row_payload[spatial_spec.lon_col] = float(test_block.iloc[local_idx][spatial_spec.lon_col])
            rows.append(row_payload)

    return pd.DataFrame(rows)


def build_metadata(
    config: RunConfig,
    spatial_spec_by_mode: Mapping[str, SpatialCVSpec],
    dataset_by_mode: Mapping[str, pd.DataFrame],
    best_configs: Mapping[str, SelectedConfig],
    saved_artifact_paths: Mapping[str, str],
    summary_path: Path,
    folds_path: Path,
    best_configs_path: Path,
    fold_specs_path: str,
    group_summary_paths: Mapping[str, str],
    feature_importance_paths: Mapping[str, str],
    oof_prediction_paths: Mapping[str, str],
    feature_ranges_by_mode: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> Dict[str, Any]:
    mode_to_role = {ENV_MODE: ENV_ROLE, NUCLIDE_MODE: PRIMARY_ROLE}
    mode_to_feature_set = {mode: cfg.feature_set for mode, cfg in best_configs.items()}
    mode_to_required_features = {mode: get_feature_columns(cfg.feature_set) for mode, cfg in best_configs.items()}

    metadata = {
        "artifact_name": "mvp_b_metadata",
        "artifact_version": "1.0.0",
        "created_at_utc": utc_now_iso(),
        "project_stage": "MVP-B",
        "task_type": "static_tabular_regression",
        "target_name": config.target_col,
        "prediction_column_name": PREDICTION_COLUMN,
        "mode_to_model_role": mode_to_role,
        "mode_to_artifact": saved_artifact_paths,
        "selected_feature_set_by_mode": mode_to_feature_set,
        "required_features_by_mode": mode_to_required_features,
        "feature_ranges_by_mode": {
            mode: {feature: dict(bounds) for feature, bounds in ranges.items()}
            for mode, ranges in feature_ranges_by_mode.items()
        },
        "excluded_columns": list(EXCLUDED_COLUMNS),
        "training_sources": {
            "env_dataset_path": config.env_dataset,
            "nuclide_dataset_path": config.nuclide_dataset,
        },
        "training_rows": {mode: int(len(df)) for mode, df in dataset_by_mode.items()},
        "spatial_cv_by_mode": {mode: spec.to_dict() for mode, spec in spatial_spec_by_mode.items()},
        "selection_policy": {
            "ranking_priority": [
                "max cv_test_r2_mean",
                "min gap_r2",
                "min cv_test_r2_std",
                "min cv_test_rmse_mean",
            ],
            "evaluation_mode": "spatial_groupkfold",
        },
        "selected_configs": {mode: cfg.to_dict() for mode, cfg in best_configs.items()},
        "feature_set_registry": get_specs_as_dict(),
        "model_registry": get_registry_as_dict(),
        "outputs": {
            "summary_csv": str(summary_path),
            "fold_metrics_csv": str(folds_path),
            "best_configs_csv": str(best_configs_path),
            "fold_specs_csv": fold_specs_path,
            "group_summaries": dict(group_summary_paths),
            "feature_importances": dict(feature_importance_paths),
            "oof_predictions": dict(oof_prediction_paths),
        },
        "notes": [
            "MVP-B training uses spatial GroupKFold as the default evaluation mode.",
            "Coordinates are used only for spatial grouping and are excluded from model features.",
            "This metadata describes the selected working configurations for env and primary modes.",
        ],
    }
    return metadata


def render_model_passport(
    best_configs: Mapping[str, SelectedConfig],
    metadata: Mapping[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# MVP-B model passport")
    lines.append("")
    lines.append(f"Generated at UTC: {metadata['created_at_utc']}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(
        "MVP-B is the current spatially evaluated and reproducibly trained working model package "
        "for `target_dose_rate` prediction."
    )
    lines.append("")
    lines.append("## Selection policy")
    lines.append("")
    lines.append("Models are ranked by:")
    lines.append("")
    lines.append("1. highest `cv_test_r2_mean`")
    lines.append("2. lowest `gap_r2`")
    lines.append("3. lowest `cv_test_r2_std`")
    lines.append("4. lowest `cv_test_rmse_mean`")
    lines.append("")

    for mode in (ENV_MODE, NUCLIDE_MODE):
        selected = best_configs.get(mode)
        if selected is None:
            continue
        title = "Env reference model" if mode == ENV_MODE else "Primary model"
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"- Role: `{selected.role}`")
        lines.append(f"- Mode: `{selected.mode}`")
        lines.append(f"- Feature set: `{selected.feature_set}`")
        lines.append(f"- Model: `{selected.model_name}`")
        lines.append(f"- Candidate ID: `{selected.candidate_id}`")
        lines.append(f"- Rows used: `{selected.n_rows}`")
        lines.append(f"- Number of features: `{selected.n_features}`")
        lines.append(f"- Spatial CV test R² mean: `{selected.cv_test_r2_mean:.6f}`")
        lines.append(f"- Spatial CV test R² std: `{selected.cv_test_r2_std:.6f}`")
        lines.append(f"- Spatial CV train R² mean: `{selected.cv_train_r2_mean:.6f}`")
        lines.append(f"- Gap R²: `{selected.gap_r2:.6f}`")
        lines.append(f"- Spatial CV test RMSE mean: `{selected.cv_test_rmse_mean:.6f}`")
        lines.append(f"- Spatial CV test MAE mean: `{selected.cv_test_mae_mean:.6f}`")
        lines.append("")
        lines.append("Parameters:")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(selected.params, ensure_ascii=False, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")

    lines.append("## Known limitations")
    lines.append("")
    lines.append("- Spatial robustness is prioritized over ordinary random-split score.")
    lines.append("- Contamination-related features remain the strongest predictive source.")
    lines.append("- Env-only performance is expected to be weaker and should be treated as contextual.")
    lines.append("- This package is intended for tabular prediction and controlled scenario comparison.")
    lines.append("")
    return "".join(lines)


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return float("nan")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=False)


def write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
