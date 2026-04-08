from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_RANDOM_STATE = 42
DEFAULT_TARGET_COL = "target_dose_rate"
DEFAULT_PROXY_TARGET_COL = "target_dose_rate_0_1m"

DEFAULT_DATASETS = {
    "env": PROJECT_ROOT / "data" / "processed" / "train_env_v1.csv",
    "nuclide": PROJECT_ROOT / "data" / "processed" / "train_nuclide_v1.csv",
}

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "spatial_feature_ablation_outputs"

SOIL_FEATURES = [
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
]

TERRAIN_FEATURES = [
    "elevation_m",
    "slope_deg_final",
    "twi_scaled",
]

CONTAMINATION_NUCLIDES = [
    "cs137_kBq_m2",
    "sr90_kBq_m2",
    "ratio_cs_sr",
]

NATURAL_NUCLIDES = [
    "k40_Bq_kg",
    "ra226_Bq_kg",
    "th232_Bq_kg",
]

ALL_ENV = SOIL_FEATURES + TERRAIN_FEATURES
ALL_NUCLIDE = ALL_ENV + CONTAMINATION_NUCLIDES + NATURAL_NUCLIDES


@dataclass
class Config:
    target_col: str = DEFAULT_TARGET_COL
    proxy_target_col: str = DEFAULT_PROXY_TARGET_COL
    random_state: int = DEFAULT_RANDOM_STATE
    cv_splits: int = 5
    use_log1p_target: bool = False
    output_dir: Path = DEFAULT_OUTPUT_DIR
    block_size_deg: float = 0.02
    lat_col: str = "latitude"
    lon_col: str = "longitude"
    with_gradient_boosting: bool = True
    include_linear: bool = True
    include_ridge: bool = True
    include_random_forest: bool = True
    include_extra_trees: bool = True


@dataclass
class DatasetBundle:
    name: str
    df: pd.DataFrame
    target_col: str
    lat_col: str
    lon_col: str


def safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    return (
        float(mean_absolute_error(y_true, y_pred)),
        safe_rmse(y_true, y_pred),
        float(r2_score(y_true, y_pred)),
    )


def maybe_wrap_target_transform(model, cfg: Config):
    if not cfg.use_log1p_target:
        return model
    return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)


def make_model_registry(cfg: Config) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {}

    if cfg.include_linear:
        models["linear_regression"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )

    if cfg.include_ridge:
        models["ridge_cv"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=np.logspace(-3, 3, 13))),
            ]
        )

    if cfg.include_random_forest:
        models["random_forest"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=250,
                        min_samples_leaf=2,
                        random_state=cfg.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    if cfg.include_extra_trees:
        models["extra_trees"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=250,
                        min_samples_leaf=2,
                        random_state=cfg.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    if cfg.with_gradient_boosting:
        models["gradient_boosting"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=250,
                        learning_rate=0.05,
                        max_depth=4,
                        random_state=cfg.random_state,
                    ),
                ),
            ]
        )

    return {name: maybe_wrap_target_transform(model, cfg) for name, model in models.items()}


def load_dataset(name: str, path: Path, cfg: Config) -> DatasetBundle:
    print(f"[DEBUG] Loading dataset: {name}")
    print(f"[DEBUG] Path: {path}")
    print(f"[DEBUG] Exists: {path.exists()}")

    df = pd.read_csv(path)

    required_cols = [cfg.target_col, cfg.lat_col, cfg.lon_col]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"[{name}] Missing required columns: {missing_required}")

    if df[cfg.target_col].isna().any():
        raise ValueError(f"[{name}] Target contains NaN values.")
    if df[cfg.lat_col].isna().any() or df[cfg.lon_col].isna().any():
        raise ValueError(f"[{name}] Spatial validation requires non-null latitude/longitude.")

    duplicated_codes = int(df["Code"].duplicated().sum()) if "Code" in df.columns else 0
    if duplicated_codes:
        warnings.warn(
            f"[{name}] Found {duplicated_codes} duplicated Code values. Rows are kept as-is.",
            RuntimeWarning,
        )

    return DatasetBundle(
        name=name,
        df=df,
        target_col=cfg.target_col,
        lat_col=cfg.lat_col,
        lon_col=cfg.lon_col,
    )


def make_spatial_groups(df: pd.DataFrame, cfg: Config) -> pd.Series:
    lat_bin = np.floor(df[cfg.lat_col] / cfg.block_size_deg).astype(int)
    lon_bin = np.floor(df[cfg.lon_col] / cfg.block_size_deg).astype(int)
    return lat_bin.astype(str) + "_" + lon_bin.astype(str)


def save_group_summary(bundle: DatasetBundle, groups: pd.Series, cfg: Config) -> None:
    work = bundle.df[[bundle.lat_col, bundle.lon_col]].copy()
    work["spatial_group"] = groups.values

    summary = (
        work.groupby("spatial_group", as_index=False)
        .agg(
            n_points=(bundle.lat_col, "size"),
            lat_min=(bundle.lat_col, "min"),
            lat_max=(bundle.lat_col, "max"),
            lon_min=(bundle.lon_col, "min"),
            lon_max=(bundle.lon_col, "max"),
        )
        .sort_values("n_points", ascending=False)
    )

    summary.to_csv(cfg.output_dir / f"spatial_groups_{bundle.name}.csv", index=False)


def available_feature_sets(bundle: DatasetBundle) -> Dict[str, List[str]]:
    cols = set(bundle.df.columns)

    def present(features: List[str]) -> List[str]:
        return [f for f in features if f in cols]

    env = present(ALL_ENV)
    contam = present(CONTAMINATION_NUCLIDES)
    natural = present(NATURAL_NUCLIDES)

    sets: Dict[str, List[str]] = {
        "env_only": env,
        "soil_only": present(SOIL_FEATURES),
        "terrain_only": present(TERRAIN_FEATURES),
        "soil_plus_terrain": env,
    }

    if bundle.name == "nuclide":
        sets.update(
            {
                "nuclide_only": contam + natural,
                "contamination_only": contam,
                "natural_only": natural,
                "env_plus_full_nuclide": env + contam + natural,
                "env_plus_contamination_only": env + contam,
                "env_plus_natural_only": env + natural,
                "env_plus_no_natural": env + contam,
                "env_plus_no_ratio": env + [f for f in contam + natural if f != "ratio_cs_sr"],
                "env_plus_no_k40_ra226_th232": env + contam,
                "env_plus_cs137_only": env + [f for f in ["cs137_kBq_m2"] if f in cols],
                "env_plus_sr90_only": env + [f for f in ["sr90_kBq_m2"] if f in cols],
                "env_plus_cs137_sr90": env + [f for f in ["cs137_kBq_m2", "sr90_kBq_m2"] if f in cols],
            }
        )

    clean_sets = {}
    for name, features in sets.items():
        unique_features = []
        seen = set()
        for f in features:
            if f not in seen:
                unique_features.append(f)
                seen.add(f)
        if unique_features:
            clean_sets[name] = unique_features

    return clean_sets


def run_group_cv_for_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    dataset_name: str,
    feature_set_name: str,
    model_name: str,
    cfg: Config,
) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    gkf = GroupKFold(n_splits=cfg.cv_splits)

    fold_rows = []
    pred_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        fitted = model.fit(X_train, y_train)
        train_pred = fitted.predict(X_train)
        test_pred = fitted.predict(X_test)

        train_mae, train_rmse, train_r2 = evaluate_metrics(y_train, train_pred)
        test_mae, test_rmse, test_r2 = evaluate_metrics(y_test, test_pred)

        fold_rows.append(
            {
                "dataset": dataset_name,
                "feature_set": feature_set_name,
                "model": model_name,
                "fold": fold_idx,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "train_groups": groups.iloc[train_idx].nunique(),
                "test_groups": groups.iloc[test_idx].nunique(),
                "train_mae": train_mae,
                "test_mae": test_mae,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "train_r2": train_r2,
                "test_r2": test_r2,
            }
        )

        pred_rows.append(
            pd.DataFrame(
                {
                    "dataset": dataset_name,
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "fold": fold_idx,
                    "row_index": X_test.index,
                    "y_true": y_test.values,
                    "y_pred": test_pred,
                    "abs_error": np.abs(y_test.values - test_pred),
                    "spatial_group": groups.iloc[test_idx].values,
                }
            )
        )

    folds_df = pd.DataFrame(fold_rows)
    preds_df = pd.concat(pred_rows, ignore_index=True)

    summary = {
        "dataset": dataset_name,
        "feature_set": feature_set_name,
        "model": model_name,
        "n_rows": len(X),
        "n_features": X.shape[1],
        "feature_columns": json.dumps(list(X.columns), ensure_ascii=False),
        "n_spatial_groups": groups.nunique(),
        "block_size_deg": cfg.block_size_deg,
        "cv_train_mae_mean": float(folds_df["train_mae"].mean()),
        "cv_test_mae_mean": float(folds_df["test_mae"].mean()),
        "cv_train_rmse_mean": float(folds_df["train_rmse"].mean()),
        "cv_test_rmse_mean": float(folds_df["test_rmse"].mean()),
        "cv_train_r2_mean": float(folds_df["train_r2"].mean()),
        "cv_test_r2_mean": float(folds_df["test_r2"].mean()),
        "cv_test_mae_std": float(folds_df["test_mae"].std()),
        "cv_test_rmse_std": float(folds_df["test_rmse"].std()),
        "cv_test_r2_std": float(folds_df["test_r2"].std()),
    }

    return summary, folds_df, preds_df


def save_dataset_summary(bundles: List[DatasetBundle], cfg: Config) -> None:
    rows = []
    for bundle in bundles:
        df = bundle.df
        rows.append(
            {
                "dataset": bundle.name,
                "rows": len(df),
                "columns": len(df.columns),
                "target_min": float(df[cfg.target_col].min()),
                "target_max": float(df[cfg.target_col].max()),
                "target_mean": float(df[cfg.target_col].mean()),
                "target_std": float(df[cfg.target_col].std()),
                "missing_total": int(df.isna().sum().sum()),
                "duplicated_code_count": int(df["Code"].duplicated().sum()) if "Code" in df.columns else 0,
                "lat_min": float(df[cfg.lat_col].min()),
                "lat_max": float(df[cfg.lat_col].max()),
                "lon_min": float(df[cfg.lon_col].min()),
                "lon_max": float(df[cfg.lon_col].max()),
            }
        )
    pd.DataFrame(rows).to_csv(cfg.output_dir / "dataset_summary.csv", index=False)


def save_feature_set_catalog(bundles: List[DatasetBundle], cfg: Config) -> None:
    rows = []
    for bundle in bundles:
        feature_sets = available_feature_sets(bundle)
        for set_name, features in feature_sets.items():
            rows.append(
                {
                    "dataset": bundle.name,
                    "feature_set": set_name,
                    "n_features": len(features),
                    "feature_columns": json.dumps(features, ensure_ascii=False),
                }
            )
    pd.DataFrame(rows).to_csv(cfg.output_dir / "feature_set_catalog.csv", index=False)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Spatial feature ablation for Ivankiv dose_rate datasets.")
    parser.add_argument("--target", default=DEFAULT_TARGET_COL)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--log-target", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--block-size-deg", type=float, default=0.02)
    parser.add_argument("--lat-col", default="latitude")
    parser.add_argument("--lon-col", default="longitude")
    parser.add_argument("--no-gradient-boosting", action="store_true")
    parser.add_argument("--linear-only", action="store_true")
    parser.add_argument("--tree-only", action="store_true")
    args = parser.parse_args()

    include_linear = True
    include_ridge = True
    include_random_forest = True
    include_extra_trees = True

    if args.linear_only:
        include_random_forest = False
        include_extra_trees = False
    if args.tree_only:
        include_linear = False
        include_ridge = False

    return Config(
        target_col=args.target,
        cv_splits=args.cv_splits,
        use_log1p_target=args.log_target,
        output_dir=Path(args.output_dir),
        block_size_deg=args.block_size_deg,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
        with_gradient_boosting=not args.no_gradient_boosting,
        include_linear=include_linear,
        include_ridge=include_ridge,
        include_random_forest=include_random_forest,
        include_extra_trees=include_extra_trees,
    )


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    bundles = [load_dataset(name, path, cfg) for name, path in DEFAULT_DATASETS.items()]
    save_dataset_summary(bundles, cfg)
    save_feature_set_catalog(bundles, cfg)

    models = make_model_registry(cfg)

    summary_rows = []
    all_folds = []
    all_preds = []

    for bundle in bundles:
        print(f"\n=== Dataset: {bundle.name} ===")
        groups = make_spatial_groups(bundle.df, cfg)
        print(f"Spatial groups: {groups.nunique()} | block_size_deg={cfg.block_size_deg}")
        save_group_summary(bundle, groups, cfg)

        y = bundle.df[bundle.target_col]
        feature_sets = available_feature_sets(bundle)

        for feature_set_name, feature_cols in feature_sets.items():
            X = bundle.df[feature_cols].copy()
            print(f"\n  Feature set: {feature_set_name} | n_features={len(feature_cols)}")
            print("  Columns:", ", ".join(feature_cols))

            for model_name, model in models.items():
                print(f"    -> {model_name}")
                summary, folds_df, preds_df = run_group_cv_for_model(
                    model=model,
                    X=X,
                    y=y,
                    groups=groups,
                    dataset_name=bundle.name,
                    feature_set_name=feature_set_name,
                    model_name=model_name,
                    cfg=cfg,
                )
                summary_rows.append(summary)
                all_folds.append(folds_df)
                all_preds.append(preds_df)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["dataset", "cv_test_rmse_mean", "cv_test_mae_mean", "cv_test_r2_mean"],
        ascending=[True, True, True, False],
    )
    folds_df = pd.concat(all_folds, ignore_index=True)
    preds_df = pd.concat(all_preds, ignore_index=True)

    summary_df["r2_gap_train_test"] = summary_df["cv_train_r2_mean"] - summary_df["cv_test_r2_mean"]
    summary_df["rmse_gap_train_test"] = summary_df["cv_test_rmse_mean"] - summary_df["cv_train_rmse_mean"]
    summary_df["mae_gap_train_test"] = summary_df["cv_test_mae_mean"] - summary_df["cv_train_mae_mean"]

    summary_df.to_csv(cfg.output_dir / "spatial_feature_ablation_summary.csv", index=False)
    folds_df.to_csv(cfg.output_dir / "spatial_feature_ablation_fold_results.csv", index=False)
    preds_df.to_csv(cfg.output_dir / "spatial_feature_ablation_predictions.csv", index=False)

    best_by_dataset = (
        summary_df.sort_values(
            ["dataset", "cv_test_rmse_mean", "cv_test_mae_mean", "cv_test_r2_mean"],
            ascending=[True, True, True, False],
        )
        .groupby("dataset", as_index=False)
        .first()
    )
    best_by_dataset.to_csv(cfg.output_dir / "spatial_feature_ablation_best_by_dataset.csv", index=False)

    best_by_feature_set = (
        summary_df.sort_values(
            ["dataset", "feature_set", "cv_test_rmse_mean", "cv_test_mae_mean", "cv_test_r2_mean"],
            ascending=[True, True, True, True, False],
        )
        .groupby(["dataset", "feature_set"], as_index=False)
        .first()
    )
    best_by_feature_set.to_csv(cfg.output_dir / "spatial_feature_ablation_best_by_feature_set.csv", index=False)

    print("\nSaved results:")
    print("-", cfg.output_dir / "spatial_feature_ablation_summary.csv")
    print("-", cfg.output_dir / "spatial_feature_ablation_fold_results.csv")
    print("-", cfg.output_dir / "spatial_feature_ablation_predictions.csv")
    print("-", cfg.output_dir / "spatial_feature_ablation_best_by_dataset.csv")
    print("-", cfg.output_dir / "spatial_feature_ablation_best_by_feature_set.csv")
    print("\nTop rows:")
    print(summary_df.head(20).to_string(index=False))

    print("\nNotes:")
    print("- Validation is GroupKFold over spatial grid blocks.")
    print("- Coordinates are used only for grouping, not as model features.")
    print("- Recommended reference setting: --block-size-deg 0.02")


if __name__ == "__main__":
    main()