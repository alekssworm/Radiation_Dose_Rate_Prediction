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

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "spatial_baseline_outputs"


@dataclass
class Config:
    target_col: str = DEFAULT_TARGET_COL
    proxy_target_col: str = DEFAULT_PROXY_TARGET_COL
    random_state: int = DEFAULT_RANDOM_STATE
    cv_splits: int = 5
    use_log1p_target: bool = False
    drop_coords: bool = True
    drop_code: bool = True
    drop_proxy_targets: bool = True
    include_xgboost: bool = False
    include_catboost: bool = False
    output_dir: Path = DEFAULT_OUTPUT_DIR
    block_size_deg: float = 0.01
    lat_col: str = "latitude"
    lon_col: str = "longitude"


@dataclass
class DatasetBundle:
    name: str
    df: pd.DataFrame
    feature_cols: List[str]
    target_col: str
    lat_col: str
    lon_col: str


def safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_feature_columns(df: pd.DataFrame, cfg: Config) -> List[str]:
    excluded = {cfg.target_col}
    if cfg.drop_proxy_targets:
        excluded.add(cfg.proxy_target_col)
    if cfg.drop_coords:
        excluded.update([cfg.lat_col, cfg.lon_col])
    if cfg.drop_code:
        excluded.add("Code")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in excluded]


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

    feature_cols = build_feature_columns(df, cfg)
    if not feature_cols:
        raise ValueError(f"[{name}] No usable feature columns after exclusions.")

    duplicated_codes = int(df["Code"].duplicated().sum()) if "Code" in df.columns else 0
    if duplicated_codes:
        warnings.warn(
            f"[{name}] Found {duplicated_codes} duplicated Code values. Rows are kept as-is.",
            RuntimeWarning,
        )

    return DatasetBundle(
        name=name,
        df=df,
        feature_cols=feature_cols,
        target_col=cfg.target_col,
        lat_col=cfg.lat_col,
        lon_col=cfg.lon_col,
    )


def maybe_wrap_target_transform(model, cfg: Config):
    if not cfg.use_log1p_target:
        return model
    return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)


def make_model_registry(cfg: Config) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {
        "linear_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "ridge_cv": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=np.logspace(-3, 3, 13))),
            ]
        ),
        "random_forest": Pipeline(
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
        ),
        "extra_trees": Pipeline(
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
        ),
        "gradient_boosting": Pipeline(
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
        ),
    }

    if cfg.include_xgboost:
        try:
            from xgboost import XGBRegressor

            models["xgboost"] = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        XGBRegressor(
                            n_estimators=300,
                            max_depth=5,
                            learning_rate=0.05,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            reg_lambda=1.0,
                            objective="reg:squarederror",
                            random_state=cfg.random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
        except Exception as exc:
            warnings.warn(f"xgboost requested, but import failed: {exc}")

    if cfg.include_catboost:
        try:
            from catboost import CatBoostRegressor

            models["catboost"] = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        CatBoostRegressor(
                            iterations=300,
                            depth=6,
                            learning_rate=0.05,
                            loss_function="RMSE",
                            verbose=False,
                            random_state=cfg.random_state,
                        ),
                    ),
                ]
            )
        except Exception as exc:
            warnings.warn(f"catboost requested, but import failed: {exc}")

    return {name: maybe_wrap_target_transform(model, cfg) for name, model in models.items()}


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    return (
        float(mean_absolute_error(y_true, y_pred)),
        safe_rmse(y_true, y_pred),
        float(r2_score(y_true, y_pred)),
    )


def make_spatial_groups(df: pd.DataFrame, cfg: Config) -> pd.Series:
    lat_bin = np.floor(df[cfg.lat_col] / cfg.block_size_deg).astype(int)
    lon_bin = np.floor(df[cfg.lon_col] / cfg.block_size_deg).astype(int)
    groups = lat_bin.astype(str) + "_" + lon_bin.astype(str)
    return groups


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


def run_group_cv_for_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    dataset_name: str,
    model_name: str,
    cfg: Config,
) -> Tuple[dict, pd.DataFrame]:
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
        "model": model_name,
        "n_rows": len(X),
        "n_features": X.shape[1],
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

    return summary, folds_df.merge(
        preds_df.groupby(["dataset", "model", "fold"], as_index=False).agg(
            pred_rows=("row_index", "count")
        ),
        on=["dataset", "model", "fold"],
        how="left",
    ), preds_df


def save_dataset_summary(bundles: List[DatasetBundle], cfg: Config) -> None:
    rows = []
    for bundle in bundles:
        df = bundle.df
        rows.append(
            {
                "dataset": bundle.name,
                "rows": len(df),
                "columns": len(df.columns),
                "n_features_used": len(bundle.feature_cols),
                "target_min": float(df[cfg.target_col].min()),
                "target_max": float(df[cfg.target_col].max()),
                "target_mean": float(df[cfg.target_col].mean()),
                "target_std": float(df[cfg.target_col].std()),
                "missing_total": int(df.isna().sum().sum()),
                "duplicated_code_count": int(df["Code"].duplicated().sum()) if "Code" in df.columns else 0,
                "feature_columns": json.dumps(bundle.feature_cols, ensure_ascii=False),
                "lat_min": float(df[cfg.lat_col].min()),
                "lat_max": float(df[cfg.lat_col].max()),
                "lon_min": float(df[cfg.lon_col].min()),
                "lon_max": float(df[cfg.lon_col].max()),
            }
        )
    pd.DataFrame(rows).to_csv(cfg.output_dir / "dataset_summary.csv", index=False)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Spatial baseline regression for Ivankiv dose_rate datasets.")
    parser.add_argument("--target", default=DEFAULT_TARGET_COL)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--log-target", action="store_true")
    parser.add_argument("--include-coords", action="store_true")
    parser.add_argument("--keep-code", action="store_true")
    parser.add_argument("--include-proxy-target", action="store_true")
    parser.add_argument("--with-xgboost", action="store_true")
    parser.add_argument("--with-catboost", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--block-size-deg", type=float, default=0.01)
    parser.add_argument("--lat-col", default="latitude")
    parser.add_argument("--lon-col", default="longitude")
    args = parser.parse_args()

    return Config(
        target_col=args.target,
        cv_splits=args.cv_splits,
        use_log1p_target=args.log_target,
        drop_coords=not args.include_coords,
        drop_code=not args.keep_code,
        drop_proxy_targets=not args.include_proxy_target,
        include_xgboost=args.with_xgboost,
        include_catboost=args.with_catboost,
        output_dir=Path(args.output_dir),
        block_size_deg=args.block_size_deg,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
    )


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    bundles = [load_dataset(name, path, cfg) for name, path in DEFAULT_DATASETS.items()]
    save_dataset_summary(bundles, cfg)
    models = make_model_registry(cfg)

    summary_rows = []
    all_folds = []
    all_preds = []

    for bundle in bundles:
        print(f"\n=== Dataset: {bundle.name} ===")
        print(f"Rows: {len(bundle.df)} | Features used: {len(bundle.feature_cols)}")
        print("Features:", ", ".join(bundle.feature_cols))

        groups = make_spatial_groups(bundle.df, cfg)
        print(f"Spatial groups: {groups.nunique()} | block_size_deg={cfg.block_size_deg}")
        save_group_summary(bundle, groups, cfg)

        X = bundle.df[bundle.feature_cols]
        y = bundle.df[bundle.target_col]

        for model_name, model in models.items():
            print(f"  -> {model_name}")
            summary, folds_df, preds_df = run_group_cv_for_model(
                model=model,
                X=X,
                y=y,
                groups=groups,
                dataset_name=bundle.name,
                model_name=model_name,
                cfg=cfg,
            )

            summary_rows.append(summary)
            all_folds.append(folds_df)
            all_preds.append(preds_df)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["dataset", "cv_test_rmse_mean", "cv_test_mae_mean"],
        ascending=[True, True, True],
    )
    folds_df = pd.concat(all_folds, ignore_index=True)
    preds_df = pd.concat(all_preds, ignore_index=True)

    summary_df.to_csv(cfg.output_dir / "spatial_baseline_results_summary.csv", index=False)
    folds_df.to_csv(cfg.output_dir / "spatial_cv_fold_results.csv", index=False)
    preds_df.to_csv(cfg.output_dir / "spatial_cv_predictions.csv", index=False)

    print("\nSaved results:")
    print("-", cfg.output_dir / "spatial_baseline_results_summary.csv")
    print("-", cfg.output_dir / "spatial_cv_fold_results.csv")
    print("-", cfg.output_dir / "spatial_cv_predictions.csv")
    print(summary_df.to_string(index=False))

    print("\nNotes:")
    print("- Validation is GroupKFold over spatial grid blocks.")
    print("- Coordinates are used for grouping, not necessarily as model features.")
    print("- block_size_deg controls spatial strictness; try 0.01 and 0.02 for sensitivity checks.")


if __name__ == "__main__":
    main()