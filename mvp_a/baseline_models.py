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
from sklearn.model_selection import KFold, cross_validate, train_test_split
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

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "baseline_outputs"
@dataclass
class Config:
    target_col: str = DEFAULT_TARGET_COL
    proxy_target_col: str = DEFAULT_PROXY_TARGET_COL
    test_size: float = 0.2
    random_state: int = DEFAULT_RANDOM_STATE
    run_cv: bool = True
    cv_splits: int = 5
    use_log1p_target: bool = False
    drop_coords: bool = True
    drop_code: bool = True
    drop_proxy_targets: bool = True
    include_xgboost: bool = False
    include_catboost: bool = False
    output_dir: Path = DEFAULT_OUTPUT_DIR


@dataclass
class DatasetBundle:
    name: str
    df: pd.DataFrame
    feature_cols: List[str]
    target_col: str


@dataclass
class EvaluationResult:
    dataset: str
    model: str
    n_rows: int
    n_features: int
    train_mae: float
    test_mae: float
    train_rmse: float
    test_rmse: float
    train_r2: float
    test_r2: float
    cv_mae_mean: Optional[float] = None
    cv_mae_std: Optional[float] = None
    cv_rmse_mean: Optional[float] = None
    cv_rmse_std: Optional[float] = None
    cv_r2_mean: Optional[float] = None
    cv_r2_std: Optional[float] = None


def safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_bins(y: pd.Series, max_bins: int = 5) -> Optional[pd.Series]:
    try:
        n_unique = y.nunique(dropna=True)
        n_bins = min(max_bins, n_unique)
        if n_bins < 2:
            return None
        bins = pd.qcut(y, q=n_bins, duplicates="drop")
        if bins.nunique() < 2:
            return None
        return bins.astype(str)
    except Exception:
        return None


def build_feature_columns(df: pd.DataFrame, cfg: Config) -> List[str]:
    excluded = {cfg.target_col}
    if cfg.drop_proxy_targets:
        excluded.add(cfg.proxy_target_col)
    if cfg.drop_coords:
        excluded.update(["latitude", "longitude"])
    if cfg.drop_code:
        excluded.add("Code")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in excluded]


def load_dataset(name: str, path: Path, cfg: Config) -> DatasetBundle:
    print(f"[DEBUG] Loading dataset: {name}")
    print(f"[DEBUG] Path: {path}")
    print(f"[DEBUG] Exists: {path.exists()}")

    df = pd.read_csv(path)
    if cfg.target_col not in df.columns:
        raise ValueError(f"[{name}] Missing target column: {cfg.target_col}")
    if df[cfg.target_col].isna().any():
        raise ValueError(f"[{name}] Target contains NaN values.")

    feature_cols = build_feature_columns(df, cfg)
    if not feature_cols:
        raise ValueError(f"[{name}] No usable feature columns after exclusions.")

    duplicated_codes = int(df["Code"].duplicated().sum()) if "Code" in df.columns else 0
    if duplicated_codes:
        warnings.warn(
            f"[{name}] Found {duplicated_codes} duplicated Code values. Rows are kept as-is.",
            RuntimeWarning,
        )

    return DatasetBundle(name=name, df=df, feature_cols=feature_cols, target_col=cfg.target_col)


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


def extract_feature_importance(fitted_model, feature_cols: List[str]) -> Optional[pd.DataFrame]:
    estimator = fitted_model.regressor_ if hasattr(fitted_model, "regressor_") else fitted_model

    if not hasattr(estimator, "named_steps"):
        return None

    model = estimator.named_steps.get("model")
    if model is None:
        return None

    if hasattr(model, "feature_importances_"):
        return pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values(
            "importance", ascending=False
        )

    if hasattr(model, "coef_"):
        coef = np.ravel(model.coef_)
        return pd.DataFrame(
            {"feature": feature_cols, "coefficient": coef, "abs_coefficient": np.abs(coef)}
        ).sort_values("abs_coefficient", ascending=False)

    return None


def run_cross_validation(model, X: pd.DataFrame, y: pd.Series, cfg: Config) -> Dict[str, Optional[float]]:
    cv = KFold(n_splits=cfg.cv_splits, shuffle=True, random_state=cfg.random_state)
    scoring = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
    return {
        "cv_mae_mean": float(-scores["test_mae"].mean()),
        "cv_mae_std": float(scores["test_mae"].std()),
        "cv_rmse_mean": float(-scores["test_rmse"].mean()),
        "cv_rmse_std": float(scores["test_rmse"].std()),
        "cv_r2_mean": float(scores["test_r2"].mean()),
        "cv_r2_std": float(scores["test_r2"].std()),
    }


def save_predictions(output_dir: Path, dataset_name: str, model_name: str, test_index: pd.Index, y_true: pd.Series, y_pred: np.ndarray) -> None:
    pd.DataFrame(
        {
            "row_index": test_index,
            "y_true": y_true.values,
            "y_pred": y_pred,
            "abs_error": np.abs(y_true.values - y_pred),
        }
    ).to_csv(output_dir / f"predictions_{dataset_name}_{model_name}.csv", index=False)


def evaluate_dataset(bundle: DatasetBundle, models: Dict[str, Pipeline], cfg: Config) -> List[EvaluationResult]:
    X = bundle.df[bundle.feature_cols]
    y = bundle.df[bundle.target_col]
    stratify_bins = regression_bins(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=stratify_bins if stratify_bins is not None else None,
    )

    results: List[EvaluationResult] = []
    for model_name, model in models.items():
        print(f"  -> {model_name}")
        fitted = model.fit(X_train, y_train)
        train_pred = fitted.predict(X_train)
        test_pred = fitted.predict(X_test)

        train_mae, train_rmse, train_r2 = evaluate_metrics(y_train, train_pred)
        test_mae, test_rmse, test_r2 = evaluate_metrics(y_test, test_pred)
        cv_stats = run_cross_validation(model, X, y, cfg) if cfg.run_cv else {}

        results.append(
            EvaluationResult(
                dataset=bundle.name,
                model=model_name,
                n_rows=len(bundle.df),
                n_features=len(bundle.feature_cols),
                train_mae=train_mae,
                test_mae=test_mae,
                train_rmse=train_rmse,
                test_rmse=test_rmse,
                train_r2=train_r2,
                test_r2=test_r2,
                **cv_stats,
            )
        )

        save_predictions(cfg.output_dir, bundle.name, model_name, X_test.index, y_test, test_pred)
        importance_df = extract_feature_importance(fitted, bundle.feature_cols)
        if importance_df is not None:
            importance_df.to_csv(cfg.output_dir / f"feature_importance_{bundle.name}_{model_name}.csv", index=False)

    return results


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
            }
        )
    pd.DataFrame(rows).to_csv(cfg.output_dir / "dataset_summary.csv", index=False)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Baseline regression for Ivankiv dose_rate datasets.")
    parser.add_argument("--target", default=DEFAULT_TARGET_COL)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--no-cv", action="store_true")
    parser.add_argument("--log-target", action="store_true")
    parser.add_argument("--include-coords", action="store_true")
    parser.add_argument("--keep-code", action="store_true")
    parser.add_argument("--include-proxy-target", action="store_true")
    parser.add_argument("--with-xgboost", action="store_true")
    parser.add_argument("--with-catboost", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    return Config(
        target_col=args.target,
        test_size=args.test_size,
        run_cv=not args.no_cv,
        cv_splits=args.cv_splits,
        use_log1p_target=args.log_target,
        drop_coords=not args.include_coords,
        drop_code=not args.keep_code,
        drop_proxy_targets=not args.include_proxy_target,
        include_xgboost=args.with_xgboost,
        include_catboost=args.with_catboost,
        output_dir=Path(args.output_dir),
    )


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    bundles = [load_dataset(name, path, cfg) for name, path in DEFAULT_DATASETS.items()]
    save_dataset_summary(bundles, cfg)
    models = make_model_registry(cfg)

    all_results: List[EvaluationResult] = []
    for bundle in bundles:
        print(f"\n=== Dataset: {bundle.name} ===")
        print(f"Rows: {len(bundle.df)} | Features used: {len(bundle.feature_cols)}")
        print("Features:", ", ".join(bundle.feature_cols))
        all_results.extend(evaluate_dataset(bundle, models, cfg))

    results_df = pd.DataFrame([vars(r) for r in all_results]).sort_values(
        ["dataset", "test_rmse", "test_mae"], ascending=[True, True, True]
    )
    results_path = cfg.output_dir / "baseline_results_summary.csv"
    results_df.to_csv(results_path, index=False)

    print("\nSaved results:", results_path)
    print(results_df.to_string(index=False))
    print("\nNotes:")
    print("- By default, Code / latitude / longitude are excluded from features.")
    print("- By default, target_dose_rate_0_1m is excluded to avoid leakage / proxy effects.")
    print("- Current validation = random split + KFold. Later, spatial validation is recommended.")


if __name__ == "__main__":
    main()