from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "baseline_outputs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "baseline_analysis"


def safe_divide(a: float, b: float) -> float:
    if b == 0:
        return np.nan
    return float(a / b)


def prepare_summary_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Расширяет baseline_results_summary.csv дополнительными аналитическими колонками.
    """
    df = results_df.copy()

    df["r2_gap_train_test"] = df["train_r2"] - df["test_r2"]
    df["rmse_gap_train_test"] = df["test_rmse"] - df["train_rmse"]
    df["mae_gap_train_test"] = df["test_mae"] - df["train_mae"]
    df["rmse_ratio_test_train"] = [
        safe_divide(test_rmse, train_rmse)
        for train_rmse, test_rmse in zip(df["train_rmse"], df["test_rmse"])
    ]

    if {"cv_r2_mean", "test_r2"}.issubset(df.columns):
        df["cv_test_r2_gap"] = df["cv_r2_mean"] - df["test_r2"]
    else:
        df["cv_test_r2_gap"] = np.nan

    order_df = df.sort_values(
        ["dataset", "test_rmse", "test_mae", "test_r2"],
        ascending=[True, True, True, False],
    ).copy()

    order_df["rank_within_dataset"] = order_df.groupby("dataset").cumcount() + 1
    return order_df


def get_best_models(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Выбирает лучшую модель внутри каждого датасета.
    Критерий: минимальный test_rmse, затем test_mae, затем максимальный test_r2.
    """
    best_rows = []
    for dataset_name, group in summary_df.groupby("dataset", sort=False):
        best_rows.append(
            group.sort_values(
                ["test_rmse", "test_mae", "test_r2"],
                ascending=[True, True, False],
            ).iloc[0]
        )
    return pd.DataFrame(best_rows).reset_index(drop=True)


def parse_prediction_filename(path: Path) -> Optional[Tuple[str, str]]:
    match = re.match(r"^predictions_(?P<dataset>[^_]+)_(?P<model>.+)\.csv$", path.name)
    if not match:
        return None
    return match.group("dataset"), match.group("model")


def parse_importance_filename(path: Path) -> Optional[Tuple[str, str]]:
    match = re.match(r"^feature_importance_(?P<dataset>[^_]+)_(?P<model>.+)\.csv$", path.name)
    if not match:
        return None
    return match.group("dataset"), match.group("model")


def summarize_prediction_files(input_dir: Path) -> pd.DataFrame:
    """
    Собирает summary по prediction-файлам:
    bias, median abs error, p90 abs error, max abs error, correlation true/pred.
    """
    rows = []

    for path in sorted(input_dir.glob("predictions_*.csv")):
        parsed = parse_prediction_filename(path)
        if parsed is None:
            continue

        dataset_name, model_name = parsed
        df = pd.read_csv(path)

        residual = df["y_pred"] - df["y_true"]
        abs_error = np.abs(residual)

        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "n_test_rows": len(df),
                "mean_error": float(residual.mean()),
                "median_error": float(np.median(residual)),
                "mae_from_predictions": float(abs_error.mean()),
                "median_abs_error": float(np.median(abs_error)),
                "p90_abs_error": float(np.quantile(abs_error, 0.90)),
                "max_abs_error": float(abs_error.max()),
                "corr_true_pred": float(np.corrcoef(df["y_true"], df["y_pred"])[0, 1]) if len(df) > 1 else np.nan,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["dataset", "mae_from_predictions", "p90_abs_error"])


def choose_importance_column(df: pd.DataFrame) -> Optional[str]:
    for col in ("importance", "abs_coefficient", "coefficient"):
        if col in df.columns:
            return col
    return None


def aggregate_feature_importance(input_dir: Path) -> pd.DataFrame:
    """
    Агрегирует importance по моделям внутри датасета.
    Для линейных моделей использует abs(coef), для деревьев -- feature_importances_.
    Затем нормализует внутри модели и усредняет.
    """
    rows = []

    for path in sorted(input_dir.glob("feature_importance_*.csv")):
        parsed = parse_importance_filename(path)
        if parsed is None:
            continue

        dataset_name, model_name = parsed
        df = pd.read_csv(path)

        score_col = choose_importance_column(df)
        if score_col is None or "feature" not in df.columns:
            continue

        work = df[["feature", score_col]].copy()
        work = work.rename(columns={score_col: "raw_score"})
        work["raw_score"] = work["raw_score"].astype(float).abs()

        total = work["raw_score"].sum()
        work["normalized_score"] = work["raw_score"] / total if total > 0 else 0.0
        work["dataset"] = dataset_name
        work["model"] = model_name

        rows.append(work)

    if not rows:
        return pd.DataFrame()

    long_df = pd.concat(rows, ignore_index=True)

    agg = (
        long_df.groupby(["dataset", "feature"], as_index=False)
        .agg(
            mean_normalized_score=("normalized_score", "mean"),
            median_normalized_score=("normalized_score", "median"),
            models_count=("model", "nunique"),
        )
        .sort_values(["dataset", "mean_normalized_score"], ascending=[True, False])
    )

    return agg


def plot_scatter_true_vs_pred(pred_df: pd.DataFrame, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(pred_df["y_true"], pred_df["y_pred"], alpha=0.7)

    min_val = float(min(pred_df["y_true"].min(), pred_df["y_pred"].min()))
    max_val = float(max(pred_df["y_true"].max(), pred_df["y_pred"].max()))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    ax.set_xlabel("y_true")
    ax.set_ylabel("y_pred")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_residuals(pred_df: pd.DataFrame, title: str, output_path: Path) -> None:
    residual = pred_df["y_pred"] - pred_df["y_true"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(pred_df["y_true"], residual, alpha=0.7)
    ax.axhline(0.0, linestyle="--")

    ax.set_xlabel("y_true")
    ax.set_ylabel("residual = y_pred - y_true")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_abs_error_hist(pred_df: pd.DataFrame, title: str, output_path: Path) -> None:
    abs_error = np.abs(pred_df["y_pred"] - pred_df["y_true"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(abs_error, bins=30)

    ax.set_xlabel("absolute error")
    ax.set_ylabel("count")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_top_features(feature_df: pd.DataFrame, title: str, output_path: Path, top_n: int = 10) -> None:
    top_df = (
        feature_df.sort_values("mean_normalized_score", ascending=False)
        .head(top_n)
        .sort_values("mean_normalized_score")
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_df["feature"], top_df["mean_normalized_score"])

    ax.set_xlabel("mean normalized importance")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_prediction_diagnostics_for_best_models(
    best_models_df: pd.DataFrame,
    input_dir: Path,
    output_dir: Path,
) -> None:
    """
    Для лучшей модели каждого датасета строит:
    - y_true vs y_pred
    - residuals vs y_true
    - histogram(abs_error)
    - top20 worst predictions
    """
    for _, row in best_models_df.iterrows():
        dataset_name = row["dataset"]
        model_name = row["model"]

        pred_path = input_dir / f"predictions_{dataset_name}_{model_name}.csv"
        if not pred_path.exists():
            continue

        pred_df = pd.read_csv(pred_path)
        prefix = output_dir / f"{dataset_name}_{model_name}"

        plot_scatter_true_vs_pred(
            pred_df,
            title=f"{dataset_name} | {model_name} | y_true vs y_pred",
            output_path=prefix.with_name(prefix.name + "_scatter_true_vs_pred.png"),
        )

        plot_residuals(
            pred_df,
            title=f"{dataset_name} | {model_name} | residuals vs y_true",
            output_path=prefix.with_name(prefix.name + "_residuals.png"),
        )

        plot_abs_error_hist(
            pred_df,
            title=f"{dataset_name} | {model_name} | abs error histogram",
            output_path=prefix.with_name(prefix.name + "_abs_error_hist.png"),
        )

        worst_df = pred_df.assign(
            abs_error=np.abs(pred_df["y_pred"] - pred_df["y_true"])
        ).sort_values("abs_error", ascending=False)

        worst_df.head(20).to_csv(
            prefix.with_name(prefix.name + "_top20_worst_predictions.csv"),
            index=False,
        )


def save_feature_plots(
    agg_importance_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 10,
) -> None:
    for dataset_name, group in agg_importance_df.groupby("dataset", sort=False):
        plot_top_features(
            feature_df=group,
            title=f"{dataset_name} | aggregated feature importance",
            output_path=output_dir / f"{dataset_name}_aggregated_feature_importance.png",
            top_n=top_n,
        )


def fmt_or_nan(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value:.3f}"


def build_markdown_report(
    summary_df: pd.DataFrame,
    best_models_df: pd.DataFrame,
    pred_error_df: pd.DataFrame,
) -> str:
    """
    Строит короткий markdown-report по baseline-analysis.
    """
    lines: List[str] = []
    lines.append("# Baseline analysis report")
    lines.append("")
    lines.append("## Best model by dataset")
    lines.append("")

    for _, row in best_models_df.iterrows():
        lines.append(
            f"- **{row['dataset']}**: `{row['model']}` | "
            f"test_R2={row['test_r2']:.3f}, test_RMSE={row['test_rmse']:.4f}, "
            f"cv_R2={fmt_or_nan(row.get('cv_r2_mean', np.nan))}"
        )
    lines.append("")

    if {"env", "nuclide"}.issubset(set(best_models_df["dataset"])):
        env_row = best_models_df.loc[best_models_df["dataset"] == "env"].iloc[0]
        nuc_row = best_models_df.loc[best_models_df["dataset"] == "nuclide"].iloc[0]

        lines.append("## Env vs nuclide")
        lines.append("")
        lines.append(
            f"- Best env model: `{env_row['model']}` with test_R2={env_row['test_r2']:.3f} "
            f"and cv_R2={fmt_or_nan(env_row.get('cv_r2_mean', np.nan))}."
        )
        lines.append(
            f"- Best nuclide model: `{nuc_row['model']}` with test_R2={nuc_row['test_r2']:.3f} "
            f"and cv_R2={fmt_or_nan(nuc_row.get('cv_r2_mean', np.nan))}."
        )
        lines.append(
            f"- Absolute gain in test_R2 from env to nuclide: {(nuc_row['test_r2'] - env_row['test_r2']):.3f}."
        )

        cv_gain = nuc_row.get("cv_r2_mean", np.nan) - env_row.get("cv_r2_mean", np.nan)
        lines.append(f"- Absolute gain in cv_R2 from env to nuclide: {fmt_or_nan(cv_gain)}.")
        lines.append("")

    lines.append("## Overfitting view")
    lines.append("")
    for _, row in best_models_df.iterrows():
        lines.append(
            f"- `{row['dataset']} / {row['model']}`: "
            f"train_R2={row['train_r2']:.3f}, "
            f"test_R2={row['test_r2']:.3f}, "
            f"gap={row['r2_gap_train_test']:.3f}."
        )
    lines.append("")

    if not pred_error_df.empty:
        lines.append("## Prediction error notes")
        lines.append("")
        merged = pred_error_df.merge(
            best_models_df[["dataset", "model"]],
            on=["dataset", "model"],
            how="inner",
        )

        for _, row in merged.iterrows():
            lines.append(
                f"- `{row['dataset']} / {row['model']}`: "
                f"median_abs_error={row['median_abs_error']:.4f}, "
                f"p90_abs_error={row['p90_abs_error']:.4f}, "
                f"max_abs_error={row['max_abs_error']:.4f}."
            )
        lines.append("")

    lines.append("## Recommended next steps")
    lines.append("")
    lines.append("- Inspect aggregated feature importance for physical plausibility.")
    lines.append("- Review top worst predictions for systematic failures.")
    lines.append("- Re-run baseline with `--log-target` and compare.")
    lines.append("- Add spatial validation, because random split may be optimistic for geospatial data.")
    lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-hoc analysis for baseline regression outputs.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-n-features", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = input_dir / "baseline_results_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    raw_summary_df = pd.read_csv(summary_path)
    summary_df = prepare_summary_table(raw_summary_df)
    summary_df.to_csv(output_dir / "baseline_results_enriched.csv", index=False)

    best_models_df = get_best_models(summary_df)
    best_models_df.to_csv(output_dir / "best_models_by_dataset.csv", index=False)

    pred_error_df = summarize_prediction_files(input_dir)
    if not pred_error_df.empty:
        pred_error_df.to_csv(output_dir / "prediction_error_summary.csv", index=False)

    agg_importance_df = aggregate_feature_importance(input_dir)
    if not agg_importance_df.empty:
        agg_importance_df.to_csv(output_dir / "feature_importance_aggregated.csv", index=False)
        save_feature_plots(
            agg_importance_df,
            output_dir=output_dir,
            top_n=args.top_n_features,
        )

    save_prediction_diagnostics_for_best_models(
        best_models_df,
        input_dir=input_dir,
        output_dir=output_dir,
    )

    report_text = build_markdown_report(summary_df, best_models_df, pred_error_df)
    (output_dir / "baseline_analysis_report.md").write_text(report_text, encoding="utf-8")

    print("Saved analysis outputs to:", output_dir)
    print("Created files:")
    print("- baseline_results_enriched.csv")
    print("- best_models_by_dataset.csv")
    if not pred_error_df.empty:
        print("- prediction_error_summary.csv")
    if not agg_importance_df.empty:
        print("- feature_importance_aggregated.csv")
        print("- aggregated feature importance plots")
    print("- diagnostics plots for best models")
    print("- baseline_analysis_report.md")


if __name__ == "__main__":
    main()