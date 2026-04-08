"""Compare two MVP-B scenario prediction tables.

This script merges two prediction CSVs produced by `predict_mvp_b.py` and
computes row-wise and aggregate deltas.

Main outputs:
- merged comparison table with baseline/policy predictions and deltas;
- summary CSV with aggregate statistics;
- top-changed rows by absolute delta;
- top-changed rows by percent delta.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


DEFAULT_PREDICTION_COLUMN = "predicted_target_dose_rate"
DEFAULT_ID_CANDIDATES = ["point_id", "source_row_index", "Code"]
DEFAULT_OUTPUT_DIR = Path("mvp_b") / "outputs" / "scenario_comparisons"
DEFAULT_TOP_N = 20


@dataclass(frozen=True)
class CompareConfig:
    baseline_csv: str
    scenario_csv: str
    output_dir: str
    comparison_name: str
    id_col: str | None
    prediction_col: str
    top_n: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two MVP-B scenario prediction CSVs.")
    parser.add_argument(
        "--baseline-csv",
        required=True,
        type=str,
        help="Path to baseline prediction CSV.",
    )
    parser.add_argument(
        "--scenario-csv",
        required=True,
        type=str,
        help="Path to scenario prediction CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        type=str,
        help="Directory where comparison outputs will be saved.",
    )
    parser.add_argument(
        "--comparison-name",
        default=None,
        type=str,
        help="Optional output name prefix. Defaults to '<baseline>__vs__<scenario>'.",
    )
    parser.add_argument(
        "--id-col",
        default=None,
        type=str,
        help="Explicit merge key. If omitted, the script will try common identifier columns.",
    )
    parser.add_argument(
        "--prediction-col",
        default=DEFAULT_PREDICTION_COLUMN,
        type=str,
        help="Prediction column name.",
    )
    parser.add_argument(
        "--top-n",
        default=DEFAULT_TOP_N,
        type=int,
        help="Number of top changed rows to export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison_name = args.comparison_name or make_default_comparison_name(
        baseline_csv=args.baseline_csv,
        scenario_csv=args.scenario_csv,
    )
    config = CompareConfig(
        baseline_csv=args.baseline_csv,
        scenario_csv=args.scenario_csv,
        output_dir=args.output_dir,
        comparison_name=comparison_name,
        id_col=args.id_col,
        prediction_col=args.prediction_col,
        top_n=int(args.top_n),
    )

    baseline_df = load_prediction_csv(Path(config.baseline_csv), config.prediction_col)
    scenario_df = load_prediction_csv(Path(config.scenario_csv), config.prediction_col)

    id_col = resolve_id_column(
        baseline_df=baseline_df,
        scenario_df=scenario_df,
        explicit_id_col=config.id_col,
    )

    baseline_name = resolve_scenario_label(baseline_df, fallback="baseline")
    scenario_name = resolve_scenario_label(scenario_df, fallback="scenario")

    comparison_df = build_comparison_table(
        baseline_df=baseline_df,
        scenario_df=scenario_df,
        id_col=id_col,
        prediction_col=config.prediction_col,
        baseline_name=baseline_name,
        scenario_name=scenario_name,
    )

    summary_df = build_summary_table(
        comparison_df=comparison_df,
        baseline_name=baseline_name,
        scenario_name=scenario_name,
    )
    top_abs_df = build_top_changed_table(comparison_df, top_n=config.top_n, sort_col="delta_abs_abs")
    top_pct_df = build_top_changed_table(comparison_df, top_n=config.top_n, sort_col="delta_pct_abs")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = output_dir / f"{config.comparison_name}_comparison.csv"
    summary_path = output_dir / f"{config.comparison_name}_summary.csv"
    top_abs_path = output_dir / f"{config.comparison_name}_top_abs_delta.csv"
    top_pct_path = output_dir / f"{config.comparison_name}_top_pct_delta.csv"
    summary_json_path = output_dir / f"{config.comparison_name}_summary.json"

    comparison_df.to_csv(comparison_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    top_abs_df.to_csv(top_abs_path, index=False)
    top_pct_df.to_csv(top_pct_path, index=False)
    write_json(summary_json_path, summary_df.iloc[0].to_dict())

    print(f"[OK] Merge key: {id_col}")
    print(f"[OK] Baseline label: {baseline_name}")
    print(f"[OK] Scenario label: {scenario_name}")
    print(f"[OK] Rows compared: {len(comparison_df)}")
    print(f"[OK] Comparison CSV: {comparison_path}")
    print(f"[OK] Summary CSV: {summary_path}")
    print(f"[OK] Top abs delta CSV: {top_abs_path}")
    print(f"[OK] Top pct delta CSV: {top_pct_path}")


def load_prediction_csv(path: Path, prediction_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prediction CSV was not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Prediction CSV is empty: {path}")
    if prediction_col not in df.columns:
        raise KeyError(f"Prediction column {prediction_col!r} is missing from: {path}")

    return df.reset_index(drop=True)


def resolve_id_column(
    baseline_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    explicit_id_col: str | None,
) -> str:
    if explicit_id_col is not None:
        if explicit_id_col not in baseline_df.columns:
            raise KeyError(f"Explicit id column {explicit_id_col!r} is missing from baseline CSV.")
        if explicit_id_col not in scenario_df.columns:
            raise KeyError(f"Explicit id column {explicit_id_col!r} is missing from scenario CSV.")
        return explicit_id_col

    for candidate in DEFAULT_ID_CANDIDATES:
        if candidate in baseline_df.columns and candidate in scenario_df.columns:
            return candidate

    shared_columns = sorted(set(baseline_df.columns).intersection(set(scenario_df.columns)))
    raise KeyError(
        "Could not resolve a shared identifier column automatically. "
        f"Tried {DEFAULT_ID_CANDIDATES}. Shared columns were: {shared_columns}"
    )


def resolve_scenario_label(df: pd.DataFrame, fallback: str) -> str:
    if "scenario_name" in df.columns and df["scenario_name"].nunique(dropna=True) == 1:
        value = str(df["scenario_name"].dropna().iloc[0])
        if value:
            return value
    return fallback


def make_default_comparison_name(baseline_csv: str, scenario_csv: str) -> str:
    baseline_name = Path(baseline_csv).stem
    scenario_name = Path(scenario_csv).stem
    return f"{baseline_name}__vs__{scenario_name}"


def build_comparison_table(
    baseline_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    id_col: str,
    prediction_col: str,
    baseline_name: str,
    scenario_name: str,
) -> pd.DataFrame:
    left = baseline_df.copy()
    right = scenario_df.copy()

    common_context_cols = [
        col for col in ["latitude", "longitude", "model_mode", "feature_set", "model_name"]
        if col in left.columns and col in right.columns and col != id_col
    ]

    left_keep = [id_col, prediction_col, *common_context_cols]
    right_keep = [id_col, prediction_col]

    left = left[left_keep].copy()
    right = right[right_keep].copy()

    left = left.rename(columns={prediction_col: f"prediction_{baseline_name}"})
    right = right.rename(columns={prediction_col: f"prediction_{scenario_name}"})

    merged = left.merge(right, on=id_col, how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("The merged comparison table is empty. Check that both files share the same identifier values.")

    baseline_pred_col = f"prediction_{baseline_name}"
    scenario_pred_col = f"prediction_{scenario_name}"

    merged["delta_abs"] = merged[scenario_pred_col] - merged[baseline_pred_col]
    merged["delta_abs_abs"] = merged["delta_abs"].abs()

    baseline_nonzero = merged[baseline_pred_col].replace(0.0, np.nan)
    merged["delta_pct"] = (merged["delta_abs"] / baseline_nonzero) * 100.0
    merged["delta_pct_abs"] = merged["delta_pct"].abs()

    merged["baseline_label"] = baseline_name
    merged["scenario_label"] = scenario_name
    return merged


def build_summary_table(
    comparison_df: pd.DataFrame,
    baseline_name: str,
    scenario_name: str,
) -> pd.DataFrame:
    baseline_pred_col = f"prediction_{baseline_name}"
    scenario_pred_col = f"prediction_{scenario_name}"

    summary = {
        "baseline_label": baseline_name,
        "scenario_label": scenario_name,
        "n_rows": int(len(comparison_df)),
        "baseline_mean": float(comparison_df[baseline_pred_col].mean()),
        "scenario_mean": float(comparison_df[scenario_pred_col].mean()),
        "baseline_median": float(comparison_df[baseline_pred_col].median()),
        "scenario_median": float(comparison_df[scenario_pred_col].median()),
        "delta_abs_mean": float(comparison_df["delta_abs"].mean()),
        "delta_abs_median": float(comparison_df["delta_abs"].median()),
        "delta_abs_min": float(comparison_df["delta_abs"].min()),
        "delta_abs_max": float(comparison_df["delta_abs"].max()),
        "delta_abs_std": float(comparison_df["delta_abs"].std(ddof=0)),
        "delta_pct_mean": float(comparison_df["delta_pct"].dropna().mean()),
        "delta_pct_median": float(comparison_df["delta_pct"].dropna().median()),
        "delta_pct_min": float(comparison_df["delta_pct"].dropna().min()),
        "delta_pct_max": float(comparison_df["delta_pct"].dropna().max()),
        "rows_delta_positive": int((comparison_df["delta_abs"] > 0).sum()),
        "rows_delta_negative": int((comparison_df["delta_abs"] < 0).sum()),
        "rows_delta_zero": int((comparison_df["delta_abs"] == 0).sum()),
    }
    return pd.DataFrame([summary])


def build_top_changed_table(
    comparison_df: pd.DataFrame,
    top_n: int,
    sort_col: str,
) -> pd.DataFrame:
    if sort_col not in comparison_df.columns:
        raise KeyError(f"Sort column is missing from comparison table: {sort_col!r}")

    top_df = comparison_df.sort_values(sort_col, ascending=False).head(top_n).reset_index(drop=True)
    return top_df


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
