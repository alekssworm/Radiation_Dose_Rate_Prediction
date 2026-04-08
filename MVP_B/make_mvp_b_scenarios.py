"""Create scenario input CSV files for MVP-B prediction.

This script takes a base nuclide-capable tabular dataset and generates a small
set of scenario CSVs for controlled comparison runs with `predict_mvp_b.py`.

Default scenario pack:
- baseline
- cs137_plus_20
- cs137_minus_20
- sr90_plus_20
- remediation_light

The generated files are intended for `nuclide_mode` prediction with the current
MVP-B primary model based on the `env_plus_no_ratio` feature set.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd


DEFAULT_ID_COLUMN = "point_id"
DEFAULT_OUTPUT_DIR = Path("MVP_B") / "examples" / "scenarios"
TARGET_COLUMNS_TO_DROP = ["target_dose_rate", "target_dose_rate_0_1m"]

REQUIRED_NUCLIDE_SCENARIO_COLUMNS = [
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
    "cs137_kBq_m2",
    "sr90_kBq_m2",
    "k40_Bq_kg",
    "ra226_Bq_kg",
    "th232_Bq_kg",
]

OPTIONAL_CONTEXT_COLUMNS = [
    "latitude",
    "longitude",
    "Code",
    "ratio_cs_sr",
]


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    description: str
    transform: Callable[[pd.DataFrame], pd.DataFrame]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create scenario CSVs for MVP-B prediction.")
    parser.add_argument(
        "--input-csv",
        required=True,
        type=str,
        help="Path to a base nuclide-capable input CSV, e.g. train_nuclide_v1.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        type=str,
        help="Directory where scenario CSVs will be saved.",
    )
    parser.add_argument(
        "--id-col",
        default=DEFAULT_ID_COLUMN,
        type=str,
        help="Preferred row identifier column. If missing, it will be created.",
    )
    parser.add_argument(
        "--keep-target-columns",
        action="store_true",
        help="Keep target columns if they exist. By default they are dropped for inference files.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write only scenario-required columns plus identifier/context columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)

    base_df = load_base_dataframe(input_path)
    base_df = ensure_id_column(base_df, id_col=args.id_col)
    validate_required_columns(base_df)

    scenario_base = prepare_base_scenario_frame(
        df=base_df,
        id_col=args.id_col,
        keep_target_columns=bool(args.keep_target_columns),
        compact=bool(args.compact),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_specs = build_default_scenarios()

    print(f"[OK] Loaded base input: {input_path}")
    print(f"[OK] Rows: {len(scenario_base)}")
    print(f"[OK] Output directory: {output_dir}")

    manifest_rows: List[Dict[str, object]] = []

    for spec in scenario_specs:
        scenario_df = spec.transform(scenario_base.copy())
        scenario_df["scenario_name"] = spec.name

        output_path = output_dir / f"scenario_{spec.name}.csv"
        scenario_df.to_csv(output_path, index=False)

        manifest_rows.append(
            {
                "scenario_name": spec.name,
                "description": spec.description,
                "output_csv": str(output_path),
                "n_rows": int(len(scenario_df)),
                "cs137_min": float(scenario_df["cs137_kBq_m2"].min()),
                "cs137_max": float(scenario_df["cs137_kBq_m2"].max()),
                "sr90_min": float(scenario_df["sr90_kBq_m2"].min()),
                "sr90_max": float(scenario_df["sr90_kBq_m2"].max()),
            }
        )

        print(f"[OK] Saved scenario: {spec.name} -> {output_path}")

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "scenario_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"[OK] Saved manifest: {manifest_path}")


def load_base_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Base input CSV was not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Base input CSV is empty: {path}")

    return df.reset_index(drop=True)


def ensure_id_column(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    out = df.copy()
    if id_col not in out.columns:
        out[id_col] = np.arange(len(out), dtype=int)
    return out


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_NUCLIDE_SCENARIO_COLUMNS if col not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(
            "Base input CSV is missing required columns for MVP-B nuclide scenarios: "
            f"{missing_str}"
        )

    null_columns = [col for col in REQUIRED_NUCLIDE_SCENARIO_COLUMNS if df[col].isna().any()]
    if null_columns:
        null_str = ", ".join(null_columns)
        raise ValueError(f"Required scenario columns contain null values: {null_str}")


def prepare_base_scenario_frame(
    df: pd.DataFrame,
    id_col: str,
    keep_target_columns: bool,
    compact: bool,
) -> pd.DataFrame:
    out = df.copy()

    if not keep_target_columns:
        drop_cols = [col for col in TARGET_COLUMNS_TO_DROP if col in out.columns]
        if drop_cols:
            out = out.drop(columns=drop_cols)

    if compact:
        columns: List[str] = [id_col]
        columns.extend(col for col in OPTIONAL_CONTEXT_COLUMNS if col in out.columns and col not in columns)
        columns.extend(col for col in REQUIRED_NUCLIDE_SCENARIO_COLUMNS if col in out.columns and col not in columns)
        out = out[columns].copy()

    return out


def build_default_scenarios() -> List[ScenarioSpec]:
    return [
        ScenarioSpec(
            name="baseline",
            description="Unmodified baseline contamination table.",
            transform=lambda df: df,
        ),
        ScenarioSpec(
            name="cs137_plus_20",
            description="Increase Cs-137 by 20 percent.",
            transform=lambda df: scale_column(df, "cs137_kBq_m2", 1.20),
        ),
        ScenarioSpec(
            name="cs137_minus_20",
            description="Decrease Cs-137 by 20 percent.",
            transform=lambda df: scale_column(df, "cs137_kBq_m2", 0.80),
        ),
        ScenarioSpec(
            name="sr90_plus_20",
            description="Increase Sr-90 by 20 percent.",
            transform=lambda df: scale_column(df, "sr90_kBq_m2", 1.20),
        ),
        ScenarioSpec(
            name="remediation_light",
            description="Light remediation scenario: Cs-137 x0.70 and Sr-90 x0.85.",
            transform=light_remediation_transform,
        ),
    ]


def scale_column(df: pd.DataFrame, column: str, factor: float) -> pd.DataFrame:
    out = df.copy()
    out[column] = out[column].astype(float) * float(factor)
    return out


def light_remediation_transform(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cs137_kBq_m2"] = out["cs137_kBq_m2"].astype(float) * 0.70
    out["sr90_kBq_m2"] = out["sr90_kBq_m2"].astype(float) * 0.85
    return out


if __name__ == "__main__":
    main()
