"""
Quality gate / validation for the synthetic fleet dataset.

This script validates the three raw CSV tables:
  - data/raw/vehicle_day.csv
  - data/raw/dtc_event.csv
  - data/raw/work_order.csv

It writes:
  - outputs/tables/validation_report.json   (machine-readable)
  - outputs/tables/validation_summary_table.csv/.tex  (LaTeX-friendly)

Run:
  python -m src.data.validate --config config.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from src.utils.artifacts import save_json, save_table


@dataclass
class Check:
    """Represents a single validation check and its result."""
    name: str
    passed: bool
    details: Dict[str, Any]


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config (UTF-8) into a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml did not parse into a dictionary.")
    return cfg


def read_csv_required(path: Path) -> pd.DataFrame:
    """Read a CSV that must exist; fail fast if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def coverage_non_null(df: pd.DataFrame, cols: List[str]) -> float:
    """Average non-null coverage across (rows x selected columns)."""
    if len(cols) == 0:
        return 1.0
    return float(df[cols].notna().mean().mean())


def check_required_columns(df: pd.DataFrame, required: List[str], table: str) -> Check:
    """Verify that all required columns are present."""
    missing = [c for c in required if c not in df.columns]
    return Check(
        name=f"{table}: required columns present",
        passed=(len(missing) == 0),
        details={"missing_columns": missing, "n_cols": int(df.shape[1])},
    )


def check_min_coverage(df: pd.DataFrame, cols: List[str], min_cov: float, table: str) -> Check:
    """Verify the table is sufficiently complete (non-null coverage)."""
    cov = coverage_non_null(df, cols)
    return Check(
        name=f"{table}: non-null coverage >= {min_cov}",
        passed=(cov >= min_cov),
        details={"coverage": cov, "min_required": min_cov},
    )


def check_unique_key(df: pd.DataFrame, key_cols: List[str], table: str) -> Check:
    """Verify uniqueness of key columns."""
    if df.empty:
        return Check(name=f"{table}: unique key {key_cols}", passed=True, details={"note": "empty table"})
    dup = int(df.duplicated(subset=key_cols).sum())
    return Check(name=f"{table}: unique key {key_cols}", passed=(dup == 0), details={"duplicates": dup})


def check_range(df: pd.DataFrame, col: str, lo: float, hi: float, table: str) -> Check:
    """Verify numeric values are within [lo, hi]."""
    if col not in df.columns:
        return Check(name=f"{table}: range {col} in [{lo},{hi}]", passed=False, details={"error": "missing_col"})
    s = pd.to_numeric(df[col], errors="coerce")
    bad = ((s < lo) | (s > hi)) & s.notna()
    n_bad = int(bad.sum())
    details: Dict[str, Any] = {"n_bad": n_bad}
    if s.notna().any():
        details["min"] = float(np.nanmin(s.values))
        details["max"] = float(np.nanmax(s.values))
    else:
        details["min"] = None
        details["max"] = None
    return Check(name=f"{table}: range {col} in [{lo},{hi}]", passed=(n_bad == 0), details=details)


def check_in_set(df: pd.DataFrame, col: str, allowed: List[Any], table: str, allow_empty: bool = False) -> Check:
    """Verify categorical values belong to a known allowed set."""
    if col not in df.columns:
        return Check(name=f"{table}: {col} in allowed set", passed=False, details={"error": "missing_col"})
    s = df[col].astype("string")
    if allow_empty:
        s = s[(s.notna()) & (s != "")]
    else:
        s = s[s.notna()]
    bad = ~s.isin(allowed)
    n_bad = int(bad.sum())
    examples = s[bad].head(5).tolist()
    return Check(name=f"{table}: {col} in allowed set", passed=(n_bad == 0), details={"n_bad": n_bad, "examples": examples})


def check_binary(df: pd.DataFrame, col: str, table: str) -> Check:
    """Verify column values are binary (0/1)."""
    if col not in df.columns:
        return Check(name=f"{table}: {col} binary", passed=False, details={"error": "missing_col"})
    s = pd.to_numeric(df[col], errors="coerce")
    bad = (~s.isin([0, 1])) & s.notna()
    return Check(name=f"{table}: {col} binary", passed=(int(bad.sum()) == 0), details={"n_bad": int(bad.sum())})


def check_datetime_parseable(df: pd.DataFrame, col: str, table: str) -> Check:
    """Verify datetime strings can be parsed (no NaT after parsing)."""
    if col not in df.columns:
        return Check(name=f"{table}: {col} parseable datetime", passed=False, details={"error": "missing_col"})
    parsed = pd.to_datetime(df[col], errors="coerce")
    return Check(name=f"{table}: {col} parseable datetime", passed=(int(parsed.isna().sum()) == 0), details={"n_bad": int(parsed.isna().sum())})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    raw_dir = Path(cfg["paths"]["data_raw_dir"])
    out_tables = Path(cfg["paths"]["tables_dir"])

    p_vehicle = raw_dir / "vehicle_day.csv"
    p_dtc = raw_dir / "dtc_event.csv"
    p_wo = raw_dir / "work_order.csv"

    df_vehicle = read_csv_required(p_vehicle)
    df_dtc = read_csv_required(p_dtc)
    df_wo = read_csv_required(p_wo)

    subsystems = list(cfg["simulation"]["subsystems"])
    route_types = list(cfg["simulation"]["route_weather"]["route_types"]) + ["shop"]
    min_cov = float(cfg["agent"]["guardrails"]["min_required_coverage"])
    telem = list(cfg["simulation"]["telematics_features"])

    checks: List[Check] = []

    # vehicle_day schema is our "main contract" because models/train depend on it.
    required_vehicle = (
        ["date","vehicle_id","route_type","hilliness_index","stop_go_index","ambient_temp_c","precipitation",
         "duty_cycle","miles_today","mileage_total"]
        + telem
        + ["dtc_count_today","spc_alert_today","breakdown_7d","breakdown_30d","subsystem_label"]
    )

    checks.append(check_required_columns(df_vehicle, required_vehicle, "vehicle_day"))
    checks.append(check_min_coverage(df_vehicle, required_vehicle, min_cov, "vehicle_day"))
    checks.append(check_unique_key(df_vehicle, ["vehicle_id", "date"], "vehicle_day"))
    checks.append(check_datetime_parseable(df_vehicle, "date", "vehicle_day"))

    checks.append(check_in_set(df_vehicle, "route_type", route_types, "vehicle_day"))
    checks.append(check_in_set(df_vehicle, "subsystem_label", subsystems, "vehicle_day", allow_empty=True))

    checks.append(check_range(df_vehicle, "hilliness_index", 0.0, 1.0, "vehicle_day"))
    checks.append(check_range(df_vehicle, "stop_go_index", 0.0, 1.0, "vehicle_day"))
    checks.append(check_range(df_vehicle, "duty_cycle", 0.0, 1.0, "vehicle_day"))
    checks.append(check_range(df_vehicle, "idle_pct", 0.0, 100.0, "vehicle_day"))
    checks.append(check_range(df_vehicle, "miles_today", 0.0, 2000.0, "vehicle_day"))
    checks.append(check_range(df_vehicle, "mileage_total", 0.0, 2_000_000.0, "vehicle_day"))
    checks.append(check_range(df_vehicle, "ambient_temp_c", -60.0, 60.0, "vehicle_day"))

    checks.append(check_binary(df_vehicle, "precipitation", "vehicle_day"))
    checks.append(check_binary(df_vehicle, "spc_alert_today", "vehicle_day"))
    checks.append(check_binary(df_vehicle, "breakdown_7d", "vehicle_day"))
    checks.append(check_binary(df_vehicle, "breakdown_30d", "vehicle_day"))

    # dtc_event checks
    required_dtc = ["timestamp", "vehicle_id", "dtc_code", "subsystem", "severity", "count"]
    checks.append(check_required_columns(df_dtc, required_dtc, "dtc_event"))
    checks.append(check_min_coverage(df_dtc, required_dtc, min_cov, "dtc_event"))
    if not df_dtc.empty:
        checks.append(check_datetime_parseable(df_dtc, "timestamp", "dtc_event"))
        checks.append(check_in_set(df_dtc, "subsystem", subsystems, "dtc_event"))
        checks.append(check_range(df_dtc, "severity", 1.0, 3.0, "dtc_event"))
        checks.append(check_range(df_dtc, "count", 1.0, 10_000.0, "dtc_event"))

    # work_order checks
    required_wo = ["open_date","close_date","vehicle_id","subsystem","action","parts_lead_time_days","downtime_days","notes"]
    checks.append(check_required_columns(df_wo, required_wo, "work_order"))
    checks.append(check_min_coverage(df_wo, required_wo, min_cov, "work_order"))
    if not df_wo.empty:
        checks.append(check_datetime_parseable(df_wo, "open_date", "work_order"))
        checks.append(check_datetime_parseable(df_wo, "close_date", "work_order"))
        checks.append(check_in_set(df_wo, "subsystem", subsystems, "work_order"))
        checks.append(check_range(df_wo, "parts_lead_time_days", 0.0, 365.0, "work_order"))
        checks.append(check_range(df_wo, "downtime_days", 1.0, 365.0, "work_order"))

        od = pd.to_datetime(df_wo["open_date"], errors="coerce")
        cd = pd.to_datetime(df_wo["close_date"], errors="coerce")
        bad = (cd < od) & od.notna() & cd.notna()
        checks.append(Check(name="work_order: close_date >= open_date", passed=(int(bad.sum()) == 0), details={"n_bad": int(bad.sum())}))

    passed = sum(1 for c in checks if c.passed)
    failed = [c for c in checks if not c.passed]

    report = {
        "files": {"vehicle_day": str(p_vehicle), "dtc_event": str(p_dtc), "work_order": str(p_wo)},
        "shapes": {"vehicle_day": [int(df_vehicle.shape[0]), int(df_vehicle.shape[1])],
                   "dtc_event": [int(df_dtc.shape[0]), int(df_dtc.shape[1])],
                   "work_order": [int(df_wo.shape[0]), int(df_wo.shape[1])]},
        "checks_total": len(checks),
        "checks_passed": passed,
        "checks_failed": len(failed),
        "failed": [{"name": c.name, "details": c.details} for c in failed],
    }

    # Machine-readable report (for CI / debugging)
    save_json(report, out_tables / "validation_report.json")

    # LaTeX-friendly summary table (for the final report)
    summary_df = pd.DataFrame(
        [
            {"item": "vehicle_day rows", "value": int(df_vehicle.shape[0])},
            {"item": "vehicle_day cols", "value": int(df_vehicle.shape[1])},
            {"item": "dtc_event rows", "value": int(df_dtc.shape[0])},
            {"item": "work_order rows", "value": int(df_wo.shape[0])},
            {"item": "checks passed", "value": int(passed)},
            {"item": "checks failed", "value": int(len(failed))},
        ]
    )
    save_table(
        summary_df,
        out_dir=out_tables,
        stem="validation_summary_table",
        caption="Data validation summary (quality gate).",
        label="tab:validation_summary",
        index=False,
        float_decimals=0,
    )

    print("=== Data validation summary ===")
    print(f"vehicle_day: rows={df_vehicle.shape[0]} cols={df_vehicle.shape[1]}")
    print(f"dtc_event:   rows={df_dtc.shape[0]} cols={df_dtc.shape[1]}")
    print(f"work_order:  rows={df_wo.shape[0]} cols={df_wo.shape[1]}")
    print(f"checks: passed {passed}/{len(checks)}")
    print("report: outputs/tables/validation_report.json")
    print("latex table: outputs/tables/validation_summary_table.tex")

    if failed:
        print("\\nFAILED CHECKS:")
        for c in failed:
            print(f"- {c.name}: {c.details}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()