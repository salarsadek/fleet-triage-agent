"""
src/reporting/eda_stats.py

Generate EDA + statistical artifacts (LaTeX-ready) for the Fleet Maintenance Triage Agent.

Outputs (relative to repo root):
- outputs/tables/eda_overview_table.(csv/tex)
- outputs/tables/work_orders_by_subsystem.(csv/tex)
- outputs/tables/cox_hazard_ratios.(csv/tex)
- outputs/figures/km_duty_cycle.(pdf/png)

Design notes
- Robust to small synthetic datasets (few events).
- Uses a simple survival proxy: time-to-first work order per vehicle.
- Drops near-constant covariates before Cox fit to improve numerical stability.
- Adds a small penalizer for Cox to reduce numerical issues.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

# Prefer project utilities for consistent LaTeX/CSV formatting across the repo.
try:
    from src.utils.artifacts import save_table  # type: ignore
except Exception:
    save_table = None


def _ensure_dir(p: Path) -> None:
    """Create directory if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    """Read CSV with a clear error if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def _to_dt(s: pd.Series) -> pd.Series:
    """Parse datetimes robustly (malformed -> NaT)."""
    return pd.to_datetime(s, errors="coerce")


def _write_json(obj: Dict[str, Any], out_path: Path) -> None:
    """Write a small JSON artifact."""
    _ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _save_table_fallback(
    df: pd.DataFrame,
    out_dir: Path,
    stem: str,
    caption: str,
    label: str,
    index: bool = False,
) -> None:
    """
    Fallback: write CSV + a simple LaTeX table without relying on project utilities.
    This keeps the module runnable even if utils/artifacts.py changes.
    """
    _ensure_dir(out_dir)

    csv_path = out_dir / f"{stem}.csv"
    tex_path = out_dir / f"{stem}.tex"

    df.to_csv(csv_path, index=index)

    safe = df.copy()
    safe.columns = [str(c).replace("_", r"\_") for c in safe.columns]
    for c in safe.columns:
        if safe[c].dtype == "object":
            safe[c] = safe[c].astype(str).str.replace("_", r"\_", regex=False)

    tabular = safe.to_latex(index=index, escape=False)

    tex = "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            tabular,
            r"\end{table}",
            "",
        ]
    )
    tex_path.write_text(tex, encoding="utf-8")


def _save_table(
    df: pd.DataFrame,
    out_dir: Path,
    stem: str,
    caption: str,
    label: str,
    index: bool = False,
) -> None:
    """
    Save table as CSV + LaTeX, using project formatter when available.

    Robustness: different repos implement save_table with different parameter names.
    We try several call styles (keyword + positional), and fall back to a safe writer.
    """
    if save_table is None:
        _save_table_fallback(df, out_dir, stem, caption, label, index=index)
        return

    # 1) Try common keyword variants
    keyword_variants = [
        {"df": df, "out_dir": str(out_dir), "name": stem, "caption": caption, "label": label, "index": index},
        {"df": df, "out_dir": str(out_dir), "filename": stem, "caption": caption, "label": label, "index": index},
        {"df": df, "out_dir": str(out_dir), "stem": stem, "caption": caption, "label": label, "index": index},
        {"df": df, "out_dir": str(out_dir), "table_name": stem, "caption": caption, "label": label, "index": index},
    ]
    for kwargs in keyword_variants:
        try:
            save_table(**kwargs)  # type: ignore[misc]
            return
        except TypeError:
            continue
        except Exception:
            # If the helper exists but fails for data reasons, do not crash EDA.
            break

    # 2) Try positional variants
    positional_variants = [
        (df, str(out_dir), stem, caption, label, index),
        (df, str(out_dir), stem, caption, label),
        (df, out_dir, stem, caption, label),
    ]
    for args in positional_variants:
        try:
            save_table(*args)  # type: ignore[misc]
            return
        except TypeError:
            continue
        except Exception:
            break

    # 3) Fall back if project helper is incompatible
    _save_table_fallback(df, out_dir, stem, caption, label, index=index)


def _save_fig(fig, out_pdf: Path, out_png: Path, dpi: int = 150) -> None:
    """Save figure as PDF (LaTeX) + PNG (preview)."""
    _ensure_dir(out_pdf.parent)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=dpi)


@dataclass
class Paths:
    """Resolved project paths based on config.yaml."""
    repo_root: Path
    data_raw_dir: Path
    outputs_tables: Path
    outputs_figures: Path


def _load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_paths(cfg: Dict[str, Any], repo_root: Path) -> Paths:
    """Resolve paths from config with sensible defaults."""
    paths = cfg.get("paths", {})
    data_raw_dir = repo_root / Path(paths.get("data_raw_dir", "data/raw"))
    tables_dir = repo_root / Path(paths.get("tables_dir", "outputs/tables"))
    figures_dir = repo_root / Path(paths.get("figures_dir", "outputs/figures"))
    return Paths(
        repo_root=repo_root,
        data_raw_dir=data_raw_dir,
        outputs_tables=tables_dir,
        outputs_figures=figures_dir,
    )


def _build_overview_table(
    vehicle_day: pd.DataFrame, dtc_event: pd.DataFrame, work_order: pd.DataFrame
) -> pd.DataFrame:
    """Create a compact overview table used in the report."""
    n_rows = int(vehicle_day.shape[0])
    n_cols = int(vehicle_day.shape[1])

    vehicles = vehicle_day["vehicle_id"].nunique() if "vehicle_id" in vehicle_day.columns else np.nan
    date_min = vehicle_day["date_dt"].min() if "date_dt" in vehicle_day.columns else pd.NaT
    date_max = vehicle_day["date_dt"].max() if "date_dt" in vehicle_day.columns else pd.NaT
    n_days = (date_max - date_min).days + 1 if pd.notna(date_min) and pd.notna(date_max) else np.nan

    prev_7 = float(vehicle_day["breakdown_7d"].mean()) if "breakdown_7d" in vehicle_day.columns else np.nan
    prev_30 = float(vehicle_day["breakdown_30d"].mean()) if "breakdown_30d" in vehicle_day.columns else np.nan

    dtc_rate = float(dtc_event.shape[0]) / float(n_rows) if n_rows > 0 else np.nan

    rows = [
        ("vehicle_day rows", n_rows),
        ("vehicle_day cols", n_cols),
        ("n_vehicles", int(vehicles) if pd.notna(vehicles) else np.nan),
        ("date_min", str(date_min.date()) if pd.notna(date_min) else ""),
        ("date_max", str(date_max.date()) if pd.notna(date_max) else ""),
        ("n_days", int(n_days) if pd.notna(n_days) else np.nan),
        ("dtc_event rows", int(dtc_event.shape[0])),
        ("dtc events per vehicle-day", round(dtc_rate, 5) if pd.notna(dtc_rate) else np.nan),
        ("work_order rows", int(work_order.shape[0])),
        ("label prevalence breakdown_7d", round(prev_7, 5) if pd.notna(prev_7) else np.nan),
        ("label prevalence breakdown_30d", round(prev_30, 5) if pd.notna(prev_30) else np.nan),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def _work_orders_by_subsystem(work_order: pd.DataFrame) -> pd.DataFrame:
    """Summarize work orders by subsystem (counts + optional numeric means)."""
    if work_order.empty or "subsystem" not in work_order.columns:
        return pd.DataFrame([("no_work_orders", 0)], columns=["subsystem", "n_work_orders"])

    df = work_order.copy()

    numeric_cols: List[str] = []
    for c in ["downtime_days", "parts_lead_time_days", "cost"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            numeric_cols.append(c)

    grp = df.groupby("subsystem", dropna=False)
    out = grp.size().rename("n_work_orders").reset_index().sort_values("n_work_orders", ascending=False)

    for c in numeric_cols:
        out[f"mean_{c}"] = grp[c].mean().values

    return out


def _build_vehicle_survival_dataset(
    vehicle_day: pd.DataFrame, work_order: pd.DataFrame
) -> pd.DataFrame:
    """
    Build a per-vehicle survival dataset.

    duration_days: days since start to first work order (or censored at last day)
    event_work_order: 1 if the vehicle has any work order, else 0
    covariates: stable per-vehicle summaries from daily data
    """
    if "vehicle_id" not in vehicle_day.columns or "date_dt" not in vehicle_day.columns:
        raise ValueError("vehicle_day must include 'vehicle_id' and parsed 'date_dt'.")

    df = vehicle_day.copy()
    start_date = df["date_dt"].min().normalize()
    end_date = df["date_dt"].max().normalize()
    total_days = int((end_date - start_date).days + 1)

    wo_first = None
    if (
        not work_order.empty
        and "vehicle_id" in work_order.columns
        and "open_date_dt" in work_order.columns
    ):
        wo_first = (
            work_order.dropna(subset=["open_date_dt"])
            .groupby("vehicle_id")["open_date_dt"]
            .min()
            .apply(lambda x: pd.to_datetime(x).normalize())
        )

    vehicles = sorted(df["vehicle_id"].unique().tolist())
    rows: List[Dict[str, Any]] = []

    for vid in vehicles:
        df_v = df[df["vehicle_id"] == vid].sort_values("date_dt")

        event = 0
        event_date = None
        if wo_first is not None and vid in wo_first.index and pd.notna(wo_first.loc[vid]):
            event = 1
            event_date = wo_first.loc[vid]

        if event_date is None:
            duration = total_days
        else:
            duration = int((event_date - start_date).days + 1)
            duration = max(duration, 1)

        duty_cycle_mean = float(df_v["duty_cycle"].mean()) if "duty_cycle" in df_v.columns else np.nan
        ambient_mean = float(df_v["ambient_temp_c"].mean()) if "ambient_temp_c" in df_v.columns else np.nan

        if "ambient_temp_c" in df_v.columns:
            amb = pd.to_numeric(df_v["ambient_temp_c"], errors="coerce").fillna(ambient_mean)
            temp_extreme_share = float(((amb < -10.0) | (amb > 30.0)).mean())
        else:
            temp_extreme_share = np.nan

        if "dtc_count_today" in df_v.columns:
            dtc_rate = float(pd.to_numeric(df_v["dtc_count_today"], errors="coerce").fillna(0.0).mean())
        else:
            dtc_rate = np.nan

        spc_alert_share = (
            float(pd.to_numeric(df_v["spc_alert_today"], errors="coerce").fillna(0.0).mean())
            if "spc_alert_today" in df_v.columns
            else np.nan
        )

        rows.append(
            {
                "vehicle_id": vid,
                "duration_days": duration,
                "event_work_order": event,
                "duty_cycle_mean": duty_cycle_mean,
                "ambient_temp_mean": ambient_mean,
                "temp_extreme_share": temp_extreme_share,
                "dtc_rate_mean": dtc_rate,
                "spc_alert_share": spc_alert_share,
            }
        )

    out = pd.DataFrame(rows)

    for c in ["duty_cycle_mean", "ambient_temp_mean", "temp_extreme_share", "dtc_rate_mean", "spc_alert_share"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    return out


def _fit_cox_and_export(df_surv: pd.DataFrame, out_tables: Path) -> None:
    """Fit Cox PH model and export hazard ratios table (robust to small data)."""
    try:
        from lifelines import CoxPHFitter
        from lifelines.exceptions import ConvergenceWarning
    except Exception as e:
        note = pd.DataFrame([{"note": f"lifelines not available: {e}"}])
        _save_table(
            note,
            out_tables,
            "cox_hazard_ratios",
            caption="Cox model hazard ratios (lifelines not available).",
            label="tab:cox_hazard_ratios",
            index=False,
        )
        return

    if df_surv.empty or int(df_surv["event_work_order"].sum()) == 0:
        note = pd.DataFrame([{"note": "No events available for Cox model (no work orders)."}])
        _save_table(
            note,
            out_tables,
            "cox_hazard_ratios",
            caption="Cox model hazard ratios (no events in data).",
            label="tab:cox_hazard_ratios",
            index=False,
        )
        return

    duration_col = "duration_days"
    event_col = "event_work_order"

    covariates = [
        "duty_cycle_mean",
        "ambient_temp_mean",
        "temp_extreme_share",
        "dtc_rate_mean",
        "spc_alert_share",
    ]
    covariates = [c for c in covariates if c in df_surv.columns]

    min_var = 1e-4
    variances = df_surv[covariates].var(numeric_only=True).fillna(0.0)
    low_var = variances[variances < min_var].index.tolist()
    if low_var:
        print(f"[eda_stats] Dropping low-variance Cox covariates: {low_var}")
        covariates = [c for c in covariates if c not in low_var]

    if not covariates:
        note = pd.DataFrame([{"note": "All Cox covariates were near-constant; nothing to fit."}])
        _save_table(
            note,
            out_tables,
            "cox_hazard_ratios",
            caption="Cox model hazard ratios (covariates near-constant).",
            label="tab:cox_hazard_ratios",
            index=False,
        )
        return

    cph = CoxPHFitter(penalizer=0.05)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        try:
            df_fit = df_surv[[duration_col, event_col] + covariates].copy()
            cph.fit(df_fit, duration_col=duration_col, event_col=event_col)

            summ = cph.summary.reset_index().rename(columns={"index": "covariate"})
            keep = [
                c for c in [
                    "covariate",
                    "coef",
                    "se(coef)",
                    "p",
                    "exp(coef)",
                    "exp(coef) lower 95%",
                    "exp(coef) upper 95%",
                ]
                if c in summ.columns
            ]
            out = summ[keep].copy().rename(
                columns={
                    "se(coef)": "se_coef",
                    "exp(coef)": "hazard_ratio",
                    "exp(coef) lower 95%": "hr_ci95_low",
                    "exp(coef) upper 95%": "hr_ci95_high",
                }
            )

            _save_table(
                out,
                out_tables,
                "cox_hazard_ratios",
                caption="Cox proportional hazards model (time-to-first work order).",
                label="tab:cox_hazard_ratios",
                index=False,
            )
        except Exception as e:
            fail = pd.DataFrame([{"note": f"Cox fit failed: {type(e).__name__}: {e}"}])
            _save_table(
                fail,
                out_tables,
                "cox_hazard_ratios",
                caption="Cox model hazard ratios (fit failed).",
                label="tab:cox_hazard_ratios",
                index=False,
            )


def _km_plot_by_duty_cycle(df_surv: pd.DataFrame, out_figures: Path) -> None:
    """Kaplan–Meier survival plot split by median duty cycle."""
    try:
        import matplotlib.pyplot as plt
        from lifelines import KaplanMeierFitter
    except Exception as e:
        print(f"[eda_stats] Skipping KM plot (missing deps): {e}")
        return

    if df_surv.empty or int(df_surv["event_work_order"].sum()) == 0:
        print("[eda_stats] Skipping KM plot (empty or no events).")
        return

    if "duty_cycle_mean" not in df_surv.columns:
        print("[eda_stats] Skipping KM plot (missing duty_cycle_mean).")
        return

    median_dc = float(df_surv["duty_cycle_mean"].median())
    df = df_surv.copy()
    df["duty_group"] = np.where(df["duty_cycle_mean"] >= median_dc, "high_duty", "low_duty")

    fig = plt.figure(figsize=(7.0, 4.2))
    ax = fig.add_subplot(1, 1, 1)

    kmf = KaplanMeierFitter()
    for group in ["low_duty", "high_duty"]:
        g = df[df["duty_group"] == group]
        kmf.fit(
            durations=g["duration_days"],
            event_observed=g["event_work_order"],
            label=f"{group} (n={len(g)})",
        )
        kmf.plot_survival_function(ax=ax)

    ax.set_title("Kaplan–Meier survival: time-to-first work order by duty-cycle group")
    ax.set_xlabel("Days since start")
    ax.set_ylabel("Survival probability")
    ax.grid(True, alpha=0.25)

    _save_fig(fig, out_pdf=out_figures / "km_duty_cycle.pdf", out_png=out_figures / "km_duty_cycle.png", dpi=150)
    plt.close(fig)


def generate(cfg: Dict[str, Any], repo_root: Path) -> None:
    """Main generation routine for EDA + stats artifacts."""
    paths = _resolve_paths(cfg, repo_root)

    out_tables = paths.outputs_tables
    out_figures = paths.outputs_figures
    _ensure_dir(out_tables)
    _ensure_dir(out_figures)

    vehicle_day = _read_csv(paths.data_raw_dir / "vehicle_day.csv")
    dtc_event = _read_csv(paths.data_raw_dir / "dtc_event.csv")
    work_order = _read_csv(paths.data_raw_dir / "work_order.csv")

    if "date" in vehicle_day.columns:
        vehicle_day["date_dt"] = _to_dt(vehicle_day["date"])
    elif "date_dt" not in vehicle_day.columns:
        raise ValueError("vehicle_day.csv must include 'date' (or pre-parsed 'date_dt').")

    if "timestamp" in dtc_event.columns:
        dtc_event["timestamp_dt"] = _to_dt(dtc_event["timestamp"])

    if "open_date" in work_order.columns:
        work_order["open_date_dt"] = _to_dt(work_order["open_date"])
    if "close_date" in work_order.columns:
        work_order["close_date_dt"] = _to_dt(work_order["close_date"])

    overview = _build_overview_table(vehicle_day, dtc_event, work_order)
    _save_table(
        overview,
        out_tables,
        "eda_overview_table",
        caption="EDA overview of the synthetic fleet dataset.",
        label="tab:eda_overview_table",
        index=False,
    )
    _write_json({row["metric"]: row["value"] for _, row in overview.iterrows()}, out_tables / "eda_overview.json")

    wo_by_sub = _work_orders_by_subsystem(work_order)
    _save_table(
        wo_by_sub,
        out_tables,
        "work_orders_by_subsystem",
        caption="Work orders by subsystem (synthetic).",
        label="tab:work_orders_by_subsystem",
        index=False,
    )

    df_surv = _build_vehicle_survival_dataset(vehicle_day, work_order)
    _km_plot_by_duty_cycle(df_surv, out_figures)
    _fit_cox_and_export(df_surv, out_tables)

    print("=== EDA + Stats artifacts generated (LaTeX-ready) ===")
    print("tables: outputs/tables/eda_overview_table.(csv/tex)")
    print("tables: outputs/tables/work_orders_by_subsystem.(csv/tex)")
    print("tables: outputs/tables/cox_hazard_ratios.(csv/tex)")
    print("figure: outputs/figures/km_duty_cycle.pdf and outputs/figures/km_duty_cycle.png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    cfg = _load_config(Path(args.config))
    generate(cfg, repo_root)


if __name__ == "__main__":
    main()