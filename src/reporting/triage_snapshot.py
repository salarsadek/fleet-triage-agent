"""
Export a LaTeX-ready triage snapshot (tables + figures) that matches the Streamlit app.

Why this exists
---------------
- The Streamlit app is interactive, but a portfolio project also needs a clean, reproducible report.
- This script produces deterministic outputs in outputs/tables and outputs/figures.
- Later, a LaTeX report can simply \\input{} the generated .tex tables and include the PDFs.

What it exports
---------------
- Top-K service queue (risk-only or cost-optimized)
- Abstained vehicles with reasons (guardrails)
- OOD counts + summary JSON
- Similar-case evidence tables for the top N queue vehicles
- Risk histogram, cost-vs-risk scatter, mean risk by route table

Run
---
python -m src.reporting.triage_snapshot --config .\config.yaml --horizon 30 --model hgb --k 10 --ranking cost --evidence 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors


# -----------------------------
# Basic utilities (robust, no surprises)
# -----------------------------
def repo_root() -> Path:
    """src/reporting/triage_snapshot.py -> repo root is parents[2]."""
    return Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict:
    """Load YAML config with a clear error message on failure."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config did not parse as a dictionary.")
    return cfg


def ensure_dir(p: Path) -> None:
    """Create a directory if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


def save_table_csv_tex(
    df: pd.DataFrame,
    out_tables: Path,
    stem: str,
    caption: str,
    label: str,
    index: bool = False,
) -> Tuple[Path, Path]:
    """
    Save a table as CSV + a LaTeX table snippet.

    Note:
    - We intentionally avoid pandas 'booktabs=' arguments (compat issues).
    - We wrap the tabular in a full table environment so LaTeX can \\input{} it.
    """
    ensure_dir(out_tables)
    csv_path = out_tables / f"{stem}.csv"
    tex_path = out_tables / f"{stem}.tex"

    df.to_csv(csv_path, index=index)

    tabular = df.to_latex(index=index, escape=True)
    tex = "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            tabular,
            r"\end{table}",
            "",
        ]
    )
    tex_path.write_text(tex, encoding="utf-8")
    return csv_path, tex_path


def save_fig_pdf_png(fig: plt.Figure, out_figs: Path, stem: str) -> Tuple[Path, Path]:
    """Save a matplotlib figure as PDF (LaTeX) + PNG (preview)."""
    ensure_dir(out_figs)
    pdf_path = out_figs / f"{stem}.pdf"
    png_path = out_figs / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


# -----------------------------
# Feature space (must match training/app)
# -----------------------------
def feature_columns(cfg: dict) -> Tuple[List[str], List[str], List[str]]:
    """
    Return (numeric_cols, categorical_cols, all_cols).
    Must be aligned with the training pipeline.
    """
    telem = list(cfg["simulation"]["telematics_features"])

    numeric_cols = [
        "hilliness_index",
        "stop_go_index",
        "ambient_temp_c",
        "precipitation",
        "duty_cycle",
        "miles_today",
        "mileage_total",
        "dtc_count_today",
        "spc_alert_today",
    ] + telem

    categorical_cols = ["route_type"]
    all_cols = numeric_cols + categorical_cols
    return numeric_cols, categorical_cols, all_cols


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """Impute + scale numeric; impute + one-hot categorical."""
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # OneHotEncoder API differs slightly across sklearn versions
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )


# -----------------------------
# OOD + retrieval + business cost logic (same idea as the app)
# -----------------------------
def compute_train_end_date(vehicle_day: pd.DataFrame, train_days: int) -> pd.Timestamp:
    """Compute end date for the training split based on earliest date + train_days."""
    dates = pd.to_datetime(vehicle_day["date_dt"]).dropna().dt.normalize().unique()
    dates = np.array(sorted(dates))
    if len(dates) == 0:
        raise ValueError("No valid dates found in vehicle_day.")
    start = dates[0]
    return pd.Timestamp(start) + pd.Timedelta(days=int(train_days - 1))


def infer_subsystem_map(
    df_latest: pd.DataFrame,
    dtc_event: pd.DataFrame,
    latest_date: pd.Timestamp,
    lookback_days: int = 30,
) -> pd.Series:
    """Infer subsystem per vehicle from most common DTC subsystem in a lookback window."""
    if dtc_event.empty or "timestamp_dt" not in dtc_event.columns or "subsystem" not in dtc_event.columns:
        return pd.Series(["unknown"] * len(df_latest), index=df_latest.index)

    if "vehicle_id" not in dtc_event.columns:
        return pd.Series(["unknown"] * len(df_latest), index=df_latest.index)

    start = latest_date - pd.Timedelta(days=lookback_days)
    df_d = dtc_event.copy()
    df_d = df_d[df_d["timestamp_dt"].notna()].copy()
    df_d = df_d[df_d["timestamp_dt"] >= start].copy()

    if df_d.empty:
        return pd.Series(["unknown"] * len(df_latest), index=df_latest.index)

    grp = df_d.groupby(["vehicle_id", "subsystem"]).size().reset_index(name="n")
    top = grp.sort_values(["vehicle_id", "n"], ascending=[True, False]).groupby("vehicle_id").head(1)
    mapping = dict(zip(top["vehicle_id"].astype(str), top["subsystem"].astype(str)))

    return df_latest["vehicle_id"].astype(str).map(mapping).fillna("unknown")


def work_order_medians(work_order: pd.DataFrame) -> pd.DataFrame:
    """Median downtime + lead time by subsystem."""
    required = {"subsystem", "downtime_days", "parts_lead_time_days"}
    if work_order.empty or not required.issubset(set(work_order.columns)):
        return pd.DataFrame(columns=["subsystem", "downtime_med", "lead_med"])

    df = work_order.copy()
    df = df[df["subsystem"].notna()].copy()

    meds = df.groupby("subsystem", as_index=False).agg(
        downtime_med=("downtime_days", "median"),
        lead_med=("parts_lead_time_days", "median"),
    )
    return meds


def compute_cost_score(df_latest: pd.DataFrame, cfg: dict, wo_meds: pd.DataFrame) -> pd.Series:
    """Cost score = risk * (downtime_cost*downtime + lead_penalty*lead) * severity_multiplier."""
    cm = cfg["queue"]["cost_model"]
    downtime_cost = float(cm.get("downtime_cost_per_day", 1.0))
    lead_penalty = float(cm.get("parts_lead_time_penalty_per_day", 0.2))
    sev_map = dict(cm.get("severity_multiplier_by_subsystem", {}))

    # Robust defaults if there are few/no work orders
    default_downtime = float(np.nanmedian(wo_meds["downtime_med"])) if not wo_meds.empty else 2.0
    default_lead = float(np.nanmedian(wo_meds["lead_med"])) if not wo_meds.empty else 5.0

    if not wo_meds.empty and "subsystem_guess" in df_latest.columns:
        tmp = df_latest[["subsystem_guess"]].merge(
            wo_meds, left_on="subsystem_guess", right_on="subsystem", how="left"
        )
        downtime = tmp["downtime_med"].fillna(default_downtime).astype(float).values
        lead = tmp["lead_med"].fillna(default_lead).astype(float).values
    else:
        downtime = np.full(len(df_latest), default_downtime, dtype=float)
        lead = np.full(len(df_latest), default_lead, dtype=float)

    base_cost = downtime_cost * downtime + lead_penalty * lead
    sev = df_latest["subsystem_guess"].map(lambda s: float(sev_map.get(str(s), 1.0))).astype(float).values
    risk = df_latest["risk_score"].astype(float).values
    return pd.Series(risk * base_cost * sev, index=df_latest.index)


def triage_status(cfg: dict, df_latest: pd.DataFrame, ood_flags: np.ndarray, all_cols: List[str]) -> Tuple[pd.Series, pd.Series]:
    """Guardrails: coverage + confidence + optional OOD abstention."""
    min_cov = float(cfg["agent"]["guardrails"].get("min_required_coverage", 0.95))
    min_conf = float(cfg["agent"]["guardrails"].get("min_confidence", 0.55))
    abstain_if_ood = bool(cfg["agent"]["guardrails"].get("abstain_if_ood", True))

    # Coverage over all input columns used in modeling
    cov = df_latest[all_cols].notna().mean(axis=1).astype(float)

    # Confidence proxy: too close to 0.5 -> low confidence
    margin = max(min_conf - 0.5, 0.0)
    low_conf = (df_latest["risk_score"].astype(float) - 0.5).abs() < margin

    reasons = []
    for i in range(len(df_latest)):
        r: List[str] = []
        if float(cov.iloc[i]) < min_cov:
            r.append("low_coverage")
        if bool(low_conf.iloc[i]):
            r.append("low_confidence")
        if abstain_if_ood and bool(ood_flags[i]):
            r.append("ood")
        reasons.append(", ".join(r) if r else "")

    reasons_s = pd.Series(reasons, index=df_latest.index)
    status_s = pd.Series(np.where(reasons_s == "", "OK", "ABSTAIN"), index=df_latest.index)
    return status_s, reasons_s


def risk_band(p: float) -> str:
    if p >= 0.70:
        return "Critical"
    if p >= 0.55:
        return "High"
    if p >= 0.35:
        return "Medium"
    return "Low"


def recommended_action(p: float, reasons: str) -> str:
    """Human-readable action, respecting abstention reasons."""
    if reasons:
        return f"Abstain ({reasons}): collect more evidence / monitor"
    if p >= 0.70:
        return "Schedule inspection ASAP"
    if p >= 0.55:
        return "Schedule inspection next available slot"
    if p >= 0.35:
        return "Monitor (increase watchlist frequency)"
    return "No action (routine monitoring)"


def retrieve_similar_cases(
    nn: NearestNeighbors,
    preprocessor: ColumnTransformer,
    meta: pd.DataFrame,
    query_row: pd.DataFrame,
    query_vehicle_id: str,
    query_date: pd.Timestamp,
    n_cases: int,
) -> pd.DataFrame:
    """Nearest-neighbor retrieval with a strict past-only filter."""
    Xq = preprocessor.transform(query_row)
    dists, idxs = nn.kneighbors(Xq, n_neighbors=50, return_distance=True)
    dists = dists.flatten()
    idxs = idxs.flatten()

    cand = meta.iloc[idxs].copy()
    cand["distance"] = dists
    cand["similarity"] = 1.0 / (1.0 + cand["distance"].astype(float))

    # Only past evidence to avoid leakage
    if "date_dt" in cand.columns:
        cand = cand[cand["date_dt"] < query_date].copy()

    if "vehicle_id" in cand.columns and "date_dt" in cand.columns:
        cand = cand[~((cand["vehicle_id"].astype(str) == query_vehicle_id) & (cand["date_dt"] == query_date))].copy()

    cand = cand.sort_values("similarity", ascending=False).head(n_cases).copy()

    keep = [c for c in ["vehicle_id", "date", "route_type", "duty_cycle", "dtc_count_today", "spc_alert_today",
                        "breakdown_7d", "breakdown_30d", "similarity", "distance"] if c in cand.columns]
    return cand[keep]


# -----------------------------
# Main entrypoint
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--horizon", type=int, default=30, choices=[7, 30])
    p.add_argument("--model", type=str, default="hgb", choices=["logreg", "hgb"])
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--ranking", type=str, default="cost", choices=["cost", "risk"])
    p.add_argument("--evidence", type=int, default=5, help="How many similar-case rows to export per top vehicle.")
    p.add_argument("--evidence_topn", type=int, default=5, help="How many queue vehicles to export evidence for.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path

    cfg = load_yaml(cfg_path)

    # Resolve paths from config
    raw_dir = root / cfg["paths"]["data_raw_dir"]
    out_tables = root / cfg["paths"]["tables_dir"]
    out_figs = root / cfg["paths"]["figures_dir"]
    out_reports = root / cfg["paths"]["reports_dir"]
    models_dir = root / cfg["paths"]["outputs_dir"] / "models"

    ensure_dir(out_tables)
    ensure_dir(out_figs)
    ensure_dir(out_reports)

    # Load input tables
    vehicle_path = raw_dir / "vehicle_day.csv"
    dtc_path = raw_dir / "dtc_event.csv"
    wo_path = raw_dir / "work_order.csv"

    if not vehicle_path.exists():
        raise FileNotFoundError(f"Missing {vehicle_path}. Run: .\\scripts\\run.ps1 data")

    vehicle_day = pd.read_csv(vehicle_path)
    dtc_event = pd.read_csv(dtc_path) if dtc_path.exists() else pd.DataFrame()
    work_order = pd.read_csv(wo_path) if wo_path.exists() else pd.DataFrame()

    vehicle_day["date_dt"] = pd.to_datetime(vehicle_day.get("date"), errors="coerce")
    vehicle_day = vehicle_day[vehicle_day["date_dt"].notna()].copy()
    if vehicle_day.empty:
        raise ValueError("vehicle_day has no valid dates.")

    if not dtc_event.empty and "timestamp" in dtc_event.columns:
        dtc_event["timestamp_dt"] = pd.to_datetime(dtc_event["timestamp"], errors="coerce")
    else:
        dtc_event["timestamp_dt"] = pd.NaT

    # Load selected model
    model_file = models_dir / f"risk_{args.horizon}d_{args.model}.joblib"
    if not model_file.exists():
        raise FileNotFoundError(f"Missing model file: {model_file}. Run: .\\scripts\\run.ps1 train")

    model = joblib.load(model_file)

    # Build snapshot (latest day)
    latest_date = vehicle_day["date_dt"].max().normalize()
    df_latest = vehicle_day[vehicle_day["date_dt"].dt.normalize() == latest_date].copy()

    # Feature columns
    numeric_cols, categorical_cols, all_cols = feature_columns(cfg)
    missing = [c for c in all_cols if c not in df_latest.columns]
    if missing:
        raise ValueError(f"Latest snapshot missing required columns: {missing}")

    # Predict risk
    df_latest["risk_score"] = model.predict_proba(df_latest[all_cols])[:, 1].astype(float)
    df_latest["risk_pct"] = (100.0 * df_latest["risk_score"]).round(1)
    df_latest["risk_band"] = df_latest["risk_score"].apply(lambda p: risk_band(float(p)))

    # Fit preprocessor + OOD + retrieval index (time-split training slice)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    train_days = int(cfg["modeling"]["time_split_days"]["train"])
    train_end = compute_train_end_date(vehicle_day, train_days=train_days)
    df_train = vehicle_day[vehicle_day["date_dt"] <= train_end].copy()
    if df_train.empty:
        df_train = vehicle_day.copy()

    preprocessor.fit(df_train[all_cols])

    X_train = preprocessor.transform(df_train[all_cols])
    X_all = preprocessor.transform(vehicle_day[all_cols])

    iso = IsolationForest(
        n_estimators=200,
        contamination=float(cfg["agent"]["ood"].get("contamination", 0.02)),
        random_state=int(cfg["project"]["random_seed"]),
    )
    iso.fit(X_train)

    nn = NearestNeighbors(n_neighbors=50, metric="euclidean")
    nn.fit(X_all)

    # Meta for retrieval (“citations”)
    meta_cols = [c for c in ["vehicle_id", "date", "date_dt", "route_type", "duty_cycle",
                             "dtc_count_today", "spc_alert_today", "breakdown_7d", "breakdown_30d"] if c in vehicle_day.columns]
    meta = vehicle_day[meta_cols].copy()

    # OOD flags for snapshot
    ood_pred = iso.predict(preprocessor.transform(df_latest[all_cols]))
    df_latest["ood_flag"] = (ood_pred == -1)

    # Business-cost ranking
    df_latest["subsystem_guess"] = infer_subsystem_map(df_latest, dtc_event, latest_date, lookback_days=30)
    wo_meds = work_order_medians(work_order)
    df_latest["cost_score"] = compute_cost_score(df_latest, cfg, wo_meds).astype(float)

    # Guardrails
    status_s, reasons_s = triage_status(cfg, df_latest, df_latest["ood_flag"].values, all_cols)
    df_latest["triage_status"] = status_s
    df_latest["abstain_reasons"] = reasons_s
    df_latest["recommended_action"] = [
        recommended_action(float(p), str(r))
        for p, r in zip(df_latest["risk_score"].values, df_latest["abstain_reasons"].values)
    ]

    # Build queue
    sort_col = "cost_score" if args.ranking == "cost" else "risk_score"
    df_ok = df_latest[df_latest["triage_status"] == "OK"].copy()
    queue = df_ok.sort_values(sort_col, ascending=False).head(int(args.k)).copy()

    # Save tables
    stamp = f"{latest_date.date()}_{args.horizon}d_{args.model}_{args.ranking}"

    show_cols = [
        "vehicle_id",
        "route_type",
        "duty_cycle",
        "dtc_count_today",
        "spc_alert_today",
        "risk_pct",
        "risk_band",
        "ood_flag",
        "subsystem_guess",
        "cost_score",
        "recommended_action",
    ]
    show_cols = [c for c in show_cols if c in queue.columns]

    save_table_csv_tex(
        queue[show_cols],
        out_tables,
        stem=f"service_queue_{stamp}",
        caption=f"Service queue (Top-{args.k}) on {latest_date.date()} ({args.ranking} ranking).",
        label=f"tab:service_queue_{args.horizon}d_{args.model}_{args.ranking}",
        index=False,
    )

    df_ab = df_latest[df_latest["triage_status"] == "ABSTAIN"].copy()
    ab_cols = [c for c in ["vehicle_id", "route_type", "risk_pct", "risk_band", "ood_flag", "abstain_reasons"] if c in df_ab.columns]
    save_table_csv_tex(
        df_ab.sort_values("risk_score", ascending=False).head(50)[ab_cols],
        out_tables,
        stem=f"abstained_vehicles_{stamp}",
        caption=f"Abstained vehicles (Top 50 by risk) on {latest_date.date()} and reasons.",
        label=f"tab:abstained_{args.horizon}d_{args.model}_{args.ranking}",
        index=False,
    )

    # Mean risk by route table
    route_means = df_latest.groupby("route_type", as_index=False)["risk_score"].mean().sort_values("risk_score", ascending=False)
    route_means = route_means.rename(columns={"risk_score": "mean_risk"})
    save_table_csv_tex(
        route_means,
        out_tables,
        stem=f"mean_risk_by_route_{stamp}",
        caption=f"Mean predicted risk by route type on {latest_date.date()}.",
        label=f"tab:mean_risk_by_route_{args.horizon}d_{args.model}_{args.ranking}",
        index=False,
    )

    # Save summary JSON (handy for later report automation)
    summary = {
        "date": str(latest_date.date()),
        "horizon_days": int(args.horizon),
        "model": str(args.model),
        "ranking": str(args.ranking),
        "n_vehicles": int(df_latest["vehicle_id"].nunique()),
        "n_actionable_ok": int((df_latest["triage_status"] == "OK").sum()),
        "n_abstain": int((df_latest["triage_status"] == "ABSTAIN").sum()),
        "n_ood": int(df_latest["ood_flag"].sum()),
        "k_queue": int(args.k),
    }
    (out_tables / f"triage_summary_{stamp}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Figures
    fig = plt.figure()
    ax = plt.gca()
    ax.hist(df_latest["risk_score"].values, bins=20)
    ax.set_title(f"Fleet risk distribution ({latest_date.date()}, {args.horizon}d, {args.model})")
    ax.set_xlabel("Predicted breakdown risk")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    save_fig_pdf_png(fig, out_figs, stem=f"fleet_risk_hist_{stamp}")

    fig = plt.figure()
    ax = plt.gca()
    x = df_latest["risk_score"].astype(float).values
    y = df_latest["cost_score"].astype(float).values
    ax.scatter(x, y)
    ax.set_title(f"Cost score vs risk ({latest_date.date()}, {args.horizon}d, {args.model})")
    ax.set_xlabel("Risk score")
    ax.set_ylabel("Cost score")
    ax.grid(True, alpha=0.3)
    save_fig_pdf_png(fig, out_figs, stem=f"fleet_cost_vs_risk_{stamp}")

    # Evidence tables for top N queue vehicles
    topn = min(int(args.evidence_topn), len(queue))
    for i in range(topn):
        vid = str(queue.iloc[i]["vehicle_id"])
        qrow = df_latest[df_latest["vehicle_id"].astype(str) == vid].head(1)
        if qrow.empty:
            continue

        sim = retrieve_similar_cases(
            nn=nn,
            preprocessor=preprocessor,
            meta=meta,
            query_row=qrow[all_cols],
            query_vehicle_id=vid,
            query_date=latest_date,
            n_cases=int(args.evidence),
        )

        if sim.empty:
            continue

        # Make filename safe-ish (vehicle ids like V0001 are fine)
        save_table_csv_tex(
            sim,
            out_tables,
            stem=f"similar_cases_{stamp}_{vid}",
            caption=f"Similar historical cases for vehicle {vid} (as of {latest_date.date()}).",
            label=f"tab:similar_cases_{args.horizon}d_{args.model}_{args.ranking}_{vid}",
            index=False,
        )

    # Create a small LaTeX snippet for easy inclusion later
    snippet = "\n".join(
        [
            r"\section{Fleet Triage Snapshot}",
            r"\noindent This section is auto-generated from the triage exporter and matches the Streamlit view.",
            "",
            rf"\input{{../outputs/tables/service_queue_{stamp}.tex}}",
            rf"\input{{../outputs/tables/abstained_vehicles_{stamp}.tex}}",
            rf"\input{{../outputs/tables/mean_risk_by_route_{stamp}.tex}}",
            "",
            r"\begin{figure}[htbp]",
            r"\centering",
            rf"\includegraphics[width=0.85\linewidth]{{../outputs/figures/fleet_risk_hist_{stamp}.pdf}}",
            r"\caption{Fleet risk score distribution (snapshot).}",
            rf"\label{{fig:fleet_risk_hist_{stamp}}}",
            r"\end{figure}",
            "",
            r"\begin{figure}[htbp]",
            r"\centering",
            rf"\includegraphics[width=0.85\linewidth]{{../outputs/figures/fleet_cost_vs_risk_{stamp}.pdf}}",
            r"\caption{Cost score vs risk score (snapshot).}",
            rf"\label{{fig:fleet_cost_vs_risk_{stamp}}}",
            r"\end{figure}",
            "",
        ]
    )
    (out_reports / f"triage_snapshot_{stamp}.tex").write_text(snippet, encoding="utf-8")

    print("=== Triage snapshot exported (LaTeX-ready) ===")
    print(f"tables:  outputs/tables/service_queue_{stamp}.(csv/tex)")
    print(f"tables:  outputs/tables/abstained_vehicles_{stamp}.(csv/tex)")
    print(f"tables:  outputs/tables/mean_risk_by_route_{stamp}.(csv/tex)")
    print(f"tables:  outputs/tables/triage_summary_{stamp}.json")
    print(f"figures: outputs/figures/fleet_risk_hist_{stamp}.pdf/png")
    print(f"figures: outputs/figures/fleet_cost_vs_risk_{stamp}.pdf/png")
    print(f"report:  outputs/reports/triage_snapshot_{stamp}.tex")


if __name__ == "__main__":
    main()