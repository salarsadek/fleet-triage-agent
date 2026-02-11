"""
Streamlit GUI for the Fleet Maintenance Triage Agent (Synthetic Demo).

Upgrades included
----------------
1) OOD flag (IsolationForest)
   - Flags vehicles whose feature patterns look out-of-distribution (OOD)
   - Supports guardrail abstention: "abstain_if_ood"

2) Retrieval of similar historical cases (NearestNeighbors)
   - For a selected vehicle/day, retrieve similar historical vehicle-days
   - Display as "evidence/citations" with outcomes (breakdown_7d / breakdown_30d)

3) Cost-optimized queue
   - Rank service queue by: cost_score = risk_score * expected_cost * severity_multiplier
   - expected_cost estimated from historical work orders (median downtime + lead time)
   - subsystem guessed from recent DTC subsystem (fallback: unknown)

Robustness goals
----------------
- Avoid crashes: validate files/columns, safe fallbacks, clear user messages.
- Caching is invalidated noting file signatures (size + mtime) to avoid stale data.
- No custom classes used as cached keys (Streamlit rerun-safe).

Run
---
.\scripts\run.ps1 app
(or) .\.venv\Scripts\streamlit.exe run app\streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import yaml
import matplotlib.pyplot as plt

# sklearn pieces used for OOD + retrieval + preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors


# -----------------------------
# Paths + config
# -----------------------------
def project_root() -> Path:
    """Resolve project root reliably: app/streamlit_app.py -> parents[1] is repo root."""
    return Path(__file__).resolve().parents[1]


def abs_path(rel_path: str) -> Path:
    """Convert config relative paths into absolute paths under project root."""
    return project_root() / rel_path


def safe_file_sig(p: Path) -> Tuple[str, int, float]:
    """
    File signature used for cache invalidation.
    If file is missing, return a signature that will trigger errors downstream (but not crash here).
    """
    try:
        s = p.stat()
        return (str(p), int(s.st_size), float(s.st_mtime))
    except FileNotFoundError:
        return (str(p), -1, -1.0)


@st.cache_data(show_spinner=False)
def load_config(cfg_sig: Tuple[str, int, float]) -> dict:
    """Load config.yaml (cached using file signature to avoid stale reads)."""
    cfg_path = Path(cfg_sig[0])
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml did not parse into a dictionary.")
    return cfg


# -----------------------------
# Data loading (cache keyed by file signatures)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_tables(
    vehicle_sig: Tuple[str, int, float],
    dtc_sig: Tuple[str, int, float],
    wo_sig: Tuple[str, int, float],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three raw tables with date parsing."""
    vehicle_path = Path(vehicle_sig[0])
    dtc_path = Path(dtc_sig[0])
    wo_path = Path(wo_sig[0])

    vehicle_day = pd.read_csv(vehicle_path)
    dtc_event = pd.read_csv(dtc_path)
    work_order = pd.read_csv(wo_path)

    vehicle_day["date_dt"] = pd.to_datetime(vehicle_day.get("date"), errors="coerce")

    if not dtc_event.empty and "timestamp" in dtc_event.columns:
        dtc_event["timestamp_dt"] = pd.to_datetime(dtc_event["timestamp"], errors="coerce")
    else:
        dtc_event["timestamp_dt"] = pd.NaT

    if not work_order.empty and "open_date" in work_order.columns:
        work_order["open_date_dt"] = pd.to_datetime(work_order["open_date"], errors="coerce")
        work_order["close_date_dt"] = pd.to_datetime(work_order.get("close_date"), errors="coerce")
    else:
        work_order["open_date_dt"] = pd.NaT
        work_order["close_date_dt"] = pd.NaT

    return vehicle_day, dtc_event, work_order


# -----------------------------
# Models loading (cache keyed by model file signatures)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_models(model_sigs: Tuple[Tuple[str, int, float], ...]) -> Dict[Tuple[int, str], object]:
    """
    Load trained models. Keys are tuples: (horizon_days, model_name).

    model_sigs is a tuple of file signatures for all expected model files.
    """
    models: Dict[Tuple[int, str], object] = {}

    for (path_str, size, _mtime) in model_sigs:
        p = Path(path_str)
        if size <= 0:
            continue  # missing file
        name = p.name  # risk_7d_logreg.joblib
        try:
            parts = name.replace(".joblib", "").split("_")  # ["risk", "7d", "logreg"]
            horizon = int(parts[1].replace("d", ""))
            model_name = parts[2]
            models[(horizon, model_name)] = joblib.load(p)
        except Exception:
            # Robustness: a single bad model file should not crash the app
            continue

    return models


def predict_proba(model: object, X: pd.DataFrame) -> np.ndarray:
    """Unified probability interface."""
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


# -----------------------------
# Feature space for OOD + retrieval (cached)
# -----------------------------
def feature_columns(cfg: dict) -> List[str]:
    """Must match src/models/train.py (MVP feature set)."""
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
    return numeric_cols + categorical_cols


def build_preprocessor(cfg: dict) -> ColumnTransformer:
    """Preprocessing consistent with training: impute + scale numeric, impute + one-hot categorical."""
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

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def compute_train_end_date(vehicle_day: pd.DataFrame, train_days: int) -> pd.Timestamp:
    """Compute the end date of the training split based on earliest date and train_days."""
    dates = pd.to_datetime(vehicle_day["date_dt"]).dropna().dt.normalize().unique()
    dates = np.array(sorted(dates))
    if len(dates) == 0:
        raise ValueError("vehicle_day has no valid dates.")
    start = dates[0]
    return pd.Timestamp(start) + pd.Timedelta(days=int(train_days - 1))


@st.cache_resource(show_spinner=False)
def build_feature_space(
    vehicle_sig: Tuple[str, int, float],
    cfg_sig: Tuple[str, int, float],
) -> Dict[str, object]:
    """
    Build feature space objects for:
      - OOD detection: IsolationForest on training feature distribution
      - Retrieval: NearestNeighbors index on all historical rows

    Cached by file signatures to update when data/config changes.
    """
    cfg = load_config(cfg_sig)
    vehicle_path = Path(vehicle_sig[0])
    vehicle_day = pd.read_csv(vehicle_path)
    vehicle_day["date_dt"] = pd.to_datetime(vehicle_day.get("date"), errors="coerce")

    # Keep only rows with valid dates (robustness)
    vehicle_day = vehicle_day[vehicle_day["date_dt"].notna()].copy()
    if vehicle_day.empty:
        raise ValueError("vehicle_day.csv contains no rows with valid dates.")

    X_cols = feature_columns(cfg)
    missing_cols = [c for c in X_cols if c not in vehicle_day.columns]
    if missing_cols:
        raise ValueError(f"vehicle_day is missing required feature columns: {missing_cols}")

    # Fit preprocessor on training slice (time-based)
    train_days = int(cfg["modeling"]["time_split_days"]["train"])
    train_end = compute_train_end_date(vehicle_day, train_days=train_days)
    df_train = vehicle_day[vehicle_day["date_dt"] <= train_end].copy()
    if df_train.empty:
        # Fall back: if split fails, fit on all rows (still allows OOD/retrieval)
        df_train = vehicle_day.copy()

    preprocessor = build_preprocessor(cfg)
    preprocessor.fit(df_train[X_cols])

    X_all = preprocessor.transform(vehicle_day[X_cols])

    # OOD model
    contamination = float(cfg["agent"]["ood"].get("contamination", 0.02))
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=int(cfg["project"]["random_seed"]),
    )
    iso.fit(preprocessor.transform(df_train[X_cols]))

    # Retrieval index (neighbors on the same feature space)
    nn = NearestNeighbors(n_neighbors=50, algorithm="auto", metric="euclidean")
    nn.fit(X_all)

    # Metadata for showing â€œcitationsâ€
    meta_cols = [
        "vehicle_id",
        "date",
        "date_dt",
        "route_type",
        "duty_cycle",
        "dtc_count_today",
        "spc_alert_today",
        "breakdown_7d",
        "breakdown_30d",
    ]
    meta = vehicle_day[[c for c in meta_cols if c in vehicle_day.columns]].copy()

    return {
        "cfg": cfg,
        "X_cols": X_cols,
        "preprocessor": preprocessor,
        "X_all": X_all,
        "iso": iso,
        "nn": nn,
        "meta": meta,
    }


def is_ood(feature_space: Dict[str, object], X_df: pd.DataFrame) -> np.ndarray:
    """Compute OOD flags for a batch of rows."""
    pre = feature_space["preprocessor"]
    iso = feature_space["iso"]
    X = pre.transform(X_df)
    pred = iso.predict(X)  # 1=inlier, -1=outlier
    return (pred == -1)


def retrieve_similar_cases(
    feature_space: Dict[str, object],
    query_row: pd.DataFrame,
    query_vid: int,
    query_date: pd.Timestamp,
    n_cases: int = 5,
) -> pd.DataFrame:
    """
    Retrieve similar historical cases as evidence.

    Strategy:
    - Query the NN index with the selected row
    - Filter to past dates only (< query_date)
    - Return top-n with similarity score + outcomes
    """
    pre = feature_space["preprocessor"]
    nn = feature_space["nn"]
    meta = feature_space["meta"]

    Xq = pre.transform(query_row)
    dists, idxs = nn.kneighbors(Xq, n_neighbors=50, return_distance=True)

    dists = dists.flatten()
    idxs = idxs.flatten()

    cand = meta.iloc[idxs].copy()
    cand["distance"] = dists
    cand["similarity"] = 1.0 / (1.0 + cand["distance"].astype(float))

    # Past-only filter (avoid leakage in â€œevidenceâ€)
    if "date_dt" in cand.columns:
        cand = cand[cand["date_dt"] < query_date].copy()

    # Avoid returning the exact same vehicle/day if it slips through
    if "vehicle_id" in cand.columns and "date_dt" in cand.columns:
        cand = cand[~((cand["vehicle_id"] == query_vid) & (cand["date_dt"] == query_date))].copy()

    cand = cand.sort_values("similarity", ascending=False).head(n_cases).copy()

    # Keep a clean display set
    keep = [c for c in ["vehicle_id", "date", "route_type", "duty_cycle", "dtc_count_today", "spc_alert_today",
                        "breakdown_7d", "breakdown_30d", "similarity", "distance"] if c in cand.columns]
    return cand[keep]


# -----------------------------
# Subsystem inference + cost score
# -----------------------------
def infer_subsystem_map(
    df_latest: pd.DataFrame,
    dtc_event: pd.DataFrame,
    latest_date: pd.Timestamp,
    lookback_days: int = 30,
) -> pd.Series:
    """
    Infer subsystem for each vehicle based on recent DTC events subsystem.
    Fallback: 'unknown' if no events.

    This is MVP heuristic (fast + understandable).
    """
    if dtc_event.empty or "timestamp_dt" not in dtc_event.columns or "subsystem" not in dtc_event.columns:
        return pd.Series(["unknown"] * len(df_latest), index=df_latest.index)

    start = (latest_date - pd.Timedelta(days=lookback_days)).to_pydatetime()
    df_d = dtc_event.copy()
    df_d = df_d[df_d["timestamp_dt"].notna()].copy()
    df_d = df_d[df_d["timestamp_dt"] >= pd.Timestamp(start)].copy()

    if df_d.empty or "vehicle_id" not in df_d.columns:
        return pd.Series(["unknown"] * len(df_latest), index=df_latest.index)

    # Most common subsystem per vehicle in lookback window
    grp = df_d.groupby(["vehicle_id", "subsystem"]).size().reset_index(name="n")
    idx = grp.sort_values(["vehicle_id", "n"], ascending=[True, False]).groupby("vehicle_id").head(1)
    mapping = dict(zip(idx["vehicle_id"].astype(str), idx["subsystem"].astype(str)))

    return df_latest["vehicle_id"].astype(str).map(mapping).fillna("unknown")


def work_order_medians(work_order: pd.DataFrame) -> pd.DataFrame:
    """
    Compute median downtime and lead time by subsystem.
    If work_order is empty, returns an empty table (caller must fall back).
    """
    required = {"subsystem", "downtime_days", "parts_lead_time_days"}
    if work_order.empty or not required.issubset(set(work_order.columns)):
        return pd.DataFrame(columns=["subsystem", "downtime_med", "lead_med"])

    df = work_order.copy()
    df = df[df["subsystem"].notna()].copy()

    med = df.groupby("subsystem", as_index=False).agg(
        downtime_med=("downtime_days", "median"),
        lead_med=("parts_lead_time_days", "median"),
    )
    return med


def compute_cost_score(
    df_latest: pd.DataFrame,
    cfg: dict,
    wo_meds: pd.DataFrame,
) -> pd.Series:
    """
    Compute cost score:
      cost_score = risk_score * (downtime_cost_per_day*downtime + parts_penalty_per_day*lead_time) * severity_multiplier
    """
    cm = cfg["queue"]["cost_model"]
    downtime_cost = float(cm.get("downtime_cost_per_day", 1.0))
    lead_penalty = float(cm.get("parts_lead_time_penalty_per_day", 0.2))
    sev_map = dict(cm.get("severity_multiplier_by_subsystem", {}))

    # Default expectations if we have no work order medians
    default_downtime = float(np.nanmedian(wo_meds["downtime_med"])) if not wo_meds.empty else 2.0
    default_lead = float(np.nanmedian(wo_meds["lead_med"])) if not wo_meds.empty else 5.0

    # Join medians by subsystem if available
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

    if "subsystem_guess" in df_latest.columns:
        sev = df_latest["subsystem_guess"].map(lambda s: float(sev_map.get(str(s), 1.0))).astype(float).values
    else:
        sev = np.ones(len(df_latest), dtype=float)

    risk = df_latest["risk_score"].astype(float).values
    return pd.Series(risk * base_cost * sev, index=df_latest.index)


# -----------------------------
# Guardrails (abstention logic)
# -----------------------------
def triage_status(
    cfg: dict,
    df_latest: pd.DataFrame,
    ood_flags: np.ndarray,
) -> Tuple[pd.Series, pd.Series]:
    """
    Determine triage status and abstention reasons.

    Reasons include:
      - low_coverage: too many missing required inputs
      - low_confidence: probability too close to 0.5
      - ood: out-of-distribution (if abstain_if_ood enabled)
    """
    min_cov = float(cfg["agent"]["guardrails"].get("min_required_coverage", 0.95))
    min_conf = float(cfg["agent"]["guardrails"].get("min_confidence", 0.55))
    abstain_if_ood = bool(cfg["agent"]["guardrails"].get("abstain_if_ood", True))

    # Coverage over feature columns (assumes X_cols present)
    X_cols = feature_columns(cfg)
    cov = df_latest[X_cols].notna().mean(axis=1).astype(float)

    # Confidence proxy: distance from 0.5 (simple MVP rule)
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


# -----------------------------
# UI helpers
# -----------------------------
def st_df(df: pd.DataFrame) -> None:
    """Compatibility wrapper for Streamlit dataframe sizing."""
    try:
        st.dataframe(df, width="stretch")
    except TypeError:
        st.dataframe(df, use_container_width=True)


def plot_hist(values: np.ndarray, title: str, xlabel: str) -> None:
    """Small histogram helper."""
    fig = plt.figure()
    ax = plt.gca()
    ax.hist(values, bins=20)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def plot_line(dates: pd.Series, y: np.ndarray, title: str, ylabel: str) -> None:
    """Small line-plot helper."""
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(dates, y)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def risk_band(p: float) -> str:
    """Human-friendly risk bucket."""
    if p >= 0.70:
        return "Critical"
    if p >= 0.55:
        return "High"
    if p >= 0.35:
        return "Medium"
    return "Low"


def recommend_action(p: float, reasons: str) -> str:
    """Recommended action depends on abstention reasons (if any) + risk thresholds."""
    if reasons:
        # If we abstain, be explicit about why.
        return f"Abstain ({reasons}): collect more evidence / monitor"

    if p >= 0.70:
        return "Schedule inspection ASAP"
    if p >= 0.55:
        return "Schedule inspection next available slot"
    if p >= 0.35:
        return "Monitor (increase watchlist frequency)"
    return "No action (routine monitoring)"


def safe_int(x: object, default: int = 0) -> int:
    """Robust int conversion."""
    try:
        return int(x)
    except Exception:
        return default


def main() -> None:
    st.set_page_config(page_title="Fleet Triage Agent", layout="wide")

    # File signatures for caching (prevents stale caches after regeneration)
    cfg_path = project_root() / "config.yaml"
    cfg_sig = safe_file_sig(cfg_path)

    cfg = load_config(cfg_sig)

    raw_dir = abs_path(cfg["paths"]["data_raw_dir"])
    vehicle_sig = safe_file_sig(raw_dir / "vehicle_day.csv")
    dtc_sig = safe_file_sig(raw_dir / "dtc_event.csv")
    wo_sig = safe_file_sig(raw_dir / "work_order.csv")

    # Load data
    try:
        vehicle_day, dtc_event, work_order = load_tables(vehicle_sig, dtc_sig, wo_sig)
    except FileNotFoundError as e:
        st.error(f"Missing input file: {e}. Did you run: .\\scripts\\run.ps1 data ?")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load input data: {e}")
        st.stop()

    # Basic checks
    if "date_dt" not in vehicle_day.columns or vehicle_day["date_dt"].isna().any():
        st.error("vehicle_day.csv has invalid/missing dates in 'date'. Re-run data generation/validation.")
        st.stop()

    # Load models
    models_dir = abs_path(cfg["paths"]["outputs_dir"]) / "models"
    expected = []
    for h in [7, 30]:
        for name in ["logreg", "hgb"]:
            expected.append(safe_file_sig(models_dir / f"risk_{h}d_{name}.joblib"))
    model_sigs = tuple(expected)

    models = load_models(model_sigs)
    if len(models) == 0:
        st.error("No trained models found in outputs/models. Run: .\\scripts\\run.ps1 train")
        st.stop()

    # Build OOD + retrieval feature space
    try:
        feature_space = build_feature_space(vehicle_sig, cfg_sig)
    except Exception as e:
        st.error(f"Failed to build OOD/retrieval feature space: {e}")
        st.stop()

    X_cols = feature_space["X_cols"]

    st.title("Fleet Maintenance Triage Agent")
    st.caption("Synthetic portfolio demo: Top-K service queue + OOD flag + similar-case evidence + cost ranking.")

    # Sidebar controls
    st.sidebar.header("Controls")

    available_keys = sorted(models.keys())
    horizons = sorted({k[0] for k in available_keys})
    horizon = st.sidebar.selectbox(
        "Risk horizon (days)",
        horizons,
        index=0,
        help="Prediction target: probability of a breakdown within the next N days.",
    )

    model_names = sorted({k[1] for k in available_keys if k[0] == horizon})
    model_name = st.sidebar.selectbox(
        "Model",
        model_names,
        index=0,
        help="logreg = interpretable baseline; hgb = stronger nonlinear model.",
    )

    model = models[(horizon, model_name)]

    k_default = int(cfg["modeling"]["evaluation"]["k_service_per_week"])
    k_queue = st.sidebar.slider(
        "Queue size (Top-K)",
        min_value=1,
        max_value=50,
        value=min(k_default, 50),
        help="How many vehicles you plan to schedule in the next weekly service batch.",
    )

    ranking = st.sidebar.radio(
        "Queue ranking",
        ["Cost-optimized (risk Ã— cost)", "Risk-only"],
        help="Cost-optimized uses downtime + parts lead time + subsystem severity weights.",
    )

    page = st.sidebar.radio("Page", ["Fleet overview", "Vehicle detail"])

    with st.sidebar.expander("How to read this", expanded=True):
        st.markdown(
            """
- **Risk score** is the predicted probability of a breakdown within the selected horizon.
- **OOD** flags unusual patterns compared to training distribution (may require extra caution).
- **Abstain** appears when the agent lacks confidence / coverage / sees OOD (guardrails).
- **Similar cases** are retrieved historical vehicle-days that look most like the selected one.
- **Cost-optimized queue** ranks by risk Ã— expected operational cost (downtime + parts lead time).
            """.strip()
        )

    # Latest date snapshot
    latest_date = vehicle_day["date_dt"].max()
    df_latest = vehicle_day[vehicle_day["date_dt"] == latest_date].copy()

    missing_cols = [c for c in X_cols if c not in df_latest.columns]
    if missing_cols:
        st.error(f"Latest snapshot missing required feature columns: {missing_cols}")
        st.stop()

    # Risk predictions (latest day snapshot)
    try:
        df_latest["risk_score"] = predict_proba(model, df_latest[X_cols])
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.stop()

    df_latest["risk_pct"] = (100.0 * df_latest["risk_score"]).round(1)
    df_latest["risk_band"] = df_latest["risk_score"].apply(lambda p: risk_band(float(p)))

    # OOD flags
    try:
        ood_flags = is_ood(feature_space, df_latest[X_cols])
    except Exception:
        # If OOD fails, do not crash the app; mark as unknown/False
        ood_flags = np.zeros(len(df_latest), dtype=bool)

    df_latest["ood_flag"] = ood_flags

    # Subsystem inference + work-order medians (for cost)
    df_latest["subsystem_guess"] = infer_subsystem_map(df_latest, dtc_event, latest_date, lookback_days=30)
    wo_meds = work_order_medians(work_order)
    df_latest["cost_score"] = compute_cost_score(df_latest, cfg, wo_meds)
    df_latest["cost_score"] = df_latest["cost_score"].astype(float)

    # Guardrails: status + reasons
    status_s, reasons_s = triage_status(cfg, df_latest, ood_flags)
    df_latest["triage_status"] = status_s
    df_latest["abstain_reasons"] = reasons_s
    df_latest["recommended_action"] = [
        recommend_action(float(p), str(r))
        for p, r in zip(df_latest["risk_score"].values, df_latest["abstain_reasons"].values)
    ]

    # -----------------------------
    # Fleet overview
    # -----------------------------
    if page == "Fleet overview":
        st.subheader("Fleet overview (latest day)")

        colA, colB, colC, colD, colE = st.columns(5)
        colA.metric("Latest date", str(pd.to_datetime(latest_date).date()))
        colB.metric("Vehicles", int(df_latest["vehicle_id"].nunique()))
        colC.metric("Actionable (OK)", int((df_latest["triage_status"] == "OK").sum()))
        colD.metric("Abstain", int((df_latest["triage_status"] == "ABSTAIN").sum()))
        colE.metric("OOD flagged", int(df_latest["ood_flag"].sum()))

        # Filters
        st.markdown("### Filters")
        route_options = sorted(df_latest["route_type"].dropna().unique().tolist())
        selected_routes = st.multiselect("Route type", route_options, default=route_options)

        band_options = ["Low", "Medium", "High", "Critical"]
        selected_bands = st.multiselect("Risk band", band_options, default=band_options)

        df_f = df_latest.copy()
        if selected_routes:
            df_f = df_f[df_f["route_type"].isin(selected_routes)].copy()
        if selected_bands:
            df_f = df_f[df_f["risk_band"].isin(selected_bands)].copy()

        # Choose ranking
        sort_col = "cost_score" if ranking.startswith("Cost-optimized") else "risk_score"
        df_ok = df_f[df_f["triage_status"] == "OK"].copy()

        queue = df_ok.sort_values(sort_col, ascending=False).head(k_queue).copy()

        st.markdown("### Suggested service queue (Top-K actionable vehicles)")
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
        st_df(queue[show_cols])

        # Export
        csv_bytes = queue[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download queue (CSV)",
            data=csv_bytes,
            file_name=f"service_queue_{pd.to_datetime(latest_date).date()}_{horizon}d_{model_name}.csv",
            mime="text/csv",
        )

        # Abstained list (show reasons)
        with st.expander("Vehicles that were abstained (needs more evidence)", expanded=False):
            df_ab = df_f[df_f["triage_status"] == "ABSTAIN"].copy()
            if df_ab.empty:
                st.write("None (all vehicles actionable under current guardrails).")
            else:
                ab_cols = [
                    "vehicle_id",
                    "route_type",
                    "risk_pct",
                    "risk_band",
                    "ood_flag",
                    "abstain_reasons",
                ]
                ab_cols = [c for c in ab_cols if c in df_ab.columns]
                st_df(df_ab.sort_values("risk_score", ascending=False)[ab_cols].head(50))

        # Charts
        st.markdown("### Risk distribution")
        plot_hist(df_f["risk_score"].values, "Fleet risk score distribution", "Predicted breakdown risk")

        st.markdown("### Mean risk by route type")
        route_means = df_f.groupby("route_type", as_index=False)["risk_score"].mean().sort_values("risk_score", ascending=False)
        route_means = route_means.rename(columns={"risk_score": "mean_risk"})
        st_df(route_means)

        st.info("Tip: open **Vehicle detail** for similar-case evidence + signal trends + maintenance history.")

    # -----------------------------
    # Vehicle detail
    # -----------------------------
    else:
        st.subheader("Vehicle detail")

        vehicle_ids = sorted(vehicle_day["vehicle_id"].unique().tolist())
        vid = st.selectbox("Select vehicle", vehicle_ids, index=0)

        df_v = vehicle_day[vehicle_day["vehicle_id"] == vid].sort_values("date_dt").copy()
        if df_v.empty:
            st.warning("No data found for this vehicle.")
            st.stop()

        # Predict risk over time
        try:
            df_v["risk_score"] = predict_proba(model, df_v[X_cols])
        except Exception as e:
            st.error(f"Prediction failed for this vehicle: {e}")
            st.stop()

        df_v["risk_pct"] = (100.0 * df_v["risk_score"]).round(1)
        df_v["risk_band"] = df_v["risk_score"].apply(lambda p: risk_band(float(p)))

        latest_row = df_v.iloc[-1]
        latest_row_df = df_v.tail(1)[X_cols].copy()

        # OOD for latest row
        try:
            ood_latest = bool(is_ood(feature_space, latest_row_df)[0])
        except Exception:
            ood_latest = False

        # Guardrails reasons for latest row (reuse triage logic on a single-row frame)
        df_one = df_latest[df_latest["vehicle_id"] == vid].copy()
        if df_one.empty:
            # If vehicle is not present in latest day snapshot, compute reasons manually (fallback)
            df_one = df_v.tail(1).copy()
            df_one["risk_score"] = float(latest_row["risk_score"])
            df_one["ood_flag"] = ood_latest
            df_one["triage_status"] = "OK"
            df_one["abstain_reasons"] = ""
        else:
            # df_one already contains status/reasons from latest snapshot computation above
            pass

        reasons = str(df_one["abstain_reasons"].iloc[0]) if "abstain_reasons" in df_one.columns else ""
        action = recommend_action(float(latest_row["risk_score"]), reasons)

        # Summary cards
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Latest date", str(pd.to_datetime(latest_row["date_dt"]).date()))
        c2.metric(f"Risk ({horizon}d)", f"{float(latest_row['risk_score']):.3f} ({float(latest_row['risk_pct']):.1f}%)")
        c3.metric("Risk band", str(latest_row["risk_band"]))
        c4.metric("OOD", "Yes" if ood_latest else "No")
        c5.metric("Recommended action", action)

        tabs = st.tabs(["Overview", "Similar cases (evidence)", "Telematics", "DTC events", "Work orders"])

        with tabs[0]:
            st.markdown("### Explanation")
            st.markdown(
                """
- **Risk trend**: how predicted breakdown probability evolves over time.
- **OOD**: the latest day looks unusual compared to the training distribution.
- **Abstain**: the agent refuses to recommend action when guardrails trigger (coverage/confidence/OOD).
                """.strip()
            )
            if reasons:
                st.warning(f"Guardrails triggered â†’ abstain reasons: **{reasons}**")

            st.markdown("### Risk trend")
            plot_line(df_v["date_dt"], df_v["risk_score"].values, f"Risk trend (vehicle {vid})", "Predicted breakdown risk")

            st.markdown("### Recent context (last 14 days)")
            recent_cols = [
                "date",
                "route_type",
                "duty_cycle",
                "ambient_temp_c",
                "dtc_count_today",
                "spc_alert_today",
                "risk_pct",
                "risk_band",
            ]
            recent_cols = [c for c in recent_cols if c in df_v.columns]
            st_df(df_v.tail(14)[recent_cols])

        with tabs[1]:
            st.markdown("### Retrieved similar historical cases (citations)")
            st.caption("These are similar vehicle-days from the past (before the selected date), shown as supporting evidence.")

            try:
                sim = retrieve_similar_cases(
                    feature_space=feature_space,
                    query_row=df_v.tail(1)[X_cols],
                    query_vid=str(vid),
                    query_date=pd.to_datetime(latest_row["date_dt"]).normalize(),
                    n_cases=5,
                )
                if sim.empty:
                    st.write("No similar cases found (after applying past-only filtering).")
                else:
                    st_df(sim)

                    # Simple summary: how many similar cases broke down
                    bd7 = sim["breakdown_7d"].mean() if "breakdown_7d" in sim.columns else float("nan")
                    bd30 = sim["breakdown_30d"].mean() if "breakdown_30d" in sim.columns else float("nan")
                    st.info(f"In the retrieved cases: breakdown_7d rate â‰ˆ {bd7:.2f} ; breakdown_30d rate â‰ˆ {bd30:.2f}")
            except Exception as e:
                st.warning(f"Similar-case retrieval failed (non-fatal): {e}")

        with tabs[2]:
            st.markdown("### Telematics signals")
            telem_cols = list(cfg["simulation"]["telematics_features"])
            existing_telem = [c for c in telem_cols if c in df_v.columns]

            default_signals = [c for c in ["coolant_temp_max_c", "egt_max_c", "battery_v_min"] if c in existing_telem]
            selected = st.multiselect(
                "Signals to plot",
                existing_telem,
                default=default_signals if default_signals else existing_telem[:3],
            )

            if not selected:
                st.warning("Select at least one signal to plot.")
            else:
                for col in selected:
                    plot_line(df_v["date_dt"], df_v[col].values, f"{col} over time (vehicle {vid})", col)

        with tabs[3]:
            st.markdown("### DTC events (latest 50)")
            if dtc_event.empty:
                st.write("No DTC events table.")
            else:
                df_d = dtc_event[dtc_event["vehicle_id"] == vid].copy() if "vehicle_id" in dtc_event.columns else pd.DataFrame()
                if df_d.empty:
                    st.write("No DTC events for this vehicle.")
                else:
                    df_d = df_d.sort_values("timestamp_dt")
                    cols = ["timestamp", "dtc_code", "subsystem", "severity", "count"]
                    cols = [c for c in cols if c in df_d.columns]
                    st_df(df_d.tail(50)[cols])

        with tabs[4]:
            st.markdown("### Work orders / maintenance history")
            if work_order.empty:
                st.write("No work orders table.")
            else:
                df_w = work_order[work_order["vehicle_id"] == vid].copy() if "vehicle_id" in work_order.columns else pd.DataFrame()
                if df_w.empty:
                    st.write("No work orders for this vehicle.")
                else:
                    cols = ["open_date", "close_date", "subsystem", "action", "parts_lead_time_days", "downtime_days", "notes"]
                    cols = [c for c in cols if c in df_w.columns]
                    st_df(df_w[cols].sort_values("open_date"))

    st.caption("Synthetic demo: recommendations are for portfolio illustration; validate before real deployment.")


if __name__ == "__main__":
    main()