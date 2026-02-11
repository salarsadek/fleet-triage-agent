"""
Train risk models (7d and 30d) and export LaTeX-ready artifacts.

Why this script exists
----------------------
We want a portfolio project where the final report is auto-generated from outputs/.
So this script saves:
  - tables as CSV + TEX (\\input{...} in LaTeX)
  - figures as PDF + PNG (LaTeX uses PDF, PNG is quick preview)
  - fitted models as .joblib (used later by the agent + Streamlit app)

Models (MVP)
------------
- Baseline: Logistic Regression (interpretable)
- Advanced: HistGradientBoostingClassifier (captures nonlinearities)

Evaluation (MVP)
----------------
- AUPRC (Average Precision): good for rare events
- Brier score: probability quality / calibration
- Precision@K on Mondays: mimics weekly service-queue decision points
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve

from src.utils.artifacts import ensure_dir, save_table, save_json


# -----------------------------
# Config loading
# -----------------------------
def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config (UTF-8) into a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml did not parse into a dictionary.")
    return cfg


# -----------------------------
# IO helpers for figures
# -----------------------------
def save_fig(fig, out_dir: Path, stem: str) -> Tuple[Path, Path]:
    """
    Save figure as:
      - PDF (LaTeX-friendly)
      - PNG (quick preview)
    """
    ensure_dir(out_dir)
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    return pdf_path, png_path


# -----------------------------
# Data loading + time split
# -----------------------------
def load_vehicle_day(raw_dir: Path) -> pd.DataFrame:
    """Load daily vehicle table and parse date column."""
    df = pd.read_csv(raw_dir / "vehicle_day.csv")
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date_dt"].isna().any():
        raise ValueError("Found unparseable dates in vehicle_day.date")
    return df


def time_split_by_days(df: pd.DataFrame, train_days: int, val_days: int, test_days: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Global date-based split to avoid leakage (future -> past).
    This is a clean and standard baseline for time series classification MVPs.
    """
    unique_dates = np.array(sorted(df["date_dt"].dt.normalize().unique()))
    need = train_days + val_days + test_days
    if len(unique_dates) < need:
        raise ValueError(f"Not enough unique days for split. Need {need}, have {len(unique_dates)}.")

    start = unique_dates[0]
    train_end = start + np.timedelta64(train_days - 1, "D")
    val_end = train_end + np.timedelta64(val_days, "D")
    test_end = val_end + np.timedelta64(test_days, "D")

    train = df[df["date_dt"] <= train_end].copy()
    val = df[(df["date_dt"] > train_end) & (df["date_dt"] <= val_end)].copy()
    test = df[(df["date_dt"] > val_end) & (df["date_dt"] <= test_end)].copy()
    return train, val, test


# -----------------------------
# Feature pipeline
# -----------------------------
def build_preprocessor(cfg: Dict[str, Any]) -> Tuple[ColumnTransformer, List[str]]:
    """
    Build preprocessing:
      - numeric: median imputation + scaling
      - categorical: mode imputation + one-hot encoding
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
    feature_cols = numeric_cols + categorical_cols

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Keep this robust across scikit-learn versions.
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor, feature_cols


# -----------------------------
# Business-like metric: Precision@K on Mondays
# -----------------------------
def precision_at_k_mondays(df_test: pd.DataFrame, y_col: str, proba: np.ndarray, k: int) -> Tuple[float, pd.DataFrame]:
    """
    Evaluate how good the model is at selecting the top-K vehicles *on Mondays*.

    Rationale: fleets often plan service queues weekly (e.g., Monday morning).
    """
    tmp = df_test.copy()
    tmp["pred"] = proba
    tmp["weekday"] = tmp["date_dt"].dt.weekday
    tmp = tmp[tmp["weekday"] == 0].copy()  # Monday only

    rows = []
    for d, g in tmp.groupby(tmp["date_dt"].dt.normalize()):
        g2 = g.sort_values("pred", ascending=False).head(k)
        prec = float(g2[y_col].mean()) if len(g2) > 0 else float("nan")
        rows.append({"monday_date": str(pd.to_datetime(d).date()), "precision_at_k": prec, "k": int(len(g2))})

    out = pd.DataFrame(rows)
    mean_prec = float(out["precision_at_k"].mean()) if not out.empty else float("nan")
    return mean_prec, out


# -----------------------------
# Calibration + plots
# -----------------------------
def maybe_calibrate(estimator: Pipeline, X_val: pd.DataFrame, y_val: pd.Series, method: str):
    """
    Calibrate probabilities if validation contains both classes.
    If not, return the raw estimator (still usable).
    """
    uniq = set(pd.Series(y_val).dropna().unique().tolist())
    if uniq != {0, 1}:
        return estimator, False

    calib = CalibratedClassifierCV(estimator=estimator, method=method, cv="prefit")
    calib.fit(X_val, y_val)
    return calib, True


def plot_calibration(y_true: np.ndarray, proba: np.ndarray, title: str, out_figures: Path, stem: str) -> None:
    """Reliability diagram (calibration curve)."""
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")

    fig = plt.figure()
    ax = plt.gca()
    ax.plot(mean_pred, frac_pos, marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.grid(True, alpha=0.3)
    save_fig(fig, out_figures, stem)


def plot_pr_curve(y_true: np.ndarray, proba: np.ndarray, title: str, out_figures: Path, stem: str) -> None:
    """Precision–Recall curve (more informative than ROC for rare events)."""
    precision, recall, _ = precision_recall_curve(y_true, proba)

    fig = plt.figure()
    ax = plt.gca()
    ax.plot(recall, precision)
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True, alpha=0.3)
    save_fig(fig, out_figures, stem)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    raw_dir = Path(cfg["paths"]["data_raw_dir"])
    out_tables = Path(cfg["paths"]["tables_dir"])
    out_figures = Path(cfg["paths"]["figures_dir"])
    out_models = Path(cfg["paths"]["outputs_dir"]) / "models"

    ensure_dir(out_tables)
    ensure_dir(out_figures)
    ensure_dir(out_models)

    seed = int(cfg["project"]["random_seed"])
    k_week = int(cfg["modeling"]["evaluation"]["k_service_per_week"])
    calib_method = str(cfg["modeling"]["calibration"]["method"])

    df = load_vehicle_day(raw_dir)

    split = cfg["modeling"]["time_split_days"]
    df_train, df_val, df_test = time_split_by_days(
        df,
        train_days=int(split["train"]),
        val_days=int(split["val"]),
        test_days=int(split["test"]),
    )

    preprocessor, feature_cols = build_preprocessor(cfg)

    horizons = [7, 30]
    label_map = {7: "breakdown_7d", 30: "breakdown_30d"}

    metrics_rows = []
    p_at_k_all = []

    for h in horizons:
        y_col = label_map[h]

        X_train = df_train[feature_cols]
        y_train = df_train[y_col].astype(int)

        X_val = df_val[feature_cols]
        y_val = df_val[y_col].astype(int)

        X_test = df_test[feature_cols]
        y_test = df_test[y_col].astype(int).values

        # -----------------------------
        # Model A: Logistic Regression (baseline)
        # -----------------------------
        logreg = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        )

        logreg_pipe = Pipeline([("prep", preprocessor), ("clf", logreg)])
        logreg_pipe.fit(X_train, y_train)

        logreg_model, logreg_cal = maybe_calibrate(logreg_pipe, X_val, y_val, calib_method)
        proba_logreg = logreg_model.predict_proba(X_test)[:, 1]

        auprc = float(average_precision_score(y_test, proba_logreg))
        brier = float(brier_score_loss(y_test, proba_logreg))

        p_at_k, p_tbl = precision_at_k_mondays(df_test, y_col, proba_logreg, k=k_week)
        p_tbl["horizon_days"] = h
        p_tbl["model"] = "logreg"
        p_at_k_all.append(p_tbl)

        plot_calibration(y_test, proba_logreg, f"Calibration: LogReg ({h}d)", out_figures, f"calibration_logreg_{h}d")
        plot_pr_curve(y_test, proba_logreg, f"PR curve: LogReg ({h}d)", out_figures, f"pr_logreg_{h}d")

        joblib.dump(logreg_model, out_models / f"risk_{h}d_logreg.joblib")

        # Save test predictions (useful later for Streamlit + debugging)
        pred_path = out_tables / f"test_predictions_{h}d_logreg.csv"
        df_test[["vehicle_id", "date"]].assign(y_true=y_test, proba=proba_logreg).to_csv(pred_path, index=False)

        metrics_rows.append(
            {
                "task": f"risk_{h}d",
                "model": "logreg",
                "calibrated": int(logreg_cal),
                "auprc": auprc,
                "brier": brier,
                "precision_at_k_mondays_mean": p_at_k,
            }
        )

        # -----------------------------
        # Model B: HistGradientBoosting (advanced)
        # -----------------------------
        hgb = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=300,
            random_state=seed,
        )

        hgb_pipe = Pipeline([("prep", preprocessor), ("clf", hgb)])

        # Simple weighting for imbalance (stable and easy to explain)
        pos_rate = float(y_train.mean())
        w_pos = 0.5 / max(pos_rate, 1e-6)
        w_neg = 0.5 / max(1.0 - pos_rate, 1e-6)
        sample_weight = np.where(y_train.values == 1, w_pos, w_neg).astype(float)

        hgb_pipe.fit(X_train, y_train, clf__sample_weight=sample_weight)

        hgb_model, hgb_cal = maybe_calibrate(hgb_pipe, X_val, y_val, calib_method)
        proba_hgb = hgb_model.predict_proba(X_test)[:, 1]

        auprc2 = float(average_precision_score(y_test, proba_hgb))
        brier2 = float(brier_score_loss(y_test, proba_hgb))

        p_at_k2, p_tbl2 = precision_at_k_mondays(df_test, y_col, proba_hgb, k=k_week)
        p_tbl2["horizon_days"] = h
        p_tbl2["model"] = "hgb"
        p_at_k_all.append(p_tbl2)

        plot_calibration(y_test, proba_hgb, f"Calibration: HGB ({h}d)", out_figures, f"calibration_hgb_{h}d")
        plot_pr_curve(y_test, proba_hgb, f"PR curve: HGB ({h}d)", out_figures, f"pr_hgb_{h}d")

        joblib.dump(hgb_model, out_models / f"risk_{h}d_hgb.joblib")

        pred_path2 = out_tables / f"test_predictions_{h}d_hgb.csv"
        df_test[["vehicle_id", "date"]].assign(y_true=y_test, proba=proba_hgb).to_csv(pred_path2, index=False)

        metrics_rows.append(
            {
                "task": f"risk_{h}d",
                "model": "hgb",
                "calibrated": int(hgb_cal),
                "auprc": auprc2,
                "brier": brier2,
                "precision_at_k_mondays_mean": p_at_k2,
            }
        )

    # -----------------------------
    # Export LaTeX-ready tables
    # -----------------------------
    metrics_df = pd.DataFrame(metrics_rows).sort_values(["task", "model"])
    save_table(
        metrics_df,
        out_dir=out_tables,
        stem="risk_metrics",
        caption="Risk model performance on the test split.",
        label="tab:risk_metrics",
        index=False,
        float_decimals=4,
    )

    p_at_k_df = pd.concat(p_at_k_all, ignore_index=True) if len(p_at_k_all) else pd.DataFrame()
    save_table(
        p_at_k_df,
        out_dir=out_tables,
        stem="precision_at_k_mondays",
        caption="Precision@K evaluated on Mondays (weekly service planning proxy).",
        label="tab:precision_at_k_mondays",
        index=False,
        float_decimals=4,
    )

    # Also save a small JSON summary (useful for agent/report automation).
    summary = {
        "horizons": horizons,
        "k_service_per_week": k_week,
        "calibration_method": calib_method,
        "n_train_rows": int(df_train.shape[0]),
        "n_val_rows": int(df_val.shape[0]),
        "n_test_rows": int(df_test.shape[0]),
    }
    save_json(summary, out_tables / "train_run_summary.json")

    print("=== Training artifacts generated (LaTeX-ready) ===")
    print("tables: outputs/tables/risk_metrics.(csv/tex)")
    print("tables: outputs/tables/precision_at_k_mondays.(csv/tex)")
    print("tables: outputs/tables/test_predictions_*.csv")
    print("figures: outputs/figures/calibration_*.pdf + pr_*.pdf")
    print("models: outputs/models/*.joblib")


if __name__ == "__main__":
    main()