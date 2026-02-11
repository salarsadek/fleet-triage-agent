"""
Create stable "latest" aliases for date-stamped triage snapshot artifacts.

Why this exists
---------------
The triage snapshot exporter produces stamp-based filenames, e.g.
  service_queue_2024-06-28_30d_hgb_cost.tex
For automated LaTeX, we want stable filenames:
  service_queue_latest.tex
  fleet_risk_hist_latest.pdf
  triage_snapshot_latest.tex

This script finds the latest triage stamp and copies the relevant files.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml


@dataclass(frozen=True)
class Paths:
    root: Path
    out_tables: Path
    out_figs: Path
    out_reports: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config did not parse as a dictionary.")
    return cfg


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def newest_file(pattern: str, folder: Path) -> Optional[Path]:
    files = list(folder.glob(pattern))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def extract_stamp(service_queue_tex: Path) -> str:
    name = service_queue_tex.name
    if not name.startswith("service_queue_") or not name.endswith(".tex"):
        raise ValueError(f"Unexpected filename format: {name}")
    return name[len("service_queue_") : -len(".tex")]


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    shutil.copyfile(src, dst)
    return True


def save_index_table(df: pd.DataFrame, out_tables: Path, stem: str, caption: str, label: str) -> None:
    ensure_dir(out_tables)
    csv_path = out_tables / f"{stem}.csv"
    tex_path = out_tables / f"{stem}.tex"
    df.to_csv(csv_path, index=False)

    tabular = df.to_latex(index=False, escape=True)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--top_similar", type=int, default=5, help="How many similar_cases tables to alias to *_latest_XX.*")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path

    cfg = load_yaml(cfg_path)

    out_tables = root / cfg["paths"]["tables_dir"]
    out_figs = root / cfg["paths"]["figures_dir"]
    out_reports = root / cfg["paths"]["reports_dir"]

    ensure_dir(out_tables)
    ensure_dir(out_figs)
    ensure_dir(out_reports)

    sq = newest_file("service_queue_*.tex", out_tables)
    if sq is None:
        raise FileNotFoundError("No service_queue_*.tex found. Run triage snapshot exporter first.")

    stamp = extract_stamp(sq)

    # Core triage artifacts
    copies = [
        (out_tables / f"service_queue_{stamp}.tex", out_tables / "service_queue_latest.tex"),
        (out_tables / f"service_queue_{stamp}.csv", out_tables / "service_queue_latest.csv"),
        (out_tables / f"abstained_vehicles_{stamp}.tex", out_tables / "abstained_vehicles_latest.tex"),
        (out_tables / f"abstained_vehicles_{stamp}.csv", out_tables / "abstained_vehicles_latest.csv"),
        (out_tables / f"mean_risk_by_route_{stamp}.tex", out_tables / "mean_risk_by_route_latest.tex"),
        (out_tables / f"mean_risk_by_route_{stamp}.csv", out_tables / "mean_risk_by_route_latest.csv"),
        (out_tables / f"triage_summary_{stamp}.json", out_tables / "triage_summary_latest.json"),
        (out_figs / f"fleet_risk_hist_{stamp}.pdf", out_figs / "fleet_risk_hist_latest.pdf"),
        (out_figs / f"fleet_risk_hist_{stamp}.png", out_figs / "fleet_risk_hist_latest.png"),
        (out_figs / f"fleet_cost_vs_risk_{stamp}.pdf", out_figs / "fleet_cost_vs_risk_latest.pdf"),
        (out_figs / f"fleet_cost_vs_risk_{stamp}.png", out_figs / "fleet_cost_vs_risk_latest.png"),
        (out_reports / f"triage_snapshot_{stamp}.tex", out_reports / "triage_snapshot_latest.tex"),
    ]

    n_ok = 0
    for src, dst in copies:
        if copy_if_exists(src, dst):
            n_ok += 1

    (out_reports / "triage_stamp_latest.txt").write_text(stamp, encoding="utf-8")

    # Similar-case evidence (optional aliases)
    sim_tex = sorted(out_tables.glob(f"similar_cases_{stamp}_*.tex"), key=lambda p: p.stat().st_mtime, reverse=True)
    sim_tex = sim_tex[: int(args.top_similar)]

    index_rows: List[dict] = []
    for i, tex_path in enumerate(sim_tex, start=1):
        base = tex_path.name.replace(".tex", "")
        # similar_cases_{stamp}_{VID}
        parts = base.split("_")
        vehicle_id = parts[-1] if parts else "unknown"

        dst_tex = out_tables / f"similar_cases_latest_{i:02d}.tex"
        dst_csv = out_tables / f"similar_cases_latest_{i:02d}.csv"

        copy_if_exists(tex_path, dst_tex)
        copy_if_exists(out_tables / f"{base}.csv", dst_csv)

        index_rows.append(
            {
                "rank": i,
                "vehicle_id": vehicle_id,
                "alias_tex": dst_tex.name,
                "alias_csv": dst_csv.name,
            }
        )

    if index_rows:
        df_index = pd.DataFrame(index_rows)
        save_index_table(
            df_index,
            out_tables,
            stem="similar_cases_index_latest",
            caption="Index of aliased similar-case evidence tables (latest snapshot).",
            label="tab:similar_cases_index_latest",
        )

    print("=== Latest aliases created ===")
    print(f"stamp: {stamp}")
    print(f"copied core artifacts: {n_ok}/{len(copies)}")
    if index_rows:
        print("similar-case aliases: outputs/tables/similar_cases_latest_XX.(csv/tex)")
        print("index table: outputs/tables/similar_cases_index_latest.(csv/tex)")


if __name__ == "__main__":
    main()