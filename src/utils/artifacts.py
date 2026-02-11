"""
Artifact utilities: save tables/figures in a LaTeX-friendly way.

Design goals
------------
- Every important result should be saved as:
    * CSV (easy to inspect / re-use)
    * TEX (easy to include in LaTeX via \\input{...})
- Keep formatting stable and reproducible across runs.

Note
----
pandas.DataFrame.to_latex() does NOT support a "booktabs=" argument in some versions.
However, it typically outputs \\toprule/\\midrule/\\bottomrule by default, which requires
\\usepackage{booktabs} in the LaTeX report. We keep the report using booktabs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import json
import pandas as pd


def ensure_dir(p: Path) -> None:
    """Create directory if missing (including parents)."""
    p.mkdir(parents=True, exist_ok=True)


def dataframe_to_latex(
    df: pd.DataFrame,
    caption: Optional[str],
    label: Optional[str],
    index: bool = False,
    float_decimals: int = 3,
    table_env: bool = True,
) -> str:
    """
    Convert a DataFrame into LaTeX.

    Why we wrap our own table environment
    ------------------------------------
    We want a single .tex file that can be included directly with \\input{...}.
    That makes it easy to auto-generate a report from outputs/ without manual edits.
    """
    # Keep formatting stable: fixed decimals, "-" for missing, escape special chars.
    tabular = df.to_latex(
        index=index,
        escape=True,
        na_rep="-",
        float_format=(lambda x: f"{x:.{float_decimals}f}"),
    )

    if not table_env:
        return tabular + "\n"

    parts = [
        "% Auto-generated. Do not edit by hand.",
        r"\begin{table}[htbp]",
        r"\centering",
    ]
    if caption:
        parts.append(rf"\caption{{{caption}}}")
    if label:
        parts.append(rf"\label{{{label}}}")
    parts.append(tabular)
    parts.append(r"\end{table}")
    return "\n".join(parts) + "\n"


def save_table(
    df: pd.DataFrame,
    out_dir: Path,
    stem: str,
    caption: Optional[str] = None,
    label: Optional[str] = None,
    index: bool = False,
    float_decimals: int = 3,
) -> Dict[str, str]:
    """
    Save a DataFrame to:
      - CSV: <stem>.csv
      - TEX: <stem>.tex
    """
    ensure_dir(out_dir)
    csv_path = out_dir / f"{stem}.csv"
    tex_path = out_dir / f"{stem}.tex"

    df.to_csv(csv_path, index=index)

    tex = dataframe_to_latex(
        df=df,
        caption=caption,
        label=label,
        index=index,
        float_decimals=float_decimals,
        table_env=True,
    )
    tex_path.write_text(tex, encoding="utf-8")

    return {"csv": str(csv_path), "tex": str(tex_path)}


def save_json(obj: Any, out_path: Path) -> str:
    """Save JSON (UTF-8) with indentation."""
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return str(out_path)