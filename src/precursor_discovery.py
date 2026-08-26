"""Discover recurring precursor patterns for HSE investigation.

These grouped patterns show recurrence and SIF density. They do not prove
causality and should be investigated by HSE personnel before decisions are
made.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_reports.csv"
GROUP_COLUMNS = ["site", "activity", "life_saving_rule", "barrier_failure"]


def _sif_count(values: pd.Series) -> int:
    return int(values.astype(str).str.strip().str.lower().eq("yes").sum())


def _risk_level(report_count: int, sif_percentage: float) -> str:
    """Assign a simple transparent triage level, not a causal risk estimate."""
    if report_count >= 3 and sif_percentage >= 60.0:
        return "HIGH"
    if report_count >= 2 and sif_percentage >= 30.0:
        return "MEDIUM"
    return "LOW"


def discover_precursor_patterns(
    reports: pd.DataFrame,
    min_reports: int = 2,
) -> pd.DataFrame:
    """Return recurring grouped precursor patterns ranked for HSE review.

    Groups are formed by site, activity, Life-Saving Rule, and barrier failure.
    ``min_reports`` prevents one-off observations from being called recurring.
    """
    required = set(GROUP_COLUMNS + ["sif_potential"])
    missing = required.difference(reports.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    grouped = (
        reports.groupby(GROUP_COLUMNS, dropna=False, sort=False)
        .agg(
            report_count=("sif_potential", "size"),
            sif_potential_count=("sif_potential", _sif_count),
        )
        .reset_index()
    )
    grouped = grouped[grouped["report_count"] >= min_reports].copy()
    grouped["sif_percentage"] = (
        grouped["sif_potential_count"] / grouped["report_count"] * 100
    ).round(1)
    grouped["risk_level"] = grouped.apply(
        lambda row: _risk_level(int(row["report_count"]), float(row["sif_percentage"])),
        axis=1,
    )
    risk_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    grouped["_risk_order"] = grouped["risk_level"].map(risk_order)
    return (
        grouped.sort_values(
            ["_risk_order", "sif_percentage", "report_count"],
            ascending=[False, False, False],
        )
        .drop(columns="_risk_order")
        .reset_index(drop=True)
    )


def highest_risk_patterns(
    reports: pd.DataFrame,
    top_n: int = 10,
    min_reports: int = 2,
) -> pd.DataFrame:
    """Return the highest-ranked recurring patterns for HSE investigation."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    return discover_precursor_patterns(reports, min_reports=min_reports).head(top_n)


def load_and_discover(
    data_path: Path = DATA_PATH,
    top_n: int = 10,
    min_reports: int = 2,
) -> pd.DataFrame:
    """Load the CSV and return the highest-risk recurring patterns."""
    reports = pd.read_csv(data_path)
    return highest_risk_patterns(reports, top_n=top_n, min_reports=min_reports)


if __name__ == "__main__":
    patterns = load_and_discover()
    print(patterns.to_string(index=False))
