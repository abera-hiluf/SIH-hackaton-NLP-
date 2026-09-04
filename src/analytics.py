"""Pure Pandas calculations used by the HSE dashboard."""

from __future__ import annotations

import pandas as pd


def sif_mask(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().eq("yes")


def prepare_reports(reports: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize demo/real-compatible report fields."""
    required = {"report_id", "date", "report_text", "sif_potential", "life_saving_rule"}
    missing = required.difference(reports.columns)
    if missing:
        raise ValueError(f"Missing required report fields: {sorted(missing)}")
    output = reports.copy()
    def series_or_default(column: str, default: object) -> pd.Series:
        if column in output.columns:
            return output[column].fillna(default)
        return pd.Series(default, index=output.index)

    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    output["site"] = series_or_default("site", "Unknown")
    output["location"] = series_or_default("location", output["site"])
    output["activity"] = series_or_default("activity", "Unknown")
    output["precursor"] = series_or_default("precursor", output["life_saving_rule"])
    output["barrier_failure"] = series_or_default("barrier_failure", "Not specified")
    output["priority"] = series_or_default(
        "priority", output["sif_potential"].map({"Yes": "High", "No": "Low"})
    ).fillna("Review")
    output["sif_bool"] = sif_mask(output["sif_potential"])
    return output


def calculate_kpis(reports: pd.DataFrame) -> dict[str, int | float | str]:
    data = prepare_reports(reports)
    precursor = data["precursor"].value_counts()
    rule = data["life_saving_rule"].value_counts()
    return {
        "total_reports": len(data),
        "sif_reports": int(data["sif_bool"].sum()),
        "non_sif_reports": int((~data["sif_bool"]).sum()),
        "high_priority_reports": int(data["priority"].astype(str).str.upper().eq("HIGH").sum()),
        "top_precursor": str(precursor.index[0]) if not precursor.empty else "None",
        "top_rule": str(rule.index[0]) if not rule.empty else "None",
    }


def rank_dimension(reports: pd.DataFrame, dimension: str) -> pd.DataFrame:
    data = prepare_reports(reports)
    summary = data.groupby(dimension, dropna=False).agg(
        reports=("report_id", "count"),
        sif_count=("sif_bool", "sum"),
    ).reset_index()
    summary["sif_density"] = (summary["sif_count"] / summary["reports"] * 100).round(1)
    return summary.sort_values(["sif_density", "reports"], ascending=False).reset_index(drop=True)


def precursor_summary(reports: pd.DataFrame) -> pd.DataFrame:
    data = prepare_reports(reports)
    summary = data.groupby("precursor", dropna=False).agg(
        frequency=("report_id", "count"),
        sif_count=("sif_bool", "sum"),
        activities=("activity", lambda values: ", ".join(sorted(set(map(str, values))))),
        locations=("location", lambda values: ", ".join(sorted(set(map(str, values))))),
    ).reset_index()
    summary["sif_density"] = (summary["sif_count"] / summary["frequency"] * 100).round(1)
    return summary.sort_values(["frequency", "sif_density"], ascending=False).reset_index(drop=True)


def rule_summary(reports: pd.DataFrame) -> pd.DataFrame:
    data = prepare_reports(reports)
    summary = data.groupby("life_saving_rule", dropna=False).agg(
        frequency=("report_id", "count"),
        sif_count=("sif_bool", "sum"),
    ).reset_index()
    summary["percentage"] = (summary["frequency"] / len(data) * 100).round(1)
    return summary.sort_values("frequency", ascending=False).reset_index(drop=True)


def monthly_trend(reports: pd.DataFrame, dimension: str | None = None, value: str | None = None) -> pd.DataFrame:
    data = prepare_reports(reports).dropna(subset=["date"]).copy()
    data["month"] = data["date"].dt.to_period("M").dt.to_timestamp()
    if dimension and value:
        data = data[data[dimension].astype(str).eq(str(value))]
    trend = data.groupby("month").agg(
        reports=("report_id", "count"),
        sif_count=("sif_bool", "sum"),
    ).reset_index()
    return trend.sort_values("month")


def filter_reports(
    reports: pd.DataFrame,
    query: str = "",
    sif_potential: str = "All",
    selections: dict[str, list[str]] | None = None,
    start_date: object | None = None,
    end_date: object | None = None,
) -> pd.DataFrame:
    """Apply searchable structured filters without changing the source data."""
    data = prepare_reports(reports)
    if query.strip():
        matches = data.astype(str).apply(lambda column: column.str.contains(query, case=False, na=False))
        data = data[matches.any(axis=1)]
    if sif_potential != "All":
        data = data[data["sif_potential"].astype(str).eq(sif_potential)]
    for column, selected in (selections or {}).items():
        if selected and column in data.columns:
            data = data[data[column].astype(str).isin(selected)]
    if start_date is not None:
        data = data[data["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        data = data[data["date"] < pd.Timestamp(end_date) + pd.Timedelta(days=1)]
    return data.reset_index(drop=True)
