"""Transparent report-level extraction for evidence and investigation views."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


SIGNAL_TERMS = {
    "maintenance activity": ["maintenance", "servicing", "replacement", "repair"],
    "isolation issue": ["isolation", "lockout", "tagout", "zero energy", "pressure"],
    "worker exposure": ["worker", "entrant", "crew", "personnel", "pedestrian"],
    "hot work exposure": ["welding", "grinding", "cutting", "sparks", "flammable"],
    "height / dropped-object exposure": ["height", "edge", "scaffold", "fall", "dropped"],
    "moving-load / line-of-fire exposure": ["load", "suspended", "hose", "swing", "whip"],
    "vehicle exposure": ["vehicle", "driver", "reversing", "pedestrian", "journey"],
}


def _first_value(row: pd.Series, *columns: str) -> str:
    for column in columns:
        if column in row.index and pd.notna(row[column]) and str(row[column]).strip():
            return str(row[column])
    return "Not detected"


def extract_report_features(row: pd.Series) -> dict[str, Any]:
    """Extract only fields/signals supported by the report and structured data."""
    text = str(row.get("report_text", ""))
    lowered = text.lower()
    evidence = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    signals = [label for label, terms in SIGNAL_TERMS.items() if any(term in lowered for term in terms)]
    return {
        "activity": _first_value(row, "activity"),
        "location": _first_value(row, "location", "site"),
        "hazard": _first_value(row, "hazard"),
        "precursor": _first_value(row, "precursor", "barrier_failure", "life_saving_rule"),
        "barrier": _first_value(row, "barrier"),
        "barrier_failure": _first_value(row, "barrier_failure"),
        "signals": signals,
        "evidence": evidence[:4],
    }
