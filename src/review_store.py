"""Small CSV persistence layer for human review decisions."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REVIEW_COLUMNS = [
    "report_id", "ai_prediction", "ai_confidence", "reviewer_name",
    "reviewer_decision", "corrected_classification", "reviewer_comment", "timestamp",
]


def load_reviews(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    return pd.read_csv(path)


def save_review(path: Path, review: dict[str, object]) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_reviews(path)
    record = {column: review.get(column, "") for column in REVIEW_COLUMNS}
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    existing = existing[existing["report_id"].astype(str) != str(record["report_id"])]
    updated = pd.concat([existing, pd.DataFrame([record])], ignore_index=True)
    updated.to_csv(path, index=False)
    return updated
