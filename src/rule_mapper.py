"""Map safety-report text to a controlled Life-Saving Rule taxonomy.

The mapping uses semantic similarity only; it does not call an LLM. Results
are intended as explainable triage suggestions for HSE review.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Calibrated for the short operational narratives used by this prototype.
REVIEW_THRESHOLD = 0.15

RULE_DESCRIPTIONS = {
    "Energy Isolation": "Control hazardous energy by isolating, locking, tagging, and verifying zero energy before work.",
    "Hot Work": "Control ignition sources and combustible materials during welding, cutting, grinding, or other hot work.",
    "Confined Space": "Authorize, test, ventilate, monitor, and provide rescue controls before entering a confined space.",
    "Line of Fire": "Keep people clear of stored energy, moving equipment, suspended loads, and unexpected release paths.",
    "Working at Height": "Prevent falls and dropped objects by using safe access, edge protection, and fall protection at height.",
    "Lifting Operations": "Plan and control lifting operations with inspected equipment, a competent signaler, and an exclusion zone.",
    "Driving / Vehicle Safety": "Drive defensively, use seat belts, control vehicle movements, and secure loads during journeys.",
}


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load the embedding model once per process and keep inference repeatable."""
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def _rule_embeddings() -> tuple[list[str], np.ndarray]:
    """Create normalized embeddings for the fixed taxonomy descriptions."""
    rules = list(RULE_DESCRIPTIONS)
    embeddings = _load_model().encode(
        list(RULE_DESCRIPTIONS.values()),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return rules, embeddings


def map_report_to_rule(
    report_text: str,
    threshold: float = REVIEW_THRESHOLD,
) -> dict[str, Any]:
    """Return the closest rule and cosine similarity for one report.

    A low-confidence result is returned as ``Requires HSE Review`` instead of
    forcing a taxonomy label.
    """
    if not isinstance(report_text, str) or not report_text.strip():
        return {"life_saving_rule": "Requires HSE Review", "similarity_score": 0.0}

    rules, descriptions = _rule_embeddings()
    report_embedding = _load_model().encode(
        [report_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    scores = descriptions @ report_embedding
    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])
    best_rule = rules[best_index] if best_score >= threshold else "Requires HSE Review"
    return {
        "life_saving_rule": best_rule,
        "similarity_score": round(best_score, 4),
    }


def map_reports_to_rules(report_texts: list[str], threshold: float = REVIEW_THRESHOLD) -> list[dict[str, Any]]:
    """Map multiple reports using the same cached taxonomy embeddings."""
    return [map_report_to_rule(text, threshold=threshold) for text in report_texts]
