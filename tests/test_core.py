"""Lightweight tests for the prototype's pure calculations and persistence."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analytics import calculate_kpis, filter_reports, precursor_summary, prepare_reports
from src.review_store import load_reviews, save_review


class PrototypeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.DataFrame(
            [
                {"report_id": "1", "date": "2025-01-01", "site": "Duliajan", "activity": "Pump Maintenance", "report_text": "Isolation issue", "sif_potential": "Yes", "life_saving_rule": "Energy Isolation", "precursor": "Isolation verification failure", "barrier_failure": "Isolation Verification", "priority": "High"},
                {"report_id": "2", "date": "2025-01-02", "site": "Duliajan", "activity": "Pump Maintenance", "report_text": "Observation", "sif_potential": "No", "life_saving_rule": "Energy Isolation", "precursor": "Isolation verification failure", "barrier_failure": "Isolation Verification", "priority": "Low"},
                {"report_id": "3", "date": "2025-02-01", "site": "Digboi", "activity": "Hot Work", "report_text": "Sparks", "sif_potential": "Yes", "life_saving_rule": "Hot Work", "precursor": "Hot work control failure", "barrier_failure": "Permit", "priority": "High"},
            ]
        )

    def test_kpis_and_precursor_density(self) -> None:
        kpis = calculate_kpis(self.data)
        self.assertEqual(kpis["total_reports"], 3)
        self.assertEqual(kpis["sif_reports"], 2)
        summary = precursor_summary(self.data)
        self.assertEqual(summary.iloc[0]["frequency"], 2)
        self.assertEqual(summary.iloc[0]["sif_density"], 50.0)

    def test_filters_and_missing_optional_fields(self) -> None:
        filtered = filter_reports(self.data, query="duliajan", sif_potential="Yes")
        self.assertEqual(filtered["report_id"].tolist(), ["1"])
        minimal = self.data.drop(columns=["precursor", "priority"])
        prepared = prepare_reports(minimal)
        self.assertIn("precursor", prepared.columns)
        self.assertIn("priority", prepared.columns)

    def test_missing_required_field(self) -> None:
        with self.assertRaises(ValueError):
            prepare_reports(self.data.drop(columns=["report_text"]))

    def test_review_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.csv"
            save_review(path, {"report_id": "1", "ai_prediction": "SIF Potential", "ai_confidence": 0.9, "reviewer_name": "HSE", "reviewer_decision": "Confirm AI Assessment", "corrected_classification": "SIF Potential", "reviewer_comment": "Verified"})
            reviews = load_reviews(path)
            self.assertEqual(len(reviews), 1)
            self.assertEqual(reviews.iloc[0]["reviewer_name"], "HSE")


if __name__ == "__main__":
    unittest.main()
