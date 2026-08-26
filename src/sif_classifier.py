"""Explainable TF-IDF + Logistic Regression classifier for SIF potential."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_reports.csv"
MODEL_DIR = PROJECT_ROOT / "models"
VECTORIZER_PATH = MODEL_DIR / "sif_tfidf_vectorizer.joblib"
CLASSIFIER_PATH = MODEL_DIR / "sif_logistic_regression.joblib"


def _normalise_target(values: pd.Series) -> pd.Series:
    """Convert the dataset's Yes/No labels into binary integers."""
    return values.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})


def train_and_evaluate(
    data_path: Path = DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 26165,
) -> dict[str, Any]:
    """Train, evaluate, and save the SIF classifier artifacts."""
    reports = pd.read_csv(data_path)
    required_columns = {"report_text", "sif_potential"}
    missing = required_columns.difference(reports.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    texts = reports["report_text"].fillna("").astype(str)
    targets = _normalise_target(reports["sif_potential"])
    if targets.isna().any():
        raise ValueError("sif_potential must contain only Yes or No labels")

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        targets,
        test_size=test_size,
        random_state=random_state,
        stratify=targets,
    )

    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    classifier.fit(x_train_tfidf, y_train)
    predictions = classifier.predict(x_test_tfidf)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(classifier, CLASSIFIER_PATH)

    print("SIF classification evaluation")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}  (SIF recall)")
    print(f"F1:        {metrics['f1']:.3f}")
    print("Confusion matrix [[non-SIF, predicted SIF], [SIF, predicted SIF]]:")
    print(metrics["confusion_matrix"])
    print(f"Saved vectorizer: {VECTORIZER_PATH}")
    print(f"Saved classifier: {CLASSIFIER_PATH}")
    return metrics


def load_model(
    vectorizer_path: Path = VECTORIZER_PATH,
    classifier_path: Path = CLASSIFIER_PATH,
) -> tuple[TfidfVectorizer, LogisticRegression]:
    """Load previously trained artifacts without retraining."""
    return joblib.load(vectorizer_path), joblib.load(classifier_path)


def predict_sif_potential(report_text: str) -> dict[str, Any]:
    """Predict SIF potential for one report using saved artifacts."""
    vectorizer, classifier = load_model()
    features = vectorizer.transform([report_text])
    prediction = int(classifier.predict(features)[0])
    probability = float(classifier.predict_proba(features)[0][prediction])
    return {
        "sif_potential": "Yes" if prediction == 1 else "No",
        "sif_probability": probability,
    }


if __name__ == "__main__":
    train_and_evaluate()
