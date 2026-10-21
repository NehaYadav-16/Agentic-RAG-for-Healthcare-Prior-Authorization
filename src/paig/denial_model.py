"""Denial-risk classifier: predicts P(denied) for a prior-auth claim
before submission, using structured claim features."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL_FEATURES = ["procedure_category", "payer"]
NUMERIC_FEATURES = [
    "patient_age",
    "documentation_completeness",
    "conservative_treatment_tried",
    "criteria_met_ratio",
    "prior_denial_history",
    "peer_to_peer_requested",
    "days_to_submit",
]
TARGET = "denied"


@dataclass
class EvalMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "roc_auc": round(self.roc_auc, 4),
        }


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",
    )
    model = GradientBoostingClassifier(random_state=42)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def train_denial_model(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    X = df[features]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = EvalMetrics(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred),
        roc_auc=roc_auc_score(y_test, y_proba),
    )
    return pipeline, metrics


def predict_denial_risk(pipeline: Pipeline, claim_features: Dict) -> float:
    """Returns a denial probability in [0, 1] for a single claim dict."""
    row = pd.DataFrame([claim_features])[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    return float(pipeline.predict_proba(row)[0, 1])


def save_model(pipeline: Pipeline, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(path: str | Path) -> Pipeline:
    return joblib.load(path)
