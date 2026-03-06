from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ndcg_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import DEFAULT_MODEL_PATH, RANDOM_SEED


class PrioritizationModelAgent:
    """Trains and serves an XGBoost-based priority model."""

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.model = None
        self.model_kind = ""
        self.label_encoder = LabelEncoder()
        self.class_score_weights = {"High": 100.0, "Medium": 65.0, "Low": 30.0}

    def _build_model(self, num_classes: int):
        try:
            from xgboost import XGBClassifier

            self.model_kind = "xgboost"
            cpu_count = max(2, min(16, os.cpu_count() or 4))
            return XGBClassifier(
                n_estimators=180,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="multi:softprob",
                num_class=num_classes,
                random_state=self.random_state,
                eval_metric="mlogloss",
                tree_method="hist",
                n_jobs=cpu_count,
            )
        except Exception:
            from sklearn.ensemble import HistGradientBoostingClassifier

            self.model_kind = "hist_gradient_boosting_fallback"
            return HistGradientBoostingClassifier(random_state=self.random_state)

    @staticmethod
    def _one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
        out = np.zeros((len(y), num_classes))
        out[np.arange(len(y)), y] = 1
        return out

    def train(self, x: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> dict:
        y_encoded = self.label_encoder.fit_transform(y)
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y_encoded,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y_encoded,
        )

        self.model = self._build_model(num_classes=len(self.label_encoder.classes_))
        self.model.fit(x_train, y_train)
        return self.evaluate(x_test, y_test)

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> dict:
        if self.model is None:
            raise RuntimeError("Model has not been trained.")

        y_pred = self.model.predict(x_test)
        y_proba = self.model.predict_proba(x_test)
        labels = np.arange(len(self.label_encoder.classes_))

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        ndcg = ndcg_score(
            self._one_hot(y_test, num_classes=len(self.label_encoder.classes_)),
            y_proba,
        )

        report = classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(y_test, y_pred, labels=labels).tolist()
        return {
            "model_kind": self.model_kind,
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision_weighted": round(float(precision), 4),
            "recall_weighted": round(float(recall), 4),
            "f1_weighted": round(float(f1), 4),
            "ndcg": round(float(ndcg), 4),
            "classification_report": report,
            "confusion_matrix": matrix,
        }

    def _probability_to_scores(self, probabilities: np.ndarray) -> np.ndarray:
        class_weights = np.array(
            [self.class_score_weights.get(label, 50.0) for label in self.label_encoder.classes_]
        )
        scores = probabilities @ class_weights
        return np.clip(np.round(scores, 2), 0, 100)

    def predict(self, x: np.ndarray):
        if self.model is None:
            raise RuntimeError("Model is not loaded or trained.")

        y_encoded = self.model.predict(x)
        y_proba = self.model.predict_proba(x)
        labels = self.label_encoder.inverse_transform(y_encoded.astype(int))
        scores = self._probability_to_scores(y_proba)
        return labels, scores, y_proba

    def save(self, model_path: Path = DEFAULT_MODEL_PATH) -> Path:
        if self.model is None:
            raise RuntimeError("Model is not trained and cannot be saved.")

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": self.model,
            "model_kind": self.model_kind,
            "label_encoder": self.label_encoder,
            "class_score_weights": self.class_score_weights,
        }
        joblib.dump(artifact, model_path)
        return model_path

    @classmethod
    def load(cls, model_path: Path = DEFAULT_MODEL_PATH) -> "PrioritizationModelAgent":
        artifact = joblib.load(model_path)
        agent = cls()
        agent.model = artifact["model"]
        agent.model_kind = artifact.get("model_kind", "")
        agent.label_encoder = artifact["label_encoder"]
        agent.class_score_weights = artifact.get("class_score_weights", agent.class_score_weights)
        return agent
