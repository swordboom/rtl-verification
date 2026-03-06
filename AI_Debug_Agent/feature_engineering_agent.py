from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import DEFAULT_FEATURE_PIPELINE_PATH


class FeatureEngineeringAgent:
    """Builds model-ready features from tabular + log text inputs."""

    severity_map = {"fatal": 3, "error": 2, "warning": 1}
    module_criticality_map = {
        "ALU": 1.3,
        "Cache": 1.6,
        "MemoryCtrl": 1.7,
        "DMA": 1.2,
        "UART": 1.0,
        "PCIe": 1.5,
    }

    def __init__(
        self,
        use_text_embeddings: bool = True,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        max_tfidf_features: int = 128,
    ):
        self.use_text_embeddings = use_text_embeddings
        self.embedding_model_name = embedding_model_name
        self.max_tfidf_features = max_tfidf_features

        self.numeric_features = [
            "coverage_drop",
            "failure_frequency",
            "historical_bug_count",
            "avg_fix_time",
            "assertion_failures",
            "severity_score",
            "module_criticality",
            "coverage_freq_interaction",
        ]
        self.categorical_features = [
            "module_name",
            "error_code",
            "assertion_type",
            "regression_suite",
            "severity",
        ]
        self.base_feature_order = self.numeric_features + self.categorical_features

        self.preprocessor: Optional[ColumnTransformer] = None
        self.text_backend: Optional[str] = None
        self.text_model = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.text_feature_names: list[str] = []

    @staticmethod
    def _to_dense(x: np.ndarray) -> np.ndarray:
        return x.toarray() if hasattr(x, "toarray") else x

    def _build_preprocessor(self) -> ColumnTransformer:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

        return ColumnTransformer(
            transformers=[
                ("numeric", StandardScaler(), self.numeric_features),
                ("categorical", encoder, self.categorical_features),
            ],
            remainder="drop",
        )

    def add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["severity"] = out["severity"].fillna("warning").str.lower()
        out["module_name"] = out["module_name"].fillna("UART")
        out["error_code"] = out["error_code"].fillna("E_UNKNOWN")
        out["assertion_type"] = out["assertion_type"].fillna("assert_stable")
        out["regression_suite"] = out["regression_suite"].fillna("nightly_run")
        out["log_message"] = out.get("log_message", "").fillna("")

        out["coverage_drop"] = out["coverage_drop"].astype(float)
        out["failure_frequency"] = out["failure_frequency"].astype(int)
        out["historical_bug_count"] = out["historical_bug_count"].astype(int)
        out["avg_fix_time"] = out["avg_fix_time"].astype(int)
        out["assertion_failures"] = out["assertion_failures"].astype(int)

        out["severity_score"] = out["severity"].map(self.severity_map).fillna(1).astype(int)
        out["module_criticality"] = (
            out["module_name"].map(self.module_criticality_map).fillna(1.0).astype(float)
        )
        out["coverage_freq_interaction"] = out["coverage_drop"] * out["failure_frequency"]
        return out

    def _fit_text_features(self, messages: pd.Series) -> np.ndarray:
        if not self.use_text_embeddings:
            self.text_backend = "disabled"
            self.text_feature_names = []
            return np.zeros((len(messages), 0))

        try:
            from sentence_transformers import SentenceTransformer

            self.text_model = SentenceTransformer(self.embedding_model_name)
            embeddings = self.text_model.encode(
                messages.astype(str).tolist(),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self.text_backend = "sentence_transformers"
            self.text_feature_names = [f"emb_{idx:03d}" for idx in range(embeddings.shape[1])]
            return embeddings
        except Exception:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_tfidf_features, ngram_range=(1, 2)
            )
            tfidf = self.tfidf_vectorizer.fit_transform(messages.astype(str))
            self.text_backend = "tfidf"
            self.text_feature_names = [
                f"tfidf_{term}" for term in self.tfidf_vectorizer.get_feature_names_out()
            ]
            return self._to_dense(tfidf)

    def _transform_text_features(self, messages: pd.Series) -> np.ndarray:
        if self.text_backend == "disabled":
            return np.zeros((len(messages), 0))

        if self.text_backend == "sentence_transformers":
            if self.text_model is None:
                from sentence_transformers import SentenceTransformer

                self.text_model = SentenceTransformer(self.embedding_model_name)
            return self.text_model.encode(
                messages.astype(str).tolist(),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

        if self.text_backend == "tfidf":
            if self.tfidf_vectorizer is None:
                raise RuntimeError("TF-IDF vectorizer is not available.")
            tfidf = self.tfidf_vectorizer.transform(messages.astype(str))
            return self._to_dense(tfidf)

        return np.zeros((len(messages), 0))

    def fit_transform(self, df: pd.DataFrame, target_col: str = "priority_label"):
        out = self.add_engineered_features(df)
        if self.preprocessor is None:
            self.preprocessor = self._build_preprocessor()

        tabular = self.preprocessor.fit_transform(out[self.base_feature_order])
        tabular = self._to_dense(tabular)
        text = self._fit_text_features(out["log_message"])
        x = np.hstack([tabular, text]) if text.shape[1] else tabular
        y = out[target_col].values if target_col in out.columns else None
        return x, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None:
            raise RuntimeError("FeatureEngineeringAgent is not fitted. Call fit_transform first.")

        out = self.add_engineered_features(df)
        tabular = self.preprocessor.transform(out[self.base_feature_order])
        tabular = self._to_dense(tabular)
        text = self._transform_text_features(out["log_message"])
        return np.hstack([tabular, text]) if text.shape[1] else tabular

    def get_feature_names(self) -> list[str]:
        if self.preprocessor is None:
            return []

        tabular_names = self.preprocessor.get_feature_names_out().tolist()
        return tabular_names + self.text_feature_names

    def save(self, output_path: Path = DEFAULT_FEATURE_PIPELINE_PATH) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "use_text_embeddings": self.use_text_embeddings,
            "embedding_model_name": self.embedding_model_name,
            "max_tfidf_features": self.max_tfidf_features,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "base_feature_order": self.base_feature_order,
            "preprocessor": self.preprocessor,
            "text_backend": self.text_backend,
            "tfidf_vectorizer": self.tfidf_vectorizer,
            "text_feature_names": self.text_feature_names,
        }
        joblib.dump(artifact, output_path)
        return output_path

    @classmethod
    def load(cls, artifact_path: Path = DEFAULT_FEATURE_PIPELINE_PATH) -> "FeatureEngineeringAgent":
        artifact = joblib.load(artifact_path)
        agent = cls(
            use_text_embeddings=artifact["use_text_embeddings"],
            embedding_model_name=artifact["embedding_model_name"],
            max_tfidf_features=artifact["max_tfidf_features"],
        )
        agent.numeric_features = artifact["numeric_features"]
        agent.categorical_features = artifact["categorical_features"]
        agent.base_feature_order = artifact["base_feature_order"]
        agent.preprocessor = artifact["preprocessor"]
        agent.text_backend = artifact["text_backend"]
        agent.tfidf_vectorizer = artifact["tfidf_vectorizer"]
        agent.text_feature_names = artifact["text_feature_names"]
        return agent
