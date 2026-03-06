from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .prioritization_model_agent import PrioritizationModelAgent


class ExplanationAgent:
    """Provides SHAP explanations and advanced analytics helpers."""

    def __init__(self, model_agent: PrioritizationModelAgent):
        self.model_agent = model_agent
        self._shap_explainer = None
        self._shap_enabled = None

    def _ensure_shap(self) -> bool:
        if self._shap_enabled is not None:
            return self._shap_enabled

        if self.model_agent.model is None:
            self._shap_enabled = False
            return False

        try:
            import shap

            self._shap_explainer = shap.TreeExplainer(self.model_agent.model)
            self._shap_enabled = True
        except Exception:
            self._shap_enabled = False
        return self._shap_enabled

    def explain_instance(
        self,
        x_row: np.ndarray,
        feature_names: list[str],
        top_k: int = 5,
    ) -> dict[str, Any]:
        x = np.asarray(x_row).reshape(1, -1)
        labels, scores, probabilities = self.model_agent.predict(x)
        predicted_label = labels[0]
        priority_score = float(scores[0])

        response = {
            "predicted_label": predicted_label,
            "priority_score": priority_score,
            "top_factors": [],
            "explanation_backend": "feature_importance_fallback",
        }

        if self._ensure_shap():
            shap_values = self._shap_explainer.shap_values(x)
            class_index = int(np.argmax(probabilities[0]))
            values = self._select_shap_vector(shap_values, class_index=class_index)

            ranked = np.asarray(np.argsort(np.abs(values))[::-1][:top_k]).reshape(-1)
            response["top_factors"] = [
                {
                    "feature": feature_names[int(idx)]
                    if int(idx) < len(feature_names)
                    else f"f_{int(idx)}",
                    "contribution": round(float(values[int(idx)]), 4),
                    "direction": "increase" if float(values[int(idx)]) >= 0 else "decrease",
                }
                for idx in ranked
            ]
            response["explanation_backend"] = "shap"
            return response

        if hasattr(self.model_agent.model, "feature_importances_"):
            importances = self.model_agent.model.feature_importances_
            ranked = np.argsort(importances)[::-1][:top_k]
            response["top_factors"] = [
                {
                    "feature": feature_names[idx] if idx < len(feature_names) else f"f_{idx}",
                    "importance": round(float(importances[idx]), 4),
                    "value": round(float(x[0][idx]), 4),
                }
                for idx in ranked
            ]
        return response

    @staticmethod
    def _select_shap_vector(shap_values: Any, class_index: int) -> np.ndarray:
        """Normalize SHAP outputs to a 1D feature-contribution vector."""
        if isinstance(shap_values, list):
            selected = np.asarray(shap_values[class_index])
            if selected.ndim >= 2:
                return selected[0].reshape(-1)
            return selected.reshape(-1)

        arr = np.asarray(shap_values)
        if arr.ndim == 1:
            return arr.reshape(-1)
        if arr.ndim == 2:
            return arr[0].reshape(-1)
        if arr.ndim == 3:
            # Common multi-class format: (n_samples, n_features, n_classes)
            if arr.shape[2] > class_index:
                return arr[0, :, class_index].reshape(-1)
            # Alternate format: (n_classes, n_samples, n_features)
            if arr.shape[0] > class_index:
                return arr[class_index, 0, :].reshape(-1)

        return arr.reshape(-1)

    def suggest_root_cause(self, record: dict[str, Any]) -> str:
        module = str(record.get("module_name", "")).lower()
        error_code = str(record.get("error_code", "")).upper()
        assertion = str(record.get("assertion_type", "")).lower()
        log_text = str(record.get("log_message", "")).lower()

        if "cache" in module or "COHERENCY" in error_code or "cache" in log_text:
            return "Possible cache coherency issue under invalidation/writeback race."
        if (
            "memory" in module
            or "REFRESH" in error_code
            or "memory" in log_text
            or "stale read" in log_text
            or "page fault" in log_text
        ):
            return "Possible memory timing/refresh window issue causing stale reads."
        if (
            "pcie" in module
            or "LTSSM" in error_code
            or "network" in log_text
            or "nic" in log_text
            or "link down" in log_text
        ):
            return "Possible PCIe link training or protocol state transition instability."
        if "dma" in module or "DMA" in error_code or "disk" in log_text or "io" in log_text:
            return "Possible DMA descriptor alignment/count mismatch in burst transfer path."
        if "uart" in module or "serial" in log_text or "com port" in log_text:
            return "Possible UART framing/parity issue or FIFO overrun under burst traffic."
        if "timeout" in log_text or "hung" in log_text or "deadlock" in log_text:
            return "Possible handshake deadlock or timeout due to missing ready/ack transition."
        if "overflow" in log_text or "underflow" in log_text:
            return "Possible buffer boundary handling issue causing overflow/underflow condition."
        if "permission" in log_text or "access denied" in log_text:
            return "Possible privilege or access-control path mismatch in system integration."
        if assertion == "assert_ordering":
            return "Ordering assertion indicates scoreboard mismatch across pipeline stages."
        if assertion == "assert_handshake":
            return "Handshake assertion indicates valid/ready protocol sequencing issue."
        return "Review waveform around failing assertion and compare expected vs observed state transitions."

    def detect_module_trends(
        self, df: pd.DataFrame, timestamp_col: str = "timestamp", module_col: str = "module_name"
    ) -> list[dict[str, Any]]:
        if df.empty or timestamp_col not in df or module_col not in df:
            return []

        out = df[[timestamp_col, module_col]].dropna().copy()
        out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
        out = out.dropna(subset=[timestamp_col])
        if out.empty:
            return []

        out["week"] = out[timestamp_col].dt.to_period("W").apply(lambda p: p.start_time)
        weekly = out.groupby(["week", module_col]).size().reset_index(name="failures")

        trends = []
        for module_name, group in weekly.groupby(module_col):
            group = group.sort_values("week")
            if len(group) < 2:
                slope = 0.0
            else:
                x = np.arange(len(group))
                y = group["failures"].to_numpy(dtype=float)
                slope = float(np.polyfit(x, y, 1)[0])
            trends.append(
                {
                    "module_name": module_name,
                    "weekly_failure_slope": round(slope, 4),
                    "latest_week_failures": int(group["failures"].iloc[-1]),
                }
            )

        return sorted(trends, key=lambda item: item["weekly_failure_slope"], reverse=True)

    def cluster_failures(self, df: pd.DataFrame, n_clusters: int = 4) -> dict[str, Any]:
        numeric_cols = [
            col
            for col in [
                "coverage_drop",
                "failure_frequency",
                "historical_bug_count",
                "avg_fix_time",
                "assertion_failures",
            ]
            if col in df.columns
        ]
        if not numeric_cols or df.empty:
            return {"cluster_profile": [], "clustered_preview": []}

        matrix = df[numeric_cols].fillna(0.0).to_numpy()
        scaled = StandardScaler().fit_transform(matrix)
        distinct_points = np.unique(np.round(scaled, decimals=8), axis=0)
        distinct_count = int(distinct_points.shape[0])

        # Avoid convergence warnings for tiny/duplicate uploaded samples.
        if distinct_count <= 1:
            out = df.copy()
            out["cluster_id"] = 0
            profile = (
                out.groupby("cluster_id")[numeric_cols]
                .mean()
                .round(3)
                .assign(cluster_size=out.groupby("cluster_id").size())
                .reset_index()
                .to_dict(orient="records")
            )
            preview = out.head(25).to_dict(orient="records")
            return {"cluster_profile": profile, "clustered_preview": preview}

        clusters = min(max(2, n_clusters), len(df), distinct_count)
        kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled)

        out = df.copy()
        out["cluster_id"] = labels
        profile = (
            out.groupby("cluster_id")[numeric_cols]
            .mean()
            .round(3)
            .assign(cluster_size=out.groupby("cluster_id").size())
            .reset_index()
            .to_dict(orient="records")
        )
        preview = out.head(25).to_dict(orient="records")
        return {"cluster_profile": profile, "clustered_preview": preview}

    def git_fix_insights(self, repo_path: Path = Path(".")) -> dict[str, Any]:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_path), "log", "--pretty=format:%ad|%s", "--date=short"],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return {"fix_commits": 0, "recent_fix_messages": []}

        lines = [line for line in result.stdout.splitlines() if line.strip()]
        fix_messages = []
        for line in lines:
            date, _, message = line.partition("|")
            lower_msg = message.lower()
            if "fix" in lower_msg or "bug" in lower_msg or "hotfix" in lower_msg:
                fix_messages.append({"date": date, "message": message})
            if len(fix_messages) >= 15:
                break
        return {"fix_commits": len(fix_messages), "recent_fix_messages": fix_messages}
