from __future__ import annotations

import os
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_FEATURE_PIPELINE_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SQLITE_PATH,
)
from .data_ingestion_agent import DataIngestionAgent
from .explanation_agent import ExplanationAgent
from .evaluation_report import generate_evaluation_report
from .feature_engineering_agent import FeatureEngineeringAgent
from .log_parser_agent import LogParserAgent
from .prioritization_model_agent import PrioritizationModelAgent

# Avoid loky CPU-core detection warning on Windows systems without `wmic`.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")


class TrainRequest(BaseModel):
    row_count: int = Field(default=12000, ge=10000, le=50000)
    speed_profile: str = Field(default="fast")


class ParseLogRequest(BaseModel):
    log: str


class PredictRequest(BaseModel):
    timestamp: Optional[str] = None
    module_name: str = "MemoryCtrl"
    error_code: str = "E_UNKNOWN"
    severity: str = "error"
    coverage_drop: float = 5.0
    failure_frequency: int = 12
    historical_bug_count: int = 5
    avg_fix_time: int = 4
    assertion_type: str = "assert_stable"
    regression_suite: str = "nightly_run"
    assertion_failures: int = 3
    log_message: str = ""


@dataclass
class AgentState:
    ingestion_agent: DataIngestionAgent
    parser_agent: LogParserAgent
    feature_agent: FeatureEngineeringAgent
    model_agent: PrioritizationModelAgent
    explanation_agent: ExplanationAgent
    uploaded_logs_df: Optional[pd.DataFrame] = None


def _build_state() -> AgentState:
    model_agent = PrioritizationModelAgent()
    feature_agent = FeatureEngineeringAgent(use_text_embeddings=True)
    return AgentState(
        ingestion_agent=DataIngestionAgent(),
        parser_agent=LogParserAgent(),
        feature_agent=feature_agent,
        model_agent=model_agent,
        explanation_agent=ExplanationAgent(model_agent),
    )


def _prepare_predict_frame(record: dict[str, Any]) -> pd.DataFrame:
    if not record.get("timestamp"):
        record["timestamp"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    if not record.get("log_message"):
        record["log_message"] = (
            f"[{record['severity'].upper()}] Module: {record['module_name']} "
            f"{record['assertion_type']} triggered with {record['error_code']}."
        )
    return pd.DataFrame([record])


def _payload_to_dict(payload: BaseModel) -> dict[str, Any]:
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    return payload.dict()


def _ensure_model_ready(state: AgentState):
    if state.model_agent.model is None or state.feature_agent.preprocessor is None:
        raise HTTPException(status_code=400, detail="Model is not trained. Call /train first.")


def _ensure_required_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    defaults = {
        "timestamp": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "module_name": "MemoryCtrl",
        "error_code": "E_UNKNOWN",
        "severity": "error",
        "coverage_drop": 0.0,
        "assertion_type": "assert_stable",
        "regression_suite": "nightly_run",
        "failure_frequency": 10,
        "historical_bug_count": 4,
        "avg_fix_time": 4,
        "assertion_failures": 3,
        "log_message": "",
    }
    for key, value in defaults.items():
        if key not in out.columns:
            out[key] = value
    return out


def _populate_runtime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    severity_weight = {"fatal": 1.0, "error": 0.65, "warning": 0.35}

    if "log_message" not in out.columns:
        out["log_message"] = ""

    signatures = (
        out["module_name"].astype(str)
        + "|"
        + out["error_code"].astype(str)
        + "|"
        + out["assertion_type"].astype(str)
    )
    signature_count = signatures.value_counts()
    msg_lengths = out["log_message"].fillna("").str.len().to_numpy(dtype=float)
    word_counts = out["log_message"].fillna("").str.split().str.len().fillna(0).to_numpy(dtype=float)
    numeric_counts = out["log_message"].fillna("").str.count(r"\d+").to_numpy(dtype=float)
    severity_factor = (
        out["severity"].astype(str).str.lower().map(severity_weight).fillna(0.4).to_numpy(dtype=float)
    )

    inferred_freq = np.clip(np.round(4 + severity_factor * 18 + np.log1p(word_counts) * 6), 1, 50).astype(int)
    inferred_hist = np.clip(
        np.round(1 + severity_factor * 8 + np.minimum(numeric_counts, 8) * 0.7),
        0,
        15,
    ).astype(int)
    inferred_fix = np.clip(np.round(1 + severity_factor * 6 + np.log1p(msg_lengths) * 0.4), 1, 10).astype(int)
    inferred_assertions = np.clip(
        np.round(1 + severity_factor * 8 + np.minimum(word_counts, 40) * 0.15),
        0,
        20,
    ).astype(int)

    out["failure_frequency"] = inferred_freq
    out["historical_bug_count"] = inferred_hist
    out["avg_fix_time"] = inferred_fix
    out["assertion_failures"] = inferred_assertions

    out["failure_frequency"] = out["failure_frequency"] + signatures.map(signature_count).fillna(1).astype(int) - 1
    out["failure_frequency"] = out["failure_frequency"].clip(1, 50)

    if "coverage_drop" in out.columns:
        inferred_cov = np.clip(np.round(severity_factor * 10 + np.log1p(msg_lengths) * 0.8, 2), 0, 20)
        out["coverage_drop"] = np.where(out["coverage_drop"].astype(float) <= 0.0, inferred_cov, out["coverage_drop"])

    return out


def _analytics_payload(df: pd.DataFrame, module_filter: Optional[str] = None) -> dict[str, Any]:
    source_df = df.copy()
    if module_filter:
        source_df = source_df[source_df["module_name"].astype(str) == module_filter]
        if source_df.empty:
            return {
                "rows": 0,
                "columns": df.columns.tolist(),
                "module_filter": module_filter,
                "severity_distribution": {},
                "priority_distribution": {},
                "coverage_by_module": [],
                "module_severity_heatmap": {"modules": [], "severities": [], "values": []},
                "trend_detection": [],
                "cluster_profile": [],
                "git_fix_insights": state.explanation_agent.git_fix_insights(
                    Path(__file__).resolve().parent.parent
                ),
            }

    severity_dist = source_df["severity"].value_counts().to_dict()
    priority_dist = (
        source_df["priority_label"].value_counts().to_dict()
        if "priority_label" in source_df.columns
        else {}
    )
    coverage_by_module = (
        source_df.groupby("module_name")["coverage_drop"]
        .mean()
        .round(2)
        .sort_values(ascending=False)
        .reset_index(name="avg_coverage_drop")
        .to_dict(orient="records")
    )
    heatmap = pd.crosstab(source_df["module_name"], source_df["severity"])
    trends = state.explanation_agent.detect_module_trends(source_df)
    clusters = state.explanation_agent.cluster_failures(source_df.head(2000))
    git_insights = state.explanation_agent.git_fix_insights(Path(__file__).resolve().parent.parent)
    return {
        "rows": int(len(source_df)),
        "columns": source_df.columns.tolist(),
        "module_filter": module_filter,
        "severity_distribution": severity_dist,
        "priority_distribution": priority_dist,
        "coverage_by_module": coverage_by_module,
        "module_severity_heatmap": {
            "modules": heatmap.index.tolist(),
            "severities": heatmap.columns.tolist(),
            "values": heatmap.values.tolist(),
        },
        "trend_detection": trends,
        "cluster_profile": clusters["cluster_profile"],
        "git_fix_insights": git_insights,
    }


def _calibrate_priority_score(model_score: float, row: dict[str, Any], class_probabilities: dict[str, float]) -> float:
    coverage = float(row.get("coverage_drop", 0.0))
    frequency = float(row.get("failure_frequency", 0.0))
    historical = float(row.get("historical_bug_count", 0.0))
    fix_time = float(row.get("avg_fix_time", 0.0))
    assertions = float(row.get("assertion_failures", 0.0))
    severity = str(row.get("severity", "warning")).lower()

    impact = (
        (coverage / 20.0) * 0.30
        + (frequency / 50.0) * 0.25
        + (historical / 15.0) * 0.20
        + (fix_time / 10.0) * 0.15
        + (assertions / 20.0) * 0.10
    ) * 100.0
    confidence = (max(class_probabilities.values()) if class_probabilities else 0.0) * 100.0
    severity_adjust = {"fatal": 8.0, "error": 0.0, "warning": -6.0}.get(severity, 0.0)

    score = (0.60 * float(model_score)) + (0.30 * impact) + (0.10 * confidence) + severity_adjust
    return round(float(np.clip(score, 0.0, 100.0)), 2)


def _runtime_component_status() -> dict[str, Any]:
    def module_available(name: str) -> bool:
        return importlib.util.find_spec(name) is not None

    status = {
        "xgboost_installed": module_available("xgboost"),
        "shap_installed": module_available("shap"),
        "sentence_transformers_installed": module_available("sentence_transformers"),
        "model_loaded": state.model_agent.model is not None,
        "model_kind": state.model_agent.model_kind or "uninitialized",
        "feature_pipeline_loaded": state.feature_agent.preprocessor is not None,
        "embedding_backend": state.feature_agent.text_backend or "unknown",
    }

    # Optional deep checks.
    try:
        if status["shap_installed"] and status["model_loaded"]:
            status["shap_ready"] = bool(state.explanation_agent._ensure_shap())
        else:
            status["shap_ready"] = False
    except Exception:
        status["shap_ready"] = False

    try:
        if status["sentence_transformers_installed"]:
            from sentence_transformers import SentenceTransformer

            probe = SentenceTransformer("all-MiniLM-L6-v2")
            emb = probe.encode(["health check"], convert_to_numpy=True, show_progress_bar=False)
            status["minilm_model_loaded"] = bool(getattr(emb, "shape", (0,))[0] == 1)
            status["minilm_embedding_dim"] = int(emb.shape[1]) if len(emb.shape) == 2 else None
        else:
            status["minilm_model_loaded"] = False
            status["minilm_embedding_dim"] = None
    except Exception as exc:
        status["minilm_model_loaded"] = False
        status["minilm_embedding_dim"] = None
        status["minilm_error"] = str(exc)

    return status


def _predict_in_batches(x: np.ndarray, batch_size: int = 8192):
    labels_all = []
    scores_all = []
    probs_all = []
    total = len(x)
    if total == 0:
        return np.array([]), np.array([]), np.zeros((0, 0))

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        labels, scores, probs = state.model_agent.predict(x[start:end])
        labels_all.extend(labels.tolist() if hasattr(labels, "tolist") else list(labels))
        scores_all.extend(scores.tolist() if hasattr(scores, "tolist") else list(scores))
        probs_all.append(probs)

    probabilities = np.vstack(probs_all) if probs_all else np.zeros((0, 0))
    return np.array(labels_all), np.array(scores_all), probabilities


state = _build_state()
app = FastAPI(title="AI Debug Prioritization Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dashboard_dir = Path(__file__).resolve().parent / "dashboard"
app.mount("/dashboard", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")


@app.on_event("startup")
def load_artifacts_on_startup():
    if DEFAULT_MODEL_PATH.exists() and DEFAULT_FEATURE_PIPELINE_PATH.exists():
        state.model_agent = PrioritizationModelAgent.load(DEFAULT_MODEL_PATH)
        state.feature_agent = FeatureEngineeringAgent.load(DEFAULT_FEATURE_PIPELINE_PATH)
        state.explanation_agent = ExplanationAgent(state.model_agent)


@app.get("/")
def home_redirect():
    return RedirectResponse(url="/dashboard/index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_ready": bool(
            state.model_agent.model is not None and state.feature_agent.preprocessor is not None
        ),
        "dataset_exists": DEFAULT_DATASET_PATH.exists(),
        "db_exists": DEFAULT_SQLITE_PATH.exists(),
    }


@app.post("/train")
def train(request: TrainRequest):
    if request.speed_profile not in {"fast", "balanced"}:
        raise HTTPException(status_code=400, detail="speed_profile must be fast or balanced")

    if request.speed_profile == "fast":
        state.feature_agent = FeatureEngineeringAgent(
            use_text_embeddings=True,
            max_tfidf_features=256,
            large_input_threshold=4000,
            embedding_batch_size=512,
        )
    else:
        state.feature_agent = FeatureEngineeringAgent(
            use_text_embeddings=True,
            max_tfidf_features=256,
            large_input_threshold=12000,
            embedding_batch_size=256,
        )

    df = state.ingestion_agent.build_dataset_and_store(
        row_count=request.row_count,
        dataset_path=DEFAULT_DATASET_PATH,
        db_path=DEFAULT_SQLITE_PATH,
    )
    x, y = state.feature_agent.fit_transform(df, target_col="priority_label")
    metrics = state.model_agent.train(x, y)
    state.feature_agent.save(DEFAULT_FEATURE_PIPELINE_PATH)
    state.model_agent.save(DEFAULT_MODEL_PATH)
    state.explanation_agent = ExplanationAgent(state.model_agent)
    return {
        "rows": len(df),
        "speed_profile": request.speed_profile,
        "text_backend": state.feature_agent.text_backend,
        "dataset_path": str(DEFAULT_DATASET_PATH),
        "sqlite_path": str(DEFAULT_SQLITE_PATH),
        "model_path": str(DEFAULT_MODEL_PATH),
        "metrics": metrics,
    }


@app.post("/parse-log")
def parse_log(payload: ParseLogRequest):
    return state.parser_agent.parse_log(payload.log)


@app.post("/predict")
def predict(payload: PredictRequest):
    _ensure_model_ready(state)
    record = _payload_to_dict(payload)
    df = _prepare_predict_frame(record)
    x = state.feature_agent.transform(df)
    labels, scores, probabilities = state.model_agent.predict(x)
    explanation = state.explanation_agent.explain_instance(
        x[0], state.feature_agent.get_feature_names(), top_k=6
    )
    return {
        "predicted_priority_label": labels[0],
        "priority_score": float(scores[0]),
        "class_probabilities": {
            label: round(float(prob), 4)
            for label, prob in zip(state.model_agent.label_encoder.classes_, probabilities[0])
        },
        "root_cause_suggestion": state.explanation_agent.suggest_root_cause(record),
        "explanation": explanation,
    }


@app.post("/predict-from-log")
def predict_from_log(payload: ParseLogRequest):
    _ensure_model_ready(state)
    parsed = state.parser_agent.parse_log(payload.log)
    df = _prepare_predict_frame(parsed)
    df = _ensure_required_prediction_columns(df)
    df = _populate_runtime_features(df)
    record = df.iloc[0].to_dict()
    x = state.feature_agent.transform(df)
    labels, scores, probabilities = state.model_agent.predict(x)
    class_probs = {
        label: round(float(prob), 4)
        for label, prob in zip(state.model_agent.label_encoder.classes_, probabilities[0])
    }
    calibrated_score = _calibrate_priority_score(float(scores[0]), record, class_probs)

    enriched = df.copy()
    enriched["priority_label"] = labels
    enriched["priority_score"] = calibrated_score
    state.uploaded_logs_df = (
        enriched
        if state.uploaded_logs_df is None
        else pd.concat([state.uploaded_logs_df, enriched], ignore_index=True).tail(2000)
    )

    return {
        "parsed": parsed,
        "predicted_priority_label": labels[0],
        "priority_score": calibrated_score,
        "class_probabilities": class_probs,
        "root_cause_suggestion": state.explanation_agent.suggest_root_cause(record),
        "analytics_source": "uploaded",
    }


@app.post("/upload-logs")
async def upload_logs(file: UploadFile = File(...)):
    _ensure_model_ready(state)
    raw = (await file.read()).decode("utf-8", errors="ignore")
    logs = [line.strip() for line in raw.splitlines() if line.strip()]
    if not logs:
        raise HTTPException(status_code=400, detail="No logs found in uploaded file.")

    parsed_df = state.parser_agent.parse_logs(logs, remove_duplicates=True)
    duplicate_count = max(0, len(logs) - len(parsed_df))
    if parsed_df.empty:
        raise HTTPException(
            status_code=400,
            detail="No unique/valid logs found after preprocessing.",
        )
    parsed_df = _ensure_required_prediction_columns(parsed_df)
    parsed_df["test_name"] = parsed_df.get("test_name").fillna(parsed_df["regression_suite"])
    parsed_df = _populate_runtime_features(parsed_df)

    x = state.feature_agent.transform(parsed_df)
    labels, scores, probabilities = _predict_in_batches(x, batch_size=8192)
    enriched_df = parsed_df.copy()
    enriched_df["priority_label"] = labels
    calibrated_scores = []
    for idx in range(len(enriched_df)):
        probs = {
            label: float(prob)
            for label, prob in zip(state.model_agent.label_encoder.classes_, probabilities[idx])
        }
        calibrated_scores.append(
            _calibrate_priority_score(float(scores[idx]), enriched_df.iloc[idx].to_dict(), probs)
        )
    enriched_df["priority_score"] = calibrated_scores
    state.uploaded_logs_df = enriched_df.sort_values("priority_score", ascending=False).copy()

    unique_failure_cols = ["module_name", "error_code", "severity", "assertion_type", "error_category"]
    unique_failures = (
        enriched_df.groupby(unique_failure_cols, dropna=False)
        .size()
        .reset_index(name="occurrences")
        .sort_values("occurrences", ascending=False)
        .to_dict(orient="records")
    )

    categorized_summary = (
        enriched_df.groupby("error_category")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .to_dict(orient="records")
    )

    impact_analysis = (
        enriched_df.groupby(["module_name", "priority_label"], dropna=False)
        .agg(
            tests_affected=("test_name", "nunique"),
            avg_priority_score=("priority_score", "mean"),
            total_failures=("module_name", "size"),
        )
        .reset_index()
        .sort_values(["avg_priority_score", "tests_affected"], ascending=[False, False])
    )
    impact_analysis["avg_priority_score"] = impact_analysis["avg_priority_score"].round(2)

    git_changes = state.explanation_agent.git_fix_insights(Path(__file__).resolve().parent.parent)
    recent_messages = git_changes.get("recent_fix_messages", [])
    module_names = {str(m).lower() for m in enriched_df["module_name"].dropna().unique().tolist()}
    related_recent_changes = []
    for item in recent_messages:
        msg = item.get("message", "")
        msg_lower = msg.lower()
        if any(module in msg_lower for module in module_names):
            related_recent_changes.append(item)
        if len(related_recent_changes) >= 8:
            break

    uploaded_records = []
    for idx, row in parsed_df.reset_index(drop=True).iterrows():
        parsed_record = row.to_dict()
        class_probs = {
            label: round(float(prob), 4)
            for label, prob in zip(state.model_agent.label_encoder.classes_, probabilities[idx])
        }
        calibrated = float(calibrated_scores[idx])
        prediction = {
            "predicted_priority_label": labels[idx],
            "priority_score": calibrated,
            "class_probabilities": class_probs,
            "root_cause_suggestion": state.explanation_agent.suggest_root_cause(parsed_record),
        }
        uploaded_records.append(
            {
                "index": idx + 1,
                "raw_log": logs[idx] if idx < len(logs) else parsed_record.get("log_message", ""),
                "parsed": parsed_record,
                "prediction": prediction,
            }
        )
    uploaded_records = sorted(
        uploaded_records,
        key=lambda item: float(item["prediction"]["priority_score"]),
        reverse=True,
    )

    top_debug_start_points = []
    for record in unique_failures[:6]:
        top_debug_start_points.append(
            {
                "module_name": record.get("module_name"),
                "error_code": record.get("error_code"),
                "severity": record.get("severity"),
                "occurrences": int(record.get("occurrences", 0)),
                "suggested_start": state.explanation_agent.suggest_root_cause(record),
            }
        )

    return {
        "total_logs": len(uploaded_records),
        "preprocessing_summary": {
            "raw_log_lines": len(logs),
            "unique_logs_after_dedup": int(len(parsed_df)),
            "duplicates_removed": int(duplicate_count),
        },
        "uploaded_records": uploaded_records,
        "prioritized_failure_list": sorted(
            [
                {
                    "module_name": row["module_name"],
                    "error_code": row["error_code"],
                    "severity": row["severity"],
                    "priority_label": row["priority_label"],
                    "priority_score": round(float(row["priority_score"]), 2),
                    "error_category": row.get("error_category"),
                }
                for _, row in state.uploaded_logs_df.iterrows()
            ],
            key=lambda item: item["priority_score"],
            reverse=True,
        ),
        "categorized_error_summary": categorized_summary,
        "unique_failures": unique_failures,
        "impact_analysis": impact_analysis.to_dict(orient="records"),
        "suggested_starting_points": top_debug_start_points,
        "related_recent_changes": related_recent_changes,
        "analytics_source": "uploaded",
    }


@app.get("/analytics")
def analytics(
    source: str = Query(default="auto"),
    module_name: Optional[str] = Query(default=None),
):
    if source not in {"auto", "uploaded", "dataset"}:
        raise HTTPException(status_code=400, detail="source must be one of: auto, uploaded, dataset")
    if source in ("auto", "uploaded") and state.uploaded_logs_df is not None:
        payload = _analytics_payload(state.uploaded_logs_df, module_filter=module_name)
        payload["source"] = "uploaded"
        return payload

    if source == "uploaded" and state.uploaded_logs_df is None:
        raise HTTPException(status_code=404, detail="No uploaded log analytics available yet.")

    if not DEFAULT_DATASET_PATH.exists():
        raise HTTPException(status_code=404, detail="Dataset not found. Run /train first.")
    df = pd.read_csv(DEFAULT_DATASET_PATH)
    payload = _analytics_payload(df, module_filter=module_name)
    payload["source"] = "dataset"
    return payload


@app.get("/demo-scenario")
def demo_scenario():
    return {
        "manual_workflow": {
            "steps": [
                "Engineer reads logs",
                "Engineer identifies likely root cause",
                "Engineer assigns priority manually",
            ],
            "time_minutes": 30,
        },
        "ai_workflow": {
            "steps": [
                "Upload logs",
                "AI parser and model analyze failures",
                "Priority ranking with explanation generated",
            ],
            "time_seconds": 10,
        },
    }


@app.get("/runtime-status")
def runtime_status():
    return _runtime_component_status()


@app.get("/evaluation-report")
def evaluation_report(rows: int = Query(default=12000, ge=1000, le=50000)):
    report = generate_evaluation_report(rows=rows)
    return report
