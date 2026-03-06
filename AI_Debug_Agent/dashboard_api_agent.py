from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
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
from .feature_engineering_agent import FeatureEngineeringAgent
from .log_parser_agent import LogParserAgent
from .prioritization_model_agent import PrioritizationModelAgent


class TrainRequest(BaseModel):
    row_count: int = Field(default=12000, ge=10000, le=50000)


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
    defaults = {
        "failure_frequency": 10,
        "historical_bug_count": 4,
        "avg_fix_time": 4,
        "assertion_failures": 3,
    }
    record = {**defaults, **parsed}
    df = _prepare_predict_frame(record)
    x = state.feature_agent.transform(df)
    labels, scores, probabilities = state.model_agent.predict(x)
    return {
        "parsed": parsed,
        "predicted_priority_label": labels[0],
        "priority_score": float(scores[0]),
        "class_probabilities": {
            label: round(float(prob), 4)
            for label, prob in zip(state.model_agent.label_encoder.classes_, probabilities[0])
        },
        "root_cause_suggestion": state.explanation_agent.suggest_root_cause(record),
    }


@app.post("/upload-logs")
async def upload_logs(file: UploadFile = File(...)):
    _ensure_model_ready(state)
    raw = (await file.read()).decode("utf-8", errors="ignore")
    logs = [line.strip() for line in raw.splitlines() if line.strip()]
    if not logs:
        raise HTTPException(status_code=400, detail="No logs found in uploaded file.")

    parsed_df = state.parser_agent.parse_logs(logs)
    parsed_df["failure_frequency"] = 10
    parsed_df["historical_bug_count"] = 4
    parsed_df["avg_fix_time"] = 4
    parsed_df["assertion_failures"] = 3

    x = state.feature_agent.transform(parsed_df)
    labels, scores, probabilities = state.model_agent.predict(x)
    results = []
    for idx, row in parsed_df.reset_index(drop=True).iterrows():
        results.append(
            {
                "log_message": row["log_message"],
                "module_name": row["module_name"],
                "severity": row["severity"],
                "predicted_priority_label": labels[idx],
                "priority_score": float(scores[idx]),
                "class_probabilities": {
                    label: round(float(prob), 4)
                    for label, prob in zip(state.model_agent.label_encoder.classes_, probabilities[idx])
                },
                "root_cause_suggestion": state.explanation_agent.suggest_root_cause(row.to_dict()),
            }
        )
    return {"total_logs": len(results), "results": results}


@app.get("/analytics")
def analytics():
    if not DEFAULT_DATASET_PATH.exists():
        raise HTTPException(status_code=404, detail="Dataset not found. Run /train first.")

    df = pd.read_csv(DEFAULT_DATASET_PATH)
    severity_dist = df["severity"].value_counts().to_dict()
    priority_dist = df["priority_label"].value_counts().to_dict()

    coverage_by_module = (
        df.groupby("module_name")["coverage_drop"]
        .mean()
        .round(2)
        .sort_values(ascending=False)
        .reset_index(name="avg_coverage_drop")
        .to_dict(orient="records")
    )

    heatmap = pd.crosstab(df["module_name"], df["severity"])
    trends = state.explanation_agent.detect_module_trends(df)
    clusters = state.explanation_agent.cluster_failures(df.head(2000))
    git_insights = state.explanation_agent.git_fix_insights(Path(__file__).resolve().parent.parent)

    return {
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
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
