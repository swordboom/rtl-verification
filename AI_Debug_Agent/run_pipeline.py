from __future__ import annotations

import argparse

from .config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_FEATURE_PIPELINE_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SQLITE_PATH,
)
from .data_ingestion_agent import DataIngestionAgent
from .explanation_agent import ExplanationAgent
from .feature_engineering_agent import FeatureEngineeringAgent
from .prioritization_model_agent import PrioritizationModelAgent


def run(rows: int = 12000):
    ingestion_agent = DataIngestionAgent()
    feature_agent = FeatureEngineeringAgent(use_text_embeddings=True)
    model_agent = PrioritizationModelAgent()

    df = ingestion_agent.build_dataset_and_store(
        row_count=rows,
        dataset_path=DEFAULT_DATASET_PATH,
        db_path=DEFAULT_SQLITE_PATH,
    )
    x, y = feature_agent.fit_transform(df, target_col="priority_label")
    metrics = model_agent.train(x, y)

    feature_agent.save(DEFAULT_FEATURE_PIPELINE_PATH)
    model_agent.save(DEFAULT_MODEL_PATH)

    explanation_agent = ExplanationAgent(model_agent)
    sample_explanation = explanation_agent.explain_instance(
        x_row=x[0],
        feature_names=feature_agent.get_feature_names(),
        top_k=5,
    )

    return {
        "rows": len(df),
        "dataset_path": str(DEFAULT_DATASET_PATH),
        "sqlite_path": str(DEFAULT_SQLITE_PATH),
        "model_path": str(DEFAULT_MODEL_PATH),
        "feature_pipeline_path": str(DEFAULT_FEATURE_PIPELINE_PATH),
        "metrics": metrics,
        "sample_explanation": sample_explanation,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full AI Debug Agent training pipeline.")
    parser.add_argument("--rows", type=int, default=12000, help="Number of synthetic rows to generate.")
    args = parser.parse_args()
    output = run(rows=args.rows)
    for key, value in output.items():
        print(f"{key}: {value}")
