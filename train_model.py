from pathlib import Path

import pandas as pd

from AI_Debug_Agent.config import DEFAULT_DATASET_PATH, DEFAULT_FEATURE_PIPELINE_PATH, DEFAULT_MODEL_PATH
from AI_Debug_Agent.data_ingestion_agent import DataIngestionAgent
from AI_Debug_Agent.feature_engineering_agent import FeatureEngineeringAgent
from AI_Debug_Agent.prioritization_model_agent import PrioritizationModelAgent


def _load_or_build_dataset() -> pd.DataFrame:
    source_path = Path("rtl_regression_dataset.csv")
    if source_path.exists():
        df = pd.read_csv(source_path)
    elif DEFAULT_DATASET_PATH.exists():
        df = pd.read_csv(DEFAULT_DATASET_PATH)
    else:
        df = DataIngestionAgent().generate_synthetic_dataset(row_count=12000)

    rename_map = {
        "module": "module_name",
        "frequency": "failure_frequency",
        "recurrence": "historical_bug_count",
        "fix_time": "avg_fix_time",
        "priority": "priority_label",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    defaults = {
        "module_name": "MemoryCtrl",
        "error_code": "E_UNKNOWN",
        "severity": "error",
        "coverage_drop": 0.0,
        "failure_frequency": 1,
        "historical_bug_count": 0,
        "avg_fix_time": 1,
        "assertion_type": "assert_stable",
        "regression_suite": "nightly_run",
        "assertion_failures": 0,
        "log_message": "",
        "priority_label": "Medium",
    }
    for key, value in defaults.items():
        if key not in df.columns:
            df[key] = value
    return df


def main():
    df = _load_or_build_dataset()
    feature_agent = FeatureEngineeringAgent(use_text_embeddings=True)
    model_agent = PrioritizationModelAgent()

    x, y = feature_agent.fit_transform(df, target_col="priority_label")
    metrics = model_agent.train(x, y)
    feature_agent.save(DEFAULT_FEATURE_PIPELINE_PATH)
    model_agent.save(DEFAULT_MODEL_PATH)

    print("Training complete.")
    print(f"Model artifact: {DEFAULT_MODEL_PATH}")
    print(f"Feature pipeline artifact: {DEFAULT_FEATURE_PIPELINE_PATH}")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
