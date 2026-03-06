from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATASET_PATH = PROJECT_ROOT / "AI_Debug_Agent" / "dataset" / "rtl_dataset.csv"
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "AI_Debug_Agent" / "dataset" / "rtl_failures.db"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "AI_Debug_Agent" / "models" / "xgboost_model.pkl"
DEFAULT_FEATURE_PIPELINE_PATH = PROJECT_ROOT / "AI_Debug_Agent" / "models" / "feature_pipeline.pkl"

RANDOM_SEED = 42
