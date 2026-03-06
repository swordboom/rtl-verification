from pathlib import Path

from AI_Debug_Agent.config import DEFAULT_DATASET_PATH, DEFAULT_SQLITE_PATH
from AI_Debug_Agent.data_ingestion_agent import DataIngestionAgent


def main():
    agent = DataIngestionAgent()
    df = agent.build_dataset_and_store(
        row_count=12000,
        dataset_path=DEFAULT_DATASET_PATH,
        db_path=DEFAULT_SQLITE_PATH,
    )

    # Compatibility export for legacy scripts in this repo.
    legacy_path = Path("rtl_regression_dataset.csv")
    df.to_csv(legacy_path, index=False)
    print(f"Dataset generated: {len(df)} rows")
    print(f"Primary CSV: {DEFAULT_DATASET_PATH}")
    print(f"Legacy CSV: {legacy_path.resolve()}")
    print(f"SQLite DB: {DEFAULT_SQLITE_PATH}")


if __name__ == "__main__":
    main()
