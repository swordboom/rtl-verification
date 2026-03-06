from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import DEFAULT_DATASET_PATH, DEFAULT_SQLITE_PATH, RANDOM_SEED

MODULES = ["ALU", "Cache", "MemoryCtrl", "DMA", "UART", "PCIe"]
SEVERITY_LEVELS = ["fatal", "error", "warning"]
ASSERTION_TYPES = [
    "assert_stable",
    "assert_never",
    "assert_handshake",
    "assert_fsm_state",
    "assert_ordering",
]
REGRESSION_SUITES = ["nightly_run", "sanity", "power_aware", "full_chip", "smoke"]

ERROR_CODES: Dict[str, list[str]] = {
    "ALU": ["E_ALU_OVERFLOW", "E_ALU_ZERO_DIV", "E_ALU_OPCODE"],
    "Cache": ["E_CACHE_COHERENCY", "E_CACHE_EVICT", "E_CACHE_TAG"],
    "MemoryCtrl": ["E_MEM_REFRESH", "E_MEM_TIMING", "E_MEM_STALE_READ"],
    "DMA": ["E_DMA_COUNT_MISMATCH", "E_DMA_ALIGN", "E_DMA_TIMEOUT"],
    "UART": ["E_UART_FRAMING", "E_UART_PARITY", "E_UART_FIFO"],
    "PCIe": ["E_PCIE_LINK", "E_PCIE_TLP", "E_PCIE_LTSSM"],
}

SEVERITY_WEIGHTS = {"fatal": 3.0, "error": 2.0, "warning": 1.0}


@dataclass
class DataIngestionConfig:
    row_count: int = 20000
    seed: int = RANDOM_SEED


class DataIngestionAgent:
    """Generates and loads synthetic RTL verification failure datasets."""

    def __init__(self, config: Optional[DataIngestionConfig] = None):
        self.config = config or DataIngestionConfig()
        self._rng = random.Random(self.config.seed)
        self._np_rng = np.random.default_rng(self.config.seed)

    def _sample_severity(self) -> str:
        return self._rng.choices(SEVERITY_LEVELS, weights=[0.16, 0.52, 0.32], k=1)[0]

    def _sample_metrics(self, severity: str) -> dict[str, float]:
        if severity == "fatal":
            return {
                "coverage_drop": round(self._rng.uniform(10.0, 20.0), 2),
                "failure_frequency": self._rng.randint(20, 50),
                "historical_bug_count": self._rng.randint(8, 15),
                "avg_fix_time": self._rng.randint(6, 10),
                "assertion_failures": self._rng.randint(8, 20),
            }
        if severity == "error":
            return {
                "coverage_drop": round(self._rng.uniform(5.0, 15.0), 2),
                "failure_frequency": self._rng.randint(8, 35),
                "historical_bug_count": self._rng.randint(3, 12),
                "avg_fix_time": self._rng.randint(3, 8),
                "assertion_failures": self._rng.randint(3, 15),
            }
        return {
            "coverage_drop": round(self._rng.uniform(0.0, 8.0), 2),
            "failure_frequency": self._rng.randint(1, 18),
            "historical_bug_count": self._rng.randint(0, 8),
            "avg_fix_time": self._rng.randint(1, 5),
            "assertion_failures": self._rng.randint(0, 8),
        }

    def _priority_label(self, severity: str, coverage_drop: float, failure_frequency: int) -> str:
        if severity == "fatal" or coverage_drop >= 14.0 or failure_frequency >= 35:
            return "High"
        if severity == "error" or coverage_drop >= 7.0 or failure_frequency >= 15:
            return "Medium"
        return "Low"

    def _priority_score(self, row: dict[str, float]) -> int:
        score = (
            row["coverage_drop"] * 2.8
            + row["failure_frequency"] * 1.1
            + row["historical_bug_count"] * 2.0
            + row["avg_fix_time"] * 3.2
            + row["assertion_failures"] * 1.4
            + SEVERITY_WEIGHTS[row["severity"]] * 10.0
        )
        return int(np.clip(round(score), 0, 100))

    def _timestamp(self) -> str:
        now = datetime.utcnow()
        delta = timedelta(
            days=self._rng.randint(0, 90),
            hours=self._rng.randint(0, 23),
            minutes=self._rng.randint(0, 59),
            seconds=self._rng.randint(0, 59),
        )
        return (now - delta).strftime("%Y-%m-%d %H:%M:%S")

    def _build_log_message(
        self,
        severity: str,
        module: str,
        error_code: str,
        assertion_type: str,
        coverage_drop: float,
        regression_suite: str,
    ) -> str:
        line_no = self._rng.randint(60, 900)
        return (
            f"[{severity.upper()}] Module: {module} {assertion_type} triggered with {error_code} at line {line_no}. "
            f"Coverage drop detected: {coverage_drop}%. Regression: {regression_suite}"
        )

    def generate_synthetic_dataset(self, row_count: Optional[int] = None) -> pd.DataFrame:
        target_rows = row_count or self.config.row_count
        records = []
        for idx in range(target_rows):
            module = self._rng.choice(MODULES)
            severity = self._sample_severity()
            metrics = self._sample_metrics(severity)
            error_code = self._rng.choice(ERROR_CODES[module])
            assertion_type = self._rng.choice(ASSERTION_TYPES)
            regression_suite = self._rng.choice(REGRESSION_SUITES)

            record = {
                "failure_id": f"FAIL_{idx + 1:05d}",
                "timestamp": self._timestamp(),
                "module_name": module,
                "error_code": error_code,
                "severity": severity,
                **metrics,
                "assertion_type": assertion_type,
                "regression_suite": regression_suite,
            }
            record["log_message"] = self._build_log_message(
                severity=severity,
                module=module,
                error_code=error_code,
                assertion_type=assertion_type,
                coverage_drop=record["coverage_drop"],
                regression_suite=regression_suite,
            )
            record["priority_label"] = self._priority_label(
                severity, record["coverage_drop"], record["failure_frequency"]
            )
            record["priority_score"] = self._priority_score(record)
            records.append(record)

        return pd.DataFrame(records)

    def save_dataset(self, df: pd.DataFrame, output_path: Path = DEFAULT_DATASET_PATH) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return output_path

    def load_dataset(self, dataset_path: Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
        return pd.read_csv(dataset_path)

    def ingest_to_sqlite(
        self,
        df: pd.DataFrame,
        db_path: Path = DEFAULT_SQLITE_PATH,
        table_name: str = "rtl_failures",
        if_exists: str = "replace",
    ) -> Path:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        return db_path

    def build_dataset_and_store(
        self,
        row_count: Optional[int] = None,
        dataset_path: Path = DEFAULT_DATASET_PATH,
        db_path: Path = DEFAULT_SQLITE_PATH,
    ) -> pd.DataFrame:
        df = self.generate_synthetic_dataset(row_count=row_count)
        self.save_dataset(df, dataset_path)
        self.ingest_to_sqlite(df, db_path=db_path)
        return df
