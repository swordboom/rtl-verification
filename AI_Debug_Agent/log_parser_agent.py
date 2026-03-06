from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import pandas as pd

SEVERITY_PATTERN = re.compile(r"\[(FATAL|ERROR|WARNING)\]", re.IGNORECASE)
MODULE_PATTERN = re.compile(r"Module:\s*([A-Za-z0-9_]+)", re.IGNORECASE)
ERROR_CODE_PATTERN = re.compile(r"(E_[A-Z0-9_]+)")
COVERAGE_PATTERN = re.compile(r"Coverage drop detected:\s*([0-9]+(?:\.[0-9]+)?)%", re.IGNORECASE)
REGRESSION_PATTERN = re.compile(r"Regression:\s*([A-Za-z0-9_\-]+)", re.IGNORECASE)
LINE_PATTERN = re.compile(r"line\s*([0-9]+)", re.IGNORECASE)

ASSERTION_TOKENS = [
    "assert_stable",
    "assert_never",
    "assert_handshake",
    "assert_fsm_state",
    "assert_ordering",
]


class LogParserAgent:
    """Parses semi-structured RTL verification logs into structured fields."""

    def parse_log(self, log: str) -> dict[str, Any]:
        severity_match = SEVERITY_PATTERN.search(log)
        module_match = MODULE_PATTERN.search(log)
        error_code_match = ERROR_CODE_PATTERN.search(log)
        coverage_match = COVERAGE_PATTERN.search(log)
        regression_match = REGRESSION_PATTERN.search(log)
        line_match = LINE_PATTERN.search(log)

        severity = (
            severity_match.group(1).lower()
            if severity_match
            else ("error" if "error" in log.lower() else "warning")
        )

        assertion_type = "assert_stable"
        for token in ASSERTION_TOKENS:
            if token in log.lower():
                assertion_type = token
                break

        return {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "module_name": module_match.group(1) if module_match else "MemoryCtrl",
            "error_code": error_code_match.group(1) if error_code_match else "E_UNKNOWN",
            "severity": severity,
            "coverage_drop": float(coverage_match.group(1)) if coverage_match else 0.0,
            "assertion_type": assertion_type,
            "regression_suite": regression_match.group(1) if regression_match else "nightly_run",
            "line_no": int(line_match.group(1)) if line_match else -1,
            "log_message": log.strip(),
        }

    def parse_logs(self, logs: list[str]) -> pd.DataFrame:
        records = [self.parse_log(log) for log in logs if log and log.strip()]
        return pd.DataFrame(records)
