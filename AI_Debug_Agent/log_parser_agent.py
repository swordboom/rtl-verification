from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import pandas as pd

SEVERITY_PATTERN = re.compile(r"\[(FATAL|ERROR|WARNING)\]", re.IGNORECASE)
MODULE_PATTERN = re.compile(r"Module:\s*([A-Za-z0-9_]+)", re.IGNORECASE)
ERROR_CODE_PATTERN = re.compile(r"(E_[A-Z0-9_]+)")
COVERAGE_PATTERN = re.compile(r"Coverage drop detected:\s*([0-9]+(?:\.[0-9]+)?)%", re.IGNORECASE)
PERCENT_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)%")
REGRESSION_PATTERN = re.compile(r"Regression:\s*([A-Za-z0-9_\-]+)", re.IGNORECASE)
LINE_PATTERN = re.compile(r"line\s*([0-9]+)", re.IGNORECASE)
TEST_PATTERN = re.compile(r"Test:\s*([A-Za-z0-9_\-./]+)", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]*")

ASSERTION_TOKENS = [
    "assert_stable",
    "assert_never",
    "assert_handshake",
    "assert_fsm_state",
    "assert_ordering",
]

MODULE_KEYWORDS = {
    "memoryctrl": "MemoryCtrl",
    "memory": "MemoryCtrl",
    "dram": "MemoryCtrl",
    "cache": "Cache",
    "pcie": "PCIe",
    "uart": "UART",
    "dma": "DMA",
    "alu": "ALU",
    "kernel-power": "MemoryCtrl",
    "disk": "DMA",
    "network": "PCIe",
}

SEVERITY_KEYWORDS = {
    "fatal": "fatal",
    "critical": "fatal",
    "panic": "fatal",
    "error": "error",
    "failed": "error",
    "warning": "warning",
    "warn": "warning",
}


class LogParserAgent:
    """Parses semi-structured RTL verification logs into structured fields."""

    @staticmethod
    def _infer_module(lower_log: str) -> str:
        for keyword, module in MODULE_KEYWORDS.items():
            if keyword in lower_log:
                return module
        return "ALU"

    @staticmethod
    def _infer_severity(lower_log: str) -> str:
        for keyword, severity in SEVERITY_KEYWORDS.items():
            if keyword in lower_log:
                return severity
        return "warning"

    @staticmethod
    def tokenize_log(log: str) -> list[str]:
        return [token.lower() for token in TOKEN_PATTERN.findall(log)]

    @staticmethod
    def _normalized_log_signature(log: str) -> str:
        tokens = LogParserAgent.tokenize_log(log)
        return " ".join(tokens)

    def parse_log(self, log: str) -> dict[str, Any]:
        severity_match = SEVERITY_PATTERN.search(log)
        module_match = MODULE_PATTERN.search(log)
        error_code_match = ERROR_CODE_PATTERN.search(log)
        coverage_match = COVERAGE_PATTERN.search(log)
        percent_match = PERCENT_PATTERN.search(log)
        regression_match = REGRESSION_PATTERN.search(log)
        line_match = LINE_PATTERN.search(log)
        test_match = TEST_PATTERN.search(log)
        lower_log = log.lower()

        severity = (
            severity_match.group(1).lower()
            if severity_match
            else self._infer_severity(lower_log)
        )

        assertion_type = "assert_stable"
        for token in ASSERTION_TOKENS:
            if token in lower_log:
                assertion_type = token
                break

        if "uvm" in lower_log:
            error_category = "UVM"
        elif "sva" in lower_log or "assert" in lower_log:
            error_category = "SVA"
        else:
            error_category = "MODULE"

        module_name = module_match.group(1) if module_match else self._infer_module(lower_log)
        coverage_drop = 0.0
        if coverage_match:
            coverage_drop = float(coverage_match.group(1))
        elif percent_match:
            coverage_drop = float(percent_match.group(1))

        tokens = self.tokenize_log(log)
        return {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "module_name": module_name,
            "error_code": error_code_match.group(1) if error_code_match else "E_UNKNOWN",
            "severity": severity,
            "coverage_drop": coverage_drop,
            "assertion_type": assertion_type,
            "regression_suite": regression_match.group(1) if regression_match else "nightly_run",
            "test_name": test_match.group(1) if test_match else None,
            "error_category": error_category,
            "line_no": int(line_match.group(1)) if line_match else -1,
            "token_count": len(tokens),
            "tokens": tokens,
            "normalized_signature": self._normalized_log_signature(log),
            "log_message": log.strip(),
        }

    def parse_logs(self, logs: list[str], remove_duplicates: bool = True) -> pd.DataFrame:
        seen_signatures = set()
        records = []
        for log in logs:
            if not log or not log.strip():
                continue
            parsed = self.parse_log(log)
            if remove_duplicates:
                sig = parsed.get("normalized_signature", "")
                if sig in seen_signatures:
                    continue
                seen_signatures.add(sig)
            records.append(parsed)
        return pd.DataFrame(records)
