# RTL Failure Dataset Feature Description

## Overview

This dataset contains synthetic RTL verification failure records for hardware regression analysis, bug prioritization, and severity modeling. Each row represents one failure event observed during a verification run.

## Dataset Size

- Total rows: 12,000
- File format: CSV
- Primary deliverables: `schema.json`, `feature_description.md`, `rtl_failure_dataset.csv`

## Column Definitions

### failure_id

Unique failure identifier in the format `FAIL_00001`.

### module

Hardware block associated with the failure.

Allowed values:
- alu
- cache_controller
- memory_controller
- uart
- pcie
- dram_interface
- dma
- branch_predictor
- pipeline_controller
- interrupt_controller

### severity

Verification triage class for the failure.

Allowed values:
- fatal
- error
- warning

### coverage_drop

Integer value representing how much verification coverage was lost because of the failure.

Range: 0 to 15

### frequency

Integer value representing how often the issue appears across regression runs.

Range: 1 to 10

### recurrence

Integer value representing how many times the issue reappears after initial observation.

Range: 0 to 5

### fix_time

Integer estimate of engineering effort required to resolve the failure.

Range: 1 to 5

### log_message

Short textual failure description generated to resemble RTL regression logs.

### priority

Derived integer priority used for scheduling fixes.

Mapping:
- 1 = fatal
- 2 = error
- 3 = warning

## Generation Logic

- `priority` is deterministically derived from `severity`.
- `fatal` failures are biased toward higher `coverage_drop`, `frequency`, and `fix_time` values.
- `warning` failures are biased toward lower impact values.
- `log_message` is selected from module-specific message templates.

## Typical Use Cases

- Failure severity classification
- Verification dashboard prototyping
- Regression triage analytics
- Priority prediction and ranking models
