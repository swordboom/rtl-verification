# AI Debug Prioritization Agent (RTL Verification)

AI-based system for parsing regression logs, prioritizing failures, generating explainable scores, and visualizing analytics for hackathon/demo workflows.

## Overview

This project provides:

- Synthetic RTL failure dataset generation (10k-50k+ rows).
- Log/CSV upload ingestion for real-world datasets.
- Failure prioritization (`High`/`Medium`/`Low`) with `priority_score` (`0-100`).
- Explainability (SHAP when available) and root-cause hints.
- Dashboard analytics (severity distribution, priority histogram, heatmap, coverage impact).
- Workflow-aligned evaluation report generation.

## Tech Stack

- Backend: Python, FastAPI, Pandas, NumPy, scikit-learn, XGBoost, SHAP
- NLP: sentence-transformers (`all-MiniLM-L6-v2`) with TF-IDF fallback/fast mode
- Storage: SQLite + CSV
- Frontend: HTML + JavaScript
- Charts: Chart.js + Plotly

## Project Structure

```text
rtl-verification/
│
├── AI_Debug_Agent/
│   ├── __init__.py
│   ├── config.py
│   ├── data_ingestion_agent.py
│   ├── log_parser_agent.py
│   ├── feature_engineering_agent.py
│   ├── prioritization_model_agent.py
│   ├── explanation_agent.py
│   ├── dashboard_api_agent.py
│   ├── run_pipeline.py
│   ├── evaluation_report.py
│   │
│   ├── dashboard/
│   │   ├── index.html
│   │   ├── app.js
│   │   └── styles.css
│   │
│   ├── dataset/
│   │   └── .gitkeep
│   │
│   └── models/
│       └── .gitkeep
│
├── backend/
│   ├── main.py
│   ├── model.py
│   ├── parser.py
│   ├── feature_engineering.py
│   └── explainability.py
│
├── dataset_builder.py
├── train_model.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

1. Train model + generate artifacts:

```bash
python -m AI_Debug_Agent.run_pipeline --rows 12000
```

2. Start API and dashboard:

```bash
uvicorn AI_Debug_Agent.dashboard_api_agent:app --reload
```

3. Open:

- Dashboard: `http://127.0.0.1:8000/dashboard/index.html`
- API docs: `http://127.0.0.1:8000/docs`

## Training Profiles (Large Scale)

`POST /train` accepts:

- `row_count`: `10000` to `50000`
- `speed_profile`: `fast` or `balanced`

Use `fast` for high-volume uploads (10k-100k logs). It is optimized for throughput and switches earlier to scalable TF-IDF text features.

Example:

```json
{
  "row_count": 12000,
  "speed_profile": "fast"
}
```

## Upload Behavior

`POST /upload-logs` supports:

- `.log`, `.txt`: line-wise parsing with dedup of exact duplicate lines
- `.csv`: structured ingestion preserving row counts

Preprocessing summary is returned:

- `raw_log_lines`
- `unique_logs_after_dedup`
- `duplicates_removed`

## Uploaded Dataset Metrics (Separate Panel)

Uploaded metrics are available via:

- `GET /uploaded-metrics`
- `uploaded_model_metrics` in upload response

Metrics run in 3 modes:

1. `exact_labels`
   - Uses labels from `priority_label`, `priority`, `label`, `target`, `ground_truth`, `y`, `class`.
   - Supports mappings: `High/Medium/Low`, `1/2/3`, `fatal/error/warning`.

2. `estimated_from_severity`
   - Proxy labels from uploaded `severity` column.
   - Not true ground-truth accuracy.

3. `confidence_based_estimate`
   - No labels available; uses model confidence/entropy estimates.
   - Not true ground-truth accuracy.

## API Endpoints

- `GET /health`
- `GET /runtime-status`
  - Verifies XGBoost/SHAP/sentence-transformers availability and runtime readiness.
- `POST /train`
- `POST /parse-log`
- `POST /predict`
- `POST /predict-from-log`
- `POST /upload-logs`
- `GET /uploaded-metrics`
- `GET /analytics?source=auto|uploaded|dataset&module_name=<module>`
- `GET /evaluation-report?rows=<n>`
- `GET /demo-scenario`

## Workflow-Aligned Features

- Unique failure identification
- UVM/SVA/module-wise categorization
- Prioritized failure list (descending by score)
- Categorized error summary
- Suggested starting points for debug
- Impact analysis (`tests_affected`, module impact)
- Recent git change hints related to modules
- Dashboard module filter for analytics
- Deduplication and tokenization preprocessing

## Formal Evaluation Report

Generate from CLI:

```bash
python -m AI_Debug_Agent.evaluation_report --rows 12000
```

Artifacts:

- `reports/workflow_evaluation_report.json`
- `reports/workflow_evaluation_report.md`

Includes:

- Debug effort reduction estimate
- Clustering/categorization quality
- Prioritization clarity
- Model metrics: Accuracy, Precision, Recall, F1, NDCG
- Optional learning-to-rank check (`XGBRanker`)

## Notes

- For large uploads, keep model trained in `fast` profile before calling `/upload-logs`.
- SHAP explanations are used when available; otherwise feature-importance fallback is used.
- If MiniLM cannot be loaded, TF-IDF fallback is used.
