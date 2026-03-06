# AI Debug Prioritization Agent for RTL Verification

This project builds a modular AI coding-agent style system that parses RTL regression failures, engineers features, predicts bug priority, explains predictions, and serves a dashboard/API for hackathon demos.

## What This Solves

Manual RTL triage is slow and inconsistent. This system automates:

- Parsing raw verification logs into structured failure records.
- Scoring failures with a priority label (`High`, `Medium`, `Low`) and priority score (`0-100`).
- Explaining why a bug was prioritized (SHAP where available).
- Visualizing failure distributions and trends.

## Tech Stack

- Backend: Python, FastAPI, Pandas, scikit-learn, XGBoost, SHAP
- AI + NLP: Sentence Transformers (`all-MiniLM-L6-v2`) with TF-IDF fallback
- Database: SQLite (default hackathon option)
- Frontend: HTML + JavaScript
- Visualization: Chart.js, Plotly, Seaborn-ready data outputs

## Project Architecture

```text
rtl-verification/
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
│   ├── dataset/
│   │   ├── rtl_dataset.csv
│   │   └── rtl_failures.db
│   ├── models/
│   │   ├── xgboost_model.pkl
│   │   └── feature_pipeline.pkl
│   └── dashboard/
│       ├── index.html
│       ├── app.js
│       └── styles.css
├── backend/
│   ├── main.py
│   ├── model.py
│   ├── parser.py
│   ├── feature_engineering.py
│   └── explainability.py
├── dataset_builder.py
├── train_model.py
└── requirements.txt
```

## Agent Modules

- `data_ingestion_agent.py`: Generates synthetic RTL failures (10k-50k), writes CSV + SQLite.
- `log_parser_agent.py`: Parses logs like:
  - `[ERROR] Module: MemoryCtrl ... Coverage drop detected: 8%. Regression: nightly_run`
- `feature_engineering_agent.py`: Builds tabular features + text embeddings, normalizes, encodes categories.
- `prioritization_model_agent.py`: Trains XGBoost classifier and outputs class probabilities + priority score.
- `explanation_agent.py`: SHAP-based explanations, root-cause hints, trend detection, clustering, git-fix insights.
- `dashboard_api_agent.py`: FastAPI endpoints for train/predict/analytics/dashboard.

## Synthetic Dataset Schema

Generated columns include:

- `failure_id`
- `timestamp`
- `module_name`
- `error_code`
- `severity`
- `coverage_drop`
- `failure_frequency`
- `historical_bug_count`
- `avg_fix_time`
- `assertion_type`
- `regression_suite`
- `assertion_failures`
- `log_message`
- `priority_label`
- `priority_score`

Severity-conditioned distributions bias fatal failures toward higher impact values.

## Setup

1. Create environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Optional: build dataset only:

```bash
python dataset_builder.py
```

3. Optional: train model from script:

```bash
python train_model.py
```

## Run Full Pipeline

```bash
python -m AI_Debug_Agent.run_pipeline --rows 12000
```

This generates:

- `AI_Debug_Agent/dataset/rtl_dataset.csv`
- `AI_Debug_Agent/dataset/rtl_failures.db`
- `AI_Debug_Agent/models/xgboost_model.pkl`
- `AI_Debug_Agent/models/feature_pipeline.pkl`

## Run API + Dashboard

```bash
uvicorn AI_Debug_Agent.dashboard_api_agent:app --reload
```

Open:

- Dashboard UI: `http://127.0.0.1:8000/dashboard/index.html`
- OpenAPI docs: `http://127.0.0.1:8000/docs`

## API Endpoints

- `GET /health` -> service and artifact status
- `POST /train` -> generate dataset + train model
- `POST /parse-log` -> parse raw log into structured fields
- `POST /predict` -> predict from structured payload
- `POST /predict-from-log` -> parse + predict from raw log
- `POST /upload-logs` -> batch score uploaded `.log/.txt` file
- `GET /analytics` -> distributions, heatmap data, trends, clustering, git insights
- `GET /demo-scenario` -> manual vs AI demo comparison

## Priority Model + Score

- Classifier predicts `High/Medium/Low`.
- `priority_score` is computed from class probabilities into a 0-100 scale.
- Suggested interpretation:
  - `80-100`: urgent
  - `50-79`: medium scheduling priority
  - `<50`: low urgency / monitor

## Explainability and Advanced Features

- SHAP local explanation (if SHAP is available)
- Top contributing features for each prediction
- Root-cause suggestions by module/error/assertion pattern
- Trend detection (module-level weekly slope)
- Failure clustering (KMeans)
- Git fix insight extraction (`git log`)

## Evaluation Metrics

Training reports:

- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1 score (weighted)
- NDCG (ranking quality)
- Confusion matrix
- Full per-class classification report

## Hackathon Demo Flow

- Manual flow: read logs -> inspect bug -> prioritize (`~30 min`)
- AI flow: upload logs -> model scores + explains -> ranked output (`~10 sec`)

## Notes

- If `sentence-transformers` model download is unavailable, the system falls back to TF-IDF text features.
- `PrioritizationModelAgent` uses XGBoost when available; a sklearn gradient boosting fallback is included for resilience.
