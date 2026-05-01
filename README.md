# Distributed Data Analysis Project

End-to-end distributed data pipeline for retail revenue forecasting. The pipeline cleans raw data, aggregates daily revenue, engineers lag/rolling features, and runs a production model stored in HDFS. A Flask API serves predictions and metrics, while a Streamlit UI provides interactive prediction and monitoring.

## Features

- Spark-based cleaning, aggregation, and feature engineering
- Daily batch pipeline writes features and predictions to HDFS
- Model serving API with prediction logs and metrics
- Streamlit UI for predictions, monitoring, and logs
- HDFS services via Docker Compose

## Project Structure

```
api/
  api.py            # Simple prediction API (legacy)
  ui.py             # Simple Streamlit UI (legacy)
app/
  app.py            # Full Flask API with metrics/logs
  pipeline.py       # Daily batch pipeline
  streamlit_app.py  # Full Streamlit UI
data/
  online_retail_sheet1.csv
  online_retail_sheet2.csv
mlartifacts/
  ...
notebooks/
  DDA_Final_Project.ipynb
  DDA_Machine_Learning.ipynb
  setup_hdfs.ipynb
output/
```

## Requirements

- Python 3.9+ recommended
- Java 8+ (for Spark/HDFS)
- Docker and Docker Compose (for local HDFS cluster)

If you need a dependency list, use the notebooks and scripts as the source of truth (PySpark, Flask, Streamlit, pandas, numpy, requests, MLflow).

## Services (HDFS)

Bring up HDFS with Docker Compose:

```
docker compose up -d
```

This starts:

- NameNode: http://localhost:9870
- HDFS RPC: hdfs://localhost:9005

The `hdfs-init` service creates `/user/muhammed_mahmoud` in HDFS.

## Data Flow

1. Raw CSVs live in [data/](data/).
2. Notebooks load data and write bronze Parquet to HDFS:
    - `hdfs://localhost:9005/user/muhammed_mahmoud/bronze/sheet1/online_retail_sheet1.parquet`
    - `hdfs://localhost:9005/user/muhammed_mahmoud/bronze/sheet2/online_retail_sheet2.parquet`
3. The pipeline builds daily features and writes:
    - Features: `.../features/daily`
    - Predictions: `.../predictions/YYYY-MM-DD`
4. The API reads the production model from:
    - `hdfs://localhost:9005/user/muhammed_mahmoud/models/production/RandomForest`

## Run the Pipeline

```
python app/pipeline.py
```

## API Server (Full)

```
python app/app.py
```

Endpoints:

- `GET /` health message
- `GET /health` model load status
- `POST /predict` list of feature records
- `GET /metrics` prediction stats and optional error metrics
- `GET /logs?n=50` recent prediction logs

Prediction logs are stored at `/tmp/prediction_logs.jsonl` and actuals at `/tmp/actuals.parquet`.

## Streamlit UI (Full)

```
streamlit run app/streamlit_app.py
```

The UI calls the API at `http://localhost:8000` and exposes three pages:
Predict, Monitor, and Logs.

## MLflow Tracking

Run MLflow with HDFS-backed artifacts:

```
mlflow ui --port 5000
```

## Notebooks

- [notebooks/setup_hdfs.ipynb](notebooks/setup_hdfs.ipynb): HDFS setup and checks
- [notebooks/DDA_Final_Project.ipynb](notebooks/DDA_Final_Project.ipynb): End-to-end analysis
- [notebooks/DDA_Machine_Learning.ipynb](notebooks/DDA_Machine_Learning.ipynb): Model training workflow

Run them in this order to set up HDFS, prepare data, and train the model. The notebooks write outputs to HDFS and MLflow.

## Output

Artifacts, plots, and exported files are written to [output/](output/).

## License

This project is for academic use.
