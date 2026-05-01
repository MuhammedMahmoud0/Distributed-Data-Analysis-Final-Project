from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F
import json
import logging
from datetime import datetime, UTC

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Spark + Model ──────────────────────────────────────────────────────────
spark = SparkSession.builder.appName("ModelServing").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

MODEL_PATH = (
    "hdfs://localhost:9005/user/muhammed_mahmoud/models/production/RandomForest"
)
PREDICTION_LOG = "/tmp/prediction_logs.jsonl"
ACTUALS_PATH = "/tmp/actuals.parquet"

logger.info("Loading model from HDFS...")
model = PipelineModel.load(MODEL_PATH)
logger.info("Model loaded successfully.")

app = Flask(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────
def log_prediction(input_record: dict, prediction: float):
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event_date": input_record.get("event_date"),
        "prediction": prediction,
    }
    with open(PREDICTION_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({"message": "Time Series Forecast API running"})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data or not isinstance(data, list):
            return jsonify({"error": "Expected a JSON list of records"}), 400

        logger.info(f"Prediction request — {len(data)} record(s)")

        df = spark.createDataFrame(data)
        df = df.withColumn("event_date", F.to_date("event_date"))

        pred_df = model.transform(df)
        result = (
            pred_df.select("event_date", "prediction")
            .toPandas()
            .to_dict(orient="records")
        )

        for i, record in enumerate(result):
            log_prediction(data[i], record["prediction"])

        for r in result:
            r["event_date"] = str(r["event_date"])

        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/metrics", methods=["GET"])
def metrics():
    import pandas as pd
    import numpy as np
    import os

    try:
        if not os.path.exists(PREDICTION_LOG):
            return jsonify({"error": "No prediction logs yet"}), 404

        logs = pd.read_json(PREDICTION_LOG, lines=True)

        response = {
            "total_predictions": int(len(logs)),
            "last_prediction_at": str(logs["timestamp"].max()),
            "date_range": {
                "from": str(logs["event_date"].min()),
                "to": str(logs["event_date"].max()),
            },
            "prediction_stats": {
                "mean": round(float(logs["prediction"].mean()), 2),
                "min": round(float(logs["prediction"].min()), 2),
                "max": round(float(logs["prediction"].max()), 2),
                "std": round(float(logs["prediction"].std()), 2),
            },
        }

        # Compute error metrics if actuals exist
        if os.path.exists(ACTUALS_PATH):
            actuals = pd.read_parquet(ACTUALS_PATH)

            # Normalise both to plain "YYYY-MM-DD" strings before merging
            logs["event_date"] = pd.to_datetime(logs["event_date"]).dt.strftime(
                "%Y-%m-%d"
            )
            actuals["event_date"] = pd.to_datetime(actuals["event_date"]).dt.strftime(
                "%Y-%m-%d"
            )

            merged = logs.merge(actuals, on="event_date", how="inner")
            logger.info(f"Merged {len(merged)} rows for error metrics")

            if len(merged) > 0:
                errors = merged["prediction"] - merged["actual"]
                response["error_metrics"] = {
                    "n_compared": int(len(merged)),
                    "RMSE": round(float(np.sqrt((errors**2).mean())), 2),
                    "MAE": round(float(errors.abs().mean()), 2),
                    "MAPE": round(
                        float((errors.abs() / merged["actual"]).mean() * 100), 2
                    ),
                }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Metrics error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/logs", methods=["GET"])
def get_logs():
    import pandas as pd
    import os

    if not os.path.exists(PREDICTION_LOG):
        return jsonify([])

    n = int(request.args.get("n", 50))
    logs = pd.read_json(PREDICTION_LOG, lines=True).tail(n)
    return jsonify(logs.to_dict(orient="records"))


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
