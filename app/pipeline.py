"""
pipeline.py — Daily data pipeline
Run manually:  python pipeline.py
Schedule:      0 2 * * * /path/to/python_env/bin/python /path/to/pipeline.py
"""

import json
import logging
from datetime import datetime, UTC

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, StringType, FloatType, TimestampType
from pyspark.ml import PipelineModel

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
HDFS_BASE = "hdfs://localhost:9005/user/muhammed_mahmoud"
BRONZE_SHEET1 = f"{HDFS_BASE}/bronze/sheet1/online_retail_sheet1.parquet"
BRONZE_SHEET2 = f"{HDFS_BASE}/bronze/sheet2/online_retail_sheet2.parquet"
FEAT_PATH = f"{HDFS_BASE}/features/daily"
PRED_PATH = f"{HDFS_BASE}/predictions"
MODEL_PATH = f"{HDFS_BASE}/models/production/RandomForest"

PRED_LOG = "/tmp/prediction_logs.jsonl"
ACTUALS_PATH = "/tmp/actuals.parquet"

# ── Spark ──────────────────────────────────────────────────────────────────
spark = SparkSession.builder.appName("DailyPipeline").getOrCreate()
spark.sparkContext.setLogLevel("WARN")


# ── Step 1: Cleaning ───────────────────────────────────────────────────────
def normalize_nulls(sdf):
    for col_name in sdf.columns:
        sdf = sdf.withColumn(
            col_name,
            F.when(
                F.trim(F.col(col_name).cast("string")).isin(
                    "", "nan", "NaN", "None", "none"
                ),
                F.lit(None),
            ).otherwise(F.col(col_name)),
        )
    return sdf


def clean(df):
    logger.info("Cleaning raw data...")

    df = normalize_nulls(df)

    df = df.withColumn(
        "StockCode",
        F.when(
            F.col("StockCode").rlike("^[0-9]"),
            F.regexp_extract(F.col("StockCode"), r"(\d{5})", 1),
        ).otherwise(None),
    )

    junk_values = [
        "21494",
        "22719",
        "22467",
        "20713",
        "?",
        "??",
        "???",
        "????",
        "?missing",
        "?? missing",
        "???missing",
        "????missing",
        "? sold as sets?",
        "?sold as sets?",
        "?lost",
        "???lost",
        "????damages????",
        "?display?",
        "?sold individually?",
    ]
    df = df.withColumn(
        "Description",
        F.when(F.col("Description").isin(junk_values), None).otherwise(
            F.col("Description")
        ),
    )
    df = df.withColumn(
        "Description",
        F.regexp_replace(F.col("Description"), r"^\*", ""),
    )

    df = df.withColumn("Quantity", F.abs(F.col("Quantity")))
    df = df.withColumn("Price", F.abs(F.col("Price")))

    min_positive_price = (
        df.filter(F.col("Price") > 0).agg(F.min("Price")).collect()[0][0]
    )
    df = df.withColumn(
        "Price",
        F.when(F.col("Price") == 0, min_positive_price).otherwise(F.col("Price")),
    )

    df = df.fillna({"Customer ID": 0})

    df = (
        df.withColumn("Invoice", F.col("Invoice").cast(StringType()))
        .withColumn("StockCode", F.col("StockCode").cast(IntegerType()))
        .withColumn("Description", F.col("Description").cast(StringType()))
        .withColumn("Quantity", F.col("Quantity").cast(IntegerType()))
        .withColumn("InvoiceDate", F.col("InvoiceDate").cast(TimestampType()))
        .withColumn("Price", F.col("Price").cast(FloatType()))
        .withColumn("Customer ID", F.col("Customer ID").cast(IntegerType()))
        .withColumn("Country", F.col("Country").cast(StringType()))
    )

    logger.info(f"Clean rows: {df.count()}")
    return df


# ── Step 2: Aggregate to daily revenue ────────────────────────────────────
def aggregate_daily(df):
    logger.info("Aggregating to daily revenue...")
    df_base = (
        df.withColumn("InvoiceDate", F.to_timestamp(F.col("InvoiceDate")))
        .withColumn("Quantity", F.col("Quantity").cast("double"))
        .withColumn("Price", F.col("Price").cast("double"))
        .withColumn("Revenue", F.col("Quantity") * F.col("Price"))
        .dropna(subset=["InvoiceDate", "Revenue"])
    )
    df_daily = (
        df_base.withColumn("event_date", F.to_date("InvoiceDate"))
        .groupBy("event_date")
        .agg(F.sum("Revenue").alias("target"))
        .orderBy("event_date")
    )
    logger.info(f"Daily rows: {df_daily.count()}")
    return df_daily


# ── Step 3: Feature engineering ────────────────────────────────────────────
def build_features(df_daily):
    logger.info("Building lag & rolling features...")

    w = Window.orderBy("event_date")
    w7 = Window.orderBy("event_date").rowsBetween(-6, -1)
    w28 = Window.orderBy("event_date").rowsBetween(-27, -1)

    df_features = (
        df_daily.withColumn("lag_1", F.lag("target", 1).over(w))
        .withColumn("lag_7", F.lag("target", 7).over(w))
        .withColumn("lag_14", F.lag("target", 14).over(w))
        .withColumn("lag_28", F.lag("target", 28).over(w))
        .withColumn("rolling_mean_7", F.avg("target").over(w7))
        .withColumn("rolling_std_7", F.stddev("target").over(w7))
        .withColumn("rolling_mean_28", F.avg("target").over(w28))
        .withColumn("day_of_week", F.dayofweek("event_date").cast("int"))
        .withColumn("week_of_year", F.weekofyear("event_date").cast("int"))
        .withColumn("month", F.month("event_date").cast("int"))
        .withColumn("quarter", F.quarter("event_date").cast("int"))
        .withColumn(
            "is_weekend",
            F.dayofweek("event_date").isin(1, 7).cast("int"),
        )
        .dropna()
    )

    logger.info(f"Feature rows (after dropna): {df_features.count()}")
    return df_features


# ── Step 4: Run model & save predictions ──────────────────────────────────
def run_predictions(df_features):
    logger.info("Loading model and running predictions...")
    model = PipelineModel.load(MODEL_PATH)
    pred_df = model.transform(df_features)

    # Save batch predictions to HDFS
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    out_path = f"{PRED_PATH}/{today}"
    pred_df.select("event_date", "target", "prediction").write.mode(
        "overwrite"
    ).parquet(out_path)
    logger.info(f"Predictions saved to HDFS: {out_path}")

    # Collect once — reuse for all local writes
    results = pred_df.select(
        F.col("event_date"),
        F.col("target").alias("actual"),
        F.col("prediction"),
    ).toPandas()

    # 1. Save actuals so /metrics can compute RMSE/MAE
    results[["event_date", "actual"]].to_parquet(ACTUALS_PATH, index=False)
    logger.info(f"Actuals saved to {ACTUALS_PATH}")

    # 2. Overwrite prediction log so /metrics and /logs show full dataset
    now_ts = datetime.now(UTC).isoformat()
    with open(PRED_LOG, "w") as f:
        for _, row in results.iterrows():
            entry = {
                "timestamp": now_ts,
                "event_date": str(row["event_date"]),
                "prediction": float(row["prediction"]),
            }
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Wrote {len(results)} rows to {PRED_LOG}")


# ── Step 5: Drift check ───────────────────────────────────────────────────
def check_drift(df_features):
    logger.info("Checking for data drift...")
    stats = df_features.select(
        F.mean("lag_1").alias("mean_lag1"),
        F.stddev("lag_1").alias("std_lag1"),
        F.mean("rolling_mean_7").alias("mean_roll7"),
    ).collect()[0]

    logger.info(
        f"Drift stats — mean_lag1={stats['mean_lag1']:.2f}  "
        f"std_lag1={stats['std_lag1']:.2f}  "
        f"mean_roll7={stats['mean_roll7']:.2f}"
    )

    if stats["std_lag1"] and stats["std_lag1"] > 100_000:
        logger.warning("HIGH VARIANCE in lag_1 — possible data issue or demand spike")


# ── Main ───────────────────────────────────────────────────────────────────
def run_pipeline():
    logger.info("=" * 50)
    logger.info(f"Pipeline started at {datetime.now(UTC).isoformat()}")

    logger.info(f"Reading sheet1 from {BRONZE_SHEET1}")
    df1 = spark.read.parquet(BRONZE_SHEET1)

    logger.info(f"Reading sheet2 from {BRONZE_SHEET2}")
    df2 = spark.read.parquet(BRONZE_SHEET2)

    raw = df1.union(df2)
    logger.info(f"Combined rows before cleaning: {raw.count()}")

    clean_df = clean(raw)
    daily_df = aggregate_daily(clean_df)
    feat_df = build_features(daily_df)

    feat_df.write.mode("overwrite").parquet(FEAT_PATH)
    logger.info(f"Features saved to {FEAT_PATH}")

    check_drift(feat_df)
    run_predictions(feat_df)

    logger.info(f"Pipeline finished at {datetime.now(UTC).isoformat()}")
    logger.info("=" * 50)


if __name__ == "__main__":
    run_pipeline()
