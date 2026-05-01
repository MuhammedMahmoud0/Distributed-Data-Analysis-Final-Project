from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F

# Initialize Spark
spark = SparkSession.builder.appName("ModelServing").getOrCreate()

# Load production model from HDFS
MODEL_PATH = (
    "hdfs://localhost:9005/user/muhammed_mahmoud/models/production/RandomForest"
)
model = PipelineModel.load(MODEL_PATH)

app = Flask(__name__)


@app.route("/")
def home():
    return {"message": "Time Series Forecast API running"}


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Expect list of records
        df = spark.createDataFrame(data)

        # Ensure correct types
        df = df.withColumn("event_date", F.to_date("event_date"))

        # Run prediction
        pred_df = model.transform(df)

        result = (
            pred_df.select("event_date", "prediction")
            .toPandas()
            .to_dict(orient="records")
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
