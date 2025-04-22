from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, pandas_udf
from pyspark.sql.types import StructType, StructField, StringType
import pandas as pd
import joblib

# ---------------------------
# 1. Initialize Spark Session
# ---------------------------
spark = SparkSession.builder \
    .appName("KafkaToSnowflakePrediction") \
    .config("spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0,"
            "net.snowflake:snowflake-jdbc:3.13.25,"
            "net.snowflake:spark-snowflake_2.12:2.9.3-spark_3.3") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ---------------------------
# 2. Kafka Message Schema
# ---------------------------
schema = StructType([
    StructField("claim", StringType()),
    StructField("label", StringType()),
    StructField("entity_redacted_claim", StringType())
])

# ---------------------------
# 3. Pandas UDF for Prediction
# ---------------------------
@pandas_udf(StringType())
def predict_label_udf(text_series: pd.Series) -> pd.Series:
    try:
        model = joblib.load("logistic_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
    except Exception as e:
        print(f"[!] Model/Vectorizer Load Failed: {e}")
        return pd.Series(["ERROR"] * len(text_series))

    def predict(text):
        try:
            if not text or not isinstance(text, str):
                return "UNKNOWN"
            return str(model.predict(vectorizer.transform([text]))[0])
        except Exception as e:
            return "ERROR"

    return text_series.apply(predict)

# ---------------------------
# 4. Read Kafka Stream
# ---------------------------
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "redacted-claims") \
    .option("startingOffsets", "latest") \
    .load()

value_df = kafka_df.selectExpr("CAST(value AS STRING)")
parsed_df = value_df.select(from_json(col("value"), schema).alias("data")).select("data.*")

# ---------------------------
# 5. Apply Prediction
# ---------------------------
predicted_df = parsed_df.withColumn("predicted_label", predict_label_udf(col("entity_redacted_claim")))

# ---------------------------
# 6. Snowflake Configuration
# ---------------------------
sfOptions = {
    "sfURL": "dezodls-xz86953.snowflakecomputing.com",
    "sfDatabase": "DATA228_FP",
    "sfSchema": "HALLUCINATION",
    "sfWarehouse": "compute_wh",
    "sfRole": "accountadmin",
    "sfUser": "adityatekale",
    "sfPassword": "Aditya@data228",
    "usestagingtable": "OFF",
    "tempDir": "file:///tmp"
}

# ---------------------------
# 7. Write Batch to Snowflake
# ---------------------------
def write_to_snowflake(batch_df, batch_id):
    try:
        batch_df.write \
            .format("snowflake") \
            .options(**sfOptions) \
            .option("dbtable", "hallucination_predictions") \
            .mode("append") \
            .save()
        print(f"[✓] Batch {batch_id} successfully written to Snowflake.")
    except Exception as e:
        print(f"[!] Batch {batch_id} failed: {e}")

# ---------------------------
# 8. Start Streaming Query
# ---------------------------
final_df = predicted_df.select("claim", "entity_redacted_claim", "label", "predicted_label")

query = final_df.writeStream \
    .foreachBatch(write_to_snowflake) \
    .outputMode("append") \
    .option("checkpointLocation", "/tmp/hallucination_checkpoint") \
    .trigger(processingTime="30 seconds") \
    .start()

query.awaitTermination()