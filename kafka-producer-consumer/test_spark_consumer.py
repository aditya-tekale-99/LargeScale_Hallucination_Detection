from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import joblib

# Step 1: Initialize Spark Session
spark = SparkSession.builder \
    .appName("KafkaToSnowflakePrediction") \
    .config("spark.jars.packages", "net.snowflake:snowflake-jdbc:RELEASE,net.snowflake:spark-snowflake_2.12:RELEASE") \
    .getOrCreate()

# Step 2: Load Pretrained Model and Vectorizer
model = joblib.load('logistic_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Step 3: Define a UDF for Prediction
def predict_label(text):
    if text is None:
        return "UNKNOWN"
    X = vectorizer.transform([text])
    pred = model.predict(X)
    return pred[0]

predict_udf = udf(predict_label, StringType())

# Step 4: Read from Kafka topic
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "redacted-claims") \
    .option("startingOffsets", "latest") \
    .load()

# Step 5: Prepare Kafka messages
value_df = kafka_df.selectExpr("CAST(value AS STRING)")

from pyspark.sql.functions import from_json
from pyspark.sql.types import StructType, StructField

# Define the schema based on what your producer sends
schema = StructType([
    StructField("claim", StringType()),
    StructField("label", StringType()),
    StructField("entity_redacted_claim", StringType())
])

parsed_df = value_df.select(from_json(col("value"), schema).alias("data")).select("data.*")

# Step 6: Predict hallucination label
predicted_df = parsed_df.withColumn("predicted_label", predict_udf(col("entity_redacted_claim")))

# Step 7: Write to Snowflake
sfOptions = {
  "sfURL": "<your_account>.snowflakecomputing.com",
  "sfDatabase": "<your_database>",
  "sfSchema": "<your_schema>",
  "sfWarehouse": "<your_warehouse>",
  "sfRole": "<your_role>",
  "sfUser": "<your_user>",
  "sfPassword": "<your_password>"
}

# Save only important columns
final_df = predicted_df.select("claim", "entity_redacted_claim", "label", "predicted_label")

# Instead of write_to_snowflake, just print to console
query = predicted_df.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \
    .start()

query.awaitTermination()
