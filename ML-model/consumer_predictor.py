from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_json, struct
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml import PipelineModel

# --- Define schema of the Kafka message value ---
schema = StructType([
    StructField("claim", StringType()),
    StructField("entity_redacted_claim", StringType()),
    StructField("label", IntegerType())
])

# --- Initialize Spark session ---
spark = SparkSession.builder \
    .appName("ClaimPredictor") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# --- Load Kafka stream (start from earliest) ---
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "fever-stream") \
    .option("startingOffsets", "earliest") \
    .load()

# --- Parse JSON inside Kafka value field ---
parsed_df = df.selectExpr("CAST(value AS STRING) as json_str") \
    .select(from_json(col("json_str"), schema).alias("data")) \
    .select("data.*")

# --- DEBUG: print incoming rows to console (optional) ---
parsed_df.writeStream \
    .format("console") \
    .outputMode("append") \
    .start()

# --- Load pre-trained ML pipeline model ---
model = PipelineModel.load("models/claim_classifier")  # <-- Update path if different

# --- Apply prediction ---
predicted_df = model.transform(parsed_df)

# --- Select and rename output fields ---
output_df = predicted_df.select(
    col("claim"),
    col("entity_redacted_claim"),
    col("label"),
    col("prediction").cast("int").alias("predicted_label")
)

# --- Send prediction to Kafka topic ---
output_df.select(to_json(struct("*")).alias("value")) \
    .writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "predicted-claims") \
    .option("checkpointLocation", "./checkpoints/predictor") \
    .start() \
    .awaitTermination()


# command to run: spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5 consumer_predictor.py