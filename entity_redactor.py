from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType
import re

# --- Hallucination confidence score UDF ---
def map_confidence(label):
    return 1 if label in ["REFUTES", "NOT ENOUGH INFO"] else 0

# --- Redaction UDF ---
def redact_entities(claim, entities):
    if not claim or not entities:
        return claim
    for entity in entities:
        if entity:
            claim = re.sub(rf'\b{re.escape(entity)}\b', '[REDACTED]', claim, flags=re.IGNORECASE)
    return claim

redact_udf = udf(redact_entities, StringType())

confidence_udf = udf(map_confidence, IntegerType())

# --- Spark Session ---
spark = SparkSession.builder \
    .appName("EntityRedactor") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# --- Schema ---
schema = StructType([
    StructField("id", IntegerType()),
    StructField("claim", StringType()),
    StructField("label", StringType()),
    StructField("entities", ArrayType(StringType()))
])

# --- Kafka Source ---
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "fever-stream") \
    .load()

# --- Parse + Redact ---
parsed_df = df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

redacted_df = parsed_df \
    .withColumn("claim", redact_udf(col("claim"), col("entities"))) \
    .withColumn("hallucination_confidence", confidence_udf(col("label")))

# --- Write to redacted topic ---
redacted_df.selectExpr("to_json(struct(*)) AS value") \
    .writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "redacted-claims") \
    .option("checkpointLocation", "./checkpoints/redactor") \
    .start() \
    .awaitTermination()
