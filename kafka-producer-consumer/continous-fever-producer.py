# fever_streaming_producer.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import to_json, struct, lit
import time

# Define schema
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

schema = StructType([
    StructField("claim", StringType()),
    StructField("entity_redacted_claim", StringType()),
    StructField("label", IntegerType())
])

# Start Spark session
spark = SparkSession.builder \
    .appName("FEVERStreamingProducer") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Read data once (batch)
df = spark.read.schema(schema).json("../datasets/fever_preprocessed.jsonl")

# Define loop-based streaming emulation
while True:
    print("[INFO] Sending batch to Kafka...")

    df.withColumn("key", lit("fever")) \
      .selectExpr("CAST(key AS STRING)", "to_json(struct(*)) AS value") \
      .write \
      .format("kafka") \
      .option("kafka.bootstrap.servers", "localhost:9092") \
      .option("topic", "fever-stream") \
      .save()

    time.sleep(5)  # wait 5 seconds before next round (adjust as needed)