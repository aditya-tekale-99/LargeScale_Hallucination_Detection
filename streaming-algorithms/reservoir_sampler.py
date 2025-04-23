from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType
import random
import json
from kafka import KafkaProducer
import time

# --- Parameters ---
k = 100
reservoir = []
seen_count = 0
last_push_time = 0
PUSH_INTERVAL = 10

# --- Kafka Producer ---
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# --- Reservoir Logic ---
def update_reservoir(batch_df, batch_id):
    global reservoir, seen_count, last_push_time

    refuted = batch_df.filter(col("label") == "REFUTES").collect()

    for row in refuted:
        seen_count += 1
        item = row.asDict()
        if len(reservoir) < k:
            reservoir.append(item)
        else:
            j = random.randint(0, seen_count - 1)
            if j < k:
                reservoir[j] = item

    print(f"\n--- Reservoir Sample (size={len(reservoir)}) ---")
    for entry in reservoir[:5]:
        print(f"[{entry['id']}] {entry['claim']}")
    print("...")

    current_time = time.time()
    if current_time - last_push_time >= PUSH_INTERVAL:
        try:
            producer.send("reservoir-sample", value=reservoir)
            producer.flush()
            last_push_time = current_time
            print("[✓] Reservoir sample published.")
        except Exception as e:
            print(f"[✗] Kafka publish failed: {e}")

# --- Spark Session ---
spark = SparkSession.builder \
    .appName("ReservoirSampler") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# --- Schema ---
schema = StructType([
    StructField("id", IntegerType()),
    StructField("claim", StringType()),
    StructField("label", StringType()),
    StructField("entities", ArrayType(StringType())),
    StructField("hallucination_confidence", IntegerType())
])

# --- Kafka Source: redacted-claims ---
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "redacted-claims") \
    .load()

parsed_df = df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

query = parsed_df.writeStream \
    .foreachBatch(update_reservoir) \
    .outputMode("update") \
    .option("checkpointLocation", "./checkpoints/reservoir") \
    .start()

query.awaitTermination()
