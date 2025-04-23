from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, IntegerType
import json
from kafka import KafkaProducer
from datetime import datetime
import time

# --- DGIM Class ---
class DGIM:
    def __init__(self, window_size):
        self.window_size = window_size
        self.buckets = []
        self.current_time = 0

    def update(self, bit):
        self.current_time += 1
        if bit == 1:
            self.buckets.append((self.current_time, 1))
        i = 0
        while i < len(self.buckets) - 2:
            if self.buckets[i][1] == self.buckets[i + 1][1]:
                self.buckets[i] = (self.buckets[i + 1][0], self.buckets[i][1] * 2)
                self.buckets.pop(i + 1)
            else:
                i += 1
        oldest = self.current_time - self.window_size
        self.buckets = [b for b in self.buckets if b[0] > oldest]

    def count(self):
        if not self.buckets:
            return 0
        total = 0
        sizes = {}
        for _, size in self.buckets:
            sizes[size] = sizes.get(size, 0) + 1
        for size, count in sizes.items():
            total += size * (count - 1)
            total += size // 2
        return total

# --- Initialize DGIM ---
dgim = DGIM(window_size=1000)
last_push_time = 0
PUSH_INTERVAL = 10  # seconds

# --- Kafka Producer ---
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# --- Batch Processing Function ---
def process_batch(batch_df, batch_id):
    global dgim, last_push_time

    bits = batch_df.select("hallucination_confidence").rdd.flatMap(lambda x: x).collect()
    for bit in bits:
        if bit is not None:
            dgim.update(int(bit))

    est_count = dgim.count()
    print(f"\n--- Batch {batch_id} | DGIM estimate (last 1000): {est_count}")

    current_time = time.time()
    if current_time - last_push_time >= PUSH_INTERVAL:
        try:
            message = {"estimated_hallucinations": est_count, 
                        "timestamp": datetime.utcnow().isoformat()}
            producer.send("hallucination-estimate", value=message)
            producer.flush()
            last_push_time = current_time
            print("[✓] DGIM estimate published:", message)
        except Exception as e:
            print(f"[✗] Kafka publish failed: {e}")

# --- Spark Session ---
spark = SparkSession.builder \
    .appName("DGIMTracker") \
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

# --- Kafka Source ---
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "redacted-claims") \
    .load()

parsed_df = df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

# --- Stream Logic ---
query = parsed_df.writeStream \
    .foreachBatch(process_batch) \
    .outputMode("update") \
    .option("checkpointLocation", "./checkpoints/dgim") \
    .start()

query.awaitTermination()
