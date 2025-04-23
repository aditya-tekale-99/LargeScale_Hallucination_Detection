from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StringType, ArrayType, IntegerType
from pybloom_live import BloomFilter
from kafka import KafkaProducer
from datetime import datetime
import json

# 1. Spark Setup
spark = SparkSession.builder \
    .appName("Kafka-Bloom-Entity-Filter") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# 2. Kafka Source
df_raw = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "fever-stream") \
    .option("startingOffsets", "earliest") \
    .load()

# 3. Define message schema
schema = StructType() \
    .add("id", IntegerType()) \
    .add("claim", StringType()) \
    .add("label", StringType()) \
    .add("entities", ArrayType(StringType()))

# 4. Parse Kafka message
df_json = df_raw.selectExpr("CAST(value AS STRING) as json") \
    .select(from_json("json", schema).alias("data")) \
    .select("data.*")

# 5. Global Bloom Filter
bloom = BloomFilter(capacity=10000, error_rate=0.001)

# 6. Processing Logic
def process_batch(batch_df, batch_id):
    rows = batch_df.select("entities").collect()

    output = []

    for row in rows:
        if row["entities"]:
            for entity in row["entities"]:
                seen = entity in bloom
                if not seen:
                    bloom.add(entity)
                output.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "entity": entity,
                    "seen_before": seen
                })

    # Send to Kafka topic
    if output:
        producer = KafkaProducer(
            bootstrap_servers='localhost:9092',
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        for record in output:
            producer.send("fever-bloom-output", value=record)
        producer.flush()
        producer.close()

        print(f"→ Sent {len(output)} Bloom results to Kafka.")

# 7. Stream it
df_json.writeStream \
    .foreachBatch(process_batch) \
    .outputMode("append") \
    .option("checkpointLocation", "./checkpoints/bloomFilter") \
    .start() \
    .awaitTermination()