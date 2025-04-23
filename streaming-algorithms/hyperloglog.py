from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StringType, ArrayType, IntegerType
from datasketch import HyperLogLog
from kafka import KafkaProducer
from datetime import datetime
import json

# 1. Spark Setup
spark = SparkSession.builder \
    .appName("Kafka-HLL-Cumulative-Estimator") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# 2. Kafka Input Stream
df_raw = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "redacted-claims") \
    .option("startingOffsets", "earliest") \
    .load()

# 3. JSON Schema for message decoding
schema = StructType() \
    .add("id", IntegerType()) \
    .add("claim", StringType()) \
    .add("label", StringType()) \
    .add("entities", ArrayType(StringType())) \
    .add("hallucination_confidence", IntegerType())

# 4. Parse Kafka value column as JSON
df_json = df_raw.selectExpr("CAST(value AS STRING) as json") \
    .select(from_json("json", schema).alias("data")) \
    .select("data.*")

# 5. Global HyperLogLog instance (cumulative)
hll = HyperLogLog(p=14)

# 6. Function to send HLL count to Kafka
def send_to_kafka(batch_df, batch_id):
    rows = batch_df.select("entities").collect()

    for row in rows:
        entities = row["entities"]
        if entities:
            for e in entities:
                hll.update(e.encode('utf-8'))

    # Get cumulative estimate
    msg = {
        "timestamp": datetime.utcnow().isoformat(),
        "estimated_unique_entities": int(hll.count())
    }

    # Send message to Kafka
    producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    producer.send("hll-output", value=msg)
    producer.flush()
    producer.close()

    print(f"→ Sent to Kafka: {msg}")

# 7. Stream the data and apply custom sink
df_json.writeStream \
    .foreachBatch(send_to_kafka) \
    .outputMode("append") \
    .option("checkpointLocation", "./checkpoints/hll") \
    .start() \
    .awaitTermination()