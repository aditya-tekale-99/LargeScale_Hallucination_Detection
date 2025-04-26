from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

schema = StructType([
    StructField("claim", StringType()),
    StructField("entity_redacted_claim", StringType()),
    StructField("label", StringType()),
    StructField("predicted_label", StringType())
])

spark = SparkSession.builder \
    .appName("KafkaToSnowflake") \
    .config("spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
            "net.snowflake:snowflake-jdbc:3.13.30,"
            "net.snowflake:spark-snowflake_2.12:3.0.0") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "2g") \
    .config("spark.sql.shuffle.partitions", "10") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "predicted-labels") \
    .option("startingOffsets", "latest") \
    .option("maxOffsetsPerTrigger", 1000) \
    .load()


parsed_df = df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")).select("data.*") \
    .withColumn("predicted_label", col("predicted_label").cast("double")) \
    .withColumn("timestamp", current_timestamp())

sfOptions = {
    "sfURL": "xxxxxxx-xxxxxxx.snowflakecomputing.com",
    "sfUser": "adityatekale",
    "sfPassword": "xxxx228",
    "sfDatabase": "DATA228_FP",
    "sfSchema": "HALLUCINATION",
    "sfWarehouse": "compute_wh",
    "sfRole": "accountadmin",
    "usestagingtable": "OFF",
    "partition_size_mb": "64",
    "use_exponential_backoff": "on" 
}

def write_to_snowflake(batch_df, batch_id):
    try:
        batch_df.repartition(4).write \
            .format("snowflake") \
            .options(**sfOptions) \
            .option("dbtable", "predicted_label") \
            .mode("append") \
            .save()
        print(f"[✓] Batch {batch_id} sent to Snowflake.")
    except Exception as e:
        print(f"[!] Batch {batch_id} error: {e}")

query = parsed_df.writeStream \
    .foreachBatch(write_to_snowflake) \
    .outputMode("append") \
    .option("checkpointLocation", "/tmp/checkpoint_rawstream_snowflake") \
    .trigger(processingTime="30 seconds") \
    .start()

query.awaitTermination()
# command to run the code: spark-submit \
#  --driver-memory 4g \
#  --executor-memory 4g \
#  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,net.snowflake:snowflake-jdbc:3.13.30,net.snowflake:spark-snowflake_2.12:3.0.0 \
#  topic-snowflake.py
