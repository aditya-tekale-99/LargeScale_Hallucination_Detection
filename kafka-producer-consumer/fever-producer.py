from pyspark.sql import SparkSession
from pyspark.sql.functions import to_json, struct
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# --- Define schema based on your data ---
schema = StructType([
    StructField("claim", StringType()),
    StructField("entity_redacted_claim", StringType()),
    StructField("label", IntegerType())
])

# --- Start Spark session ---
spark = SparkSession.builder \
    .appName("FEVERProducer") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5") \
    .getOrCreate()

# --- Read the JSON data ---
df = spark.read.schema(schema).json("/Users/adi/Downloads/SJSU/Sem 2/DATA 228/Final Project/LargeScale_Hallucination_Detection/datasets/fever_preprocessed.jsonl")

# --- Serialize and write to Kafka topic ---
df.select(to_json(struct("*")).alias("value")) \
    .write \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "fever-stream") \
    .save()
 






#command to run: spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5 fever-producer.py
