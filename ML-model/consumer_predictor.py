from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_json, struct, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml import PipelineModel

schema = StructType([
    StructField("claim", StringType()),
    StructField("entity_redacted_claim", StringType()),
    StructField("label", IntegerType())
])

spark = SparkSession.builder \
    .appName("ClaimPredictor") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "fever-stream") \
    .option("startingOffsets", "earliest") \
    .option("failOnDataLoss", "false") \
    .load()

parsed_df = df.selectExpr("CAST(value AS STRING) as json_str") \
    .select(from_json(col("json_str"), schema).alias("data")) \
    .select("data.*")

try:
    model = PipelineModel.load("models/claim_classifier")
    
    predicted_df = model.transform(parsed_df)
    
    
    #output_df = predicted_df.select(to_json(struct("claim", "entity_redacted_claim", "label", "predicted_label")).alias("value"))
    output_df = predicted_df.select(to_json(struct("claim", "entity_redacted_claim", "label", col("prediction").alias("predicted_label"))).alias("value"))
    
    kafka_query = output_df.writeStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("topic", "predicted-labels") \
        .option("checkpointLocation", "/tmp/predicted-claims-checkpoint-new") \
        .outputMode("append") \
        .trigger(processingTime="5 seconds") \
        .start()
    
    kafka_query.awaitTermination()
except Exception as e:
    print(f"Error in model loading or processing: {e}")
    import traceback
    traceback.print_exc()

#command to run the code: spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5 consumer_predictor.py
