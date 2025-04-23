from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import time
import json
import random
from typing import List, Dict
from pymongo import MongoClient

# 1. Schema
schema = StructType([
    StructField("claim", StringType()),
    StructField("entity_redacted_claim", StringType()),
    StructField("label", StringType()),
    StructField("predicted_label", StringType())
])

# 2. Spark Init
spark = SparkSession.builder \
    .appName("KafkaToMongoDB_Reservoir") \
    .config("spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# MongoDB Connection
def get_mongo_client():
    """Create and return a MongoDB client"""
    # Update these connection details
    mongo_uri = "mongodb://localhost:27017/"
    client = MongoClient(mongo_uri)
    return client

# 3. Reservoir Sampling Implementation
class ReservoirSampler:
    def __init__(self, reservoir_size=100):
        """Initialize a reservoir sampler with specified reservoir size"""
        self.reservoir_size = reservoir_size
        self.reservoir = []
        self.count = 0
        
    def add(self, item):
        """Add an item to the reservoir using reservoir sampling algorithm"""
        self.count += 1
        
        # If the reservoir is not full, just append the item
        if len(self.reservoir) < self.reservoir_size:
            self.reservoir.append(item)
        else:
            # Randomly decide whether to include the new item
            # The probability of keeping the new item is reservoir_size/count
            j = random.randint(0, self.count - 1)
            if j < self.reservoir_size:
                self.reservoir[j] = item
    
    def get_samples(self):
        """Return the current reservoir samples"""
        return self.reservoir
    
    def to_dict(self):
        """Convert reservoir state to a serializable dictionary"""
        return {
            "reservoir_size": self.reservoir_size,
            "count": self.count,
            "reservoir": self.reservoir
        }
    
    @classmethod
    def from_dict(cls, data):
        """Reconstruct a ReservoirSampler from a dictionary"""
        sampler = cls(data["reservoir_size"])
        sampler.count = data["count"]
        sampler.reservoir = data["reservoir"]
        return sampler

def serialize_reservoir(reservoir):
    """Serialize Reservoir Sampler state to JSON string"""
    if reservoir is None:
        return "{}"
    return json.dumps(reservoir.to_dict())

# 4. Kafka Stream
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "predicted-labels") \
    .option("startingOffsets", "latest") \
    .load()

parsed_df = df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")).select("data.*") \
    .withColumn("predicted_label", col("predicted_label").cast("double")) \
    .withColumn("timestamp", current_timestamp())

# 5. Batch Writer with Reservoir Sampling
def process_batch(batch_df, batch_id):
    try:
        # Initialize MongoDB client
        mongo_client = get_mongo_client()
        db = mongo_client["hallucination_detection"]  # Database name
        reservoir_state_collection = db["reservoir_state"]  # Collection for state
        reservoir_samples_collection = db["reservoir_samples"]  # Collection for samples
        stats_collection = db["reservoir_stats"]  # Collection for statistics
        
        # Initialize processing stats
        row_count = 0
        
        # Try to load existing Reservoir Sampler state
        try:
            latest_reservoir = reservoir_state_collection.find_one(
                sort=[("updated_at", -1)]  # Get the most recent state
            )
            
            if latest_reservoir and "reservoir_state" in latest_reservoir:
                # Load the existing state
                reservoir = ReservoirSampler.from_dict(json.loads(latest_reservoir["reservoir_state"]))
                print(f"[✓] Loaded existing Reservoir Sampler state from MongoDB.")
            else:
                # Initialize a new Reservoir Sampler
                reservoir_size = 100  # Configure your reservoir size
                reservoir = ReservoirSampler(reservoir_size)
                print(f"[i] Initialized new Reservoir Sampler with size {reservoir_size}.")
        except Exception as e:
            # If there's an error, start with a fresh Reservoir Sampler
            reservoir_size = 100  # Configure your reservoir size
            reservoir = ReservoirSampler(reservoir_size)
            print(f"[!] Error loading Reservoir Sampler state: {e}")
            print(f"[i] Initialized new Reservoir Sampler with size {reservoir_size}.")
        
        # Classifications to track
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        # Process each row in the batch
        for row in batch_df.toLocalIterator():
            row_count += 1
            
            # Prepare the row for storage
            row_dict = row.asDict()
            # Convert timestamp to string for MongoDB compatibility
            if "timestamp" in row_dict and hasattr(row_dict["timestamp"], "strftime"):
                row_dict["timestamp"] = row_dict["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            
            # Track prediction accuracy
            predicted_positive = row["predicted_label"] >= 0.5
            actual_positive = row["label"].lower() == "true"
            
            if predicted_positive and actual_positive:
                true_positives += 1
            elif predicted_positive and not actual_positive:
                false_positives += 1
            elif not predicted_positive and not actual_positive:
                true_negatives += 1
            elif not predicted_positive and actual_positive:
                false_negatives += 1
            
            # Add to reservoir sampler
            # We'll add comprehensive information
            row_dict["batch_id"] = batch_id
            row_dict["processed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            row_dict["predicted_positive"] = predicted_positive
            row_dict["actual_positive"] = actual_positive
            
            # Determine prediction category
            if predicted_positive and actual_positive:
                row_dict["prediction_category"] = "true_positive"
            elif predicted_positive and not actual_positive:
                row_dict["prediction_category"] = "false_positive"
            elif not predicted_positive and not actual_positive:
                row_dict["prediction_category"] = "true_negative"
            else:
                row_dict["prediction_category"] = "false_negative"
                
            # Add to reservoir 
            reservoir.add(row_dict)
            
        if row_count == 0:
            print(f"[i] Batch {batch_id} is empty, skipping processing.")
            mongo_client.close()
            return
            
        print(f"[i] Batch {batch_id} contains {row_count} records.")
        
        # Save Reservoir Sampler state to MongoDB
        try:
            reservoir_state_json = serialize_reservoir(reservoir)
            reservoir_state_record = {
                "reservoir_state": reservoir_state_json,
                "updated_at": int(time.time()),
                "batch_id": batch_id
            }
            
            # Insert Reservoir state
            reservoir_state_collection.insert_one(reservoir_state_record)
            print(f"[✓] Successfully saved Reservoir Sampler state to MongoDB.")
        except Exception as e:
            print(f"[!] Error saving Reservoir Sampler state to MongoDB: {e}")
        
        # Save statistics to MongoDB
        try:
            # Calculate accuracy metrics
            accuracy = (true_positives + true_negatives) / row_count if row_count > 0 else 0
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            stats_record = {
                "batch_id": batch_id,
                "batch_size": row_count,
                "reservoir_sample_count": len(reservoir.get_samples()),
                "reservoir_size": reservoir.reservoir_size,
                "total_processed": reservoir.count,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "true_negatives": true_negatives,
                "false_negatives": false_negatives,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Insert statistics
            stats_collection.insert_one(stats_record)
            print(f"[✓] Successfully saved statistics to MongoDB.")
            
            # Print the statistics
            print(f"[✓] Reservoir Sampling Results for Batch {batch_id}:")
            print(f"    - Batch size: {stats_record['batch_size']}")
            print(f"    - Reservoir samples: {stats_record['reservoir_sample_count']} of {stats_record['reservoir_size']}")
            print(f"    - Total processed by reservoir: {stats_record['total_processed']}")
            print(f"    - Accuracy: {stats_record['accuracy']:.4f}")
            print(f"    - Precision: {stats_record['precision']:.4f}")
            print(f"    - Recall: {stats_record['recall']:.4f}")
            print(f"    - F1 Score: {stats_record['f1_score']:.4f}")
        except Exception as e:
            print(f"[!] Error saving statistics to MongoDB: {e}")
        
        # Save reservoir samples to MongoDB (replacing previous samples)
        try:
            samples = reservoir.get_samples()
            if samples:
                # First delete previous samples
                reservoir_samples_collection.delete_many({})
                
                # Then insert new samples
                reservoir_samples_collection.insert_many(samples)
                print(f"[✓] Successfully saved {len(samples)} reservoir samples to MongoDB.")
                
                # Sample distribution analysis
                categories = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
                for sample in samples:
                    if "prediction_category" in sample:
                        categories[sample["prediction_category"]] = categories.get(sample["prediction_category"], 0) + 1
                
                print(f"    - Sample distribution:")
                for category, count in categories.items():
                    percentage = (count / len(samples)) * 100 if len(samples) > 0 else 0
                    print(f"      - {category}: {count} ({percentage:.1f}%)")
        except Exception as e:
            print(f"[!] Error saving reservoir samples to MongoDB: {e}")
        
        # Close MongoDB connection
        mongo_client.close()
        print(f"[✓] Batch {batch_id} processed with Reservoir Sampling.")
    except Exception as e:
        print(f"[!] Batch {batch_id} error: {e}")

# 6. Start Stream
query = parsed_df.writeStream \
    .foreachBatch(process_batch) \
    .outputMode("append") \
    .option("checkpointLocation", "/tmp/checkpoint_reservoir_sampling") \
    .trigger(processingTime="30 seconds") \
    .start()

query.awaitTermination()