from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, current_timestamp, udf, lit, array, collect_list
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, ArrayType, IntegerType, MapType
import time
import json
import os
from typing import List, Dict
from collections import defaultdict
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
    .appName("KafkaToMongoDB_DGIM") \
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

# 3. DGIM Algorithm Implementation
class DGIMBucket:
    def __init__(self, size, timestamp):
        self.size = size  # Size of bucket (powers of 2)
        self.timestamp = timestamp  # When this bucket was created

    def __repr__(self):
        return f"Bucket(size={self.size}, timestamp={self.timestamp})"

class DGIM:
    def __init__(self, window_size):
        self.window_size = window_size
        self.current_timestamp = 0
        self.buckets = defaultdict(list)  # Dictionary mapping bucket sizes to lists of buckets
        self.total_ones = 0

    def add_bit(self, bit):
        """Process a new bit in the stream"""
        self.current_timestamp += 1
        
        # Remove expired buckets (outside the window)
        self._remove_expired()
        
        # If the bit is 1, create a new bucket of size 1
        if bit == 1:
            self.buckets[1].append(DGIMBucket(1, self.current_timestamp))
            self.total_ones += 1
        
        # Merge buckets if necessary (maintain at most 2 buckets of each size)
        self._merge_buckets()

    def _remove_expired(self):
        """Remove buckets that are outside the sliding window"""
        expired_timestamp = self.current_timestamp - self.window_size
        
        for size in list(self.buckets.keys()):
            # Remove buckets with timestamps outside the window
            original_count = len(self.buckets[size])
            self.buckets[size] = [b for b in self.buckets[size] if b.timestamp > expired_timestamp]
            removed_count = original_count - len(self.buckets[size])
            self.total_ones -= removed_count * size
            
            # Remove empty bucket lists
            if not self.buckets[size]:
                del self.buckets[size]

    def _merge_buckets(self):
        """Merge buckets to maintain at most 2 buckets of each size"""
        for size in sorted(self.buckets.keys()):
            # If we have more than 2 buckets of this size
            while len(self.buckets[size]) > 2:
                # Take the two oldest buckets
                oldest = self.buckets[size].pop(0)
                second_oldest = self.buckets[size].pop(0)
                
                # Create a new bucket of twice the size
                new_size = size * 2
                new_timestamp = second_oldest.timestamp  # Use the newer timestamp
                self.buckets[new_size].append(DGIMBucket(new_size, new_timestamp))

    def count_ones(self):
        """Estimate the number of 1's in the current window"""
        if not self.buckets:
            return 0
            
        total = 0
        # Add all complete buckets
        for size, bucket_list in self.buckets.items():
            # Add all buckets except possibly the last one
            total += size * len(bucket_list)
            
        return total
        
    def to_dict(self):
        """Convert the DGIM state to a serializable dictionary"""
        result = {
            "window_size": self.window_size,
            "current_timestamp": self.current_timestamp,
            "buckets": {str(size): [{"size": b.size, "timestamp": b.timestamp} for b in buckets] 
                         for size, buckets in self.buckets.items()},
            "total_ones": self.total_ones
        }
        return result
        
    @classmethod
    def from_dict(cls, data):
        """Reconstruct a DGIM object from a dictionary"""
        dgim = cls(data["window_size"])
        dgim.current_timestamp = data["current_timestamp"]
        dgim.total_ones = data["total_ones"]
        
        for size_str, bucket_list in data["buckets"].items():
            size = int(size_str)
            dgim.buckets[size] = [DGIMBucket(b["size"], b["timestamp"]) for b in bucket_list]
            
        return dgim

# Define function to serialize DGIM state
def serialize_dgim(dgim):
    """Serialize DGIM state to JSON string"""
    if dgim is None:
        return "{}"
    return json.dumps(dgim.to_dict())

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
    .withColumn("timestamp", current_timestamp()) \
    .withColumn("is_positive", (col("predicted_label") >= 0.5).cast("integer"))  # Convert to binary for DGIM

# 5. Batch Writer with DGIM processing
def process_with_dgim(batch_df, batch_id):
    try:
        # Initialize MongoDB client
        mongo_client = get_mongo_client()
        db = mongo_client["hallucination_detection"]  # Database name
        dgim_state_collection = db["dgim_state"]  # Collection for state
        dgim_stats_collection = db["dgim_stats"]  # Collection for statistics
        raw_data_collection = db["raw_data_sample"]  # Collection for raw data samples
        
        # Process the batch without collecting all data at once
        # This avoids memory issues by processing one row at a time
        row_count = 0
        positive_count = 0
        sample_rows = []  # Store a small sample of rows
        
        # Try to load existing DGIM state from MongoDB
        try:
            latest_state = dgim_state_collection.find_one(
                sort=[("updated_at", -1)]  # Get the most recent state
            )
            
            if latest_state and "dgim_state" in latest_state:
                # Load the existing state
                dgim = DGIM.from_dict(json.loads(latest_state["dgim_state"]))
                print(f"[✓] Loaded existing DGIM state from MongoDB.")
            else:
                # Initialize a new DGIM instance
                window_size = 1000  # Configure your sliding window size
                dgim = DGIM(window_size)
                print(f"[i] Initialized new DGIM with window size {window_size}.")
        except Exception as e:
            # If there's an error, start with a fresh DGIM instance
            window_size = 1000  # Configure your sliding window size
            dgim = DGIM(window_size)
            print(f"[!] Error loading DGIM state: {e}")
            print(f"[i] Initialized new DGIM with window size {window_size}.")
        
        # Use a more memory-efficient row iterator
        for row in batch_df.toLocalIterator():
            row_count += 1
            
            # Ensure we're processing an integer bit
            bit_value = 1 if row["is_positive"] == 1 else 0
            if bit_value == 1:
                positive_count += 1
            
            # Add to DGIM
            dgim.add_bit(bit_value)
            
            # Keep a small sample of rows for raw data storage
            if row_count <= 5:  # Only save first 5 rows as sample
                row_dict = row.asDict()
                # Convert timestamp to string for MongoDB compatibility
                if "timestamp" in row_dict and hasattr(row_dict["timestamp"], "strftime"):
                    row_dict["timestamp"] = row_dict["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                sample_rows.append(row_dict)
            
        if row_count == 0:
            print(f"[i] Batch {batch_id} is empty, skipping DGIM processing.")
            mongo_client.close()
            return
            
        print(f"[i] Batch {batch_id} contains {row_count} records. Actual positives: {positive_count}")
        print(f"[i] DGIM approximation of positives: {dgim.count_ones()}")
        
        # Save DGIM state to MongoDB
        try:
            dgim_state_json = serialize_dgim(dgim)
            dgim_state_record = {
                "dgim_state": dgim_state_json,
                "updated_at": int(time.time()),
                "batch_id": batch_id
            }
            
            # Insert or update DGIM state
            dgim_state_collection.insert_one(dgim_state_record)
            print(f"[✓] Successfully saved DGIM state to MongoDB.")
        except Exception as e:
            print(f"[!] Error saving DGIM state to MongoDB: {e}")
        
        # Save statistics to MongoDB
        try:
            error_percentage = abs(dgim.count_ones() - positive_count) / positive_count * 100 if positive_count > 0 else 0.0
            
            stats_record = {
                "batch_id": batch_id,
                "stream_position": dgim.current_timestamp,
                "window_size": dgim.window_size,
                "approximate_count": dgim.count_ones(),
                "actual_count": positive_count,
                "batch_size": row_count,
                "positive_ratio": float(dgim.count_ones()) / dgim.window_size if dgim.window_size > 0 else 0.0,
                "error_percentage": error_percentage,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Insert statistics
            dgim_stats_collection.insert_one(stats_record)
            print(f"[✓] Successfully saved DGIM statistics to MongoDB.")
            
            # Print the statistics
            print(f"[✓] DGIM Algorithm Results for Batch {batch_id}:")
            print(f"    - Stream position: {stats_record['stream_position']}")
            print(f"    - Window size: {stats_record['window_size']}")
            print(f"    - Approximate positive count: {stats_record['approximate_count']}")
            print(f"    - Actual positive count: {stats_record['actual_count']}")
            print(f"    - Batch size: {stats_record['batch_size']}")
            print(f"    - Positive ratio: {stats_record['positive_ratio']:.4f}")
            print(f"    - Approximation error: {stats_record['error_percentage']:.2f}%")
        except Exception as e:
            print(f"[!] Error saving DGIM statistics to MongoDB: {e}")
        
        # Save raw data sample to MongoDB
        if sample_rows:
            try:
                # Insert raw data samples
                raw_data_collection.insert_many(sample_rows)
                print(f"[✓] Successfully saved {len(sample_rows)} sample records to MongoDB.")
            except Exception as e:
                print(f"[!] Error saving raw data samples to MongoDB: {e}")
        
        # Close MongoDB connection
        mongo_client.close()
        print(f"[✓] Batch {batch_id} processed with DGIM and sent to MongoDB.")
    except Exception as e:
        print(f"[!] Batch {batch_id} error: {e}")

# 6. Start Stream
query = parsed_df.writeStream \
    .foreachBatch(process_with_dgim) \
    .outputMode("append") \
    .option("checkpointLocation", "/tmp/checkpoint_dgim_mongodb") \
    .trigger(processingTime="30 seconds") \
    .start()

query.awaitTermination()