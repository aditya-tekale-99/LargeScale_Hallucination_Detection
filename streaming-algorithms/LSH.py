from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from datasketch import MinHash, MinHashLSH
import time
import json
import re
import random
from typing import List, Dict
from pymongo import MongoClient
from collections import Counter
import gc

# 1. Schema
schema = StructType([
    StructField("claim", StringType()),
    StructField("entity_redacted_claim", StringType()),
    StructField("label", StringType()),
    StructField("predicted_label", StringType())
])

# 2. Spark Init
spark = SparkSession.builder \
    .appName("KafkaToMongoDB_LSH") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryoserializer.buffer.max", "1024m") \
    .config("spark.sql.shuffle.partitions", "10") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# MongoDB Connection
def get_mongo_client():
    """Create and return a MongoDB client"""
    # Update these connection details
    mongo_uri = "mongodb://localhost:27017/"
    client = MongoClient(mongo_uri)
    return client

# 3. LSH State Manager
class LSHState:
    def __init__(self, threshold=0.7, num_perm=64):
        """Initialize LSH state with parameters"""
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.claims = {}  # claim_id -> claim_text
        self.claim_count = 0
        self.similarity_clusters = {}  # For tracking clusters
        self.pattern_counts = Counter()  # For tracking common patterns
        
    # Utility: Create word-level n-grams
    def get_ngrams(self, tokens, n=3):
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        
    # Process a single claim
    def process_claim(self, claim_text, entity_redacted_text=None):
        """Process a claim through LSH, return similar claims if any"""
        self.claim_count += 1
        claim_id = f"claim_{self.claim_count}"
        
        # Use entity_redacted_text if available, otherwise use claim_text
        text_to_process = entity_redacted_text if entity_redacted_text else claim_text
        
        # Tokenize and generate 3-gram shingles
        tokens = re.sub(r'[^\w\s<>]', '', text_to_process.lower()).split()
        shingles = self.get_ngrams(tokens, n=3)
        
        # Track common n-grams (or patterns)
        for shingle in shingles:
            self.pattern_counts[shingle] += 1
        
        # Create MinHash
        minhash = MinHash(num_perm=self.num_perm)
        for shingle in shingles:
            minhash.update(shingle.encode('utf-8'))
        
        # Query for similar claims
        similar_claim_ids = self.lsh.query(minhash) if self.lsh is not None else []
        
        # Prepare result object
        result = {
            "claim_id": claim_id,
            "claim_text": claim_text,
            "entity_redacted_text": entity_redacted_text,
            "similar_claims": [
                {"id": similar_id, "text": self.claims.get(similar_id, "Unknown claim")} 
                for similar_id in similar_claim_ids
            ]
        }
        
        # Update similarity clusters
        if similar_claim_ids:
            cluster_id = min(similar_claim_ids)  # Use smallest ID as cluster identifier
            if cluster_id not in self.similarity_clusters:
                self.similarity_clusters[cluster_id] = []
            
            # Add this claim to cluster
            self.similarity_clusters[cluster_id].append({
                "id": claim_id,
                "text": claim_text
            })
            
            # Add all similar claims to this cluster too
            for similar_id in similar_claim_ids:
                if similar_id != cluster_id:
                    similar_text = self.claims.get(similar_id, "Unknown claim")
                    if not any(item["id"] == similar_id for item in self.similarity_clusters[cluster_id]):
                        self.similarity_clusters[cluster_id].append({
                            "id": similar_id,
                            "text": similar_text
                        })
        
        # Add to LSH index
        self.lsh.insert(claim_id, minhash)
        self.claims[claim_id] = claim_text
        
        return result if similar_claim_ids else None
    
    def cleanup(self):
        """Clear internal data structures to free memory"""
        self.claims.clear()
        self.similarity_clusters.clear()
        self.pattern_counts.clear()
    
    def get_common_patterns(self, top_n=20):
        """Get the most common patterns/n-grams found in claims"""
        return self.pattern_counts.most_common(top_n)

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

# 5. Batch Writer with LSH processing
def process_batch_lsh(batch_df, batch_id):
    try:
        # Initialize MongoDB client
        mongo_client = get_mongo_client()
        db = mongo_client["hallucination_detection"]  # Database name
        lsh_patterns_collection = db["lsh_patterns"]  # Collection for common patterns
        
        # Stats tracking
        row_count = 0
        positive_claims_count = 0
        similarity_count = 0
        
        # Initialize a new LSH state with reduced parameters
        threshold = 0.7  # LSH similarity threshold
        num_perm = 64    # Reduced from 128 to 64
        lsh_state = LSHState(threshold, num_perm)
        print(f"[i] Initialized new LSH with threshold {threshold} and {num_perm} permutations.")
        
        # Limit the number of claims we process
        max_claims_to_process = 1000
        
        # Get positive claims from the batch with size limitation
        positive_claims = batch_df.filter(col("predicted_label") >= 0.5).limit(max_claims_to_process).collect()
        
        for row in positive_claims:
            row_count += 1
            positive_claims_count += 1
            
            claim_text = row["claim"]
            entity_redacted_text = row["entity_redacted_claim"]
            
            # Process claim through LSH
            result = lsh_state.process_claim(claim_text, entity_redacted_text)
            
            # If similar claims were found, count it
            if result and result["similar_claims"]:
                similarity_count += 1
        
        if row_count == 0:
            print(f"[i] Batch {batch_id} contains no positive claims, skipping LSH processing.")
            mongo_client.close()
            return
            
        print(f"[i] Batch {batch_id} contains {row_count} total records, {positive_claims_count} positive claims.")
        print(f"[i] Found {similarity_count} claims with similarities.")
        
        # Save common patterns to MongoDB
        try:
            # Get common patterns from LSH state
            patterns = lsh_state.get_common_patterns(top_n=50)
            
            if patterns:
                # Format patterns for MongoDB
                pattern_record = {
                    "batch_id": batch_id,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "patterns": [{"pattern": pattern, "count": count} for pattern, count in patterns]
                }
                
                # Clear previous patterns and insert new ones
                #lsh_patterns_collection.delete_many({})
                lsh_patterns_collection.insert_one(pattern_record)
                print(f"[✓] Successfully saved {len(patterns)} common patterns to MongoDB.")
                
                # Print top patterns
                print(f"[✓] Most common patterns in claims:")
                for pattern, count in patterns[:10]:  # Show top 10
                    print(f"    - '{pattern}': {count} occurrences")
            else:
                print(f"[i] No patterns to save.")
        except Exception as e:
            print(f"[!] Error saving common patterns to MongoDB: {e}")
        
        # Clean up LSH state to free memory
        lsh_state.cleanup()
        
        # Close MongoDB connection
        mongo_client.close()
        
        # Force garbage collection
        gc.collect()
        
        print(f"[✓] Batch {batch_id} processed with LSH and sent to MongoDB.")
    except Exception as e:
        print(f"[!] Batch {batch_id} error: {e}")

# 6. Start Stream
query = parsed_df.writeStream \
    .foreachBatch(process_batch_lsh) \
    .outputMode("append") \
    .option("checkpointLocation", "/tmp/checkpoint_lsh") \
    .trigger(processingTime="15 seconds") \
    .start()

query.awaitTermination()