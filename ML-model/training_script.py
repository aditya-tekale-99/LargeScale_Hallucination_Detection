from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# --- Start Spark ---
spark = SparkSession.builder \
    .appName("TrainClaimClassifier") \
    .getOrCreate()

# --- Load preprocessed FEVER data (must have entity_redacted_claim and label) ---
df = spark.read.json("/Users/adi/Downloads/SJSU/Sem 2/DATA 228/Final Project/LargeScale_Hallucination_Detection/datasets/fever_preprocessed.jsonl")

# --- Tokenization + TF + IDF ---
tokenizer = Tokenizer(inputCol="entity_redacted_claim", outputCol="words")
tf = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="features")

# --- Model ---
lr = LogisticRegression(featuresCol="features", labelCol="label")

# --- Pipeline ---
pipeline = Pipeline(stages=[tokenizer, tf, idf, lr])

# --- Train/test split ---
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# --- Train ---
model = pipeline.fit(train_df)

# --- Evaluate ---
predictions = model.transform(test_df)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy: {accuracy:.4f}")

# --- Save pipeline model ---
model.write().overwrite().save("models/claim_classifier")


# command to run: python3  training_script.py