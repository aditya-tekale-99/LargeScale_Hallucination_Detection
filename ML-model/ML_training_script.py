from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder \
    .appName("TrainClaimClassifier") \
    .getOrCreate()

df = spark.read.json("/Users/adi/Downloads/SJSU/Sem 2/DATA 228/Final Project/LargeScale_Hallucination_Detection/datasets/fever_preprocessed.jsonl")

tokenizer = Tokenizer(inputCol="entity_redacted_claim", outputCol="words")
tf = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="features")

lr = LogisticRegression(featuresCol="features", labelCol="label")

pipeline = Pipeline(stages=[tokenizer, tf, idf, lr])

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

model = pipeline.fit(train_df)

predictions = model.transform(test_df)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy: {accuracy:.4f}")

model.write().overwrite().save("models/claim_classifier")
