from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import explode, col
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

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

# --------------------------------
# --- Explainable AI Analysis ---
# --------------------------------

# Extract the LogisticRegression model from the pipeline
lr_model = model.stages[-1]

# Get feature importances (coefficients)
coefficients = lr_model.coefficients.toArray()

# Create a list of (feature_index, coefficient_value) tuples
feature_importances = [(i, float(coef)) for i, coef in enumerate(coefficients)]

# Sort by absolute coefficient value to find most important features
top_features = sorted(feature_importances, key=lambda x: abs(x[1]), reverse=True)[:30]

# Print the top features
print("\n===== TOP FEATURES INFLUENCING HALLUCINATION PREDICTION =====")
print("Feature Index | Coefficient | Influence")
print("-" * 50)
for idx, weight in top_features:
    influence = "Increases hallucination likelihood" if weight > 0 else "Decreases hallucination likelihood"
    print(f"{idx:12d} | {weight:+10.4f} | {influence}")

os.makedirs("explainable_ai_output", exist_ok=True)

# Save feature importance to CSV
feature_df = pd.DataFrame(top_features, columns=["feature_index", "importance"])
feature_df["influence"] = feature_df["importance"].apply(
    lambda x: "Increases hallucination likelihood" if x > 0 else "Decreases hallucination likelihood"
)
feature_df.to_csv("explainable_ai_output/feature_importance.csv", index=False)

print("\n\n===== ATTEMPTING TO MAP FEATURES TO WORDS =====")
print("Note: Due to hashing collisions, multiple words may map to the same feature")

# Get unique words
tokenized_data = tokenizer.transform(df)
words_df = tokenized_data.select(explode(col("words")).alias("word")).distinct()
words = [row.word for row in words_df.collect()]

# Create word-to-feature mapping
feature_to_words = {}
for word in words:
    # Simplified hash function - approximation of HashingTF's murmur hash
    # Note: This is an approximation and may not exactly match
    hash_value = abs(hash(word)) % 10000
    if hash_value not in feature_to_words:
        feature_to_words[hash_value] = []
    feature_to_words[hash_value].append(word)

# Print possible words for top features
print("\nTop Features with Possible Associated Words:")
print("-" * 70)
word_mappings = []
for idx, weight in top_features:
    possible_words = feature_to_words.get(idx, ["Unknown (hash collision)"])
    influence = "Increases hallucination likelihood" if weight > 0 else "Decreases hallucination likelihood"
    print(f"Feature {idx}: {weight:+.4f} - {influence}")
    print(f"  Possible words: {', '.join(possible_words[:10])}")
    if len(possible_words) > 10:
        print(f"  ...and {len(possible_words)-10} more words")
    
    # Store for CSV
    word_mappings.append({
        "feature_index": idx,
        "coefficient": weight,
        "influence": influence,
        "possible_words": "|".join(possible_words[:20])
    })

# Save word mappings to CSV
word_df = pd.DataFrame(word_mappings)
word_df.to_csv("explainable_ai_output/feature_word_mapping.csv", index=False)

# --- Visualize feature importances ---

# Plot positive vs negative coefficients
pos_features = [(i, w) for i, w in top_features if w > 0]
neg_features = [(i, w) for i, w in top_features if w < 0]

plt.figure(figsize=(12, 10))

# Plot positive coefficients (features that predict hallucination)
plt.subplot(2, 1, 1)
if pos_features:
    pos_indices = [i for i, _ in pos_features]
    pos_values = [w for _, w in pos_features]
    plt.bar(range(len(pos_features)), pos_values, color='red')
    plt.xticks(range(len(pos_features)), pos_indices, rotation=90)
    plt.title('Features That Increase Hallucination Likelihood')
    plt.ylabel('Coefficient Value')

# Plot negative coefficients (features that predict non-hallucination)
plt.subplot(2, 1, 2)
if neg_features:
    neg_indices = [i for i, _ in neg_features]
    neg_values = [w for _, w in neg_features]
    plt.bar(range(len(neg_features)), neg_values, color='blue')
    plt.xticks(range(len(neg_features)), neg_indices, rotation=90)
    plt.title('Features That Decrease Hallucination Likelihood')
    plt.ylabel('Coefficient Value')

plt.tight_layout()
plt.savefig("explainable_ai_output/feature_importance_chart.png")

plt.figure(figsize=(14, 8))
indices = [i for i, _ in top_features]
values = [abs(w) for _, w in top_features]
colors = ['red' if w > 0 else 'blue' for _, w in top_features]

plt.bar(range(len(top_features)), values, color=colors)
plt.xticks(range(len(top_features)), indices, rotation=90)
plt.title('Top Features by Absolute Importance')
plt.xlabel('Feature Index')
plt.ylabel('Absolute Coefficient Value')
plt.legend(['Increases Hallucination', 'Decreases Hallucination'])
plt.tight_layout()
plt.savefig("explainable_ai_output/absolute_feature_importance.png")

# --- Generate feature importance report ---
with open("explainable_ai_output/feature_importance_report.txt", "w") as f:
    f.write("HALLUCINATION DETECTION MODEL - FEATURE IMPORTANCE ANALYSIS\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Model Accuracy: {accuracy:.4f}\n\n")
    
    f.write("TOP FEATURES INCREASING HALLUCINATION LIKELIHOOD:\n")
    f.write("-" * 60 + "\n")
    for idx, weight in [(i, w) for i, w in top_features if w > 0][:15]:
        possible_words = feature_to_words.get(idx, ["Unknown"])[:5]
        f.write(f"Feature {idx}: {weight:+.4f} - Possible words: {', '.join(possible_words)}\n")
    
    f.write("\nTOP FEATURES DECREASING HALLUCINATION LIKELIHOOD:\n")
    f.write("-" * 60 + "\n")
    for idx, weight in [(i, w) for i, w in top_features if w < 0][:15]:
        possible_words = feature_to_words.get(idx, ["Unknown"])[:5]
        f.write(f"Feature {idx}: {weight:+.4f} - Possible words: {', '.join(possible_words)}\n")
    
    f.write("\nNOTE: Due to hash collisions in the feature extraction process, ")
    f.write("word-to-feature mappings are approximate and may not be exact.\n\n")
    
    f.write("INTERPRETATION GUIDE:\n")
    f.write("-" * 60 + "\n")
    f.write("Positive coefficients: Features that increase the likelihood of classifying a claim as hallucinating\n")
    f.write("Negative coefficients: Features that decrease the likelihood of classifying a claim as hallucinating\n")
    f.write("The magnitude (absolute value) indicates how strongly the feature influences the prediction\n")

print("\n===== EXPLAINABLE AI ANALYSIS COMPLETE =====")
print(f"Results saved to the 'explainable_ai_output' directory")
