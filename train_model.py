import json
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load FEVER dataset (no entities field)
input_file = 'fever_cleaned.jsonl'

claims = []
labels = []

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        if 'claim' in entry and 'label' in entry:
            claims.append(entry['claim'])
            labels.append(entry['label'])

print(f"Loaded {len(claims)} entries.")

# Step 2: Apply Simple Regex-Based Entity Redaction
def simple_redact(claim):
    # Replace capitalized words (likely proper nouns) with <ENTITY>
    return re.sub(r'\b[A-Z][a-z]+\b', '<ENTITY>', claim)

redacted_claims = [simple_redact(c) for c in claims]

# Step 3: Prepare DataFrame
df = pd.DataFrame({
    'entity_redacted_claim': redacted_claims,
    'label': labels
})

print(df.head())

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['entity_redacted_claim'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# Step 5: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 6: Train Logistic Regression Model
model = LogisticRegression(max_iter=500)
model.fit(X_train_tfidf, y_train)

# Step 7: Evaluate Model
y_pred = model.predict(X_test_tfidf)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Accuracy Score ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Step 8: Save Model and Vectorizer
import joblib
joblib.dump(model, 'logistic_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("\nModel and vectorizer saved!")
