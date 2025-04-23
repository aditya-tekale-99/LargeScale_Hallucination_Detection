import json
import re
import time
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
import numpy as np
from kafka import KafkaConsumer


# --- Ensure stopwords are available ---
nltk.download('stopwords')
from nltk.corpus import stopwords

# --- Config ---
WORDCLOUD_PATH = "wordcloud.png"
MAX_WORDS = 100
DP_NOISE_SCALE = 1.0  # Laplace noise scale

# --- Load Latest Reservoir Sample from Kafka ---
def get_latest_reservoir():
    consumer = KafkaConsumer(
        "reservoir-sample",
        bootstrap_servers='localhost:9092',
        auto_offset_reset='latest',
        enable_auto_commit=False,
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        consumer_timeout_ms=1000
    )

    try:
        messages = list(consumer)
        if not messages:
            print("[!] No new reservoir sample found.")
            return []
        return messages[-1].value  # latest batch
    except Exception as e:
        print(f"[✗] Error reading from Kafka: {e}")
        return []
    finally:
        consumer.close()

# --- Tokenize & Clean ---
def extract_words(text):
    stop_words = set(stopwords.words('english'))
    tokens = re.findall(r'\[REDACTED\]|\w+', text.lower())
    return [
        word for word in tokens
        if word not in stop_words and word != "[redacted]" and "redacted" not in word
    ]

# --- Word Cloud Generator with DP ---
def generate_wordcloud():
    reservoir = get_latest_reservoir()
    if not reservoir:
        return

    # Extract and count words
    all_words = []
    for entry in reservoir:
        if entry.get("label") == "REFUTES":
            all_words.extend(extract_words(entry.get("claim", "")))

    raw_word_freq = Counter(all_words)

    # Apply DP
    dp_word_freq = {
        word: max(0, count + np.random.laplace(loc=0, scale=DP_NOISE_SCALE))
        for word, count in raw_word_freq.items()
    }

    top_words = dict(Counter(dp_word_freq).most_common(MAX_WORDS))
    if not top_words:
        print("[!] No words to display.")
        return

    # Generate and save word cloud
    wc = WordCloud(width=800, height=400, background_color='white')
    wc.generate_from_frequencies(top_words)
    wc.to_file(WORDCLOUD_PATH)
    print(f"[✓] Word cloud saved to {WORDCLOUD_PATH}")

# --- Run in Loop ---
if __name__ == "__main__":
    while True:
        generate_wordcloud()
        time.sleep(5)  # every 5s
