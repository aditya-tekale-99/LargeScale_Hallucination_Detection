import json
import time
from kafka import KafkaProducer
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load multilingual NER model
tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Kafka setup
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

topic = 'fever-stream'
input_file = 'fever_cleaned.jsonl'

i = 0

# Start streaming
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        if i == 1000:
            break
        try:
            entry = json.loads(line)
            claim = entry['claim']
            entities = ner_pipeline(claim)
            entity_list = list(set(ent['word'] for ent in entities))

            enriched_entry = {
                'id': entry['id'],
                'claim': claim,
                'label': entry['label'],
                'entities': entity_list
            }

            producer.send(topic, enriched_entry)
            print(f"Sent: {enriched_entry}")
            i += 1
            time.sleep(1)  # Simulate streaming delay

        except Exception as e:
            print(f"Error: {e}")

producer.flush()
producer.close()
