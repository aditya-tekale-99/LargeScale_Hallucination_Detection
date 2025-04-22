import time
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

topic = 'redacted-claims'

test_messages = [
    {"claim": "Entity invented the telephone.", "label": "SUPPORTED", "entity_redacted_claim": "<ENTITY> invented the telephone."},
    {"claim": "The Moon is made of cheese.", "label": "REFUTES", "entity_redacted_claim": "The <ENTITY> is made of cheese."},
    {"claim": "Stranger Things is a TV series.", "label": "SUPPORTED", "entity_redacted_claim": "<ENTITY> <ENTITY> is a TV series."}
]

while True:
    for message in test_messages:
        producer.send(topic, value=message)
        print(f"Sent: {message}")
        time.sleep(1)