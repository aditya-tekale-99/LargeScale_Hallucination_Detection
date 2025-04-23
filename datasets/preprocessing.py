import json
import spacy
from tqdm import tqdm

# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")

input_file = 'train.jsonl'
output_file = 'fever_preprocessed.jsonl'

def extract_entities(text):
    doc = nlp(text)
    return list(set(ent.text for ent in doc.ents))

def redact_entities(claim, entities):
    for entity in sorted(entities, key=len, reverse=True):
        claim = claim.replace(entity, "<ENTITY>")
    return claim

count = 0
with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
    for line in tqdm(f_in, desc="Processing entries"):
        entry = json.loads(line)

        # Check for required fields
        if 'claim' not in entry or 'label' not in entry:
            continue

        # Convert label to binary
        label = entry['label']
        if label == "SUPPORTS":
            binary_label = 0
        elif label in ["REFUTES", "NOT ENOUGH INFO"]:
            binary_label = 1
        else:
            continue  # Skip unknown labels

        # Extract and redact entities
        claim = entry['claim']
        entities = extract_entities(claim)
        redacted_claim = redact_entities(claim, entities)

        # Save only necessary fields
        cleaned_entry = {
            "claim": claim,
            "entity_redacted_claim": redacted_claim,
            "label": binary_label
        }

        f_out.write(json.dumps(cleaned_entry) + '\n')
        count += 1

print(f"Saved {count} preprocessed entries to {output_file}")
