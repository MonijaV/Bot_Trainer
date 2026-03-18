import json
import os

# This function loads the intents.json file
def load_intents(filepath="data/intents.json"):
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find {filepath}")
    
    # Open and read the JSON file
    with open(filepath, "r") as f:
        data = json.load(f)
    
    return data

# This function loads the evaluation dataset
def load_eval_dataset(filepath="data/eval_dataset.json"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find {filepath}")
    
    with open(filepath, "r") as f:
        data = json.load(f)
    
    return data["test_samples"]

# This function returns just the list of intent names
def get_intent_names(intents_data):
    return [intent["name"] for intent in intents_data["intents"]]

# This function returns examples for a specific intent
def get_intent_examples(intents_data, intent_name):
    for intent in intents_data["intents"]:
        if intent["name"] == intent_name:
            return intent["examples"]
    return []

# Quick test — run this file directly to check everything loads
if __name__ == "__main__":
    intents = load_intents()
    print("Intents loaded successfully!")
    print("Intent names:", get_intent_names(intents))
    
    eval_data = load_eval_dataset()
    print(f"\nEval dataset loaded! Total samples: {len(eval_data)}")
    print("First sample:", eval_data[0])