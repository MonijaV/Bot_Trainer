import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.llm_client import call_llm

def load_prompt(filename: str) -> str:
    """Loads a prompt template from the prompts/ folder."""
    filepath = os.path.join("prompts", filename)
    with open(filepath, "r") as f:
        return f.read()

def parse_json_response(response: str) -> dict:
    """
    Safely parses JSON from LLM response.
    Handles cases where LLM adds extra text accidentally.
    """
    if not response:
        return {}
    
    try:
        # First try: direct parse
        return json.loads(response)
    except json.JSONDecodeError:
        # Second try: find JSON inside the response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end != 0:
            try:
                return json.loads(response[start:end])
            except json.JSONDecodeError:
                pass
    
    # If all fails return empty dict
    return {}

def classify_intent(user_text: str) -> dict:
    """
    Sends user text to LLM and returns intent + confidence.
    
    Returns: {"intent": "book_flight", "confidence": 0.95}
    """
    # Load the intent prompt template
    prompt_template = load_prompt("intent_prompt.txt")
    
    # Fill in the user text
    prompt = prompt_template.replace("{user_text}", user_text)
    
    # Call the LLM
    response = call_llm(prompt)
    
    # Parse the JSON response
    result = parse_json_response(response)
    
    # Validate the result has required keys
    if "intent" not in result:
        result["intent"] = "unknown"
    if "confidence" not in result:
        result["confidence"] = 0.0
    
    return result

def extract_entities(user_text: str, intent: str) -> dict:
    """
    Sends user text + intent to LLM and returns extracted entities.
    
    Returns: {"location": "Delhi", "date": "tomorrow"}
    """
    # Load the entity prompt template
    prompt_template = load_prompt("entity_prompt.txt")
    
    # Fill in the user text and intent
    prompt = prompt_template.replace("{user_text}", user_text)
    prompt = prompt.replace("{intent}", intent)
    
    # Call the LLM
    response = call_llm(prompt)
    
    # Parse and return
    return parse_json_response(response)

def predict(user_text: str) -> dict:
    """
    Main function — takes raw user text and returns full NLU output.
    This is the function all other parts of the app will call.
    
    Returns:
    {
        "user_text": "Book a flight to Delhi tomorrow",
        "intent": "book_flight",
        "confidence": 0.95,
        "entities": {"location": "Delhi", "date": "tomorrow"}
    }
    """
    print(f"\nProcessing: '{user_text}'")
    
    # Step 1: Classify intent
    intent_result = classify_intent(user_text)
    intent = intent_result.get("intent", "unknown")
    confidence = intent_result.get("confidence", 0.0)
    
    print(f"Intent detected: {intent} (confidence: {confidence})")
    
    # Step 2: Extract entities
    entities = extract_entities(user_text, intent)
    
    print(f"Entities found: {entities}")
    
    # Step 3: Return full result
    return {
        "user_text": user_text,
        "intent": intent,
        "confidence": confidence,
        "entities": entities
    }


# Quick test — run this file directly
if __name__ == "__main__":
    test_inputs = [
        "Book a flight to Delhi tomorrow",
        "Order me a pizza",
        "What is the weather in Chennai today?",
        "Hello there",
        "Bye"
    ]
    
    print("=" * 50)
    print("NLU PIPELINE TEST")
    print("=" * 50)
    
    for text in test_inputs:
        result = predict(text)
        print("\nFINAL RESULT:")
        print(json.dumps(result, indent=2))
        print("-" * 40)