import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.nlu_pipeline import classify_intent

def get_intent(user_text: str) -> dict:
    """
    Clean wrapper to get intent from user text.
    
    Returns:
        {"intent": "book_flight", "confidence": 0.97}
    """
    result = classify_intent(user_text)
    return result


if __name__ == "__main__":
    tests = [
        "Book a flight to Mumbai",
        "Play some music",
        "What is my balance?",
    ]
    for t in tests:
        print(f"Input: {t}")
        print(f"Result: {get_intent(t)}")
        print("-" * 30)