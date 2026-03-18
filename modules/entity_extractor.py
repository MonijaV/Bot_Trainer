import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.nlu_pipeline import extract_entities

def get_entities(user_text: str, intent: str) -> dict:
    """
    Clean wrapper to extract entities from user text.
    
    Returns:
        {"location": "Mumbai", "date": "tomorrow"}
    """
    result = extract_entities(user_text, intent)
    return result


if __name__ == "__main__":
    tests = [
        ("Book a flight to Mumbai tomorrow", "book_flight"),
        ("Play Kesariya by Arijit Singh", "play_music"),
        ("Remind me to call mom at 9pm", "set_reminder"),
    ]
    for text, intent in tests:
        print(f"Input : {text}")
        print(f"Intent: {intent}")
        print(f"Entities: {get_entities(text, intent)}")
        print("-" * 30)