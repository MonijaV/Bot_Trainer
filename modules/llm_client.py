import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from groq import Groq
from dotenv import load_dotenv

# Load the API key from .env file
load_dotenv()

# Create the Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def call_llm(prompt: str) -> str:
    """
    Sends a prompt to Groq LLM and returns the response text.
    
    Args:
        prompt: The full prompt string to send
    
    Returns:
        The LLM's response as a string
    """
    try:
        response = client.chat.completions.create(
           model="llama-3.3-70b-versatile",  # Current free Groq model
            messages=[
                {
                    "role": "system",
                    "content": "You are an NLU engine. You always respond with valid JSON only. No extra text, no explanation, no markdown, no code blocks. Just raw JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,   # Low temperature = more consistent outputs
            max_tokens=500
        )
        
        # Extract the text response
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"LLM API error: {e}")
        return None
# Quick test — run this file directly
if __name__ == "__main__":
    test_prompt = 'Return this exact JSON: {"status": "connected", "model": "llama3-8b-8192"}'
    result = call_llm(test_prompt)
    print("LLM Response:", result)
    print("Connection successful!" if result else "Connection failed!")