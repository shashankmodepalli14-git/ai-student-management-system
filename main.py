import sys
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env file
load_dotenv()

# Initialize client
client = genai.Client()

def ask_llm(user_input):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_input,
    )
    return response.text
    

user_input = input("Enter your prompt: ")

output = ask_llm(user_input)

print("\n--- LLM Response ---")
print(output)