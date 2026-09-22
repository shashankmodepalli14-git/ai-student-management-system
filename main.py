import sys
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env file
load_dotenv()

# Initialize client
client = genai.Client()

if len(sys.argv) > 1:
    prompt = " ".join(sys.argv[1:])
else:
    prompt = input("Enter your prompt: ")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
)

print("\n--- LLM Response ---")
print(response.text)