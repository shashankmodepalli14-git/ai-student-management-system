import sys
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables (.env file with GEMINI_API_KEY)
load_dotenv()

# Initialize client
client = genai.Client()

def load_students(file_path: str = "students.json"):
    """Loads student records from a local JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'. Make sure it exists in the same folder.")
        sys.exit(1)

# Load student records from JSON
students_data = load_students("students.json")

# Formulate system prompt including the JSON database context
SYSTEM_INSTRUCTION = f"""
You are an AI Student Management Assistant. 
You have access to the following student database (JSON format):

{json.dumps(students_data, indent=2)}

Guidelines:
1. Answer the user's questions accurately using ONLY the student information provided in the database above.
2. Maintain a helpful, concise, and professional tone.
3. If asked about a student who isn't in the database or skills/details not listed, state clearly that the information is not found in the database.
"""

def ask_llm(user_input: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_input,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,  # Low temperature keeps answers factual based on JSON
        ),
    )
    return response.text


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        user_input = input("Enter your question about students: ")

    output = ask_llm(user_input)

    print("\n--- LLM Response ---")
    print(output)