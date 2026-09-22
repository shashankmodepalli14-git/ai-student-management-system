import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

# Initialize client
client = genai.Client()

def ask_llm(user_input):
    SYSTEM_INSTRUCTION = """
        You are an AI Student Management Assistant. Your task is to help track, organize, and manage student information.
        Follow these behavior guidelines:
        1. Be helpful, concise, and accurate when answering queries regarding students.
        2. Maintain a professional and administrative tone.
        3. If asked about student data, present details (like student_id, name, skills, bio) clearly.
        4. If a query is unrelated to student management or academic topics, politely remind the user of your core purpose.
        """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_input,
        config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.3,  # Lower temperature keeps answers focused and deterministic
            ),
    )
    return response.text
    

user_input = input("Enter your prompt: ")

output = ask_llm(user_input)

print("\n--- LLM Response ---")
print(output)