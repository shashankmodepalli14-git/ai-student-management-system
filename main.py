import sys
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# 1. Load environment variables (.env file containing GEMINI_API_KEY)
load_dotenv()

# 2. Helper function to load student dataset from JSON
def load_students(file_path: str = "students.json"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'. Make sure it exists in the project root.")
        sys.exit(1)

# Load student records
students_data = load_students("students.json")

# 3. Initialize Gemini through LangChain
# LangChain automatically looks for GEMINI_API_KEY in environment variables
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

# 4. Construct System Message with student database context
system_instruction = f"""
You are an AI Student Management Assistant powered by LangChain.
You have access to the following student database (JSON format):

{json.dumps(students_data, indent=2)}

Guidelines:
1. Answer the user's questions accurately using ONLY the student information provided in the database above.
2. Maintain a professional and concise tone.
3. If asked about information not present in the database, explicitly state that it was not found.
"""

def ask_llm(user_input: str) -> str:
    """Sends messages array (System + Human) to Gemini via LangChain."""
    messages = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=user_input)
    ]
    
    # LangChain invoke returns an AIMessage object; .content gives the text string
    response = llm.invoke(messages)
    return response.content


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        user_input = input("Enter your question about students: ")

    if not user_input.strip():
        user_input = "Show me all students who know Python."

    print(f"\nPrompt: {user_input}")
    
    # Run query through LangChain model
    output = ask_llm(user_input)

    print("\n--- LangChain Response ---")
    print(output)