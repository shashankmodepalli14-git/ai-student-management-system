import sys
import json
from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Load environment variables (.env file containing GEMINI_API_KEY)
load_dotenv()

STUDENTS_FILE = "students.json"

# Helper functions to read and write JSON
def load_students(file_path: str = STUDENTS_FILE) -> List[Dict[str, Any]]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_students(data: List[Dict[str, Any]], file_path: str = STUDENTS_FILE):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

students_data = load_students()


# 2. Define the Structured Output format using Pydantic
class CRUDInstruction(BaseModel):
    action: str = Field(description="Action: 'CREATE', 'READ', 'UPDATE', 'DELETE', or 'UNKNOWN'")
    student_id: Optional[Any] = Field(None, description="The ID of the student")
    data: Optional[Dict[str, Any]] = Field(None, description="Student details for CREATE or UPDATE")


# 3. Initialize Gemini with Structured Output
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
structured_llm = llm.with_structured_output(CRUDInstruction)


# 4. Define Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI Database Assistant. Interpret the user's request into a CRUD command.
    
Actions:
- CREATE: Adding a new student. Put student fields into 'data'.
- READ: Viewing or searching student info.
- UPDATE: Changing details of an existing student. Provide 'student_id' and updated 'data'.
- DELETE: Removing a student. Provide 'student_id'.
- UNKNOWN: Request is not related to CRUD.
"""),
    ("human", "{user_input}")
])

# Create the LangChain processing chain
crud_chain = prompt | structured_llm


# 5. Python Database Executor
def execute_crud(instruction: CRUDInstruction) -> str:
    global students_data
    action = instruction.action.upper()

    # --- CREATE ---
    if action == "CREATE":
        new_student = instruction.data or {}
        if instruction.student_id and "id" not in new_student:
            new_student["id"] = instruction.student_id
            
        students_data.append(new_student)
        save_students(students_data)
        return f"Added student: {new_student}"

    # --- READ ---
    elif action == "READ":
        if instruction.student_id is not None:
            matches = [s for s in students_data if str(s.get("id")).lower() == str(instruction.student_id).lower()]
            return json.dumps(matches, indent=2) if matches else " Student not found."
        return json.dumps(students_data, indent=2)

    # --- UPDATE ---
    elif action == "UPDATE":
        if instruction.student_id is None:
            return " Need a student_id to perform update."
        
        for s in students_data:
            if str(s.get("id")) == str(instruction.student_id):
                if instruction.data:
                    s.update(instruction.data)
                save_students(students_data)
                return f" Updated student ID {instruction.student_id}."
        return f" Student ID {instruction.student_id} not found."

    # --- DELETE ---
    elif action == "DELETE":
        if instruction.student_id is None:
            return "Need a student_id to perform delete."
        
        initial_length = len(students_data)
        students_data = [s for s in students_data if str(s.get("id")) != str(instruction.student_id)]
        
        if len(students_data) < initial_length:
            save_students(students_data)
            return f" Deleted student ID {instruction.student_id}."
        return f" Student ID {instruction.student_id} not found."

    else:
        return " Could not understand the CRUD action."


if __name__ == "__main__":
    user_input = input("Enter CRUD command: ")

    if not user_input.strip():
        user_input = "Add a new student named Rahul with ID 105 who knows Python and FastAPI."

    print(f"\nUser Command: '{user_input}'")

    # Step A: Let Gemini parse the command into structured format
    parsed_intent: CRUDInstruction = crud_chain.invoke({"user_input": user_input})
    print(f"\nParsed Action: {parsed_intent.action}")
    print(f"Parsed ID: {parsed_intent.student_id}")
    print(f"Parsed Data: {parsed_intent.data}")

    # Step B: Let Python update the actual file
    output = execute_crud(parsed_intent)
    print("\n--- Result ---")
    print(output)