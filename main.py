import sys
import json
import os
from typing import Optional, Any, Dict, List
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

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


# ==========================================
# TASK 2: Expose CRUD functions as LangChain @tool
# ==========================================

@tool
def create_student(student_id: Any, name: str, age: Optional[int] = None, skills: Optional[List[str]] = None) -> str:
    """Creates/adds a new student record to students.json. Requires student_id and name."""
    global students_data
    
    # Check for existing duplicate ID
    if any(str(s.get("id")) == str(student_id) for s in students_data):
        return f"Error: Student with ID {student_id} already exists."
    
    new_student = {
        "id": student_id,
        "name": name
    }
    if age is not None:
        new_student["age"] = age
    if skills is not None:
        new_student["skills"] = skills
        
    students_data.append(new_student)
    save_students(students_data)
    return f"Successfully added student: {new_student}"

@tool
def read_student(student_id: Optional[Any] = None) -> str:
    """Reads or searches student details. Pass student_id to get a specific student, or leave empty to list all."""
    global students_data
    if student_id is not None:
        matches = [s for s in students_data if str(s.get("id")).lower() == str(student_id).lower()]
        return json.dumps(matches, indent=2) if matches else f"Student with ID '{student_id}' not found."
    return json.dumps(students_data, indent=2)

@tool
def update_student(student_id: Any, name: Optional[str] = None, age: Optional[int] = None, skills: Optional[List[str]] = None) -> str:
    """Updates an existing student's details in students.json using their student_id."""
    global students_data
    for s in students_data:
        if str(s.get("id")) == str(student_id):
            if name is not None:
                s["name"] = name
            if age is not None:
                s["age"] = age
            if skills is not None:
                s["skills"] = skills
            save_students(students_data)
            return f"Successfully updated student ID {student_id}: {s}"
    return f"Student ID {student_id} not found."

@tool
def delete_student(student_id: Any) -> str:
    """Deletes a student record from students.json by student_id."""
    global students_data
    initial_length = len(students_data)
    students_data = [s for s in students_data if str(s.get("id")) != str(student_id)]
    
    if len(students_data) < initial_length:
        save_students(students_data)
        return f"Successfully deleted student ID {student_id}."
    return f"Student ID {student_id} not found."


# Package all tools into a list
tools = [create_student, read_student, update_student, delete_student]


# ==========================================
# TASK 3: Connect tools to Gemini and execute
# ==========================================

# 1. Initialize Gemini Model
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.0)

# 2. Bind the tools to the Gemini model
llm_with_tools = llm.bind_tools(tools)

def run_agent(user_input: str):
    """Passes user query to model, inspects tool choices, and executes the selected tool function."""
    messages = [
        SystemMessage(content="You are a helpful student database manager. Use the provided tools to execute student CRUD operations."),
        HumanMessage(content=user_input)
    ]
    
    # Let Gemini decide which tool to call and with what arguments
    ai_msg = llm_with_tools.invoke(messages)
    
    # If Gemini decided to call a tool:
    if ai_msg.tool_calls:
        print("\n--- Tool Calls ---")
        tool_map = {
            "create_student": create_student,
            "read_student": read_student,
            "update_student": update_student,
            "delete_student": delete_student
        }
        
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"Tool Name: {tool_name}")
            print(f"Arguments: {tool_args}")
            
            # Execute the tool function pythonically
            selected_tool = tool_map[tool_name]
            result = selected_tool.invoke(tool_args)
            print("\n---  Output ---")
            print(result)
    else:
        print("\n--- Direct Response (No Tool Call Needed) ---")
        print(ai_msg.content)


if __name__ == "__main__":
    user_input = input("Enter CRUD command: ")

    if not user_input.strip():
        user_input = "Add a new student named Rahul with ID 105 who knows Python and FastAPI."

    print(f"\nUser Command: '{user_input}'")
    run_agent(user_input)