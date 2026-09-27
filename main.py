import sys
import json
import os
import uuid
from typing import Optional, Any, Dict, List
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, BaseMessage

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

# Normalization helper so IDs match regardless of type (int/str) or casing
def normalize_id(val: Any) -> str:
    return str(val).strip().lower() if val is not None else ""

# Helper to cleanly extract plain text from Gemini content blocks
def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts)
    return str(content)

students_data = load_students()


# ==========================================
# TASK 2: CRUD Functions as LangChain @tool
# ==========================================

@tool
def create_student(student_id: Any, name: str, age: Optional[int] = None, skills: Optional[List[str]] = None) -> str:
    """Creates/adds a new student record to students.json. Requires student_id and name."""
    global students_data
    
    target_id = normalize_id(student_id)
    if any(normalize_id(s.get("id")) == target_id for s in students_data):
        return f"Error: Student with ID {student_id} already exists."
    
    new_student = {"id": student_id, "name": name}
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
    if student_id is not None and str(student_id).strip() != "":
        target_id = normalize_id(student_id)
        matches = [s for s in students_data if normalize_id(s.get("id")) == target_id]
        return json.dumps(matches, indent=2) if matches else f"Student with ID '{student_id}' not found."
    return json.dumps(students_data, indent=2)

@tool
def update_student(student_id: Any, name: Optional[str] = None, age: Optional[int] = None, skills: Optional[List[str]] = None) -> str:
    """Updates an existing student's details in students.json using their student_id."""
    global students_data
    target_id = normalize_id(student_id)
    
    for s in students_data:
        if normalize_id(s.get("id")) == target_id:
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
    target_id = normalize_id(student_id)
    
    target = next((s for s in students_data if normalize_id(s.get("id")) == target_id), None)
    if not target:
        return f"Operation aborted: Student ID {student_id} not found."

    print(f"\n⚠️  DESTRUCTIVE ACTION WARNING ⚠️")
    print(f"Target Record: {target}")
    confirm = input(f"Are you sure you want to permanently delete student ID {student_id}? (yes/no): ").strip().lower()
    
    if confirm not in ["yes", "y"]:
        return f"Deletion cancelled by user. Student ID {student_id} was NOT deleted."

    initial_length = len(students_data)
    students_data = [s for s in students_data if normalize_id(s.get("id")) != target_id]
    
    if len(students_data) < initial_length:
        save_students(students_data)
        return f"Successfully deleted student ID {student_id}."
    
    return f"Student ID {student_id} not found."


tools = [create_student, read_student, update_student, delete_student]


# ==========================================
# TASK 3, 4 & PHASE 4: Model Binding & Memory
# ==========================================

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
llm_with_tools = llm.bind_tools(tools)

# Global dictionary to store message history across active chat sessions
sessions: Dict[str, List[BaseMessage]] = {}

def get_or_create_session(session_id: str) -> List[BaseMessage]:
    """Retrieves existing history for a session or initializes a new session with SystemMessage."""
    if session_id not in sessions:
        sessions[session_id] = [
            SystemMessage(content="You are a helpful student database manager. Use the provided tools to execute student CRUD operations and summarize tool results clearly for the user.")
        ]
    return sessions[session_id]

def run_agent(session_id: str, user_input: str):
    """Executes the ReAct loop maintaining context in memory via session_id."""
    
    messages = get_or_create_session(session_id)
    messages.append(HumanMessage(content=user_input))
    
    tool_map = {
        "create_student": create_student,
        "read_student": read_student,
        "update_student": update_student,
        "delete_student": delete_student
    }
    
    # Run dynamic tool invocation loop until agent reaches final text response
    while True:
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        
        # If model requests tool calls, execute them and feed back ToolMessages
        if ai_msg.tool_calls:
            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                
                print(f"\n[Model Selected Tool: {tool_name} with args: {tool_args}]")
                
                selected_tool = tool_map[tool_name]
                tool_result = selected_tool.invoke(tool_args)
                
                print(f"[Tool Result Output: {tool_result}]")
                
                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_id
                    )
                )
        else:
            # Final text answer reached, output to user
            print("\n--- Agent Response ---")
            print(extract_text_content(ai_msg.content))
            break


if __name__ == "__main__":
    # Task 1: Generate unique session_id for current interactive run
    session_id = str(uuid.uuid4())
    print(f"Started Chat Session ID: {session_id}")
    print("Type 'exit' or 'quit' to end the session.\n")
    
    while True:
        try:
            user_input = input("\nUser: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("Ending session. Goodbye!")
                break
            if not user_input:
                continue
                
            run_agent(session_id, user_input)
            
        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Exiting.")
            break