import sys
import json
import os
from typing import Optional, Any, Dict, List
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

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
# TASK 3 & 4: Model Binding & Complete Loop
# ==========================================

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
llm_with_tools = llm.bind_tools(tools)

def run_agent(user_input: str):
    """Executes the complete ReAct loop: User -> LLM -> Tool Execution -> ToolMessage -> Final LLM Response."""
    
    messages = [
        SystemMessage(content="You are a helpful student database manager. Use the provided tools to execute student CRUD operations and summarize tool results clearly for the user."),
        HumanMessage(content=user_input)
    ]
    
    # 1. First pass: User -> LLM
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    
    # 2. Check if LLM generated tool call requests
    if ai_msg.tool_calls:
        tool_map = {
            "create_student": create_student,
            "read_student": read_student,
            "update_student": update_student,
            "delete_student": delete_student
        }
        
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
        
        # 3. Final pass: Send full message history back to LLM for final response synthesis
        final_response = llm_with_tools.invoke(messages)
        
        print("\n--- Final Agent Response ---")
        print(extract_text_content(final_response.content))
        
    else:
        print("\n--- Direct Response (No Tool Call Needed) ---")
        print(extract_text_content(ai_msg.content))


if __name__ == "__main__":
    user_input = input("Enter CRUD command: ")

    if not user_input.strip():
        user_input = "Delete student with ID 105."

    print(f"\nUser Command: '{user_input}'")
    run_agent(user_input)