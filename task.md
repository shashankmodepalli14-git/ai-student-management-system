GenAI Student Management AI — Project Curriculum
Project
Build an AI assistant that manages student information using natural language.
Student fields: student_id, name, skills, bio
Example requests:

"Show me students who know Python."
"Add React to student S101's skills."
"Change S102's bio."
"Who has experience with machine learning?"
"Find students similar to S101."
"Update S103's skills to Python, SQL and FastAPI."
Stack
Python → Gemini 3.6 Flash → LangChain → Tool Calling → ChromaDB → RAG → LangGraph
Phase 1 — LLM Fundamentals
Day 1 — First LLM API
Task 1: Use google-genai and Gemini 3.6 Flash to send a prompt and print the response.
Outcome: API key, Python SDK, first LLM response
Task 2: Create a reusable ask_llm(prompt) function.
Outcome: reusable LLM function, prompt input/output
Task 3: Add system instructions for a Student Management Assistant.
Outcome: system prompt, controlled assistant behavior
Task 4: Ask the model to extract student information from natural language.
Outcome: information extraction, structured thinking
Task 5: Return student information as JSON using structured output.
Outcome: JSON response, schema, validation
Phase 2 — LangChain
Day 6 — LangChain Basics
Task 1: Install LangChain and connect Gemini through LangChain.
Outcome: LangChain model, Gemini integration
Task 2: Create a prompt template for student queries.
Outcome: prompt template, variables
Task 3: Build a LangChain chain: prompt → model → parser.
Outcome: LCEL chain, output parser
Task 4: Store students in a local JSON file and load them with Python.
Outcome: JSON database, CRUD basics
Task 5: Use LangChain to interpret natural-language CRUD requests.
Outcome: natural-language CRUD, chain-based application
Phase 3 — Tool Calling
Day 11 — Give the LLM Actions
Task 1: Create Python functions get_student, add_student, update_student, delete_student.
Outcome: CRUD tools, Python functions
Task 2: Expose the functions as LangChain tools.
Outcome: tool definitions, tool schemas
Task 3: Connect tools to Gemini and let the model select the appropriate tool.
Outcome: function calling, tool selection
Task 4: Implement the complete loop: user → LLM → tool → result → LLM.
Outcome: AI agent loop, tool execution
Task 5: Add confirmation before destructive operations such as delete.
Outcome: human confirmation, safer agent behavior
Phase 4 — RAG + ChromaDB
Day 16 — Give the AI Student Knowledge
Task 1: Convert student records into documents and create embeddings using a free/local-compatible embedding approach.
Outcome: documents, embeddings
Task 2: Store the embeddings in local ChromaDB.
Outcome: vector database, persistence
Task 3: Implement semantic search for student information.
Outcome: similarity search, retrieval
Task 4: Build a RAG chain: question → retrieve students → Gemini → answer.
Outcome: RAG pipeline, contextual answers
Task 5: Combine RAG with student CRUD tools.
Outcome: retrieval + actions, intelligent student assistant
Phase 5 — LangGraph
Day 21 — Build the Agent Workflow
Task 1: Create a LangGraph state containing the user request, retrieved data and tool results.
Outcome: graph state, typed state
Task 2: Create nodes for LLM reasoning, retrieval and tool execution.
Outcome: graph nodes, agent workflow
Task 3: Add conditional routing between normal questions, RAG and tools.
Outcome: conditional edges, routing
Task 4: Add human approval before sensitive modifications.
Outcome: human-in-the-loop, controlled execution
Task 5: Connect everything into the final Student Management Agent.
Outcome: LangGraph agent, RAG, tools, memory/state, complete application
Example Data
[
  {
    "student_id": "S101",
    "name": "Arjun Kumar",
    "skills": ["Python", "FastAPI", "SQL"],
    "bio": "Backend developer interested in AI and distributed systems."
  },
  {
    "student_id": "S102",
    "name": "Priya Sharma",
    "skills": ["Python", "Machine Learning", "TensorFlow"],
    "bio": "Computer science student interested in machine learning and NLP."
  },
  {
    "student_id": "S103",
    "name": "Rahul Verma",
    "skills": ["JavaScript", "React", "Node.js"],
    "bio": "Frontend-focused student who enjoys building web applications."
  },
  {
    "student_id": "S104",
    "name": "Sneha Reddy",
    "skills": ["Python", "SQL", "Data Analysis"],
    "bio": "Data-oriented student interested in analytics and visualization."
  }
]
Final Capabilities
"Who knows Python?" → RAG/search
"Tell me about Priya." → RAG
"Add LangChain to S101's skills." → tool calling
"Remove React from S103." → tool calling
"Find students interested in AI." → semantic retrieval
"Update S104's bio and then tell me what skills she has." → LangGraph orchestration

Can you prepare a dpf out of thsi content