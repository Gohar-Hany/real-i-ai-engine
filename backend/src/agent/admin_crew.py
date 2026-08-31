"""
Admin Agent Crew Runner — Processes instructor task creation requests.

Uses CrewAI to analyze natural language educational requests, extract task
parameters, and notify the assistant agent via webhook.
"""

import os
import re
import json
import logging
import datetime
import requests

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from helpers.config import get_settings
from .prompts import ADMIN_AGENT_SYSTEM_PROMPT
from .crew_runner import OpenRouterLLM

logger = logging.getLogger("uvicorn.error")

# Disable CrewAI telemetry
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"


def _get_admin_llm():
    """Create the LLM instance for the admin agent."""
    settings = get_settings()
    api_key = settings.OPENAI_API_KEY
    api_url = settings.OPENAI_API_URL or "https://openrouter.ai/api/v1"
    model_name = settings.GENERATION_MODEL_ID or "openai/gpt-4o-mini"

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured in settings")

    if not model_name.startswith("openrouter/"):
        model_name = f"openrouter/{model_name}"

    return OpenRouterLLM(
        model=model_name,
        base_url=api_url,
        api_key=api_key,
        temperature=0.1,
    )


def run_admin_crew(
    user_request: str,
    chat_history: list = None,
) -> dict:
    """
    Execute the Admin Crew to process a user request.

    Workflow:
        1. CrewAI agent analyzes the natural language request in context of chat history.
        2. If confirmed, uses the 'Register Task Guideline' tool to write to DB and notify students' AI.
        3. Returns the status, task_id, and response message.
    """
    settings = get_settings()
    llm = _get_admin_llm()

    # Force environment variables for LiteLLM/Instructor
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
    os.environ["OPENAI_API_BASE"] = settings.OPENAI_API_URL or "https://openrouter.ai/api/v1"
    os.environ["OPENAI_BASE_URL"] = settings.OPENAI_API_URL or "https://openrouter.ai/api/v1"

    # Query projects and their uploaded assets to build a content-aware mapping for LLM course-mapping
    project_mapping_info = []
    existing_courses = []
    try:
        from pymongo import MongoClient
        client = MongoClient(settings.MONGODB_URL)
        db = client[settings.MONGODB_DATABASE or "reali-db"]
        
        projects = list(db["projects"].find({}))
        assets = list(db["assets"].find({}))
        
        for p in projects:
            p_id = p.get("project_id")
            if not p_id:
                continue
            existing_courses.append(p_id)
            # Find assets for this project
            p_obj_id = p.get("_id")
            p_assets = [a.get("asset_name") for a in assets if a.get("asset_project_id") == p_obj_id]
            assets_str = ", ".join(p_assets) if p_assets else "No uploaded files"
            project_mapping_info.append(f"- Course ID '{p_id}': contains files [{assets_str}]")
            
        client.close()
    except Exception as ex:
        logger.error(f"[Admin Agent] Failed to query existing courses: {ex}")

    mapping_str = "\n".join(project_mapping_info)
    if not existing_courses:
        existing_courses = ["General"]
    courses_list_str = ", ".join(existing_courses)

    # Accumulator to collect task registration details from the tool execution
    created_tasks = []

    @tool("Register Task Guideline")
    def register_task_guideline(
        task_type: str,
        description: str,
        course: str,
        priority: str,
        notes: str = ""
    ) -> str:
        """
        Registers a task or guideline in the system. Use this tool ONLY when the instructor
        has explicitly confirmed they want to create/save the task.

        Args:
            task_type: Must be one of: Quiz, Assignment, Flashcards, Study Guide, Summary, Exam, Guideline
            description: A clear, concise explanation of the task
            course: The course ID (e.g., 'testproject1' or '1') or 'General'
            priority: Priority level (High, Medium, Low)
            notes: Any extra parameters (e.g. number of MCQs, chapter numbers)
        """
        try:
            from pymongo import MongoClient
            client = MongoClient(settings.MONGODB_URL)
            db = client[settings.MONGODB_DATABASE or "reali-db"]
            
            # 1. Generate task ID
            guidelines = list(db["instructor_guidelines"].find({}))
            max_num = 0
            for g in guidelines:
                t_match = re.search(r"T(\d+)", g.get("task_id", ""))
                if t_match:
                    num = int(t_match.group(1))
                    if num > max_num:
                        max_num = num
            task_id = f"T{max_num + 1:03d}"
            
            # 2. Normalize course/project_id
            course_clean = course.strip()
            project_id = re.sub(r'[^a-zA-Z0-9]', '', course_clean.lower())
            if not project_id:
                project_id = "general"
                
            # Verify project exists in DB
            project = db["projects"].find_one({"project_id": project_id})
            if not project and project_id != "general":
                # Fallback to the first available project
                existing_p = db["projects"].find_one({})
                if existing_p:
                    project_id = existing_p["project_id"]
                    
            # 3. Save to MongoDB
            created_at_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            guideline_doc = {
                "project_id": project_id,
                "task_id": task_id,
                "task_type": task_type,
                "description": description,
                "priority": priority,
                "status": "Pending",
                "notes": notes,
                "created_at": created_at_str,
                "is_active": True
            }
            db["instructor_guidelines"].insert_one(guideline_doc)
            client.close()
            
            # 4. Trigger Webhook to notify students' assistant agent
            webhook_url = getattr(settings, "ASSISTANT_WEBHOOK_URL", None)
            if not webhook_url:
                webhook_url = "http://localhost:5000/api/v1/agent/webhook/task"
                
            payload = {
                "task_id": task_id,
                "description": description,
                "course": project_id,
                "task_type": task_type,
                "priority": priority,
                "notes": notes,
                "created_at": created_at_str,
                "is_active": True
            }
            
            try:
                res = requests.post(webhook_url, json=payload, timeout=5)
                webhook_status = f"Webhook success (HTTP {res.status_code})"
            except Exception as webhook_err:
                webhook_status = f"Webhook failed: {str(webhook_err)}"
                
            task_info = {
                "task_id": task_id,
                "status": "created",
                "task_type": task_type,
                "course": project_id,
                "priority": priority,
                "description": description,
                "notes": notes
            }
            created_tasks.append(task_info)
            
            return f"SUCCESS: Task {task_id} registered successfully in database. {webhook_status}. Make sure to report the generated Task ID {task_id} to the instructor in your final message."
            
        except Exception as e:
            logger.error(f"[Register Tool] Error: {e}")
            return f"ERROR: Failed to register task: {str(e)}"

    # Define the Admin Agent
    admin_agent = Agent(
        role="Admin Agent / Request Analyzer",
        goal=(
            "Analyze natural language requests from the instructor, discuss details, "
            "and call the Register Task Guideline tool ONLY after user confirmation."
        ),
        backstory=ADMIN_AGENT_SYSTEM_PROMPT,
        tools=[register_task_guideline],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=4
    )

    # Format chat history
    formatted_history = ""
    if chat_history:
        for msg in chat_history:
            role = "Instructor" if msg["role"] == "user" else "REAL_i_Admin"
            formatted_history += f"{role}: {msg['content']}\n"

    # Single conversational task
    chat_task = Task(
        description=f"""
You are chatting with the instructor (user).
Here is the previous conversation history:
{formatted_history}

User's new message: "{user_request}"
Here is the list of courses and their materials in the system:
{mapping_str}

Please perform the following instructions:
1. If the user is requesting to create a task (quiz, assignment, summary, guidelines, study guide, flashcards):
   - Propose the details clearly (Type, Course name matching existing IDs: {courses_list_str}, Description, Priority, Notes).
   - Ask for confirmation first. Do NOT call the 'Register Task Guideline' tool yet.
2. If the user is confirming a task draft that you previously proposed in the history:
   - Call the 'Register Task Guideline' tool to persist the task.
3. If it is a greeting or general talk, respond conversationally.
""",
        expected_output="A conversational, helpful text response to the instructor.",
        agent=admin_agent,
    )

    # Run the Crew
    admin_crew = Crew(
        agents=[admin_agent],
        tasks=[chat_task],
        process=Process.sequential,
        verbose=True,
    )

    logger.info(f"[Admin Agent] Processing request: '{user_request}'")
    result = admin_crew.kickoff()

    response_text = str(result.raw)

    if created_tasks:
        task_info = created_tasks[0]
        return {
            "task_id": task_info["task_id"],
            "status": "created",
            "message": response_text,
            "task_type": task_info["task_type"],
            "course": task_info["course"],
            "notes": task_info["notes"],
            "description": task_info["description"]
        }
    else:
        return {
            "task_id": "NONE",
            "status": "chat",
            "message": response_text
        }
