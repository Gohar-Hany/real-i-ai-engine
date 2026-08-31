from fastapi import APIRouter, Request, status, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from routes.schemes.admin import TaskCreateRequest, TaskCreateResponse
from agent.admin_crew import run_admin_crew
from models.db_schemes.instructor_guideline import InstructorGuideline
from models.InstructorGuidelineModel import InstructorGuidelineModel
from models.UserModel import UserModel
from routes.agent import generate_and_save_quiz_background
import logging
import re
import datetime
import jwt
import os

logger = logging.getLogger("uvicorn.error")

admin_router = APIRouter(
    prefix="/api/v1/admin",
    tags=["api_v1", "admin"],
)


@admin_router.post(
    "/task/create",
    response_model=TaskCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new task and queue it in Google Sheets memory",
)
async def create_task(request: Request, payload: TaskCreateRequest, background_tasks: BackgroundTasks):
    """
    Receives a natural language request, interacts with the Admin Crew,
    and returns the agent's text response and task registration status.
    """
    logger.info(f"[Admin] Received message: '{payload.request}', session_id: {payload.session_id}")
    db_client = request.app.db_client
    
    # 1. Get or create session
    from agent import session_manager
    session_id = await session_manager.get_or_create_session(db_client, payload.session_id, "admin")
    
    # 2. Retrieve session history for conversation context
    history = await session_manager.get_history(db_client, session_id)
    
    # 3. Execute the Admin Agent Crew
    try:
        result = await run_in_threadpool(
            run_admin_crew,
            user_request=payload.request,
            chat_history=history
        )

        task_id = result.get("task_id", "NONE")
        status_msg = result.get("status", "chat")
        message_text = result.get("message", "")
        
        # 4. Save messages to session history
        await session_manager.add_message(db_client, session_id, "user", payload.request)
        await session_manager.add_message(db_client, session_id, "assistant", message_text)
        
        # 5. Fallback background quiz generation trigger (local safety net if webhook has loopback issues)
        if status_msg == "created" and result.get("task_type", "").lower() == "quiz":
            project_id = result.get("course", "general")
            notes = result.get("notes") or ""
            description = result.get("description") or ""
            topic = notes or description or "General Topic"
            
            num_questions = 5  # default
            notes_text = notes + " " + description
            num_match = re.search(r'(\d+)\s*(?:MCQs?|questions?|سؤال|أسئلة)', notes_text, re.IGNORECASE)
            if num_match:
                num_questions = int(num_match.group(1))
                
            background_tasks.add_task(
                generate_and_save_quiz_background,
                request.app,
                project_id,
                task_id,
                topic,
                num_questions
            )
            logger.info(f"[Admin Route] Enqueued background quiz generation for task {task_id}")
        
        return TaskCreateResponse(
            task_id=task_id,
            status=status_msg,
            message=message_text,
            session_id=session_id
        )
            
    except Exception as e:
        logger.exception("[Admin] Error during request processing")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Failed to process request: {str(e)}"},
        )


@admin_router.get("/health", status_code=status.HTTP_200_OK, summary="Admin Agent Health Check")
async def admin_health_check():
    """Returns the health status of the admin agent service."""
    return {"status": "healthy", "service": "Admin Agent"}


@admin_router.get("/guidelines")
async def get_guidelines(request: Request):
    try:
        guideline_model = await InstructorGuidelineModel.create_instance(db_client=request.app.db_client)
        guidelines = await guideline_model.get_all_guidelines()
        return [
            {
                "_id": str(g.id),
                "project_id": g.project_id,
                "task_id": g.task_id,
                "task_type": g.task_type,
                "description": g.description,
                "priority": g.priority,
                "status": g.status,
                "notes": g.notes,
                "created_at": g.created_at,
                "is_active": g.is_active
            }
            for g in guidelines
        ]
    except Exception as e:
        logger.error(f"Error listing guidelines: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})


@admin_router.post("/guidelines")
async def create_or_update_guideline(request: Request, payload: dict, background_tasks: BackgroundTasks):
    try:
        guideline_model = await InstructorGuidelineModel.create_instance(db_client=request.app.db_client)
        
        task_id = payload.get("task_id")
        if not task_id:
            guidelines = await guideline_model.get_all_guidelines()
            max_num = 0
            for g in guidelines:
                match = re.search(r"T(\d+)", g.task_id)
                if match:
                    num = int(match.group(1))
                    if num > max_num:
                        max_num = num
            task_id = f"T{max_num + 1:03d}"
        
        course_name = payload.get("course") or payload.get("project_id") or "General"
        project_id = re.sub(r'[^a-zA-Z0-9]', '', course_name.lower())
        if not project_id:
            project_id = "general"

        from models.ProjectModel import ProjectModel
        project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
        project = await project_model.get_project(project_id=project_id)
        if not project:
            return JSONResponse(status_code=404, content={"detail": f"Project '{project_id}' does not exist."})
        
        task_type = payload.get("task_type", "Quiz")
        guideline = InstructorGuideline(
            project_id=project_id,
            task_id=task_id,
            task_type=task_type,
            description=payload.get("description", ""),
            priority=payload.get("priority", "Medium"),
            status=payload.get("status", "Pending"),
            notes=payload.get("notes", ""),
            created_at=payload.get("created_at") or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            is_active=payload.get("is_active", True) if payload.get("is_active") is not None else True
        )
        
        await guideline_model.create_or_update_guideline(guideline)

        # Trigger background quiz generation if task_type is Quiz
        if task_type.lower() == "quiz":
            topic = payload.get("notes") or payload.get("description") or "General Topic"
            # Extract num_questions from notes/description (e.g., "10 MCQs", "20 questions", "15 سؤال")
            num_questions = 5  # default
            notes_text = (payload.get("notes") or "") + " " + (payload.get("description") or "")
            num_match = re.search(r'(\d+)\s*(?:MCQs?|questions?|سؤال|أسئلة)', notes_text, re.IGNORECASE)
            if num_match:
                num_questions = int(num_match.group(1))
            background_tasks.add_task(
                generate_and_save_quiz_background,
                request.app,
                project_id,
                task_id,
                topic,
                num_questions
            )

        return {"status": "success", "task_id": task_id}
    except Exception as e:
        logger.error(f"Error creating/updating guideline: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})


@admin_router.put("/guidelines/{task_id}/toggle")
async def toggle_guideline(request: Request, task_id: str):
    try:
        doc = await request.app.db_client["instructor_guidelines"].find_one({"task_id": task_id})
        if not doc:
            return JSONResponse(status_code=404, content={"detail": "Guideline not found"})
        
        new_active = not doc.get("is_active", True)
        await request.app.db_client["instructor_guidelines"].update_one(
            {"task_id": task_id},
            {"$set": {"is_active": new_active}}
        )
        return {"status": "success", "task_id": task_id, "is_active": new_active}
    except Exception as e:
        logger.error(f"Error toggling guideline: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})


@admin_router.delete("/guidelines/{task_id}")
async def delete_guideline(request: Request, task_id: str):
    try:
        guideline_model = await InstructorGuidelineModel.create_instance(db_client=request.app.db_client)
        deleted = await guideline_model.delete_guideline(task_id)
        if not deleted:
            return JSONResponse(status_code=404, content={"detail": "Guideline not found"})
        return {"status": "success", "task_id": task_id}
    except Exception as e:
        logger.error(f"Error deleting guideline: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})


SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "your-super-secret-jwt-key")
ALGORITHM = "HS256"


async def get_admin_user(request: Request):
    """Helper to extract and validate the admin user from the JWT token."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        role = payload.get("role")
        if not user_id or role not in ("superadmin", "admin"):
            raise HTTPException(status_code=403, detail="Admin access required")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user_model = await UserModel.create_instance(request.app.db_client)
    user = await user_model.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@admin_router.get("/users")
async def list_users(request: Request):
    """List all registered users (admin only)."""
    await get_admin_user(request)
    user_model = await UserModel.create_instance(request.app.db_client)
    users = await user_model.get_all_users()
    return users


@admin_router.put("/users/{user_id}/role")
async def update_user_role(request: Request, user_id: str):
    """Change a user's role. Only a superadmin can perform this action."""
    admin_user = await get_admin_user(request)
    if admin_user.get("role") != "superadmin":
        raise HTTPException(status_code=403, detail="Super Admin privileges required to change roles")

    body = await request.json()
    new_role = body.get("role")
    if new_role not in ("superadmin", "admin", "instructor", "student"):
        raise HTTPException(status_code=400, detail="Role must be 'superadmin', 'admin', 'instructor', or 'student'")

    # Prevent super admin from demoting themselves
    admin_id = str(admin_user.get("_id", admin_user.get("id")))
    if user_id == admin_id:
        raise HTTPException(status_code=400, detail="Cannot change your own role to prevent lockout")

    user_model = await UserModel.create_instance(request.app.db_client)
    success = await user_model.update_user_role(user_id, new_role)
    if not success:
        raise HTTPException(status_code=404, detail="User not found or role unchanged")
    return {"status": "success", "user_id": user_id, "new_role": new_role}


@admin_router.get("/users/{user_id}/results")
async def get_user_results(request: Request, user_id: str):
    """Fetch all quiz results for a specific student (admin only)."""
    await get_admin_user(request)
    cursor = request.app.db_client["student_results"].find({"student_id": user_id})
    results = []
    async for doc in cursor:
        results.append({
            "task_id": doc.get("task_id"),
            "score": doc.get("score"),
            "total": doc.get("total"),
            "answers": doc.get("answers"),
            "completed_at": doc.get("completed_at")
        })
    return {"user_id": user_id, "results": results}
