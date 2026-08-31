from pydantic import BaseModel, Field
from typing import Optional


class TaskCreateRequest(BaseModel):
    """Request schema for creating a new admin task."""
    request: str = Field(
        ...,
        description="The natural language request for a task.",
        example="Create a quiz about Machine Learning Chapter 3 with 20 MCQ questions",
    )
    session_id: Optional[str] = Field(
        None,
        description="The session ID to maintain admin conversation history",
    )


class TaskCreateResponse(BaseModel):
    """Response schema for a created admin task."""
    task_id: str = Field(..., description="The generated task ID, e.g., T001, or 'NONE'")
    status: str = Field(..., description="The status of the task creation request.")
    message: str = Field(..., description="The confirmation or conversational message response.")
    session_id: str = Field(..., description="The session ID for the admin chat session.")
