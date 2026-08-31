from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

class QuizRequest(BaseModel):
    topic: str
    num_questions: Optional[int] = 5

class QuizQuestionSchema(BaseModel):
    question: str
    options: Dict[str, str]
    correct_answer: str
    explanation: str

class QuizResponseSchema(BaseModel):
    topic: str
    questions: List[QuizQuestionSchema]

class QuizResponse(BaseModel):
    quiz: QuizResponseSchema

class SessionClearResponse(BaseModel):
    status: str

class TaskWebhookRequest(BaseModel):
    task_id: str
    description: str
    course: str
    task_type: str
    priority: str
    notes: Optional[str] = ""
    created_at: Optional[str] = None
    is_active: Optional[bool] = True

class QuizResultRequest(BaseModel):
    student_id: Optional[str] = "unknown"
    task_id: str
    score: int
    total: int
    answers: Dict[str, Any]


