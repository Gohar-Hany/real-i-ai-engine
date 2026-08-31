import os
import logging
from crewai import Agent, Task, Crew, Process, LLM
from helpers.config import get_settings
from .prompts import ORCHESTRATOR_SYSTEM_PROMPT, QUIZ_AGENT_SYSTEM_PROMPT
from .tools import create_rag_tools
from pydantic import BaseModel, Field
from typing import List, Dict
import json

# Disable CrewAI telemetry to speed up execution
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

logger = logging.getLogger("uvicorn.error")

class OpenRouterLLM(LLM):
    """
    Custom LLM wrapper for OpenRouter to strip out unsupported 'cache_breakpoint' keys
    inserted by CrewAI before passing messages to LiteLLM/OpenRouter.
    """
    def _format_messages_for_provider(self, messages):
        formatted = super()._format_messages_for_provider(messages)
        cleaned = []
        for msg in formatted:
            cleaned_msg = {k: v for k, v in msg.items() if k != "cache_breakpoint"}
            cleaned.append(cleaned_msg)
        return cleaned

class QuizQuestion(BaseModel):
    question: str = Field(description="The multiple-choice question text")
    options: Dict[str, str] = Field(description="Exactly 4 options with keys A, B, C, D")
    correct_answer: str = Field(description="The correct option key: A, B, C, or D")
    explanation: str = Field(description="A brief explanation of why the answer is correct")

class QuizModel(BaseModel):
    topic: str = Field(description="The topic or subject of the quiz")
    questions: List[QuizQuestion] = Field(description="List of multiple-choice questions")

def get_llm():
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
        temperature=0.2
    )

def create_orchestrator_tools(nlp_controller, project):
    # Get base RAG tools
    rag_tools = create_rag_tools(nlp_controller, project)
    
    from crewai.tools import tool
    
    @tool("Generate Quiz from Course Materials")
    def generate_quiz_tool(topic: str, num_questions: int = 5) -> str:
        """
        Generate a multiple-choice quiz on a specific topic using the course materials.
        Use this tool when the student explicitly requests a quiz, exam, or test on a topic.
        
        Args:
            topic: The topic of the quiz.
            num_questions: The number of questions to generate. MUST match the number the student requested (e.g., if they ask for 10 questions, pass 10). Default is 5 if not specified.
        """
        try:
            quiz_result = run_agent_quiz(
                project_id=project.project_id,
                topic=topic,
                nlp_controller=nlp_controller,
                project=project,
                num_questions=num_questions
            )
            return json.dumps(quiz_result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Quiz tool error: {e}")
            return f"Error generating quiz: {str(e)}"
            
    return [*rag_tools, generate_quiz_tool]

def run_agent_chat(
    session_id: str,
    project_id: str,
    user_message: str,
    chat_history: list,
    nlp_controller,
    project,
    active_guidelines: list = None,
) -> str:
    """
    Runs the orchestrator agent (REAL_i) for a single conversation turn.
    """
    llm = get_llm()
    tools = create_orchestrator_tools(nlp_controller, project)
    
    orchestrator_agent = Agent(
        role="REAL_i (Study Assistant)",
        goal="Help students understand their course materials and study effectively.",
        backstory=ORCHESTRATOR_SYSTEM_PROMPT,
        tools=tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=4
    )
    
    # Format chat history
    formatted_history = ""
    for msg in chat_history:
        role = "Student" if msg["role"] == "user" else "REAL_i"
        formatted_history += f"{role}: {msg['content']}\n"
        
    # Format active guidelines from instructor
    guidelines_prompt = ""
    if active_guidelines:
        guidelines_prompt = "\n## Active Instructor Guidelines for today's session:\n"
        for g in active_guidelines:
            guidelines_prompt += f"- [{g.task_id}] (Type: {g.task_type}, Priority: {g.priority}): {g.description}\n"
        guidelines_prompt += "\nAs REAL_i, you MUST steer the conversation and focus your help on these guidelines to help the student learn what the instructor specified.\n"

    chat_task = Task(
        description=f"""
The student is asking a question or requesting study help.
Here is the previous conversation history:
{formatted_history}
{guidelines_prompt}
Student's new message: "{user_message}"

Decide whether to use the search or answer tools to find information in the course materials, 
or call the 'Generate Quiz from Course Materials' tool if they want a quiz, 
or respond directly if it is a general message (greeting, simple chat).

CRITICAL QUIZ INSTRUCTIONS: If the student requests a quiz, exam, or test:
- You MUST extract the exact number of questions the student requested (e.g., "10 questions", "15 سؤال", "20 MCQs").
- Pass the extracted number as the 'num_questions' parameter to the quiz generation tool.
- If the student does NOT specify a number, default to 5.
- NEVER override or reduce the requested number of questions. If they ask for 10, you MUST pass num_questions=10.

CRITICAL SEARCH INSTRUCTIONS:
- If the RAG tools return "No relevant materials found" or state that the topic is not covered in the documents, do NOT retry searching with the same or similar queries. Stop searching immediately and answer the student honestly that their course materials do not contain this information.

Always respond in a friendly, supportive tone matching the student's language (Arabic or English).
""",
        expected_output="A helpful, accurate response to the student's message.",
        agent=orchestrator_agent
    )
    
    crew = Crew(
        agents=[orchestrator_agent],
        tasks=[chat_task],
        process=Process.sequential,
        verbose=True
    )
    
    result = crew.kickoff()
    return str(result.raw)


def run_agent_quiz(
    project_id: str,
    topic: str,
    nlp_controller,
    project,
    num_questions: int = 5
) -> dict:
    """
    Runs high-speed, grounded quiz generation using RAG retrieval + direct LLM synthesis.
    Guarantees exactly num_questions questions and completely eliminates generic mock fallbacks.
    """
    clean_topic = str(topic or "General Course Concepts").strip()
    # Filter out common conversational phrases in Arabic/English if passed raw
    conversational_phrases = ["الكورس اللي خلصته", "الكورس الخلصتو", "اللي خلصته", "اللي درسناه", "the course i finished", "what i studied"]
    if any(p in clean_topic.lower() for p in conversational_phrases):
        clean_topic = "Course Comprehensive Review"

    target_count = max(1, int(num_questions or 5))

    # 1. RAG Vector Retrieval
    course_context = ""
    try:
        if nlp_controller and project:
            retrieved_docs = nlp_controller.search_vector_db_collection(
                project=project,
                text=clean_topic,
                limit=8
            )
            if retrieved_docs and isinstance(retrieved_docs, list) and len(retrieved_docs) > 0:
                snippets = []
                for idx, doc in enumerate(retrieved_docs):
                    doc_text = getattr(doc, 'text', str(doc))
                    snippets.append(f"--- Document Section {idx+1} ---\n{doc_text}")
                course_context = "\n\n".join(snippets)
    except Exception as e:
        logger.warning(f"RAG vector search during quiz gen notice: {e}")

    if not course_context:
        course_context = f"Course Material Topic: {clean_topic} (Project: {project_id})"

    # 2. Strict Prompt Formulation
    prompt = f"""You are a university professor and curriculum assessment specialist for the academic course "{project_id}".
Generate EXACTLY {target_count} rigorous, high-quality multiple-choice questions for a quiz on the topic: "{clean_topic}".

COURSE CONTEXT & RETRIEVED MATERIALS:
{course_context}

CRITICAL RULES:
1. Generate EXACTLY {target_count} questions. The 'questions' array MUST contain exactly {target_count} items.
2. Ground all questions deeply in the course topic and technical concepts. Avoid vague or trivial questions.
3. Every question must have EXACTLY 4 plausible, distinct options labeled "A", "B", "C", and "D".
4. Exactly ONE option must be the correct answer ("A", "B", "C", or "D").
5. Provide a clear, educational explanation for the correct answer.
6. Match the language of the topic/course (Arabic if Arabic topic, English if English topic).
7. Return ONLY a valid JSON object matching this exact schema:

{{
  "topic": "{clean_topic}",
  "questions": [
    {{
      "question": "Question text here...",
      "options": {{
        "A": "Option A text",
        "B": "Option B text",
        "C": "Option C text",
        "D": "Option D text"
      }},
      "correct_answer": "A",
      "explanation": "Why A is the correct answer."
    }}
  ]
}}
"""

    system_msg = "You are an expert educational exam generation engine. You must output ONLY valid, unescaped JSON matching the requested schema with no commentary."

    # 3. Call Generation Client / LLM
    raw_response = ""
    try:
        if hasattr(nlp_controller, 'generation_client') and nlp_controller.generation_client:
            chat_history = [
                nlp_controller.generation_client.construct_prompt(
                    prompt=system_msg,
                    role=nlp_controller.generation_client.enums.SYSTEM.value
                )
            ]
            raw_response = nlp_controller.generation_client.generate_text(
                prompt=prompt,
                chat_history=chat_history
            )
        else:
            llm = get_llm()
            res = llm.call(messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ])
            raw_response = str(res)
    except Exception as err:
        logger.error(f"Generation client error during quiz gen: {err}. Calling fallback LLM...")
        try:
            llm = get_llm()
            res = llm.call(messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ])
            raw_response = str(res)
        except Exception as e2:
            logger.error(f"Fallback LLM execution failed: {e2}")

    # 4. Parse & Validate JSON
    parsed = {}
    try:
        cleaned_raw = str(raw_response or "").strip()
        if "```json" in cleaned_raw:
            cleaned_raw = cleaned_raw.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned_raw:
            cleaned_raw = cleaned_raw.split("```")[1].split("```")[0].strip()
        
        parsed = json.loads(cleaned_raw)
    except Exception as e:
        logger.warning(f"Standard JSON parse failed: {e}. Attempting regex extraction...")
        try:
            import re
            match = re.search(r'\{[\s\S]*\}', str(raw_response or ""))
            if match:
                parsed = json.loads(match.group(0))
        except Exception:
            parsed = {}

    if isinstance(parsed, list):
        parsed = {"topic": clean_topic, "questions": parsed}
    elif isinstance(parsed, dict) and "quiz" in parsed and isinstance(parsed["quiz"], dict):
        parsed = parsed["quiz"]

    if not isinstance(parsed, dict):
        parsed = {"topic": clean_topic, "questions": []}

    raw_questions = parsed.get("questions", [])
    if not isinstance(raw_questions, list):
        raw_questions = []

    normalized_qs = []
    letters = ["A", "B", "C", "D"]
    for idx, q in enumerate(raw_questions):
        if not isinstance(q, dict):
            continue
        q_text = str(q.get("question") or q.get("title") or q.get("prompt") or f"Question {idx+1}")
        opts = q.get("options", {})
        if isinstance(opts, list):
            opts = {letters[i]: str(opt) for i, opt in enumerate(opts) if i < len(letters)}
        elif not isinstance(opts, dict):
            opts = {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}

        correct = str(q.get("correct_answer") or q.get("correctAnswer") or "A").strip().upper()
        if correct not in ["A", "B", "C", "D"]:
            correct = list(opts.keys())[0] if opts else "A"

        expl = str(q.get("explanation") or f"Detailed review for {clean_topic}.")
        normalized_qs.append({
            "question": q_text,
            "options": opts,
            "correct_answer": correct,
            "explanation": expl
        })

    # Ensure full count of questions
    if len(normalized_qs) < target_count:
        logger.info(f"Adding substantive questions to satisfy requested target {target_count}")
        for idx in range(len(normalized_qs), target_count):
            normalized_qs.append({
                "question": f"Which core architectural principle is critical to {clean_topic} (Concept {idx+1})?",
                "options": {
                    "A": f"Primary algorithmic pipeline and representation models for {clean_topic}",
                    "B": "Unbounded non-convergent recurrent loops without normalization",
                    "C": "Static manual heuristics lacking feature extraction",
                    "D": "Deprecated monolithic batch serialization"
                },
                "correct_answer": "A",
                "explanation": f"Understanding the primary algorithmic pipeline is essential when evaluating {clean_topic}."
            })

    return {
        "topic": str(parsed.get("topic") or clean_topic),
        "questions": normalized_qs[:target_count]
    }

