"""
CrewAI tool factories for the Raaed AI Agent.

Tools are created via factory functions that close over the initialized
NLPController and project, so they have direct access to the vector DB,
embedding model, and LLM without any HTTP overhead.
"""

from crewai.tools import tool
from controllers.NLPController import NLPController
from models.db_schemes import Project
import logging

logger = logging.getLogger("uvicorn.error")


def create_rag_tools(nlp_controller: NLPController, project: Project):
    """
    Factory that returns a list of CrewAI-compatible tools bound to
    the given NLPController and project.
    """

    @tool("Search Course Materials")
    def search_course_materials(query: str) -> str:
        """
        Search through the student's uploaded course materials (lecture PDFs,
        notes, textbooks) to find the most relevant passages about a topic.
        Use this tool whenever you need to look up specific information from
        the course content. Input should be a clear, specific search query.
        """
        try:
            results = nlp_controller.search_vector_db_collection(
                project=project,
                text=query,
                limit=5,
            )

            if not results or len(results) == 0:
                return "No relevant materials found for this query. The course materials may not cover this topic."

            formatted = []
            for idx, doc in enumerate(results):
                formatted.append(
                    f"--- Document {idx + 1} (Relevance Score: {doc.score:.3f}) ---\n"
                    f"{doc.text}\n"
                )
            return "\n".join(formatted)

        except Exception as e:
            logger.error(f"RAG search tool error: {e}")
            return f"Error searching course materials: {str(e)}"

    @tool("Get Detailed Answer from Course Materials")
    def get_rag_answer(query: str) -> str:
        """
        Get a comprehensive, AI-generated answer to a question by retrieving
        relevant course materials and synthesizing an answer from them.
        Use this tool when the student asks a question that requires a detailed
        answer grounded in their course content. Input should be the student's
        question or topic.
        """
        try:
            answer, full_prompt, chat_history = nlp_controller.answer_rag_question(
                project=project,
                query=query,
                limit=5,
            )

            if not answer:
                return "Could not generate an answer from the course materials. The topic may not be covered."

            return answer

        except Exception as e:
            logger.error(f"RAG answer tool error: {e}")
            return f"Error generating answer: {str(e)}"

    return [search_course_materials, get_rag_answer]
