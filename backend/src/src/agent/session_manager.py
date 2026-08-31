import uuid
import time
from typing import Optional

class SessionManager:
    """MongoDB-backed session store for chat conversations."""

    def __init__(self, max_history_length: int = 50):
        self.max_history_length = max_history_length

    async def create_session(self, db_client, project_id: str) -> str:
        """Create a new chat session and return its ID."""
        session_id = str(uuid.uuid4())
        session_doc = {
            "session_id": session_id,
            "project_id": project_id,
            "history": [],
            "created_at": time.time(),
            "last_active": time.time(),
        }
        await db_client["chat_sessions"].insert_one(session_doc)
        return session_id

    async def get_or_create_session(self, db_client, session_id: Optional[str], project_id: str) -> str:
        """Return existing session_id if valid, otherwise create a new one."""
        if session_id:
            session = await db_client["chat_sessions"].find_one({"session_id": session_id})
            if session:
                await db_client["chat_sessions"].update_one({"session_id": session_id}, {"$set": {"last_active": time.time()}})
                return session_id
        return await self.create_session(db_client, project_id)

    async def get_history(self, db_client, session_id: str) -> list:
        """Return the chat history for a session."""
        session = await db_client["chat_sessions"].find_one({"session_id": session_id})
        if not session:
            return []
        return session.get("history", [])

    async def get_project_id(self, db_client, session_id: str) -> Optional[str]:
        """Return the project_id associated with a session."""
        session = await db_client["chat_sessions"].find_one({"session_id": session_id})
        if not session:
            return None
        return session.get("project_id")

    async def add_message(self, db_client, session_id: str, role: str, content: str):
        """Add a message to the session history."""
        session = await db_client["chat_sessions"].find_one({"session_id": session_id})
        if not session:
            return

        history = session.get("history", [])
        history.append({
            "role": role,
            "content": content,
        })
        
        # Trim history if it exceeds the maximum length
        if len(history) > self.max_history_length:
            history = history[-self.max_history_length:]

        await db_client["chat_sessions"].update_one(
            {"session_id": session_id},
            {"$set": {"history": history, "last_active": time.time()}}
        )

    async def clear_session(self, db_client, session_id: str) -> bool:
        """Delete a session and its history. Returns True if session existed."""
        result = await db_client["chat_sessions"].delete_one({"session_id": session_id})
        return result.deleted_count > 0

    async def get_active_sessions_count(self, db_client) -> int:
        """Return the number of active sessions."""
        return await db_client["chat_sessions"].count_documents({})

