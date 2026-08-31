from fastapi import APIRouter, Depends, Request
import os
from helpers.config import get_settings, Settings

base_router = APIRouter(
    prefix="/api/v1",
    tags=["api_v1"],
)

@base_router.get("/")
@base_router.get("/health")
async def welcome(request: Request, app_settings: Settings = Depends(get_settings)):

    app_name = app_settings.APP_NAME
    app_version = app_settings.APP_VERSION

    # Connection statuses
    mongodb_status = "offline"
    try:
        await request.app.mongo_conn.admin.command("ping")
        mongodb_status = "online"
    except Exception:
        pass

    qdrant_status = "offline"
    if hasattr(request.app, "vectordb_client") and request.app.vectordb_client and request.app.vectordb_client.client:
        try:
            request.app.vectordb_client.client.get_collections()
            qdrant_status = "online"
        except Exception:
            pass

    openai_status = "offline"
    if hasattr(request.app, "generation_client") and request.app.generation_client:
        if getattr(request.app.generation_client, "api_key", None):
            openai_status = "online"

    # Statistics
    projects_count = 0
    documents_count = 0
    guidelines_count = 0
    try:
        projects_count = await request.app.db_client["projects"].count_documents({})
        documents_count = await request.app.db_client["assets"].count_documents({})
        guidelines_count = await request.app.db_client["instructor_guidelines"].count_documents({})
    except Exception:
        pass

    return {
        "app_name": app_name,
        "app_version": app_version,
        "services": {
            "mongodb": mongodb_status,
            "qdrant": qdrant_status,
            "openai": openai_status
        },
        "stats": {
            "projects_count": projects_count,
            "documents_count": documents_count,
            "guidelines_count": guidelines_count
        }
    }

