import os
# Hot-reloaded to apply new Google Sheets env configurations
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import base, data, nlp, agent, admin, auth
from motor.motor_asyncio import AsyncIOMotorClient
from helpers.config import get_settings
from helpers.db_init import init_database
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from stores.llm.templates.template_parser import TemplateParser

import logging

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern lifespan context manager (replaces deprecated on_event).
    Startup logic runs before `yield`, shutdown logic runs after.
    """
    settings = get_settings()

    # ── MongoDB Connection ──────────────────────────────────────────────
    logger.info(f"[Startup] Connecting to MongoDB: {settings.MONGODB_URL[:30]}...")
    app.mongo_conn = AsyncIOMotorClient(
        settings.MONGODB_URL,
        serverSelectionTimeoutMS=5000,
    )
    app.db_client = app.mongo_conn[settings.MONGODB_DATABASE]

    # Health check: fail fast with a clear message if MongoDB is unreachable
    try:
        await app.mongo_conn.admin.command("ping")
        logger.info("[Startup] MongoDB connection successful ✓")
    except Exception as e:
        logger.error(f"[Startup] MongoDB connection FAILED: {e}")
        logger.error(
            "[Startup] Check MONGODB_URL in .env. "
            "Ensure MongoDB is running and accessible."
        )
        raise

    # Auto-create collections & indexes if they don't exist
    await init_database(app.db_client)

    # ── LLM & Embedding Clients ────────────────────────────────────────
    llm_provider_factory = LLMProviderFactory(settings)
    vectordb_provider_factory = VectorDBProviderFactory(settings)

    # generation client
    app.generation_client = llm_provider_factory.create(provider=settings.GENERATION_BACKEND)
    app.generation_client.set_generation_model(model_id = settings.GENERATION_MODEL_ID)

    # embedding client
    app.embedding_client = llm_provider_factory.create(provider=settings.EMBEDDING_BACKEND)
    app.embedding_client.set_embedding_model(model_id=settings.EMBEDDING_MODEL_ID,
                                             embedding_size=settings.EMBEDDING_MODEL_SIZE)
    
    # vector db client
    app.vectordb_client = vectordb_provider_factory.create(
        provider=settings.VECTOR_DB_BACKEND
    )
    app.vectordb_client.connect()

    # Reranker client (optional, defaults to fast cosine vector retrieval)
    app.reranker_client = None
    if getattr(settings, 'ENABLE_LOCAL_RERANKER', 0) in (1, '1', True):
        try:
            logger.info("Loading Reranker Model (BGE)...")
            from sentence_transformers import CrossEncoder
            app.reranker_client = CrossEncoder('BAAI/bge-reranker-v2-m3')
        except Exception as e:
            logger.warning(f"Local CrossEncoder not enabled/available: {e}")

    app.template_parser = TemplateParser(
        language=settings.PRIMARY_LANG,
        default_language=settings.DEFAULT_LANG,
    )

    app.google_sheets_provider = None

    app._assistant_webhook_url = getattr(
        settings, 'ASSISTANT_WEBHOOK_URL',
        'http://localhost:5000/api/v1/agent/webhook/task'
    )

    logger.info("[Startup] All services initialized ✓")

    # ── App runs here ───────────────────────────────────────────────────
    yield

    # ── Shutdown ────────────────────────────────────────────────────────
    logger.info("[Shutdown] Closing connections...")
    app.mongo_conn.close()
    app.vectordb_client.disconnect()
    if hasattr(app, 'google_sheets_provider') and app.google_sheets_provider:
        app.google_sheets_provider.disconnect()
    logger.info("[Shutdown] Done ✓")


app = FastAPI(
    title="REAL_i AI Engine API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://localhost:4000",
    ],
    allow_origin_regex=r"https://.*\.azurestaticapps\.net|https://.*\.vercel\.app|https://.*\.railway\.app|https://.*\.up\.railway\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(nlp.nlp_router)
app.include_router(agent.agent_router)
app.include_router(admin.admin_router)
app.include_router(auth.auth_router)

