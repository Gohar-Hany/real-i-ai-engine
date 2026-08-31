from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):

    APP_NAME: str = "RAAED"
    APP_VERSION: str = "1.0"

    FILE_ALLOWED_TYPES: list = ["text/plain", "application/pdf"]
    FILE_MAX_SIZE: int = 10
    FILE_DEFAULT_CHUNK_SIZE: int = 512000

    MONGODB_URL: str
    MONGODB_DATABASE: str = "raad-rag"

    GENERATION_BACKEND: str = "OPENAI"
    EMBEDDING_BACKEND: str = "OPENAI"

    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_URL: Optional[str] = None
    COHERE_API_KEY: Optional[str] = None

    GENERATION_MODEL_ID: Optional[str] = "openai/gpt-4o-mini"
    EMBEDDING_MODEL_ID: Optional[str] = "openai/text-embedding-3-small"
    EMBEDDING_MODEL_SIZE: Optional[int] = 1536
    INPUT_DAFAULT_MAX_CHARACTERS: Optional[int] = 16384
    GENERATION_DAFAULT_MAX_TOKENS: Optional[int] = 1000
    GENERATION_DAFAULT_TEMPERATURE: Optional[float] = 0.2

    VECTOR_DB_BACKEND: str = "QDRANT"
    VECTOR_DB_PATH: str = "raad_qdrant_db"
    VECTOR_DB_DISTANCE_METHOD: Optional[str] = "cosine"

    PRIMARY_LANG: str = "en"
    DEFAULT_LANG: str = "en"
    ENABLE_LOCAL_RERANKER: int = 0

    # ── Google Sheets (Admin Agent) ──────────────────────────────────
    GOOGLE_SPREADSHEET_ID: Optional[str] = None
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None
    ASSISTANT_WEBHOOK_URL: str = "http://localhost:5000/api/v1/agent/webhook/task"

    class Config:
        env_file = ".env"
        extra = "ignore"

def get_settings():
    settings = Settings()
    # Strip outer quotes from all string settings to avoid Docker --env-file issues
    for key, value in list(settings.__dict__.items()):
        if isinstance(value, str):
            setattr(settings, key, value.strip("'\""))
    return settings
