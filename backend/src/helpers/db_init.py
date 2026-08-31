"""
Database Initialization Helper for Raaed RAG Pipeline.

Ensures all required MongoDB collections and indexes exist.
This module is idempotent — safe to run on every application startup.

Usage:
    # As part of FastAPI startup (called from main.py):
    from helpers.db_init import init_database
    await init_database(db_client)

    # Standalone verification:
    python -m helpers.db_init
"""

import logging
from motor.motor_asyncio import AsyncIOMotorDatabase
from models.enums.DataBaseEnum import DataBaseEnum
from models.db_schemes import Project, Asset, DataChunk

logger = logging.getLogger("uvicorn.error")


# Map collection enum names to their schema classes (which define indexes)
COLLECTION_SCHEMA_MAP = {
    DataBaseEnum.COLLECTION_PROJECT_NAME.value: Project,
    DataBaseEnum.COLLECTION_ASSET_NAME.value: Asset,
    DataBaseEnum.COLLECTION_CHUNK_NAME.value: DataChunk,
}


async def init_database(db_client: AsyncIOMotorDatabase) -> None:
    """
    Ensure all required collections and their indexes exist.

    This function is idempotent:
    - Collections that already exist are skipped.
    - Indexes that already exist are skipped (create_index is idempotent).

    Args:
        db_client: An AsyncIOMotorDatabase instance (e.g., app.db_client).
    """
    existing_collections = await db_client.list_collection_names()
    logger.info(f"[DB Init] Existing collections: {existing_collections}")

    for collection_name, schema_cls in COLLECTION_SCHEMA_MAP.items():
        # Create collection if it doesn't exist
        if collection_name not in existing_collections:
            await db_client.create_collection(collection_name)
            logger.info(f"[DB Init] Created collection: {collection_name}")
        else:
            logger.info(f"[DB Init] Collection already exists: {collection_name}")

        # Create indexes (idempotent — MongoDB skips if index already exists)
        collection = db_client[collection_name]
        indexes = schema_cls.get_indexes()
        for index in indexes:
            await collection.create_index(
                index["key"],
                name=index["name"],
                unique=index["unique"],
            )
            logger.info(
                f"[DB Init] Ensured index '{index['name']}' on {collection_name} "
                f"(unique={index['unique']})"
            )

    logger.info("[DB Init] Database initialization complete.")


async def verify_database(db_client: AsyncIOMotorDatabase) -> dict:
    """
    Verify that all collections and indexes exist. Returns a status dict.
    Useful for health-check endpoints or startup diagnostics.
    """
    status = {"ok": True, "collections": {}}

    existing_collections = await db_client.list_collection_names()

    for collection_name, schema_cls in COLLECTION_SCHEMA_MAP.items():
        col_status = {
            "exists": collection_name in existing_collections,
            "indexes": [],
        }

        if col_status["exists"]:
            collection = db_client[collection_name]
            index_info = await collection.index_information()
            col_status["indexes"] = list(index_info.keys())
        else:
            col_status["exists"] = False
            status["ok"] = False

        status["collections"][collection_name] = col_status

    return status
