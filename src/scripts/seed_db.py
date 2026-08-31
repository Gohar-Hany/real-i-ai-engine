"""
Raaed RAG Pipeline — Standalone MongoDB Setup & Verification Script.

This script can be run independently to:
  1. Connect to MongoDB (Atlas or local)
  2. Create the database and all required collections
  3. Create all indexes
  4. Verify the setup
  5. Optionally insert sample test data

Usage:
    cd src
    python -m scripts.seed_db              # Setup + verify
    python -m scripts.seed_db --seed       # Setup + verify + insert sample data
    python -m scripts.seed_db --verify     # Verify only (no creation)
"""

import asyncio
import argparse
import sys
import os

# Ensure the src directory is on the path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from motor.motor_asyncio import AsyncIOMotorClient
from helpers.config import get_settings
from helpers.db_init import init_database, verify_database


def print_header(text: str):
    print(f"\n{'━' * 60}")
    print(f"  {text}")
    print(f"{'━' * 60}\n")


def print_ok(text: str):
    print(f"  ✓ {text}")


def print_err(text: str):
    print(f"  ✗ {text}")


async def run_setup(verify_only: bool = False, seed: bool = False):
    settings = get_settings()

    print_header("Raaed MongoDB Setup")
    print(f"  MongoDB URL:  {settings.MONGODB_URL[:40]}...")
    print(f"  Database:     {settings.MONGODB_DATABASE}")
    print()

    # ── Connect ─────────────────────────────────────────────────────────
    print_header("Step 1: Connecting to MongoDB")
    client = AsyncIOMotorClient(settings.MONGODB_URL, serverSelectionTimeoutMS=5000)

    try:
        result = await client.admin.command("ping")
        print_ok(f"MongoDB ping successful: {result}")
    except Exception as e:
        print_err(f"MongoDB connection FAILED: {e}")
        print()
        print("  Troubleshooting:")
        print("  - Check MONGODB_URL in .env")
        print("  - For Atlas: ensure your IP is whitelisted in Network Access")
        print("  - For local: ensure mongod is running")
        sys.exit(1)

    db = client[settings.MONGODB_DATABASE]

    # ── Create collections & indexes ────────────────────────────────────
    if not verify_only:
        print_header("Step 2: Creating Collections & Indexes")
        await init_database(db)
        print_ok("Database initialization complete")
    else:
        print_header("Step 2: Skipped (--verify mode)")

    # ── Verify ──────────────────────────────────────────────────────────
    print_header("Step 3: Verification")
    status = await verify_database(db)

    for col_name, col_info in status["collections"].items():
        if col_info["exists"]:
            print_ok(f"Collection '{col_name}' exists")
            for idx in col_info["indexes"]:
                print(f"        Index: {idx}")
        else:
            print_err(f"Collection '{col_name}' MISSING")

    print()
    if status["ok"]:
        print_ok("All collections and indexes verified ✓")
    else:
        print_err("Some collections are missing! Run without --verify to create them.")

    # ── Seed sample data ────────────────────────────────────────────────
    if seed:
        print_header("Step 4: Inserting Sample Data")
        from bson import ObjectId

        projects_col = db["projects"]
        assets_col = db["assets"]
        chunks_col = db["chunks"]

        # Check if sample project already exists
        existing = await projects_col.find_one({"project_id": "testproject"})
        if existing:
            print_ok("Sample project 'testproject' already exists, skipping seed")
        else:
            # Insert a sample project
            project_result = await projects_col.insert_one({
                "project_id": "testproject"
            })
            project_oid = project_result.inserted_id
            print_ok(f"Inserted sample project: _id={project_oid}")

            # Insert a sample asset
            from datetime import datetime, timezone
            asset_result = await assets_col.insert_one({
                "asset_project_id": project_oid,
                "asset_type": "file",
                "asset_name": "sample_document.pdf",
                "asset_size": 1024,
                "asset_config": None,
                "asset_pushed_at": datetime.now(timezone.utc),
            })
            asset_oid = asset_result.inserted_id
            print_ok(f"Inserted sample asset: _id={asset_oid}")

            # Insert a sample chunk
            chunk_result = await chunks_col.insert_one({
                "chunk_text": "This is a sample chunk for testing the Raaed RAG pipeline.",
                "chunk_metadata": {
                    "section_heading": "Introduction",
                    "chunk_index": 0,
                    "source": "sample_document.pdf",
                },
                "chunk_order": 1,
                "chunk_project_id": project_oid,
                "chunk_asset_id": asset_oid,
            })
            print_ok(f"Inserted sample chunk: _id={chunk_result.inserted_id}")

        print_ok("Seed data complete")

    # ── Done ────────────────────────────────────────────────────────────
    print_header("Setup Complete!")
    print("  Your MongoDB database is ready.")
    print("  Start the app with:")
    print("    cd src && uvicorn main:app --reload --host 0.0.0.0 --port 5000")
    print()

    client.close()


def main():
    parser = argparse.ArgumentParser(description="Raaed MongoDB Setup Script")
    parser.add_argument("--verify", action="store_true", help="Verify only, don't create")
    parser.add_argument("--seed", action="store_true", help="Insert sample test data")
    args = parser.parse_args()

    asyncio.run(run_setup(verify_only=args.verify, seed=args.seed))


if __name__ == "__main__":
    main()
