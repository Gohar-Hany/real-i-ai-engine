"""
embedder.py
-----------
RAG Project - Digital Pioneers Initiative | AI Learning Assistant v2.0
Module: Vector Embedding & Storage

Strategy: 
1. Load chunks from the semantic chunker.
2. Generate embeddings using Alibaba-NLP/gte-multilingual-base.
3. Store in ChromaDB for persistent local retrieval.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Third-party imports
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: Missing dependencies. Please install them using:")
    print("pip install chromadb sentence-transformers")
    sys.exit(1)

from chunking import semantic_chunker as chunker_module

# ──────────────────────────────────────────────
# Core Embedding Logic
# ──────────────────────────────────────────────

class VectorStoreManager:
    def __init__(self, db_path: str = "./data/chroma_db", model_name: str = "Alibaba-NLP/gte-multilingual-base"):
        """
        Initialize the embedding model and ChromaDB client.
        """
        print(f"Loading embedding model: {model_name}...")
        import torch
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        
        # Patch for GTE Multilingual Base to fix uninitialized position_ids
        if "gte-multilingual-base" in model_name:
            print("Applying position_ids patch for GTE model...")
            base_transformer = self.model[0].auto_model
            if hasattr(base_transformer, "embeddings") and hasattr(base_transformer.embeddings, "position_ids"):
                max_positions = base_transformer.config.max_position_embeddings
                base_transformer.embeddings.position_ids = torch.arange(max_positions).expand((1, -1))
        
        print(f"Initializing ChromaDB at: {db_path}...")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="document_chunks",
            metadata={"hnsw:space": "cosine"}
        )

    def embed_and_store(self, chunks: List[Dict[str, Any]]):
        """
        Convert chunks to embeddings and store them in ChromaDB.
        """
        if not chunks:
            print("No chunks to process.")
            return

        print(f"Processing {len(chunks)} chunks...")
        
        documents = [c["chunk_content"] for c in chunks]
        ids = [c["chunk_id"] for c in chunks]
        
        # Prepare metadata (ensure all values are primitive types for ChromaDB)
        metadatas = []
        for c in chunks:
            metadata = {
                "source": c.get("source", "unknown"),
                "source_path": str(c.get("source_path", "")),
                "section_heading": c.get("section_heading", ""),
                "chunk_index": c.get("chunk_index", 0),
                "token_count": c.get("token_count", 0)
            }
            if "page_number" in c:
                metadata["page_number"] = c["page_number"]
            metadatas.append(metadata)

        # Generate embeddings
        print("Generating embeddings (this may take a moment)...")
        embeddings = self.model.encode(documents).tolist()

        # Debug: Print the first chunk to verify cleaning
        if chunks:
            print("\n--- Debug: First Chunk Content ---")
            print(chunks[0]["chunk_content"][:500])
            print("----------------------------------\n")

        # Add to ChromaDB
        print("Storing in ChromaDB...")
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        print("Success: Chunks stored successfully.")

def process_file(md_path: str, store_manager: VectorStoreManager):
    """
    Process a single markdown file: read, chunk, and embed.
    """
    path = Path(md_path)
    if not path.exists():
        print(f"Error: File not found: {md_path}")
        return

    print(f"\n--- Processing: {path.name} ---")
    content = path.read_text(encoding="utf-8")

    # Format for chunker
    documents = [
        {
            "page_content": content,
            "source": str(path.absolute()),
        }
    ]

    # Run semantic chunking
    chunks = chunker_module.chunk_documents(documents)
    
    # Store in vector DB
    store_manager.embed_and_store(chunks)

# ──────────────────────────────────────────────
# Main Execution
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    # Default path for testing
    default_test_file = r"data\output\Python Session 1.md"
    
    target_file = sys.argv[1] if len(sys.argv) > 1 else default_test_file
    
    # Initialize the manager
    manager = VectorStoreManager()
    
    # Process the file
    process_file(target_file, manager)
    
    # Optional: Quick verify
    print("\n--- Quick Verification ---")
    query_text = "why is Python So Popular?"
    query_embedding = manager.model.encode([query_text]).tolist()
    
    results = manager.collection.query(
        query_embeddings=query_embedding,
        n_results=5
    )
    
    print(f"Top results for '{query_text}':")
    for i, doc in enumerate(results['documents'][0]):
        print(f"\n[{i+1}] {results['metadatas'][0][i]['section_heading']} (Score: {results['distances'][0][i]:.4f})")
        print(f"Content snippet: {doc[:200]}...")
