# RAG Pipeline: Embedding & Retrieval Documentation

## 1. Architecture Overview
We have implemented a **Two-Stage Retrieval** pipeline to ensure maximum accuracy and relevance for the AI Learning Assistant.

*   **Stage 1 (Vector Search)**: Uses `Alibaba-NLP/gte-multilingual-base` to find the top 10 most similar candidates from the database.
*   **Stage 2 (Reranking)**: Uses `BAAI/bge-reranker-v2-m3` to re-score those 10 candidates and bubble the absolute best answer to Rank #1.

---

## 2. Why We Chose These Models

| Model | Role | Why it was the best choice? |
| :--- | :--- | :--- |
| **GTE Multilingual Base** | **Embedding** | High-tier accuracy (~73%+ MTEB), efficient 768-dimension vectors, and excellent multilingual support for diverse documents. |
| **BGE Reranker v2-m3** | **Reranking** | The enterprise standard for RAG. It effectively eliminates "noise" by performing a deep cross-encoding check between the query and the retrieved text. |

### Comparison Summary:
- **Nomic Embed**: Great for low-cost storage, but we prioritized raw accuracy.
- **BGE (M3 Model)**: Excellent, but we found **GTE** to be faster for initial retrieval while using the **BGE Reranker** as a specialized second pass for the best of both worlds.

---

## 3. Implementation Details

### The GTE "Position IDs" Patch
During development, we encountered a critical `IndexError` when loading the GTE model. This was caused by uninitialized memory in the model's custom implementation.
**Solution**: We applied a manual patch to initialize `position_ids`:
```python
base_transformer = model[0].auto_model
max_positions = base_transformer.config.max_position_embeddings
base_transformer.embeddings.position_ids = torch.arange(max_positions).expand((1, -1))
```

### Search Metric: Cosine Similarity
We configured ChromaDB to use **Cosine Similarity** (`hnsw:space: cosine`) instead of the default L2 distance. This is mathematically better for normalized embeddings like GTE and ensures higher semantic alignment.

---

## 4. Challenges & Solutions

### Issue 1: Redundant Heading Noise
**Problem**: The document processing created multiple duplicate headers (e.g., `## Python Programming` repeated 4 times). This diluted the "density" of the actual features list, making it harder to retrieve.
**Solution**: We updated `semantic_chunker.py` with a regex-based deduplication pass that collapses consecutive identical headings.

### Issue 2: Large Model Download
**Problem**: The BGE Reranker is ~2.27 GB, causing initial timeouts and progress bar suppression in the CLI.
**Solution**: The script handles the download gracefully, and subsequent runs are instant once the model is cached in the HuggingFace hub.

### Issue 3: "Keyword Dilution"
**Problem**: Search queries like "python feature" were sometimes outranked by sections like "Syntax" because they shared many common keywords.
**Solution**: The **BGE Reranker** solved this. It looks at the *intent* of the question and increased the score of the technical feature list from Rank #3 to **Rank #1 with 99% confidence**.

---

## 5. How to Run
1.  **Embedding**: `python embedder.py "path/to/file.md"`
2.  **Searching**: `python reranked_search.py "your question here"`

---
---

# Technical Deep-Dive: AI Learning Assistant Embedding Engine

This section provides a detailed look into the internal mechanics and design of the pipeline.

## 6. Phase 1: Semantic Chunking (`semantic_chunker.py`)
Standard character-based chunking often breaks context. Our pipeline uses **Structure-Aware Semantic Chunking**.

### A. Pre-processing (The "Slide Fix")
Because the source PDFs (processed via Docling) are slide-based, they often contain "furniture" noise.
-   **Heading Deduplication**: Redundant headers are collapsed using a robust regex pass.
-   **Empty Heading Removal**: Any heading not followed by a body (immediately followed by another heading) is discarded.
-   **Noise Filtering**: Specific slide artifacts like "Session Agenda" or "Digilians" are stripped out.

### B. Splitting & Merging
-   **Boundary Detection**: Splits occur at Markdown heading boundaries (`#` to `######`).
-   **Small Section Merging**: Sections smaller than **80 tokens** are merged into the following section to preserve context.
-   **Oversized Section Splitting**: Sections larger than **1000 tokens** are split using a `RecursiveCharacterTextSplitter` with a **17% overlap** (170 tokens) to ensure no information is lost at the boundaries.

---

## 7. Phase 2: Embedding & Storage (`embedder.py`)

### A. The Model: `Alibaba-NLP/gte-multilingual-base`
-   **Architecture**: Based on a modern transformer architecture optimized for retrieval.
-   **Dimensions**: 768-dimensional dense vectors.
-   **Context Window**: 8,192 tokens (though we optimize at ~1,000 for better granularity).

### B. Vector Database: `ChromaDB`
-   **Configuration**: Persistent local storage in `./chroma_db`.
-   **Metadata Schema**:
    -   `chunk_id`: Unique identifier (Source_Index).
    -   `source`: Filename.
    -   `section_heading`: The nearest Markdown header.
    -   `token_count`: Used for context window management.
    -   `chunk_index`: Preserves original document order.

---

## 8. Phase 3: Two-Stage Retrieval (`reranked_search.py`)

### Step 1: Candidate Retrieval (Bi-Encoder)
-   The user query is embedded using GTE with an instruction prefix: `Instruct: retrieve semantically similar documents.`.
-   ChromaDB performs a fast Approximate Nearest Neighbor (ANN) search to find the **Top 10** candidates.

### Step 2: High-Precision Reranking (Cross-Encoder)
-   **Model**: `BAAI/bge-reranker-v2-m3`.
-   Unlike GTE (which looks at abstract vectors), the **Cross-Encoder** looks at the Query and the Document *together*.
-   It performs a deep attention-based comparison to assign a final relevance score (0.0 to 1.0).
-   **Benefit**: In our tests, this moved the "Python Features" list from Rank #3 to **Rank #1 with 99.5% confidence**.
