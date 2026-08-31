# 📘 REAL_i — Complete System Documentation

> **REAL_i (REAL_i)** — An AI-powered educational assistant platform for university students.  
> _Digital Pioneers Initiative | AI Learning Assistant v2.0_

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [High-Level System Architecture](#2-high-level-system-architecture)
3. [Detailed Component Architecture](#3-detailed-component-architecture)
4. [Data Flow — End-to-End Pipeline](#4-data-flow--end-to-end-pipeline)
5. [Module Reference](#5-module-reference)
   - [5.1 PDF Extraction Pipeline](#51-pdf-extraction-pipeline)
   - [5.2 Semantic Chunking Engine](#52-semantic-chunking-engine)
   - [5.3 Embedding & Vector Storage](#53-embedding--vector-storage)
   - [5.4 RAG Pipeline (Retrieval + Generation)](#54-rag-pipeline-retrieval--generation)
   - [5.5 AI Agent System (CrewAI)](#55-ai-agent-system-crewai)
   - [5.6 Admin Agent Platform](#56-admin-agent-platform)
   - [5.7 Inter-Agent Communication](#57-inter-agent-communication)
6. [Database Architecture](#6-database-architecture)
7. [API Reference](#7-api-reference)
8. [Configuration Reference](#8-configuration-reference)
9. [Directory Structure](#9-directory-structure)
10. [Deployment & Running](#10-deployment--running)
11. [RAG Evaluation Framework & Performance Metrics](#11-rag-evaluation-framework--performance-metrics)

---

## 1. Project Overview

**REAL_i** is a Retrieval-Augmented Generation (RAG) platform designed to help university students study their course materials more effectively. The system processes uploaded lecture PDFs, extracts and chunks the content semantically, stores it in a vector database, and provides an AI-powered conversational assistant that can answer questions, generate quizzes, and follow instructor guidelines — all grounded in the actual course materials.

### Key Capabilities

| Capability | Description |
|---|---|
| 📄 PDF Processing | Extracts text from lecture slides/PDFs using Docling + OCR fallback |
| 🔪 Semantic Chunking | Splits documents into meaningful sections based on Markdown headings |
| 🔍 Vector Search | Embeds chunks and performs cosine similarity search via Qdrant |
| 🔁 BGE Reranking | Re-scores retrieved documents with a cross-encoder for precision |
| 🤖 AI Study Assistant | Conversational agent ("REAL_i") that answers questions from course materials |
| 📝 Quiz Generator | Sub-agent that generates structured MCQ quizzes from retrieved content |
| 👨‍🏫 Admin Agent | Instructor-facing agent that creates tasks and steers the assistant via webhooks |
| 📊 Shared Memory | Google Sheets-based task queue + MongoDB guidelines for inter-agent communication |

---

## 2. High-Level System Architecture

```mermaid
graph TB
    subgraph "👨‍🏫 Instructor Side"
        INS[Instructor]
        ADMIN_API["Admin Agent API<br/>(FastAPI :8001)"]
        CREW["CrewAI Admin Crew<br/>(Task Analyzer)"]
        GS["Google Sheets<br/>(Shared Memory)"]
    end

    subgraph "🎓 Student Side"
        STU[Student]
    end

    subgraph "🧠 REAL_i Core Platform (FastAPI :8000)"
        direction TB
        MAIN["main.py<br/>(FastAPI Lifespan)"]

        subgraph "API Layer"
            BASE_R["/api/v1/ (Health)"]
            DATA_R["/api/v1/data/ (Upload & Process)"]
            NLP_R["/api/v1/nlp/ (Index, Search, Answer)"]
            AGENT_R["/api/v1/agent/ (Chat, Quiz, Webhook)"]
        end

        subgraph "Controllers"
            DC[DataController]
            PC[ProcessController]
            NLPC[NLPController]
        end

        subgraph "Agent Module (CrewAI)"
            ORCH["Orchestrator Agent<br/>(REAL_i)"]
            QUIZ_A["Quiz Generator<br/>Sub-Agent"]
            SESS[SessionManager]
        end

        subgraph "Processing Pipeline"
            EXT["Extraction<br/>(Docling + OCR)"]
            CHUNK["Semantic Chunker<br/>(Heading-based)"]
        end

        subgraph "Storage Layer"
            MONGO[(MongoDB Atlas)]
            QDRANT[(Qdrant Vector DB)]
        end

        subgraph "AI Models"
            GTE["GTE Multilingual<br/>(Embedding, 768-dim)"]
            BGE["BGE Reranker v2<br/>(Cross-Encoder)"]
            LLM["OpenAI GPT-4o-mini<br/>(via OpenRouter)"]
        end
    end

    INS -->|"Natural language task"| ADMIN_API
    ADMIN_API --> CREW
    CREW -->|"Structured record"| GS
    CREW -->|"Webhook trigger"| AGENT_R
    AGENT_R -->|"Save guideline"| MONGO

    STU -->|"Chat / Quiz request"| AGENT_R
    AGENT_R --> ORCH
    ORCH -->|"RAG Search"| NLPC
    ORCH -->|"Delegate quiz"| QUIZ_A
    QUIZ_A -->|"RAG Search"| NLPC

    DATA_R --> DC
    DATA_R --> PC
    PC --> EXT
    EXT --> CHUNK

    NLP_R --> NLPC
    NLPC --> GTE
    NLPC --> QDRANT
    NLPC --> BGE
    NLPC --> LLM

    MAIN --> MONGO
    MAIN --> QDRANT
    MAIN --> GTE
    MAIN --> BGE
    MAIN --> LLM
```

---

## 3. Detailed Component Architecture

### 3.1 REAL_i Core Platform Architecture

```mermaid
graph LR
    subgraph "FastAPI Application (src/)"
        direction TB

        subgraph "Routes Layer"
            R1["base.py<br/>GET /api/v1/"]
            R2["data.py<br/>POST /upload, /process"]
            R3["nlp.py<br/>POST /index/push, /search, /answer"]
            R4["agent.py<br/>POST /chat, /quiz, /webhook"]
        end

        subgraph "Controllers Layer"
            C1[BaseController]
            C2[DataController]
            C3[ProjectController]
            C4[ProcessController]
            C5[NLPController]
        end

        subgraph "Models Layer (MongoDB)"
            M1[ProjectModel]
            M2[ChunkModel]
            M3[AssetModel]
            M4[InstructorGuidelineModel]
        end

        subgraph "Stores Layer"
            S1["LLM Providers<br/>(OpenAI, CoHere, Local)"]
            S2["VectorDB Providers<br/>(Qdrant, ChromaDB)"]
            S3["Template Parser<br/>(Locale-based prompts)"]
        end
    end

    R2 --> C2 & C4
    R3 --> C5
    R4 --> C5
    C4 --> C3
    C2 --> C1
    C5 --> S1 & S2 & S3
    R2 --> M1 & M2 & M3
    R3 --> M1
    R4 --> M1 & M4
```

### 3.2 Agent System Architecture

```mermaid
graph TB
    subgraph "Agent Module (src/agent/)"
        INIT["__init__.py<br/>(Exports + SessionManager singleton)"]
        PROMPTS["prompts.py<br/>(System prompts for both agents)"]
        TOOLS["tools.py<br/>(RAG tool factories)"]
        CR["crew_runner.py<br/>(CrewAI execution engine)"]
        SM["session_manager.py<br/>(In-memory chat history)"]
    end

    subgraph "CrewAI Crews"
        direction TB
        ORCH_CREW["Orchestrator Crew"]
        QUIZ_CREW["Quiz Generator Crew"]
    end

    subgraph "Tools Available"
        T1["🔍 Search Course Materials"]
        T2["📖 Get Detailed RAG Answer"]
        T3["📝 Generate Quiz (delegates to Quiz Crew)"]
    end

    CR --> ORCH_CREW & QUIZ_CREW
    ORCH_CREW --> T1 & T2 & T3
    QUIZ_CREW --> T1 & T2
    T1 & T2 -->|"Direct access"| NLPC2[NLPController]
    T3 -->|"Internally calls"| QUIZ_CREW
    INIT --> CR & SM
```

### 3.3 Admin Agent Platform Architecture

```mermaid
graph LR
    subgraph "Admin Agent Platform (admin_agent_platform/)"
        MAIN_A["main.py<br/>(FastAPI :8001)"]
        CREW_A["crew.py<br/>(CrewAI Admin Crew)"]
        GSW["google_sheets_writer.py<br/>(Google Sheets API)"]
    end

    subgraph "External Services"
        GS2["Google Sheets<br/>(Shared Memory)"]
        WH["REAL_i Webhook<br/>POST /api/v1/agent/webhook/task"]
    end

    MAIN_A -->|"POST /task/create"| CREW_A
    CREW_A -->|"1. Analyze request"| CREW_A
    CREW_A -->|"2. Extract parameters"| CREW_A
    CREW_A -->|"3. Write record"| GSW
    GSW --> GS2
    CREW_A -->|"4. Trigger webhook"| WH
```

---

## 4. Data Flow — End-to-End Pipeline

### 4.1 Document Ingestion Flow

```mermaid
sequenceDiagram
    actor User
    participant API as FastAPI
    participant DC as DataController
    participant PC as ProcessController
    participant EXT as Extraction Pipeline
    participant CHK as Semantic Chunker
    participant DB as MongoDB
    participant VDB as Qdrant
    participant EMB as GTE Embedding

    User->>API: POST /api/v1/data/upload/{project_id}
    API->>DC: validate_uploaded_file(file)
    DC-->>API: ✅ Valid
    API->>DB: Save file metadata (Asset)
    API-->>User: { file_id, signal: "success" }

    User->>API: POST /api/v1/data/process/{project_id}
    API->>PC: get_file_content(file_id)
    PC->>EXT: run_unified_pipeline(pdf_path)

    Note over EXT: Step 1: Docling parses PDF structure<br/>(headings, text, tables, code)
    Note over EXT: Step 2: Native VLM visual description<br/>(SmolVLM-256M on GPU, auto-disabled on CPU)
    Note over EXT: Step 3: EasyOCR text extraction for scanned PDF pages
    Note over EXT: Step 4: CPU/GPU hardware auto-detection fallback
    Note over EXT: Step 5: Image metadata placeholder replacement

    EXT-->>PC: Cleaned Markdown & JSON tuple
    PC->>CHK: chunk_documents(markdown)

    Note over CHK: 1. Clean slide artifacts<br/>2. Split on headings<br/>3. Merge small sections<br/>4. Split oversized sections

    CHK-->>PC: List[DataChunk]
    PC-->>API: chunks
    API->>DB: insert_many_chunks(chunks)
    API-->>User: { chunks_inserted, files_processed }

    User->>API: POST /api/v1/nlp/index/push/{project_id}
    API->>DB: get_project_chunks(page_by_page)
    loop For each page of chunks
        API->>EMB: embed_text(chunk_text)
        EMB-->>API: vector (768-dim)
        API->>VDB: insert_many(vectors, texts, metadata)
    end
    API-->>User: { inserted_items_count }
```

### 4.2 RAG Query Flow

```mermaid
sequenceDiagram
    actor Student
    participant API as FastAPI
    participant NLPC as NLPController
    participant EMB as GTE Embedding
    participant VDB as Qdrant
    participant RR as BGE Reranker
    participant LLM as GPT-4o-mini
    participant TP as Template Parser

    Student->>API: POST /api/v1/nlp/index/answer/{project_id}
    API->>NLPC: answer_rag_question(project, query, limit=10)

    Note over NLPC: Step 1: Vector Retrieval
    NLPC->>EMB: embed_text(query)
    EMB-->>NLPC: query_vector (768-dim)
    NLPC->>VDB: search_by_vector(query_vector, limit=20)
    VDB-->>NLPC: Top-20 candidates

    Note over NLPC: Step 2: BGE Reranking
    NLPC->>RR: predict([[query, doc1], [query, doc2], ...])
    RR-->>NLPC: reranked_scores
    Note over NLPC: Sort by reranker score,<br/>take top-10

    Note over NLPC: Step 3: Prompt Construction
    NLPC->>TP: get("rag", "system_prompt")
    NLPC->>TP: get("rag", "document_prompt") × N docs
    NLPC->>TP: get("rag", "footer_prompt")

    Note over NLPC: Step 4: LLM Generation
    NLPC->>LLM: generate_text(full_prompt, chat_history)
    LLM-->>NLPC: answer

    NLPC-->>API: (answer, full_prompt, chat_history)
    API-->>Student: { signal: "success", answer, full_prompt }
```

### 4.3 Agent Chat Flow

```mermaid
sequenceDiagram
    actor Student
    participant API as Agent Routes
    participant SM as SessionManager
    participant GM as GuidelineModel
    participant CR as CrewRunner
    participant ORCH as Orchestrator Agent
    participant TOOLS as RAG Tools
    participant NLPC as NLPController

    Student->>API: POST /api/v1/agent/chat/{project_id}
    API->>SM: get_or_create_session(session_id, project_id)
    SM-->>API: session_id
    API->>SM: get_history(session_id)
    SM-->>API: chat_history[]

    API->>GM: get_active_guidelines(project_id)
    GM-->>API: active_guidelines[]

    API->>CR: run_agent_chat(message, history, guidelines)

    Note over CR: Creates CrewAI Orchestrator Agent<br/>with system prompt + tools + guidelines

    CR->>ORCH: Execute chat task
    ORCH->>ORCH: Analyze student message

    alt Student asks a course question
        ORCH->>TOOLS: search_course_materials(query)
        TOOLS->>NLPC: search_vector_db_collection()
        NLPC-->>TOOLS: retrieved documents
        TOOLS-->>ORCH: formatted results
        ORCH->>ORCH: Synthesize answer from results
    else Student requests a quiz
        ORCH->>TOOLS: generate_quiz_tool(topic)
        Note over TOOLS: Internally spawns<br/>Quiz Generator Crew
        TOOLS-->>ORCH: JSON quiz
    else General greeting/chat
        ORCH->>ORCH: Respond directly
    end

    ORCH-->>CR: final response text
    CR-->>API: response

    API->>SM: add_message("user", message)
    API->>SM: add_message("assistant", response)
    API-->>Student: { response, session_id }
```

### 4.4 Instructor → Admin → Assistant Flow

```mermaid
sequenceDiagram
    actor Instructor
    participant ADMIN as Admin Agent API
    participant CREW as CrewAI Admin Crew
    participant GS as Google Sheets
    participant WH as REAL_i Webhook
    participant DB as MongoDB Guidelines
    actor Student
    participant REAL_i as REAL_i Agent

    Instructor->>ADMIN: POST /task/create<br/>"Focus on Module 5 with 20 MCQs"

    ADMIN->>CREW: run_admin_crew(request)

    Note over CREW: Task 1: Analyze request intent<br/>Task 2: Format task record<br/>Task 3: Generate structured JSON

    CREW->>GS: write_task_to_google_sheets(task_data)
    GS-->>CREW: "SUCCESS: Task T003 created"

    CREW->>WH: POST /api/v1/agent/webhook/task
    Note over WH: Payload: {task_id, description,<br/>course, task_type, priority, notes}

    WH->>WH: Normalize course → project_id<br/>"Machine Learning" → "machinelearning"
    WH->>DB: create_or_update_guideline(guideline)
    WH-->>CREW: { status: "success" }

    CREW-->>ADMIN: { task_id: "T003", status: "created" }
    ADMIN-->>Instructor: 201 Created

    Note over DB: Guideline is now ACTIVE<br/>for project "machinelearning"

    Student->>REAL_i: POST /chat/machinelearning<br/>"What should I study today?"
    REAL_i->>DB: get_active_guidelines("machinelearning")
    DB-->>REAL_i: [{ description: "Focus on Module 5..." }]

    Note over REAL_i: Guidelines injected into<br/>orchestrator task prompt

    REAL_i-->>Student: "The instructor asked us to focus<br/>on Module 5: Neural Networks today..."
```

---

## 5. Module Reference

### 5.1 PDF Extraction Pipeline

**Location:** `src/controllers/_extraction/`

The extraction pipeline converts uploaded PDFs into clean structural Markdown and JSON. It has been refactored into a single, unified, hardware-aware pipeline designed for optimal performance on both CPU and GPU:

| File | Purpose |
|---|---|
| `unified_pipeline.py` | Orchestrates the entire extraction process: layouts, OCR, tables, formulas, and VLM descriptions. |
| `__init__.py` | Exports the main entry point: `run_unified_pipeline`. |

**Hardware-Aware Execution Model:**
- **On GPU (Production/T4 Server)**: Uses CUDA-accelerated Docling layout parser, EasyOCR with GPU support, and dynamic `SmolVLM-256M-Instruct` vision-language model for deep image and diagram descriptions.
- **On CPU (Local Developer Environment)**: Detects CUDA unavailability, switches to CPU fallback for standard Docling and EasyOCR, and **automatically disables VLM descriptions** to prevent severe CPU processing bottleneck and memory freezes.

**Pipeline Steps:**
1. **Caching initialization**: Uses cache decorator (`@lru_cache`) on the DocumentConverter to avoid reloading heavy deep-learning model weights on repeated requests.
2. **Page-chunked Processing**: Large PDFs (>20 pages) are automatically split and processed in chunks to keep memory usage minimal and prevent server timeout errors.
3. **Docling Structure Extraction**: Extracts semantic hierarchy, paragraphs, headings, lists, table structures, and code formulas.
4. **OCR Fallback**: Invokes EasyOCR to read text out of scanned PDF pages or embedded images.
5. **VLM Picture Description (GPU only)**: Feeds diagrams, flowcharts, and math pictures to the SmolVLM model to get markdown-grounded text descriptions.
6. **Token/Whitespace Cleanup**: Removes base64 images and excessive spacing to optimize tokens for subsequent RAG embedding.

---

### 5.2 Semantic Chunking Engine

**Location:** `src/chunking/semantic_chunker.py`

Splits cleaned Markdown into semantically coherent chunks based on heading structure.

| Parameter | Value | Purpose |
|---|---|---|
| `MAX_TOKENS` | 1000 | Maximum tokens per chunk |
| `MIN_TOKENS` | 80 | Minimum tokens — smaller sections are merged |
| `OVERLAP_RATIO` | 0.17 | 17% overlap between split sub-chunks |
| `ENCODING_MODEL` | `cl100k_base` | Tiktoken encoding for token counting |

**Algorithm:**
1. **Pre-processing** — Clean slide artifacts (empty headings, duplicates, base64 noise)
2. **Section Splitting** — Split at every `##` heading boundary
3. **Merge Small** — Sections < `MIN_TOKENS` merged with the next section
4. **Split Large** — Sections > `MAX_TOKENS` recursively split using `RecursiveCharacterTextSplitter`

**Output per chunk:**
```python
{
    "chunk_id": "Python_Session_1.md_c0",
    "chunk_content": "## Heading\n\nBody text...",
    "source": "Python_Session_1.md",
    "section_heading": "Heading",
    "chunk_index": 0,
    "token_count": 450
}
```

---

### 5.3 Embedding & Vector Storage

#### Embedding Model
| Property | Value |
|---|---|
| Model | `Alibaba-NLP/gte-multilingual-base` |
| Type | Local (runs on device) |
| Dimensions | 768 |
| Multilingual | ✅ Arabic + English |

#### Vector Database — Qdrant
| Property | Value |
|---|---|
| Backend | Qdrant (local persistent storage) |
| Distance Method | Cosine Similarity |
| Storage Path | `src/assets/database/reali_qdrant_db/` |
| Collection Naming | `collection_{project_id}` |

---

### 5.4 RAG Pipeline (Retrieval + Generation)

**Location:** `src/controllers/NLPController.py`

The RAG pipeline implements a **two-stage retrieval** system:

```mermaid
graph LR
    Q[Query] --> EMB[GTE Embedding]
    EMB --> VS["Vector Search<br/>(Top 2×limit)"]
    VS --> RR["BGE Reranker<br/>(Cross-Encoder)"]
    RR --> TOP["Top-N Documents"]
    TOP --> PROMPT["Prompt Construction<br/>(System + Docs + Footer)"]
    PROMPT --> LLM["GPT-4o-mini<br/>(via OpenRouter)"]
    LLM --> ANS[Answer]
```

**Stage 1 — Vector Retrieval:**
- Embed query using GTE multilingual
- Search Qdrant for `2 × limit` candidates (over-retrieve)
- Returns documents with cosine similarity scores

**Stage 2 — BGE Reranking:**
- Model: `BAAI/bge-reranker-v2-m3` (Cross-Encoder)
- Re-scores all candidate pairs `[query, document]`
- Sort by reranker score descending
- Return top `limit` documents

**Stage 3 — Answer Generation:**
- Construct prompt from locale templates (`stores/llm/templates/locales/en/rag.py`)
- Send to `openai/gpt-4o-mini` via OpenRouter
- Return grounded answer

---

### 5.5 AI Agent System (CrewAI)

**Location:** `src/agent/`

The agent system uses **CrewAI** to create intelligent, tool-using agents that sit on top of the RAG pipeline.

#### Orchestrator Agent ("REAL_i")

| Property | Value |
|---|---|
| Role | Study Assistant |
| Personality | Encouraging, supportive, patient — a study companion |
| Language | Matches the student (Arabic or English) |
| Framework | CrewAI with OpenRouter LLM |
| Tools | Search Course Materials, Get RAG Answer, Generate Quiz |

**Behavioral Rules:**
- Uses RAG tools for subject-specific questions
- Responds directly for general conversation
- Delegates quiz generation to the Quiz sub-agent
- Injects active instructor guidelines into its reasoning

#### Quiz Generator Sub-Agent

| Property | Value |
|---|---|
| Role | Quiz Generator Specialist |
| Output | Structured JSON (`QuizModel`) |
| Format | MCQ with 4 options (A/B/C/D), correct answer, explanation |
| Source | Always grounded in retrieved course materials |

#### Session Manager

| Property | Value |
|---|---|
| Storage | In-memory (Python dict) |
| Max History | 50 messages per session |
| Session ID | UUID4 |
| Persistence | Ephemeral (lost on server restart) |

---

### 5.6 Admin Agent Platform

**Location:** `admin_agent_platform/`

A separate FastAPI service that processes instructor requests using CrewAI.

#### Workflow
1. Instructor sends a natural language request (e.g., "Create a quiz about ML Chapter 3 with 20 MCQs")
2. CrewAI Admin Crew analyzes intent → extracts parameters → formats structured JSON
3. Record is written to Google Sheets (Shared Memory)
4. Webhook notification is sent to the REAL_i Assistant

#### Google Sheets Schema ("Shared Memory")
| Column | Description |
|---|---|
| Task_ID | Auto-generated (T001, T002, ...) |
| Task_Type | Quiz, Assignment, Flashcards, Study Guide, Summary, Exam |
| Description | Clear description of the task |
| Course | Subject name (default: "General") |
| Priority | High, Medium, Low |
| Assigned_Agent | Always "TA" |
| Status | Pending, In Progress, Completed |
| Created_At | Timestamp |
| Notes | Extracted parameters (MCQ count, chapters, etc.) |

---

### 5.7 Inter-Agent Communication

The Admin Agent and Assistant Agent communicate through two mechanisms:

#### 1. Webhook Trigger (Real-Time)
When the Admin Agent creates a task, it immediately sends an HTTP POST to the Assistant Agent's webhook endpoint. This stores the instructor's guideline in MongoDB as an active directive.

#### 2. Active Guidelines Injection
When a student chats with the Assistant Agent, it queries MongoDB for all active guidelines for the student's `project_id`. If any exist, they are injected into the orchestrator's task description, steering the conversation toward the instructor's objectives.

```mermaid
graph TB
    subgraph "Communication Layer"
        direction LR
        GS3["Google Sheets<br/>(Persistent Record)"]
        MDB["MongoDB<br/>(Active Guidelines)"]
        WH2["Webhook HTTP POST<br/>(Real-Time Trigger)"]
    end

    ADMIN2[Admin Agent] -->|"1. Write record"| GS3
    ADMIN2 -->|"2. Trigger"| WH2
    WH2 -->|"3. Save guideline"| MDB
    MDB -->|"4. Query on chat"| REAL_i2[REAL_i Agent]
```

---

## 6. Database Architecture

### 6.1 MongoDB Collections

```mermaid
erDiagram
    projects {
        ObjectId _id PK
        string project_id UK "Alphanumeric identifier"
    }

    assets {
        ObjectId _id PK
        ObjectId asset_project_id FK
        string asset_type "FILE"
        string asset_name UK "Unique file identifier"
        int asset_size "Bytes"
        dict asset_config
        datetime asset_pushed_at
    }

    chunks {
        ObjectId _id PK
        string chunk_text "Chunk content"
        dict chunk_metadata "section_heading, token_count, source"
        int chunk_order "1-indexed order"
        ObjectId chunk_project_id FK
        ObjectId chunk_asset_id FK
    }

    instructor_guidelines {
        ObjectId _id PK
        string project_id FK "Normalized course name"
        string task_id UK "e.g. T001"
        string task_type "Quiz, Assignment, etc."
        string description "Instructor directive"
        string priority "High, Medium, Low"
        string status "Pending"
        string notes "Parameters"
        string created_at "Timestamp"
        bool is_active "true = actively guiding"
    }

    projects ||--o{ assets : "has"
    projects ||--o{ chunks : "has"
    projects ||--o{ instructor_guidelines : "has"
    assets ||--o{ chunks : "produced"
```

### 6.2 Vector Database (Qdrant)

Each project gets its own Qdrant collection named `collection_{project_id}`.

| Field | Type | Description |
|---|---|---|
| `id` | int | Auto-incremented chunk index |
| `vector` | float[768] | GTE embedding vector |
| `text` | string | Original chunk text |
| `metadata` | dict | section_heading, token_count, source, source_path |

---

## 7. API Reference

### 7.1 Base Routes

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/` | Health check — returns app name and version |

---

### 7.2 Data Routes (`/api/v1/data/`)

#### `POST /api/v1/data/upload/{project_id}`

Upload a file (PDF or TXT) to a project.

| Parameter | Type | Location | Description |
|---|---|---|---|
| `project_id` | string | Path | Alphanumeric project identifier |
| `file` | UploadFile | Body (multipart) | The file to upload |

**Response (200):**
```json
{
    "signal": "file_upload_success",
    "file_id": "648a1f2b..."
}
```

#### `POST /api/v1/data/process/{project_id}`

Extract text from uploaded files and store as chunks in MongoDB.

| Parameter | Type | Location | Description |
|---|---|---|---|
| `project_id` | string | Path | Project identifier |
| `file_id` | string? | Body | Specific file to process (optional — processes all if omitted) |
| `chunk_size` | int? | Body | Chunk size parameter (default: 100) |
| `overlap_size` | int? | Body | Overlap size parameter (default: 20) |
| `do_reset` | int? | Body | If 1, deletes existing chunks first |

**Response (200):**
```json
{
    "signal": "processing_success",
    "inserted_chunks": 42,
    "processed_files": 1
}
```

---

### 7.3 NLP Routes (`/api/v1/nlp/`)

#### `POST /api/v1/nlp/index/push/{project_id}`

Index all chunks from MongoDB into the Qdrant vector database.

| Parameter | Type | Location | Description |
|---|---|---|---|
| `project_id` | string | Path | Project identifier |
| `do_reset` | int? | Body | If 1, resets the vector collection |

**Response (200):**
```json
{
    "signal": "insert_into_vectordb_success",
    "inserted_items_count": 42
}
```

#### `GET /api/v1/nlp/index/info/{project_id}`

Get vector collection metadata.

**Response (200):**
```json
{
    "signal": "vectordb_collection_retrieved",
    "collection_info": { "vectors_count": 42, "status": "green" }
}
```

#### `POST /api/v1/nlp/index/search/{project_id}`

Perform semantic vector search with optional BGE reranking.

| Parameter | Type | Location | Description |
|---|---|---|---|
| `project_id` | string | Path | Project identifier |
| `text` | string | Body | Search query |
| `limit` | int? | Body | Number of results (default: 5) |

**Response (200):**
```json
{
    "signal": "vectordb_search_success",
    "results": [
        { "text": "...", "score": 0.892 }
    ]
}
```

#### `POST /api/v1/nlp/index/answer/{project_id}`

Full RAG pipeline: retrieve → rerank → generate answer.

| Parameter | Type | Location | Description |
|---|---|---|---|
| `project_id` | string | Path | Project identifier |
| `text` | string | Body | User's question |
| `limit` | int? | Body | Number of documents to retrieve (default: 5) |

**Response (200):**
```json
{
    "signal": "rag_answer_success",
    "answer": "Based on the course materials, ...",
    "full_prompt": "...",
    "chat_history": [...]
}
```

---

### 7.4 Agent Routes (`/api/v1/agent/`)

#### `POST /api/v1/agent/chat/{project_id}`

Multi-turn chat with the REAL_i Study Assistant.

| Parameter | Type | Location | Description |
|---|---|---|---|
| `project_id` | string | Path | Project identifier |
| `message` | string | Body | Student's message |
| `session_id` | string? | Body | Existing session ID (creates new if omitted) |

**Response (200):**
```json
{
    "response": "Hello! Based on Module 5...",
    "session_id": "a1b2c3d4-..."
}
```

#### `POST /api/v1/agent/quiz/{project_id}`

Generate a structured MCQ quiz from course materials.

| Parameter | Type | Location | Description |
|---|---|---|---|
| `project_id` | string | Path | Project identifier |
| `topic` | string | Body | Quiz topic |
| `num_questions` | int? | Body | Number of questions (default: 5) |

**Response (200):**
```json
{
    "quiz": {
        "topic": "Neural Networks",
        "questions": [
            {
                "question": "What is the activation function...?",
                "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
                "correct_answer": "B",
                "explanation": "B is correct because..."
            }
        ]
    }
}
```

#### `DELETE /api/v1/agent/session/{session_id}`

Clear a chat session's history.

**Response (200):**
```json
{ "status": "cleared" }
```

#### `POST /api/v1/agent/webhook/task`

Receive task notifications from the Admin Agent.

| Parameter | Type | Location | Description |
|---|---|---|---|
| `task_id` | string | Body | Task ID (e.g., "T003") |
| `description` | string | Body | Task description |
| `course` | string | Body | Course name (auto-normalized to project_id) |
| `task_type` | string | Body | Quiz, Assignment, etc. |
| `priority` | string | Body | High, Medium, Low |
| `notes` | string? | Body | Additional parameters |

**Response (200):**
```json
{
    "status": "success",
    "message": "Guideline successfully registered for project: machinelearning",
    "project_id": "machinelearning",
    "task_id": "T003"
}
```

---

### 7.5 Admin Agent API (`admin_agent_platform/`)

#### `POST /task/create`

Create a new task via natural language.

| Parameter | Type | Location | Description |
|---|---|---|---|
| `request` | string | Body | Natural language task request |

**Response (201):**
```json
{
    "task_id": "T003",
    "status": "created"
}
```

#### `GET /health`

Health check for the Admin Agent service.

**Response (200):**
```json
{ "status": "healthy", "service": "Admin Agent Platform" }
```

---

## 8. Configuration Reference

### 8.1 Core Platform (`src/.env`)

| Variable | Description | Example |
|---|---|---|
| `APP_NAME` | Application name | `REAL_i` |
| `APP_VERSION` | Application version | `0.1` |
| `OPENAI_API_KEY` | OpenRouter API key | `sk-or-v1-...` |
| `OPENAI_API_URL` | LLM API base URL | `https://openrouter.ai/api/v1` |
| `MONGODB_URL` | MongoDB Atlas connection string | `mongodb+srv://...` |
| `MONGODB_DATABASE` | Database name | `reali-db` |
| `GENERATION_BACKEND` | LLM provider | `OPENAI` |
| `EMBEDDING_BACKEND` | Embedding provider | `LOCAL` |
| `GENERATION_MODEL_ID` | Generation model | `openai/gpt-4o-mini` |
| `EMBEDDING_MODEL_ID` | Embedding model | `Alibaba-NLP/gte-multilingual-base` |
| `EMBEDDING_MODEL_SIZE` | Embedding dimensions | `768` |
| `VECTOR_DB_BACKEND` | Vector DB provider | `QDRANT` |
| `VECTOR_DB_PATH` | Vector DB storage path | `reali_qdrant_db` |
| `VECTOR_DB_DISTANCE_METHOD` | Similarity metric | `cosine` |
| `PRIMARY_LANG` | Primary template language | `en` |
| `INPUT_DAFAULT_MAX_CHARACTERS` | Max input characters | `1024` |
| `GENERATION_DAFAULT_MAX_TOKENS` | Max generation tokens | `200` |
| `GENERATION_DAFAULT_TEMPERATURE` | LLM temperature | `0.1` |

### 8.2 Admin Agent (`admin_agent_platform/.env`)

| Variable | Description | Example |
|---|---|---|
| `OPENAI_API_KEY` | OpenRouter API key | `sk-or-v1-...` |
| `OPENAI_API_URL` | LLM API base URL | `https://openrouter.ai/api/v1` |
| `GENERATION_MODEL_ID` | Model for task analysis | `openai/gpt-4o-mini` |
| `GOOGLE_SPREADSHEET_ID` | Google Sheets spreadsheet ID | `1wMtkgZ...` |
| `GOOGLE_CLIENT_ID` | OAuth2 client ID | `72549...` |
| `GOOGLE_CLIENT_SECRET` | OAuth2 client secret | `GOCSPX-...` |
| `ASSISTANT_WEBHOOK_URL` | REAL_i webhook endpoint | `http://localhost:8000/api/v1/agent/webhook/task` |

---

## 9. Directory Structure

```
REAL_i-Graduation-Project/
│
├── src/                                    # 🧠 Core Platform
│   ├── main.py                             # FastAPI app + lifespan (startup/shutdown)
│   ├── .env                                # Environment configuration
│   │
│   ├── agent/                              # 🤖 AI Agent System
│   │   ├── __init__.py                     # Exports + SessionManager singleton
│   │   ├── crew_runner.py                  # CrewAI execution engine (Orchestrator + Quiz)
│   │   ├── prompts.py                      # System prompts for both agents
│   │   ├── session_manager.py              # In-memory chat session store
│   │   └── tools.py                        # CrewAI tool factories (RAG wrappers)
│   │
│   ├── controllers/                        # 🎮 Business Logic Controllers
│   │   ├── BaseController.py               # Base class (paths, config)
│   │   ├── DataController.py               # File validation
│   │   ├── NLPController.py                # RAG: embed, search, rerank, answer
│   │   ├── ProcessController.py            # PDF extraction + chunking orchestration
│   │   └── ProjectController.py            # Project directory management
│   │
│   ├── extraction/                         # 📄 PDF → Markdown Pipeline
│   │   ├── docling_pipeline.py             # Docling structural extraction
│   │   ├── local_pdf_pipeline.py           # PyMuPDF + PaddleOCR + Ollama Vision
│   │   └── merged_pipeline.py              # Hybrid orchestrator pipeline
│   │
│   ├── chunking/                           # 🔪 Text Chunking
│   │   └── semantic_chunker.py             # Heading-based semantic chunking
│   │
│   ├── search/                             # 🔍 Search Utilities
│   │   └── reranked_search.py              # Standalone two-stage search script
│   │
│   ├── routes/                             # 🌐 API Routes
│   │   ├── base.py                         # GET / (health check)
│   │   ├── data.py                         # Upload + Process endpoints
│   │   ├── nlp.py                          # Index, Search, Answer endpoints
│   │   ├── agent.py                        # Chat, Quiz, Webhook endpoints
│   │   └── schemes/                        # Pydantic request/response schemas
│   │       ├── data.py
│   │       ├── nlp.py
│   │       └── agent.py
│   │
│   ├── models/                             # 📦 MongoDB Models
│   │   ├── ProjectModel.py
│   │   ├── ChunkModel.py
│   │   ├── AssetModel.py
│   │   ├── InstructorGuidelineModel.py
│   │   ├── BaseDataModel.py
│   │   ├── db_schemes/                     # Pydantic schemas for MongoDB documents
│   │   │   ├── project.py
│   │   │   ├── data_chunk.py
│   │   │   ├── asset.py
│   │   │   └── instructor_guideline.py
│   │   └── enums/
│   │       ├── DataBaseEnum.py
│   │       ├── ResponseEnums.py
│   │       ├── AssetTypeEnum.py
│   │       └── ProcessingEnum.py
│   │
│   ├── stores/                             # 🏪 External Service Providers
│   │   ├── llm/                            # LLM Providers
│   │   │   ├── LLMInterface.py             # Abstract base class
│   │   │   ├── LLMEnums.py                 # Provider + role enums
│   │   │   ├── LLMProviderFactory.py       # Factory pattern
│   │   │   ├── providers/
│   │   │   │   ├── OpenAIProvider.py        # OpenAI/OpenRouter implementation
│   │   │   │   ├── CoHereProvider.py        # Cohere implementation
│   │   │   │   └── LocalEmbeddingProvider.py# Local GTE embedding
│   │   │   └── templates/
│   │   │       ├── template_parser.py       # Locale-based template loading
│   │   │       └── locales/en/
│   │   │           ├── rag.py               # RAG prompt templates
│   │   │           └── quiz.py              # Quiz generation templates
│   │   └── vectordb/                       # Vector DB Providers
│   │       ├── VectorDBInterface.py         # Abstract base class
│   │       ├── VectorDBEnums.py             # Provider enums
│   │       ├── VectorDBProviderFactory.py   # Factory pattern
│   │       └── providers/
│   │           ├── QdrantDBProvider.py       # Qdrant implementation
│   │           └── ChromaDBProvider.py       # ChromaDB implementation
│   │
│   ├── helpers/                            # 🔧 Utilities
│   │   ├── config.py                       # Pydantic Settings loader
│   │   └── db_init.py                      # MongoDB collection + index initializer
│   │
│   └── assets/                             # 📁 Local Storage
│       ├── files/{project_id}/             # Uploaded files per project
│       └── database/reali_qdrant_db/        # Qdrant persistent storage
│
├── admin_agent_platform/                   # 👨‍🏫 Admin Agent (Separate Service)
│   ├── main.py                             # FastAPI app (port 8001)
│   ├── crew.py                             # CrewAI Admin Crew + webhook trigger
│   ├── google_sheets_writer.py             # Google Sheets API integration
│   ├── .env                                # Admin agent configuration
│   ├── requirements.txt
│   └── test_crew.py                        # Test script
│
├── environment.yml                         # Conda environment specification
├── requirements.txt                        # Python dependencies
├── setup_env.sh                            # Environment setup script
└── README.md
```

---

## 10. Deployment & Running

### 10.1 Prerequisites

- Python 3.10+ (Conda recommended)
- MongoDB Atlas account (or local MongoDB)
- OpenRouter API key
- Google Cloud OAuth credentials (for Admin Agent)

### 10.2 Running the Core Platform

```bash
cd src
uvicorn main:app --reload --port 8000
```

The server starts up by:
1. Connecting to MongoDB Atlas
2. Initializing LLM provider (OpenRouter → GPT-4o-mini)
3. Initializing local embedding model (GTE Multilingual, 768-dim)
4. Connecting to Qdrant vector database
5. Loading BGE Reranker cross-encoder model
6. Setting up template parser (English locale)

**API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### 10.3 Running the Admin Agent

```bash
cd admin_agent_platform
uvicorn main:app --reload --port 8001
```

**API Docs:** [http://localhost:8001/docs](http://localhost:8001/docs)

### 10.4 Typical Usage Flow

```bash
# 1. Upload a PDF
curl -X POST http://localhost:8000/api/v1/data/upload/ml_course \
  -F "file=@lecture_5.pdf"

# 2. Process the PDF (extract + chunk)
curl -X POST http://localhost:8000/api/v1/data/process/ml_course \
  -H "Content-Type: application/json" \
  -d '{"do_reset": 0}'

# 3. Index chunks into vector DB
curl -X POST http://localhost:8000/api/v1/nlp/index/push/ml_course \
  -H "Content-Type: application/json" \
  -d '{"do_reset": 0}'

# 4. Chat with REAL_i
curl -X POST http://localhost:8000/api/v1/agent/chat/ml_course \
  -H "Content-Type: application/json" \
  -d '{"message": "What is gradient descent?"}'

# 5. Generate a quiz
curl -X POST http://localhost:8000/api/v1/agent/quiz/ml_course \
  -H "Content-Type: application/json" \
  -d '{"topic": "neural networks", "num_questions": 5}'
```

## 11. RAG Evaluation Framework & Performance Metrics

To ensure the reliability, accuracy, and performance of the REAL_i RAG system for the graduation project presentation, a comprehensive evaluation framework is built directly into the codebase. 

### 11.1 Framework Structure

The evaluation suite is located in `src/evaluation/` and consists of the following components:

- **`generate_dataset.py`**: Retrieves course document chunks from MongoDB and uses the production LLM (`gpt-4o-mini`) with robust parsing fallbacks to generate a static evaluation dataset of 15 Question-Answer-Context triples.
- **`evaluate_rag.py`**: The evaluation runner script that loads the dataset, executes tests on both the retrieval and generation stages, computes multiple metrics, and compares results.
- **`test_dataset.json`**: The static evaluation Q&A dataset representing standard student queries and textbook answers.
- **`evaluation_results.json`**: Output containing granular metrics and generated answers for every test case.

---

### 11.2 Evaluation Metrics

The framework evaluates the RAG pipeline across three dimensions:

#### 1. Retrieval Quality (Vector Search vs BGE Reranker)
This evaluates how effectively the vector database retrieves the exact textbook chunk containing the answer. We compare standard cosine similarity vector search alone against our two-stage pipeline with a **BGE Cross-Encoder Reranker** (`BAAI/bge-reranker-v2-m3`).

| Metric | Description | Vector Search (GTE Only) | Reranked Search (GTE + BGE) |
| :--- | :--- | :---: | :---: |
| **Hit Rate @ 1** | Is the correct document ranked 1st? | 66.7% | **100.0%** |
| **Hit Rate @ 3** | Is the correct document in the top 3? | 100.0% | **100.0%** |
| **Hit Rate @ 5** | Is the correct document in the top 5? | 100.0% | **100.0%** |
| **MRR** | Mean Reciprocal Rank of the correct document | 0.8222 | **1.0000** |
| **NDCG @ 5** | Normalized Discounted Cumulative Gain | 0.8682 | **1.0000** |

* **Key Outcome**: The BGE Reranker successfully boosts Hit Rate @ 1 to **100.0%**, ensuring that the AI assistant always reads the exact, relevant section of the textbook before generating answers.

#### 2. Generation Quality (RAG vs Raw LLM Baseline)
This evaluates the quality, accuracy, and truthfulness of the generated answers. We compare the RAG system output against a raw LLM (`gpt-4o-mini` without textbook context).

| Metric | Evaluation Method | RAG System | Raw LLM Baseline |
| :--- | :--- | :---: | :---: |
| **Faithfulness** | LLM-as-a-Judge (Detects hallucinations) | **100.0%** | N/A |
| **Answer Relevancy** | LLM-as-a-Judge (Directly answers question) | **99.3%** | 100.0% |
| **Context Relevancy** | LLM-as-a-Judge (Text relevance filter) | **90.7%** | N/A |
| **ROUGE-L F1** | Overlap with ground-truth textbook answers | **0.5356** | 0.3972 |
| **Semantic Similarity**| Cosine similarity of text embeddings | **0.8961** | 0.8553 |

* **Key Outcome**: Grounding the model in course documents achieves **100.0% Faithfulness** (hallucination-free) while matching the textbook content significantly better (ROUGE-L F1 of **0.5356** vs **0.3972**).

#### 3. System Latency & Performance (CPU on WSL)
- **Average Retrieval Latency (GTE + BGE)**: 7.229 seconds
- **Average Generation Latency**: 9.292 seconds (includes LLM-as-a-Judge evaluation calls)
- **Average Total Request Latency**: 16.521 seconds
- **P95 Latency**: 24.809 seconds

---

### 11.3 Pipeline Optimizations & Bug Fixes
The evaluation framework identified and helped resolve two critical production issues:
1. **RAG Refusal Bug**: We discovered the question was missing from the LLM prompt. We updated `src/controllers/NLPController.py` to dynamically inject the question.
2. **Prompt Truncation**: The environment variable `INPUT_DAFAULT_MAX_CHARACTERS` in `.env` was limiting inputs to `1024` characters, truncating long textbook contexts. We increased this to `16384`.
* **Fix Impact**: These changes raised **Answer Relevancy from 26% to 99.3%** and **Faithfulness from 55.3% to 100.0%**.

---

### 11.4 Running the Evaluation

To run the evaluation suite locally:
1. Stop the FastAPI server (to unlock the local Qdrant SQLite files).
2. Execute the evaluation script inside `src/`:
   ```bash
   wsl /home/gohar/miniconda3/envs/raaed/bin/python evaluation/evaluate_rag.py
   ```
3. Granular results are saved in `src/evaluation/evaluation_results.json`.

---

> _This documentation was generated for the REAL_i Graduation Project — Digital Pioneers Initiative._  
> _Last updated: June 2026_
