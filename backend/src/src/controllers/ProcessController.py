from .BaseController import BaseController
from .ProjectController import ProjectController
import os
from pathlib import Path
from models import ProcessingEnum
from ._extraction import run_unified_pipeline

class CustomDocument:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata

class ProcessController(BaseController):

    def __init__(self, project_id: str):
        super().__init__()
        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=project_id)

    def get_file_extension(self, file_id: str):
        return os.path.splitext(file_id)[-1].lower()

    def get_file_content(self, file_id: str):
        file_ext = self.get_file_extension(file_id=file_id)
        file_path = os.path.join(self.project_path, file_id)

        if not os.path.exists(file_path):
            return None

        if file_ext == ProcessingEnum.TXT.value:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                print(f"Error reading text file: {e}")
                return None

        if file_ext == ProcessingEnum.PDF.value:
            # Create a separate output directory for this project
            output_dir = os.path.join(self.project_path, "output")
            os.makedirs(output_dir, exist_ok=True)

            try:
                # Run unified GPU-accelerated pipeline (Docling + SmolVLM)
                result = run_unified_pipeline(
                    source=file_path,
                    output_dir=Path(output_dir),
                    fast_tables=True,
                )

                if result is None:
                    return None

                # Directly use the returned markdown (no disk I/O needed)
                markdown, json_data, report = result
                return markdown if markdown else None
            except Exception as e:
                print(f"Error during unified pipeline processing: {e}")
                return None
        
        return None

    def process_file_content(self, file_content: str, file_id: str,
                             chunk_size: int = 100, overlap_size: int = 20):
        if not file_content:
            return []

        documents = [
            {
                "page_content": file_content,
                "source": file_id
            }
        ]

        # Run semantic chunking (which is heading-based and cleans slides artifacts)
        chunks = chunk_documents(documents)

        # Wrap in CustomDocument objects to conform to the route expectations
        doc_chunks = []
        for i, chunk in enumerate(chunks):
            doc_chunks.append(
                CustomDocument(
                    page_content=chunk["chunk_content"],
                    metadata={
                        "section_heading": chunk.get("section_heading", ""),
                        "chunk_index": chunk.get("chunk_index", i),
                        "token_count": chunk.get("token_count", 0),
                        "source": chunk.get("source", file_id),
                        "source_path": chunk.get("source_path", file_id)
                    }
                )
            )

        return doc_chunks


# ============================================================
# SEMANTIC CHUNKING LOGIC (Inlined)
# ============================================================


import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

MAX_TOKENS: int      = 1000
MIN_TOKENS: int      = 80
OVERLAP_RATIO: float = 0.17
OVERLAP_TOKENS: int  = int(MAX_TOKENS * OVERLAP_RATIO)

ENCODING_MODEL: str = "cl100k_base"
ENCODER = tiktoken.get_encoding(ENCODING_MODEL)

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")


# ──────────────────────────────────────────────
# Token counting
# ──────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """
    Count tokens in a text string using tiktoken.

    Args:
        text (str): Input text.

    Returns:
        int: Token count.

    Raises:
        TypeError: If text is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return len(ENCODER.encode(text))


# ──────────────────────────────────────────────
# Pre-processing — fix Atef's slide-based output
# ──────────────────────────────────────────────

def _is_heading(line: str) -> bool:
    """Return True if the line is a Markdown heading."""
    return bool(HEADING_PATTERN.match(line.strip()))

def _clean_markdown(text: str) -> str:
    """
    Clean Docling Markdown output before chunking.
    """
    # Remove full markdown image tags with base64 data
    text = re.sub(
        r'!\[.*?\]\(data:image\/[^;]+;base64,.*?\)',
        '',
        text,
        flags=re.DOTALL
    )

    # Remove any remaining very long base64-like strings
    text = re.sub(
        r'[A-Za-z0-9+/]{300,}={0,2}',
        '',
        text
    )

    # Remove broken leftover image markdown parts
    text = re.sub(
        r'!\[.*?\]\(.*?\)',
        '',
        text,
        flags=re.DOTALL
    )

    # Pass 0: Remove specific "noise" lines (Slide furniture)
    noise_patterns = [
        r"## Digilians",
        r"Session \d+\s*:",
        r"## Session Agenda",
        r"- Python programming language",
        r"- Data Containers",
    ]
    
    lines = text.split("\n")
    cleaned_noise = []
    for line in lines:
        if any(re.search(pat, line, re.IGNORECASE) for pat in noise_patterns):
            continue
        cleaned_noise.append(line)
    
    lines = cleaned_noise
    processed: List[str] = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # If it's a heading
        if HEADING_PATTERN.match(stripped):
            # Find next non-blank line
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            
            # If next non-blank is also a heading, skip THIS heading
            if j < len(lines) and HEADING_PATTERN.match(lines[j].strip()):
                i = j
                continue
            
            # If THIS heading is identical to the last heading added, skip it
            # (First we need to find the last heading in 'processed')
            last_h = None
            for prev_line in reversed(processed):
                if prev_line.strip():
                    if HEADING_PATTERN.match(prev_line.strip()):
                        last_h = prev_line.strip().lower()
                    break
            
            if last_h == stripped.lower():
                # Skip duplicate heading
                i += 1
                continue
                
        processed.append(line)
        i += 1

    cleaned = "\n".join(processed)

    # Remove lines that are mostly unreadable/base64 fragments
    cleaned_lines = []
    for line in cleaned.splitlines():
        stripped = line.strip()

        if len(stripped) > 150 and re.fullmatch(r"[A-Za-z0-9+/=]+", stripped):
            continue

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)

    # Collapse 3+ blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()
# ──────────────────────────────────────────────
# Section splitting
# ──────────────────────────────────────────────

def _split_into_sections(text: str) -> List[Tuple[str, str]]:
    """
    Split Markdown into sections at every heading boundary.

    Args:
        text (str): Cleaned Markdown.

    Returns:
        List of (heading_line, body) tuples.
    """
    sections: List[Tuple[str, str]] = []
    matches = list(re.finditer(r"^(#{1,6})\s+(.+)$", text, re.MULTILINE))

    if not matches:
        return [("", text.strip())]

    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    for i, match in enumerate(matches):
        heading_line = match.group(0).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append((heading_line, body))

    return sections


def _merge_small_sections(
    sections: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """
    Merge sections smaller than MIN_TOKENS into the following section.
    """
    if not sections:
        return []

    merged: List[Tuple[str, str]] = []
    i = 0

    while i < len(sections):
        curr_heading, curr_body = sections[i]
        
        # Calculate tokens for the current consolidated section
        combined_text = f"{curr_heading}\n\n{curr_body}".strip() if curr_heading else curr_body
        
        if count_tokens(combined_text) < MIN_TOKENS and i + 1 < len(sections):
            next_heading, next_body = sections[i + 1]
            
            # Merge current into next
            # We preserve the first non-empty heading
            new_heading = curr_heading or next_heading
            
            # The body of the next section now includes the current heading (if any) and current body
            new_body = combined_text
            if next_heading and next_heading != curr_heading:
                new_body += f"\n\n{next_heading}"
            if next_body:
                new_body += f"\n\n{next_body}"
            
            # Remove the current heading from the start of new_body if it's the same as new_heading
            # because chunk_documents will prepend it again
            if new_heading and new_body.startswith(new_heading):
                new_body = new_body[len(new_heading):].strip()

            sections[i + 1] = (new_heading, new_body)
        else:
            merged.append((curr_heading, curr_body))
        
        i += 1

    return merged


# ──────────────────────────────────────────────
# Oversized section splitter
# ──────────────────────────────────────────────

_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name=ENCODING_MODEL,
    chunk_size=MAX_TOKENS,
    chunk_overlap=OVERLAP_TOKENS,
    separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
)


def _split_large_section(heading: str, body: str) -> List[str]:
    """
    Split a section that exceeds MAX_TOKENS.
    Heading is prepended to every sub-chunk so each is self-contained.

    Args:
        heading : Section heading line.
        body    : Section body text.

    Returns:
        List of chunk strings.
    """
    full_text = f"{heading}\n\n{body}".strip() if heading else body

    if count_tokens(full_text) <= MAX_TOKENS:
        return [full_text]

    sub_chunks = _splitter.split_text(body)

    if heading:
        return [f"{heading}\n\n{sub.strip()}" for sub in sub_chunks if sub.strip()]
    return [sub.strip() for sub in sub_chunks if sub.strip()]


# ──────────────────────────────────────────────
# Core chunking function
# ──────────────────────────────────────────────

def chunk_documents(
    documents: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Semantically chunk Markdown documents by heading structure.

    Applies pre-processing to remove empty slide-title headings
    from Atef's Docling output before splitting into chunks.

    Expected input (Atef's output format):
        [
            {
                "page_content": "## Heading\\n\\nBody...",
                "source": "Python_Session_1.md",
                "page_number": 1          # optional
            }
        ]

    Returns:
        List[Dict]: Each dict contains:
            - "chunk_id"        (str) : e.g. 'Python_Session_1.md_c0'
            - "chunk_content"   (str) : Heading + body text
            - "source"          (str) : Source file name
            - "source_path"     (str) : Full original path
            - "section_heading" (str) : The ## heading label
            - "chunk_index"     (int) : Global chunk index
            - "token_count"     (int) : Number of tokens in the chunk
            - "page_number"     (int) : Page number if provided

    Raises:
        TypeError:  If documents is not a list or fields have wrong types.
        ValueError: If a document is missing required keys.
    """
    if not isinstance(documents, list):
        raise TypeError("documents must be a list of dictionaries")

    required_keys = {"page_content", "source"}
    all_chunks: List[Dict[str, Any]] = []
    global_index: int = 0

    for doc_index, doc in enumerate(documents):
        if not isinstance(doc, dict):
            raise TypeError(f"Document at index {doc_index} must be a dictionary")

        missing = required_keys - doc.keys()
        if missing:
            raise ValueError(
                f"Document at index {doc_index} is missing keys: {missing}."
            )

        raw_text: str              = doc["page_content"]
        source: str                = doc["source"]
        page_number: Optional[int] = doc.get("page_number")

        if not isinstance(raw_text, str):
            raise TypeError(f"'page_content' must be str at index {doc_index}")
        if not isinstance(source, str):
            raise TypeError(f"'source' must be str at index {doc_index}")

        raw_text = raw_text.strip()
        if not raw_text:
            continue

        source_name: str = Path(source).name

        # Step 0: remove empty slide-title headings
        clean_text = _clean_markdown(raw_text)

        # Step 1: split on headings
        sections = _split_into_sections(clean_text)

        # Step 2: merge tiny sections
        sections = _merge_small_sections(sections)

        # Step 3: build final chunks
        for heading, body in sections:
            chunk_texts = _split_large_section(heading, body)

            for chunk_text in chunk_texts:
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue

                first_line = chunk_text.split("\n")[0].strip()
                section_heading = (
                    first_line.lstrip("#").strip()
                    if first_line.startswith("#")
                    else heading.lstrip("#").strip()
                )

                chunk_dict: Dict[str, Any] = {
                    "chunk_id":        f"{source_name}_c{global_index}",
                    "chunk_content":   chunk_text,
                    "source":          source_name,
                    "source_path":     source,
                    "section_heading": section_heading,
                    "chunk_index":     global_index,
                    "token_count":     count_tokens(chunk_text),
                }

                if page_number is not None:
                    chunk_dict["page_number"] = page_number

                all_chunks.append(chunk_dict)
                global_index += 1

    return all_chunks


# ──────────────────────────────────────────────
# Utility: print chunk stats
# ──────────────────────────────────────────────

def print_chunk_summary(chunks: List[Dict[str, Any]]) -> None:
    """
    Print a formatted summary of the generated chunks.

    Args:
        chunks (List[Dict]): Output of chunk_documents().
    """
    if not chunks:
        print("No chunks generated.")
        return

    token_counts = [c["token_count"] for c in chunks]
    sources = sorted({c["source"] for c in chunks})

    print("=" * 60)
    print("Semantic Chunking Summary")
    print("=" * 60)
    print(f"Total chunks : {len(chunks)}")
    print(f"Sources      : {', '.join(sources)}")
    print(f"Min tokens   : {min(token_counts)}")
    print(f"Max tokens   : {max(token_counts)}")
    print(f"Avg tokens   : {sum(token_counts) / len(token_counts):.2f}")
    print("=" * 60)
    print(f"\n{'#':<4} {'Heading':<38} {'Tokens':>6}")
    print("-" * 52)
    for c in chunks:
        heading = (c["section_heading"] or "(no heading)")[:36]
        print(f"{c['chunk_index']:<4} {heading:<38} {c['token_count']:>6}")
    print("=" * 60)


# ──────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────

