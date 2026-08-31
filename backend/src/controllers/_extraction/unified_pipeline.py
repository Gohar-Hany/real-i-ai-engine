"""
Unified PDF Extraction Pipeline
================================
A single, optimized document extraction pipeline built on Docling with native
GPU-accelerated VLM picture descriptions. Replaces the previous 3-file setup
(docling_pipeline, local_pdf_pipeline, merged_pipeline).

Features:
  - OCR via EasyOCR (GPU-accelerated)
  - Table structure extraction (accurate or fast mode)
  - Code block detection & formula enrichment
  - Native VLM picture description (SmolVLM-256M on GPU)
  - Picture classification
  - Page-chunked conversion for large PDFs
  - Validation report with coverage metrics

Optimized for T4 GPU (16GB VRAM).
"""

from __future__ import annotations

import copy
import json
import logging
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any
try:
    from pypdf import PdfReader
except ImportError:
    import subprocess
    import sys
    print("[Runtime Setup] Installing missing dependency: pypdf...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])
    from pypdf import PdfReader
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    EasyOcrOptions,
    PdfPipelineOptions,
    PictureDescriptionVlmEngineOptions,
    TableStructureOptions,
)
from docling.datamodel.settings import settings
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)
from docling_core.types.doc.base import ImageRefMode

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
log = logging.getLogger("unified_pipeline")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PAGE_CHUNK_SIZE = 20

PICTURE_DESCRIPTION_PROMPT = (
    "You are a precise document content extractor. "
    "Analyze this image and extract ONLY the valuable textual content it contains. "
    "Follow these rules strictly:\n"
    "- If the image contains a mathematical formula or equation, transcribe it in LaTeX format (e.g. $E = mc^2$).\n"
    "- If it is a chart or graph, describe the chart type, axes, data series, trends, and key values.\n"
    "- If it is a diagram or flowchart, describe every component, label, and connection/flow.\n"
    "- If it contains text (like a slide or document), transcribe ALL the text content.\n"
    "- If it is purely decorative or a logo with no educational/informational value, respond with exactly: [NO_CONTENT]\n"
    "- Do NOT describe the visual appearance of the image itself.\n"
    "- Return ONLY the extracted content — no commentary, no preamble."
)

# Pre-compiled regex patterns for speed
_RE_EXCESS_NEWLINES = re.compile(r'\n{4,}')
_RE_IMAGE_PLACEHOLDER = "<!-- image -->"
_RE_NO_CONTENT = "[NO_CONTENT]"


# ─────────────────────────────────────────────────────────────────────────────
# Validation Report
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ValidationReport:
    """Post-conversion quality and completeness report."""

    source: str = ""
    total_pages_expected: int = 0
    total_pages_extracted: int = 0
    total_text_elements: int = 0
    total_tables: int = 0
    total_pictures: int = 0
    total_code_blocks: int = 0
    empty_pages: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    processing_time_sec: float = 0.0
    status: str = "UNKNOWN"

    @property
    def coverage_pct(self) -> float:
        if self.total_pages_expected == 0:
            return 0.0
        return (self.total_pages_extracted / self.total_pages_expected) * 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "status": self.status,
            "processing_time_sec": round(self.processing_time_sec, 2),
            "page_coverage": {
                "expected": self.total_pages_expected,
                "extracted": self.total_pages_extracted,
                "coverage_pct": round(self.coverage_pct, 1),
                "empty_pages": self.empty_pages,
            },
            "content_counts": {
                "text_elements": self.total_text_elements,
                "tables": self.total_tables,
                "pictures": self.total_pictures,
                "code_blocks": self.total_code_blocks,
            },
            "warnings": self.warnings,
            "errors": self.errors,
        }

    def summary(self) -> str:
        lines = [
            f"  Status          : {self.status}",
            f"  Processing Time : {self.processing_time_sec:.1f}s",
            f"  Page Coverage   : {self.total_pages_extracted}/{self.total_pages_expected} ({self.coverage_pct:.0f}%)",
            f"  Text Elements   : {self.total_text_elements}",
            f"  Tables          : {self.total_tables}",
            f"  Pictures        : {self.total_pictures}",
            f"  Code Blocks     : {self.total_code_blocks}",
        ]
        if self.empty_pages:
            lines.append(f"  Empty Pages     : {self.empty_pages}")
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"    ⚠  {w}")
        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"    ✗  {e}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Builder — Optimized for T4 GPU
# ─────────────────────────────────────────────────────────────────────────────
def build_pipeline_options(
    no_code: bool = False,
    fast_tables: bool = False,
) -> PdfPipelineOptions:
    """
    PdfPipelineOptions optimized for T4 GPU (16GB VRAM).
    Enables ALL features: OCR, tables, code, formulas, picture descriptions.
    """
    # Global performance — T4 has enough VRAM for higher concurrency
    settings.perf.doc_batch_concurrency = 2
    settings.perf.page_batch_size = 16
    settings.perf.elements_batch_size = 16

    import torch
    cuda_available = torch.cuda.is_available()

    opts = PdfPipelineOptions()

    if cuda_available:
        # ── Accelerator: Force CUDA ──────────────────────────────────────────
        opts.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device="cuda",
        )
        # ── OCR: EasyOCR with GPU ────────────────────────────────────────────
        opts.do_ocr = True
        opts.ocr_options = EasyOcrOptions(
            lang=["en"],
            use_gpu=True,
            confidence_threshold=0.4,
        )
        log.info("OCR engine: EasyOCR (GPU)")
    else:
        # ── Accelerator: CPU fallback ────────────────────────────────────────
        opts.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device="cpu",
        )
        # ── OCR: EasyOCR with CPU ────────────────────────────────────────────
        opts.do_ocr = True
        opts.ocr_options = EasyOcrOptions(
            lang=["en"],
            use_gpu=False,
            confidence_threshold=0.4,
        )
        log.info("OCR engine: EasyOCR (CPU) - CUDA not available")

    # ── Table Structure ──────────────────────────────────────────────────
    opts.do_table_structure = True
    if fast_tables:
        opts.table_structure_options = TableStructureOptions(
            do_cell_matching=True,
            mode="fast",
        )
    else:
        opts.table_structure_options = TableStructureOptions(
            do_cell_matching=True,
            mode="accurate",
        )

    # ── Code & Formula Enrichment ────────────────────────────────────────
    opts.do_code_enrichment = not no_code
    opts.do_formula_enrichment = True

    # ── Image Handling ───────────────────────────────────────────────────
    opts.generate_picture_images = True
    opts.generate_page_images = False
    opts.images_scale = 1.5  # Good balance of quality vs speed

    # ── Native VLM Picture Description ─────────────────
    if cuda_available:
        opts.do_picture_description = True
        opts.picture_description_options = PictureDescriptionVlmEngineOptions.from_preset(
            "smolvlm"  # SmolVLM-256M — fast, fits easily in T4 VRAM
        )
        opts.picture_description_options.prompt = PICTURE_DESCRIPTION_PROMPT
        opts.picture_description_options.picture_area_threshold = 0.03
        log.info("VLM picture descriptions: ENABLED (SmolVLM on GPU)")
    else:
        opts.do_picture_description = False
        log.warning("VLM picture descriptions: DISABLED (CUDA not available, skipping VLM to avoid CPU freeze)")

    # ── Picture Classification ───────────────────────────────────────────
    opts.do_picture_classification = True

    opts.document_timeout = None
    return opts


@lru_cache(maxsize=4)
def _build_converter_cached(
    no_code: bool,
    fast_tables: bool,
) -> DocumentConverter:
    """
    Build and cache the DocumentConverter.
    Uses lru_cache to avoid re-loading heavy ML models on repeated calls.
    The cache key is the combination of (no_code, fast_tables).
    """
    pipeline_opts = build_pipeline_options(no_code=no_code, fast_tables=fast_tables)
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_opts),
        }
    )


def build_converter(
    no_code: bool = False,
    fast_tables: bool = False,
) -> DocumentConverter:
    """Public API to get a (cached) DocumentConverter."""
    return _build_converter_cached(no_code=no_code, fast_tables=fast_tables)


# ─────────────────────────────────────────────────────────────────────────────
# Document Processing
# ─────────────────────────────────────────────────────────────────────────────
def get_pdf_page_count(source: str) -> int:
    """Get page count using pypdf (fast, no heavy dependencies)."""
    try:
        return len(PdfReader(source).pages)
    except Exception as e:
        log.warning(f"Could not read page count: {e}")
        return 0


def process_document(
    converter: DocumentConverter,
    source: str,
    chunk_size: int = DEFAULT_PAGE_CHUNK_SIZE,
) -> tuple[str, dict, ValidationReport]:
    """
    Process a document with optional page-chunking for large PDFs.
    Returns (markdown_output, json_output, validation_report).
    """
    report = ValidationReport(source=source)
    t0 = time.time()

    source_path = Path(source)
    is_pdf = source_path.suffix.lower() == ".pdf"
    total_pages = get_pdf_page_count(source) if is_pdf else 0
    report.total_pages_expected = total_pages if total_pages > 0 else 1

    use_chunking = is_pdf and chunk_size > 0 and total_pages > chunk_size

    md_chunks: list[str] = []
    json_chunks: list[dict] = []
    all_errors: list[str] = []
    pages_extracted = 0

    def _convert_and_collect(page_range=None, chunk_label=""):
        nonlocal pages_extracted
        kwargs: dict[str, Any] = {"source": source, "raises_on_error": False}
        if page_range:
            kwargs["page_range"] = page_range

        try:
            result = converter.convert(**kwargs)
            if result.status in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
                md_chunks.append(
                    result.document.export_to_markdown(
                        image_mode=ImageRefMode.PLACEHOLDER,
                    )
                )
                json_chunks.append(result.document.export_to_dict())

                # Safe page count
                try:
                    page_count = len(result.pages) if hasattr(result, 'pages') and result.pages else 0
                    if page_count == 0 and page_range:
                        page_count = page_range[1] - page_range[0] + 1
                    elif page_count == 0:
                        page_count = total_pages if total_pages > 0 else 1
                    pages_extracted += page_count
                except Exception:
                    if page_range:
                        pages_extracted += page_range[1] - page_range[0] + 1
                    else:
                        pages_extracted += total_pages if total_pages > 0 else 1

                for err in result.errors:
                    msg = str(err.error_message) if hasattr(err, 'error_message') else str(err)
                    all_errors.append(f"{chunk_label} warning: {msg}")
            else:
                all_errors.append(f"{chunk_label} FAILED: {result.status}")
                log.error(f"{chunk_label} failed: {result.status}")
        except Exception as exc:
            all_errors.append(f"{chunk_label} exception: {type(exc).__name__}: {exc}")
            log.exception(f"Conversion failed: {chunk_label}")

    if use_chunking:
        log.info(f"Large PDF ({total_pages} pages) — chunking by {chunk_size}.")
        chunk_start = 1
        chunk_idx = 0
        while chunk_start <= total_pages:
            chunk_end = min(chunk_start + chunk_size - 1, total_pages)
            chunk_idx += 1
            log.info(f"  Chunk {chunk_idx}: pages {chunk_start}–{chunk_end}")
            _convert_and_collect(
                page_range=(chunk_start, chunk_end),
                chunk_label=f"Chunk {chunk_idx} (p{chunk_start}-{chunk_end})",
            )
            chunk_start = chunk_end + 1
    else:
        log.info(f"Converting {source} in a single pass...")
        _convert_and_collect(chunk_label="Single-pass")

    # Merge
    final_md = "\n\n---\n\n".join(c.strip() for c in md_chunks if c.strip())
    final_json = _merge_json_chunks(json_chunks)

    # Report
    report.total_pages_extracted = pages_extracted
    report.errors = all_errors
    report.total_text_elements = len(final_json.get("texts", []))
    report.total_tables = len(final_json.get("tables", []))
    report.total_pictures = len(final_json.get("pictures", []))
    report.total_code_blocks = final_md.count("```") // 2

    if report.total_pages_extracted < report.total_pages_expected:
        missing = report.total_pages_expected - report.total_pages_extracted
        report.warnings.append(f"{missing} page(s) missing from extraction.")
    if report.total_text_elements == 0:
        report.warnings.append("No text elements found.")

    report.processing_time_sec = time.time() - t0
    report.status = "PASS" if not report.errors else "PASS_WITH_WARNINGS"
    if report.coverage_pct < 80:
        report.status = "FAIL"

    return final_md, final_json, report


def _merge_json_chunks(chunks: list[dict]) -> dict:
    """Merge multiple JSON document dicts into one unified structure."""
    if not chunks:
        return {}
    if len(chunks) == 1:
        return chunks[0]
    merged = copy.deepcopy(chunks[0])
    for chunk in chunks[1:]:
        if "body" in chunk and "children" in chunk["body"]:
            merged.setdefault("body", {}).setdefault("children", []).extend(
                chunk["body"]["children"]
            )
        for key in ("texts", "tables", "pictures", "groups", "key_value_items", "form_items"):
            if key in chunk:
                merged.setdefault(key, []).extend(chunk[key])
        if "pages" in chunk:
            merged.setdefault("pages", {}).update(chunk["pages"])
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Post-Processing
# ─────────────────────────────────────────────────────────────────────────────
def inject_picture_descriptions(markdown: str, json_data: dict) -> str:
    """Replace <!-- image --> placeholders with VLM descriptions from metadata."""
    pictures = json_data.get("pictures", [])

    descriptions: list[str] = []
    for pic in pictures:
        desc = ""
        meta = pic.get("meta", {})
        if meta and isinstance(meta, dict) and "description" in meta:
            desc_data = meta["description"]
            if isinstance(desc_data, dict):
                desc = desc_data.get("text", "").strip()
            elif isinstance(desc_data, str):
                desc = desc_data.strip()
        if desc and desc != _RE_NO_CONTENT:
            descriptions.append(desc)
        else:
            descriptions.append("")

    result_parts: list[str] = []
    remaining = markdown
    desc_idx = 0

    while _RE_IMAGE_PLACEHOLDER in remaining and desc_idx < len(descriptions):
        before, _, remaining = remaining.partition(_RE_IMAGE_PLACEHOLDER)
        result_parts.append(before)
        if descriptions[desc_idx]:
            result_parts.append(f"\n\n{descriptions[desc_idx]}\n\n")
        desc_idx += 1

    result_parts.append(remaining)
    return "".join(result_parts)


def cleanup_markdown(markdown: str) -> str:
    """Remove excessive blank lines and normalize whitespace."""
    markdown = _RE_EXCESS_NEWLINES.sub('\n\n\n', markdown)
    lines = [line if line.strip() else '' for line in markdown.split('\n')]
    return '\n'.join(lines).strip() + '\n'


# ─────────────────────────────────────────────────────────────────────────────
# Output Saving (optional — used when run standalone)
# ─────────────────────────────────────────────────────────────────────────────
def save_outputs(
    source: str,
    markdown: str,
    json_data: dict,
    report: ValidationReport,
    output_dir: Path,
) -> dict[str, str]:
    """Write all outputs to the output directory. Returns a map of file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(source).stem
    paths: dict[str, str] = {}

    # Markdown
    md_path = output_dir / f"{stem}.md"
    md_path.write_text(markdown, encoding="utf-8")
    paths["markdown"] = str(md_path)
    log.info(f"  Markdown → {md_path}")

    # JSON (full structured document)
    json_path = output_dir / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    paths["json"] = str(json_path)
    log.info(f"  JSON → {json_path}")

    # Table of contents
    toc: list[dict] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith('#'):
            level = len(stripped) - len(stripped.lstrip('#'))
            title = stripped.lstrip('#').strip()
            if title:
                toc.append({"level": level, "title": title})
    toc_path = output_dir / f"{stem}_toc.json"
    toc_path.write_text(json.dumps(toc, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["toc"] = str(toc_path)
    log.info(f"  TOC → {toc_path}")

    # Validation report
    report_path = output_dir / f"{stem}_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    paths["report"] = str(report_path)
    log.info(f"  Report → {report_path}")

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestration — Public API
# ─────────────────────────────────────────────────────────────────────────────
def run_unified_pipeline(
    source: str,
    output_dir: Path | None = None,
    chunk_size: int = DEFAULT_PAGE_CHUNK_SIZE,
    no_code: bool = False,
    fast_tables: bool = False,
) -> tuple[str, dict, ValidationReport] | None:
    """
    Run the full unified extraction pipeline on a PDF or image file.

    Args:
        source:      Path to the PDF or image file.
        output_dir:  Optional directory to save outputs (markdown, JSON, report).
                     If None, outputs are returned but not saved to disk.
        chunk_size:  Number of pages per chunk for large PDFs.
        no_code:     If True, disable code enrichment (faster).
        fast_tables: If True, use fast table extraction mode.

    Returns:
        Tuple of (markdown, json_data, validation_report), or None if source
        file not found.
    """
    if not Path(source).exists():
        log.error(f"Source file not found: {source}")
        return None

    t0 = time.time()
    log.info("=" * 60)
    log.info("  UNIFIED PDF EXTRACTION PIPELINE (DOCLING NATIVE + T4 GPU)")
    log.info("=" * 60)
    log.info(f"  Source: {source}")
    log.info(f"  Chunk size: {chunk_size} pages")
    log.info("=" * 60)

    # Build (or retrieve cached) converter
    converter = build_converter(no_code=no_code, fast_tables=fast_tables)

    # Process document
    markdown, json_data, report = process_document(converter, source, chunk_size)

    # Post-processing
    markdown = inject_picture_descriptions(markdown, json_data)
    markdown = cleanup_markdown(markdown)

    # Save outputs to disk if output_dir is specified
    if output_dir is not None:
        log.info("Saving outputs...")
        save_outputs(source, markdown, json_data, report, output_dir)

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("  PIPELINE COMPLETE")
    log.info("=" * 60)
    log.info(report.summary())
    log.info(f"  Total elapsed: {elapsed:.1f}s")
    log.info("=" * 60)

    return markdown, json_data, report


# ─────────────────────────────────────────────────────────────────────────────
# Module-level exports
# ─────────────────────────────────────────────────────────────────────────────
__all__ = [
    "run_unified_pipeline",
    "build_converter",
    "build_pipeline_options",
    "process_document",
    "save_outputs",
    "ValidationReport",
    "DEFAULT_PAGE_CHUNK_SIZE",
]
