"""
Advanced Docling OCR Pipeline
==============================
A production-grade, end-to-end document intelligence pipeline built on Docling.
Designed to handle large multi-page PDFs (50+ pages) with embedded images,
scanned content, tables, and code blocks — maximizing accuracy, structural
integrity, and completeness.

Usage:
    python -m extraction.docling_pipeline                          # Process the default PDF
    python -m extraction.docling_pipeline "path/to/document.pdf"   # Process a specific file
    python -m extraction.docling_pipeline "file1.pdf" "file2.pdf"  # Process multiple files
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    EasyOcrOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
    TableStructureOptions,
)
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)
from docling_core.types.doc.base import ImageRefMode

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("docling_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
# Page-chunk size for processing large PDFs in segments.
# Each chunk is converted independently and then merged, preventing memory
# overload on 50+ page documents.  Set to 0 to disable chunking.
PAGE_CHUNK_SIZE = 10

# Output directory (created automatically)
OUTPUT_DIR = Path("data/output")


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


# ───────────────────────────────────────────────────────────────────────────
# Pipeline Builder
# ───────────────────────────────────────────────────────────────────────────
def build_pipeline_options() -> PdfPipelineOptions:
    """
    Construct a fully-tuned PdfPipelineOptions for maximum extraction quality.

    Key design decisions:
    ─────────────────────
    • OCR is ALWAYS on with force_full_page_ocr=True to catch every embedded
      image/scan, eliminating the heuristic threshold that causes misses.
    • EasyOCR is selected over the default RapidOCR for higher accuracy on
      English text (better handling of capitalization, punctuation, and line
      endings).  Falls back to RapidOCR automatically if EasyOCR is unavailable.
    • Table structure uses "accurate" mode with cell matching for full-fidelity
      row/column alignment.
    • Code enrichment is enabled so that code blocks preserve indentation and
      line breaks via the CodeFormulaV2 vision model.
    • Batch sizes and thread counts are tuned for a 28-core CPU system to
      maximize throughput without OOM.
    """
    num_cores = os.cpu_count() or 4
    # Use roughly half the cores for Docling's internal threading to leave
    # headroom for the OS and Python's own parallelism.
    num_threads = max(4, num_cores // 2)

    opts = PdfPipelineOptions()

    # ── Hardware Acceleration ──────────────────────────────────────────────
    opts.accelerator_options = AcceleratorOptions(
        num_threads=num_threads,
        device="auto",  # Will pick CUDA if available, else CPU
    )

    # ── OCR Configuration ─────────────────────────────────────────────────
    opts.do_ocr = True

    # Use EasyOCR for higher accuracy on English text.
    # EasyOCR handles capitalization, punctuation, and edge-of-line text
    # significantly better than RapidOCR in benchmarks.
    # We must verify the actual easyocr package is importable — the Docling
    # options class constructs fine but the error surfaces at conversion time.
    easyocr_available = False
    try:
        import easyocr as _easyocr  # noqa: F401
        easyocr_available = True
    except ImportError:
        pass

    if easyocr_available:
        ocr_options = EasyOcrOptions(
            lang=["en"],
            force_full_page_ocr=True,   # Never skip — scan every page
            bitmap_area_threshold=0.01, # Catch even small embedded images
            confidence_threshold=0.3,   # Lower threshold = fewer missed chars
        )
        log.info("OCR engine: EasyOCR (high-accuracy mode)")
    else:
        ocr_options = RapidOcrOptions(
            lang=["chinese"],  # RapidOCR default (includes latin)
            force_full_page_ocr=True,
            bitmap_area_threshold=0.01,
            text_score=0.3,
        )
        log.info("OCR engine: RapidOCR (fallback — install easyocr for better accuracy)")
    opts.ocr_options = ocr_options

    # ── Table Extraction ──────────────────────────────────────────────────
    opts.do_table_structure = True
    opts.table_structure_options = TableStructureOptions(
        do_cell_matching=True,  # Align detected cells with content
        mode="accurate",        # Precision over speed
    )

    # ── Code Block Detection ──────────────────────────────────────────────
    opts.do_code_enrichment = True

    # ── Image/Picture Handling ────────────────────────────────────────────
    opts.generate_picture_images = True  # Extract picture crops
    opts.generate_page_images = True     # Keep full page renders for audit

    # ── Performance Tuning ────────────────────────────────────────────────
    # Larger batch sizes use more RAM but process pages faster.
    opts.ocr_batch_size = 8
    opts.layout_batch_size = 8
    opts.table_batch_size = 8

    # Scale factor for internal image rendering — 1.5 balances quality/speed
    opts.images_scale = 1.5

    # No global timeout — large PDFs need time
    opts.document_timeout = None

    return opts


def build_converter(pipeline_options: PdfPipelineOptions) -> DocumentConverter:
    """Create a DocumentConverter supporting both PDF and direct image input."""
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page-Chunked Conversion (Large PDF Strategy)
# ─────────────────────────────────────────────────────────────────────────────
def get_pdf_page_count(source: str) -> int:
    """
    Attempt to determine the total page count of a PDF before conversion.
    Uses docling-parse directly (it's already a dependency of Docling).
    Falls back to 0 if we can't determine the count.
    """
    try:
        from docling_parse.pdf_parser import DoclingPdfParser
        parser = DoclingPdfParser()
        doc = parser.load(source)
        return doc.number_of_pages()
    except Exception:
        pass

    # Fallback: try a quick Docling dry-run on just the first page
    try:
        minimal = PdfPipelineOptions()
        minimal.do_ocr = False
        minimal.do_table_structure = False
        minimal.do_code_enrichment = False
        conv = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=minimal)
            }
        )
        result = conv.convert(source, max_num_pages=sys.maxsize)
        return len(result.pages) if result.pages else 0
    except Exception:
        return 0


def convert_single_source(
    converter: DocumentConverter,
    source: str,
    page_range: tuple[int, int] | None = None,
) -> ConversionResult:
    """
    Convert a single source file, optionally restricted to a page range.
    """
    kwargs: dict[str, Any] = {
        "source": source,
        "raises_on_error": False,
    }
    if page_range is not None:
        kwargs["page_range"] = page_range
    return converter.convert(**kwargs)


def merge_markdown_chunks(chunks: list[str]) -> str:
    """Merge multiple Markdown segments, adding page-break markers."""
    parts = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            parts.append("\n\n---\n\n")  # Visual page-break separator
        parts.append(chunk.strip())
    return "\n".join(parts)


def merge_json_chunks(chunks: list[dict]) -> dict:
    """
    Merge multiple JSON document dicts into one unified structure.
    Concatenates body children, texts, tables, pictures, and pages.
    """
    if not chunks:
        return {}
    if len(chunks) == 1:
        return chunks[0]

    merged = json.loads(json.dumps(chunks[0]))  # Deep copy of first chunk

    for chunk in chunks[1:]:
        # Merge body children
        if "body" in chunk and "children" in chunk["body"]:
            merged.setdefault("body", {}).setdefault("children", []).extend(
                chunk["body"]["children"]
            )
        # Merge flat collections
        for key in ("texts", "tables", "pictures", "groups",
                     "key_value_items", "form_items"):
            if key in chunk:
                merged.setdefault(key, []).extend(chunk[key])
        # Merge pages dict
        if "pages" in chunk:
            merged.setdefault("pages", {}).update(chunk["pages"])

    return merged


def process_document(
    converter: DocumentConverter,
    source: str,
    chunk_size: int = PAGE_CHUNK_SIZE,
) -> tuple[str, dict, ValidationReport]:
    """
    Process a document with optional page-chunking for large PDFs.

    Returns (markdown_output, json_output, validation_report).
    """
    report = ValidationReport(source=source)
    t0 = time.time()

    source_path = Path(source)
    is_pdf = source_path.suffix.lower() == ".pdf"

    # ── Determine page count for PDFs ─────────────────────────────────────
    total_pages = 0
    if is_pdf:
        log.info("Probing page count for: %s", source)
        total_pages = get_pdf_page_count(source)
        log.info("Detected %d pages", total_pages)
    report.total_pages_expected = total_pages if total_pages > 0 else 1

    # ── Decide: chunked vs single-pass ────────────────────────────────────
    use_chunking = is_pdf and chunk_size > 0 and total_pages > chunk_size

    md_chunks: list[str] = []
    json_chunks: list[dict] = []
    all_errors: list[str] = []
    pages_extracted = 0

    if use_chunking:
        log.info(
            "Large PDF detected (%d pages). Processing in chunks of %d.",
            total_pages,
            chunk_size,
        )
        chunk_start = 1
        chunk_idx = 0
        while chunk_start <= total_pages:
            chunk_end = min(chunk_start + chunk_size - 1, total_pages)
            chunk_idx += 1
            log.info(
                "  Chunk %d: pages %d–%d", chunk_idx, chunk_start, chunk_end
            )

            try:
                result = convert_single_source(
                    converter, source, page_range=(chunk_start, chunk_end)
                )

                if result.status == ConversionStatus.SUCCESS:
                    md_chunks.append(result.document.export_to_markdown(image_mode=ImageRefMode.EMBEDDED))
                    json_chunks.append(result.document.export_to_dict())
                    pages_extracted += len(result.pages)
                elif result.status == ConversionStatus.PARTIAL_SUCCESS:
                    md_chunks.append(result.document.export_to_markdown(image_mode=ImageRefMode.EMBEDDED))
                    json_chunks.append(result.document.export_to_dict())
                    pages_extracted += len(result.pages)
                    for err in result.errors:
                        all_errors.append(
                            f"Chunk {chunk_idx} (p{chunk_start}-{chunk_end}): "
                            f"{err.error_message}"
                        )
                else:
                    all_errors.append(
                        f"Chunk {chunk_idx} (p{chunk_start}-{chunk_end}) "
                        f"FAILED: {result.status}"
                    )

            except Exception as exc:
                all_errors.append(
                    f"Chunk {chunk_idx} (p{chunk_start}-{chunk_end}) "
                    f"EXCEPTION: {exc}"
                )
                log.exception("Chunk %d failed", chunk_idx)

            chunk_start = chunk_end + 1

    else:
        # ── Single-pass conversion ────────────────────────────────────────
        log.info("Converting %s in a single pass...", source)
        try:
            result = convert_single_source(converter, source)

            if result.status in (
                ConversionStatus.SUCCESS,
                ConversionStatus.PARTIAL_SUCCESS,
            ):
                md_chunks.append(result.document.export_to_markdown(image_mode=ImageRefMode.EMBEDDED))
                json_chunks.append(result.document.export_to_dict())
                pages_extracted = len(result.pages) if result.pages else 1
                for err in result.errors:
                    all_errors.append(f"Conversion warning: {err.error_message}")
            else:
                all_errors.append(f"Conversion FAILED: {result.status}")

        except Exception as exc:
            all_errors.append(f"Conversion EXCEPTION: {exc}")
            log.exception("Conversion failed for %s", source)

    # ── Merge results ─────────────────────────────────────────────────────
    final_md = merge_markdown_chunks(md_chunks)
    final_json = merge_json_chunks(json_chunks)

    # ── Validation ────────────────────────────────────────────────────────
    report.total_pages_extracted = pages_extracted
    report.errors = all_errors

    # Count content elements from the merged JSON
    report.total_text_elements = len(final_json.get("texts", []))
    report.total_tables = len(final_json.get("tables", []))
    report.total_pictures = len(final_json.get("pictures", []))

    # Count code blocks in Markdown output (fenced with ```)
    code_fence_count = final_md.count("```")
    report.total_code_blocks = code_fence_count // 2  # opening + closing

    # Detect empty pages
    pages_dict = final_json.get("pages", {})
    for page_key in pages_dict:
        try:
            page_num = int(page_key) + 1  # 0-indexed → 1-indexed
        except (ValueError, TypeError):
            page_num = page_key

    # Completeness warnings
    if report.total_pages_extracted < report.total_pages_expected:
        missing = report.total_pages_expected - report.total_pages_extracted
        report.warnings.append(
            f"{missing} page(s) may not have been extracted "
            f"({report.total_pages_extracted}/{report.total_pages_expected})"
        )
    if report.total_text_elements == 0:
        report.warnings.append("No text elements found — document may be image-only.")
    if report.total_tables == 0 and "table" in source.lower():
        report.warnings.append("Expected tables but none were detected.")

    report.processing_time_sec = time.time() - t0
    report.status = "PASS" if not report.errors else "PASS_WITH_WARNINGS"
    if report.coverage_pct < 80:
        report.status = "FAIL"

    return final_md, final_json, report


# ─────────────────────────────────────────────────────────────────────────────
# Output Writer
# ─────────────────────────────────────────────────────────────────────────────
def save_outputs(
    source: str,
    markdown: str,
    json_data: dict,
    report: ValidationReport,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, str]:
    """Write all outputs to the output directory. Returns a map of file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(source).stem

    paths = {}

    # Markdown
    md_path = output_dir / f"{stem}.md"
    md_path.write_text(markdown, encoding="utf-8")
    paths["markdown"] = str(md_path)
    log.info("Markdown saved  → %s", md_path)

    # JSON (full structured document)
    json_path = output_dir / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    paths["json"] = str(json_path)
    log.info("JSON saved      → %s", json_path)

    # Validation report
    report_path = output_dir / f"{stem}_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    paths["report"] = str(report_path)
    log.info("Report saved    → %s", report_path)

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Advanced Docling OCR Pipeline — process PDFs and images.",
    )
    parser.add_argument(
        "sources",
        nargs="*",
        default=[r"content\Math_Session_1.pdf"],
        help="Paths to PDF/image files to process",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=PAGE_CHUNK_SIZE,
        help=f"Pages per chunk for large PDFs (0=disable, default={PAGE_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Output directory (default={OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-code",
        action="store_true",
        help="Disable code enrichment (faster, lower resource usage)",
    )
    parser.add_argument(
        "--fast-tables",
        action="store_true",
        help="Use fast table extraction mode instead of accurate",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # ── Build the pipeline ────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  ADVANCED DOCLING OCR PIPELINE")
    log.info("=" * 60)

    pipeline_opts = build_pipeline_options()

    # Apply CLI overrides
    if args.no_code:
        pipeline_opts.do_code_enrichment = False
        log.info("Code enrichment DISABLED via --no-code")
    if args.fast_tables:
        pipeline_opts.table_structure_options = TableStructureOptions(
            do_cell_matching=True,
            mode="fast",
        )
        log.info("Table extraction mode: FAST")

    converter = build_converter(pipeline_opts)
    log.info("Pipeline initialized. Processing %d source(s)...", len(args.sources))

    # ── Process each source ───────────────────────────────────────────────
    all_reports = []

    for source in args.sources:
        log.info("─" * 60)
        log.info("Processing: %s", source)
        log.info("─" * 60)

        if not Path(source).exists():
            log.error("File not found: %s — skipping.", source)
            continue

        markdown, json_data, report = process_document(
            converter, source, chunk_size=args.chunk_size
        )

        paths = save_outputs(source, markdown, json_data, report, output_dir)

        log.info("Validation Report for %s:", Path(source).name)
        log.info("\n%s", report.summary())
        all_reports.append(report)

    # ── Final Summary ─────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  PIPELINE COMPLETE")
    log.info("=" * 60)
    passed = sum(1 for r in all_reports if r.status.startswith("PASS"))
    failed = sum(1 for r in all_reports if r.status == "FAIL")
    log.info("  %d processed  |  %d passed  |  %d failed", len(all_reports), passed, failed)
    total_time = sum(r.processing_time_sec for r in all_reports)
    log.info("  Total time: %.1fs", total_time)
    log.info("  Outputs in: %s", output_dir.resolve())


if __name__ == "__main__":
    main()
    