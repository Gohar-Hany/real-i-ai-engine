"""
Merged PDF Extraction Pipeline
===============================
Combines Doc.py (Docling) for excellent document structure with
local_pdf_pipeline.py's OCR/vision capabilities for image-based content.

Features:
  - Primary extraction via Docling (headings, sections, tables, code, math)
  - Automatic logo/watermark detection and removal (small repeated images)
  - Fallback to PyMuPDF + PaddleOCR + Ollama Vision for content-poor pages
  - Clean, deduplicated Markdown output

Usage:
    python -m extraction.merged_pipeline "document.pdf"
    python -m extraction.merged_pipeline "document.pdf" --output-dir data/output
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

import fitz  # PyMuPDF

# ─── Import from docling_pipeline (primary pipeline) ──────────────────────
from extraction.docling_pipeline import (
    build_pipeline_options,
    build_converter,
    process_document,
    save_outputs,
    ValidationReport,
    PAGE_CHUNK_SIZE,
)

# ─── Import from local_pdf_pipeline (fallback) ────────────────────────────
from extraction.local_pdf_pipeline import (
    extract_text_from_image,
    _is_ocr_meaningful,
    _init_ocr,
    PAGE_RENDER_DPI,
)

try:
    from extraction.local_pdf_pipeline import (
        get_image_vision_description,
        post_process_text,
        OLLAMA_AVAILABLE,
    )
except ImportError:
    OLLAMA_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("merged_pipeline")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("data/output")
REFERENCE_LOGOS_DIR = Path("data/reference_logos")

# Logo detection thresholds
LOGO_AREA_THRESHOLD = 40_000    # px² — images smaller than this are candidates
LOGO_REPEAT_THRESHOLD = 2      # images appearing on N+ pages = logos
MIN_PAGE_TEXT_CHARS = 30        # minimum chars for a page to be "content-rich"

# Regex patterns for Markdown parsing
IMAGE_PATTERN = re.compile(
    r'!\[Image\]\(data:image/[^;]+;base64,([A-Za-z0-9+/=\n]+)\)'
)
IMAGE_LINE_PATTERN = re.compile(
    r'^!\[Image\]\(data:image/[^;]+;base64,[A-Za-z0-9+/=\n]+\)$'
)
HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)


# ─────────────────────────────────────────────────────────────────────────────
# Logo Detection & Filtering
# ─────────────────────────────────────────────────────────────────────────────
def _decode_base64_image_size(b64_data: str) -> Optional[Tuple[int, int]]:
    """Decode base64 image data and return (width, height), or None."""
    if not PIL_AVAILABLE:
        return None
    try:
        img_bytes = base64.b64decode(b64_data)
        img = Image.open(io.BytesIO(img_bytes))
        return img.size
    except Exception:
        return None


def _b64_signature(b64_data: str) -> str:
    """Compute a short hash signature of base64 image data for dedup."""
    return hashlib.md5(b64_data[:2000].encode()).hexdigest()


def detect_logo_images(markdown: str) -> Set[str]:
    """
    Find logo images in Markdown by detecting small, repeated base64 images.
    Returns set of base64-prefix signatures that are logos.
    """
    matches = IMAGE_PATTERN.findall(markdown)
    if not matches:
        return set()

    # Group by signature → track count and size
    sig_info: Dict[str, Dict[str, Any]] = {}
    b64_to_sig: Dict[str, str] = {}

    for b64_data in matches:
        b64_clean = b64_data.replace('\n', '')
        sig = _b64_signature(b64_clean)
        b64_to_sig[b64_clean] = sig

        if sig in sig_info:
            sig_info[sig]['count'] += 1
            continue

        size = _decode_base64_image_size(b64_clean)
        w, h = size if size else (0, 0)
        sig_info[sig] = {
            'count': 1,
            'width': w,
            'height': h,
            'area': w * h,
            'b64_prefix': b64_clean[:200],  # store prefix for matching
        }

    # Identify logos: small AND repeated
    logo_prefixes: Set[str] = set()
    for sig, info in sig_info.items():
        is_small = info['area'] < LOGO_AREA_THRESHOLD
        is_repeated = info['count'] >= LOGO_REPEAT_THRESHOLD

        if is_small and is_repeated:
            log.info(
                "  Logo detected: %d×%d (%d px²), appears %d times",
                info['width'], info['height'], info['area'], info['count']
            )
            logo_prefixes.add(info['b64_prefix'])

    return logo_prefixes


def save_reference_logos(
    markdown: str,
    logo_prefixes: Set[str],
    output_dir: Path = REFERENCE_LOGOS_DIR,
) -> None:
    """Save detected logo images as reference PNG files."""
    if not PIL_AVAILABLE or not logo_prefixes:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    matches = IMAGE_PATTERN.findall(markdown)
    saved = set()

    for b64_data in matches:
        b64_clean = b64_data.replace('\n', '')
        prefix = b64_clean[:200]
        if prefix in logo_prefixes and prefix not in saved:
            saved.add(prefix)
            try:
                img_bytes = base64.b64decode(b64_clean)
                img = Image.open(io.BytesIO(img_bytes))
                idx = len(saved)
                filepath = output_dir / f"logo_{idx}.png"
                img.save(str(filepath))
                log.info("  Reference logo saved → %s", filepath)
            except Exception as exc:
                log.warning("  Could not save reference logo: %s", exc)


def filter_logos_from_markdown(
    markdown: str, logo_prefixes: Set[str]
) -> str:
    """Remove all image tags whose base64 data matches identified logos."""
    if not logo_prefixes:
        return markdown

    lines = markdown.split('\n')
    filtered = []
    removed = 0

    for line in lines:
        is_logo = False
        for prefix in logo_prefixes:
            if prefix in line:
                is_logo = True
                removed += 1
                break
        if not is_logo:
            filtered.append(line)

    log.info("  Removed %d logo image line(s) from Markdown", removed)
    return '\n'.join(filtered)


# ─────────────────────────────────────────────────────────────────────────────
# Image → Text Replacement (replaces ALL remaining images with descriptions)
# ─────────────────────────────────────────────────────────────────────────────

# Full image tag pattern (matches the entire ![Image](...) including data URI)
FULL_IMAGE_TAG = re.compile(
    r'!\[Image\]\(data:image/([^;]+);base64,([A-Za-z0-9+/=\n]+)\)'
)

# Prompt for the vision model to extract useful content from images
IMAGE_CONTENT_PROMPT = (
    "You are a precise document content extractor. "
    "Analyze this image and extract ONLY the valuable textual content it contains. "
    "Follow these rules strictly:\n"
    "- If the image contains a mathematical formula or equation, transcribe it in LaTeX format (e.g. $E = mc^2$).\n"
    "- If it is a chart or graph, describe the chart type, axes, data series, trends, and key values.\n"
    "- If it is a diagram or schematic, describe every component, label, and flow.\n"
    "- If it contains text (like a slide or document), transcribe ALL the text content.\n"
    "- If it is purely decorative or a logo with no educational/informational value, respond with exactly: [NO_CONTENT]\n"
    "- Do NOT describe the visual appearance of the image itself (colors, layout, etc.) unless relevant to the data.\n"
    "- Return ONLY the extracted content — no commentary, no preamble."
)


def _extract_content_from_b64_image(
    b64_data: str,
    img_format: str,
    vision_model: str = "minicpm-v",
    text_model: str = "phi3:mini",
) -> str:
    """
    Decode a base64 image, run OCR + optional vision, and return
    the extracted textual content. Returns empty string if no useful
    content is found.
    """
    import tempfile

    # Decode image to temp file
    try:
        img_bytes = base64.b64decode(b64_data)
    except Exception:
        return ""

    suffix = f".{img_format}" if img_format else ".png"
    with tempfile.NamedTemporaryFile(
        suffix=suffix, delete=False, prefix="img_content_"
    ) as tmp:
        tmp.write(img_bytes)
        tmp_path = tmp.name

    try:
        content_parts: List[str] = []

        # Step 1: Try OCR first
        ocr_text = extract_text_from_image(tmp_path)
        ocr_good = _is_ocr_meaningful(ocr_text)

        if ocr_good:
            # Clean OCR text with LLM if available
            if OLLAMA_AVAILABLE:
                try:
                    cleaned = post_process_text(
                        ocr_text, model=text_model, timeout=60
                    )
                    content_parts.append(cleaned)
                except Exception:
                    content_parts.append(ocr_text)
            else:
                content_parts.append(ocr_text)

        # Step 2: Vision description for richer content extraction
        if OLLAMA_AVAILABLE:
            try:
                response = _ollama_client().chat(
                    model=vision_model,
                    messages=[{
                        "role": "user",
                        "content": IMAGE_CONTENT_PROMPT,
                        "images": [tmp_path],
                    }],
                )
                desc = response.get("message", {}).get("content", "").strip()

                if (
                    desc
                    and "[NO_CONTENT]" not in desc
                    and "[Vision description timed out]" not in desc
                    and "[Vision description failed" not in desc
                ):
                    # If OCR already got text, only add vision if it provides
                    # significantly different/additional content
                    if not ocr_good:
                        content_parts.append(desc)
                    elif len(desc) > len(ocr_text) * 1.5:
                        # Vision provided substantially more — use it instead
                        content_parts = [desc]
            except Exception as exc:
                log.debug("  Vision extraction failed: %s", exc)

        return "\n\n".join(content_parts) if content_parts else ""

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _ollama_client():
    """Get an Ollama client instance."""
    import ollama as _ollama
    return _ollama.Client(host="http://localhost:11434")


def replace_images_with_descriptions(
    markdown: str,
    vision_model: str = "minicpm-v",
    text_model: str = "phi3:mini",
) -> Tuple[str, int]:
    """
    Find ALL remaining base64-embedded images in the Markdown,
    extract their textual content (equations, descriptions, data),
    and replace the image tag with the extracted text.

    Images with no useful content are removed entirely.

    Returns (new_markdown, count_of_images_processed).
    """
    # Initialize OCR engine once
    _init_ocr()

    matches = list(FULL_IMAGE_TAG.finditer(markdown))
    if not matches:
        return markdown, 0

    log.info("  Found %d content image(s) to process...", len(matches))

    # Process in reverse order so string indices stay valid
    images_processed = 0
    images_with_content = 0
    images_removed = 0

    for match in reversed(matches):
        img_format = match.group(1)  # png, jpeg, etc.
        b64_data = match.group(2).replace('\n', '')

        images_processed += 1
        log.info(
            "  Processing image %d/%d (format: %s, ~%d KB)...",
            len(matches) - images_processed + 1, len(matches),
            img_format, len(b64_data) * 3 // 4 // 1024
        )

        # Extract content from the image
        content = _extract_content_from_b64_image(
            b64_data, img_format,
            vision_model=vision_model,
            text_model=text_model,
        )

        if content.strip():
            # Replace image tag with extracted content
            replacement = f"\n{content.strip()}\n"
            images_with_content += 1
        else:
            # No useful content — just remove the image
            replacement = ""
            images_removed += 1

        markdown = markdown[:match.start()] + replacement + markdown[match.end():]

    log.info(
        "  Image processing complete: %d processed, %d had content, %d removed",
        images_processed, images_with_content, images_removed
    )
    return markdown, images_processed


# ─────────────────────────────────────────────────────────────────────────────
# Page Quality Assessment
# ─────────────────────────────────────────────────────────────────────────────
def _get_page_text_map(json_data: dict) -> Dict[int, str]:
    """
    Build a map of page_number → concatenated text from the JSON export.
    Uses Docling's texts array with provenance info.
    """
    page_texts: Dict[int, List[str]] = {}

    for text_item in json_data.get("texts", []):
        text = text_item.get("text", "").strip()
        if not text:
            continue

        # Get label — skip headers/footers (noise)
        label = text_item.get("label", "")
        if isinstance(label, str):
            label_lower = label.split(".")[-1].lower()
        else:
            label_lower = str(label).split(".")[-1].lower()

        if label_lower in ("page_header", "page_footer"):
            continue

        # Get page number from provenance
        prov = text_item.get("prov", [])
        if prov and isinstance(prov, list) and len(prov) > 0:
            page_no = prov[0].get("page_no", prov[0].get("page", 0))
        else:
            continue

        page_texts.setdefault(page_no, []).append(text)

    return {p: " ".join(texts) for p, texts in page_texts.items()}


def _get_page_headings_map(json_data: dict) -> Dict[int, List[str]]:
    """
    Build a map of page_number → list of heading texts from JSON export.
    """
    page_headings: Dict[int, List[str]] = {}

    for text_item in json_data.get("texts", []):
        label = text_item.get("label", "")
        if isinstance(label, str):
            label_lower = label.split(".")[-1].lower()
        else:
            label_lower = str(label).split(".")[-1].lower()

        if label_lower not in ("title", "section_header"):
            continue

        text = text_item.get("text", "").strip()
        if not text:
            continue

        prov = text_item.get("prov", [])
        if prov and isinstance(prov, list) and len(prov) > 0:
            page_no = prov[0].get("page_no", prov[0].get("page", 0))
        else:
            continue

        page_headings.setdefault(page_no, []).append(text)

    return page_headings


def find_content_poor_pages(
    json_data: dict,
    total_pages: int,
    min_chars: int = MIN_PAGE_TEXT_CHARS,
) -> List[int]:
    """
    Identify pages where Doc.py failed to extract meaningful text.
    Returns list of 1-indexed page numbers.
    """
    page_text_map = _get_page_text_map(json_data)
    poor_pages = []

    for page_no in range(1, total_pages + 1):
        text = page_text_map.get(page_no, "")
        # Strip heading text — we want to check body content
        # (headings alone don't count as "meaningful content")
        if len(text.strip()) < min_chars:
            poor_pages.append(page_no)
            log.info(
                "  Page %d: content-poor (%d chars < %d threshold)",
                page_no, len(text.strip()), min_chars
            )

    return poor_pages


# ─────────────────────────────────────────────────────────────────────────────
# Fallback: PyMuPDF + OCR + Optional Vision
# ─────────────────────────────────────────────────────────────────────────────
def extract_fallback_for_pages(
    pdf_path: str,
    pages: List[int],
    vision_model: str = "minicpm-v",
    text_model: str = "phi3:mini",
    dpi: int = 150,
) -> Dict[int, str]:
    """
    For each page number in `pages`, render via PyMuPDF, OCR, and
    optionally describe via Ollama vision.
    Returns page_number → extracted text content.
    """
    if not pages:
        return {}

    log.info("Fallback extraction for %d content-poor page(s): %s", len(pages), pages)
    result: Dict[int, str] = {}
    doc = fitz.open(pdf_path)

    # Initialize OCR engine once
    _init_ocr()

    for page_no in pages:
        page_idx = page_no - 1  # fitz is 0-indexed
        if page_idx < 0 or page_idx >= len(doc):
            log.warning("  Page %d out of range, skipping", page_no)
            continue

        page = doc.load_page(page_idx)
        log.info("  Fallback: rendering page %d at %d DPI...", page_no, dpi)

        # Render page to image
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pixmap = page.get_pixmap(matrix=mat, alpha=False)

        # Save to temp file for OCR
        with tempfile.NamedTemporaryFile(
            suffix=".png", delete=False, prefix=f"page_{page_no}_"
        ) as tmp:
            pixmap.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Step 1: OCR
            ocr_text = extract_text_from_image(tmp_path)
            ocr_good = _is_ocr_meaningful(ocr_text)

            content_parts: List[str] = []

            if ocr_good:
                # Clean OCR text with LLM if available
                if OLLAMA_AVAILABLE:
                    try:
                        cleaned = post_process_text(
                            ocr_text, model=text_model, timeout=60
                        )
                        content_parts.append(cleaned)
                    except Exception:
                        content_parts.append(ocr_text)
                else:
                    content_parts.append(ocr_text)
                log.info("  Page %d: OCR extracted %d chars", page_no, len(ocr_text))

            # Step 2: Vision description (always for full-page renders)
            if OLLAMA_AVAILABLE:
                try:
                    desc = get_image_vision_description(
                        tmp_path,
                        model=vision_model,
                        timeout=120,
                        context_hint="",
                    )
                    if (
                        desc
                        and "[Vision description timed out]" not in desc
                        and "[Vision description failed" not in desc
                    ):
                        if not ocr_good:
                            content_parts.append(desc)
                        # If OCR was good, vision is redundant — skip
                except Exception as exc:
                    log.warning("  Vision fallback failed for page %d: %s", page_no, exc)
            elif not ocr_good:
                log.warning(
                    "  Page %d: no OCR text and Ollama unavailable", page_no
                )

            result[page_no] = "\n\n".join(content_parts) if content_parts else ""

        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    doc.close()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Fallback Content Injection
# ─────────────────────────────────────────────────────────────────────────────
def inject_fallback_content(
    markdown: str,
    json_data: dict,
    fallback_content: Dict[int, str],
) -> str:
    """
    Insert fallback-extracted text into the Markdown at the correct positions.

    Strategy: find the heading for each content-poor page, then insert the
    fallback text immediately after that heading block.
    """
    if not fallback_content:
        return markdown

    page_headings = _get_page_headings_map(json_data)

    for page_no, content in fallback_content.items():
        if not content.strip():
            continue

        headings = page_headings.get(page_no, [])
        if not headings:
            # No heading found — append at end
            markdown += f"\n\n<!-- Page {page_no} (fallback) -->\n\n{content}\n"
            log.info("  Page %d: fallback appended at end (no heading found)", page_no)
            continue

        # Find the LAST heading for this page and insert content after it
        target_heading = headings[-1]
        inserted = False

        lines = markdown.split('\n')
        new_lines = []
        for i, line in enumerate(lines):
            new_lines.append(line)
            # Check if this line contains the target heading
            stripped = line.strip()
            if (
                not inserted
                and stripped.startswith('#')
                and target_heading in stripped
            ):
                # Insert fallback content after this heading
                # Skip any immediately following blank lines first
                new_lines.append("")
                new_lines.append(content)
                new_lines.append("")
                inserted = True
                log.info(
                    "  Page %d: fallback inserted after heading '%s'",
                    page_no, target_heading[:50]
                )

        if inserted:
            markdown = '\n'.join(new_lines)
        else:
            # Heading not found in Markdown — append at end
            markdown += f"\n\n<!-- Page {page_no} (fallback) -->\n\n{content}\n"
            log.info(
                "  Page %d: heading '%s' not found, fallback appended at end",
                page_no, target_heading[:50]
            )

    return markdown


# ─────────────────────────────────────────────────────────────────────────────
# Final Cleanup
# ─────────────────────────────────────────────────────────────────────────────
def cleanup_markdown(markdown: str) -> str:
    """Remove excessive blank lines, empty sections, and artifacts."""
    # Collapse 3+ consecutive blank lines into 2
    markdown = re.sub(r'\n{4,}', '\n\n\n', markdown)

    # Remove lines that are just whitespace
    lines = markdown.split('\n')
    lines = [line if line.strip() else '' for line in lines]

    # Remove empty sections (heading followed immediately by another heading)
    # This is optional — sometimes a heading-only section is intentional
    cleaned = []
    for i, line in enumerate(lines):
        cleaned.append(line)

    return '\n'.join(cleaned).strip() + '\n'


# ─────────────────────────────────────────────────────────────────────────────
# Get total page count
# ─────────────────────────────────────────────────────────────────────────────
def get_total_pages(pdf_path: str) -> int:
    """Get total page count using PyMuPDF (reliable)."""
    try:
        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_merged_pipeline(
    source: str,
    output_dir: Path = OUTPUT_DIR,
    chunk_size: int = PAGE_CHUNK_SIZE,
    vision_model: str = "minicpm-v",
    text_model: str = "phi3:mini",
    dpi: int = 150,
    no_code: bool = False,
    fast_tables: bool = False,
) -> None:
    """
    Full merged pipeline:
      1. Docling primary extraction (structure, text, tables)
      2. Logo detection & removal
      3. Content quality assessment per page
      4. PyMuPDF + OCR fallback for content-poor pages
      5. Clean Markdown output
    """
    if not Path(source).exists():
        log.error("File not found: %s", source)
        return

    t0 = time.time()

    # ── Step 1: Primary extraction via Docling ────────────────────────────
    log.info("=" * 60)
    log.info("  MERGED PDF EXTRACTION PIPELINE")
    log.info("=" * 60)
    log.info("Step 1: Docling primary extraction...")

    from docling.datamodel.pipeline_options import TableStructureOptions

    pipeline_opts = build_pipeline_options()
    if no_code:
        pipeline_opts.do_code_enrichment = False
    if fast_tables:
        pipeline_opts.table_structure_options = TableStructureOptions(
            do_cell_matching=True, mode="fast"
        )

    converter = build_converter(pipeline_opts)
    markdown, json_data, report = process_document(
        converter, source, chunk_size=chunk_size
    )
    log.info("  Docling extraction complete: %s", report.status)

    total_pages = get_total_pages(source)
    if total_pages == 0:
        total_pages = report.total_pages_expected or 1
    log.info("  Total pages: %d", total_pages)

    # ── Step 2: Logo detection & removal ──────────────────────────────────
    log.info("Step 2: Detecting and removing logo images...")
    logo_prefixes = detect_logo_images(markdown)

    if logo_prefixes:
        save_reference_logos(markdown, logo_prefixes)
        markdown = filter_logos_from_markdown(markdown, logo_prefixes)
        log.info("  Detected %d unique logo pattern(s)", len(logo_prefixes))
    else:
        log.info("  No repeated logo images detected")

    # ── Step 2b: Replace remaining images with text descriptions ──────────
    log.info("Step 2b: Extracting content from remaining images...")
    markdown, images_replaced = replace_images_with_descriptions(
        markdown,
        vision_model=vision_model,
        text_model=text_model,
    )
    log.info("  Processed %d image(s)", images_replaced)

    # ── Step 3: Content quality assessment ─────────────────────────────────
    log.info("Step 3: Assessing page content quality...")
    poor_pages = find_content_poor_pages(json_data, total_pages)

    if poor_pages:
        log.info("  Found %d content-poor page(s): %s", len(poor_pages), poor_pages)
    else:
        log.info("  All pages have sufficient text content")

    # ── Step 4: Fallback extraction ───────────────────────────────────────
    fallback_content: Dict[int, str] = {}
    if poor_pages:
        log.info("Step 4: Running fallback extraction (PyMuPDF + OCR)...")
        fallback_content = extract_fallback_for_pages(
            source,
            poor_pages,
            vision_model=vision_model,
            text_model=text_model,
            dpi=dpi,
        )
        pages_recovered = sum(1 for v in fallback_content.values() if v.strip())
        log.info("  Recovered content for %d/%d pages", pages_recovered, len(poor_pages))
    else:
        log.info("Step 4: No fallback needed — skipping")

    # ── Step 5: Inject fallback content ───────────────────────────────────
    if fallback_content:
        log.info("Step 5: Injecting fallback content into Markdown...")
        markdown = inject_fallback_content(markdown, json_data, fallback_content)

    # ── Step 6: Final cleanup ─────────────────────────────────────────────
    log.info("Step 6: Final cleanup...")
    markdown = cleanup_markdown(markdown)

    # Also clean up failure markers that may remain
    markdown = markdown.replace("[Vision description timed out]", "")
    markdown = re.sub(r'\[Vision description failed:.*?\]', '', markdown)

    # ── Step 7: Save outputs ──────────────────────────────────────────────
    log.info("Step 7: Saving outputs...")
    paths = save_outputs(source, markdown, json_data, report, output_dir)

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("  MERGED PIPELINE COMPLETE")
    log.info("=" * 60)
    log.info("  Total time: %.1fs", elapsed)
    log.info("  Pages: %d total, %d content-poor, %d recovered via fallback",
             total_pages, len(poor_pages),
             sum(1 for v in fallback_content.values() if v.strip()))
    log.info("  Logo patterns removed: %d", len(logo_prefixes))
    log.info("  Images replaced with text: %d", images_replaced)
    log.info("  Outputs in: %s", output_dir.resolve())
    for key, path in paths.items():
        log.info("    %s → %s", key, path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merged PDF Extraction Pipeline — Docling + OCR fallback.",
    )
    parser.add_argument(
        "sources",
        nargs="*",
        default=[r"Math_Session_1.pdf"],
        help="Paths to PDF files to process",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=PAGE_CHUNK_SIZE,
        help=f"Pages per chunk for large PDFs (default={PAGE_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_DIR),
        help=f"Output directory (default={OUTPUT_DIR})",
    )
    parser.add_argument(
        "--vision-model", default="minicpm-v",
        help="Ollama vision model for fallback descriptions",
    )
    parser.add_argument(
        "--text-model", default="phi3:mini",
        help="Ollama text model for OCR cleanup",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="DPI for full-page renders in fallback (default=150)",
    )
    parser.add_argument(
        "--no-code", action="store_true",
        help="Disable code enrichment (faster)",
    )
    parser.add_argument(
        "--fast-tables", action="store_true",
        help="Use fast table extraction mode",
    )
    args = parser.parse_args()

    for source in args.sources:
        run_merged_pipeline(
            source=source,
            output_dir=Path(args.output_dir),
            chunk_size=args.chunk_size,
            vision_model=args.vision_model,
            text_model=args.text_model,
            dpi=args.dpi,
            no_code=args.no_code,
            fast_tables=args.fast_tables,
        )


if __name__ == "__main__":
    main()
