"""
PDF Extraction Pipeline
=======================
A robust, multi-layer pipeline for extracting ALL content from PDF documents:
  - Structured text and tables via Docling
  - Embedded raster images via PyMuPDF (get_images)
  - Vector/drawn graphics via PyMuPDF full-page pixmap rendering
  - OCR on image regions via PaddleOCR
  - Visual description of complex images / charts via Ollama Vision
  - Formula handling via Ollama Vision instead of silently dropping them
  - Clean Markdown reconstruction preserving reading order
  - Structured JSON output derived from Docling's native export (not LLM hallucination)

Usage:
    python -m extraction.local_pdf_pipeline document.pdf [--vision-model minicpm-v] [--text-model phi3:mini] [--dpi 150]
"""

import os
import sys
import json
import logging
import argparse
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import fitz  # PyMuPDF
from PIL import Image

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("ollama package not found. Vision and text post-processing will be disabled.")

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PaddleOCR = None
    PADDLEOCR_AVAILABLE = False

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_DIR   = Path("data/output")
IMAGES_DIR   = OUTPUT_DIR / "extracted_images"
PAGE_IMG_DIR = OUTPUT_DIR / "page_renders"          # full-page pixmap renders
MIN_IMAGE_AREA = 1000                               # px² — ignore tiny decorative images
PAGE_RENDER_DPI = 150                               # DPI for full-page renders

# ---------------------------------------------------------------------------
# Global OCR engine (initialised once)
# ---------------------------------------------------------------------------
ocr_engine: Optional[Any] = None

def _init_ocr() -> Optional[Any]:
    """Initialise PaddleOCR once and cache globally."""
    global ocr_engine
    if ocr_engine is not None:
        return ocr_engine
    if not PADDLEOCR_AVAILABLE:
        log.warning("PaddleOCR not installed — OCR step will be skipped.")
        return None
    try:
        log.info("Initialising PaddleOCR …")
        ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        log.info("PaddleOCR ready.")
    except Exception as exc:
        log.warning(f"PaddleOCR init failed: {exc}")
        ocr_engine = None
    return ocr_engine


# ===========================================================================
# STAGE 1 — Docling: structured text + table extraction
# ===========================================================================

def extract_docling_structure(pdf_path: str):
    """
    Run Docling over the PDF with OCR and table-structure detection enabled.
    Returns a Docling Document object whose items preserve reading order.
    """
    log.info("Stage 1 — Docling: parsing document structure …")
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        },
    )
    result = converter.convert(pdf_path)
    log.info("Docling parsing complete.")
    return result.document


# ===========================================================================
# STAGE 2 — PyMuPDF: dual image extraction strategy
# ===========================================================================

def _is_image_worth_keeping(width: int, height: int) -> bool:
    """Filter out tiny decorative images (icons, bullets, dividers)."""
    return (width * height) >= MIN_IMAGE_AREA


def extract_pymupdf_images(
    pdf_path: str,
    output_dir: Path = IMAGES_DIR,
) -> List[Dict[str, Any]]:
    """
    Two-pass image extraction using PyMuPDF:

    Pass A — Embedded raster images (get_images):
        Extracts JPEG / PNG / JBIG2 images stored inside the PDF's resource
        dictionary.  These are the raw originals at full resolution.

    Pass B — Full-page pixmap render (get_pixmap):
        Renders each page as a raster image at PAGE_RENDER_DPI.  This catches
        everything that Pass A misses: vector drawings, charts drawn with PDF
        path operators, SVG-like graphics, and mixed-content pages.
        A page-render is only saved when the page actually contains vector
        graphics (i.e. it has drawing commands beyond simple text).

    Returns a list of dicts with keys:
        page       — 1-indexed page number
        path       — absolute path to the saved image file
        index      — image index within the page
        source     — "embedded" | "page_render"
        width, height
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    PAGE_IMG_DIR.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    images_info: List[Dict[str, Any]] = []

    log.info(f"Stage 2 — PyMuPDF: extracting images from {pdf_path} …")

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_label = page_num + 1   # 1-indexed

        # ------------------------------------------------------------------
        # Pass A: embedded raster images
        # ------------------------------------------------------------------
        embedded_on_page: List[Dict[str, Any]] = []
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                base_image   = doc.extract_image(xref)
                image_bytes  = base_image["image"]
                image_ext    = base_image["ext"]
                img_width    = base_image.get("width",  0)
                img_height   = base_image.get("height", 0)
            except Exception as exc:
                log.warning(f"  Could not extract image xref={xref} on page {page_label}: {exc}")
                continue

            if not _is_image_worth_keeping(img_width, img_height):
                log.debug(f"  Skipping tiny image {img_width}×{img_height} on page {page_label}")
                continue

            filename = f"page_{page_label}_img_{img_index + 1}.{image_ext}"
            filepath = output_dir / filename
            filepath.write_bytes(image_bytes)

            entry = {
                "page":   page_label,
                "path":   str(filepath),
                "index":  img_index + 1,
                "source": "embedded",
                "width":  img_width,
                "height": img_height,
            }
            embedded_on_page.append(entry)
            images_info.append(entry)

        # ------------------------------------------------------------------
        # Pass B: full-page pixmap for vector/drawn graphics
        #
        # Heuristic: render the page if it contains any drawing paths
        # (lines, curves, fills) — i.e. it's not purely a text page.
        # We detect this via page.get_drawings(), which returns an empty
        # list for text-only pages.
        # ------------------------------------------------------------------
        drawings = page.get_drawings()
        has_vector_content = len(drawings) > 0

        if has_vector_content:
            log.info(f"  Page {page_label}: {len(drawings)} vector drawing(s) detected — rendering full page.")
            try:
                mat      = fitz.Matrix(PAGE_RENDER_DPI / 72, PAGE_RENDER_DPI / 72)
                pixmap   = page.get_pixmap(matrix=mat, alpha=False)
                filename = f"page_{page_label}_render.png"
                filepath = PAGE_IMG_DIR / filename
                pixmap.save(str(filepath))

                images_info.append({
                    "page":   page_label,
                    "path":   str(filepath),
                    "index":  0,            # 0 = full-page render sentinel
                    "source": "page_render",
                    "width":  pixmap.width,
                    "height": pixmap.height,
                })
                log.info(f"  Saved full-page render → {filepath}")
            except Exception as exc:
                log.error(f"  Full-page render failed for page {page_label}: {exc}")

    doc.close()
    log.info(f"Image extraction complete — {len(images_info)} image(s) total.")
    return images_info


# ===========================================================================
# STAGE 3 — OCR: text extraction from image files
# ===========================================================================

def extract_text_from_image(image_path: str) -> str:
    """
    Run PaddleOCR on a single image and return the concatenated text.
    Returns an empty string if OCR is unavailable or produces no output.
    """
    engine = _init_ocr()
    if engine is None:
        return ""
    try:
        result = engine.ocr(image_path, cls=True)
        if not result or not result[0]:
            return ""
        lines = [line[1][0] for line in result[0] if line and line[1]]
        return "\n".join(lines).strip()
    except Exception as exc:
        log.error(f"OCR failed for {image_path}: {exc}")
        return ""


def _is_ocr_meaningful(text: str, min_chars: int = 15) -> bool:
    """
    Decide whether OCR output is semantically useful.
    A few stray characters from axis labels don't constitute a description.
    """
    # Require a minimum character count AND at least one word of ≥3 chars
    if len(text) < min_chars:
        return False
    words = [w for w in text.split() if len(w) >= 3]
    return len(words) >= 2


# ===========================================================================
# STAGE 4 — Ollama helpers: vision + text cleanup
# ===========================================================================

def _run_with_timeout(fn, timeout_seconds: int = 120):
    """Execute `fn` in a thread pool with a hard timeout."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        return future.result(timeout=timeout_seconds)


def _ollama_client():
    return ollama.Client(host="http://localhost:11434")


def get_image_vision_description(
    image_path: str,
    model: str = "minicpm-v",
    timeout: int = 120,
    context_hint: str = "",
) -> str:
    """
    Ask the Ollama vision model for a detailed description of an image.

    `context_hint` is optional surrounding text (e.g. a section heading)
    to help the model produce a more relevant description.

    Returns a human-readable description string, or an error sentinel.
    """
    if not OLLAMA_AVAILABLE:
        return "[Vision model unavailable — ollama not installed]"

    hint_clause = (
        f'This image appears in the section: "{context_hint}". ' if context_hint else ""
    )
    prompt = (
        f"{hint_clause}"
        "Describe this image thoroughly and precisely. "
        "If it is a chart or graph, identify the chart type, explain all axes, "
        "data series, trends, and key values. "
        "If it is a diagram or schematic, explain every component, label, and flow. "
        "If it contains a mathematical formula or equation, transcribe it in LaTeX. "
        "If it is a photograph or illustration, describe the subject and any text overlaid. "
        "Be specific — do not summarise vaguely."
    )
    log.info(f"  Querying Ollama vision ({model}): {Path(image_path).name}")
    try:
        def _call():
            response = _ollama_client().chat(
                model=model,
                messages=[{"role": "user", "content": prompt, "images": [image_path]}],
            )
            return response.get("message", {}).get("content", "").strip()

        return _run_with_timeout(_call, timeout_seconds=timeout)
    except FuturesTimeoutError:
        log.warning(f"  Vision model timed out after {timeout}s for {image_path}.")
        return "[Vision description timed out]"
    except Exception as exc:
        log.error(f"  Vision model failed for {image_path}: {exc}")
        return f"[Vision description failed: {exc}]"


def post_process_text(
    text: str,
    model: str = "phi3:mini",
    timeout: int = 60,
) -> str:
    """
    Use a lightweight LLM to correct OCR artefacts and normalise formatting.
    Falls back to the raw text on timeout or failure — never loses data.
    """
    if not OLLAMA_AVAILABLE or not text.strip():
        return text

    prompt = (
        "You are a professional document editor. "
        "The text below was extracted via OCR and may contain errors: "
        "broken hyphenation, merged words, garbled characters, or stray symbols. "
        "Fix only genuine OCR errors. Do NOT add, remove, or rewrite content. "
        "Return the corrected text only — no commentary, no markdown wrappers.\n\n"
        f"{text}"
    )
    log.info("  Post-processing OCR text …")
    try:
        def _call():
            response = _ollama_client().chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.get("message", {}).get("content", "").strip()

        return _run_with_timeout(_call, timeout_seconds=timeout)
    except FuturesTimeoutError:
        log.warning(f"  Text post-processing timed out — using raw OCR text.")
        return text
    except Exception as exc:
        log.error(f"  Text post-processing failed: {exc}")
        return text


# ===========================================================================
# STAGE 5 — Image processing: choose the best content strategy per image
# ===========================================================================

class ImageContentStrategy:
    """
    Decides and executes the best extraction strategy for a single image:

    1. Run OCR first (fast, zero network).
    2. If OCR returns meaningful text (prose, tables, captions) → clean with LLM.
    3. Always ALSO run vision model for complex images (charts, diagrams,
       full-page renders) regardless of OCR output, then merge both.
    4. For embedded images with good OCR text only → skip vision to save time.
    """

    ALWAYS_USE_VISION_SOURCES = {"page_render"}
    COMPLEX_EXTENSIONS = {".png", ".jpg", ".jpeg"}   # embedded images to double-check

    def __init__(
        self,
        vision_model: str = "minicpm-v",
        text_model:   str = "phi3:mini",
        vision_timeout: int = 120,
        text_timeout:   int = 60,
    ):
        self.vision_model   = vision_model
        self.text_model     = text_model
        self.vision_timeout = vision_timeout
        self.text_timeout   = text_timeout

    def process(
        self,
        image_info: Dict[str, Any],
        context_hint: str = "",
    ) -> str:
        """
        Returns a Markdown-formatted content block for one image.
        Always includes an image link, then text/description content below it.
        """
        path   = image_info["path"]
        source = image_info.get("source", "embedded")
        ext    = Path(path).suffix.lower()

        md_parts: List[str] = [f"\n![image]({path})\n"]

        # --- OCR pass (always attempt) ---
        ocr_raw  = extract_text_from_image(path)
        ocr_good = _is_ocr_meaningful(ocr_raw)

        # --- Decide whether to invoke vision ---
        needs_vision = (
            source in self.ALWAYS_USE_VISION_SOURCES   # full-page render
            or not ocr_good                            # OCR found nothing useful
        )

        if ocr_good:
            cleaned = post_process_text(ocr_raw, model=self.text_model, timeout=self.text_timeout)
            md_parts.append(f"**Extracted Text (OCR):**\n{cleaned}\n")

        if needs_vision:
            desc = get_image_vision_description(
                path,
                model=self.vision_model,
                timeout=self.vision_timeout,
                context_hint=context_hint,
            )
            if ocr_good:
                md_parts.append(f"**Visual Description:**\n{desc}\n")
            else:
                md_parts.append(f"**Description:**\n{desc}\n")

        return "\n".join(md_parts)


# ===========================================================================
# STAGE 6 — Formula handling
# ===========================================================================

def handle_formula_item(item, vision_model: str, vision_timeout: int) -> str:
    """
    Docling marks undecodable formulas as 'formula-not-decoded'.
    Instead of silently dropping them, we:
      1. Check if Docling attached an image to the item (it often does).
      2. If yes → send to vision model to transcribe as LaTeX.
      3. If no  → log a warning and emit a placeholder so the user knows.
    """
    text = item.text.strip()

    # Try to get an image from the item
    item_image_path: Optional[str] = None
    if hasattr(item, "image") and item.image is not None:
        try:
            # Docling may store the image as a PIL Image or a URI
            img_obj = item.image
            if hasattr(img_obj, "pil_image") and img_obj.pil_image is not None:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    img_obj.pil_image.save(tmp.name)
                    item_image_path = tmp.name
            elif hasattr(img_obj, "uri") and img_obj.uri:
                item_image_path = str(img_obj.uri)
        except Exception as exc:
            log.warning(f"  Could not extract formula image: {exc}")

    if item_image_path and OLLAMA_AVAILABLE:
        log.info("  Sending formula image to vision model for transcription …")
        latex = get_image_vision_description(
            item_image_path,
            model=vision_model,
            timeout=vision_timeout,
            context_hint="mathematical formula or equation",
        )
        # Clean up temp file
        if item_image_path.startswith(tempfile.gettempdir()):
            try:
                os.unlink(item_image_path)
            except OSError:
                pass
        return f"\n$$\n{latex}\n$$\n"
    else:
        log.warning("  Formula image unavailable — emitting placeholder.")
        raw_fallback = text.replace("formula-not-decoded", "").strip()
        return f"\n<!-- FORMULA NOT DECODED: {raw_fallback or 'unknown'} -->\n"


# ===========================================================================
# STAGE 7 — Markdown reconstruction
# ===========================================================================

def _label_str(item) -> str:
    """Safely extract a Docling item's label as a lowercase string."""
    label = getattr(item, "label", None)
    if label is None:
        return "text"
    return str(label).split(".")[-1].lower()


def _get_item_page(item) -> Optional[int]:
    """Return the 1-indexed page number from an item's provenance, or None."""
    if hasattr(item, "prov") and item.prov:
        return item.prov[0].page_no
    return None


def reconstruct_markdown(
    docling_doc,
    images_info: List[Dict[str, Any]],
    output_md: Path,
    vision_model: str = "minicpm-v",
    text_model:   str = "phi3:mini",
) -> str:
    """
    Walks through Docling's structured items in reading order and builds a
    clean Markdown string.  Images are injected at the correct page boundary.

    Special handling:
      - Headings: depth derived from `level`; None → h2
      - Formulas:  sent to vision model instead of being dropped
      - Page headers/footers: skipped (noise)
      - PictureItem: position is known from Docling; content comes from our
        PyMuPDF-extracted files matched by page number
      - All images (embedded + page-renders) are processed via ImageContentStrategy
    """
    log.info("Stage 7 — Reconstructing Markdown …")

    # Group images by page for O(1) lookup
    images_by_page: Dict[int, List[Dict[str, Any]]] = {}
    for img in images_info:
        images_by_page.setdefault(img["page"], []).append(img)

    strategy = ImageContentStrategy(
        vision_model=vision_model,
        text_model=text_model,
    )

    md_lines: List[str] = []
    current_page = 0
    last_section_heading = ""   # carry-forward for context hints

    def flush_images_for_page(page_num: int):
        """Inject all images belonging to `page_num` into the Markdown stream."""
        if page_num not in images_by_page:
            return
        for img in images_by_page.pop(page_num):
            content_block = strategy.process(img, context_hint=last_section_heading)
            md_lines.append(content_block)

    # ------------------------------------------------------------------
    # Walk Docling items
    # ------------------------------------------------------------------
    for item, level in docling_doc.iterate_items():
        item_page = _get_item_page(item)

        # Advance page counter and flush images that belong to earlier pages
        if item_page is not None:
            while current_page < item_page:
                current_page += 1
                flush_images_for_page(current_page)

        item_type  = item.__class__.__name__
        label      = _label_str(item)

        # --------------------------------------------------------------
        # TextItem
        # --------------------------------------------------------------
        if item_type == "TextItem":
            text = item.text.strip()
            if not text:
                continue

            # Formula: never drop — send to vision
            if "formula-not-decoded" in text:
                md_lines.append(
                    handle_formula_item(item, vision_model=vision_model, vision_timeout=120)
                )
                continue

            if label == "title":
                md_lines.append(f"# {text}\n")
                last_section_heading = text

            elif label == "section_header":
                # level is an int or None; guard against 0 (falsy but valid)
                depth = max(2, level) if (level is not None) else 2
                prefix = "#" * depth
                md_lines.append(f"{prefix} {text}\n")
                last_section_heading = text

            elif label == "list_item":
                md_lines.append(f"- {text}")

            elif label == "formula":
                # Properly decoded formula from Docling
                md_lines.append(f"\n$$\n{text}\n$$\n")

            elif label in ("page_header", "page_footer"):
                pass  # skip — pure noise in reconstructed output

            else:
                md_lines.append(f"{text}\n")

        # --------------------------------------------------------------
        # TableItem
        # --------------------------------------------------------------
        elif item_type == "TableItem":
            try:
                table_md = item.export_to_markdown()
                md_lines.append(f"\n{table_md}\n")
            except Exception as exc:
                log.warning(f"  Table export failed: {exc}")

        # --------------------------------------------------------------
        # PictureItem
        # We know WHERE in the document the picture lives (from Docling),
        # but we use our higher-quality PyMuPDF extraction for the actual
        # image bytes.  Match by page number.
        # --------------------------------------------------------------
        elif item_type == "PictureItem":
            if item_page and item_page in images_by_page:
                # Pop the first image on this page (best positional proxy)
                img = images_by_page[item_page].pop(0)
                if not images_by_page[item_page]:
                    del images_by_page[item_page]
                content_block = strategy.process(img, context_hint=last_section_heading)
                md_lines.append(content_block)
            # If no PyMuPDF image matches, fall through silently — the
            # full-page render (if present) will catch it later.

    # ------------------------------------------------------------------
    # Flush any remaining images (pages after the last text element)
    # ------------------------------------------------------------------
    if images_by_page:
        remaining_pages = sorted(images_by_page.keys())
        log.info(f"  Flushing images from {len(remaining_pages)} remaining page(s) …")
        for page_num in remaining_pages:
            for img in images_by_page[page_num]:
                content_block = strategy.process(img, context_hint="")
                md_lines.append(content_block)

    # ------------------------------------------------------------------
    # Write Markdown file
    # ------------------------------------------------------------------
    full_markdown = "\n".join(md_lines)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(full_markdown, encoding="utf-8")
    log.info(f"Markdown saved → {output_md}")
    return full_markdown


# ===========================================================================
# STAGE 8 — JSON output (Docling-native, not LLM-generated)
# ===========================================================================

def export_json(
    docling_doc,
    full_markdown: str,
    output_json: Path,
) -> None:
    """
    Produces two JSON files:

    1. <name>_structured.json  — Docling's native structural export.
       This is deterministic, lossless, and covers every element Docling
       identified (sections, tables, pictures, formulas, page layout).

    2. <name>_toc.json         — A lightweight table-of-contents extracted
       directly from the Markdown (headings + page references).
       Useful for navigation without parsing the full structure.

    We deliberately avoid asking an LLM to 'convert Markdown to JSON' because
    that process is non-deterministic and prone to hallucination on long docs.
    """
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # --- 1. Docling native export ---
    try:
        native_data = docling_doc.export_to_dict()
        output_json.write_text(json.dumps(native_data, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info(f"Structured JSON saved → {output_json}")
    except Exception as exc:
        log.error(f"Docling native JSON export failed: {exc}")

    # --- 2. Table of contents ---
    toc: List[Dict[str, Any]] = []
    for line in full_markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped.lstrip("#").strip()
            toc.append({"level": level, "title": title})

    toc_path = output_json.with_name(output_json.stem.replace("_structured", "") + "_toc.json")
    toc_path.write_text(json.dumps(toc, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Table of contents JSON saved → {toc_path}")


# ===========================================================================
# Main pipeline orchestrator
# ===========================================================================

def process_pdf_pipeline(
    pdf_path: str,
    output_md:      Path = OUTPUT_DIR / "output.md",
    output_json:    Path = OUTPUT_DIR / "output_structured.json",
    vision_model:   str  = "minicpm-v",
    text_model:     str  = "phi3:mini",
    page_render_dpi: int = PAGE_RENDER_DPI,
) -> None:
    """
    Full pipeline:
        1  Docling   — structured text, tables, layout
        2  PyMuPDF   — embedded images + full-page renders for vector graphics
        3  PaddleOCR — text inside images
        4  Ollama    — vision descriptions + OCR cleanup + formula transcription
        5  Markdown  — clean, ordered, rich output
        6  JSON      — deterministic structural export (Docling-native)
    """
    if not os.path.exists(pdf_path):
        log.error(f"PDF not found: {pdf_path}")
        return

    # Expose DPI setting globally so extract_pymupdf_images can use it
    global PAGE_RENDER_DPI
    PAGE_RENDER_DPI = page_render_dpi

    try:
        # Stage 1
        docling_doc = extract_docling_structure(pdf_path)

        # Stage 2
        images_info = extract_pymupdf_images(pdf_path, output_dir=IMAGES_DIR)

        # Stages 3-7 combined in reconstruction
        full_markdown = reconstruct_markdown(
            docling_doc,
            images_info,
            output_md,
            vision_model=vision_model,
            text_model=text_model,
        )

        # Stage 8
        export_json(docling_doc, full_markdown, output_json)

        log.info("=" * 60)
        log.info("Pipeline completed successfully.")
        log.info(f"  Markdown  → {output_md}")
        log.info(f"  JSON      → {output_json}")
        log.info("=" * 60)

    except Exception as exc:
        log.exception(f"Pipeline failed: {exc}")
        raise


# ===========================================================================
# CLI entry point
# ===========================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Robust multi-layer PDF content extraction pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("pdf", help="Path to the input PDF file.")
    p.add_argument(
        "--vision-model", default="minicpm-v",
        help="Ollama model for image/formula vision description.",
    )
    p.add_argument(
        "--text-model", default="phi3:mini",
        help="Ollama model for OCR text cleanup.",
    )
    p.add_argument(
        "--dpi", type=int, default=150,
        help="DPI for full-page pixmap renders (higher = sharper but slower).",
    )
    p.add_argument(
        "--output-dir", default=str(OUTPUT_DIR),
        help="Root directory for all pipeline outputs.",
    )
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Override globals with CLI values
    OUTPUT_DIR   = out_dir
    IMAGES_DIR   = out_dir / "extracted_images"
    PAGE_IMG_DIR = out_dir / "page_renders"

    base     = Path(args.pdf).stem
    out_md   = out_dir / f"{base}_output.md"
    out_json = out_dir / f"{base}_structured.json"

    process_pdf_pipeline(
        pdf_path        = args.pdf,
        output_md       = out_md,
        output_json     = out_json,
        vision_model    = args.vision_model,
        text_model      = args.text_model,
        page_render_dpi = args.dpi,
    )