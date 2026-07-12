"""OCR fallback for scanned/image-only PDFs.

pdfplumber (and any text-layer extractor) returns nothing for a résumé that
was scanned from paper or exported as a flattened image — there's no text
object to read, only pixels. Rather than fail the whole upload, we rasterize
each page with pypdfium2 (already a transitive dependency, no external
poppler binary needed — unlike pdf2image) and run Tesseract over the image.

Like the transformer NER model, this is optional and fails soft: if
Tesseract isn't installed on the host, OCR is skipped with a logged warning
rather than crashing the request. Résumé OCR only has to clear a fairly low
bar (find section headers, skills, dates) so Tesseract's general-purpose
English model is sufficient — no fine-tuning needed.
"""

from __future__ import annotations

import io
import logging
from functools import lru_cache

import pypdfium2 as pdfium

from ..config import get_settings

logger = logging.getLogger("cv_parser.ocr")


class OcrEngine:
    """Thin, fail-soft wrapper around pytesseract, mirroring TransformerNer's
    lazy-load-and-degrade-gracefully shape in ner.py."""

    def __init__(self) -> None:
        self._load_attempted = False
        self.available = False
        self._pytesseract = None

    def _ensure_loaded(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        settings = get_settings()
        if not settings.enable_ocr:
            logger.info("OCR disabled via CV_PARSER_ENABLE_OCR=0")
            return
        try:
            import pytesseract

            # get_tesseract_version() actually invokes the binary — this is
            # the cheapest way to fail fast if Tesseract isn't installed,
            # rather than discovering that mid-request on the first real OCR
            # call.
            pytesseract.get_tesseract_version()
            self._pytesseract = pytesseract
            self.available = True
            logger.info("Tesseract OCR available (%s).", pytesseract.get_tesseract_version())
        except Exception as exc:  # noqa: BLE001 - must never crash the pipeline
            logger.warning(
                "Tesseract OCR unavailable (%s). Scanned/image-only PDFs will "
                "return no text rather than being OCR'd.",
                exc,
            )
            self.available = False

    def extract_text(self, data: bytes, max_pages: int) -> tuple[str, list[str]]:
        """Rasterize up to `max_pages` pages and OCR each. Returns
        (concatenated_text, warnings)."""
        self._ensure_loaded()
        warnings: list[str] = []
        if not self.available or self._pytesseract is None:
            return "", ["OCR unavailable — Tesseract is not installed on this host."]

        settings = get_settings()
        scale = settings.ocr_dpi / 72.0
        lines: list[str] = []
        try:
            pdf = pdfium.PdfDocument(io.BytesIO(data))
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR: failed to open PDF for rasterization (%s).", exc)
            return "", [f"OCR failed: could not open PDF ({exc})."]

        try:
            page_count = len(pdf)
            pages_to_process = min(page_count, max_pages)
            if page_count > max_pages:
                warnings.append(
                    f"PDF has {page_count} pages; OCR was limited to the first "
                    f"{max_pages} for performance."
                )
            for i in range(pages_to_process):
                try:
                    page = pdf[i]
                    bitmap = page.render(scale=scale)
                    image = bitmap.to_pil()
                    page_text = self._pytesseract.image_to_string(image)
                    if page_text.strip():
                        lines.append(page_text)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("OCR failed on page %d (%s).", i + 1, exc)
                    warnings.append(f"OCR failed on page {i + 1}.")
        finally:
            pdf.close()

        return "\n".join(lines), warnings


@lru_cache(maxsize=1)
def get_ocr_engine() -> OcrEngine:
    return OcrEngine()
