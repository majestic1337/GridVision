"""Page-level synthesis for tiled scanned PDF pages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class OCRText:
    text: str
    confidence: float = 1.0


@dataclass(frozen=True)
class SynthesizedPage:
    page_number: int
    png_bytes: bytes
    ocr: OCRText


class TiledPageSynthesizer:
    """Render all strips on a page as one image and OCR it once."""

    def __init__(self, dpi: int = 200, ocr_enabled: bool = True, ocr_engine: str = "rapidocr"):
        self.dpi = max(72, int(dpi))
        self.ocr_enabled = bool(ocr_enabled)
        self.ocr_engine = ocr_engine
        self._ocr = None

    def render_page(self, page: Any) -> Any:
        matrix = self._matrix(page)
        return page.get_pixmap(matrix=matrix, alpha=False)

    def synthesize(self, page: Any, page_number: int) -> SynthesizedPage:
        pixmap = self.render_page(page)
        png_bytes = pixmap.tobytes("png")
        ocr = self.extract_text(png_bytes)
        return SynthesizedPage(page_number=page_number, png_bytes=png_bytes, ocr=ocr)

    def synthesize_page(self, page: Any, page_number: int) -> SynthesizedPage:
        return self.synthesize(page, page_number)

    def extract_text(self, image: Any) -> OCRText:
        if not self.ocr_enabled:
            return OCRText("")
        try:
            engine = self._get_ocr()
            if engine is None:
                return OCRText("")
            result = engine(image)
            texts = list(getattr(result, "txts", None) or [])
            scores = list(getattr(result, "scores", None) or [])
            confidence = sum(float(score) for score in scores) / len(scores) if scores else 1.0
            # Preserve line boundaries as paragraph boundaries. This keeps
            # adjacent OCR rows from being merged by downstream chunking.
            return OCRText("\n\n".join(str(text).strip() for text in texts if str(text).strip()), confidence)
        except Exception:
            # OCR is an optional enhancement. The synthesized image still
            # exists for later recovery without making ingestion fail.
            return OCRText("")

    def _get_ocr(self) -> Optional[Any]:
        if self._ocr is not None:
            return self._ocr
        if self.ocr_engine.lower() not in {"rapidocr", "tesseract", "auto"}:
            return None
        try:
            from rapidocr import RapidOCR

            self._ocr = RapidOCR()
        except Exception:
            self._ocr = None
        return self._ocr

    def _matrix(self, page: Any) -> Any:
        try:
            import fitz

            scale = self.dpi / 72.0
            return fitz.Matrix(scale, scale)
        except ImportError:
            return None


