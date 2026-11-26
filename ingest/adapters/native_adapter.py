"""Native PyMuPDF extraction adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass(frozen=True)
class NativeTextBlock:
    text: str
    bbox: Optional[List[float]]
    subtype: str = "body"


class NativePdfAdapter:
    """Extract text blocks without invoking Docling or OCR."""

    def extract(self, page: Any) -> List[NativeTextBlock]:
        return self.extract_page(page)

    def extract_page(self, page: Any) -> List[NativeTextBlock]:
        try:
            blocks = page.get_text("blocks") or []
        except (AttributeError, TypeError, RuntimeError):
            return []

        extracted: List[NativeTextBlock] = []
        for block in blocks:
            if len(block) < 5:
                continue
            text = str(block[4] or "").strip()
            if not text:
                continue
            bbox = [float(value) for value in block[:4]]
            extracted.append(NativeTextBlock(text=text, bbox=bbox, subtype="body"))
        extracted.sort(key=lambda item: (item.bbox[1] if item.bbox else 0.0, item.bbox[0] if item.bbox else 0.0))
        return extracted

    def extract_vector_marker(self, page: Any, page_number: int) -> Optional[str]:
        try:
            get_drawings = getattr(page, "get_cdrawings", None) or getattr(page, "get_drawings", None)
            drawing_count = len(get_drawings() or []) if get_drawings else 0
        except (AttributeError, TypeError, RuntimeError):
            drawing_count = 0
        if drawing_count <= 0:
            return None
        return f"Vector graphic schematic on page {page_number}; drawing paths: {drawing_count}."
