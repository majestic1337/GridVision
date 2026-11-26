"""Fast, dependency-light PDF page profiling for adaptive extraction."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

try:
    import fitz
except ImportError:  # pragma: no cover - profiling can still be unit tested with fakes
    fitz = None


@dataclass(frozen=True)
class PageProfile:
    page_number: int
    page_idx: int
    page_width: float
    page_height: float
    native_char_count: int
    raster_coverage: float
    drawing_count: int
    image_count: int
    thumbnail_variance: float
    image_rects: Tuple[Tuple[float, float, float, float], ...] = field(default_factory=tuple)
    classification: str = "blank"

    @property
    def is_native(self) -> bool:
        return self.classification == "native"

    @property
    def is_vector_heavy(self) -> bool:
        return self.classification == "vector"

    @property
    def is_tiled_scan(self) -> bool:
        return self.classification == "tiled_scan"

    @property
    def native_chars(self) -> int:
        return self.native_char_count

    @property
    def drawings(self) -> int:
        return self.drawing_count


class DocumentProfiler:
    """Inspect pages using signals that are cheap to obtain with PyMuPDF.

    The profiler intentionally does not perform OCR or Docling conversion. It
    only decides which extraction path should handle each page.
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        config = config or {}
        self.min_native_chars = int(config.get("min_native_chars", 150))
        self.raster_coverage_threshold = float(
            config.get("scanned_raster_coverage_threshold", 0.70)
        )
        self.tiled_slice_count_threshold = int(
            config.get("tiled_slice_count_threshold", 3)
        )
        self.thumbnail_variance_threshold = float(
            config.get("thumbnail_variance_threshold", 1.0)
        )

    def profile_document(self, document: Any) -> List[PageProfile]:
        return [self.profile_page(document.load_page(index), index) for index in range(document.page_count)]

    def profile(self, document: Any) -> List[PageProfile]:
        """Compatibility alias for callers that use the shorter API name."""
        return self.profile_document(document)

    def profile_page(self, page: Any, page_idx: int = 0) -> PageProfile:
        rect = getattr(page, "rect", None)
        width = max(float(getattr(rect, "width", 0.0)), 1.0)
        height = max(float(getattr(rect, "height", 0.0)), 1.0)
        text = self._page_text(page)
        native_char_count = len(text)
        image_rects = tuple(self._image_rects(page))
        coverage = self._union_area(image_rects, width, height) / (width * height)
        drawing_count = self._drawing_count(page)
        # Native text, raster coverage, and drawing signals are enough to
        # classify most pages. Render a thumbnail only for ambiguous pages so
        # profiling stays cheap on large native manuals.
        if native_char_count >= self.min_native_chars or coverage >= self.raster_coverage_threshold or drawing_count:
            variance = 0.0
        else:
            variance = self._thumbnail_variance(page, width, height)
        image_count = len(image_rects)

        classification = self._classify(
            native_char_count=len(text),
            raster_coverage=coverage,
            drawing_count=drawing_count,
            image_count=image_count,
            thumbnail_variance=variance,
        )
        return PageProfile(
            page_number=page_idx + 1,
            page_idx=page_idx,
            page_width=width,
            page_height=height,
            native_char_count=len(text),
            raster_coverage=coverage,
            drawing_count=drawing_count,
            image_count=image_count,
            thumbnail_variance=variance,
            image_rects=image_rects,
            classification=classification,
        )

    def _classify(
        self,
        *,
        native_char_count: int,
        raster_coverage: float,
        drawing_count: int,
        image_count: int,
        thumbnail_variance: float,
    ) -> str:
        # A page made of many image strips is a scan even if each individual
        # strip has an extreme aspect ratio. Keep the signal at page level.
        tiled = (
            image_count >= self.tiled_slice_count_threshold
            and raster_coverage >= self.raster_coverage_threshold
        )
        if tiled:
            return "tiled_scan"

        scanned = raster_coverage >= self.raster_coverage_threshold or (
            image_count >= self.tiled_slice_count_threshold
            and raster_coverage >= self.raster_coverage_threshold * 0.5
        )
        if scanned:
            return "scanned"

        # Large vector-path populations are a strong schematic signal even
        # when a PDF also contains selectable labels.
        if drawing_count > 1000:
            return "vector"

        if native_char_count >= self.min_native_chars:
            return "native"

        if drawing_count > 5 and native_char_count < 50 and image_count == 0:
            return "vector"

        if native_char_count or thumbnail_variance >= self.thumbnail_variance_threshold:
            return "ocr"
        return "blank"

    @staticmethod
    def _page_text(page: Any) -> str:
        try:
            return str(page.get_text("text") or "")
        except (AttributeError, TypeError, RuntimeError):
            return ""

    @staticmethod
    def _drawing_count(page: Any) -> int:
        try:
            get_drawings = getattr(page, "get_cdrawings", None)
            if get_drawings is None:
                get_drawings = getattr(page, "get_drawings", None)
            if get_drawings is None:
                return 0
            return len(get_drawings() or [])
        except (AttributeError, TypeError, RuntimeError):
            return 0

    @staticmethod
    def _image_rects(page: Any) -> Iterable[Tuple[float, float, float, float]]:
        try:
            images = page.get_images(full=True) or []
        except (AttributeError, TypeError, RuntimeError):
            return []
        rects: List[Tuple[float, float, float, float]] = []
        for image in images:
            if not image:
                continue
            try:
                placements = page.get_image_rects(image[0]) or []
            except (AttributeError, TypeError, RuntimeError):
                continue
            for placement in placements:
                try:
                    rects.append(
                        (
                            float(placement.x0),
                            float(placement.y0),
                            float(placement.x1),
                            float(placement.y1),
                        )
                    )
                except (AttributeError, TypeError, ValueError):
                    continue
        return rects

    @staticmethod
    def _union_area(
        rects: Iterable[Tuple[float, float, float, float]],
        page_width: float,
        page_height: float,
    ) -> float:
        """Calculate rectangle union area with a small x-sweep."""
        normalized = []
        for left, top, right, bottom in rects:
            left = max(0.0, min(page_width, left))
            right = max(0.0, min(page_width, right))
            top = max(0.0, min(page_height, top))
            bottom = max(0.0, min(page_height, bottom))
            if right > left and bottom > top:
                normalized.append((left, top, right, bottom))
        if not normalized:
            return 0.0

        x_edges = sorted({edge for rect in normalized for edge in (rect[0], rect[2])})
        area = 0.0
        for left, right in zip(x_edges, x_edges[1:]):
            spans = sorted((rect[1], rect[3]) for rect in normalized if rect[0] < right and rect[2] > left)
            covered_height = 0.0
            current_top = current_bottom = None
            for top, bottom in spans:
                if current_top is None:
                    current_top, current_bottom = top, bottom
                elif top > current_bottom:
                    covered_height += current_bottom - current_top
                    current_top, current_bottom = top, bottom
                else:
                    current_bottom = max(current_bottom, bottom)
            if current_top is not None:
                covered_height += current_bottom - current_top
            area += (right - left) * covered_height
        return area

    def _thumbnail_variance(self, page: Any, width: float, height: float) -> float:
        try:
            matrix = None
            if fitz is not None:
                scale = min(64.0 / max(width, 1.0), 64.0 / max(height, 1.0))
                matrix = fitz.Matrix(scale, scale)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            samples = list(getattr(pixmap, "samples", b""))
            channels = max(int(getattr(pixmap, "n", 1)), 1)
            if not samples:
                return 0.0
            values = [sum(samples[index:index + channels]) / channels for index in range(0, len(samples), channels)]
            return float(statistics.pvariance(values)) if len(values) > 1 else 0.0
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return 0.0
