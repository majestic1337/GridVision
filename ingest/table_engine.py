"""Table detection and semantic ownership helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class TableCell:
    text: str
    bbox: Tuple[float, float, float, float]


@dataclass(frozen=True)
class TableRegion:
    bbox: Tuple[float, float, float, float]
    cells: Tuple[TableCell, ...] = field(default_factory=tuple)
    source: str = "pymupdf"

    @property
    def has_structured_content(self) -> bool:
        return bool(self.cells and any(cell.text.strip() for cell in self.cells))


class TableProcessingEngine:
    """Detect table regions and prevent duplicate raw-text ownership."""

    def detect_tables(self, page: Any) -> List[TableRegion]:
        regions = self._find_tables_api(page)
        if regions:
            return regions
        return self._find_vector_tables(page)

    def suppress_owned_text(
        self,
        items: Sequence[Tuple[Any, str]],
        table_regions: Sequence[TableRegion],
    ) -> List[Tuple[Any, str]]:
        """Drop text only when a structured table cell owns that text area."""
        owned_regions = [region for region in table_regions if region.has_structured_content]
        if not owned_regions:
            return list(items)

        filtered = []
        for item, item_type in items:
            if item_type != "text":
                filtered.append((item, item_type))
                continue
            bbox = self._bbox(item)
            if bbox and any(self._intersects_cell(bbox, region) for region in owned_regions):
                continue
            filtered.append((item, item_type))
        return filtered

    def filter_structured_items(self, items: Sequence[Tuple[Any, str]]) -> List[Tuple[Any, str]]:
        """Apply semantic ownership when table objects expose cell geometry."""
        regions = []
        for item, item_type in items:
            if item_type == "table":
                region = self._region_from_table_item(item)
                if region:
                    regions.append(region)
        return self.suppress_owned_text(items, regions)

    def _find_tables_api(self, page: Any) -> List[TableRegion]:
        find_tables = getattr(page, "find_tables", None)
        if not callable(find_tables):
            return []
        try:
            result = find_tables()
            tables = getattr(result, "tables", result) or []
        except Exception:
            return []
        regions = []
        for table in tables:
            bbox = self._bbox(getattr(table, "bbox", table))
            if not bbox:
                continue
            cells = self._cells_from_table(table)
            regions.append(TableRegion(tuple(bbox), tuple(cells), "pymupdf"))
        return regions

    def _find_vector_tables(self, page: Any) -> List[TableRegion]:
        drawings = getattr(page, "get_cdrawings", None) or getattr(page, "get_drawings", None)
        if not callable(drawings):
            return []
        try:
            paths = drawings() or []
        except Exception:
            return []
        rects = []
        horizontal = vertical = 0
        for path in paths:
            rect = self._bbox(path.get("rect") if isinstance(path, dict) else getattr(path, "rect", None))
            if not rect:
                continue
            left, top, right, bottom = rect
            if right - left > 20 and bottom - top < 3:
                horizontal += 1
            if bottom - top > 20 and right - left < 3:
                vertical += 1
            rects.append(rect)
        if horizontal < 2 or vertical < 2 or not rects:
            return []
        return [TableRegion(self._union(rects), (), "vector_heuristic")]

    def _region_from_table_item(self, item: Any) -> Optional[TableRegion]:
        bbox = self._bbox(item)
        if not bbox:
            return None
        cells = self._cells_from_table(item)
        return TableRegion(tuple(bbox), tuple(cells), "docling") if cells else None

    def _cells_from_table(self, table: Any) -> List[TableCell]:
        raw_cells = getattr(table, "cells", None) or getattr(table, "table_cells", None) or []
        cells: List[TableCell] = []
        for cell in raw_cells:
            bbox = self._bbox(cell)
            text = getattr(cell, "text", None)
            if text is None and isinstance(cell, dict):
                text = cell.get("text", "")
            if bbox and text is not None:
                cells.append(TableCell(str(text), tuple(bbox)))
        return cells

    @staticmethod
    def _bbox(value: Any) -> Optional[List[float]]:
        if value is None:
            return None
        if isinstance(value, dict):
            return TableProcessingEngine._bbox(value.get("bbox") or value.get("rect"))
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return [float(part) for part in value]
        coords = []
        names = ("x0", "y0", "x1", "y1") if any(
            hasattr(value, name) for name in ("x0", "y0", "x1", "y1")
        ) else ("l", "t", "r", "b")
        for name in names:
            if hasattr(value, name):
                coords.append(float(getattr(value, name)))
        if len(coords) == 4:
            return coords
        provenance = getattr(value, "prov", None) or []
        if provenance:
            first = getattr(provenance[0], "bbox", None)
            return TableProcessingEngine._bbox(first)
        return None

    @staticmethod
    def _union(rects: Iterable[Sequence[float]]) -> Tuple[float, float, float, float]:
        values = list(rects)
        return (
            min(rect[0] for rect in values),
            min(rect[1] for rect in values),
            max(rect[2] for rect in values),
            max(rect[3] for rect in values),
        )

    @staticmethod
    def _intersects_cell(bbox: Sequence[float], region: TableRegion) -> bool:
        for cell in region.cells:
            if not (
                bbox[2] < cell.bbox[0]
                or bbox[0] > cell.bbox[2]
                or bbox[3] < cell.bbox[1]
                or bbox[1] > cell.bbox[3]
            ):
                return True
        return False
