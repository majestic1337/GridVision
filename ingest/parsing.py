import hashlib
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Generator, Iterable
from dataclasses import dataclass
from collections import defaultdict

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except Exception:
    PYMUPDF_AVAILABLE = False

try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

PARA_Y_GAP = 32.0
HASH_CHUNK_SIZE = 4096

logger = logging.getLogger("GridVision_Parser")

@dataclass
class ParsedElement:
    element_id: str
    type: str          # text | table | image
    content: str
    page_idx: int      # 0-based
    page_number: int   # 1-based
    page_label: str
    bbox: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentParser:
    def __init__(self):
        if not DOCLING_AVAILABLE:
            logger.warning("Docling not found. Install with `pip install docling`.")
            self.converter = None
        else:
            self.converter = DocumentConverter()

        if not PYMUPDF_AVAILABLE:
            logger.warning("PyMuPDF not found. Install with `pip install pymupdf`.")

    def generate_doc_id(self, file_path: Path) -> Tuple[str, str]:
        stem = file_path.stem
        doc_slug = re.sub(r"[^\w\-]", "_", stem)

        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(HASH_CHUNK_SIZE), b""):
                hasher.update(chunk)

        return hasher.hexdigest(), doc_slug

    def parse_file(self, file_path: Path, doc_id: str) -> Generator[ParsedElement, None, None]:
        doc = self._convert_docling(file_path)
        if doc is None:
            return

        docling_items = self.extract_docling_items(doc)
        pymupdf_items = self.extract_pymupdf_images(file_path)
        all_items = docling_items + pymupdf_items

        logger.info(
            f"Found {len(all_items)} total elements "
            f"(Texts: {sum(1 for _, t in docling_items if t=='text')}, "
            f"Tables: {sum(1 for _, t in docling_items if t=='table')}, "
            f"Images(PyMuPDF placements): {len(pymupdf_items)})"
        )

        if not all_items:
            logger.warning("No elements found! Check if PDF is valid or OCR failed.")
            return

        pages_map = self.group_by_page(all_items)

        for page_no in sorted(pages_map.keys()):
            page_idx = page_no - 1
            page_label = str(page_no)

            page_items = pages_map[page_no]
            page_items = self.sort_page_items(page_items)

            page_ctx = {
                "doc_id": doc_id,
                "doc": doc,
                "page_no": page_no,
                "page_idx": page_idx,
                "page_label": page_label,
            }

            yield from self.page_to_elements(page_items, page_ctx)

    def _convert_docling(self, file_path: Path) -> Any:
        if not DOCLING_AVAILABLE or self.converter is None:
            raise RuntimeError("Docling Missing")

        logger.info(f"Parsing structure for {file_path.name}...")

        try:
            res = self.converter.convert(file_path)
            return res.document
        except Exception as e:
            logger.error(f"Docling conversion failed: {e}")
            return None

    def extract_docling_items(self, doc: Any) -> List[Tuple[Any, str]]:
        items: List[Tuple[Any, str]] = []

        if hasattr(doc, "texts"):
            for it in doc.texts:
                items.append((it, "text"))

        if hasattr(doc, "tables"):
            for it in doc.tables:
                items.append((it, "table"))

        return items

    def extract_pymupdf_images(self, file_path: Path) -> List[Tuple[Dict[str, Any], str]]:
        if not PYMUPDF_AVAILABLE:
            return []

        images: List[Tuple[Dict[str, Any], str]] = []
        doc = fitz.open(str(file_path))
        try:
            for page_idx in range(doc.page_count):
                page = doc.load_page(page_idx)
                for img in page.get_images(full=True):
                    xref = img[0]
                    rects = page.get_image_rects(xref)

                    for r in rects:
                        images.append(
                            (
                                {
                                    "kind": "pymupdf_image",
                                    "page_no": page_idx + 1,
                                    "xref": int(xref),
                                    "rect": [float(r.x0), float(r.y0), float(r.x1), float(r.y1)],
                                },
                                "image",
                            )
                        )
        finally:
            doc.close()

        return images

    def group_by_page(self, items: List[Tuple[Any, str]]) -> Dict[int, List[Tuple[Any, str]]]:
        pages_map: Dict[int, List[Tuple[Any, str]]] = defaultdict(list)

        for item, type_str in items:
            page_no = self._infer_page_no(item)
            pages_map[page_no].append((item, type_str))

        return pages_map

    def sort_page_items(self, page_items: List[Tuple[Any, str]]) -> List[Tuple[Any, str]]:
        def sort_key(entry: Tuple[Any, str]) -> Tuple[float, float]:
            bbox = self._get_safe_bbox(entry[0])
            if bbox and len(bbox) == 4:
                return (float(bbox[1]), float(bbox[0]))
            return (0.0, 0.0)

        out = list(page_items)
        out.sort(key=sort_key)
        return out

    def page_to_elements(
        self,
        page_items: List[Tuple[Any, str]],
        page_ctx: Dict[str, Any],
    ) -> Iterable[ParsedElement]:
        doc_id: str = page_ctx["doc_id"]
        doc: Any = page_ctx["doc"]
        page_no: int = page_ctx["page_no"]
        page_idx: int = page_ctx["page_idx"]
        page_label: str = page_ctx["page_label"]

        seq_counter = 0

        text_buf: List[str] = []
        raw_items_buf: List[Any] = []
        bbox_buf: List[List[float]] = []
        prev_text_bbox: Optional[List[float]] = None

        def label_to_subtype(it: Any) -> str:
            st = "body"
            if hasattr(it, "label"):
                lbl = str(it.label).upper()
                if "FOOTER" in lbl:
                    st = "footer"
                elif "HEADER" in lbl or "TITLE" in lbl:
                    st = "header"
            return st

        def bbox_union(bboxes: List[List[float]]) -> Optional[List[float]]:
            bxs = [b for b in bboxes if b and len(b) == 4]
            if not bxs:
                return None
            l = min(b[0] for b in bxs)
            t = min(b[1] for b in bxs)
            r = max(b[2] for b in bxs)
            bb = max(b[3] for b in bxs)
            return [l, t, r, bb]

        def flush_paragraph() -> Optional[ParsedElement]:
            nonlocal seq_counter, text_buf, raw_items_buf, bbox_buf, prev_text_bbox

            if not text_buf:
                return None

            paragraph_text = "\n".join([t for t in text_buf if t]).strip()
            if not paragraph_text:
                text_buf, raw_items_buf, bbox_buf = [], [], []
                prev_text_bbox = None
                return None

            seq_counter += 1
            stable_id = self._generate_stable_id(doc_id, page_idx, seq_counter, "text")

            meta = {
                "raw_item": None,
                "raw_items": raw_items_buf.copy(),
                "subtype": "body",
                "seq_id": seq_counter,
                "raw_table": None,
                "parent_doc": doc,
                "source": "docling",
            }

            el = ParsedElement(
                element_id=stable_id,
                type="text",
                content=paragraph_text,
                page_idx=page_idx,
                page_number=page_no,
                page_label=page_label,
                bbox=bbox_union(bbox_buf),
                metadata=meta,
            )

            text_buf, raw_items_buf, bbox_buf = [], [], []
            prev_text_bbox = None
            return el

        for item, base_type in page_items:
            bbox = self._get_safe_bbox(item)

            if base_type == "text":
                subtype = label_to_subtype(item)
                txt = (getattr(item, "text", "") or "").strip()

                if subtype != "body":
                    flushed = flush_paragraph()
                    if flushed:
                        yield flushed

                    seq_counter += 1
                    stable_id = self._generate_stable_id(doc_id, page_idx, seq_counter, "text")
                    meta = {
                        "raw_item": item,
                        "subtype": subtype,
                        "seq_id": seq_counter,
                        "raw_table": None,
                        "parent_doc": doc,
                        "source": "docling",
                    }
                    yield ParsedElement(
                        element_id=stable_id,
                        type="text",
                        content=txt,
                        page_idx=page_idx,
                        page_number=page_no,
                        page_label=page_label,
                        bbox=bbox,
                        metadata=meta,
                    )
                    prev_text_bbox = None
                    continue

                new_paragraph = False
                if prev_text_bbox and bbox and len(prev_text_bbox) == 4 and len(bbox) == 4:
                    y_gap = float(bbox[1]) - float(prev_text_bbox[3])
                    if y_gap > PARA_Y_GAP:
                        new_paragraph = True

                if new_paragraph:
                    flushed = flush_paragraph()
                    if flushed:
                        yield flushed

                if txt:
                    text_buf.append(txt)
                    raw_items_buf.append(item)
                    if bbox:
                        bbox_buf.append(bbox)
                    prev_text_bbox = bbox if bbox else prev_text_bbox

                continue

            flushed = flush_paragraph()
            if flushed:
                yield flushed

            if base_type == "table":
                seq_counter += 1
                stable_id = self._generate_stable_id(doc_id, page_idx, seq_counter, "table")
                meta = {
                    "raw_item": item,
                    "subtype": "body",
                    "seq_id": seq_counter,
                    "raw_table": item,
                    "parent_doc": doc,
                    "source": "docling",
                }
                yield ParsedElement(
                    element_id=stable_id,
                    type="table",
                    content="<TABLE_PLACEHOLDER>",
                    page_idx=page_idx,
                    page_number=page_no,
                    page_label=page_label,
                    bbox=bbox,
                    metadata=meta,
                )
                continue

            if base_type == "image":
                seq_counter += 1
                stable_id = self._generate_stable_id(doc_id, page_idx, seq_counter, "image")
                meta = {
                    "raw_item": item,
                    "subtype": "body",
                    "seq_id": seq_counter,
                    "raw_table": None,
                    "parent_doc": doc,
                    "source": "pymupdf"
                    if (isinstance(item, dict) and item.get("kind") == "pymupdf_image")
                    else "docling",
                }
                yield ParsedElement(
                    element_id=stable_id,
                    type="image",
                    content="",
                    page_idx=page_idx,
                    page_number=page_no,
                    page_label=page_label,
                    bbox=bbox,
                    metadata=meta,
                )
                continue

        flushed = flush_paragraph()
        if flushed:
            yield flushed

    def _generate_stable_id(self, doc_id: str, page_idx: int, seq_id: int, base_type: str) -> str:
        return f"{doc_id}_p{page_idx}_{seq_id:02d}_{base_type}"

    def _infer_page_no(self, item: Any) -> int:
        if hasattr(item, "prov") and getattr(item, "prov", None):
            return int(item.prov[0].page_no)
        if isinstance(item, dict) and item.get("kind") == "pymupdf_image":
            return int(item["page_no"])
        if hasattr(item, "page_no"):
            return int(item.page_no)
        return 1

    def _get_safe_bbox(self, item: Any) -> Optional[List[float]]:
        if isinstance(item, dict) and item.get("kind") == "pymupdf_image":
            rect = item.get("rect")
            if rect and len(rect) == 4:
                return [float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])]
            return None

        if hasattr(item, "prov") and item.prov and len(item.prov) > 0:
            if hasattr(item.prov[0], "bbox"):
                b = item.prov[0].bbox
                return [getattr(b, "l", 0), getattr(b, "t", 0), getattr(b, "r", 0), getattr(b, "b", 0)]

        if hasattr(item, "bbox") and item.bbox:
            b = item.bbox
            return [getattr(b, "l", 0), getattr(b, "t", 0), getattr(b, "r", 0), getattr(b, "b", 0)]

        return None
