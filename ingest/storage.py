import json
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from config.constants import ASSET_ID_HASH_LEN, DEFAULT_ASSET_ZOOM

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except Exception:
    PYMUPDF_AVAILABLE = False

if TYPE_CHECKING:
    from .parsing import ParsedElement


class ArtifactStore:
    def __init__(self, root_dir: Path, artifact_dir: Path, logger: logging.Logger):
        self.root_dir = root_dir
        self.artifact_dir = artifact_dir
        self.logger = logger

    def initialize(self, reset: bool = False) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        chunks_path = self.artifact_dir / "chunks.jsonl"
        assets_path = self.artifact_dir / "assets_manifest.json"
        elements_path = self.artifact_dir / "element_to_chunks.json"
        pages_path = self.artifact_dir / "page_to_assets.json"
        docs_path = self.artifact_dir / "documents.jsonl"

        if reset:
            chunks_path.write_text("", encoding="utf-8")
            assets_path.write_text("[]", encoding="utf-8")
            elements_path.write_text("{}", encoding="utf-8")
            pages_path.write_text("{}", encoding="utf-8")
            docs_path.write_text("", encoding="utf-8")
            return

        self._ensure_file(chunks_path, "")
        self._ensure_file(assets_path, "[]")
        self._ensure_file(elements_path, "{}")
        self._ensure_file(pages_path, "{}")
        self._ensure_file(docs_path, "")

    def _ensure_file(self, path: Path, default_text: str) -> None:
        if not path.exists():
            path.write_text(default_text, encoding="utf-8")

    def load_ingested_doc_ids(self) -> Set[str]:
        doc_ids: Set[str] = set()
        docs_path = self.artifact_dir / "documents.jsonl"
        if docs_path.exists() and docs_path.stat().st_size > 0:
            with open(docs_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        self.logger.warning("Skipping malformed documents.jsonl line")
                        continue
                    doc_id = record.get("doc_id")
                    if doc_id:
                        doc_ids.add(str(doc_id))
            if doc_ids:
                return doc_ids

        chunk_path = self.artifact_dir / "chunks.jsonl"
        doc_records: Dict[str, Dict[str, Any]] = {}
        if chunk_path.exists() and chunk_path.stat().st_size > 0:
            with open(chunk_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except Exception:
                        self.logger.warning("Skipping malformed chunks.jsonl line")
                        continue
                    doc_id = chunk.get("doc_id")
                    if not doc_id:
                        continue
                    doc_id = str(doc_id)
                    if doc_id not in doc_records:
                        doc_records[doc_id] = {
                            "doc_id": doc_id,
                            "doc_slug": chunk.get("doc_slug"),
                            "title": chunk.get("title"),
                            "source_uri": chunk.get("source_uri"),
                            "created_at": chunk.get("created_at"),
                            "tags": chunk.get("tags"),
                        }
                    doc_ids.add(doc_id)

        if doc_records and (not docs_path.exists() or docs_path.stat().st_size == 0):
            with open(docs_path, "a", encoding="utf-8") as f:
                for record in doc_records.values():
                    f.write(json.dumps(record) + "\n")

        return doc_ids

    def append_document_record(self, record: Dict[str, Any]) -> None:
        path = self.artifact_dir / "documents.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def write_chunks(self, data: List[Dict[str, Any]]) -> None:
        if not data:
            return
        path = self.artifact_dir / "chunks.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    def write_assets_manifest(self, assets: List[Dict[str, Any]]) -> None:
        if not assets:
            return
        current_manifest_path = self.artifact_dir / "assets_manifest.json"
        existing_manifest: List[Dict[str, Any]] = []
        if current_manifest_path.exists():
            try:
                with open(current_manifest_path, "r", encoding="utf-8") as f:
                    existing_manifest = json.load(f)
            except Exception:
                existing_manifest = []

        full_manifest = existing_manifest + assets
        self._save_json_list(full_manifest, current_manifest_path)

    def write_maps(
        self,
        element_to_chunks: Dict[str, List[str]],
        page_to_assets: Dict[str, List[str]],
    ) -> None:
        self._save_json_merge(element_to_chunks, self.artifact_dir / "element_to_chunks.json")
        self._save_json_merge(page_to_assets, self.artifact_dir / "page_to_assets.json")

    def _save_json_list(self, data: List[Dict[str, Any]], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _save_json_merge(self, data: Dict[str, Any], path: Path) -> None:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if isinstance(existing, dict):
                    existing.update(data)
                    data = existing
            except Exception:
                pass

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


class AssetManager:
    def __init__(self, root_dir: Path, asset_root: Path, logger: logging.Logger):
        self.root_dir = root_dir
        self.asset_root = asset_root
        self.logger = logger
        self.asset_dir: Optional[Path] = None
        self.doc_hash: Optional[str] = None
        self.pdf_doc = None
        self.page_total: Optional[int] = None
        self.page_to_assets: Dict[str, List[str]] = {}
        self._doc_assets: List[Dict[str, Any]] = []

    def start_document(self, doc_hash: str) -> None:
        self.doc_hash = doc_hash
        self.asset_dir = self.asset_root / doc_hash
        self.asset_dir.mkdir(parents=True, exist_ok=True)
        self._doc_assets = []

    def current_assets(self) -> List[Dict[str, Any]]:
        return list(self._doc_assets)

    def open_pdf(self, raw_path: Path) -> Optional[int]:
        self.page_total = None
        self.pdf_doc = None

        if PYMUPDF_AVAILABLE:
            try:
                self.pdf_doc = fitz.open(str(raw_path))
                self.page_total = self.pdf_doc.page_count
                self.logger.info(f"[DOC] Page count detected: {self.page_total}")
            except Exception as e:
                self.logger.warning(f"Failed to open PDF with PyMuPDF: {e}")
                self.pdf_doc = None
        else:
            self.logger.warning("PyMuPDF not installed. Image extraction may fail. Install with `pip install pymupdf`.")

        return self.page_total

    def close_pdf(self) -> None:
        if self.pdf_doc is None:
            return
        try:
            self.pdf_doc.close()
        except Exception:
            pass
        self.pdf_doc = None

    def register_element(
        self,
        element: "ParsedElement",
    ) -> Tuple[Optional[str], Optional[Path]]:
        if element.type not in ("image", "table"):
            return None, None
        if self.asset_dir is None or self.doc_hash is None:
            raise RuntimeError("AssetManager.start_document must be called before register_element")

        asset_uuid = f"asset_{hashlib.md5(element.element_id.encode()).hexdigest()[:ASSET_ID_HASH_LEN]}"
        ext = ".png" if element.type == "image" else ".html"
        asset_filename = f"{element.element_id}{ext}"
        asset_path = self.asset_dir / asset_filename

        metadata = element.metadata or {}
        raw_item = metadata.get("raw_item")
        parent_doc = metadata.get("parent_doc")

        saved = self._save_asset(
            raw_item=raw_item,
            dest_path=asset_path,
            asset_type=element.type,
            parent_doc=parent_doc,
        )

        self.logger.info(
            "[ASSET] element=%s type=%s saved=%s path=%s",
            element.element_id,
            element.type,
            saved,
            asset_path,
        )

        if not saved:
            return None, asset_path

        file_path = str(asset_path)
        try:
            file_path = str(asset_path.relative_to(self.root_dir))
        except ValueError:
            pass

        self._doc_assets.append({
            "asset_id": asset_uuid,
            "element_id": element.element_id,
            "doc_id": self.doc_hash,
            "type": f"{element.type}_asset",
            "page_label": element.page_label,
            "file_path": file_path,
            "bbox": element.bbox
        })

        page_key = f"{self.doc_hash}_{element.page_label}"
        if page_key not in self.page_to_assets:
            self.page_to_assets[page_key] = []
        self.page_to_assets[page_key].append(asset_uuid)

        return asset_uuid, asset_path

    def _save_asset(
        self,
        raw_item: Any,
        dest_path: Path,
        asset_type: str,
        parent_doc: Any = None,
        zoom: float = DEFAULT_ASSET_ZOOM,
    ) -> bool:
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            if asset_type == "image":
                if isinstance(raw_item, dict) and raw_item.get("kind") == "pymupdf_image":
                    if not PYMUPDF_AVAILABLE or self.pdf_doc is None:
                        self.logger.warning("PyMuPDF not available or pdf_doc not provided for image extraction")
                        return False

                    page_no = int(raw_item["page_no"])
                    rect = raw_item.get("rect")
                    if not rect or len(rect) != 4:
                        self.logger.warning(f"Bad rect for {dest_path.name}")
                        return False

                    page = self.pdf_doc.load_page(page_no - 1)
                    clip = fitz.Rect(rect[0], rect[1], rect[2], rect[3])

                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
                    pix.save(str(dest_path))
                    return True
                image_obj = None
                if hasattr(raw_item, "get_image") and callable(getattr(raw_item, "get_image")) and parent_doc is not None:
                    try:
                        image_obj = raw_item.get_image(parent_doc)
                    except Exception:
                        image_obj = None

                if image_obj is None and hasattr(raw_item, "image"):
                    try:
                        image_obj = raw_item.image
                    except Exception:
                        image_obj = None

                if image_obj is not None and hasattr(image_obj, "save") and callable(getattr(image_obj, "save")):
                    image_obj.save(dest_path, format="PNG")
                    return True

                self.logger.warning(f"Image object is None/unusable for {dest_path.name}")
                return False

            if asset_type == "table":
                if hasattr(raw_item, "export_to_html") and callable(getattr(raw_item, "export_to_html")):
                    try:
                        html = raw_item.export_to_html(doc=parent_doc)
                    except TypeError:
                        html = raw_item.export_to_html()
                    with open(dest_path, "w", encoding="utf-8") as f:
                        f.write(html)
                    return True

                self.logger.warning(f"Table export_to_html not available for {dest_path.name}")
                return False

            self.logger.warning(f"Unknown asset_type={asset_type} for {dest_path.name}")
            return False

        except Exception as e:
            self.logger.error(f"Failed to save asset {dest_path.name}: {e}")
            return False


class ChunkRegistry:
    def __init__(self):
        self.element_to_chunks: Dict[str, List[str]] = {}

    def record(self, element_id: str, chunk_id: str) -> None:
        if element_id not in self.element_to_chunks:
            self.element_to_chunks[element_id] = []
        self.element_to_chunks[element_id].append(chunk_id)
