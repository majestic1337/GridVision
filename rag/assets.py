import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from .schemas import Asset


class AssetResolver:
    def __init__(self, manifest_path: Path):
        self.by_asset_id: Dict[str, Dict[str, Any]] = {}
        self.by_element_id: Dict[str, Dict[str, Any]] = {}
        self.by_doc_page: Dict[str, List[Dict[str, Any]]] = {}

        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                asset_id = item.get("asset_id")
                element_id = item.get("element_id")
                doc_id = item.get("doc_id")
                page_label = item.get("page_label")

                if asset_id:
                    self.by_asset_id[str(asset_id)] = item
                if element_id:
                    self.by_element_id[str(element_id)] = item
                if doc_id and page_label is not None:
                    key = f"{doc_id}_{page_label}"
                    self.by_doc_page.setdefault(key, []).append(item)

    def _get_meta(self, doc: Document, key: str) -> Optional[Any]:
        # supports both flattened and nested payload styles
        if key in doc.metadata:
            return doc.metadata.get(key)
        nested = doc.metadata.get("metadata") or {}
        if isinstance(nested, dict):
            return nested.get(key)
        return None

    def resolve(self, docs: List[Document]) -> List[Asset]:
        assets: List[Asset] = []
        seen_asset_ids = set()

        for doc in docs:
            element_id = self._get_meta(doc, "element_id")
            linked_asset_id = self._get_meta(doc, "linked_asset_id")
            doc_id = self._get_meta(doc, "doc_id")
            page_label = self._get_meta(doc, "page_label") or self._get_meta(doc, "page_number")

            item = None

            # 1) linked_asset_id from ingestion is asset_id
            if linked_asset_id:
                item = self.by_asset_id.get(str(linked_asset_id))

            # 2) fallback: resolve by element_id
            if item is None and element_id:
                item = self.by_element_id.get(str(element_id))

            # 3) doc-linked fallback (same doc + page)
            if item is None and doc_id and page_label is not None:
                key = f"{doc_id}_{page_label}"
                for candidate in self.by_doc_page.get(key, []):
                    asset_id = str(candidate.get("asset_id"))
                    if not asset_id or asset_id in seen_asset_ids:
                        continue

                    assets.append(
                        Asset(
                            asset_id=asset_id,
                            type=str(candidate.get("type", "image_asset")),
                            file_path=str(candidate.get("file_path", "")),
                            page_label=str(candidate.get("page_label", page_label)),
                            element_id=str(candidate.get("element_id", element_id or "")),
                        )
                    )
                    seen_asset_ids.add(asset_id)
                continue

            if not item:
                continue

            asset_id = str(item.get("asset_id"))
            if not asset_id or asset_id in seen_asset_ids:
                continue

            item_page_label = item.get("page_label")
            if item_page_label is None:
                # fallback if manifest stores page_number
                item_page_label = item.get("page_number", "?")

            assets.append(
                Asset(
                    asset_id=asset_id,
                    type=str(item.get("type", "image_asset")),
                    file_path=str(item.get("file_path", "")),
                    page_label=str(item_page_label),
                    element_id=str(item.get("element_id", element_id or "")),
                )
            )
            seen_asset_ids.add(asset_id)

        return assets
