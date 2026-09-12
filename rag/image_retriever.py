from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Optional

from PIL import Image
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


class ImageRetriever:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        model: Any,
        top_k: int = 8,
        min_score: Optional[float] = None,
    ):
        self.client = client
        self.collection_name = collection_name
        self.model = model
        self.top_k = top_k
        self.min_score = min_score

    def _filter_hits(self, hits: List[Any]) -> List[Any]:
        if self.min_score is None:
            return hits
        filtered = []
        for hit in hits:
            score = getattr(hit, "score", None)
            if score is None:
                continue
            if float(score) >= self.min_score:
                filtered.append(hit)
        return filtered

    def _encode_text(self, text: str) -> Optional[List[float]]:
        if not text or not self.model:
            return None
        vec = self.model.encode([text], normalize_embeddings=True)
        if hasattr(vec[0], "tolist"):
            return vec[0].tolist()
        return list(vec[0])

    def _encode_image_bytes(self, image_bytes: bytes) -> Optional[List[float]]:
        if not image_bytes or not self.model:
            return None
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            logger.warning("Failed to load image bytes: %s", exc)
            return None
        vec = self.model.encode([img], normalize_embeddings=True)
        if hasattr(vec[0], "tolist"):
            return vec[0].tolist()
        return list(vec[0])

    def search_by_text(self, text: str) -> List[Dict[str, Any]]:
        vec = self._encode_text(text)
        if not vec:
            return []
        try:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=vec,
                limit=self.top_k,
                with_payload=True,
            )
        except Exception as exc:
            logger.warning("Image search (text) failed: %s", exc)
            return []
        hits = self._filter_hits(hits)
        return [{"score": h.score, "payload": h.payload} for h in hits]

    def search_by_image(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        vec = self._encode_image_bytes(image_bytes)
        if not vec:
            return []
        try:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=vec,
                limit=self.top_k,
                with_payload=True,
            )
        except Exception as exc:
            logger.warning("Image search (image) failed: %s", exc)
            return []
        hits = self._filter_hits(hits)
        return [{"score": h.score, "payload": h.payload} for h in hits]
