# index/assets_indexer.py
from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from config.constants import (
    DEFAULT_ASSET_COLLECTION,
    DEFAULT_QDRANT_URL,
    INDEXER_DEFAULT_BATCH_SIZE,
    QDRANT_TIMEOUT_SECONDS,
)

logger = logging.getLogger("GridVision_AssetIndexer")


@dataclass
class AssetIndexerConfig:
    collection: str
    qdrant_url: str
    qdrant_api_key: Optional[str]
    batch_size: int
    assets_manifest: Path
    chunks_file: Path
    docs_file: Path
    recreate: bool = False


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def load_docs_map(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            doc_id = str(rec.get("doc_id", ""))
            if not doc_id:
                continue
            out[doc_id] = {
                "doc_slug": rec.get("doc_slug"),
                "title": rec.get("title"),
                "source_uri": rec.get("source_uri"),
            }
    return out


def load_caption_map(chunks_path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not chunks_path.exists():
        return out
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("type") != "image_caption":
                continue
            element_id = str(rec.get("element_id", ""))
            if not element_id or element_id in out:
                continue
            out[element_id] = {
                "caption": rec.get("content"),
                "caption_chunk_id": rec.get("chunk_id"),
            }
    return out


def ensure_collection(client: QdrantClient, collection: str, dim: int, recreate: bool) -> None:
    if recreate and client.collection_exists(collection):
        client.delete_collection(collection)
    if client.collection_exists(collection):
        return
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


def index_assets(cfg: AssetIndexerConfig) -> None:
    if not cfg.assets_manifest.exists():
        raise FileNotFoundError(f"assets_manifest.json not found: {cfg.assets_manifest}")

    assets = json.loads(cfg.assets_manifest.read_text(encoding="utf-8"))
    if not isinstance(assets, list):
        raise ValueError("assets_manifest.json must be a JSON array")

    caption_map = load_caption_map(cfg.chunks_file)
    docs_map = load_docs_map(cfg.docs_file)

    # determine vector dim
    dim = None
    for item in assets:
        emb = item.get("embedding")
        if emb:
            dim = len(emb)
            break
    if dim is None:
        raise RuntimeError("No embeddings found in assets_manifest.json. Run --embed-assets first.")

    client = QdrantClient(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key,
        timeout=QDRANT_TIMEOUT_SECONDS,
    )
    ensure_collection(client, cfg.collection, dim=dim, recreate=cfg.recreate)

    batch: List[PointStruct] = []
    total = 0
    for item in assets:
        emb = item.get("embedding")
        if not emb:
            continue
        asset_id = str(item.get("asset_id", ""))
        if not asset_id:
            continue

        element_id = item.get("element_id")
        doc_id = item.get("doc_id")
        doc_meta = docs_map.get(str(doc_id), {})
        cap_meta = caption_map.get(str(element_id), {})

        payload = {
            "asset_id": asset_id,
            "element_id": element_id,
            "doc_id": doc_id,
            "doc_slug": doc_meta.get("doc_slug"),
            "title": doc_meta.get("title"),
            "source_uri": doc_meta.get("source_uri"),
            "page_label": item.get("page_label"),
            "file_path": item.get("file_path"),
            "type": item.get("type"),
            "caption": cap_meta.get("caption"),
            "caption_chunk_id": cap_meta.get("caption_chunk_id"),
            "embedding_model": item.get("embedding_model"),
        }

        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"gridvision:asset:{asset_id}"))
        batch.append(
            PointStruct(
                id=point_id,
                vector=[float(v) for v in emb],
                payload=payload,
            )
        )
        if len(batch) >= cfg.batch_size:
            client.upsert(collection_name=cfg.collection, points=batch, wait=True)
            total += len(batch)
            batch = []

    if batch:
        client.upsert(collection_name=cfg.collection, points=batch, wait=True)
        total += len(batch)

    logger.info("Asset indexing done. Upserted points: %d", total)


def parse_args() -> AssetIndexerConfig:
    p = argparse.ArgumentParser(description="Index assets_manifest.json into Qdrant for image retrieval.")
    p.add_argument("--assets-manifest", default="data/processed/artifacts/assets_manifest.json")
    p.add_argument("--chunks", default="data/processed/artifacts/chunks.jsonl")
    p.add_argument("--docs", default="data/processed/artifacts/documents.jsonl")
    p.add_argument("--collection", default=os.getenv("GV_ASSET_COLLECTION", DEFAULT_ASSET_COLLECTION))
    p.add_argument("--url", default=os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL))
    p.add_argument("--api-key", default=os.getenv("QDRANT_API_KEY"))
    p.add_argument("--batch", type=int, default=INDEXER_DEFAULT_BATCH_SIZE)
    p.add_argument("--recreate", action="store_true", help="Delete and recreate collection.")
    args = p.parse_args()

    return AssetIndexerConfig(
        collection=args.collection,
        qdrant_url=args.url,
        qdrant_api_key=args.api_key,
        batch_size=args.batch,
        assets_manifest=Path(args.assets_manifest),
        chunks_file=Path(args.chunks),
        docs_file=Path(args.docs),
        recreate=args.recreate,
    )


def main() -> None:
    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    cfg = parse_args()
    index_assets(cfg)


if __name__ == "__main__":
    main()
