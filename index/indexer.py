# index/indexer.py
from __future__ import annotations

import os
import json
import argparse
import logging
import sys
from dataclasses import dataclass,asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import time
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    SparseVector,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from config.constants import (
    DEFAULT_DENSE_MODEL,
    DEFAULT_QDRANT_URL,
    DEFAULT_SPARSE_MODEL,
    INDEXER_DEFAULT_BATCH_SIZE,
    INDEXER_DEFAULT_CHUNK_FILE,
    INDEXER_DEFAULT_SAVE_EVERY,
    INDEXER_DEFAULT_SAVEPOINT_PATH,
    INDEXER_SCROLL_LIMIT,
    INDEXER_TEST_LIMIT,
    INDEXER_TEST_QUERY,
    QDRANT_TIMEOUT_SECONDS,
)
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except Exception:
    ST_AVAILABLE = False

try:
    from fastembed import SparseTextEmbedding
    FASTEMBED_AVAILABLE = True
except Exception:
    FASTEMBED_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config.runtime import get_runtime_config


logger = logging.getLogger("GridVision_Indexer")
_RUNTIME = get_runtime_config()


@dataclass
class IndexerConfig:
    collection: str
    qdrant_url: str
    qdrant_api_key: Optional[str]
    prefer_grpc: bool

    dense_name: str = _RUNTIME.dense_vector
    sparse_name: str = _RUNTIME.sparse_vector

    dense_model: str = DEFAULT_DENSE_MODEL
    sparse_model: str = DEFAULT_SPARSE_MODEL

    batch_size: int = INDEXER_DEFAULT_BATCH_SIZE
    chunk_file: Path = INDEXER_DEFAULT_CHUNK_FILE

    do_prune: bool = True
    do_validate: bool = True
    do_test_search: bool = True

    savepoint_path: Path = INDEXER_DEFAULT_SAVEPOINT_PATH
    resume: bool = True
    reset_savepoint: bool = False
    save_every: int = INDEXER_DEFAULT_SAVE_EVERY

    # Sparse index tuning (ok defaults)
    sparse_on_disk: bool = False
    sparse_full_scan_threshold: Optional[int] = None


class DenseEmbedder:
    def __init__(self, model_name: str):
        if not ST_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers is not installed. Install it or disable dense indexing."
            )
        logger.info("Loading dense model: %s", model_name)
        device = os.getenv("GV_DENSE_DEVICE", "cpu")
        self.model = SentenceTransformer(model_name, device=device)

    def dim(self) -> int:
        # robust way: embed 1 short string
        v = self.encode(["dim_probe"])
        return len(v[0])

    def encode(self, texts: List[str]) -> List[List[float]]:
        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # SentenceTransformer returns np.ndarray; cast to python lists
        return [v.tolist() for v in vecs]


class SparseEmbedderFallback:
    """
    Fallback sparse embedder якщо в chunks.jsonl sparse_vector пустий
    або ти захочеш перераховувати sparse у indexer-і.
    """
    def __init__(self, model_name: str):
        self.model = None
        if FASTEMBED_AVAILABLE:
            logger.info("Loading sparse model (FastEmbed): %s", model_name)
            self.model = SparseTextEmbedding(model_name=model_name)
        else:
            logger.warning("fastembed not installed; sparse fallback disabled")

    def encode_one(self, text: str) -> Dict[str, List[Any]]:
        if not self.model or not text:
            return {"indices": [], "values": []}
        vec = list(self.model.embed([text]))[0]
        return {"indices": vec.indices.tolist(), "values": vec.values.tolist()}

@dataclass
class Savepoint:
    offset_lines: int = 0
    upserted: int = 0
    updated_at_unix: float = 0.0

def load_savepoint(path: Path) -> Savepoint:
    if not path.exists():
        return Savepoint()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Savepoint(
            offset_lines=int(data.get("offset_lines", 0)),
            upserted=int(data.get("upserted", 0)),
            updated_at_unix=float(data.get("updated_at_unix", 0.0)),
        )
    except Exception as e:
        logger.warning("Failed to load savepoint %s: %s. Starting from 0.", path, e)
        return Savepoint()

def save_savepoint(path: Path, sp: Savepoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sp.updated_at_unix = time.time()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(asdict(sp), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

def ensure_collection(
    client: QdrantClient,
    cfg: IndexerConfig,
    dense_dim: int,
) -> None:
    """
    Create collection with named vectors:
      - cfg.dense_name: dense vectors (size = dense_dim)
      - cfg.sparse_name: sparse vectors (inverted index)
    """
    if client.collection_exists(cfg.collection):
        logger.info("Collection exists: %s", cfg.collection)
        return

    logger.info("Creating collection: %s", cfg.collection)

    sparse_index = SparseIndexParams(
        on_disk=cfg.sparse_on_disk,
        full_scan_threshold=cfg.sparse_full_scan_threshold,
    )

    client.create_collection(
        collection_name=cfg.collection,
        vectors_config={
            cfg.dense_name: VectorParams(size=dense_dim, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            cfg.sparse_name: SparseVectorParams(index=sparse_index)
        },
    )
    logger.info("Collection created: %s", cfg.collection)


def iter_jsonl_with_skip(path: Path, skip_lines: int) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for _ in range(skip_lines):
            next(f, None)
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def collect_current_ids(chunk_file: Path) -> Dict[str, Set[str]]:
    """Return the complete desired point-id set for every document in an artifact file."""
    current_ids: Dict[str, Set[str]] = {}
    with open(chunk_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk = json.loads(line)
            doc_id = str(chunk.get("doc_id", ""))
            chunk_id = str(chunk.get("chunk_id", ""))
            if doc_id and chunk_id:
                current_ids.setdefault(doc_id, set()).add(chunk_id)
    return current_ids


def qdrant_doc_filter(doc_id: str) -> Filter:
    return Filter(
        must=[
            FieldCondition(
                key="doc_id",
                match=MatchValue(value=doc_id),
            )
        ]
    )


def get_existing_ids(client: QdrantClient, cfg: IndexerConfig, doc_id: str) -> Set[str]:
    """
    Scroll all points for a doc_id and return their IDs.
    """
    ids: Set[str] = set()
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=cfg.collection,
            scroll_filter=qdrant_doc_filter(doc_id),
            limit=INDEXER_SCROLL_LIMIT,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        for p in points:
            # id may be int/uuid/str; in our pipeline it's str (chunk_id)
            ids.add(str(p.id))
        if offset is None:
            break

    return ids


def delete_ids(client: QdrantClient, cfg: IndexerConfig, ids: Set[str]) -> None:
    if not ids:
        return
    client.delete(
        collection_name=cfg.collection,
        points_selector=list(ids),
        wait=True,
    )


def build_point(
    cfg: IndexerConfig,
    chunk: Dict[str, Any],
    dense_vec: List[float],
    sparse_vec_dict: Dict[str, Any],
) -> PointStruct:
    sv = SparseVector(
        indices=list(map(int, sparse_vec_dict.get("indices", []) or [])),
        values=list(map(float, sparse_vec_dict.get("values", []) or [])),
    )

    payload = {
        "chunk_id": str(chunk["chunk_id"]),       
        "doc_id": chunk.get("doc_id"),
        "doc_slug": chunk.get("doc_slug"),
        "title": chunk.get("title"),
        "source_uri": chunk.get("source_uri"),
        "created_at": chunk.get("created_at"),
        "tags": chunk.get("tags"),
        "chunk_index": chunk.get("chunk_index"),
        "element_id": chunk.get("element_id"),
        "type": chunk.get("type"),
        "content_hash": chunk.get("content_hash"),
        "content": chunk.get("content"),

        # --- FLATTENED metadata ---
        "page_label": (chunk.get("metadata") or {}).get("page_label"),
        "page_number": (chunk.get("metadata") or {}).get("page_number"),
        "section": (chunk.get("metadata") or {}).get("section"),
        "linked_asset_id": (chunk.get("metadata") or {}).get("linked_asset_id"),

        # keep full metadata too (optional, useful for debug)
        "metadata": chunk.get("metadata", {}),
        "provenance": chunk.get("provenance", {}),
    }

    return PointStruct(
        id=str(chunk["chunk_id"]),
        vector={
            cfg.dense_name: dense_vec,
            cfg.sparse_name: sv,
        },
        payload=payload,
    )


def upsert_batch(client: QdrantClient, cfg: IndexerConfig, points: List[PointStruct]) -> None:
    if not points:
        return
    client.upsert(
        collection_name=cfg.collection,
        points=points,
        wait=True,
    )


def index_chunks(cfg: IndexerConfig) -> None:
    if not cfg.chunk_file.exists():
        raise FileNotFoundError(f"chunks.jsonl not found: {cfg.chunk_file}")

    client = QdrantClient(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key,
        prefer_grpc=cfg.prefer_grpc,
        timeout=QDRANT_TIMEOUT_SECONDS,
    )

    dense = DenseEmbedder(cfg.dense_model)
    sparse_fallback = SparseEmbedderFallback(cfg.sparse_model)

    dense_dim = dense.dim()
    ensure_collection(client, cfg, dense_dim=dense_dim)

    # --- Savepoint (resume/reset) ---
    if getattr(cfg, "reset_savepoint", False):
        sp = Savepoint(offset_lines=0, upserted=0)
        save_savepoint(cfg.savepoint_path, sp)
        logger.info("Savepoint reset: %s", cfg.savepoint_path)
    elif getattr(cfg, "resume", True):
        sp = load_savepoint(cfg.savepoint_path)
        logger.info(
            "Resuming from savepoint: offset_lines=%d upserted=%d (%s)",
            sp.offset_lines,
            sp.upserted,
            cfg.savepoint_path,
        )
    else:
        sp = Savepoint()

    # Pruning must see the whole artifact, not only records after a savepoint.
    current_ids = collect_current_ids(cfg.chunk_file) if cfg.do_prune else {}
    expected_total = sp.upserted  # cumulative already-upserted count

    batch_texts: List[str] = []
    batch_chunks: List[Dict[str, Any]] = []

    processed_lines = 0          # processed this run (after skip)
    global_lines = sp.offset_lines  # absolute file offset

    logger.info("Indexing from: %s", cfg.chunk_file)

    for chunk in tqdm(
        iter_jsonl_with_skip(cfg.chunk_file, sp.offset_lines),
        desc="Reading chunks",
        initial=sp.offset_lines,
    ):
        processed_lines += 1
        global_lines += 1
        expected_total += 1

        text = chunk.get("content") or ""
        batch_texts.append(text)
        batch_chunks.append(chunk)

        if len(batch_chunks) >= cfg.batch_size:
            _process_batch(client, cfg, dense, sparse_fallback, batch_chunks, batch_texts)
            batch_texts = []
            batch_chunks = []

        # periodic savepoint
        save_every = int(getattr(cfg, "save_every", INDEXER_DEFAULT_SAVE_EVERY) or 0)
        if save_every > 0 and (processed_lines % save_every == 0):
            sp.offset_lines = global_lines
            sp.upserted = expected_total
            save_savepoint(cfg.savepoint_path, sp)
            logger.info(
                "Saved savepoint: offset_lines=%d upserted=%d",
                sp.offset_lines,
                sp.upserted,
            )

    if batch_chunks:
        _process_batch(client, cfg, dense, sparse_fallback, batch_chunks, batch_texts)

    # final savepoint
    sp.offset_lines = global_lines
    sp.upserted = expected_total
    save_savepoint(cfg.savepoint_path, sp)
    logger.info(
        "Final savepoint saved: offset_lines=%d upserted=%d",
        sp.offset_lines,
        sp.upserted,
    )

    logger.info("Upsert done. Expected points from file (cumulative): %d", expected_total)

    if cfg.do_prune and current_ids:
        logger.info("Prune enabled. Syncing existing points (doc_id-scoped).")
        _prune(client, cfg, current_ids)

    if cfg.do_validate:
        _validate_counts(client, cfg, expected_total)

    if cfg.do_test_search:
        _test_search(client, cfg, dense, sparse_fallback)


def _process_batch(
    client: QdrantClient,
    cfg: IndexerConfig,
    dense: DenseEmbedder,
    sparse_fallback: SparseEmbedderFallback,
    chunks: List[Dict[str, Any]],
    texts: List[str],
) -> None:
    dense_vecs = dense.encode(texts)

    points: List[PointStruct] = []
    for chunk, dvec in zip(chunks, dense_vecs):
        sparse_vec = chunk.get("sparse_vector") or {"indices": [], "values": []}

        # fallback only if empty
        if (not sparse_vec.get("indices")) and (not sparse_vec.get("values")):
            sparse_vec = sparse_fallback.encode_one(chunk.get("content") or "")

        points.append(build_point(cfg, chunk, dvec, sparse_vec))

    upsert_batch(client, cfg, points)


def _prune(client: QdrantClient, cfg: IndexerConfig, current_ids: Dict[str, Set[str]]) -> None:
    removed_total = 0

    for doc_id, ids_now in tqdm(current_ids.items(), desc="Pruning docs"):
        ids_existing = get_existing_ids(client, cfg, doc_id)
        to_delete = ids_existing - ids_now
        if to_delete:
            delete_ids(client, cfg, to_delete)
            removed_total += len(to_delete)

    logger.info("Prune completed. Removed points: %d", removed_total)


def _validate_counts(client: QdrantClient, cfg: IndexerConfig, expected_total: int) -> None:
    res = client.count(collection_name=cfg.collection, exact=True)
    actual = int(res.count)
    if actual != expected_total:
        logger.warning(
            "Validation mismatch: qdrant_count=%d vs chunks_jsonl=%d. "
            "If you index multiple runs into same collection, this is expected unless you prune all doc_ids.",
            actual, expected_total
        )
    else:
        logger.info("Validation OK: qdrant_count == chunks_jsonl == %d", actual)


def _test_search(
    client: QdrantClient,
    cfg: IndexerConfig,
    dense: DenseEmbedder,
    sparse_fallback: SparseEmbedderFallback,
) -> None:
    query = INDEXER_TEST_QUERY

    # Dense
    q_dense = dense.encode([query])[0]
    dense_resp = client.query_points(
        collection_name=cfg.collection,
        query=q_dense,
        using=cfg.dense_name,
        limit=INDEXER_TEST_LIMIT,
        with_payload=True,
    )
    dense_hits = dense_resp.points

    logger.info("Dense test search hits: %d", len(dense_hits))
    if dense_hits:
        logger.info("Dense top1 doc_slug=%s type=%s score=%s",
                    dense_hits[0].payload.get("doc_slug"),
                    dense_hits[0].payload.get("type"),
                    dense_hits[0].score)

    # Sparse
    q_sv = sparse_fallback.encode_one(query)
    q_sparse = SparseVector(indices=q_sv["indices"], values=q_sv["values"])

    sparse_resp = client.query_points(
        collection_name=cfg.collection,
        query=q_sparse,
        using=cfg.sparse_name,
        limit=INDEXER_TEST_LIMIT,
        with_payload=True,
    )
    sparse_hits = sparse_resp.points

    logger.info("Sparse test search hits: %d", len(sparse_hits))
    if sparse_hits:
        logger.info("Sparse top1 doc_slug=%s type=%s score=%s",
                    sparse_hits[0].payload.get("doc_slug"),
                    sparse_hits[0].payload.get("type"),
                    sparse_hits[0].score)


def parse_args() -> IndexerConfig:
    runtime_cfg = get_runtime_config()
    p = argparse.ArgumentParser(description="Index chunks.jsonl into Qdrant (dense + sparse named vectors).")
    p.add_argument("--chunks", default=os.getenv("GV_CHUNKS", str(INDEXER_DEFAULT_CHUNK_FILE)))
    p.add_argument("--collection", default=runtime_cfg.collection)
    p.add_argument("--url", default=os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL))
    p.add_argument("--api-key", default=os.getenv("QDRANT_API_KEY"))
    p.add_argument("--grpc", action="store_true", default=(os.getenv("QDRANT_PREFER_GRPC", "0") == "1"))
    p.add_argument("--dense-vector-name", default=runtime_cfg.dense_vector)
    p.add_argument("--sparse-vector-name", default=runtime_cfg.sparse_vector)

    p.add_argument("--savepoint", default=os.getenv("GV_INDEX_SAVEPOINT", str(INDEXER_DEFAULT_SAVEPOINT_PATH)))
    p.add_argument("--resume", action="store_true", default=(os.getenv("GV_INDEX_RESUME", "1") == "1"))
    p.add_argument("--reset-savepoint", action="store_true", default=False)
    p.add_argument("--save-every", type=int, default=int(os.getenv("GV_INDEX_SAVE_EVERY", str(INDEXER_DEFAULT_SAVE_EVERY))))

    p.add_argument("--batch", type=int, default=int(os.getenv("GV_INDEX_BATCH", str(INDEXER_DEFAULT_BATCH_SIZE))))
    p.add_argument("--no-prune", action="store_true")
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("--no-test", action="store_true")

    p.add_argument("--dense-model", default=os.getenv("GV_DENSE_MODEL", DEFAULT_DENSE_MODEL))
    p.add_argument("--sparse-model", default=os.getenv("GV_SPARSE_MODEL", DEFAULT_SPARSE_MODEL))

    args = p.parse_args()

    return IndexerConfig(
        collection=args.collection,
        qdrant_url=args.url,
        qdrant_api_key=args.api_key,
        prefer_grpc=args.grpc,
        dense_name=args.dense_vector_name,
        sparse_name=args.sparse_vector_name,
        batch_size=args.batch,
        chunk_file=Path(args.chunks),
        do_prune=(not args.no_prune),
        do_validate=(not args.no_validate),
        do_test_search=(not args.no_test),
        dense_model=args.dense_model,
        sparse_model=args.sparse_model,
        savepoint_path=Path(args.savepoint),
        resume=args.resume,
        reset_savepoint=args.reset_savepoint,
        save_every=args.save_every
    )


def main() -> None:
    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    cfg = parse_args()
    index_chunks(cfg)


if __name__ == "__main__":
    main()
