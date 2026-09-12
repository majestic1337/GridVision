#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from config import constants as const
from config.runtime import get_runtime_config
from rag.retriever import HybridQdrantRetriever


# ----------------------------
# IO helpers
# ----------------------------
def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_queries(path: Path) -> List[Dict[str, Any]]:
    """
    Supports:
      - JSONL: one query object per line
      - JSON: list[...] or {"queries":[...]} or {"items":[...]}
      - golden_redesign.json style: {"queries":[...]} or list[...]
    """
    if path.suffix.lower() == ".jsonl":
        items: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                items.append(json.loads(line))
        return items

    data = _read_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("queries", "items"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unsupported queries format: {path}")


def safe_set(values: Optional[Iterable[str]]) -> Set[str]:
    if not values:
        return set()
    return {str(v) for v in values if v}


def safe_pages(values: Optional[Iterable[str]]) -> Set[str]:
    if not values:
        return set()
    out = set()
    for v in values:
        if not v:
            continue
        out.add(str(v).strip())
    return out


# ----------------------------
# Models / retriever
# ----------------------------
def _get_model_dense(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device="cpu")


def _get_model_sparse(model_name: str):
    from fastembed import SparseTextEmbedding

    try:
        return SparseTextEmbedding(model_name=model_name)
    except Exception as exc:
        raise RuntimeError(
            "FastEmbed sparse model failed to load. Ensure the model is downloaded "
            "and the cache is writable. You can prefetch by running:\n"
            "  FASTEMBED_CACHE_PATH=/home/majestic/.cache/fastembed \\\n"
            "  python - <<'PY'\n"
            "from fastembed import SparseTextEmbedding\n"
            "SparseTextEmbedding(model_name='Qdrant/Splade_PP_en_v1')\n"
            "print('fastembed ready')\n"
            "PY"
        ) from exc


def build_retriever(top_k: int) -> HybridQdrantRetriever:
    runtime = get_runtime_config()
    return HybridQdrantRetriever(
        client=QdrantClient(
            url=os.getenv("QDRANT_URL", const.DEFAULT_QDRANT_URL),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=const.QDRANT_TIMEOUT_SECONDS,
        ),
        collection_name=os.getenv(const.ENV_COLLECTION, runtime.collection),
        dense_model=_get_model_dense(os.getenv("GV_DENSE_MODEL", const.DEFAULT_DENSE_MODEL)),
        sparse_model=_get_model_sparse(os.getenv("GV_SPARSE_MODEL", const.DEFAULT_SPARSE_MODEL)),
        top_k=top_k,
        dense_vector_name=os.getenv(const.ENV_DENSE_VECTOR, runtime.dense_vector),
        sparse_vector_name=os.getenv(const.ENV_SPARSE_VECTOR, runtime.sparse_vector),
    )


def verify_qdrant(retriever: HybridQdrantRetriever) -> None:
    try:
        ok = retriever.client.collection_exists(retriever.collection_name)
    except Exception as exc:
        raise RuntimeError("Qdrant is not reachable. Start Qdrant or set QDRANT_URL.") from exc
    if not ok:
        raise RuntimeError(
            f"Qdrant collection not found: {retriever.collection_name}. "
            "Check QDRANT_COLLECTION or ingest/index first."
        )


# ----------------------------
# Mapping builders
# ----------------------------
def load_element_mappings(element_to_chunks_path: Optional[Path]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    element_to_chunks: element_key -> [chunk_id,...]
    Also builds chunk_to_element for relaxed matching.
    """
    if not element_to_chunks_path or not element_to_chunks_path.exists():
        return {}, {}

    element_to_chunks = _read_json(element_to_chunks_path)
    if not isinstance(element_to_chunks, dict):
        raise ValueError("element_to_chunks.json must be a dict[element_key -> list[chunk_id]]")

    chunk_to_element: Dict[str, str] = {}
    norm: Dict[str, List[str]] = {}
    for el, chunks in element_to_chunks.items():
        if not isinstance(chunks, list):
            continue
        norm_chunks = [str(c) for c in chunks if c]
        norm[str(el)] = norm_chunks
        for c in norm_chunks:
            # first wins (should be unique anyway)
            chunk_to_element.setdefault(str(c), str(el))

    return norm, chunk_to_element


def load_page_to_assets(page_to_assets_path: Optional[Path]) -> Dict[str, List[str]]:
    """
    page_to_assets: "doc_id_pageLabel" -> [asset_id,...]
    Example key: "d5aabf..._245"
    """
    if not page_to_assets_path or not page_to_assets_path.exists():
        return {}

    page_to_assets = _read_json(page_to_assets_path)
    if not isinstance(page_to_assets, dict):
        raise ValueError("page_to_assets.json must be a dict[doc_id_page -> list[asset_id]]")

    out: Dict[str, List[str]] = {}
    for k, v in page_to_assets.items():
        if not isinstance(v, list):
            continue
        out[str(k)] = [str(x) for x in v if x]
    return out


def build_slug_to_doc_id_map(golden_redesign_path: Optional[Path]) -> Dict[str, str]:
    """
    From golden_redesign.json: uses doc_slug and doc_id found in queries/gold_citations.
    """
    if not golden_redesign_path or not golden_redesign_path.exists():
        return {}

    data = _read_json(golden_redesign_path)
    items: List[Dict[str, Any]]
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and isinstance(data.get("queries"), list):
        items = data["queries"]
    else:
        # try to find any list in dict
        items = []
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                items = v
                break

    slug_to_doc: Dict[str, str] = {}
    for q in items:
        # direct fields sometimes exist
        for cit in (q.get("gold_citations") or []):
            doc_id = cit.get("doc_id")
            doc_slug = cit.get("doc_slug")
            if doc_id and doc_slug:
                slug_to_doc.setdefault(str(doc_slug), str(doc_id))

        # sometimes expected_retrieval has nothing; but doc_id/doc_slug might exist at top-level
        if q.get("doc_id") and q.get("doc_slug"):
            slug_to_doc.setdefault(str(q["doc_slug"]), str(q["doc_id"]))

    return slug_to_doc


# ----------------------------
# Evaluation logic
# ----------------------------
def _to_page_key(doc_id: Optional[str], page_label: Optional[Any]) -> Optional[str]:
    if not doc_id or page_label is None:
        return None
    return f"{str(doc_id)}_{str(page_label)}"


def _parse_expected_pages_to_docid_keys(
    expected_pages: Set[str],
    slug_to_doc_id: Dict[str, str],
) -> Tuple[Set[str], List[Dict[str, str]]]:
    """
    Converts {"TM-9-254#p7"} -> {"d5aabf..._7"} using slug_to_doc_id.
    Returns (docid_page_keys, warnings)
    """
    out: Set[str] = set()
    warnings: List[Dict[str, str]] = []
    for p in expected_pages:
        raw = str(p)
        if "#p" not in raw:
            warnings.append({"issue": "expected_page_missing_#p", "value": raw})
            continue
        slug, page = raw.split("#p", 1)
        doc_id = slug_to_doc_id.get(slug)
        if not doc_id:
            warnings.append({"issue": "unknown_doc_slug_in_expected_pages", "value": slug})
            continue
        key = _to_page_key(doc_id, page)
        if key:
            out.add(key)
    return out, warnings


def compute_metrics(
    queries: List[Dict[str, Any]],
    retriever: HybridQdrantRetriever,
    chunk_to_element: Dict[str, str],
    page_to_assets: Dict[str, List[str]],
    slug_to_doc_id: Dict[str, str],
    k_values: List[int],
) -> Dict[str, Any]:
    max_k = max(k_values)
    summary: Dict[str, Any] = {"per_query": [], "by_k": {}}
    warnings: List[Dict[str, Any]] = []

    totals = {
        k: {
            # element-relaxed text metrics
            "recall_sum": 0.0,
            "precision_sum": 0.0,
            "mrr_sum": 0.0,
            "text_count": 0,
            # strict chunk metrics
            "strict_recall_sum": 0.0,
            "strict_precision_sum": 0.0,
            "strict_mrr_sum": 0.0,
            "strict_text_count": 0,
            # pages (doc_id_page key)
            "page_recall_sum": 0.0,
            "page_precision_sum": 0.0,
            "page_mrr_sum": 0.0,
            "page_count": 0,
            # assets hit (doc-linked via page_to_assets)
            "asset_hit_sum": 0.0,
            "asset_count": 0,
        }
        for k in k_values
    }

    for entry in queries:
        query = entry.get("query", "")
        qid = entry.get("id", "")

        expected = entry.get("expected_retrieval", {}) or {}
        expected_pages_raw = safe_pages(entry.get("expected_pages"))

        expected_text_chunks = safe_set(expected.get("text_chunk_ids"))
        expected_assets = safe_set(expected.get("asset_ids")) | safe_set(expected.get("image_ids")) | safe_set(expected.get("table_ids"))

        # Convert expected text chunks -> expected elements (relaxed)
        expected_elements = {chunk_to_element.get(cid, cid) for cid in expected_text_chunks}

        # Convert expected pages (slug#pX) -> doc_id_page keys
        expected_page_keys, page_map_w = _parse_expected_pages_to_docid_keys(expected_pages_raw, slug_to_doc_id)
        for w in page_map_w:
            warnings.append({"id": qid, **w})

        docs = retriever.invoke(query)

        retrieved_chunk_ids: List[str] = [
            str(doc.metadata.get("chunk_id"))
            for doc in docs
            if doc.metadata and doc.metadata.get("chunk_id") is not None
        ]
        retrieved_elements: List[str] = [chunk_to_element.get(cid, cid) for cid in retrieved_chunk_ids]

        # Build retrieved page keys in the SAME format as page_to_assets expects: doc_id_pageLabel
        retrieved_page_keys: List[str] = []
        for doc in docs:
            meta = doc.metadata or {}
            doc_id = meta.get("doc_id") or meta.get("doc_hash")
            page_label = meta.get("page_label") or meta.get("page_number")
            if doc_id is None:
                # fallback: try doc_slug -> doc_id
                doc_slug = meta.get("doc_slug") or meta.get("title")
                if doc_slug:
                    doc_id = slug_to_doc_id.get(str(doc_slug))
            key = _to_page_key(doc_id, page_label)
            if key:
                retrieved_page_keys.append(key)

        # Helper: assets retrieved "up to k" via pages in top-k results
        def assets_up_to_k(k: int) -> Set[str]:
            keys = retrieved_page_keys[:k]
            out: Set[str] = set()
            for pk in keys:
                for aid in page_to_assets.get(pk, []):
                    out.add(aid)
            return out

        per_query: Dict[str, Any] = {
            "id": qid,
            "query": query,
            "expected_text_chunks": sorted(expected_text_chunks),
            "expected_text_elements": sorted(expected_elements),
            "expected_pages": sorted(expected_pages_raw),
            "expected_page_keys": sorted(expected_page_keys),
            "expected_assets": sorted(expected_assets),
            "retrieved_top_k_chunks": retrieved_chunk_ids[:max_k],
            "retrieved_top_k_elements": retrieved_elements[:max_k],
            "retrieved_top_k_page_keys": retrieved_page_keys[:max_k],
        }

        for k in k_values:
            top_chunks = retrieved_chunk_ids[:k]
            top_chunk_set = set(top_chunks)

            top_elements = retrieved_elements[:k]
            top_element_set = set(top_elements)

            # ---- Strict chunk metrics
            if expected_text_chunks:
                hit = len(expected_text_chunks & top_chunk_set)
                strict_recall = hit / max(len(expected_text_chunks), 1)
                strict_precision = hit / max(len(top_chunks), 1)
                rank = 0
                for idx, cid in enumerate(top_chunks, 1):
                    if cid in expected_text_chunks:
                        rank = idx
                        break
                strict_mrr = 1.0 / rank if rank else 0.0

                totals[k]["strict_recall_sum"] += strict_recall
                totals[k]["strict_precision_sum"] += strict_precision
                totals[k]["strict_mrr_sum"] += strict_mrr
                totals[k]["strict_text_count"] += 1
            else:
                strict_recall = strict_precision = strict_mrr = None

            # ---- Relaxed element metrics
            if expected_elements:
                hit = len(expected_elements & top_element_set)
                recall = hit / max(len(expected_elements), 1)
                precision = hit / max(len(top_elements), 1)
                rank = 0
                for idx, el in enumerate(top_elements, 1):
                    if el in expected_elements:
                        rank = idx
                        break
                mrr = 1.0 / rank if rank else 0.0

                totals[k]["recall_sum"] += recall
                totals[k]["precision_sum"] += precision
                totals[k]["mrr_sum"] += mrr
                totals[k]["text_count"] += 1
            else:
                recall = precision = mrr = None

            # ---- Page metrics (doc_id_page keys)
            if expected_page_keys:
                top_pages = retrieved_page_keys[:k]
                top_page_set = set(top_pages)
                hitp = len(expected_page_keys & top_page_set)
                page_recall = hitp / max(len(expected_page_keys), 1)
                page_precision = hitp / max(len(top_pages), 1)
                rank = 0
                for idx, pk in enumerate(top_pages, 1):
                    if pk in expected_page_keys:
                        rank = idx
                        break
                page_mrr = 1.0 / rank if rank else 0.0

                totals[k]["page_recall_sum"] += page_recall
                totals[k]["page_precision_sum"] += page_precision
                totals[k]["page_mrr_sum"] += page_mrr
                totals[k]["page_count"] += 1
            else:
                page_recall = page_precision = page_mrr = None

            # ---- Asset hit via page_to_assets (doc-linked)
            if expected_assets:
                got_assets = assets_up_to_k(k)
                asset_hit = 1.0 if (expected_assets & got_assets) else 0.0
                totals[k]["asset_hit_sum"] += asset_hit
                totals[k]["asset_count"] += 1
            else:
                got_assets = set()
                asset_hit = None

            per_query.setdefault("metrics", {})[f"k_{k}"] = {
                "relaxed_text": {"recall": recall, "precision": precision, "mrr": mrr},
                "strict_text": {"recall": strict_recall, "precision": strict_precision, "mrr": strict_mrr},
                "pages": {"recall": page_recall, "precision": page_precision, "mrr": page_mrr},
                "assets": {"hit": asset_hit, "retrieved_asset_ids": sorted(got_assets) if expected_assets else []},
            }

        summary["per_query"].append(per_query)

    # aggregate
    for k in k_values:
        tc = totals[k]["text_count"]
        stc = totals[k]["strict_text_count"]
        pc = totals[k]["page_count"]
        ac = totals[k]["asset_count"]

        summary["by_k"][str(k)] = {
            "relaxed_text": {
                "recall": totals[k]["recall_sum"] / tc if tc else None,
                "precision": totals[k]["precision_sum"] / tc if tc else None,
                "mrr": totals[k]["mrr_sum"] / tc if tc else None,
                "queries": tc,
            },
            "strict_text": {
                "recall": totals[k]["strict_recall_sum"] / stc if stc else None,
                "precision": totals[k]["strict_precision_sum"] / stc if stc else None,
                "mrr": totals[k]["strict_mrr_sum"] / stc if stc else None,
                "queries": stc,
            },
            "pages": {
                "recall": totals[k]["page_recall_sum"] / pc if pc else None,
                "precision": totals[k]["page_precision_sum"] / pc if pc else None,
                "mrr": totals[k]["page_mrr_sum"] / pc if pc else None,
                "queries": pc,
            },
            "assets": {
                "hit_rate": totals[k]["asset_hit_sum"] / ac if ac else None,
                "queries": ac,
            },
        }

    if warnings:
        summary["warnings"] = warnings

    return summary


# ----------------------------
# CLI
# ----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation over a gold dataset (relaxed by element + page-linked assets).")

    parser.add_argument("--queries", default="/mnt/data/eval_set_30.jsonl", help="Path to queries JSON/JSONL.")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10], help="k values to evaluate.")
    parser.add_argument("--out", default="eval/retrieval_metrics.json", help="Output JSON path.")

    parser.add_argument("--golden-redesign", default="/mnt/data/golden_redesign.json", help="Path to golden_redesign.json (for doc_slug->doc_id mapping).")
    parser.add_argument("--element-to-chunks", default="/mnt/data/element_to_chunks.json", help="Path to element_to_chunks.json (for relaxed element matching).")
    parser.add_argument("--page-to-assets", default="/mnt/data/page_to_assets.json", help="Path to page_to_assets.json (for asset evaluation).")

    args = parser.parse_args()

    query_path = Path(args.queries)
    if not query_path.is_absolute():
        query_path = ROOT / query_path
    if not query_path.exists():
        raise FileNotFoundError(f"Queries file not found: {query_path}")

    k_values = sorted({int(k) for k in args.k if int(k) > 0})
    queries = load_queries(query_path)

    golden_redesign_path = Path(args.golden_redesign)
    if not golden_redesign_path.is_absolute():
        golden_redesign_path = ROOT / golden_redesign_path

    element_to_chunks_path = Path(args.element_to_chunks)
    if not element_to_chunks_path.is_absolute():
        element_to_chunks_path = ROOT / element_to_chunks_path

    page_to_assets_path = Path(args.page_to_assets)
    if not page_to_assets_path.is_absolute():
        page_to_assets_path = ROOT / page_to_assets_path

    # mappings
    _, chunk_to_element = load_element_mappings(element_to_chunks_path if element_to_chunks_path.exists() else None)
    page_to_assets = load_page_to_assets(page_to_assets_path if page_to_assets_path.exists() else None)
    slug_to_doc_id = build_slug_to_doc_id_map(golden_redesign_path if golden_redesign_path.exists() else None)

    # retriever
    retriever = build_retriever(top_k=max(k_values))
    verify_qdrant(retriever)

    summary = compute_metrics(
        queries=queries,
        retriever=retriever,
        chunk_to_element=chunk_to_element,
        page_to_assets=page_to_assets,
        slug_to_doc_id=slug_to_doc_id,
        k_values=k_values,
    )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
