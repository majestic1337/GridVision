import importlib.util
import json
import os
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "eval" / "golden_redesign.json"
CHUNKS_PATH = ROOT / "data" / "processed" / "artifacts" / "chunks.jsonl"
ASSETS_MANIFEST_PATH = ROOT / "data" / "processed" / "artifacts" / "assets_manifest.json"


@pytest.fixture(scope="session")
def golden_redesign():
    if not GOLDEN_PATH.exists():
        pytest.skip(f"Missing golden dataset: {GOLDEN_PATH}")
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, list), "golden_redesign.json must be a list of query objects"
    return data


def _iter_expected_chunks(entries):
    for entry in entries:
        expected = entry.get("expected_retrieval", {}) or {}
        for cid in expected.get("text_chunk_ids", []) or []:
            if cid:
                yield str(cid)


def _iter_expected_assets(entries):
    for entry in entries:
        expected = entry.get("expected_retrieval", {}) or {}
        for key in ("image_ids", "table_ids", "asset_ids"):
            for aid in expected.get(key, []) or []:
                if aid:
                    yield str(aid)


def test_eval_set_size_and_schema(golden_redesign):
    assert 20 <= len(golden_redesign) <= 50, "Evaluation set must contain 20–50 queries"

    required_keys = {
        "id",
        "query",
        "intent",
        "requires_image",
        "difficulty",
        "tags",
        "expected_pages",
        "expected_retrieval",
        "gold_answer",
        "gold_citations",
    }

    for entry in golden_redesign:
        missing = required_keys - set(entry.keys())
        assert not missing, f"Missing keys in entry {entry.get('id')}: {sorted(missing)}"
        expected = entry["expected_retrieval"]
        assert isinstance(expected, dict), "expected_retrieval must be an object"
        for key in ("text_chunk_ids", "image_ids", "table_ids", "asset_ids"):
            assert key in expected, f"expected_retrieval missing {key}"
            assert isinstance(expected[key], list), f"expected_retrieval.{key} must be a list"


def test_eval_has_image_queries(golden_redesign):
    image_queries = [q for q in golden_redesign if q.get("requires_image")]
    assert len(image_queries) >= 3, "Need at least 3 image-oriented queries"

    for entry in image_queries:
        expected = entry.get("expected_retrieval", {}) or {}
        has_images = any(
            expected.get(key)
            for key in ("image_ids", "table_ids", "asset_ids")
        )
        assert has_images, f"Image query {entry.get('id')} missing expected image ids"


def test_expected_pages_format(golden_redesign):
    pattern = re.compile(r".+#p\d+$")
    for entry in golden_redesign:
        pages = entry.get("expected_pages") or []
        for page in pages:
            assert pattern.match(str(page)), f"Bad expected page format: {page}"


def test_gold_citations_align_with_expected_chunks(golden_redesign):
    for entry in golden_redesign:
        expected_chunks = {
            str(cid) for cid in entry.get("expected_retrieval", {}).get("text_chunk_ids", []) or []
        }
        citations = entry.get("gold_citations") or []
        assert citations, f"Missing gold_citations for {entry.get('id')}"
        for cite in citations:
            chunk_id = str(cite.get("chunk_id") or "")
            assert chunk_id in expected_chunks, (
                f"Citation chunk_id {chunk_id} not in expected_retrieval for {entry.get('id')}"
            )
            assert cite.get("doc_slug"), f"Missing doc_slug in citations for {entry.get('id')}"
            assert cite.get("evidence_text"), f"Missing evidence_text in citations for {entry.get('id')}"


def test_expected_chunks_exist_in_chunks_file(golden_redesign):
    if not CHUNKS_PATH.exists():
        pytest.skip("chunks.jsonl not found; skipping chunk-id consistency check")

    expected = set(_iter_expected_chunks(golden_redesign))
    found = set()

    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = obj.get("chunk_id")
            if cid in expected:
                found.add(cid)
                if len(found) == len(expected):
                    break

    missing = expected - found
    assert not missing, f"Missing chunk_ids in chunks.jsonl: {sorted(list(missing))[:5]}"


def test_expected_assets_exist_in_manifest(golden_redesign):
    if not ASSETS_MANIFEST_PATH.exists():
        pytest.skip("assets_manifest.json not found; skipping asset-id consistency check")

    manifest = json.loads(ASSETS_MANIFEST_PATH.read_text(encoding="utf-8"))
    asset_ids = {str(item.get("asset_id")) for item in manifest if item.get("asset_id")}

    expected_assets = set(_iter_expected_assets(golden_redesign))
    missing = [aid for aid in expected_assets if aid not in asset_ids]
    assert not missing, f"Missing asset_ids in manifest: {missing[:5]}"


def test_prompt_includes_citations_and_idk_rule():
    chain_path = ROOT / "rag" / "chain.py"
    if not chain_path.exists():
        pytest.skip("rag/chain.py missing; skipping prompt rule check")

    content = chain_path.read_text(encoding="utf-8")
    assert "Cite sources as [S#" in content, "Prompt must enforce citation format [S#]"
    assert "I don't know" in content, 'Prompt must include "I don\'t know" fallback'


def _load_retrieval_eval_module():
    module_path = ROOT / "eval" / "run_retrieval_eval.py"
    spec = importlib.util.spec_from_file_location("run_retrieval_eval", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retrieval_metrics_computable(golden_redesign):
    if os.getenv("GV_RUN_RETRIEVAL_TESTS") != "1":
        pytest.skip("Set GV_RUN_RETRIEVAL_TESTS=1 to run live retrieval metrics")

    module = _load_retrieval_eval_module()

    retriever = module.build_retriever(top_k=5)
    try:
        module.verify_qdrant(retriever)
    except Exception as exc:
        pytest.skip(f"Qdrant not available: {exc}")

    asset_resolver = None
    if ASSETS_MANIFEST_PATH.exists():
        from rag.assets import AssetResolver
        asset_resolver = AssetResolver(ASSETS_MANIFEST_PATH)

    summary = module.compute_metrics(
        golden_redesign,
        retriever,
        asset_resolver,
        k_values=[1, 3, 5],
    )

    assert "by_k" in summary and summary["by_k"], "Expected metrics summary by_k"
    for k in ("1", "3", "5"):
        assert k in summary["by_k"], f"Missing metrics for k={k}"
