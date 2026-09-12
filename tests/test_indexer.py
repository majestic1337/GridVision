from index.indexer import IndexerConfig, build_point
from config.constants import DEFAULT_QDRANT_URL


def test_build_point_uses_named_vectors():
    cfg = IndexerConfig(
        collection="test_collection",
        qdrant_url=DEFAULT_QDRANT_URL,
        qdrant_api_key=None,
        prefer_grpc=False,
        dense_name="dense_vec",
        sparse_name="sparse_vec",
    )
    chunk = {
        "chunk_id": "c1",
        "content": "text",
        "metadata": {},
    }

    point = build_point(cfg, chunk, [0.1, 0.2], {"indices": [1], "values": [0.3]})
    assert "dense_vec" in point.vector
    assert "sparse_vec" in point.vector
