from qdrant_client.models import SparseVector

from rag.retriever import HybridQdrantRetriever


class FakeArray(list):
    def tolist(self):
        return list(self)


class FakeDense:
    def encode(self, query):
        return FakeArray([0.1, 0.2])


class FakeSparseVec:
    def __init__(self):
        self.indices = FakeArray([1, 2])
        self.values = FakeArray([0.1, 0.2])


class FakeSparse:
    def embed(self, texts):
        return [FakeSparseVec()]


class FakeHit:
    def __init__(self, payload, score=0.1):
        self.payload = payload
        self.score = score


class FakeClient:
    def __init__(self):
        self.calls = []

    def search(self, collection_name, query_vector, limit, with_payload):
        self.calls.append((collection_name, query_vector))
        return [
            FakeHit(
                {
                    "chunk_id": "c1",
                    "content": "hello",
                    "doc_slug": "doc",
                },
                0.2,
            )
        ]


def test_retriever_uses_named_vectors_and_sparse_vector():
    client = FakeClient()
    retriever = HybridQdrantRetriever(
        client=client,
        collection_name="collection",
        dense_model=FakeDense(),
        sparse_model=FakeSparse(),
        top_k=2,
    )

    docs = retriever._get_relevant_documents("query")
    assert len(client.calls) == 2

    dense_call = client.calls[0]
    sparse_call = client.calls[1]

    assert dense_call[1][0] == retriever.dense_vector_name
    assert sparse_call[1][0] == retriever.sparse_vector_name
    assert isinstance(sparse_call[1][1], SparseVector)

    assert docs[0].page_content == "hello"
    assert "rrf_score" in docs[0].metadata
