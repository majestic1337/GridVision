from typing import List, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from qdrant_client.models import SparseVector, NamedSparseVector

from config.runtime import get_runtime_config
from config.constants import RETRIEVER_TOP_K, RRF_K

_RUNTIME = get_runtime_config()

class HybridQdrantRetriever(BaseRetriever):
    client: Any  # QdrantClient
    collection_name: str
    dense_model: Any  # SentenceTransformer/BGE-M3
    sparse_model: Any # FastEmbed/Splade
    top_k: int = RETRIEVER_TOP_K   # Wide net for RRF
    dense_vector_name: str = _RUNTIME.dense_vector
    sparse_vector_name: str = _RUNTIME.sparse_vector
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        
        # 1. Encode Query
        dense_vec = self.dense_model.encode(query).tolist()
        # sparse_vec format: SparseVector(indices=[...], values=[...])
        sparse_raw = list(self.sparse_model.embed([query]))[0]
        sparse_vec = SparseVector(indices=sparse_raw.indices.tolist(), values=sparse_raw.values.tolist())

        # 2. Parallel Search (Synchronous for MVP)
        dense_hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=(self.dense_vector_name, dense_vec),
            limit=self.top_k,
            with_payload=True
        )
        sparse_hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=NamedSparseVector(
                name=self.sparse_vector_name,
                vector=sparse_vec,
            ),
            limit=self.top_k,
            with_payload=True,
        )

        # 3. RRF Fusion (Reciprocal Rank Fusion)
        # score = 1 / (k + rank)
        rrf_map = {}
        k = RRF_K
        
        def merge_hits(hits, weight=1.0):
            for rank, hit in enumerate(hits):
                doc_id = hit.payload['chunk_id']
                score = (1 / (k + rank + 1)) * weight
                if doc_id not in rrf_map:
                    rrf_map[doc_id] = {
                        "payload": hit.payload, 
                        "score": 0.0,
                        "dense_score": hit.score if weight==1.0 else 0
                    }
                rrf_map[doc_id]["score"] += score

        merge_hits(dense_hits)
        merge_hits(sparse_hits)

        # 4. Sort & Pack
        sorted_docs = sorted(rrf_map.values(), key=lambda x: x['score'], reverse=True)
        
        return [
            Document(
                page_content=item['payload']['content'],
                metadata={
                    **item['payload'],
                    "rrf_score": item['score'],
                    "dense_score": item.get('dense_score', 0)
                }
            ) for item in sorted_docs[:self.top_k] 
        ]
    
