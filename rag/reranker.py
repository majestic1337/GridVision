import logging
from typing import List
from langchain_core.documents import Document

from config.constants import DEFAULT_RERANK_MODEL, RERANK_TOP_N

logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self, model_name=DEFAULT_RERANK_MODEL, use_gpu=False):
        self.model_name = model_name
        self.active = False
        self.disabled_reason = None
        self.last_error = None
        try:
            from flashrank import Ranker
            self.ranker = Ranker(model_name=model_name)
            self.active = True
        except ImportError as exc:
            self.disabled_reason = "flashrank_not_installed"
            self.last_error = str(exc)
            logger.warning("FlashRank not found. Reranking disabled.")
        except Exception as exc:
            self.disabled_reason = "flashrank_init_failed"
            self.last_error = str(exc)
            logger.warning("FlashRank init failed. Reranking disabled: %s", exc)

    def rerank(self, query: str, docs: List[Document], top_n: int = RERANK_TOP_N) -> List[Document]:
        self.last_error = None
        if not self.active or not docs:
            return docs[:top_n]

        try:
            # FlashRank format conversion
            passages = [
                {"id": str(i), "text": d.page_content, "meta": d.metadata} 
                for i, d in enumerate(docs)
            ]
            
            from flashrank import RerankRequest
            request = RerankRequest(query=query, passages=passages)
            results = self.ranker.rerank(request)
            
            # Reconstruct Documents
            final_docs = []
            for res in results[:top_n]:
                d = Document(page_content=res['text'], metadata=res['meta'])
                d.metadata['rerank_score'] = res['score']
                final_docs.append(d)
                
            return final_docs
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Rerank failed: {e}. Fallback to RRF.")
            return docs[:top_n]

    def status(self) -> dict:
        return {
            "active": self.active,
            "model": self.model_name,
            "disabled_reason": self.disabled_reason,
            "last_error": self.last_error,
        }
