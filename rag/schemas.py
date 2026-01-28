from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Source(BaseModel):
    chunk_id: str
    doc_slug: str
    title: Optional[str] = None
    source_uri: Optional[str] = None
    page_label: str
    content_preview: str
    score: float
    type: str  # text | image_caption | table_md
    chunk_index: Optional[int] = None

class Asset(BaseModel):
    asset_id: str
    type: str  # image | table_html
    file_path: str
    page_label: str
    element_id: str

class DebugInfo(BaseModel):
    retrieval_count: int
    rerank_used: bool
    context_size_chars: int

class RAGResponse(BaseModel):
    answer: str
    sources: List[Source]
    assets: List[Asset]
    debug: DebugInfo
