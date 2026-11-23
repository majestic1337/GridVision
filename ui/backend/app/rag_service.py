import logging
import os
import re
import sys
from typing import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config import constants as const
from rag.assets import AssetResolver
from rag.chain import build_rag_chain
from rag.image_retriever import ImageRetriever
from rag.reranker import Reranker
from rag.retriever import HybridQdrantRetriever

from .assets import AssetIndex
from .settings import Settings

logger = logging.getLogger("GridVision_API_RAG")

_CITATION_RE = re.compile(r"\[(S\d+)\]")
_MIN_TOP_SCORE = float(os.getenv("GV_MIN_TOP_SCORE", "0.0"))
_ENFORCE_CITATIONS = os.getenv("GV_ENFORCE_CITATIONS", "1").lower() not in ("0", "false", "no")
_DEBUG_PROMPT = os.getenv("GV_DEBUG_PROMPT", "0").lower() not in ("0", "false", "no")
_DEBUG_PROMPT_MAX = int(os.getenv("GV_DEBUG_PROMPT_MAX_CHARS", "12000"))

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage
    _HAS_LANGCHAIN_GEMINI = True
except Exception:
    _HAS_LANGCHAIN_GEMINI = False

try:
    import google.generativeai as genai
    _HAS_GOOGLE_GENAI = True
except Exception:
    _HAS_GOOGLE_GENAI = False


VISUAL_KEYWORDS = (
    "diagram",
    "schematic",
    "figure",
    "table",
    "chart",
    "plot",
    "image",
    "photo",
    "picture",
)


@dataclass
class AnswerResult:
    answer: str
    sources: List[Dict[str, Any]]
    attachments: List[Dict[str, Any]]
    debug: Dict[str, Any]


def wants_visuals(query: str) -> bool:
    q = query.lower()
    return any(key in q for key in VISUAL_KEYWORDS)


def _unique_by_asset_id(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for item in items:
        asset_id = str(item.get("asset_id", ""))
        if not asset_id or asset_id in seen:
            continue
        seen.add(asset_id)
        out.append(item)
    return out


def _filter_attachments_by_pages(
    attachments: List[Dict[str, Any]],
    docs: List[Document],
) -> List[Dict[str, Any]]:
    if not attachments:
        return attachments
    allowed_pages = set()
    allowed_asset_ids = set()
    for doc in docs:
        meta = doc.metadata or {}
        linked_asset_id = meta.get("linked_asset_id")
        if linked_asset_id:
            allowed_asset_ids.add(str(linked_asset_id))
        page_label = meta.get("page_label") or meta.get("page_number")
        if page_label is not None:
            allowed_pages.add(str(page_label))

    if not allowed_pages and not allowed_asset_ids:
        return []

    filtered: List[Dict[str, Any]] = []
    for attachment in attachments:
        asset_id = str(attachment.get("asset_id", ""))
        if asset_id and asset_id in allowed_asset_ids:
            filtered.append(attachment)
            continue
        page_label = attachment.get("page_label")
        if page_label is None:
            continue
        if str(page_label) in allowed_pages:
            filtered.append(attachment)
    return filtered


def _format_context(docs: List[Document], limit_chars: int) -> str:
    out: List[str] = []
    total = 0
    for doc in docs:
        meta = doc.metadata or {}
        doc_title = meta.get("title") or meta.get("doc_slug", "Doc")
        page_label = meta.get("page_label") or meta.get("page_number") or "?"
        content = (doc.page_content or "").strip()
        header = f"\n--- Source: {doc_title} (Page {page_label}) ---\n"
        if meta.get("linked_asset_id"):
            header += f"[Contains visual aid: {meta['linked_asset_id']}]\n"
        entry = header + content + "\n"
        if total + len(entry) > limit_chars:
            break
        out.append(entry)
        total += len(entry)
    return "".join(out)


def _format_history(messages: List[Dict[str, Any]], limit: int) -> str:
    if not messages:
        return ""
    lines: List[str] = []
    for msg in messages[-limit:]:
        role = msg.get("role", "user")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        role_label = "User" if role == "user" else "Assistant"
        lines.append(f"{role_label}: {content}")
    return "\n".join(lines)


def _build_prompt(query: str, context: str, history: str) -> str:
    history_block = ""
    if history:
        history_block = f"\nCHAT HISTORY:\n{history}\n"
    return f"""You are GridVision, a technical support assistant.
Answer the user query based ONLY on the context below.{history_block}

RULES:
1. Cite sources as [Doc Name, Page X] using doc_slug/title and page label from headers.
2. If a section mentions a visual aid (figure/table), explicitly refer to it.
3. Be concise and technical. If the answer is not in the context, say so.

CONTEXT:
{context}

QUERY:
{query}
"""


def _build_image_prompt(query: str, context: str) -> str:
    return f"""You are GridVision, a technical assistant.
Answer based ONLY on the image context below.

RULES:
1. Cite sources as [S#] using the source labels in the context.
2. If the context does not support a claim, say "I don't know."
3. Be concise and factual.

CONTEXT:
{context}

USER QUERY:
{query or "Describe what is shown in the image."}
"""


def _docs_to_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for doc in docs:
        meta = doc.metadata or {}
        preview = (doc.page_content or "")[:const.RAG_CONTENT_PREVIEW_CHARS]
        score = (
            meta.get("rerank_score")
            or meta.get("rrf_score")
            or meta.get("score")
            or 0.0
        )
        sources.append(
            {
                "chunk_id": str(meta.get("chunk_id", "na")),
                "doc_slug": str(meta.get("doc_slug", "Unknown")),
                "title": meta.get("title"),
                "source_uri": meta.get("source_uri"),
                "page_label": str(meta.get("page_label", "?")),
                "content_preview": preview + ("..." if len(preview) >= const.RAG_CONTENT_PREVIEW_CHARS else ""),
                "score": float(score),
                "type": str(meta.get("type", "text")),
                "chunk_index": meta.get("chunk_index"),
            }
        )
    return sources


@lru_cache(maxsize=1)
def _get_dense_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device="cpu")


@lru_cache(maxsize=1)
def _get_sparse_model(model_name: str):
    from fastembed import SparseTextEmbedding
    return SparseTextEmbedding(model_name=model_name)


@lru_cache(maxsize=1)
def _get_image_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device="cpu")


@lru_cache(maxsize=1)
def _get_qdrant_client(url: str, api_key: Optional[str]) -> QdrantClient:
    return QdrantClient(url=url, api_key=api_key, timeout=const.QDRANT_TIMEOUT_SECONDS)

def _trim_debug_prompt(text: str) -> str:
    if not text:
        return ""
    if len(text) <= _DEBUG_PROMPT_MAX:
        return text
    return text[: _DEBUG_PROMPT_MAX] + "\n\n...[truncated]"


class RagService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.asset_resolver = AssetResolver(settings.assets_manifest)
        self.asset_index = AssetIndex(settings.assets_manifest)
        self.reranker = Reranker(model_name=settings.rerank_model)

    def _rerank_debug(self) -> Dict[str, Any]:
        status = self.reranker.status()
        enabled = self.settings.use_rerank
        if not enabled:
            return {
                "rerank_enabled": False,
                "rerank_active": bool(status.get("active")),
                "rerank_model": status.get("model"),
                "rerank_disabled_reason": "disabled_by_env",
                "rerank_error": None,
            }
        return {
            "rerank_enabled": True,
            "rerank_active": bool(status.get("active")),
            "rerank_model": status.get("model"),
            "rerank_disabled_reason": status.get("disabled_reason"),
            "rerank_error": status.get("last_error"),
        }

    def model_status(self) -> List[Dict[str, Any]]:
        statuses: List[Dict[str, Any]] = []

        def check_model(key: str, label: str, fn) -> None:
            try:
                fn()
                statuses.append({"key": key, "label": label, "ok": True, "detail": None})
            except Exception as exc:
                statuses.append({"key": key, "label": label, "ok": False, "detail": str(exc)})

        check_model(
            "dense",
            f"dense: {self.settings.dense_model}",
            lambda: _get_dense_model(self.settings.dense_model),
        )
        check_model(
            "sparse",
            f"sparse: {self.settings.sparse_model}",
            lambda: _get_sparse_model(self.settings.sparse_model),
        )
        check_model(
            "image",
            f"image: {self.settings.image_model}",
            lambda: _get_image_model(self.settings.image_model),
        )

        rerank = self.reranker.status()
        if not self.settings.use_rerank:
            statuses.append(
                {
                    "key": "rerank",
                    "label": f"rerank: {rerank.get('model')}",
                    "ok": False,
                    "detail": "disabled_by_env",
                }
            )
        elif rerank.get("active"):
            statuses.append(
                {
                    "key": "rerank",
                    "label": f"rerank: {rerank.get('model')}",
                    "ok": True,
                    "detail": None,
                }
            )
        else:
            detail = rerank.get("disabled_reason") or rerank.get("last_error") or "inactive"
            statuses.append(
                {
                    "key": "rerank",
                    "label": f"rerank: {rerank.get('model')}",
                    "ok": False,
                    "detail": detail,
                }
            )

        return statuses

    def _llm_error_message(self, exc: Exception) -> str:
        msg = str(exc)
        if "Resource exhausted" in msg or "429" in msg:
            return "LLM request failed (429 Resource exhausted). Please try again later."
        return "LLM request failed. Please try again later."

    def _make_retriever(self) -> HybridQdrantRetriever:
        return HybridQdrantRetriever(
            client=_get_qdrant_client(self.settings.qdrant_url, self.settings.qdrant_api_key),
            collection_name=self.settings.collection,
            dense_model=_get_dense_model(self.settings.dense_model),
            sparse_model=_get_sparse_model(self.settings.sparse_model),
            top_k=self.settings.wide_k,
            rrf_k=self.settings.rrf_k,
            dense_vector_name=self.settings.dense_vector,
            sparse_vector_name=self.settings.sparse_vector,
        )

    def _make_image_retriever(self) -> ImageRetriever:
        return ImageRetriever(
            client=_get_qdrant_client(self.settings.qdrant_url, self.settings.qdrant_api_key),
            collection_name=self.settings.asset_collection,
            model=_get_image_model(self.settings.image_model),
            top_k=self.settings.image_top_k,
            min_score=self.settings.image_min_score,
        )

    def _format_asset_context(self, hits: List[Dict[str, Any]], limit_chars: int) -> Dict[str, Any]:
        context = ""
        total = 0
        source_map = []
        for idx, hit in enumerate(hits, 1):
            payload = hit.get("payload") or {}
            sid = f"S{idx}"
            caption = (payload.get("caption") or "").strip()
            title = payload.get("title") or payload.get("doc_slug", "Asset")
            page_label = payload.get("page_label") or "?"
            header = f"\n--- Source [{sid}]: {title} (Page {page_label}) ---\n"
            entry = header + (caption or "No caption available.") + "\n"
            if total + len(entry) > limit_chars:
                break
            context += entry
            total += len(entry)
            source_map.append(
                {
                    "sid": sid,
                    "asset_id": payload.get("asset_id"),
                    "caption_chunk_id": payload.get("caption_chunk_id"),
                }
            )
        return {"context": context, "source_map": source_map}

    def _attachments_from_asset_hits(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        attachments: List[Dict[str, Any]] = []
        for hit in hits:
            payload = hit.get("payload") or {}
            score = hit.get("score")
            asset_id = payload.get("asset_id")
            if not asset_id:
                continue
            attachments.append(
                {
                    "asset_id": asset_id,
                    "type": payload.get("type", "image_asset"),
                    "file_path": payload.get("file_path", ""),
                    "page_label": str(payload.get("page_label", "?")),
                    "element_id": payload.get("element_id", ""),
                    "url": f"/assets/{asset_id}",
                    "score": float(score) if score is not None else None,
                    "doc_id": payload.get("doc_id"),
                    "doc_slug": payload.get("doc_slug"),
                    "title": payload.get("title"),
                    "source_uri": payload.get("source_uri"),
                }
            )
        return attachments

    def _passes_min_score(self, docs: List[Document]) -> bool:
        if not docs:
            return False
        meta = docs[0].metadata or {}
        top_score = meta.get("rerank_score") or meta.get("rrf_score") or 0.0
        try:
            top_score = float(top_score)
        except Exception:
            top_score = 0.0
        return top_score >= _MIN_TOP_SCORE

    def _citations_valid(self, answer: str, allowed_ids: set[str]) -> bool:
        if not allowed_ids:
            return False
        cited = set(_CITATION_RE.findall(answer or ""))
        if not cited:
            return False
        return cited.issubset(allowed_ids)

    def _generate_answer(self, prompt: str) -> str:
        if self.settings.gemini_api_key:
            if _HAS_LANGCHAIN_GEMINI:
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=self.settings.gemini_model,
                        google_api_key=self.settings.gemini_api_key,
                        temperature=self.settings.temperature,
                    )
                    msg = llm.invoke([HumanMessage(content=prompt)])
                    return getattr(msg, "content", "") or ""
                except Exception as exc:
                    logger.warning("LLM (langchain) failed: %s", exc)
                    return self._llm_error_message(exc)
            if _HAS_GOOGLE_GENAI:
                try:
                    genai.configure(api_key=self.settings.gemini_api_key)
                    model = genai.GenerativeModel(self.settings.gemini_model)
                    resp = model.generate_content(prompt)
                    return getattr(resp, "text", "") or ""
                except Exception as exc:
                    logger.warning("LLM (google-generativeai) failed: %s", exc)
                    return self._llm_error_message(exc)
        return "LLM is not configured. Provide GOOGLE_API_KEY to enable full answers."

    def answer(
        self,
        query: str,
        history: Optional[List[Dict[str, Any]]] = None,
        force_assets: Optional[bool] = None,
        use_lcel: Optional[bool] = None,
    ) -> AnswerResult:
        retriever = self._make_retriever()
        rerank_used = False
        use_lcel_effective = False
        lcel_requested = use_lcel is not False
        lcel_available = bool(self.settings.gemini_api_key) and _HAS_LANGCHAIN_GEMINI

        if lcel_requested and lcel_available:
            try:
                history_block = _format_history(history or [], self.settings.history_limit)
                llm = ChatGoogleGenerativeAI(
                    model=self.settings.gemini_model,
                    google_api_key=self.settings.gemini_api_key,
                    temperature=self.settings.temperature,
                )
                chain = build_rag_chain(
                    retriever,
                    self.reranker,
                    self.asset_resolver,
                    llm,
                    rerank_top_n=self.settings.rerank_top_n,
                    rerank_enabled=self.settings.use_rerank,
                    context_limit_chars=self.settings.context_budget,
                    return_response=False,
                    include_prompt=_DEBUG_PROMPT,
                )
                result = chain.invoke({"query": query, "history": history_block})
                docs = result.get("docs", [])
                context = result.get("context", "")
                assets = result.get("assets", [])
                answer_text = result.get("answer", "")
                source_map = result.get("source_map") or []
                prompt_text = result.get("prompt_text")

                rerank_used = self.settings.use_rerank and self.reranker.active
                allowed_ids = {item.get("sid") for item in source_map if item.get("sid")}
                if _ENFORCE_CITATIONS and not self._citations_valid(answer_text, allowed_ids):
                    answer_text = "I don't know."
                if not self._passes_min_score(docs):
                    answer_text = "I don't know."

                sources = _docs_to_sources(docs[: self.settings.top_k])
                use_lcel_effective = True
            except Exception as exc:
                logger.warning("LCEL path failed, falling back to legacy path: %s", exc)

        if not use_lcel_effective:
            docs = retriever.invoke(query)

            reranked = docs
            if self.settings.use_rerank and self.reranker.active:
                reranked = self.reranker.rerank(query, docs, top_n=self.settings.rerank_top_n)
                rerank_used = True
            else:
                reranked = docs[: self.settings.rerank_top_n]

            sources = _docs_to_sources(reranked[: self.settings.top_k])
            assets = self.asset_resolver.resolve(reranked)
            context = _format_context(reranked, self.settings.context_budget)
            history_block = _format_history(history or [], self.settings.history_limit)
            prompt = _build_prompt(query, context, history_block)
            answer_text = self._generate_answer(prompt)
            docs = reranked
            prompt_text = prompt
            if not self._passes_min_score(docs):
                answer_text = "I don't know."

        doc_ids = {str((d.metadata or {}).get("doc_id")) for d in docs if (d.metadata or {}).get("doc_id")}
        doc_slugs = {str((d.metadata or {}).get("doc_slug")) for d in docs if (d.metadata or {}).get("doc_slug")}

        include_assets = force_assets if force_assets is not None else wants_visuals(query)
        attachments: List[Dict[str, Any]] = []
        if include_assets and assets:
            for asset in assets[: self.settings.max_assets]:
                attachment = asset.model_dump()
                attachment["url"] = f"/assets/{attachment['asset_id']}"
                manifest_item = self.asset_index.get(attachment["asset_id"])
                if manifest_item:
                    attachment["doc_id"] = manifest_item.get("doc_id")
                    attachment["file_path"] = manifest_item.get("file_path", attachment.get("file_path"))
                    attachment["page_label"] = str(manifest_item.get("page_label", attachment.get("page_label", "?")))
                attachments.append(attachment)

        # Text query: augment visuals with CLIP text->image retrieval
        if include_assets:
            image_retriever = self._make_image_retriever()
            image_hits = image_retriever.search_by_text(query)
            clip_attachments = self._attachments_from_asset_hits(image_hits)
            # prioritize embedding hits first (better for visual intent)
            if doc_ids or doc_slugs:
                clip_attachments = [
                    a for a in clip_attachments
                    if (a.get("doc_id") in doc_ids) or (a.get("doc_slug") in doc_slugs)
                ]
            attachments = _unique_by_asset_id(attachments + clip_attachments)[: self.settings.max_assets]
            attachments = _filter_attachments_by_pages(attachments, docs)

        return AnswerResult(
            answer=answer_text,
            sources=sources,
            attachments=attachments,
            debug={
                "retrieval_count": len(docs),
                "rerank_used": rerank_used,
                "context_size_chars": len(context),
                "assets_available": len(assets),
                "lcel_used": use_lcel_effective,
                "prompt_text": _trim_debug_prompt(prompt_text) if _DEBUG_PROMPT and prompt_text else None,
                **self._rerank_debug(),
            },
        )

    def answer_with_image(
        self,
        image_bytes: bytes,
        query: str = "",
        history: Optional[List[Dict[str, Any]]] = None,
        force_assets: Optional[bool] = None,
    ) -> AnswerResult:
        if not image_bytes:
            return AnswerResult(
                answer="I don't know.",
                sources=[],
                attachments=[],
                debug={"reason": "no_image"},
            )

        image_retriever = self._make_image_retriever()
        image_hits = image_retriever.search_by_image(image_bytes)
        context_bundle = self._format_asset_context(image_hits, self.settings.context_budget)
        context = context_bundle["context"]
        allowed_ids = {item.get("sid") for item in context_bundle["source_map"] if item.get("sid")}

        attachments = self._attachments_from_asset_hits(image_hits)
        attachments = _unique_by_asset_id(attachments)[: self.settings.max_assets]

        if not context.strip():
            answer_text = "I don't know."
        else:
            prompt = _build_image_prompt(query, context)
            answer_text = self._generate_answer(prompt)
            if _ENFORCE_CITATIONS and not self._citations_valid(answer_text, allowed_ids):
                answer_text = "I don't know."

        return AnswerResult(
            answer=answer_text,
            sources=[],
            attachments=attachments,
            debug={
                "retrieval_count": len(image_hits),
                "context_size_chars": len(context),
                "assets_available": len(attachments),
                "image_mode": True,
                "prompt_text": _trim_debug_prompt(prompt) if _DEBUG_PROMPT and context.strip() else None,
                **self._rerank_debug(),
            },
        )

    def resolve_asset_path(self, asset_id: str) -> Optional[Path]:
        item = self.asset_index.get(asset_id)
        if not item:
            return None
        file_path = Path(str(item.get("file_path", "")))
        if file_path.is_absolute():
            return file_path
        if self.settings.assets_root:
            return self.settings.assets_root / file_path
        return self.settings.root_dir / file_path
