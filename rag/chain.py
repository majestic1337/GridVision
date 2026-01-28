from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from operator import itemgetter

from .schemas import RAGResponse, Source, DebugInfo
from .retriever import HybridQdrantRetriever
from .reranker import Reranker
from .assets import AssetResolver
from config.constants import RAG_CONTENT_PREVIEW_CHARS, RAG_CONTEXT_LIMIT_CHARS, RERANK_TOP_N

# --- SETUP ---
# (Assume clients/models initialized externally and passed here)
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def build_rag_chain(retriever, reranker, asset_resolver, llm):

    # 1. Retrieval Branch
    def retrieve_and_rank(input_dict):
        query = input_dict["query"]
        # Step A: Hybrid Search (Top-50)
        initial_docs = retriever.invoke(query)
        # Step B: Rerank (Top-5)
        final_docs = reranker.rerank(query, initial_docs, top_n=RERANK_TOP_N)
        return final_docs

    # 2. Context Formatting (With Budget)
    def format_context(docs):
        context_str = ""
        total_chars = 0
        limit = RAG_CONTEXT_LIMIT_CHARS # Budget
        
        for doc in docs:
            content = doc.page_content
            # Citing header
            doc_title = doc.metadata.get("title") or doc.metadata.get("doc_slug", "Doc")
            header = f"\n--- Source: {doc_title} (Page {doc.metadata.get('page_label', '?')}) ---\n"
            
            # Asset hint for LLM
            if doc.metadata.get('linked_asset_id'):
                header += f"[Contains visual aid: {doc.metadata['linked_asset_id']}]\n"
            
            entry = header + content + "\n"
            
            if total_chars + len(entry) > limit:
                break
            
            context_str += entry
            total_chars += len(entry)
            
        return context_str

    # 3. Prompt
    prompt = ChatPromptTemplate.from_template("""
    You are GridVision, a technical support assistant.
    Answer the user query based ONLY on the context below.
    
    RULES:
    1. Cite sources as [Doc Name, Page X].
    2. If a section mentions a visual aid (figure/table), explicitly refer to it (e.g., "See Figure 2-4").
    3. Be concise and technical.
    
    CONTEXT:
    {context}
    
    QUERY:
    {query}
    """)

    # 4. Response Packing (Converts raw data to Pydantic)
    def pack_response(input_dict):
        docs = input_dict["docs"]
        ai_msg = input_dict["ai_response"]
        assets = input_dict["assets"]
        
        sources = [
            Source(
                chunk_id=d.metadata.get('chunk_id', 'na'),
                doc_slug=d.metadata.get('doc_slug', 'Unknown'),
                title=d.metadata.get("title"),
                source_uri=d.metadata.get("source_uri"),
                page_label=str(d.metadata.get('page_label', '?')),
                score=d.metadata.get('rerank_score', d.metadata.get('rrf_score', 0)),
                type=d.metadata.get('type', 'text'),
                content_preview=d.page_content[:RAG_CONTENT_PREVIEW_CHARS] + "...",
                chunk_index=d.metadata.get("chunk_index")
            ) for d in docs
        ]
        
        return RAGResponse(
            answer=ai_msg.content,
            sources=sources,
            assets=assets,
            debug=DebugInfo(
                retrieval_count=len(docs),
                rerank_used=True,
                context_size_chars=len(input_dict["context"])
            )
        )

    # --- LCEL GRAPH ---
    
    retrieval_branch = RunnableLambda(retrieve_and_rank)
    
    main_chain = (
        RunnableParallel({
            "docs": retrieval_branch,
            "query": itemgetter("query")
        })
        .assign(
            context = lambda x: format_context(x["docs"]),
            assets = lambda x: asset_resolver.resolve(x["docs"])  # <-- ASSET JOIN BEFORE LLM
        )
        .assign(
            ai_response = (
                {"context": itemgetter("context"), "query": itemgetter("query")}
                | prompt 
                | llm
            )
        )
        | RunnableLambda(pack_response)
    )
    
    return main_chain
