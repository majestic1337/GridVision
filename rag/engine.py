import os
import torch
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "grid_vision_manuals"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

print("Initializing RAG Engine...")

client = QdrantClient(url=QDRANT_URL)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm_model = genai.GenerativeModel('gemini-2.0-flash')

def search_qdrant(query, top_k=5):
    query_vector = embedder.encode(query).tolist()
    
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )
    return hits

def ask_grid_vision(user_query):
    hits = search_qdrant(user_query)
    
    if not hits:
        return "Sorry, I could not find relevant information in the manuals.", []

    context_text = ""
    sources = []
    
    for hit in hits:
        payload = hit.payload
        score = hit.score
        
        source_str = f"--- Source: {payload.get('source', 'Unknown')} (Page {payload.get('page', 'N/A')}) ---\n"
        source_str += f"Type: {payload.get('type', 'text')}\n"
        source_str += f"Content: {payload.get('content', '')}\n"
        
        context_text += source_str + "\n"
        
        sources.append({
            "score": round(score, 3),
            "type": payload.get("type"),
            "source": payload.get("source"),
            "page": payload.get("page"),
            "content": payload.get("content"),
            "image_path": payload.get("image_path")
        })

    system_prompt = f"""
    You are an expert Technical Support AI for US Army Generators (GridVision).
    Use the Context below to answer the user's question accurately.
    
    Context contains chunks of text and descriptions of diagrams/images.
    If the context includes an Image Description, refer to it as "the diagram" or "the schematic".
    
    Rules:
    1. Base your answer ONLY on the context.
    2. If the answer involves a specific part number or connection, cite it exactly.
    3. If relevant images are found in context, explicitly mention them (e.g., "See the diagram on page 12").
    
    CONTEXT:
    {context_text}
    
    USER QUESTION:
    {user_query}
    """
    
    response = llm_model.generate_content(system_prompt)
    
    return response.text, sources

if __name__ == "__main__":
    q = "Where is the fuel pump located?"
    print(f"User: {q}")
    ans, srcs = ask_grid_vision(q)
    print(f"\nAI: {ans}")
    print(f"\nSources found: {len(srcs)}")