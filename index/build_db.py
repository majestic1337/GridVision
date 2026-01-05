import os
import json
import uuid
import torch
import pickle
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
TEXT_DIR = BASE_DIR / "data" / "processed" / "text"
IMAGE_METADATA = BASE_DIR / "data" / "processed" / "image_summaries.json"
LOG_FILE = BASE_DIR / "data" / "processed" / "processed_files_log.txt"

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "grid_vision_manuals"

EMBEDDING_MODEL = "BAAI/bge-m3" 
VECTOR_SIZE = 1024 

BATCH_SIZE = 16

def get_device():
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        return 'cuda'
    return 'cpu'

def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL)

def load_processed_log():
    if not LOG_FILE.exists():
        return set()
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def mark_as_processed(filenames):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        for name in filenames:
            f.write(f"{name}\n")

def process_text_optimized(model, client):
    processed_files = load_processed_log()
    text_files = list(TEXT_DIR.glob("*.txt"))
    new_files = [f for f in text_files if f.name not in processed_files]

    if not new_files:
        print("Text: No new files.")
        return

    print(f"Preparing {len(new_files)} text files (CPU stage)...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    all_chunks_data = [] 

    for file_path in tqdm(new_files, desc="Reading from disk"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            source_doc = file_path.stem.split("_page_")[0]
            page_num = "N/A"
            if "_page_" in file_path.stem:
                page_num = file_path.stem.split("_page_")[1].split("_")[0]

            chunks = splitter.split_text(text)
            
            for chunk in chunks:
                all_chunks_data.append({
                    "text": chunk,
                    "filename": file_path.name,
                    "payload": {
                        "type": "text",
                        "content": chunk,
                        "source": source_doc,
                        "page": page_num,
                        "file_path": file_path.name
                    }
                })
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")

    total_chunks = len(all_chunks_data)
    print(f"Starting GPU processing. Total chunks: {total_chunks}")
    print(f"Mode: FP16 (High Performance)")

    for i in tqdm(range(0, total_chunks, BATCH_SIZE), desc="GPU Encoding"):
        batch = all_chunks_data[i : i + BATCH_SIZE]
        
        texts = [item["text"] for item in batch]
        
        embeddings = model.encode(
            texts, 
            batch_size=BATCH_SIZE, 
            show_progress_bar=False, 
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        points = []
        filenames_done = set()
        
        for idx, vector in enumerate(embeddings):
            item = batch[idx]
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload=item["payload"]
            ))
            filenames_done.add(item["filename"])

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        mark_as_processed(filenames_done)

def process_images_optimized(model, client):
    processed = load_processed_log()
    if not IMAGE_METADATA.exists(): return
    
    with open(IMAGE_METADATA, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    new_images = [item for item in data if item['image_filename'] not in processed]
    if not new_images:
        print("Images: No new data.")
        return

    print(f"Processing {len(new_images)} image descriptions...")
    
    for i in tqdm(range(0, len(new_images), BATCH_SIZE), desc="GPU Images"):
        batch = new_images[i : i + BATCH_SIZE]
        texts = [f"Type: Diagram\nDesc: {x['description']}\nSource: {x['source_doc']}" for x in batch]
        
        embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False)
        
        points = []
        filenames = []
        for idx, vector in enumerate(embeddings):
            item = batch[idx]
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload={
                    "type": "image",
                    "content": item['description'],
                    "source": item['source_doc'],
                    "page": item['page_num'],
                    "image_path": item['image_path']
                }
            ))
            filenames.append(item['image_filename'])
            
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        mark_as_processed(filenames)

def main():
    device = get_device()
    
    print("Loading model into memory (FP16)...")
    model = SentenceTransformer(
        EMBEDDING_MODEL, 
        device=device,
        model_kwargs={"torch_dtype": torch.float16} 
    )
    
    client = get_qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            COLLECTION_NAME, 
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )

    process_text_optimized(model, client)
    
    process_images_optimized(model, client)
    
    print("\nFinished!")

if __name__ == "__main__":
    main()