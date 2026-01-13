import os
import json
from filelock import FileLock
import torch
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue, MatchAny
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
TEXT_DIR = BASE_DIR / "data" / "processed" / "text"
IMAGE_METADATA = BASE_DIR / "data" / "processed" / "image_summaries.json"
LOG_FILE = BASE_DIR / "data" / "processed" / "processed_files_log.txt"

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "gridvision_manuals")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3") 
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1024")) 
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
MAX_DOC_SIZE = 10_000_000

def get_device():
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        return 'cuda'
    return 'cpu'

def get_qdrant_client():
    client = QdrantClient(url=QDRANT_URL)
    try:
        client. get_collections()
        print(f"Connected to Qdrant at {QDRANT_URL}")
        return client
    except (ConnectionError, TimeoutError) as e:
        raise ConnectionError(f"Cannot reach Qdrant: {e}")

def load_processed_log():
    if not LOG_FILE.exists():
        return set()
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def mark_as_processed(filenames):
    lock = FileLock(str(LOG_FILE) + ".lock")
    with lock, open(LOG_FILE, "a", encoding="utf-8") as f:
        for name in filenames:
            f.write(f"{name}\n")

def generate_deterministic_id(content, source, page):
    key = f"{source}|{page}|{content.strip()}"
    return hashlib.md5(key.encode('utf-8', errors='ignore')).hexdigest()

def process_text_optimized(model, client):
    processed_files = load_processed_log()
    text_files = list(TEXT_DIR.glob("*.txt"))
    new_files = [f for f in text_files if f.name not in processed_files]

    if not new_files:
        print("Text: No new files.")
        return

    print(f"Preparing {len(new_files)} text files (CPU stage)...")
    
    docs_by_source = defaultdict(list)
    for file_path in new_files:
        try:
            source_doc = file_path.stem.split("_page_")[0]
            docs_by_source[source_doc].append(file_path)
        except IndexError:
            print(f"Skipping malformed filename: {file_path.name}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    all_chunks_data = [] 

    for source_doc, file_list in docs_by_source.items():
        # Sort by page number
        try:
            file_list.sort(key=lambda f: int(f.stem.split("_page_")[1].split("_")[0]))
        except (ValueError, IndexError):
            pass

        page_texts = []
        page_metadatas = []
        original_filenames = []
        current_doc_size = 0
        skip_doc = False

        for fp in file_list:
            try:
                page_num = fp.stem.split("_page_")[1].split("_")[0]
                with open(fp, "r", encoding="utf-8") as f:
                    text = f.read()

                if current_doc_size + len(text) > MAX_DOC_SIZE:
                    print(f"Warning: {source_doc} exceeds size limit ({MAX_DOC_SIZE}), skipping.")
                    skip_doc = True
                    break
                
                current_doc_size += len(text)
                page_texts.append(text)
                page_metadatas.append({
                    "source": source_doc,
                    "page": page_num,
                    "file_path": fp.name
                })
                original_filenames.append(fp.name)
            except Exception as e:
                print(f"Error reading {fp.name}: {e}")
                continue
        
        if skip_doc:
            continue

        if page_texts:
            split_docs = splitter.create_documents(page_texts, metadatas=page_metadatas)

            for doc in split_docs:
                all_chunks_data.append({
                    "text": doc.page_content,
                    "filename": f"{source_doc}_processed", 
                    "original_files": original_filenames,
                    "payload": {
                        "type": "text",
                        "content": doc.page_content,
                        "source": doc.metadata["source"],
                        "page": doc.metadata["page"]
                    }
                })

    purge_targets = defaultdict(set)
    for item in all_chunks_data:
        purge_targets[item["payload"]["source"]].add(item["payload"]["page"])

    print("Cleaning up stale vectors...")
    for source, pages in purge_targets.items():
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(key="source", match=MatchValue(value=source)),
                    FieldCondition(key="type", match=MatchValue(value="text")),
                    FieldCondition(key="page", match=MatchAny(any=list(pages))),
                ]
            )
        )

    total_chunks = len(all_chunks_data)
    print(f"Starting GPU processing. Total chunks: {total_chunks}")

    for i in tqdm(range(0, total_chunks, BATCH_SIZE), desc="GPU Encoding"):
        batch = all_chunks_data[i : i + BATCH_SIZE]
        texts = [item["text"] for item in batch]
        
        embeddings = model.encode(
            texts, batch_size=BATCH_SIZE, 
            show_progress_bar=False,
            normalize_embeddings=True  
        )

        points = []
        filenames_done = set()
        
        for idx, vector in enumerate(embeddings):
            item = batch[idx]
            point_id = generate_deterministic_id(
                item["payload"]["content"],
                item["payload"]["source"],
                item["payload"]["page"]
            )
            points.append(PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=item["payload"]
            ))
            filenames_done.update(item["original_files"])

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        mark_as_processed(filenames_done)

def process_single_image(model, item):
    """Generates PointStruct for a single image item."""
    text = f"Type: Diagram\nDesc: {item['description']}\nSource: {item['source_doc']}"
    embedding = model.encode(text)
    point_id = generate_deterministic_id(
        item['description'], item['source_doc'], item['page_num']
    )
    return PointStruct(
        id=point_id,
        vector=embedding.tolist(),
        payload={
            "type": "image",
            "content": item['description'],
            "source": item['source_doc'],
            "page": item['page_num'],
            "image_path": item['image_path']
        }
    )

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
        
        try:
            img_paths = [x['image_path'] for x in batch]
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="type", match=MatchValue(value="image")),
                        FieldCondition(key="image_path", match=MatchAny(any=img_paths))
                    ]
                )
            )
            texts = [f"Type: Diagram\nDesc: {x['description']}\nSource: {x['source_doc']}" for x in batch]
            embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False, normalize_embeddings =True)
            
            points = []
            filenames = []
            for idx, vector in enumerate(embeddings):
                item = batch[idx]
                point_id = generate_deterministic_id(
                    item['description'], item['source_doc'], item['page_num']
                )
                points.append(PointStruct(
                    id=point_id,
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

        except Exception as e:
            print(f"Error processing image batch {i//BATCH_SIZE}: {e}")
            for item in batch:
                try:
                    point = process_single_image(model, item)
                    client.upsert(collection_name=COLLECTION_NAME, points=[point])
                    mark_as_processed([item['image_filename']])
                except Exception as e2:
                    print(f"Failed to process {item['image_filename']}: {e2}")

def main():
    device = get_device()
    
    try:
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
    
    finally:
        try:
            if 'model' in locals() and device == 'cuda':
                del model
                torch.cuda.empty_cache()
        except Exception: 
            pass

if __name__ == "__main__":
    main()