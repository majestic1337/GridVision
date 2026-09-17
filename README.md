# GridVision

Multimodal RAG assistant for technical manuals. It pulls the best text chunks and
the right visuals (images + tables) into one answer.

---

## Architecture

- [Ingestion pipeline](docs/ingest_pipeline.mmd)
- [RAG pipeline](docs/rag_pipeline.mmd)

---

## What this repo does

- Ingests PDFs (text + images + tables) into a clean, structured dataset.
- Builds a hybrid retrieval layer (dense + sparse) over text chunks.
- Indexes image/table assets for visual retrieval.
- Serves a chat UI with inline sources and visuals.
- Supports image-only queries (upload a diagram, ask a question).

---

## Status (feel free to update)

- [x] Ingestion pipeline
- [x] Text + image/table extraction
- [x] Qdrant indexing
- [x] Retrieval + RAG responses
- [x] UI with sources and attachments
- [x] Retrieval evaluation script
- [ ] Answer-level evaluation and error analysis
- [ ] PRD + final report polish

---

## Quickstart (local)

### 1) Start Qdrant
```bash
docker run --rm -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 2) Python deps
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

### 3) Frontend deps
```bash
cd ui/web
npm install
```

### 4) Run tests
```bash
cd ../..
pytest -q
```

### 5) Configure environment
Create a `.env` (or export env vars in your shell):

```bash
export QDRANT_URL="http://localhost:6333"
export GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
```

Optional (recommended):
```bash
export GV_IMAGE_EMBEDDINGS=1
export GV_IMAGE_MODEL="clip-ViT-B-32"
```

### 6) Ingest PDFs
Place your PDFs in `data/raw/`, then:
```bash
python ingest/run_ingestion.py
```

Optional: backfill image embeddings for existing assets:
```bash
python ingest/run_ingestion.py --embed-assets
```

### 7) Build indexes
Text chunks:
```bash
python index/indexer.py
```

Assets (images/tables):
```bash
python index/assets_indexer.py
```

### 8) Run the backend
```bash
uvicorn ui.backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### 9) Run the UI
```bash
cd ui/web
npm run dev
```

Open the UI in your browser and ask:
- "Show the diagram for a collimating telescope."
- "Which figure shows the spiral tapered screw extractor?"
- "What does the manual say about sealing compound?"

---

## How it works (short version)

1) **Ingest** PDFs with Docling + OCR.
2) **Chunk** text and score quality.
3) **Extract** tables/images and link them to pages.
4) **Embed** text (dense + sparse) and assets.
5) **Index** everything in Qdrant.
6) **Retrieve** with hybrid search + RRF, then rerank.
7) **Answer** with citations and inline visuals.

---

## Evaluation

Retrieval metrics over a gold set:
```bash
python eval/run_retrieval_eval.py \
  --queries eval/golden_redesign.json \
  --k 1 3 5 10 \
  --out eval/retrieval_metrics.json
```

This evaluates hybrid text retrieval and direct CLIP text-to-asset retrieval. It
requires both Qdrant collections. Use `--skip-asset-eval` only when working on
the text index alone.

Answer evaluation rubric: [docs/eval/answer-quality.md](docs/eval/answer-quality.md).

---

## Data layout

- `data/raw/` - source PDFs (add yours here)
- `data/processed/artifacts/` - chunks + manifests + reports
- `data/processed/assets/` - extracted tables/images
- `data/processed/uploads/` - user-uploaded images for image queries

---

## Configuration (env vars)

Minimal:
- `QDRANT_URL` (default: http://localhost:6333)
- `GOOGLE_API_KEY` (Gemini for answer generation)

Useful:
- `GV_DENSE_MODEL` (default: `BAAI/bge-m3`)
- `GV_SPARSE_MODEL` (default: `prithivida/Splade_PP_en_v1`)
- `GV_IMAGE_MODEL` (default: `clip-ViT-B-32`)
- `GV_IMAGE_EMBEDDINGS` (set to `1` to store image embeddings)
- `GV_RERANK_MODEL` (default: `ms-marco-MiniLM-L-12-v2`)
- `GV_ASSET_COLLECTION` (default: `gridvision_assets`)
- `QDRANT_COLLECTION` (default: `gridvision_chunks`)
- `GV_ASSETS_MANIFEST` (default: `data/processed/artifacts/assets_manifest.json`)
- `GV_ASSETS_ROOT` (override asset root path)
- `GV_MAX_ASSETS` (default: 6)
- `GV_IMAGE_TOPK` (default: 8)
- `GV_IMAGE_MIN_SCORE` (default: `0.0`)
- `GV_RRF_K` (default: `60`)

UI:
- `NEXT_PUBLIC_API_URL` (default: http://localhost:8000)
- `GV_CORS_ORIGINS` (default: http://localhost:3000)

---

## Project structure

```
config/        # runtime constants
data/          # raw + processed data
eval/          # evaluation dataset + script
index/         # Qdrant indexers
ingest/        # ingestion + parsing + enrichment
rag/           # retrieval + chain logic
ui/backend/    # FastAPI backend
ui/web/        # Next.js frontend
```

---

## Notes & gotchas

- The backend only accepts English queries (Cyrillic is blocked by design).
- If you skip image embeddings, image retrieval will be limited to linked assets.
- `image_embeddings.enabled` is `false` in `metadata/ingest_config.yaml` by default.
  Set `GV_IMAGE_EMBEDDINGS=1` or flip the config to enable CLIP embeddings.

---

## Acknowledgements

- Docling for PDF parsing + OCR
- Qdrant for vector search
- LangChain (core) for orchestration
- SentenceTransformers, SPLADE, CLIP for embeddings
