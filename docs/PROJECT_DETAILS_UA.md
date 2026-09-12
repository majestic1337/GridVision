# GridVision — повний опис проєкту (детально)

Цей документ описує **всі ключові компоненти репозиторію**, як вони пов’язані між собою, у якій послідовності викликаються та які артефакти створюють.

---

## 1) Призначення проєкту (коротко)
**GridVision** — це мультимодальний RAG‑асистент для технічних мануалів: витягує найрелевантніші текстові фрагменти, а також правильні візуальні матеріали (зображення/таблиці) і показує їх у відповіді.

---

## 2) Структура репозиторію (що де лежить)
- `config/` — константи та runtime-конфіг (env‑змінні, дефолти, імена колекцій, параметри чанкінгу/ретрівалу/UI).
- `metadata/` — YAML‑конфіг інжесту (`ingest_config.yaml`).
- `ingest/` — пайплайн інжесту PDF → текст/таблиці/зображення → артефакти.
- `index/` — індексація артефактів у Qdrant (текстові чанки та візуальні ассети).
- `rag/` — ретрівер, реранкер, LCEL‑chain, резолвер ассетів.
- `ui/backend/` — FastAPI бекенд для чату, RAG‑сервіс, SQLite історія.
- `ui/web/` — Next.js фронтенд чату.
- `eval/` — скрипти та датасети для оцінки retrieval.
- `docs/` — Mermaid‑діаграми пайплайнів.
- `data/` — дані (raw/processed), артефакти, логи, завантаження користувача.
- `tests/` — юніт‑тести для інжесту, індексації та ретріверу.

---

## 3) Конфігурація (де задаються параметри)
### 3.1 `metadata/ingest_config.yaml`
Задає параметри інжесту та чанкінгу:
- `paths.*` — каталоги raw/processed/logs.
- `parser.*` — engine (Docling), OCR‑стратегія.
- `chunking.*` — розмір/перекриття чанків, пороги для абзаців.
- `quality_gates.*` — пороги якості (мін. оцінка, мін. кількість слів, щільність тексту).
- `image_embeddings.*` — вмикання/модель для embeddings зображень.

### 3.2 `config/constants.py`
Головні дефолти та регулятори:
- **Моделі**: `BAAI/bge-m3` (dense), `Splade_PP_en_v1` (sparse), `ms-marco-MiniLM-L-12-v2` (rerank), `clip-ViT-B-32` (image), `gemini-2.0-flash` (caption/LLM).
- **Параметри чанкінгу**: `DEFAULT_CHUNK_SIZE=1000`, `DEFAULT_CHUNK_OVERLAP=150`, пороги якості.
- **RAG**: `RETRIEVER_TOP_K=50`, `RRF_K=60`, `RERANK_TOP_N=5`, `RAG_CONTEXT_LIMIT_CHARS=6000`.
- **UI**: дефолти wide/top‑k, temperature, context budget, max assets.

### 3.3 Основні ENV‑змінні
- Qdrant: `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`.
- Моделі: `GV_DENSE_MODEL`, `GV_SPARSE_MODEL`, `GV_RERANK_MODEL`, `GV_IMAGE_MODEL`.
- Візуальні ембедінги: `GV_IMAGE_EMBEDDINGS=1` (вмикає), `GV_IMAGE_TOPK`.
- LLM: `GOOGLE_API_KEY`, `GV_GEMINI_MODEL`, `GV_TEMP`.
- UI/Backend: `NEXT_PUBLIC_API_URL`, `GV_CORS_ORIGINS`.

---

## 4) Інжест PDF (повний ланцюжок)
**Entry point:** `ingest/run_ingestion.py`.

### 4.1 Запуск
- `python ingest/run_ingestion.py`
- Аргументи:
  - `--config` → шлях до `ingest_config.yaml`.
  - `--raw-dir` → override папки з PDF.
  - `--pattern` → glob для PDF.
  - `--reset` → очистити артефакти.
  - `--reingest` → не пропускати вже індексовані документи.
  - `--embed-assets` → **тільки** довантажити embeddings для існуючих asset’ів.

### 4.2 Підготовка пайплайна (`IngestionProcessor`)
- `PipelineFoundation`:
  - знаходить/завантажує YAML конфіг.
  - створює директори `data/raw`, `data/processed/assets`, `data/processed/artifacts`, `data/logs`.
  - формує `run_id` та лог‑файл.
- `ArtifactStore`:
  - ініціалізує/створює файли артефактів.
  - читає вже інжестнуті `doc_id` з `documents.jsonl` або `chunks.jsonl`.
- Підключення ключових компонентів:
  - `DocumentParser` (Docling + PyMuPDF).
  - `ImageEnricher` (Gemini captions).
  - `ImageEmbedder` (CLIP/SentenceTransformers, якщо ввімкнено).
  - `SparseEncoder` (FastEmbed SPLADE).
  - `AssetManager` + `ChunkRegistry` + `IngestionReporter`.

### 4.3 Обхід PDF файлів
Для кожного PDF:
1. **Doc ID** (`doc_hash`, `doc_slug`) → MD5 по байтах PDF.
2. **Dedup**: якщо `doc_id` вже в `documents.jsonl`, файл пропускається (якщо не `--reingest`).
3. **Docling parse** + OCR: створює елементи типів `text`, `table`, `image`.
4. **PyMuPDF**: витягує позиції зображень у PDF (bbox для crop).
5. **Групування по сторінках** + сортування елементів.

### 4.4 Обробка елементів (по типу)
#### A) Текст
- Нормалізація (`TextNormalizer.normalize_keep_paragraphs`).
- Видалення header/footer (якщо `drop_headers/drop_footers`).
- Розбиття на абзаци → чанкінг (`RecursiveCharacterTextSplitter`).
- Фільтрація коротких чанків, якщо це не технічний токен.
- Оцінка якості (`QualityScorer.evaluate`).
- Побудова sparse-вектору (`SparseEncoder`).
- Формування `chunk_record` (metadata, provenance, doc_id, source_uri, tags, page info, quality).

#### B) Таблиці
- `export_to_markdown()` → `table_md` контент.
- Далі той же шлях: нормалізація → чанкінг → quality → chunk.

#### C) Зображення
- `AssetManager.register_element()`:
  - Створює `asset_id`.
  - Вирізає PNG через PyMuPDF **або** бере `raw_item.image`.
  - Для таблиць — експортує HTML.
- `ImageEnricher` (Gemini): будує JSON + caption.
- (Опційно) `ImageEmbedder` → вектор для Qdrant assets.
- Caption зберігається як `image_caption` chunk.

### 4.5 Артефакти інжесту (на виході)
Пишуться у `data/processed/artifacts/`:
- `chunks.jsonl` — кожен chunk із текстом/таблицею/зображенням (caption) + metadata + sparse vector.
- `assets_manifest.json` — список assets (asset_id, file_path, bbox, doc_id, page).
- `documents.jsonl` — реєстр документів.
- `element_to_chunks.json` — мапа `element_id -> [chunk_id]`.
- `page_to_assets.json` — мапа `doc_id_page -> [asset_id]`.
- `ingest_report.json` — підсумкова статистика по якості.

---

## 5) Індексування текстових чанків у Qdrant
**Entry point:** `index/indexer.py`

### 5.1 Вхідні дані
- `data/processed/artifacts/chunks.jsonl`.

### 5.2 Етапи
1. **Ініціалізація Qdrant**: створення колекції `gridvision_chunks` (або ENV). Вектори:
   - Dense vector (cosine).
   - Sparse vector (inverted index).
2. **Dense embeddings**: `SentenceTransformer` (модель `GV_DENSE_MODEL`).
3. **Sparse**: якщо в `chunk.sparse_vector` немає даних — обчислює через FastEmbed.
4. **Upsert** батчами.
5. **Savepoint** (`index_savepoint.json`) — для відновлення після переривань.
6. **Опціонально** prune (видалення застарілих chunk_id), validate, test-search.

### 5.3 Точки даних у Qdrant
Payload містить:
- `chunk_id`, `doc_id`, `doc_slug`, `title`, `source_uri`, `type`.
- `page_label`, `linked_asset_id`, `metadata`, `provenance`.
- `content` — текст чанка.

---

## 6) Індексування ассетів (зображень/таблиць)
**Entry point:** `index/assets_indexer.py`

### 6.1 Умова
`assets_manifest.json` **має** містити `embedding` для кожного asset (спочатку треба `--embed-assets`).

### 6.2 Етапи
1. Читає `assets_manifest.json`.
2. Визначає dim embeddings.
3. Створює Qdrant колекцію `gridvision_assets`.
4. Upsert payload:
   - `asset_id`, `element_id`, `doc_id`, `page_label`.
   - `file_path`, `caption`, `caption_chunk_id`.

---

## 7) Retrieval + RAG (текстові запити)
**Основні модулі:** `rag/retriever.py`, `rag/reranker.py`, `rag/chain.py`, `rag/assets.py`

### 7.1 Hybrid Retriever
- Dense embedding запиту → Qdrant search.
- Sparse embedding запиту → Qdrant search.
- Об’єднання через **RRF (Reciprocal Rank Fusion)**.
- Результат — топ‑K документів (`Document` з payload).

### 7.2 Реранк
- FlashRank (якщо встановлено) + `ms-marco-MiniLM-L-12-v2`.
- Якщо FlashRank недоступний — повертає Top‑N після RRF.

### 7.3 Asset Resolver
- Прив’язує зображення до текстових чанків:
  1) за `linked_asset_id`
  2) за `element_id`
  3) fallback: `doc_id + page_label`

### 7.4 LCEL chain (опційно)
`rag/chain.py` будує LCEL граф:
- Retrieve + rerank → format context → LLM → pack response.

---

## 8) Backend API (FastAPI)
**Entry point:** `ui/backend/app/main.py`

### 8.1 Основні endpoints
- `GET /health` — healthcheck.
- `GET /chats` — список чатів.
- `POST /chats` — створити чат.
- `GET /chats/{id}` — чат + історія.
- `DELETE /chats/{id}` — видалити чат.
- `POST /chats/{id}/messages` — синхронна відповідь.
- `POST /chats/{id}/messages/stream` — SSE стрімінг відповіді.
- `POST /chats/{id}/messages/image` — запит + завантажене зображення.
- `GET /assets/{asset_id}` — повертає PNG/HTML asset.
- `GET /uploads/{filename}` — повертає user‑upload.

### 8.2 Chat DB (SQLite)
- `data/processed/chat.db`.
- Таблиці: `chats`, `messages`.

### 8.3 RagService (`ui/backend/app/rag_service.py`)
- Визначає **основний RAG‑потік**:
  - Retrieve → rerank → форматування контексту → LLM → джерела/attachments.
- Опції:
  - LCEL (LangChain) якщо доступний Gemini.
  - Мінімальний score для відповіді.
  - Перевірка цитувань ([S#]).
- Візуальні запити:
  - Використовує ImageRetriever (CLIP) з `gridvision_assets`.
  - Додає attachments з найрелевантніших asset’ів.

---

## 9) Image‑RAG (пошук за зображенням)
**Потік:** `/messages/image`
1. Завантаження зображення → `data/processed/uploads/`.
2. ImageRetriever (CLIP) → пошук по `gridvision_assets`.
3. Формування короткого контексту з caption’ів.
4. LLM відповідає по контексту.

---

## 10) Frontend (Next.js)
**Entry point:** `ui/web/app/page.tsx`

### 10.1 Основні можливості
- Список чатів (ліва панель).
- Відображення повідомлень + attachments (зображення/HTML‑таблиці).
- SSE‑стрімінг відповіді.
- Діалогове вікно збільшення зображення.
- Валідація: дозволено лише **англійські** запити.

### 10.2 Комунікація з бекендом
- `NEXT_PUBLIC_API_URL` → URL FastAPI.
- Всі запити йдуть на `/chats/...`.

---

## 11) Оцінка retrieval
**Entry point:** `eval/run_retrieval_eval.py`
- Приймає список запитів (`golden_eval_50.jsonl`).
- Рахує recall/precision/mrr для текстових chunk_id.
- Окремо рахує page‑level метрики.
- Перевіряє image‑hit по `assets_manifest`.
- Результат → `eval/retrieval_metrics.json`.

---

## 12) Тести
- `tests/test_ingest_pipeline.py` — інжест: чанки/ассети/маніфест.
- `tests/test_indexer.py` — індексація Qdrant payload.
- `tests/test_retriever.py` — RRF пошук.
- `tests/test_text_utils.py` — нормалізація та scorers.
- `tests/test_eval_requirements.py` — валідація eval даних.

---

## 13) Артефакти та їх використання
- `chunks.jsonl` → текстовий індекс (`index/indexer.py`).
- `assets_manifest.json` → asset індекс (`index/assets_indexer.py`) + runtime для UI.
- `documents.jsonl` → metadata в assets index.
- `element_to_chunks.json` → дебаг/перевірки.
- `page_to_assets.json` → зв’язок сторінка ↔ ассети.
- `ingest_report.json` → метрики якості інжесту.

---

## 14) Типовий сценарій роботи (послідовність)
1. **Підготовка**: покласти PDF у `data/raw/`.
2. **Інжест**: `python ingest/run_ingestion.py`.
3. **Індекс тексту**: `python index/indexer.py`.
4. **(Опційно) Ембедінги для ассетів**:
   - `python ingest/run_ingestion.py --embed-assets`
5. **Індекс ассетів**: `python index/assets_indexer.py`.
6. **Запуск бекенда**: `uvicorn ui.backend.app.main:app --reload`.
7. **Запуск фронтенду**: `cd ui/web && npm run dev`.
8. **Запит користувача** → retriever → rerank → LLM → attachments → UI.

---

## 15) Діаграми
- `docs/ingest_pipeline.mmd` — детальна схема інжесту.
- `docs/rag_pipeline.mmd` — детальна схема RAG‑потоку.

---

## 16) Ключові залежності
- **Docling** — парсинг PDF та OCR.
- **PyMuPDF** — вирізання зображень.
- **SentenceTransformers** — dense embeddings.
- **FastEmbed** — sparse embeddings (SPLADE).
- **Qdrant** — векторна БД.
- **FastAPI** — backend API.
- **Next.js** — frontend.
- **Gemini (google.generativeai / langchain)** — LLM + image captions.

---

## 17) Важливі логічні обмеження
- Запити користувача в UI **мають бути англійською** (backend перевіряє кирилицю).
- Відповідь LLM обмежується контекстом; якщо немає підтвердження — повертає “I don’t know.”
- Для image retrieval потрібна **asset collection** та embeddings.

---

## 18) Зв’язки між файлами (коротке резюме)
- `ingest/*` → створює артефакти у `data/processed/artifacts/*`.
- `index/indexer.py` → читає `chunks.jsonl` → пише в `Qdrant`.
- `index/assets_indexer.py` → читає `assets_manifest.json` → пише в `Qdrant`.
- `rag/*` → працює поверх Qdrant + manifest.
- `ui/backend/*` → викликає rag‑pipeline, зберігає історію в SQLite.
- `ui/web/*` → показує відповідь + attachments.

---

Якщо потрібно, можу доповнити документ конкретними прикладами запуску або деталізувати певний модуль.
