from pathlib import Path
import re

# Runtime config defaults + env keys.
DEFAULT_COLLECTION = "gridvision_chunks"
DEFAULT_DENSE_VECTOR = "dense"
DEFAULT_SPARSE_VECTOR = "sparse"

ENV_COLLECTION = "QDRANT_COLLECTION"
ENV_DENSE_VECTOR = "GV_DENSE_VECTOR_NAME"
ENV_SPARSE_VECTOR = "GV_SPARSE_VECTOR_NAME"

# Model defaults.
DEFAULT_DENSE_MODEL = "BAAI/bge-m3"
DEFAULT_SPARSE_MODEL = "prithivida/Splade_PP_en_v1"
DEFAULT_RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"

# Ingestion + parsing.
PARA_Y_GAP = 32.0
HASH_CHUNK_SIZE = 4096
RE_PARA_SPLIT = re.compile(r"\n\s*\n+")

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_MIN_SCORE = 0.3
DEFAULT_MIN_PARAGRAPH_WORDS = 30
DEFAULT_MAX_PARAGRAPH_CHARS = 1500
DEFAULT_ASSET_ZOOM = 2.0
ASSET_ID_HASH_LEN = 12

# Enrichment + quality scoring.
HALLUCINATION_EXACT_NAMES = {"unknown", "component", "n/a", "not visible", ""}
HALLUCINATION_SUBSTRINGS = ("not visible", "n/a")

RE_HYPHEN_BREAK = re.compile(r"(\w+)-\s*\n\s*(\w+)")
RE_WHITESPACE = re.compile(r"\s+")
RE_INNER_WS = re.compile(r"[ \t]+")

BASE_SCORE = 0.5
TECH_TERM_BOOST_PER_MATCH = 0.1
TECH_TERM_MAX_BOOST = 0.4

LOW_WORD_COUNT_THRESHOLD = 10
LOW_WORD_PENALTY = 0.2

LOW_OCR_CONF_THRESHOLD = 0.8
LOW_OCR_PENALTY = 0.3

LOW_TEXT_DENSITY_THRESHOLD = 0.4
LOW_TEXT_DENSITY_PENALTY = 0.2

ILLEGIBLE_PENALTY = 0.4
ILLEGIBLE_PATTERNS = ("not legible", "unreadable", "blur", "unknown")

TARGET_TYPES = ("text", "ocr_text", "table_md", "image_caption")
QUALITY_SCORE_DECIMALS = 2

RE_TECH_TERMS = [
    r"\b\d{3}-\d{4}\b",                  # Standard PN (123-4567)
    r"\b[A-Z]{1,4}-\d+\b",               # Pin IDs (PIN-4, WH-12)
    r"\bTB\s*-?\s*\d+\b",                # Terminal Blocks (TB-1, TB 1)
    r"\b\d{2,}[A-Z]\d{1,}\b",            # Wire Codes (130A16 - common in schematics)
    r"\b[JKP]\d{1,3}\b",                 # Connectors (J1, P2, K16)
    r"\b(?=[A-Z0-9]{5,}\b)(?=.*[A-Z])(?=.*\d)[A-Z0-9]+\b",
]

IMAGE_ENRICH_MAX_RETRIES = 3
RATE_LIMIT_BACKOFF_BASE_SECONDS = 5

# Reporting + validation.
SCORE_BUCKETS = [
    (0.3, "0.0-0.3"),
    (0.5, "0.3-0.5"),
    (0.7, "0.5-0.7"),
    (0.9, "0.7-0.9"),
]
SCORE_BUCKET_ELSE = "0.9-1.0"
ASSET_VALIDATION_SAMPLE_SIZE = 20

# Indexer defaults.
INDEXER_DEFAULT_BATCH_SIZE = 64
INDEXER_DEFAULT_SAVE_EVERY = 500
INDEXER_SCROLL_LIMIT = 256
QDRANT_TIMEOUT_SECONDS = 120
INDEXER_TEST_QUERY = "power supply wiring diagram"
INDEXER_TEST_LIMIT = 5
INDEXER_DEFAULT_CHUNK_FILE = Path("data/processed/artifacts/chunks.jsonl")
INDEXER_DEFAULT_SAVEPOINT_PATH = Path("data/processed/artifacts/index_savepoint.json")

# Retrieval + RAG.
RETRIEVER_TOP_K = 50
RRF_K = 60
RERANK_TOP_N = 5
RAG_CONTEXT_LIMIT_CHARS = 6000
RAG_CONTENT_PREVIEW_CHARS = 100

# UI defaults.
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_WIDE_K = 50
DEFAULT_TOP_K = 50
DEFAULT_RRF_K = RRF_K
DEFAULT_RERANK_TOP_N = RERANK_TOP_N
DEFAULT_TEMPERATURE = 0.2
DEFAULT_CONTEXT_BUDGET = RAG_CONTEXT_LIMIT_CHARS
DEFAULT_MAX_ASSETS = 6

UI_WIDE_K_MIN = 10
UI_WIDE_K_MAX = 200
UI_WIDE_K_STEP = 5
UI_TOP_K_MIN = 3
UI_TOP_K_MAX = 100
UI_TOP_K_STEP = 1
UI_RRF_K_MIN = 10
UI_RRF_K_MAX = 200
UI_RRF_K_STEP = 5

UI_RERANK_TOP_N_MIN = 1
UI_RERANK_TOP_N_MAX = 20
UI_RERANK_TOP_N_STEP = 1

UI_TEMP_MIN = 0.0
UI_TEMP_MAX = 1.0
UI_TEMP_STEP = 0.05

UI_CTX_MIN = 1000
UI_CTX_MAX = 20000
UI_CTX_STEP = 500

UI_MAX_ASSETS_MIN = 0
UI_MAX_ASSETS_MAX = 20
UI_MAX_ASSETS_STEP = 1

UI_QUERY_HEIGHT = 90
UI_CONTEXT_TEXTAREA_HEIGHT = 320
UI_HTML_HEIGHT = 520
UI_COLUMNS_RATIO = (1, 2)
UI_SOURCES_PREVIEW_CHARS = 240
UI_DEFAULT_QUERY = "A summary of the parasitic losses accounted for by System Advisor"
