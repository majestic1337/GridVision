import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config import constants as const
from config.runtime import get_runtime_config


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    db_path: Path
    qdrant_url: str
    qdrant_api_key: Optional[str]
    collection: str
    asset_collection: str
    dense_vector: str
    sparse_vector: str
    dense_model: str
    sparse_model: str
    image_model: str
    rerank_model: str
    use_rerank: bool
    wide_k: int
    top_k: int
    rrf_k: int
    rerank_top_n: int
    context_budget: int
    assets_manifest: Path
    assets_root: Optional[Path]
    max_assets: int
    image_top_k: int
    image_min_score: float
    history_limit: int
    cors_origins: list[str]
    gemini_api_key: str
    gemini_model: str
    temperature: float


def _split_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def get_settings() -> Settings:
    runtime = get_runtime_config()
    root_dir = ROOT
    db_path = Path(os.getenv("GV_CHAT_DB", "data/processed/chat.db"))
    if not db_path.is_absolute():
        db_path = root_dir / db_path

    assets_manifest = Path(os.getenv("GV_ASSETS_MANIFEST", "data/processed/artifacts/assets_manifest.json"))
    if not assets_manifest.is_absolute():
        assets_manifest = root_dir / assets_manifest

    assets_root = os.getenv("GV_ASSETS_ROOT", "").strip()
    assets_root_path = Path(assets_root) if assets_root else None
    if assets_root_path and not assets_root_path.is_absolute():
        assets_root_path = root_dir / assets_root_path

    cors_origins = _split_csv(os.getenv("GV_CORS_ORIGINS", "http://localhost:3000"))

    return Settings(
        root_dir=root_dir,
        db_path=db_path,
        qdrant_url=os.getenv("QDRANT_URL", const.DEFAULT_QDRANT_URL),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        collection=os.getenv(const.ENV_COLLECTION, runtime.collection),
        asset_collection=os.getenv(const.ENV_ASSET_COLLECTION, const.DEFAULT_ASSET_COLLECTION),
        dense_vector=os.getenv(const.ENV_DENSE_VECTOR, runtime.dense_vector),
        sparse_vector=os.getenv(const.ENV_SPARSE_VECTOR, runtime.sparse_vector),
        dense_model=os.getenv("GV_DENSE_MODEL", const.DEFAULT_DENSE_MODEL),
        sparse_model=os.getenv("GV_SPARSE_MODEL", const.DEFAULT_SPARSE_MODEL),
        image_model=os.getenv(const.ENV_IMAGE_MODEL, const.DEFAULT_IMAGE_MODEL),
        rerank_model=os.getenv("GV_RERANK_MODEL", const.DEFAULT_RERANK_MODEL),
        use_rerank=os.getenv("GV_USE_RERANK", "1").lower() not in ("0", "false", "no"),
        wide_k=int(os.getenv("GV_WIDEK", str(const.DEFAULT_WIDE_K))),
        top_k=int(os.getenv("GV_TOPK", str(const.DEFAULT_TOP_K))),
        rrf_k=int(os.getenv("GV_RRF_K", str(const.DEFAULT_RRF_K))),
        rerank_top_n=int(os.getenv("GV_RERANK_TOPN", str(const.DEFAULT_RERANK_TOP_N))),
        context_budget=int(os.getenv("GV_CTX", str(const.DEFAULT_CONTEXT_BUDGET))),
        assets_manifest=assets_manifest,
        assets_root=assets_root_path,
        max_assets=int(os.getenv("GV_MAX_ASSETS", str(const.DEFAULT_MAX_ASSETS))),
        image_top_k=int(os.getenv("GV_IMAGE_TOPK", str(const.DEFAULT_IMAGE_TOP_K))),
        image_min_score=float(os.getenv(const.ENV_IMAGE_MIN_SCORE, str(const.DEFAULT_IMAGE_MIN_SCORE))),
        history_limit=int(os.getenv("GV_CHAT_HISTORY", "8")),
        cors_origins=cors_origins,
        gemini_api_key=os.getenv("GOOGLE_API_KEY", ""),
        gemini_model=os.getenv("GV_GEMINI_MODEL", const.DEFAULT_GEMINI_MODEL),
        temperature=float(os.getenv("GV_TEMP", str(const.DEFAULT_TEMPERATURE))),
    )
