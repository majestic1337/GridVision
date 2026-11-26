"""Validated configuration models for the ingestion pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsConfig(ConfigModel):
    base_dir: str = "."
    raw_data: str = "data/raw"
    processed_assets: str = "data/processed/assets"
    processed_docs: str = "data/processed/docs"
    processed_artifacts: str = "data/processed/artifacts"
    logs: str = "data/logs"


class ParserConfig(ConfigModel):
    engine: str = "docling"
    version_check: Optional[str] = None
    ocr_strategy: Optional[Literal["force", "auto", "no_ocr"]] = None
    table_mode: Optional[str] = None
    strategy: Literal["auto", "native", "docling", "ocr"] = "auto"
    ocr_engine: str = "tesseract"
    ocr_language: str = "eng"
    ocr_dpi: int = Field(default=200, gt=0)
    synthesis_enabled: bool = True


class ProfilingConfig(ConfigModel):
    enabled: bool = True
    min_native_chars: int = Field(default=150, ge=0)
    scanned_raster_coverage_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    tiled_slice_count_threshold: int = Field(default=3, ge=1)
    thumbnail_variance_threshold: float = Field(default=1.0, ge=0.0)


class TableProcessingConfig(ConfigModel):
    enabled: bool = True
    detection_mode: Literal["hybrid", "docling", "native"] = "hybrid"
    reinject_headers_on_split: bool = True


class CaptioningConfig(ConfigModel):
    enabled: bool = True
    model: str = "gemini-2.0-flash"
    timeout_seconds: float = Field(default=15, gt=0)
    max_retries: int = Field(default=3, ge=0)
    backoff_base_seconds: float = Field(default=5.0, ge=0)


class ImageProcessingConfig(ConfigModel):
    filter_decorations: bool = True
    min_area_px: int = Field(default=2500, ge=0)
    captioning: CaptioningConfig = Field(default_factory=CaptioningConfig)


class OCRRecoveryConfig(ConfigModel):
    mode: Literal["interactive", "auto", "off"] = "off"
    headless_fallback: Literal["auto", "interactive", "off"] = "auto"


class ChunkingConfig(ConfigModel):
    base_chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    table_max_chunk_size: int = Field(default=1500, gt=0)
    min_paragraph_words: int = Field(default=30, ge=0)
    max_paragraph_chars: int = Field(default=1500, gt=0)
    separators: List[str] = Field(default_factory=lambda: ["\n\n", "\n", ". ", " "])

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkingConfig":
        if self.chunk_overlap >= self.base_chunk_size:
            raise ValueError("chunk_overlap must be smaller than base_chunk_size")
        return self


class QualityGatesConfig(ConfigModel):
    min_score: float = Field(default=0.3, ge=0.0, le=1.0)
    min_word_count: int = Field(default=10, ge=0)
    min_ocr_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    min_text_density: float = Field(default=0.4, ge=0.0, le=1.0)


class EnrichmentConfig(ConfigModel):
    caption_model: str = "gemini-2.0-flash"
    timeout: float = Field(default=15, gt=0)
    max_retries: int = Field(default=3, ge=0)


class ImageEmbeddingsConfig(ConfigModel):
    enabled: bool = False
    model: str = "clip-ViT-B-32"


class IngestConfigModel(ConfigModel):
    project_name: str = "GridVision"
    version: str = "3.0.0"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)
    parser: ParserConfig = Field(default_factory=ParserConfig)
    table_processing: TableProcessingConfig = Field(default_factory=TableProcessingConfig)
    image_processing: ImageProcessingConfig = Field(default_factory=ImageProcessingConfig)
    ocr_recovery: OCRRecoveryConfig = Field(default_factory=OCRRecoveryConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    quality_gates: QualityGatesConfig = Field(default_factory=QualityGatesConfig)
    enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)
    image_embeddings: ImageEmbeddingsConfig = Field(default_factory=ImageEmbeddingsConfig)
    default_tags: List[str] = Field(default_factory=list)

    @field_validator("project_name", "version")
    @classmethod
    def validate_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be empty")
        return value


def validate_config(data: Mapping[str, Any]) -> IngestConfigModel:
    """Validate an already-loaded configuration mapping."""

    return IngestConfigModel.model_validate(dict(data))


def load_config(path: str | Path) -> IngestConfigModel:
    """Load and validate an ingestion YAML configuration file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("ingestion config root must be a mapping")
    return validate_config(data)

