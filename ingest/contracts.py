"""Pydantic contracts for the ingestion intermediate representation.

These models deliberately contain only JSON-safe values. Parser-specific objects
such as ``fitz.Document`` and Docling items must remain outside the IR boundary.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class IRModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ElementType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    HEADER = "header"
    FOOTER = "footer"
    VECTOR_GRAPHIC = "vector_graphic"


class BoundingBox(IRModel):
    """Normalized page coordinates in the range 0..1."""

    left: float
    top: float
    right: float
    bottom: float

    @model_validator(mode="after")
    def validate_order(self) -> "BoundingBox":
        if self.left > self.right or self.top > self.bottom:
            raise ValueError("bounding-box coordinates must be ordered")
        return self

    @field_validator("left", "top", "right", "bottom")
    @classmethod
    def validate_coordinate(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("bounding-box coordinates must be between 0 and 1")
        return value

    def intersects(self, other: "BoundingBox") -> bool:
        return not (
            self.right < other.left
            or self.left > other.right
            or self.bottom < other.top
            or self.top > other.bottom
        )


class DocumentRecord(IRModel):
    doc_id: str = Field(min_length=1)
    doc_slug: str = Field(min_length=1)
    title: Optional[str] = None
    source_uri: Optional[str] = None
    created_at: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    run_id: Optional[str] = None
    ingested_at: Optional[str] = None
    status: str = "completed"
    generation_id: Optional[str] = None
    stage_fingerprints: Dict[str, str] = Field(default_factory=dict)


class PageRecord(IRModel):
    doc_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    page_label: str = Field(min_length=1)
    page_idx: Optional[int] = Field(default=None, ge=0)
    page_kind: str = "unknown"
    native_char_count: int = Field(default=0, ge=0)
    raster_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    drawing_count: int = Field(default=0, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExtractedElement(IRModel):
    element_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    page_label: str = Field(min_length=1)
    element_type: ElementType = Field(
        validation_alias=AliasChoices("element_type", "type")
    )
    content: str = ""
    raw_html: Optional[str] = None
    bbox: Optional[BoundingBox] = None
    reading_order_idx: int = Field(default=0, ge=0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_adapter: str = "native"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessedChunk(IRModel):
    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    generation_id: Optional[str] = None
    element_id: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    chunk_type: str = Field(
        validation_alias=AliasChoices("chunk_type", "type"), min_length=1
    )
    content: str
    content_hash: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    page_label: str = Field(min_length=1)
    section: Optional[str] = None
    linked_asset_id: Optional[str] = None
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance: Dict[str, Any] = Field(default_factory=dict)


class ExtractedAsset(IRModel):
    asset_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    generation_id: Optional[str] = None
    element_id: str = Field(min_length=1)
    asset_type: str = Field(
        validation_alias=AliasChoices("asset_type", "type"), min_length=1
    )
    file_path: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    page_label: str = Field(min_length=1)
    bbox: Optional[List[float]] = None
    caption: Optional[str] = None
    caption_json: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        if value is not None and len(value) != 4:
            raise ValueError("asset bbox must contain four coordinates")
        return value

