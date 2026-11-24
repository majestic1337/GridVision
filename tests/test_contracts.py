import json

import pytest
from pydantic import ValidationError

from ingest.contracts import (
    BoundingBox,
    DocumentRecord,
    ElementType,
    ExtractedAsset,
    ExtractedElement,
    PageRecord,
    ProcessedChunk,
)


def test_ir_models_round_trip_without_data_loss():
    element = ExtractedElement(
        element_id="doc_p001_e001_text",
        doc_id="doc",
        page_number=1,
        page_label="1",
        element_type=ElementType.TEXT,
        content="A normalized paragraph.",
        bbox=BoundingBox(left=0.1, top=0.2, right=0.8, bottom=0.4),
        metadata={"section": "Overview"},
    )
    chunk = ProcessedChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        generation_id="v1",
        element_id=element.element_id,
        chunk_index=0,
        chunk_type="text",
        content=element.content,
        content_hash="abc123",
        page_number=1,
        page_label="1",
        quality_score=0.9,
    )
    asset = ExtractedAsset(
        asset_id="asset-1",
        doc_id="doc",
        generation_id="v1",
        element_id="doc_p001_e002_image",
        asset_type="image_asset",
        file_path="data/processed/docs/doc/v1/assets/a.png",
        page_number=1,
        page_label="1",
        caption="A diagram.",
    )
    payload = {
        "document": DocumentRecord(doc_id="doc", doc_slug="manual"),
        "page": PageRecord(doc_id="doc", page_number=1, page_label="1"),
        "element": element,
        "chunk": chunk,
        "asset": asset,
    }

    encoded = json.dumps({key: value.model_dump(mode="json") for key, value in payload.items()})
    decoded = json.loads(encoded)
    assert ExtractedElement.model_validate(decoded["element"]) == element
    assert ProcessedChunk.model_validate(decoded["chunk"]) == chunk
    assert ExtractedAsset.model_validate(decoded["asset"]) == asset


def test_ir_models_reject_invalid_values():
    with pytest.raises(ValidationError):
        BoundingBox(left=0.8, top=0.0, right=0.2, bottom=1.0)

    with pytest.raises(ValidationError):
        PageRecord(doc_id="doc", page_number=0, page_label="1")

    with pytest.raises(ValidationError):
        ExtractedElement(
            element_id="e",
            doc_id="doc",
            page_number=1,
            page_label="1",
            element_type="unsupported",
        )

