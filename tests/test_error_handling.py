from pathlib import Path

import pytest

from ingest.enrichment import ImageEnricher
from ingest.parsing import DocumentParser, DocumentParsingError


def test_docling_conversion_raises_typed_error(monkeypatch, tmp_path):
    parser = DocumentParser.__new__(DocumentParser)
    parser.converter = object()

    class BrokenConverter:
        def convert(self, path):
            raise ValueError("broken PDF")

    parser.converter = BrokenConverter()
    monkeypatch.setattr("ingest.parsing.DOCLING_AVAILABLE", True)

    with pytest.raises(DocumentParsingError, match="Docling conversion failed"):
        parser._convert_docling(tmp_path / "broken.pdf")


def test_missing_docling_raises_typed_error(monkeypatch, tmp_path):
    parser = DocumentParser.__new__(DocumentParser)
    parser.converter = None
    monkeypatch.setattr("ingest.parsing.DOCLING_AVAILABLE", False)

    with pytest.raises(DocumentParsingError, match="unavailable"):
        parser._convert_docling(tmp_path / "broken.pdf")


def test_quality_path_can_read_ocr_confidence_from_metadata():
    assert DocumentParser is not None
    from ingest.processor import IngestionProcessor

    assert IngestionProcessor._get_ocr_confidence({"ocr_confidence": 0.62}) == 0.62
    assert IngestionProcessor._get_ocr_confidence({"ocr_conf": "0.71"}) == 0.71
    assert IngestionProcessor._get_ocr_confidence({"ocr_confidence": "bad"}) == 1.0


def test_transient_caption_errors_are_retried(monkeypatch, tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"not-used-by-mock")
    enricher = ImageEnricher.__new__(ImageEnricher)
    enricher.max_retries = 3
    enricher.backoff_base_seconds = 0
    monkeypatch.setattr(enricher, "_build_image_prompt", lambda: "prompt")

    attempts = []

    class ServiceUnavailable(Exception):
        pass

    monkeypatch.setattr(
        "ingest.enrichment.google_exceptions.ServiceUnavailable",
        ServiceUnavailable,
        raising=False,
    )

    class FakeModel:
        def generate_content(self, _payload):
            attempts.append(1)
            if len(attempts) < 3:
                raise ServiceUnavailable("temporary 503")
            return type("Response", (), {"text": "{}"})()

    enricher.model = FakeModel()

    def generate_content(_payload):
        attempts.append(1)
        if len(attempts) < 3:
            raise ServiceUnavailable("temporary 503")
        return type("Response", (), {"text": "{}"})()

    enricher.model.generate_content = generate_content
    class FakeImage:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr("ingest.enrichment.Image.open", lambda _path: FakeImage())
    monkeypatch.setattr("ingest.enrichment.time.sleep", lambda _seconds: None)

    # The mocked response is enough to verify retry behavior; parsing is not
    # part of this unit test.
    assert enricher._call_model_with_retry(image_path) == "{}"
    assert len(attempts) == 3

