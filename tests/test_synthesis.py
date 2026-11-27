from pathlib import Path

import pytest

from ingest.parsing import DocumentParser
from ingest.profiler import PageProfile
from ingest.synthesizer import OCRText, TiledPageSynthesizer
from ingest.chunking import TableChunker


class FakePixmap:
    def tobytes(self, _format):
        return b"png-bytes"


class FakePage:
    def __init__(self):
        self.matrix = None

    def get_pixmap(self, matrix=None, alpha=False):
        self.matrix = matrix
        return FakePixmap()


def test_tiled_synthesizer_renders_at_configured_dpi():
    page = FakePage()
    synthesizer = TiledPageSynthesizer(dpi=200)
    synthesizer.extract_text = lambda _image: OCRText("recovered text", 0.91)

    result = synthesizer.synthesize(page, page_number=7)

    assert result.page_number == 7
    assert result.png_bytes == b"png-bytes"
    assert result.ocr.text == "recovered text"
    assert page.matrix is not None


def test_table_chunker_reinjects_header_on_continuation_chunks():
    table = "\n".join(
        [
            "| Part | Description |",
            "| --- | --- |",
            "| P-1 | Fuel pump |",
            "| P-2 | Control relay |",
            "| P-3 | Generator |",
        ]
    )

    chunks = TableChunker(max_chunk_size=70).chunk(table)

    assert len(chunks) >= 2
    assert all("| Part | Description |" in chunk for chunk in chunks)
    assert all("| --- | --- |" in chunk for chunk in chunks)
    assert "| P-1 | Fuel pump |" in chunks[0]


def test_tiled_page_suppresses_strip_image_elements():
    parser = DocumentParser(
        config={
            "parser": {"strategy": "auto", "synthesis_enabled": True},
            "profiling": {"enabled": True},
        }
    )
    profile = PageProfile(
        page_number=7,
        page_idx=6,
        page_width=616,
        page_height=793,
        native_char_count=0,
        raster_coverage=0.80,
        drawing_count=0,
        image_count=23,
        thumbnail_variance=0.0,
        classification="tiled_scan",
    )
    parser.profile_document = lambda _path: [profile]
    parser.extract_native_items = lambda *_args: []
    parser.extract_synthesized_items = lambda *_args: [
        ({"kind": "tiled_ocr", "page_no": 7, "text": "recovered text", "ocr_confidence": 0.9}, "text")
    ]
    parser.extract_pymupdf_images = lambda _path: [
        ({"kind": "pymupdf_image", "page_no": 7, "rect": [0, 0, 1, 1]}, "image")
        for _ in range(23)
    ]
    parser._convert_docling = lambda _path: (_ for _ in ()).throw(AssertionError("Docling should not run"))

    elements = list(parser.parse_file(Path("unused.pdf"), "doc"))

    assert len(elements) == 1
    assert elements[0].metadata["source"] == "tiled_synthesis"


@pytest.mark.skipif(
    not Path("data/raw_pdfs/TM-5-6115-465-12.pdf").exists(),
    reason="TM fixture is not available",
)
def test_tm_page_7_is_classified_as_tiled_scan():
    parser = DocumentParser(
        config={
            "parser": {"strategy": "auto", "ocr_dpi": 200},
            "profiling": {"enabled": True},
        }
    )
    profiles = parser.profile_document(Path("data/raw_pdfs/TM-5-6115-465-12.pdf"))
    page = profiles[6]

    assert page.page_number == 7
    assert page.image_count >= 20
    assert page.classification == "tiled_scan"
    assert page.raster_coverage > 0.70
