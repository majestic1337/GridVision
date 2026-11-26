from types import SimpleNamespace

from ingest.adapters.native_adapter import NativePdfAdapter
from ingest.profiler import DocumentProfiler


class FakePage:
    rect = SimpleNamespace(width=100, height=100)

    def __init__(self, text="", drawings=None, image_rects=None):
        self.text = text
        self.drawings = drawings or []
        self.image_rects = image_rects or {}

    def get_text(self, mode):
        if mode == "text":
            return self.text
        if mode == "blocks":
            return [(0, 0, 80, 10, self.text)] if self.text else []
        return ""

    def get_drawings(self):
        return self.drawings

    def get_images(self, full=True):
        return [(xref,) for xref in self.image_rects]

    def get_image_rects(self, xref):
        return [SimpleNamespace(x0=a, y0=b, x1=c, y1=d) for a, b, c, d in self.image_rects[xref]]

    def get_pixmap(self, matrix=None, alpha=False):
        return SimpleNamespace(samples=bytes([0, 0, 0, 255]), n=1)


def test_profiler_classifies_native_pages():
    profile = DocumentProfiler().profile_page(FakePage(text="native text " * 20))

    assert profile.classification == "native"
    assert profile.native_char_count > 150
    assert profile.raster_coverage == 0


def test_profiler_classifies_tiled_raster_pages_at_page_level():
    page = FakePage(
        image_rects={
            1: [(0, 0, 100, 35)],
            2: [(0, 35, 100, 70)],
            3: [(0, 70, 100, 100)],
        }
    )

    profile = DocumentProfiler().profile_page(page)

    assert profile.classification == "tiled_scan"
    assert profile.image_count == 3
    assert profile.raster_coverage == 1.0


def test_profiler_preserves_vector_heavy_pages():
    profile = DocumentProfiler().profile_page(FakePage(drawings=[object()] * 6))

    assert profile.classification == "vector"
    assert profile.drawing_count == 6


def test_native_adapter_extracts_sorted_text_blocks():
    page = FakePage(text="ignored")
    page.get_text = lambda mode: [
        (0, 30, 20, 40, "lower"),
        (0, 10, 20, 20, "upper"),
        (0, 0, 0, 0, "   "),
    ] if mode == "blocks" else "ignored"

    blocks = NativePdfAdapter().extract_page(page)

    assert [block.text for block in blocks] == ["upper", "lower"]
    assert blocks[0].bbox == [0.0, 10.0, 20.0, 20.0]
