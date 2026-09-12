import hashlib
import json
from pathlib import Path

from ingest.parsing import ParsedElement
from ingest.processor import IngestionProcessor
from config.constants import ASSET_ID_HASH_LEN, DEFAULT_ASSET_ZOOM


def _write_config(tmp_root: Path) -> Path:
    raw_dir = tmp_root / "raw"
    assets_dir = tmp_root / "assets"
    artifacts_dir = tmp_root / "artifacts"
    logs_dir = tmp_root / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    config_path = tmp_root / "ingest_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                'project_name: "Test"',
                'version: "test"',
                "paths:",
                f'  raw_data: "{raw_dir}"',
                f'  processed_assets: "{assets_dir}"',
                f'  processed_artifacts: "{artifacts_dir}"',
                f'  logs: "{logs_dir}"',
                "chunking:",
                "  base_chunk_size: 512",
                "  chunk_overlap: 0",
                '  separators: ["\\n\\n", "\\n", ". ", " "]',
                "quality_gates:",
                "  min_score: 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path


class DummyTable:
    def export_to_markdown(self, doc=None) -> str:
        return "|a|b|\n|1|2|"


class DummyParser:
    def generate_doc_id(self, file_path: Path):
        return "doc123", "doc_slug"

    def parse_file(self, file_path: Path, doc_id: str):
        yield ParsedElement(
            element_id="doc123_p0_01_text",
            type="text",
            content="Short text content.",
            page_idx=0,
            page_number=1,
            page_label="1",
            bbox=None,
            metadata={"raw_item": None, "parent_doc": None},
        )
        yield ParsedElement(
            element_id="doc123_p0_02_table",
            type="table",
            content="<TABLE_PLACEHOLDER>",
            page_idx=0,
            page_number=1,
            page_label="1",
            bbox=None,
            metadata={"raw_item": DummyTable(), "parent_doc": None},
        )
        yield ParsedElement(
            element_id="doc123_p0_03_image",
            type="image",
            content="",
            page_idx=0,
            page_number=1,
            page_label="1",
            bbox=None,
            metadata={"raw_item": {"kind": "fake"}, "parent_doc": None},
        )


class DummyEnricher:
    def enrich_image(self, image_path: Path):
        return "Image caption for test.", {"component_name": "Test"}


def _asset_id_for(element_id: str) -> str:
    return f"asset_{hashlib.md5(element_id.encode()).hexdigest()[:ASSET_ID_HASH_LEN]}"


def test_init_artifacts_resets_files(repo_tmp_path):
    config_path = _write_config(repo_tmp_path)
    proc = IngestionProcessor(config_path=str(config_path))
    artifacts_dir = Path(proc.config["paths"]["processed_artifacts"])

    (artifacts_dir / "chunks.jsonl").write_text("stale", encoding="utf-8")
    (artifacts_dir / "assets_manifest.json").write_text("stale", encoding="utf-8")
    (artifacts_dir / "element_to_chunks.json").write_text("stale", encoding="utf-8")
    (artifacts_dir / "page_to_assets.json").write_text("stale", encoding="utf-8")

    proc.artifacts.initialize(reset=True)

    assert (artifacts_dir / "chunks.jsonl").read_text(encoding="utf-8") == ""
    assert json.loads((artifacts_dir / "assets_manifest.json").read_text(encoding="utf-8")) == []
    assert json.loads((artifacts_dir / "element_to_chunks.json").read_text(encoding="utf-8")) == {}
    assert json.loads((artifacts_dir / "page_to_assets.json").read_text(encoding="utf-8")) == {}


def test_process_document_writes_chunks_and_assets(repo_tmp_path):
    config_path = _write_config(repo_tmp_path)
    proc = IngestionProcessor(config_path=str(config_path))
    proc.parser = DummyParser()
    proc.image_enricher = DummyEnricher()

    def fake_save_asset(
        raw_item,
        dest_path: Path,
        asset_type: str,
        parent_doc=None,
        zoom: float = DEFAULT_ASSET_ZOOM,
    ) -> bool:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(asset_type, encoding="utf-8")
        return True

    proc.asset_manager._save_asset = fake_save_asset

    raw_dir = Path(proc.config["paths"]["raw_data"])
    raw_file = raw_dir / "dummy.pdf"
    raw_file.write_bytes(b"%PDF-1.4")

    proc.process_document("dummy.pdf")

    artifacts_dir = Path(proc.config["paths"]["processed_artifacts"])
    lines = (artifacts_dir / "chunks.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    records = [json.loads(line) for line in lines]
    types = {rec["type"] for rec in records}
    assert "text" in types
    assert "table_md" in types
    assert "image_caption" in types

    manifest = json.loads((artifacts_dir / "assets_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 2
    manifest_ids = {item["asset_id"] for item in manifest}
    expected_ids = {
        _asset_id_for("doc123_p0_02_table"),
        _asset_id_for("doc123_p0_03_image"),
    }
    assert manifest_ids == expected_ids

    root_dir = proc.foundation.root_dir
    for item in manifest:
        asset_path = root_dir / item["file_path"]
        assert asset_path.exists()

    element_to_chunks = json.loads((artifacts_dir / "element_to_chunks.json").read_text(encoding="utf-8"))
    assert "doc123_p0_01_text" in element_to_chunks
    assert "doc123_p0_02_table" in element_to_chunks
    assert "doc123_p0_03_image" in element_to_chunks

    page_to_assets = json.loads((artifacts_dir / "page_to_assets.json").read_text(encoding="utf-8"))
    assert "doc123_1" in page_to_assets
    assert set(page_to_assets["doc123_1"]) == expected_ids
