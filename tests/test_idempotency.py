import json
from pathlib import Path

from ingest.processor import IngestionProcessor

from test_ingest_pipeline import DummyEnricher, DummyParser, _write_config


def _configure_processor(repo_tmp_path: Path) -> IngestionProcessor:
    processor = IngestionProcessor(
        config_path=str(_write_config(repo_tmp_path)),
        skip_existing=False,
    )
    processor.parser = DummyParser()
    processor.image_enricher = DummyEnricher()

    def fake_save_asset(raw_item, dest_path: Path, asset_type: str, parent_doc=None, zoom=1.0):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(asset_type, encoding="utf-8")
        return True

    processor.asset_manager._save_asset = fake_save_asset
    return processor


def _stable_chunk_lines(path: Path):
    """Remove generation/run timestamps while comparing published chunk data."""
    stable = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        record.pop("generation_id", None)
        record.get("provenance", {}).pop("processed_at", None)
        stable.append(json.dumps(record, sort_keys=True))
    return stable


def test_reingest_replaces_active_generation_without_duplicate_chunks(repo_tmp_path):
    processor = _configure_processor(repo_tmp_path)
    raw_file = Path(processor.config["paths"]["raw_data"]) / "dummy.pdf"
    raw_file.write_bytes(b"%PDF-1.4")

    assert processor.process_document("dummy.pdf") is True
    artifact_dir = Path(processor.config["paths"]["processed_artifacts"])
    first_lines = _stable_chunk_lines(artifact_dir / "chunks.jsonl")

    assert processor.process_document("dummy.pdf") is True
    second_lines = _stable_chunk_lines(artifact_dir / "chunks.jsonl")

    assert second_lines == first_lines
    assert len(second_lines) == len(set(second_lines))
    assert len((artifact_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()) == len(first_lines)

    generations = list((artifact_dir.parent / "docs" / "doc123").glob("v*/"))
    assert {generation.name for generation in generations} == {"v1", "v2"}
