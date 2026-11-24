import pytest
from pydantic import ValidationError

from ingest.config_schema import IngestConfigModel, load_config, validate_config


def test_current_ingest_config_validates(repo_tmp_path):
    config = load_config("metadata/ingest_config.yaml")
    assert isinstance(config, IngestConfigModel)
    assert config.parser.ocr_strategy == "auto"
    assert config.chunking.base_chunk_size == 1000


def test_config_defaults_support_new_schema():
    config = validate_config({})
    assert config.paths.processed_docs == "data/processed/docs"
    assert config.profiling.enabled is True
    assert config.parser.strategy == "auto"


def test_invalid_config_is_rejected():
    with pytest.raises(ValidationError):
        validate_config({"chunking": {"base_chunk_size": 100, "chunk_overlap": 100}})

    with pytest.raises(ValidationError):
        validate_config({"quality_gates": {"min_score": 1.5}})

