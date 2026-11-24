import logging

from ingest.foundation import PipelineFoundation


def test_ingest_child_loggers_write_to_run_log(repo_tmp_path):
    config_path = repo_tmp_path / "config.yaml"
    logs_dir = repo_tmp_path / "logs"
    config_path.write_text(
        "paths:\n"
        f"  raw_data: {repo_tmp_path / 'raw'}\n"
        f"  processed_assets: {repo_tmp_path / 'assets'}\n"
        f"  processed_artifacts: {repo_tmp_path / 'artifacts'}\n"
        f"  logs: {logs_dir}\n",
        encoding="utf-8",
    )

    foundation = PipelineFoundation(str(config_path))
    foundation.setup_directories()
    logger = foundation.setup_logging()
    logging.getLogger("gridvision.ingest.parser").warning("parser warning")
    logging.getLogger("gridvision.ingest.enrichment").warning("enrichment warning")

    log_text = (logs_dir / f"{foundation.run_id}.log").read_text(encoding="utf-8")
    assert "parser warning" in log_text
    assert "enrichment warning" in log_text
    assert logger.name == "gridvision.ingest"

