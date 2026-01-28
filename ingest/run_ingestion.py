import argparse
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from ingest.processor import IngestionProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GridVision ingestion over PDFs.")
    parser.add_argument(
        "--config",
        default=os.getenv("GV_INGEST_CONFIG", "metadata/ingest_config.yaml"),
        help="Path to ingest config YAML (default: metadata/ingest_config.yaml or GV_INGEST_CONFIG).",
    )
    parser.add_argument(
        "--raw-dir",
        default=os.getenv("GV_INGEST_RAW_DIR"),
        help="Override raw data directory (default: config paths.raw_data).",
    )
    parser.add_argument(
        "--pattern",
        default=os.getenv("GV_INGEST_PATTERN", "*.pdf"),
        help="Glob pattern for input files (default: *.pdf).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing artifacts before ingest (start fresh).",
    )
    parser.add_argument(
        "--reingest",
        action="store_true",
        help="Process PDFs even if already ingested (may duplicate entries).",
    )
    return parser.parse_args()


def _display_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    args = parse_args()
    processor = IngestionProcessor(
        config_path=args.config,
        reset_artifacts=args.reset,
        skip_existing=not args.reingest,
    )
    logger = processor.get_logger()

    raw_dir = Path(args.raw_dir) if args.raw_dir else processor.raw_data_dir()
    if not raw_dir.exists():
        logger.error("Raw data directory not found: %s", raw_dir)
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    pdf_files = sorted(raw_dir.rglob(args.pattern))
    logger.info("[INGEST] Found %s file(s) under %s", len(pdf_files), raw_dir)

    for idx, file_path in enumerate(pdf_files, 1):
        logger.info("[INGEST] (%s/%s) %s", idx, len(pdf_files), _display_path(file_path, raw_dir))
        try:
            processor.process_document(file_path)
        except Exception:
            logger.exception("[ERROR] %s", file_path)

    processor.finalize()
    logger.info("Ingestion finished")
