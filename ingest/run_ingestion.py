import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from ingest.processor import IngestionProcessor


if __name__ == "__main__":
    processor = IngestionProcessor(
        config_path="metadata/ingest_config.yaml"
    )

    raw_dir = processor.foundation.root_dir / processor.config["paths"]["raw_data"]
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    pdf_files = sorted(raw_dir.rglob("*.pdf"))
    print(f"[INGEST] Found {len(pdf_files)} PDF(s) under {raw_dir}")

    for idx, file_path in enumerate(pdf_files, 1):
        try:
            rel_path = file_path.relative_to(raw_dir)
            print(f"[INGEST] ({idx}/{len(pdf_files)}) {rel_path}")
            processor.process_document(str(rel_path))
        except Exception as e:
            print(f"[ERROR] {file_path}: {e}")

    processor.finalize()
    print("Ingestion finished")
