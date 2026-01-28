import os
import logging
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime

from config.constants import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_PARAGRAPH_CHARS,
    DEFAULT_MIN_PARAGRAPH_WORDS,
    DEFAULT_MIN_SCORE,
    LOW_OCR_CONF_THRESHOLD,
    LOW_TEXT_DENSITY_THRESHOLD,
    LOW_WORD_COUNT_THRESHOLD,
    RE_PARA_SPLIT,
)
try:
    from fastembed import SparseTextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .reporting import IngestionReporter
from .parsing import DocumentParser
from .enrichment import TextNormalizer, QualityScorer, ImageEnricher
from .foundation import PipelineFoundation
from .storage import ArtifactStore, AssetManager, ChunkRegistry

logger = logging.getLogger("GridVision_Processor")

class SparseEncoder:
    def __init__(self, model_name: str = "prithivida/Splade_PP_en_v1"):
        self.model = None
        if FASTEMBED_AVAILABLE:
            try:
                logger.info(f"Loading Sparse Model: {model_name}...")
                self.model = SparseTextEmbedding(model_name=model_name)
            except Exception as e:
                logger.error(f"Failed to load Sparse Model: {e}")
        else:
            logger.warning("FastEmbed not installed. Sparse vectors will be empty.")

    def encode(self, text: str) -> Dict[str, List[Union[int, float]]]:
        if not self.model or not text:
            return {"indices": [], "values": []}
        
        try:
            vector = list(self.model.embed([text]))[0]
            return {
                "indices": vector.indices.tolist(),
                "values": vector.values.tolist()
            }
        except Exception as e:
            logger.error(f"Sparse encoding error: {e}")
            return {"indices": [], "values": []}

class IngestionProcessor:
    def __init__(
        self,
        config_path: Optional[str] = None,
        *,
        reset_artifacts: bool = False,
        skip_existing: bool = True,
    ):
        if config_path is None:
            config_path = os.getenv("GV_INGEST_CONFIG", "metadata/ingest_config.yaml")
        self.foundation = PipelineFoundation(config_path)
        self.config = self.foundation.config
        
        self.foundation.setup_directories()
        self.logger = self.foundation.setup_logging()

        artifact_dir = self.foundation.root_dir / self.config["paths"]["processed_artifacts"]
        self.artifacts = ArtifactStore(
            root_dir=self.foundation.root_dir,
            artifact_dir=artifact_dir,
            logger=self.logger,
        )
        self.artifacts.initialize(reset=reset_artifacts)
        self.skip_existing = skip_existing
        self.ingested_doc_ids = self.artifacts.load_ingested_doc_ids()
        if self.skip_existing and self.ingested_doc_ids:
            self.logger.info(
                "[INGEST] Loaded %s ingested document(s); will skip duplicates.",
                len(self.ingested_doc_ids),
            )

        self.parser = DocumentParser()
        self.image_enricher = ImageEnricher()
        self.sparse_encoder = SparseEncoder()
        self.asset_manager = AssetManager(
            root_dir=self.foundation.root_dir,
            asset_root=self.foundation.root_dir / self.config["paths"]["processed_assets"],
            logger=self.logger,
        )
        
        chunk_conf = self.config.get("chunking", {})
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_conf.get("base_chunk_size", DEFAULT_CHUNK_SIZE),
            chunk_overlap=chunk_conf.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
            separators=chunk_conf.get("separators", ["\n\n", "\n", ". ", " "])
        )
        
        quality_gates = self.config.get("quality_gates", {})
        min_score = quality_gates.get("min_score", DEFAULT_MIN_SCORE)
        self.reporter = IngestionReporter(
            run_id=self.foundation.run_id, 
            artifact_dir=artifact_dir,
            root_dir=self.foundation.root_dir,
            min_score=min_score
        )

        self.chunk_registry = ChunkRegistry()


    def get_logger(self) -> logging.Logger:
        return self.logger

    def raw_data_dir(self) -> Path:
        paths = self.config.get("paths", {})
        raw_root = paths.get("raw_data", "data/raw")
        return self.foundation.root_dir / raw_root

    def resolve_raw_path(self, filename: Union[str, Path]) -> Path:
        raw_path = Path(filename)
        if not raw_path.is_absolute():
            raw_path = self.raw_data_dir() / raw_path
        return raw_path

    def process_document(self, filename: Union[str, Path]):
        raw_path = self.resolve_raw_path(filename)
        if not raw_path.exists():
            self.logger.error(f"File not found: {raw_path}")
            return False

        start_ts = time.monotonic()

        doc_hash, doc_slug = self.parser.generate_doc_id(raw_path)
        try:
            source_uri = str(raw_path.relative_to(self.foundation.root_dir))
        except ValueError:
            source_uri = str(raw_path)

        if self.skip_existing and doc_hash in self.ingested_doc_ids:
            self.logger.info(
                "[DOC] Skipping already ingested doc_id=%s source=%s",
                doc_hash,
                source_uri,
            )
            return False

        doc_title = raw_path.stem.replace("_", " ").strip() or doc_slug
        created_at = datetime.utcfromtimestamp(raw_path.stat().st_mtime).replace(microsecond=0).isoformat() + "Z"
        tags = self.config.get("default_tags") or []
        if not isinstance(tags, list):
            tags = [str(tags)]

        quality_gates = self.config.get("quality_gates", {})
        min_score = quality_gates.get("min_score", DEFAULT_MIN_SCORE)
        min_word_count = quality_gates.get("min_word_count", LOW_WORD_COUNT_THRESHOLD)
        min_ocr_confidence = quality_gates.get("min_ocr_confidence", LOW_OCR_CONF_THRESHOLD)
        min_text_density = quality_gates.get("min_text_density", LOW_TEXT_DENSITY_THRESHOLD)

        progress_verbose = os.getenv("GV_INGEST_VERBOSE", "1").lower() not in ("0", "false", "no")
        log_every_n = os.getenv("GV_INGEST_LOG_EVERY")
        if log_every_n is None:
            log_every_n = 1 if progress_verbose else 100
        else:
            try:
                log_every_n = int(log_every_n)
            except ValueError:
                log_every_n = 1
        if log_every_n < 1:
            log_every_n = 1

        self.asset_manager.start_document(doc_hash)

        self.logger.info(f"Processing {doc_slug} ({doc_hash})...")
        self.logger.info(f"[DOC] Source: {source_uri}")
        self.logger.info(
            "[DOC] Chunking size=%s overlap=%s min_para_words=%s max_para_chars=%s",
            self.config.get("chunking", {}).get("base_chunk_size", DEFAULT_CHUNK_SIZE),
            self.config.get("chunking", {}).get("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
            self.config.get("chunking", {}).get("min_paragraph_words", DEFAULT_MIN_PARAGRAPH_WORDS),
            self.config.get("chunking", {}).get("max_paragraph_chars", DEFAULT_MAX_PARAGRAPH_CHARS),
        )
        self.logger.info(
            "[DOC] Quality gates min_score=%s min_word_count=%s min_ocr_confidence=%s min_text_density=%s",
            min_score,
            min_word_count,
            min_ocr_confidence,
            min_text_density,
        )
        self.logger.info(f"Starting Docling conversion for {filename}...")
        self.logger.info("Please wait. Reading PDF structure & running OCR (this may take 1-3 mins)...")

        chunks_buffer = []
        page_total = self.asset_manager.open_pdf(raw_path)

        element_idx = 0
        element_counts = {"text": 0, "table": 0, "image": 0}
        chunk_total = 0
        asset_total = 0
        page_count = 0
        current_page_label = None
        page_element_count = 0
        page_chunk_count = 0
        page_asset_count = 0

        def log_page_summary() -> None:
            if current_page_label is None:
                return
            total_label = page_total if page_total is not None else "?"
            self.logger.info(
                "[PAGE] %s/%s done: elements=%s chunks=%s assets=%s",
                current_page_label,
                total_label,
                page_element_count,
                page_chunk_count,
                page_asset_count,
            )

        try:
            for element in self.parser.parse_file(raw_path, doc_hash):
                if element.page_label != current_page_label:
                    log_page_summary()
                    current_page_label = element.page_label
                    page_count += 1
                    page_element_count = 0
                    page_chunk_count = 0
                    page_asset_count = 0
                    total_label = page_total if page_total is not None else "?"
                    self.logger.info("[PAGE] %s/%s start", current_page_label, total_label)

                element_idx += 1
                element_counts[element.type] = element_counts.get(element.type, 0) + 1
                page_element_count += 1
                if element_idx % log_every_n == 0:
                    self.logger.info(
                        "[EL] #%s id=%s type=%s page=%s",
                        element_idx,
                        element.element_id,
                        element.type,
                        element.page_label,
                    )
                if element.metadata is None:
                    element.metadata = {}
                metadata = element.metadata
                parent_doc = metadata.get("parent_doc")
                linked_asset_id = None
                asset_path = None
                if element.type in ["image", "table"]:
                    linked_asset_id, asset_path = self.asset_manager.register_element(element)
                    if linked_asset_id:
                        asset_total += 1
                        page_asset_count += 1
                final_content = ""
                searchable_type = element.type

                if element.type == "text":
                    final_content = TextNormalizer.normalize_keep_paragraphs(element.content)

                elif element.type == "table":
                    raw_table = metadata.get("raw_table") or metadata.get("raw_item")
                    if hasattr(raw_table, "export_to_markdown"):
                        try:
                            md_text = raw_table.export_to_markdown(doc=parent_doc)
                        except TypeError:
                            md_text = raw_table.export_to_markdown()
                        final_content = TextNormalizer.normalize(md_text)
                        searchable_type = "table_md"

                elif element.type == "image":
                    if linked_asset_id and asset_path and asset_path.exists():
                        caption, json_data = self.image_enricher.enrich_image(asset_path)
                        if caption:
                            final_content = TextNormalizer.normalize(caption)
                            searchable_type = "image_caption"
                            metadata["caption_json"] = json_data

                if not final_content:
                    if progress_verbose:
                        self.logger.info(
                            "[SKIP] element=%s type=%s reason=no_content",
                            element.element_id,
                            element.type,
                        )
                    continue

                chunk_conf = self.config.get("chunking", {})
                min_para_words = chunk_conf.get("min_paragraph_words", DEFAULT_MIN_PARAGRAPH_WORDS)
                max_para_chars = chunk_conf.get("max_paragraph_chars", DEFAULT_MAX_PARAGRAPH_CHARS)
                base_chunk_size = chunk_conf.get("base_chunk_size", DEFAULT_CHUNK_SIZE)

                if searchable_type in ("text", "table_md", "ocr_text"):
                    prelim = paragraph_chunk_text(
                        final_content,
                        min_words=min_para_words,
                        max_chars=max_para_chars,
                    )
                    text_chunks = []
                    for ch in prelim:
                        if len(ch) > base_chunk_size:
                            text_chunks.extend(self.splitter.split_text(ch))
                        else:
                            text_chunks.append(ch)
                else:
                    text_chunks = [final_content]

                if progress_verbose:
                    self.logger.info(
                        "[CHUNK] element=%s type=%s chunks=%s",
                        element.element_id,
                        searchable_type,
                        len(text_chunks),
                    )

                for i, chunk_text in enumerate(text_chunks):
                    content_hash = TextNormalizer.compute_hash(chunk_text)
                    q_score, q_reasons = QualityScorer.evaluate(
                        chunk_text,
                        searchable_type,
                        min_word_count=min_word_count,
                        min_ocr_confidence=min_ocr_confidence,
                        min_text_density=min_text_density,
                    )

                    if q_score < min_score:
                        if progress_verbose:
                            self.logger.info(
                                "[SKIP] element=%s chunk=%s score=%s < min_score=%s",
                                element.element_id,
                                i,
                                q_score,
                                min_score,
                            )
                        continue

                    sparse_vec = self.sparse_encoder.encode(chunk_text)

                    chunk_seed = f"{element.element_id}_{i}_{content_hash}"
                    chunk_uuid = hashlib.md5(chunk_seed.encode()).hexdigest()

                    chunk_record = {
                        "chunk_id": chunk_uuid,
                        "chunk_index": i,
                        "element_id": element.element_id,
                        "doc_id": doc_hash,
                        "doc_slug": doc_slug,
                        "title": doc_title,
                        "source_uri": source_uri,
                        "created_at": created_at,
                        "tags": tags,
                        "type": searchable_type,
                        "content": chunk_text,
                        "content_hash": content_hash,
                        "sparse_vector": sparse_vec,
                        "metadata": {
                            "page_idx": element.page_idx,
                            "page_number": element.page_number,
                            "page_label": element.page_label,
                            "section": metadata.get("section") or metadata.get("subtype"),
                            "linked_asset_id": linked_asset_id,
                            "caption_json": metadata.get("caption_json"),
                            "quality_score": q_score,
                            "quality_reasons": q_reasons
                        },
                        "provenance": {
                            "processed_at": datetime.utcnow().isoformat(),
                            "pipeline_v": self.config["version"]
                        }
                    }

                    chunks_buffer.append(chunk_record)
                    chunk_total += 1
                    page_chunk_count += 1

                    if progress_verbose:
                        self.logger.info(
                            "[CHUNK] element=%s chunk=%s score=%s size=%s",
                            element.element_id,
                            i,
                            q_score,
                            len(chunk_text),
                        )

                    self.reporter.log_chunk_quality(score=q_score, chunk_type=searchable_type)

                    self.chunk_registry.record(element.element_id, chunk_uuid)

        finally:
            self.asset_manager.close_pdf()
        log_page_summary()
        assets_manifest = self.asset_manager.current_assets()
        self.artifacts.write_chunks(chunks_buffer)
        self.artifacts.write_assets_manifest(assets_manifest)
        self.artifacts.write_maps(
            self.chunk_registry.element_to_chunks,
            self.asset_manager.page_to_assets,
        )

        max_page = 0
        if chunks_buffer:
            max_page = max([ch["metadata"]["page_number"] for ch in chunks_buffer])

        self.reporter.log_document(doc_slug, max_page, len(assets_manifest))
        elapsed_s = time.monotonic() - start_ts
        self.logger.info(
            "[DOC] Finished %s. Pages=%s Elements=%s (text=%s table=%s image=%s) Chunks=%s Assets=%s Elapsed=%.1fs",
            doc_slug,
            page_count,
            element_idx,
            element_counts.get("text", 0),
            element_counts.get("table", 0),
            element_counts.get("image", 0),
            chunk_total,
            asset_total,
            elapsed_s,
        )
        if doc_hash not in self.ingested_doc_ids:
            self.artifacts.append_document_record(
                {
                    "doc_id": doc_hash,
                    "doc_slug": doc_slug,
                    "title": doc_title,
                    "source_uri": source_uri,
                    "created_at": created_at,
                    "tags": tags,
                    "run_id": self.foundation.run_id,
                    "ingested_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                }
            )
            self.ingested_doc_ids.add(doc_hash)
        return True

    def finalize(self):
        self.reporter.save_report()

def paragraph_chunk_text(
    text: str,
    *,
    min_words: int = DEFAULT_MIN_PARAGRAPH_WORDS,
    max_chars: int = DEFAULT_MAX_PARAGRAPH_CHARS,
    ) -> List[str]:
    if not text:
        return []

    paras = [p.strip() for p in RE_PARA_SPLIT.split(text) if p.strip()]

    chunks: List[str] = []
    buf: List[str] = []
    buf_words = 0

    def flush():
        nonlocal buf, buf_words
        if buf:
            chunks.append("\n\n".join(buf).strip())
        buf = []
        buf_words = 0

    for p in paras:
        p_words = len(p.split())
        if len(p) > max_chars:
            flush()
            chunks.append(p)
            continue

        buf.append(p)
        buf_words += p_words
        if buf_words >= min_words or len("\n\n".join(buf)) >= max_chars:
            flush()

    flush()
    return chunks

if __name__ == "__main__":
    proc = IngestionProcessor()
    # proc.process_document("TM-9-6115.pdf")
    proc.finalize()
