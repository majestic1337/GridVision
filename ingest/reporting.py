import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("GridVision_Reporter")

@dataclass
class QualityStats:
    total_chunks: int = 0
    low_quality_cnt: int = 0
    avg_score: float = 0.0
    min_score_threshold: float = 0.3
    
    type_dist: Dict[str, int] = field(default_factory=lambda: {
        "text": 0, "image_caption": 0, "table_md": 0, "ocr_text": 0
    })
    
    score_dist: Dict[str, int] = field(default_factory=lambda: {
        "0.0-0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, "0.7-0.9": 0, "0.9-1.0": 0
    })

    def update(self, score: float, chunk_type: str):
        self.total_chunks += 1
        self.avg_score += (score - self.avg_score) / self.total_chunks
        
        if score < self.min_score_threshold: 
            self.low_quality_cnt += 1
        
        if chunk_type not in self.type_dist:
            self.type_dist[chunk_type] = 0
        self.type_dist[chunk_type] += 1
        if score < 0.3: self.score_dist["0.0-0.3"] += 1
        elif score < 0.5: self.score_dist["0.3-0.5"] += 1
        elif score < 0.7: self.score_dist["0.5-0.7"] += 1
        elif score < 0.9: self.score_dist["0.7-0.9"] += 1
        else: self.score_dist["0.9-1.0"] += 1

@dataclass
class RunStats:
    start_time: str
    run_id: str
    docs_processed: int = 0
    total_pages: int = 0
    total_assets: int = 0
    quality: QualityStats = field(default_factory=QualityStats)
    warnings: List[str] = field(default_factory=list)

class IngestionReporter:
    def __init__(self, run_id: str, artifact_dir: Path, root_dir: Path, min_score: float = 0.3):
        self.artifact_dir = artifact_dir
        self.root_dir = root_dir
        
        self.stats = RunStats(
            start_time=datetime.utcnow().isoformat(),
            run_id=run_id
        )
        self.stats.quality.min_score_threshold = min_score

    def log_document(self, doc_slug: str, pages: int, assets: int):
        """Log high-level document stats."""
        self.stats.docs_processed += 1
        self.stats.total_pages += pages
        self.stats.total_assets += assets
        logger.info(f"Stats update: {doc_slug} | Pages: {pages} | Assets: {assets}")

    def log_chunk_quality(self, score: float, chunk_type: str):
        """Log granular chunk quality and type."""
        self.stats.quality.update(score, chunk_type)

    def log_warning(self, msg: str):
        """Collect runtime warnings."""
        self.stats.warnings.append(f"[{datetime.utcnow().isoformat()}] {msg}")
        logger.warning(f"{msg}")

    def save_report(self):
        report_path = self.artifact_dir / "ingest_report.json"
        report_data = asdict(self.stats)
        report_data["end_time"] = datetime.utcnow().isoformat()
        validation_res = self._run_validation()
        report_data["validation"] = validation_res

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
            
        logger.info(f"Ingestion Report saved to {report_path}")
        summary = (f"Summary: {self.stats.docs_processed} docs, "
                   f"{self.stats.quality.total_chunks} chunks. "
                   f"Type Dist: {self.stats.quality.type_dist}")
        logger.info(summary)

    def _run_validation(self) -> Dict[str, Any]:
        results = {"status": "PASS", "checks": []}
        
        maps = ["element_to_chunks.json", "page_to_assets.json", "chunks.jsonl", "assets_manifest.json"]
        missing = [m for m in maps if not (self.artifact_dir / m).exists()]
        
        if missing:
            results["status"] = "FAIL"
            results["checks"].append(f"Missing artifacts: {missing}")
            return results

        try:
            with open(self.artifact_dir / "assets_manifest.json", "r", encoding="utf-8") as f:
                manifest = json.load(f)
            
            if not isinstance(manifest, list):
                 raise ValueError("assets_manifest.json must be a JSON Array")
            missing_files = 0
            checked_count = 0
            
            for item in manifest[:20]: 
                checked_count += 1
                rel_path = Path(item["file_path"])
                if rel_path.is_absolute():
                    abs_path = rel_path
                else:
                    abs_path = self.root_dir / rel_path
                
                if not abs_path.exists():
                     missing_files += 1
            
            if missing_files > 0:
                results["checks"].append(f"Found {missing_files} missing asset files in sample of {checked_count}.")
                results["status"] = "WARN"
            else:
                 results["checks"].append(f"Asset file check passed (sample {checked_count}).")

        except Exception as e:
            results["checks"].append(f"Validation error: {e}")
            results["status"] = "ERROR"

        return results