"""Export the active immutable generations to legacy catalog artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List


class CatalogExporter:
    def __init__(self, artifact_dir: Path, generation_manager, logger):
        self.artifact_dir = Path(artifact_dir)
        self.generation_manager = generation_manager
        self.logger = logger

    def export(self) -> None:
        active = self.generation_manager.active_generations()
        chunks: List[Dict[str, Any]] = []
        assets: List[Dict[str, Any]] = []
        documents: List[Dict[str, Any]] = []
        element_map: Dict[str, Any] = {}
        page_map: Dict[str, Any] = {}
        for generation in active.values():
            chunks.extend(self._read_jsonl(generation / "chunks.jsonl"))
            assets.extend(self._read_json(generation / "assets_manifest.json", []))
            documents.extend(self._read_jsonl(generation / "documents.jsonl"))
            element_map.update(self._read_json(generation / "element_to_chunks.json", {}))
            page_map.update(self._read_json(generation / "page_to_assets.json", {}))

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_jsonl(self.artifact_dir / "chunks.jsonl", chunks)
        self._atomic_json(self.artifact_dir / "assets_manifest.json", assets)
        self._atomic_json(self.artifact_dir / "element_to_chunks.json", element_map)
        self._atomic_json(self.artifact_dir / "page_to_assets.json", page_map)
        self._atomic_jsonl(self.artifact_dir / "documents.jsonl", documents)
        self.logger.info("[CATALOG] Exported %s active generation(s), %s chunk(s), %s asset(s)", len(active), len(chunks), len(assets))

    @staticmethod
    def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        result = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                result.append(json.loads(line))
        return result

    @staticmethod
    def _read_json(path: Path, default):
        if not path.exists():
            return default
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, type(default)) else default
        except (OSError, ValueError):
            return default

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def _atomic_json(self, path: Path, value: Any) -> None:
        self._atomic_write(path, json.dumps(value, indent=2))

    def _atomic_jsonl(self, path: Path, values: Iterable[Dict[str, Any]]) -> None:
        text = "".join(json.dumps(value) + "\n" for value in values)
        self._atomic_write(path, text)
