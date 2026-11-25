"""Transactional storage for immutable document generations."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class Generation:
    doc_id: str
    generation_id: str
    staging_dir: Path
    final_dir: Path


class GenerationManager:
    """Creates document generations and commits them with one directory rename."""

    def __init__(self, root_dir: Path, artifact_dir: Path, logger, asset_dir: Path | None = None):
        self.root_dir = Path(root_dir)
        self.artifact_dir = Path(artifact_dir)
        self.docs_dir = self.artifact_dir.parent / "docs"
        self.asset_dir = Path(asset_dir) if asset_dir else self.root_dir / "data/processed/assets"
        self.logger = logger
        self.docs_dir.mkdir(parents=True, exist_ok=True)

    def begin(self, doc_id: str) -> Generation:
        doc_dir = self.docs_dir / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        generation_id = f"v{self._next_number(doc_dir)}"
        final_dir = doc_dir / generation_id
        staging_dir = self.docs_dir / f".staging-{doc_id}-{uuid.uuid4().hex}"
        staging_dir.mkdir(parents=True)
        return Generation(doc_id, generation_id, staging_dir, final_dir)

    def commit(self, generation: Generation) -> Path:
        """Atomically publish a fully written staging directory."""
        generation.final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(generation.staging_dir, generation.final_dir)
        self._rewrite_asset_paths(generation)
        return generation.final_dir

    def abort(self, generation: Generation) -> None:
        shutil.rmtree(generation.staging_dir, ignore_errors=True)

    def active_generations(self) -> Dict[str, Path]:
        active: Dict[str, Path] = {}
        if not self.docs_dir.exists():
            return active
        for doc_dir in self.docs_dir.iterdir():
            if not doc_dir.is_dir() or doc_dir.name.startswith("."):
                continue
            candidates = [p for p in doc_dir.iterdir() if p.is_dir() and re.fullmatch(r"v\d+", p.name)]
            if candidates:
                active[doc_dir.name] = max(candidates, key=lambda p: int(p.name[1:]))
        return active

    def reset(self) -> None:
        """Remove only generated document data and extracted assets."""
        shutil.rmtree(self.docs_dir, ignore_errors=True)
        shutil.rmtree(self.asset_dir, ignore_errors=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _next_number(doc_dir: Path) -> int:
        numbers = []
        for path in doc_dir.iterdir():
            match = re.fullmatch(r"v(\d+)", path.name)
            if match:
                numbers.append(int(match.group(1)))
        return max(numbers, default=0) + 1

    def _rewrite_asset_paths(self, generation: Generation) -> None:
        manifest_path = generation.final_dir / "assets_manifest.json"
        if not manifest_path.exists():
            return
        try:
            assets = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        changed = False
        for asset in assets:
            path = asset.get("file_path")
            if not path:
                continue
            candidate = Path(path)
            if not candidate.is_absolute():
                candidate = self.root_dir / candidate
            staging_root = generation.staging_dir.resolve()
            try:
                relative_to_staging = candidate.resolve().relative_to(staging_root)
                candidate = generation.final_dir / relative_to_staging
            except ValueError:
                pass
            try:
                asset["file_path"] = str(candidate.relative_to(self.root_dir))
                changed = True
            except ValueError:
                pass
        if changed:
            manifest_path.write_text(json.dumps(assets, indent=2), encoding="utf-8")
