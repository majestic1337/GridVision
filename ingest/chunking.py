"""Content-aware table chunking."""

from __future__ import annotations

from typing import List


class TableChunker:
    def __init__(self, max_chunk_size: int = 1500, reinject_headers: bool = True):
        self.max_chunk_size = max(1, int(max_chunk_size))
        self.reinject_headers = bool(reinject_headers)

    def chunk(self, markdown: str, max_chunk_size: int | None = None) -> List[str]:
        lines = [line.strip() for line in (markdown or "").splitlines() if line.strip()]
        if not lines:
            return []
        limit = max(1, int(max_chunk_size or self.max_chunk_size))
        header: List[str] = []
        rows = lines
        if len(lines) >= 2 and self._is_table_row(lines[0]) and self._is_separator(lines[1]):
            header = lines[:2]
            rows = lines[2:]

        if not rows:
            return ["\n".join(header)] if header else []

        chunks: List[str] = []
        current: List[str] = []
        prefix = header if self.reinject_headers and header else []

        for row in rows:
            candidate = self._join(prefix + current + [row])
            if current and len(candidate) > limit:
                chunks.append(self._join(prefix + current))
                current = [row]
            else:
                current.append(row)

        if current:
            chunks.append(self._join(prefix + current))
        return chunks

    def split(self, markdown: str, max_chunk_size: int | None = None) -> List[str]:
        return self.chunk(markdown, max_chunk_size)

    @staticmethod
    def _join(lines: List[str]) -> str:
        return "\n".join(line for line in lines if line).strip()

    @staticmethod
    def _is_table_row(line: str) -> bool:
        return line.startswith("|") and line.endswith("|")

    @staticmethod
    def _is_separator(line: str) -> bool:
        if not TableChunker._is_table_row(line):
            return False
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        return bool(cells) and all(cell and set(cell) <= set("-: ") for cell in cells)
