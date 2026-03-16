from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ImageRecord:
    path: Path
    size_bytes: int
    modified_time: float
    caption: str
    embedding: bytes


@dataclass(slots=True)
class SearchResult:
    path: Path
    caption: str
    score: float
