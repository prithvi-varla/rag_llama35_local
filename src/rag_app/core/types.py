from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Document:
    doc_id: str
    title: str
    text: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    source: str


@dataclass
class GenerationResult:
    answer: str
    used_chunks: List[RetrievalResult]
    latency_ms: float
