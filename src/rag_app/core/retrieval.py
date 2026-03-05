from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from rag_app.core.types import Chunk, RetrievalResult


@dataclass
class RetrievalConfig:
    """Holds retrieval hyperparameters for dense and hybrid retrieval."""

    bm25_top_k: int = 12
    dense_top_k: int = 12
    bm25_weight: float = 0.45
    dense_weight: float = 0.55
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


class BaselineRetriever:
    """Dense-only retriever implemented with sentence-transformers embeddings."""

    def __init__(self, chunks: List[Chunk], embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """Builds an in-memory dense index for all chunks."""
        self.chunks = chunks
        self._texts = [f"{c.title}. {c.text}" for c in chunks]
        self._embedder = SentenceTransformer(embedding_model)
        self._embeddings = self._embedder.encode(self._texts, normalize_embeddings=True, show_progress_bar=False)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Returns top-k chunks ranked by dense cosine similarity."""
        q = self._embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = np.dot(self._embeddings, q)
        idx = np.argsort(-scores)[:top_k]
        return [
            RetrievalResult(chunk=self.chunks[i], score=float(scores[i]), source="dense")
            for i in idx
        ]


class HybridRetriever:
    """Hybrid retriever combining BM25 keyword and dense semantic retrieval."""

    def __init__(self, chunks: List[Chunk], config: RetrievalConfig) -> None:
        """Builds BM25 and dense indexes once at startup."""
        self.chunks = chunks
        self.config = config
        self._texts = [f"{c.title}. {c.text}" for c in chunks]
        self._tokenized = [t.lower().split() for t in self._texts]
        self._bm25 = BM25Okapi(self._tokenized)
        self._embedder = SentenceTransformer(config.embedding_model)
        self._embeddings = self._embedder.encode(self._texts, normalize_embeddings=True, show_progress_bar=False)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Returns top-k chunks using weighted fusion of BM25 and dense scores."""
        bm25_scores = np.array(self._bm25.get_scores(query.lower().split()), dtype=np.float32)
        dense_q = self._embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        dense_scores = np.dot(self._embeddings, dense_q)

        bm25_norm = _minmax(bm25_scores)
        dense_norm = _minmax(dense_scores)
        fused = (self.config.bm25_weight * bm25_norm) + (self.config.dense_weight * dense_norm)

        idx = np.argsort(-fused)[: max(top_k, self.config.bm25_top_k, self.config.dense_top_k)]
        return [
            RetrievalResult(chunk=self.chunks[i], score=float(fused[i]), source="hybrid")
            for i in idx
        ]


def _minmax(x: np.ndarray) -> np.ndarray:
    """Normalizes any score array to [0, 1] for stable fusion."""
    if x.size == 0:
        return x
    span = x.max() - x.min()
    if span < 1e-8:
        return np.zeros_like(x)
    return (x - x.min()) / span
