from __future__ import annotations

from typing import List

from sentence_transformers import CrossEncoder

from rag_app.core.types import RetrievalResult


class CrossEncoderReranker:
    """Reranks retrieved chunks using a cross-encoder relevance model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """Loads the cross-encoder model used to score (query, chunk) pairs."""
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[RetrievalResult], top_k: int = 5) -> List[RetrievalResult]:
        """Returns top-k candidates sorted by cross-encoder relevance scores."""
        if not candidates:
            return []

        pairs = [(query, f"{c.chunk.title}. {c.chunk.text}") for c in candidates]
        scores = self.model.predict(pairs)

        rescored = [
            RetrievalResult(
                chunk=item.chunk,
                score=float(score),
                source=item.source + "+cross_encoder",
            )
            for item, score in zip(candidates, scores)
        ]
        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored[:top_k]
