from __future__ import annotations

from typing import List

from rag_app.core.types import RetrievalResult


def lexical_rerank(query: str, candidates: List[RetrievalResult], top_k: int = 5) -> List[RetrievalResult]:
    q_terms = set(query.lower().split())

    rescored = []
    for item in candidates:
        chunk_terms = set((item.chunk.title + " " + item.chunk.text).lower().split())
        overlap = len(q_terms & chunk_terms)
        adjusted = 0.7 * item.score + 0.3 * overlap
        rescored.append(
            RetrievalResult(chunk=item.chunk, score=float(adjusted), source=item.source + "+rerank")
        )

    rescored.sort(key=lambda x: x.score, reverse=True)
    return rescored[:top_k]
