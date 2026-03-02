from __future__ import annotations

from typing import List

from rag_app.core.types import RetrievalResult
from rag_app.engines.base import InferenceEngine


class MockEngine(InferenceEngine):
    def generate(self, query: str, contexts: List[RetrievalResult], max_new_tokens: int, temperature: float) -> str:
        cited = "\n".join([f"- [{c.chunk.doc_id}] {c.chunk.title}" for c in contexts])
        return (
            "This is a mock local response. Replace engine=mock with mlx or vllm for real generation.\n"
            f"Question: {query}\n"
            "Retrieved context:\n"
            f"{cited}"
        )
