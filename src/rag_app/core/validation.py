from __future__ import annotations

from typing import Dict, List, Tuple

from rag_app.core.types import RetrievalResult


def validate_retrieval_support(
    retrieved: List[RetrievalResult],
    min_supported_chunks: int = 1,
    min_relevance_score: float = 0.25,
) -> Tuple[bool, Dict[str, float]]:
    """Checks whether retrieved evidence is strong enough to safely answer."""
    if not retrieved:
        return False, {"supported_chunks": 0.0, "max_score": 0.0, "validation_pass": 0.0}

    supported = [r for r in retrieved if r.score >= min_relevance_score]
    max_score = max(r.score for r in retrieved)
    passed = len(supported) >= min_supported_chunks

    return passed, {
        "supported_chunks": float(len(supported)),
        "max_score": float(max_score),
        "validation_pass": 1.0 if passed else 0.0,
    }
