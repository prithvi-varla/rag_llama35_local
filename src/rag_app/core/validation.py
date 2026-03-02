from __future__ import annotations

import re
from typing import Dict, List, Tuple

from rag_app.core.types import RetrievalResult


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
}


def _terms(text: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in toks if t not in STOPWORDS and len(t) > 2]


def gatekeeper(query: str, retrieved: List[RetrievalResult], min_hits: int = 1) -> Tuple[bool, Dict[str, float]]:
    if not retrieved:
        return False, {"supported_chunks": 0.0}

    q = set(_terms(query))
    hits = 0
    for item in retrieved:
        chunk_terms = set(_terms(item.chunk.title + " " + item.chunk.text))
        if q & chunk_terms:
            hits += 1

    return hits >= min_hits, {"supported_chunks": float(hits)}


def auditor(query: str, retrieved: List[RetrievalResult], min_overlap_terms: int = 2) -> Tuple[bool, Dict[str, float]]:
    if not retrieved:
        return False, {"max_overlap_terms": 0.0}

    q = set(_terms(query))
    max_overlap = 0
    for item in retrieved:
        chunk_terms = set(_terms(item.chunk.title + " " + item.chunk.text))
        max_overlap = max(max_overlap, len(q & chunk_terms))

    return max_overlap >= min_overlap_terms, {"max_overlap_terms": float(max_overlap)}


def strategist(retrieved: List[RetrievalResult], min_score: float = 0.01) -> Tuple[bool, Dict[str, float]]:
    if not retrieved:
        return False, {"max_score": 0.0}
    max_score = max(r.score for r in retrieved)
    return max_score >= min_score, {"max_score": float(max_score)}


def run_validation(
    query: str,
    retrieved: List[RetrievalResult],
    min_hits: int = 1,
    min_overlap_terms: int = 2,
    min_score: float = 0.01,
) -> Tuple[bool, Dict[str, float]]:
    gate_ok, gate_metrics = gatekeeper(query, retrieved, min_hits=min_hits)
    audit_ok, audit_metrics = auditor(query, retrieved, min_overlap_terms=min_overlap_terms)
    strat_ok, strat_metrics = strategist(retrieved, min_score=min_score)

    metrics = {}
    metrics.update(gate_metrics)
    metrics.update(audit_metrics)
    metrics.update(strat_metrics)
    metrics["validation_pass"] = 1.0 if (gate_ok and audit_ok and strat_ok) else 0.0
    return gate_ok and audit_ok and strat_ok, metrics
