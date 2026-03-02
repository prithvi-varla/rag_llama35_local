from __future__ import annotations

from rag_app.core.eval import hit_at_k, mrr, summarize
from rag_app.core.types import Chunk, RetrievalResult


def _item(doc_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(chunk_id=f"{doc_id}_c", doc_id=doc_id, title="t", text="x"),
        score=score,
        source="test",
    )


def test_hit_and_mrr() -> None:
    results = [_item("doc_2", 0.9), _item("doc_5", 0.8), _item("doc_1", 0.7)]
    assert hit_at_k(results, ["doc_1"]) == 1.0
    assert mrr(results, ["doc_1"]) == 1.0 / 3.0


def test_summarize() -> None:
    rows = [
        {"hit_at_k": 1.0, "mrr": 0.5, "latency_ms": 10.0},
        {"hit_at_k": 0.0, "mrr": 0.0, "latency_ms": 20.0},
    ]
    out = summarize(rows)
    assert out["hit_at_k"] == 0.5
    assert out["mrr"] == 0.25
    assert out["latency_ms"] == 15.0
