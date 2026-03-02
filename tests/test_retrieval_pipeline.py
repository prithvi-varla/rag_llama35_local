from __future__ import annotations

from pathlib import Path

from rag_app.core.pipeline import RAGPipeline
from rag_app.engines.mock_engine import MockEngine


def test_pipeline_retrieves_chunks() -> None:
    root = Path(__file__).resolve().parent.parent
    corpus = (root / "data" / "corpus.jsonl").as_posix()

    pipeline = RAGPipeline(
        corpus_path=corpus,
        engine=MockEngine(),
        chunk_mode="semantic",
        top_k=3,
        bm25_top_k=5,
        dense_top_k=5,
        rerank_top_k=3,
    )

    result = pipeline.answer("Why combine BM25 with dense retrieval?")
    assert len(result.used_chunks) == 3
    assert result.latency_ms >= 0.0
    assert "Retrieved context" in result.answer


def test_baseline_mode_runs() -> None:
    root = Path(__file__).resolve().parent.parent
    corpus = (root / "data" / "corpus.jsonl").as_posix()

    pipeline = RAGPipeline(
        corpus_path=corpus,
        engine=MockEngine(),
        chunk_mode="fixed",
        top_k=2,
    )

    results = pipeline.retrieve("What metrics should be monitored in production RAG?")
    assert len(results) == 2
