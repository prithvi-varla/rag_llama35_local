from __future__ import annotations

import argparse
from pathlib import Path

from rag_app.core.config import load_settings
from rag_app.core.observability import RAGObservability
from rag_app.core.pipeline import RAGPipeline
from rag_app.engines.base import InferenceEngine
from rag_app.engines.mlx_engine import MLXEngine
from rag_app.engines.mock_engine import MockEngine
from rag_app.engines.vllm_engine import VLLMEngine


def build_engine(settings) -> InferenceEngine:
    """Creates the configured inference backend (mlx, vllm, or mock)."""
    engine_name = settings.inference.get("engine", "mock")
    if engine_name == "mlx":
        return MLXEngine(settings.inference["mlx"]["model_path"])
    if engine_name == "vllm":
        return VLLMEngine(
            base_url=settings.inference["vllm"]["base_url"],
            api_key=settings.inference["vllm"].get("api_key", "EMPTY"),
            model_name=settings.model["target_model"],
        )
    return MockEngine()


def main() -> None:
    """Parses CLI args, runs one RAG query, and prints answer + citations."""
    parser = argparse.ArgumentParser(description="Run one local RAG query")
    parser.add_argument("--settings", default="config/settings.yaml")
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", choices=["baseline", "hybrid"], default="hybrid")
    args = parser.parse_args()

    settings = load_settings(args.settings)
    root = Path(args.settings).resolve().parent.parent
    validation = settings.validation
    observability = RAGObservability.from_settings(settings)

    pipeline = RAGPipeline(
        corpus_path=(root / settings.paths["corpus_path"]).as_posix(),
        engine=build_engine(settings),
        chunk_mode="fixed" if args.mode == "baseline" else "semantic",
        chunk_size_chars=settings.retrieval["chunk_size_chars"],
        chunk_overlap_chars=settings.retrieval["chunk_overlap_chars"],
        top_k=settings.retrieval["top_k"],
        bm25_top_k=settings.retrieval["bm25_top_k"],
        dense_top_k=settings.retrieval["dense_top_k"],
        rerank_top_k=settings.retrieval["rerank_top_k"],
        enforce_citations=validation.get("enforce_citations", True),
        validation_min_supported_chunks=validation.get("min_supported_chunks", 1),
        validation_min_relevance_score=validation.get("min_relevance_score", 0.25),
        embedding_model=settings.retrieval.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        reranker_model=settings.retrieval.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        observability=observability,
    )

    result = pipeline.answer(
        query=args.query,
        max_new_tokens=settings.model.get("max_new_tokens", 384),
        temperature=settings.model.get("temperature", 0.1),
    )

    print()
    print("=== Query ===")
    print(args.query)
    print()
    print("=== Answer ===")
    print(result.answer)
    print()
    print(f"Latency: {result.latency_ms:.2f} ms")
    print()
    print("=== Retrieved Chunks ===")
    for i, item in enumerate(result.used_chunks, start=1):
        print(f"{i}. {item.chunk.doc_id} | {item.chunk.title} | score={item.score:.4f} | {item.source}")


if __name__ == "__main__":
    main()
