from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from rag_app.core.config import load_settings
from rag_app.core.pipeline import RAGPipeline
from rag_app.engines.base import InferenceEngine
from rag_app.engines.mlx_engine import MLXEngine
from rag_app.engines.mock_engine import MockEngine
from rag_app.engines.vllm_engine import VLLMEngine


class AskRequest(BaseModel):
    query: str


class UsedChunk(BaseModel):
    doc_id: str
    title: str
    score: float
    source: str


class AskResponse(BaseModel):
    answer: str
    latency_ms: float
    used_chunks: List[UsedChunk]


def _build_engine(engine_name: str, settings_model: dict, settings_inference: dict) -> InferenceEngine:
    if engine_name == "mlx":
        return MLXEngine(settings_inference["mlx"]["model_path"])

    if engine_name == "vllm":
        return VLLMEngine(
            base_url=settings_inference["vllm"]["base_url"],
            api_key=settings_inference["vllm"].get("api_key", "EMPTY"),
            model_name=settings_model["target_model"],
        )

    return MockEngine()


def create_app(settings_path: str = "config/settings.yaml") -> FastAPI:
    settings = load_settings(settings_path)
    engine_name = settings.inference.get("engine", "mock")
    engine = _build_engine(engine_name, settings.model, settings.inference)
    validation = settings.validation

    root = Path(settings_path).resolve().parent.parent
    corpus_path = (root / settings.paths["corpus_path"]).as_posix()

    pipeline = RAGPipeline(
        corpus_path=corpus_path,
        engine=engine,
        chunk_mode="semantic",
        chunk_size_chars=settings.retrieval["chunk_size_chars"],
        chunk_overlap_chars=settings.retrieval["chunk_overlap_chars"],
        top_k=settings.retrieval["top_k"],
        bm25_top_k=settings.retrieval["bm25_top_k"],
        dense_top_k=settings.retrieval["dense_top_k"],
        rerank_top_k=settings.retrieval["rerank_top_k"],
        enforce_citations=validation.get("enforce_citations", True),
        validation_min_hits=validation.get("min_supported_chunks", 1),
        validation_min_overlap_terms=validation.get("min_overlap_terms", 2),
        validation_min_score=validation.get("min_retrieval_score", 0.01),
    )

    app = FastAPI(title="rag-llama35-local", version="1.0.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "engine": engine_name}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        result = pipeline.answer(
            query=req.query,
            max_new_tokens=settings.model.get("max_new_tokens", 384),
            temperature=settings.model.get("temperature", 0.1),
        )
        used = [
            UsedChunk(
                doc_id=item.chunk.doc_id,
                title=item.chunk.title,
                score=item.score,
                source=item.source,
            )
            for item in result.used_chunks
        ]
        return AskResponse(answer=result.answer, latency_ms=result.latency_ms, used_chunks=used)

    return app
