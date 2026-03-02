from __future__ import annotations

import time
from typing import List

from rag_app.core.chunking import fixed_chunk_documents, semantic_chunk_documents
from rag_app.core.io import load_corpus
from rag_app.core.rerank import lexical_rerank
from rag_app.core.retrieval import BaselineRetriever, HybridRetriever, RetrievalConfig
from rag_app.core.types import GenerationResult, RetrievalResult
from rag_app.core.validation import run_validation
from rag_app.engines.base import InferenceEngine


class RAGPipeline:
    def __init__(
        self,
        corpus_path: str,
        engine: InferenceEngine,
        chunk_mode: str = "semantic",
        chunk_size_chars: int = 550,
        chunk_overlap_chars: int = 90,
        top_k: int = 5,
        bm25_top_k: int = 8,
        dense_top_k: int = 8,
        rerank_top_k: int = 5,
        enforce_citations: bool = True,
        validation_min_hits: int = 1,
        validation_min_overlap_terms: int = 2,
        validation_min_score: float = 0.01,
    ) -> None:
        self.engine = engine
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.enforce_citations = enforce_citations
        self.validation_min_hits = validation_min_hits
        self.validation_min_overlap_terms = validation_min_overlap_terms
        self.validation_min_score = validation_min_score

        docs = load_corpus(corpus_path)
        if chunk_mode == "fixed":
            chunks = fixed_chunk_documents(docs, chunk_size_chars, chunk_overlap_chars)
            self.baseline_retriever = BaselineRetriever(chunks)
            self.hybrid_retriever = None
        else:
            chunks = semantic_chunk_documents(docs, max_chars=chunk_size_chars)
            self.baseline_retriever = None
            self.hybrid_retriever = HybridRetriever(
                chunks,
                RetrievalConfig(bm25_top_k=bm25_top_k, dense_top_k=dense_top_k),
            )

    def retrieve(self, query: str) -> List[RetrievalResult]:
        if self.hybrid_retriever is not None:
            candidates = self.hybrid_retriever.retrieve(query, top_k=max(self.top_k, self.rerank_top_k))
            return lexical_rerank(query, candidates, top_k=self.top_k)

        if self.baseline_retriever is not None:
            return self.baseline_retriever.retrieve(query, top_k=self.top_k)

        raise RuntimeError("No retriever configured")

    def answer(self, query: str, max_new_tokens: int = 384, temperature: float = 0.1) -> GenerationResult:
        start = time.perf_counter()
        retrieved = self.retrieve(query)
        if self.enforce_citations:
            valid, _ = run_validation(
                query=query,
                retrieved=retrieved,
                min_hits=self.validation_min_hits,
                min_overlap_terms=self.validation_min_overlap_terms,
                min_score=self.validation_min_score,
            )
            if not valid:
                latency_ms = (time.perf_counter() - start) * 1000
                return GenerationResult(
                    answer=(
                        "I do not have enough grounded evidence in retrieved documents to answer safely. "
                        "Please refine your question or add more relevant source documents."
                    ),
                    used_chunks=retrieved,
                    latency_ms=latency_ms,
                )
        answer = self.engine.generate(query, retrieved, max_new_tokens=max_new_tokens, temperature=temperature)
        latency_ms = (time.perf_counter() - start) * 1000
        return GenerationResult(answer=answer, used_chunks=retrieved, latency_ms=latency_ms)
