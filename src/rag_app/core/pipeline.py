from __future__ import annotations

import time
from typing import List

from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import (
    CrossEncoderReranker,
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document as LCDocument
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_app.core.io import load_corpus
from rag_app.core.types import Chunk, GenerationResult, RetrievalResult
from rag_app.engines.base import InferenceEngine


class RAGPipeline:
    """Thin orchestration layer around LangChain chunking, retrieval, reranking, and filtering."""

    def __init__(
        self,
        corpus_path: str,
        engine: InferenceEngine,
        chunk_mode: str = "semantic",
        chunk_size_chars: int = 550,
        chunk_overlap_chars: int = 90,
        top_k: int = 5,
        bm25_top_k: int = 12,
        dense_top_k: int = 12,
        rerank_top_k: int = 5,
        enforce_citations: bool = True,
        validation_min_supported_chunks: int = 1,
        validation_min_relevance_score: float = 0.25,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        """Builds one shared LangChain retrieval graph and keeps it in memory for reuse."""
        self.engine = engine
        self.top_k = top_k
        self.enforce_citations = enforce_citations
        self.validation_min_supported_chunks = validation_min_supported_chunks

        docs = load_corpus(corpus_path)
        lc_docs = [
            LCDocument(
                page_content=d.text,
                metadata={"doc_id": d.doc_id, "title": d.title},
            )
            for d in docs
        ]

        separators = ["\n\n", "\n", ". ", " "] if chunk_mode == "semantic" else [" "]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size_chars,
            chunk_overlap=chunk_overlap_chars,
            separators=separators,
        )
        split_docs = splitter.split_documents(lc_docs)

        for i, item in enumerate(split_docs):
            item.metadata["chunk_id"] = f"chunk_{i}"

        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )

        vectorstore = Chroma(collection_name="rag_chunks", embedding_function=embeddings)
        vectorstore.add_documents(split_docs)

        dense_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": max(top_k, dense_top_k, rerank_top_k)},
        )

        if chunk_mode == "fixed":
            base_retriever = dense_retriever
            source_label = "langchain_dense"
        else:
            bm25 = BM25Retriever.from_documents(split_docs)
            bm25.k = max(top_k, bm25_top_k, rerank_top_k)
            base_retriever = EnsembleRetriever(
                retrievers=[bm25, dense_retriever],
                weights=[0.45, 0.55],
            )
            source_label = "langchain_hybrid"

        cross_encoder = HuggingFaceCrossEncoder(model_name=reranker_model)
        reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_k)
        emb_filter = EmbeddingsFilter(
            embeddings=embeddings,
            similarity_threshold=validation_min_relevance_score if enforce_citations else 0.0,
            k=max(top_k, validation_min_supported_chunks),
        )

        compressor = DocumentCompressorPipeline(transformers=[emb_filter, reranker])
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )
        self.source_label = source_label

    def retrieve(self, query: str) -> List[RetrievalResult]:
        """Uses LangChain retriever stack to return top grounded chunks."""
        docs = self.retriever.invoke(query)
        out: List[RetrievalResult] = []
        for d in docs[: self.top_k]:
            out.append(
                RetrievalResult(
                    chunk=Chunk(
                        chunk_id=d.metadata.get("chunk_id", "chunk_unknown"),
                        doc_id=d.metadata.get("doc_id", "doc_unknown"),
                        title=d.metadata.get("title", "Untitled"),
                        text=d.page_content,
                    ),
                    score=float(d.metadata.get("relevance_score", d.metadata.get("score", 0.0))),
                    source=self.source_label,
                )
            )
        return out

    def answer(self, query: str, max_new_tokens: int = 384, temperature: float = 0.1) -> GenerationResult:
        """Answers using retrieved LangChain chunks, or safely declines when support is low."""
        start = time.perf_counter()
        retrieved = self.retrieve(query)

        if self.enforce_citations and len(retrieved) < self.validation_min_supported_chunks:
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
