from __future__ import annotations

from typing import List

from rag_app.core.types import Chunk, Document


def fixed_chunk_documents(
    docs: List[Document], chunk_size_chars: int, chunk_overlap_chars: int
) -> List[Chunk]:
    chunks: List[Chunk] = []
    for doc in docs:
        start = 0
        idx = 0
        text = doc.text
        while start < len(text):
            end = min(len(text), start + chunk_size_chars)
            segment = text[start:end].strip()
            if segment:
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}_fixed_{idx}",
                        doc_id=doc.doc_id,
                        title=doc.title,
                        text=segment,
                    )
                )
            if end >= len(text):
                break
            start = max(0, end - chunk_overlap_chars)
            idx += 1
    return chunks


def semantic_chunk_documents(docs: List[Document], max_chars: int = 600) -> List[Chunk]:
    chunks: List[Chunk] = []
    for doc in docs:
        sentences = [s.strip() for s in doc.text.split(".") if s.strip()]
        idx = 0
        bucket = ""
        for sent in sentences:
            candidate = (bucket + ". " + sent).strip(". ")
            if len(candidate) > max_chars and bucket:
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}_sem_{idx}",
                        doc_id=doc.doc_id,
                        title=doc.title,
                        text=bucket + ".",
                    )
                )
                idx += 1
                bucket = sent
            else:
                bucket = candidate
        if bucket:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}_sem_{idx}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=bucket + ".",
                )
            )
    return chunks
