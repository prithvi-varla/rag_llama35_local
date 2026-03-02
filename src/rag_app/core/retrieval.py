from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
import re
from typing import Dict, List, Tuple

from rag_app.core.types import Chunk, RetrievalResult


@dataclass
class RetrievalConfig:
    bm25_top_k: int = 8
    dense_top_k: int = 8


class HybridRetriever:
    def __init__(self, chunks: List[Chunk], config: RetrievalConfig) -> None:
        self.chunks = chunks
        self.config = config
        self._texts = [f"{c.title}. {c.text}" for c in chunks]
        self._tokenized = [_tokenize(t) for t in self._texts]
        self._bm25 = _SimpleBM25(self._tokenized)
        self._dense = _SimpleTfidf(self._tokenized, use_bigrams=True)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        bm25_scores = self._bm25.get_scores(_tokenize(query))
        dense_scores = self._dense.score_query(_tokenize(query))
        bm25_norm = _minmax(bm25_scores)
        dense_norm = _minmax(dense_scores)

        hybrid = [0.45 * b + 0.55 * d for b, d in zip(bm25_norm, dense_norm)]
        idx = sorted(range(len(hybrid)), key=lambda i: hybrid[i], reverse=True)[:top_k]

        return [
            RetrievalResult(chunk=self.chunks[i], score=hybrid[i], source="hybrid")
            for i in idx
        ]


class BaselineRetriever:
    def __init__(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        tokens = [_tokenize(f"{c.title}. {c.text}") for c in chunks]
        self._dense = _SimpleTfidf(tokens, use_bigrams=False)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        scores = self._dense.score_query(_tokenize(query))
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            RetrievalResult(chunk=self.chunks[i], score=scores[i], source="dense_baseline")
            for i in idx
        ]


def reciprocal_rank_fusion(
    first: List[RetrievalResult],
    second: List[RetrievalResult],
    k: int = 60,
    top_k: int = 5,
) -> List[RetrievalResult]:
    scores: Dict[str, float] = {}
    by_chunk: Dict[str, RetrievalResult] = {}

    for rank, item in enumerate(first, start=1):
        scores[item.chunk.chunk_id] = scores.get(item.chunk.chunk_id, 0.0) + 1.0 / (k + rank)
        by_chunk[item.chunk.chunk_id] = item

    for rank, item in enumerate(second, start=1):
        scores[item.chunk.chunk_id] = scores.get(item.chunk.chunk_id, 0.0) + 1.0 / (k + rank)
        by_chunk[item.chunk.chunk_id] = item

    ranked_ids = sorted(scores.keys(), key=lambda cid: -scores[cid])[:top_k]
    return [
        RetrievalResult(chunk=by_chunk[cid].chunk, score=float(scores[cid]), source="rrf")
        for cid in ranked_ids
    ]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _minmax(values: List[float]) -> List[float]:
    if not values:
        return values
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


class _SimpleBM25:
    def __init__(self, tokenized_docs: List[List[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = tokenized_docs
        self.k1 = k1
        self.b = b
        self.doc_lens = [len(doc) for doc in tokenized_docs]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        self.term_freqs: List[Dict[str, int]] = []
        self.df: Dict[str, int] = {}

        for doc in tokenized_docs:
            tf: Dict[str, int] = {}
            for t in doc:
                tf[t] = tf.get(t, 0) + 1
            self.term_freqs.append(tf)
            for t in tf.keys():
                self.df[t] = self.df.get(t, 0) + 1

        self.n_docs = len(tokenized_docs)

    def idf(self, term: str) -> float:
        n_q = self.df.get(term, 0)
        return log((self.n_docs - n_q + 0.5) / (n_q + 0.5) + 1.0)

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores: List[float] = []
        for i, tf in enumerate(self.term_freqs):
            dl = self.doc_lens[i]
            score = 0.0
            for term in query_tokens:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1.0 - self.b + self.b * dl / max(1e-9, self.avgdl))
                score += self.idf(term) * (f * (self.k1 + 1.0)) / max(1e-9, denom)
            scores.append(score)
        return scores


class _SimpleTfidf:
    def __init__(self, tokenized_docs: List[List[str]], use_bigrams: bool) -> None:
        docs = [_add_bigrams(tokens) if use_bigrams else tokens for tokens in tokenized_docs]
        self.n_docs = len(docs)
        self.df: Dict[str, int] = {}
        self.docs_tf: List[Dict[str, int]] = []
        for doc in docs:
            tf: Dict[str, int] = {}
            for t in doc:
                tf[t] = tf.get(t, 0) + 1
            self.docs_tf.append(tf)
            for t in tf.keys():
                self.df[t] = self.df.get(t, 0) + 1
        self.use_bigrams = use_bigrams

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return log((1 + self.n_docs) / (1 + df)) + 1.0

    def _tfidf(self, tf: Dict[str, int]) -> Dict[str, float]:
        vec: Dict[str, float] = {}
        for t, freq in tf.items():
            vec[t] = freq * self._idf(t)
        return vec

    def score_query(self, query_tokens: List[str]) -> List[float]:
        qtoks = _add_bigrams(query_tokens) if self.use_bigrams else query_tokens
        qtf: Dict[str, int] = {}
        for t in qtoks:
            qtf[t] = qtf.get(t, 0) + 1
        qvec = self._tfidf(qtf)
        qnorm = _l2(qvec)
        if qnorm == 0.0:
            return [0.0 for _ in self.docs_tf]

        scores: List[float] = []
        for tf in self.docs_tf:
            dvec = self._tfidf(tf)
            dnorm = _l2(dvec)
            if dnorm == 0.0:
                scores.append(0.0)
                continue
            dot = 0.0
            small, large = (qvec, dvec) if len(qvec) < len(dvec) else (dvec, qvec)
            for t, v in small.items():
                dot += v * large.get(t, 0.0)
            scores.append(dot / (qnorm * dnorm))
        return scores


def _add_bigrams(tokens: List[str]) -> List[str]:
    if len(tokens) < 2:
        return tokens[:]
    out = tokens[:]
    for i in range(len(tokens) - 1):
        out.append(f"{tokens[i]}_{tokens[i+1]}")
    return out


def _l2(vec: Dict[str, float]) -> float:
    return sqrt(sum(v * v for v in vec.values()))
