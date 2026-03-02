from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

from rag_app.core.types import RetrievalResult


@dataclass
class EvalQuery:
    query_id: str
    question: str
    gold_doc_ids: List[str]


def load_eval_queries(path: str) -> List[EvalQuery]:
    rows: List[EvalQuery] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            rows.append(
                EvalQuery(
                    query_id=item["id"],
                    question=item["question"],
                    gold_doc_ids=item["gold_doc_ids"],
                )
            )
    return rows


def hit_at_k(results: List[RetrievalResult], gold_doc_ids: Iterable[str]) -> float:
    gold = set(gold_doc_ids)
    return float(any(r.chunk.doc_id in gold for r in results))


def mrr(results: List[RetrievalResult], gold_doc_ids: Iterable[str]) -> float:
    gold = set(gold_doc_ids)
    for rank, r in enumerate(results, start=1):
        if r.chunk.doc_id in gold:
            return 1.0 / rank
    return 0.0


def summarize(records: List[Dict[str, float]]) -> Dict[str, float]:
    if not records:
        return {"hit_at_k": 0.0, "mrr": 0.0, "latency_ms": 0.0}

    return {
        "hit_at_k": mean(r["hit_at_k"] for r in records),
        "mrr": mean(r["mrr"] for r in records),
        "latency_ms": mean(r["latency_ms"] for r in records),
    }
