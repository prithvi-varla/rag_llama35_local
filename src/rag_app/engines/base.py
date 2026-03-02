from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from rag_app.core.types import RetrievalResult


class InferenceEngine(ABC):
    @abstractmethod
    def generate(self, query: str, contexts: List[RetrievalResult], max_new_tokens: int, temperature: float) -> str:
        raise NotImplementedError
