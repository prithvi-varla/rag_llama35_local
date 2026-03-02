from __future__ import annotations

from typing import List

from rag_app.core.types import RetrievalResult
from rag_app.engines.base import InferenceEngine


class MLXEngine(InferenceEngine):
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def generate(self, query: str, contexts: List[RetrievalResult], max_new_tokens: int, temperature: float) -> str:
        try:
            from mlx_lm import generate, load
        except Exception as exc:
            raise RuntimeError(
                "mlx-lm is not installed. Install mlx and mlx-lm to use MLX inference."
            ) from exc

        model, tokenizer = load(self.model_path)

        context_block = "\n\n".join(
            [f"[{r.chunk.doc_id}] {r.chunk.title}\n{r.chunk.text}" for r in contexts]
        )
        prompt = (
            "You are a production RAG assistant. Use ONLY the provided context. "
            "If context is insufficient, say so explicitly.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {query}\nAnswer:"
        )

        return generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens,
            temp=temperature,
        )
