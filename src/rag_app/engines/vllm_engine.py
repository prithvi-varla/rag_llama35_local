from __future__ import annotations

from typing import List

from rag_app.core.types import RetrievalResult
from rag_app.engines.base import InferenceEngine


class VLLMEngine(InferenceEngine):
    def __init__(self, base_url: str, api_key: str, model_name: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name

    def generate(self, query: str, contexts: List[RetrievalResult], max_new_tokens: int, temperature: float) -> str:
        try:
            import httpx
        except Exception as exc:
            raise RuntimeError("httpx is required for vLLM engine. Install dependencies from requirements.txt") from exc

        context_block = "\n\n".join(
            [f"[{r.chunk.doc_id}] {r.chunk.title}\n{r.chunk.text}" for r in contexts]
        )

        prompt = (
            "You are a production RAG assistant. Use only provided context and cite doc ids.\n\n"
            f"Context:\n{context_block}\n\nQuestion: {query}\nAnswer:"
        )

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_new_tokens,
        }

        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        return data["choices"][0]["message"]["content"]
