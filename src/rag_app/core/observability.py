from __future__ import annotations

import importlib
import logging
import os
from typing import Any, Dict, List, Optional

from rag_app.core.types import RetrievalResult


class RAGObservability:
    """Best-effort observability wrapper with optional Langfuse integration."""

    def __init__(
        self,
        enabled: bool = False,
        public_key: str = "",
        secret_key: str = "",
        host: str = "",
        service_name: str = "rag-llama35-local",
        environment: str = "dev",
        debug: bool = False,
    ) -> None:
        """Initializes Langfuse if configured; otherwise remains safely disabled."""
        self.enabled = False
        self._client: Any = None
        self.service_name = service_name
        self.environment = environment
        self.debug = debug
        self._logger = logging.getLogger(__name__)

        if not enabled:
            if self.debug:
                self._logger.info("Langfuse disabled by config (observability.enabled=false).")
            return

        if not public_key or not secret_key:
            if self.debug:
                self._logger.warning("Langfuse enabled but credentials missing. Skipping initialization.")
            return

        try:
            client_ctor = _resolve_langfuse_ctor()
            if client_ctor is None:
                raise ImportError(
                    "Langfuse client constructor not found in langfuse/otel/client modules."
                )

            kwargs: Dict[str, Any] = {
                "public_key": public_key,
                "secret_key": secret_key,
            }
            if host:
                kwargs["host"] = host

            # First try explicit kwargs. If SDK signature differs, fallback to env-based init.
            try:
                self._client = client_ctor(**kwargs)
            except TypeError:
                os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
                os.environ["LANGFUSE_SECRET_KEY"] = secret_key
                if host:
                    os.environ["LANGFUSE_HOST"] = host
                self._client = client_ctor()
            self.enabled = True

            if self.debug:
                self._logger.info(
                    "Langfuse initialized host=%s public=%s secret=%s",
                    host or "<default>",
                    _mask(public_key),
                    _mask(secret_key),
                )
        except Exception:
            self._client = None
            self.enabled = False
            if self.debug:
                self._logger.exception("Langfuse initialization failed.")

    @classmethod
    def from_settings(cls, settings: Any) -> "RAGObservability":
        """Builds observability instance from settings without raising runtime errors."""
        obs = getattr(settings, "observability", {}) or {}
        app = getattr(settings, "app", {}) or {}

        # Accept both project-native keys and common aliases.
        public_key = str(
            obs.get("langfuse_public_key")
            or obs.get("public_key")
            or obs.get("api_key")
            or ""
        )
        secret_key = str(
            obs.get("langfuse_secret_key")
            or obs.get("secret_key")
            or obs.get("secret")
            or ""
        )
        host = str(
            obs.get("langfuse_host")
            or obs.get("host")
            or obs.get("base_url")
            or ""
        )

        return cls(
            enabled=bool(obs.get("enabled", False)),
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            service_name=str(app.get("name", "rag-llama35-local")),
            environment=str(app.get("env", "dev")),
            debug=bool(obs.get("debug", False)),
        )

    def start_query_trace(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Starts a query trace and returns a context object for subsequent events."""
        ctx: Dict[str, Any] = {"trace_context": None, "pipeline_span": None}
        if not self.enabled or self._client is None:
            return ctx

        meta = {"service": self.service_name, "environment": self.environment}
        if metadata:
            meta.update(metadata)

        trace_id = self._call(self._client, "create_trace_id") or None
        trace_context = {"trace_id": trace_id} if trace_id else None
        self._call(
            self._client,
            "create_event",
            trace_context=trace_context,
            name="rag_query_start",
            input={"query": query},
            metadata=meta,
        )
        pipeline_span = self._call(
            self._client,
            "start_span",
            trace_context=trace_context,
            name="pipeline",
            input={"query": query},
            metadata=meta,
        )

        ctx["trace_context"] = trace_context
        ctx["pipeline_span"] = pipeline_span
        return ctx

    def log_retrieval(self, ctx: Dict[str, Any], retrieved: List[RetrievalResult]) -> None:
        """Logs retrieval payload (doc ids/titles/scores/sources)."""
        if not self.enabled:
            return

        rows = [
            {
                "doc_id": r.chunk.doc_id,
                "title": r.chunk.title,
                "score": r.score,
                "source": r.source,
            }
            for r in retrieved
        ]
        self._call(
            self._client,
            "create_event",
            trace_context=ctx.get("trace_context"),
            name="retrieval",
            metadata={"top_k": len(rows), "chunks": rows},
        )

    def log_validation(self, ctx: Dict[str, Any], passed: bool, reason: str = "") -> None:
        """Logs validation decision before generation."""
        if not self.enabled:
            return

        self._call(
            self._client,
            "create_event",
            trace_context=ctx.get("trace_context"),
            name="validation",
            metadata={"passed": passed, "reason": reason},
        )

    def finalize_success(self, ctx: Dict[str, Any], answer: str, latency_ms: float) -> None:
        """Closes trace/span on successful completion and flushes events."""
        if not self.enabled:
            return

        self._close_span(ctx.get("pipeline_span"), {"latency_ms": latency_ms})
        self._call(
            self._client,
            "create_event",
            trace_context=ctx.get("trace_context"),
            name="rag_query_complete",
            output={"answer": answer},
            metadata={"latency_ms": latency_ms, "status": "ok"},
        )
        self._call(self._client, "flush")

    def finalize_error(self, ctx: Dict[str, Any], error_message: str) -> None:
        """Closes trace/span on error without interrupting app runtime."""
        if not self.enabled:
            return

        self._close_span(ctx.get("pipeline_span"), {"error": error_message})
        self._call(
            self._client,
            "create_event",
            trace_context=ctx.get("trace_context"),
            name="rag_query_error",
            metadata={"status": "error", "error": error_message},
        )
        self._call(self._client, "flush")

    def _call(self, obj: Any, method: str, *args: Any, **kwargs: Any) -> Any:
        """Safely executes SDK methods; all exceptions are swallowed."""
        if obj is None:
            return None
        fn = getattr(obj, method, None)
        if fn is None:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception:
            if self.debug:
                self._logger.exception("Langfuse call failed: %s", method)
            return None

    def _close_span(self, span: Any, output: Dict[str, Any]) -> None:
        """Ends span across SDK variants and records output as a separate event if needed."""
        if span is None:
            return
        try:
            span.end(output=output)
            return
        except TypeError:
            # Older SDK signatures only support end() without payload.
            self._call(span, "end")
        except Exception:
            if self.debug:
                self._logger.exception("Langfuse span close failed.")


def _mask(value: str) -> str:
    """Masks key-like values for safe debug logs."""
    if not value:
        return "<empty>"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def _resolve_langfuse_ctor() -> Any:
    """Finds a Langfuse client constructor across SDK versions/layouts."""
    candidates = (
        ("langfuse", "Langfuse"),
        ("langfuse.otel", "Langfuse"),
        ("langfuse.client", "Langfuse"),
        ("langfuse.client", "LangfuseClient"),
        ("langfuse", "LangfuseClient"),
        ("langfuse", "Client"),
    )
    for module_name, attr in candidates:
        try:
            module = importlib.import_module(module_name)
            ctor = getattr(module, attr, None)
            if ctor is not None:
                return ctor
        except Exception:
            continue
    return None
