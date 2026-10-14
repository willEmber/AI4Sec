"""Optional OpenTelemetry tracing of every LangChain / LangGraph call.

A debugging aid: with ``TRACING_OTLP_ENDPOINT`` set, each agent turn becomes a
trace — every model request with the prompt it actually sent (after the
middleware cut and annotated it), every tool call with its arguments and
result, and how long each took — exported to a collector such as Phoenix.
LangGraph puts the session's ``thread_id`` into the callback metadata, so a
conversation's turns group as one Phoenix session without anything passed here.

What it does not see: calls made through ``llm_service`` (utility passes, the
report modes' raw gateway calls) go over plain ``httpx``, not LangChain.

The provider is kept private rather than installed as the global one, so
nothing else in the process starts exporting spans by accident.
"""
from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings

logger = logging.getLogger("scholar.tracing")

_provider: Any = None
_instrumentor: Any = None


def start_tracing() -> None:
    """Instrument LangChain once, if an endpoint is configured."""
    global _provider, _instrumentor
    settings = get_settings()
    endpoint = settings.tracing_otlp_endpoint.strip()
    if not endpoint or _provider is not None:
        return

    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.semconv.resource import ResourceAttributes
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": "scholar-backend",
                ResourceAttributes.PROJECT_NAME: settings.tracing_project or "scholar",
            }
        )
    )
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    _provider, _instrumentor = provider, instrumentor
    logger.info(f"Tracing LangChain calls to {endpoint} (project {settings.tracing_project})")


def stop_tracing() -> None:
    """Flush what is buffered and undo the instrumentation."""
    global _provider, _instrumentor
    if _provider is None:
        return
    try:
        _instrumentor.uninstrument()
    finally:
        _provider.shutdown()
        _provider, _instrumentor = None, None
