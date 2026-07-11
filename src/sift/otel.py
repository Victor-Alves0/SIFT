"""OpenTelemetry bridge for SIFT's observer events.

    from sift.otel import otel_observer
    sift = Sift(observer=otel_observer())

Each ``search`` / ``execute`` / ``run_code`` event becomes a span named
``sift.<event>`` carrying the event data as attributes (``sift.path``,
``sift.ok``, ``sift.ms``, …). Honest note: observer events fire AFTER the
operation, so these are point-in-time spans with the duration as an attribute —
enough for tracing visibility and dashboards, not for waterfall nesting.

Requires the ``otel`` extra:  pip install "sift-tools[otel]"
"""
from __future__ import annotations


def otel_observer(tracer=None):
    """Build an observer callable that emits OTel spans. Pass your own tracer or
    let it use the global provider's ``sift`` tracer."""
    if tracer is None:
        from opentelemetry import trace
        tracer = trace.get_tracer("sift")

    def observer(event: str, data: dict) -> None:
        with tracer.start_as_current_span(f"sift.{event}") as span:
            for key, value in data.items():
                if value is None:
                    continue
                if not isinstance(value, (bool, int, float, str)):
                    value = str(value)
                span.set_attribute(f"sift.{key}", value)

    return observer
