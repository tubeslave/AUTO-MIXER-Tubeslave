"""
Prometheus metrics for AUTO-MIXER-Tubeslave.

Provides histograms, gauges, and counters for monitoring the mixing system.
Falls back gracefully if prometheus_client is not installed.
"""

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Info,
        start_http_server, CollectorRegistry, REGISTRY,
    )
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


class _DummyMetric:
    """No-op metric when prometheus_client is unavailable."""
    def inc(self, amount=1): pass
    def dec(self, amount=1): pass
    def set(self, value): pass
    def observe(self, value): pass
    def labels(self, *args, **kwargs): return self
    def info(self, val): pass
    def time(self):
        import contextlib
        @contextlib.contextmanager
        def _noop():
            yield
        return _noop()


class MixerMetrics:
    """Central metrics registry for the AUTO-MIXER system."""

    def __init__(self):
        self._started = False

        if HAS_PROMETHEUS:
            self.osc_latency = Histogram(
                "automixer_osc_latency_seconds",
                "OSC round-trip latency",
                buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
            )
            self.audio_processing_time = Histogram(
                "automixer_audio_processing_seconds",
                "Audio processing block time",
                buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1),
            )
            self.ws_message_time = Histogram(
                "automixer_ws_message_seconds",
                "WebSocket message handling time",
                buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5),
            )
            self.active_channels = Gauge(
                "automixer_active_channels",
                "Number of active audio channels",
            )
            self.fader_levels = Gauge(
                "automixer_fader_level",
                "Current fader level per channel",
                ["channel"],
            )
            self.lufs_levels = Gauge(
                "automixer_lufs_level",
                "Current LUFS level per channel",
                ["channel"],
            )
            self.cpu_usage = Gauge(
                "automixer_cpu_usage_percent",
                "CPU usage percentage",
            )
            self.osc_messages_sent = Counter(
                "automixer_osc_messages_sent_total",
                "Total OSC messages sent",
            )
            self.osc_messages_received = Counter(
                "automixer_osc_messages_received_total",
                "Total OSC messages received",
            )
            self.ws_connections = Counter(
                "automixer_ws_connections_total",
                "Total WebSocket connections",
            )
            self.feedback_events = Counter(
                "automixer_feedback_events_total",
                "Total feedback detection events",
            )
            self.system_info = Info(
                "automixer_system",
                "System information",
            )
        else:
            self.osc_latency = _DummyMetric()
            self.audio_processing_time = _DummyMetric()
            self.ws_message_time = _DummyMetric()
            self.active_channels = _DummyMetric()
            self.fader_levels = _DummyMetric()
            self.lufs_levels = _DummyMetric()
            self.cpu_usage = _DummyMetric()
            self.osc_messages_sent = _DummyMetric()
            self.osc_messages_received = _DummyMetric()
            self.ws_connections = _DummyMetric()
            self.feedback_events = _DummyMetric()
            self.system_info = _DummyMetric()

    def start_server(self, port: int = 9090) -> bool:
        """Start Prometheus metrics HTTP server."""
        if not HAS_PROMETHEUS:
            logger.warning("prometheus_client not available, metrics server disabled")
            return False
        if self._started:
            return True
        try:
            start_http_server(port)
            self._started = True
            logger.info(f"Prometheus metrics server started on port {port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            return False


# Singleton instance
_metrics: Optional[MixerMetrics] = None
_lock = threading.Lock()


def setup_metrics() -> MixerMetrics:
    """Initialize and return the global metrics instance."""
    global _metrics
    with _lock:
        if _metrics is None:
            _metrics = MixerMetrics()
    return _metrics


def get_metrics() -> MixerMetrics:
    """Get or create the global metrics instance."""
    if _metrics is None:
        return setup_metrics()
    return _metrics


def start_metrics_server(port: int = 9090) -> bool:
    """Start the Prometheus metrics server."""
    m = get_metrics()
    return m.start_server(port)
