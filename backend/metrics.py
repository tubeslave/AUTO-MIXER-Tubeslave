"""
Application metrics collection using Prometheus client.
Falls back to in-memory counters if prometheus_client unavailable.
"""
import time
import logging
from typing import Dict, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects application metrics."""

    def __init__(self, enable_prometheus: bool = True, port: int = 9090):
        self.port = port
        self._prom_available = False
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, list] = defaultdict(list)
        self._prom_counters = {}
        self._prom_gauges = {}
        self._prom_histograms = {}

        if enable_prometheus:
            try:
                from prometheus_client import Counter, Gauge, Histogram, start_http_server

                self._prom_counters = {
                    'osc_messages_sent': Counter('automixer_osc_messages_sent_total', 'Total OSC messages sent'),
                    'osc_messages_received': Counter('automixer_osc_messages_received_total', 'Total OSC messages received'),
                    'agent_cycles': Counter('automixer_agent_cycles_total', 'Total agent ODA cycles'),
                    'actions_applied': Counter('automixer_actions_applied_total', 'Total mixing actions applied'),
                    'feedback_events': Counter('automixer_feedback_events_total', 'Total feedback detection events'),
                    'errors': Counter('automixer_errors_total', 'Total errors'),
                }

                self._prom_gauges = {
                    'active_channels': Gauge('automixer_active_channels', 'Number of active channels'),
                    'mix_lufs': Gauge('automixer_mix_lufs', 'Current mix bus LUFS'),
                    'agent_cycle_time_ms': Gauge('automixer_agent_cycle_time_ms', 'Agent cycle duration in ms'),
                    'websocket_clients': Gauge('automixer_websocket_clients', 'Connected WebSocket clients'),
                    'cpu_usage_percent': Gauge('automixer_cpu_usage_percent', 'CPU usage percentage'),
                    'audio_buffer_fill': Gauge('automixer_audio_buffer_fill', 'Audio buffer fill level'),
                }

                self._prom_histograms = {
                    'osc_latency_ms': Histogram('automixer_osc_latency_ms', 'OSC round-trip latency',
                                                buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500]),
                    'processing_time_ms': Histogram('automixer_processing_time_ms', 'Audio processing time',
                                                    buckets=[0.5, 1, 2, 5, 10, 20, 50]),
                }

                try:
                    start_http_server(port)
                    logger.info(f"Prometheus metrics server started on port {port}")
                except OSError as e:
                    logger.warning(f"Could not start Prometheus server: {e}")

                self._prom_available = True

            except ImportError:
                logger.info("prometheus_client not available, using in-memory metrics")

    def increment(self, name: str, value: float = 1.0):
        """Increment a counter."""
        self._counters[name] += value
        if self._prom_available and name in self._prom_counters:
            self._prom_counters[name].inc(value)

    def set_gauge(self, name: str, value: float):
        """Set a gauge value."""
        self._gauges[name] = value
        if self._prom_available and name in self._prom_gauges:
            self._prom_gauges[name].set(value)

    def observe(self, name: str, value: float):
        """Record a histogram observation."""
        self._histograms[name].append(value)
        if len(self._histograms[name]) > 10000:
            self._histograms[name] = self._histograms[name][-5000:]
        if self._prom_available and name in self._prom_histograms:
            self._prom_histograms[name].observe(value)

    def get_all(self) -> Dict:
        """Get all metrics as a dict."""
        result = {'counters': dict(self._counters), 'gauges': dict(self._gauges)}
        for name, values in self._histograms.items():
            if values:
                import numpy as np
                arr = np.array(values[-1000:])
                result.setdefault('histograms', {})[name] = {
                    'count': len(values),
                    'mean': float(np.mean(arr)),
                    'p50': float(np.percentile(arr, 50)),
                    'p95': float(np.percentile(arr, 95)),
                    'p99': float(np.percentile(arr, 99)),
                    'max': float(np.max(arr)),
                }
        return result

    def timer(self, name: str):
        """Context manager for timing operations."""
        return _Timer(self, name)

class _Timer:
    def __init__(self, collector: MetricsCollector, name: str):
        self.collector = collector
        self.name = name
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed_ms = (time.perf_counter() - self.start) * 1000
        self.collector.observe(self.name, elapsed_ms)

# Singleton
_metrics: Optional[MetricsCollector] = None

def get_metrics(enable_prometheus: bool = True, port: int = 9090) -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector(enable_prometheus=enable_prometheus, port=port)
    return _metrics
