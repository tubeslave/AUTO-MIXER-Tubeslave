"""
Tests for backend/metrics.py — MixerMetrics, _DummyMetric, singleton accessors.

Tests work regardless of whether prometheus_client is installed.
"""

import pytest

try:
    from metrics import MixerMetrics, _DummyMetric, setup_metrics, get_metrics
except ImportError:
    pytest.skip("metrics module not importable", allow_module_level=True)

try:
    from prometheus_client import CollectorRegistry
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metrics():
    """Create a MixerMetrics instance, using an isolated registry when
    prometheus_client is available to avoid duplicate metric errors."""
    if HAS_PROMETHEUS:
        import metrics as _metrics_mod
        # Temporarily patch to use an isolated registry
        import prometheus_client
        registry = CollectorRegistry()
        original_histogram = prometheus_client.Histogram
        original_gauge = prometheus_client.Gauge
        original_counter = prometheus_client.Counter
        original_info = prometheus_client.Info

        def _Histogram(name, doc, buckets=None, **kwargs):
            kwargs.pop("registry", None)
            kw = {"registry": registry}
            if buckets is not None:
                kw["buckets"] = buckets
            return original_histogram(name, doc, **kw)

        def _Gauge(name, doc, labelnames=(), **kwargs):
            kwargs.pop("registry", None)
            return original_gauge(name, doc, labelnames, registry=registry)

        def _Counter(name, doc, **kwargs):
            kwargs.pop("registry", None)
            return original_counter(name, doc, registry=registry)

        def _Info(name, doc, **kwargs):
            kwargs.pop("registry", None)
            return original_info(name, doc, registry=registry)

        _metrics_mod.Histogram = _Histogram
        _metrics_mod.Gauge = _Gauge
        _metrics_mod.Counter = _Counter
        _metrics_mod.Info = _Info
        try:
            m = MixerMetrics()
        finally:
            _metrics_mod.Histogram = original_histogram
            _metrics_mod.Gauge = original_gauge
            _metrics_mod.Counter = original_counter
            _metrics_mod.Info = original_info
        return m
    else:
        return MixerMetrics()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_metric():
    """A standalone _DummyMetric instance."""
    return _DummyMetric()


@pytest.fixture
def mixer_metrics():
    """A fresh MixerMetrics instance with isolated registry."""
    return _make_metrics()


# ---------------------------------------------------------------------------
# _DummyMetric tests
# ---------------------------------------------------------------------------

class TestDummyMetric:

    def test_inc_no_error(self, dummy_metric):
        """inc() should be a no-op without raising."""
        dummy_metric.inc()
        dummy_metric.inc(5)

    def test_dec_no_error(self, dummy_metric):
        dummy_metric.dec()
        dummy_metric.dec(3)

    def test_set_no_error(self, dummy_metric):
        dummy_metric.set(42)

    def test_observe_no_error(self, dummy_metric):
        dummy_metric.observe(0.5)

    def test_labels_returns_self(self, dummy_metric):
        result = dummy_metric.labels("ch1")
        assert result is dummy_metric

    def test_time_context_manager(self, dummy_metric):
        """time() should return a no-op context manager."""
        with dummy_metric.time():
            pass  # should not raise

    def test_info_no_error(self, dummy_metric):
        dummy_metric.info({"version": "1.0"})


# ---------------------------------------------------------------------------
# MixerMetrics tests
# ---------------------------------------------------------------------------

class TestMixerMetrics:

    def test_has_expected_attributes(self, mixer_metrics):
        """MixerMetrics should have all expected metric attributes."""
        assert hasattr(mixer_metrics, "osc_latency")
        assert hasattr(mixer_metrics, "audio_processing_time")
        assert hasattr(mixer_metrics, "ws_message_time")
        assert hasattr(mixer_metrics, "active_channels")
        assert hasattr(mixer_metrics, "fader_levels")
        assert hasattr(mixer_metrics, "lufs_levels")
        assert hasattr(mixer_metrics, "cpu_usage")
        assert hasattr(mixer_metrics, "osc_messages_sent")
        assert hasattr(mixer_metrics, "osc_messages_received")
        assert hasattr(mixer_metrics, "ws_connections")
        assert hasattr(mixer_metrics, "feedback_events")
        assert hasattr(mixer_metrics, "system_info")

    def test_metrics_are_usable(self, mixer_metrics):
        """All metrics should accept basic operations without error."""
        mixer_metrics.osc_latency.observe(0.01)
        mixer_metrics.active_channels.set(8)
        mixer_metrics.osc_messages_sent.inc()
        mixer_metrics.fader_levels.labels(channel="1").set(-5.0)

    def test_start_server_without_prometheus(self):
        """start_server should return False when prometheus_client is missing."""
        if HAS_PROMETHEUS:
            pytest.skip("prometheus_client is installed; cannot test missing path")
        m = MixerMetrics()
        assert m.start_server(port=19999) is False


# ---------------------------------------------------------------------------
# Singleton accessors
# ---------------------------------------------------------------------------

class TestSingletonAccessors:

    def test_setup_metrics_returns_instance(self):
        m = setup_metrics()
        assert isinstance(m, MixerMetrics)

    def test_get_metrics_returns_same_or_new(self):
        m = get_metrics()
        assert isinstance(m, MixerMetrics)

    def test_get_metrics_consistent(self):
        """Multiple calls to get_metrics should return the same instance."""
        m1 = get_metrics()
        m2 = get_metrics()
        assert m1 is m2
