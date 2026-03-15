"""Tests for metrics module."""
import pytest
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from metrics import MetricsCollector


class TestMetricsCollector:
    """Tests for the MetricsCollector class."""

    def test_increment_counter(self):
        """increment() increases a named counter by the specified value."""
        mc = MetricsCollector(enable_prometheus=False)
        mc.increment('osc_messages_sent')
        mc.increment('osc_messages_sent')
        mc.increment('osc_messages_sent', 3.0)
        assert mc._counters['osc_messages_sent'] == 5.0

    def test_set_gauge(self):
        """set_gauge() sets a gauge to the specified value, overwriting previous."""
        mc = MetricsCollector(enable_prometheus=False)
        mc.set_gauge('active_channels', 16)
        assert mc._gauges['active_channels'] == 16
        mc.set_gauge('active_channels', 24)
        assert mc._gauges['active_channels'] == 24

    def test_observe_histogram(self):
        """observe() records histogram values and trims at 10000 entries."""
        mc = MetricsCollector(enable_prometheus=False)
        for i in range(50):
            mc.observe('osc_latency_ms', float(i))
        assert len(mc._histograms['osc_latency_ms']) == 50
        assert mc._histograms['osc_latency_ms'][0] == 0.0
        assert mc._histograms['osc_latency_ms'][-1] == 49.0

    def test_get_all_includes_counters_and_gauges(self):
        """get_all() returns a dict with counters, gauges, and histograms."""
        mc = MetricsCollector(enable_prometheus=False)
        mc.increment('errors', 2)
        mc.set_gauge('cpu_usage_percent', 45.5)
        mc.observe('processing_time_ms', 3.2)
        mc.observe('processing_time_ms', 4.8)

        result = mc.get_all()
        assert result['counters']['errors'] == 2
        assert result['gauges']['cpu_usage_percent'] == 45.5
        assert 'histograms' in result
        assert 'processing_time_ms' in result['histograms']
        hist = result['histograms']['processing_time_ms']
        assert hist['count'] == 2
        assert hist['mean'] == pytest.approx(4.0, abs=0.01)

    def test_timer_context_manager(self):
        """timer() context manager records elapsed time to the histogram."""
        mc = MetricsCollector(enable_prometheus=False)
        with mc.timer('processing_time_ms'):
            # Do a trivial operation
            total = sum(range(1000))
        assert len(mc._histograms['processing_time_ms']) == 1
        elapsed = mc._histograms['processing_time_ms'][0]
        # Should be a positive number of milliseconds
        assert elapsed >= 0.0
        assert elapsed < 5000.0  # sanity upper bound

    def test_histogram_trimming(self):
        """Histograms trim to 5000 entries when exceeding 10000."""
        mc = MetricsCollector(enable_prometheus=False)
        for i in range(10001):
            mc.observe('test_hist', float(i))
        # After exceeding 10000, the list is trimmed to the last 5000 entries
        assert len(mc._histograms['test_hist']) == 5000
