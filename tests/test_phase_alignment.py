"""
Tests for phase_alignment apply_corrections logic.

Reference channel mode: normalize to latest channel, eligibility filter,
detected/ignored_reason, applied_delay_ms.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pytest
from unittest.mock import MagicMock

try:
    from phase_alignment import PhaseAlignmentController
except ImportError:
    pytest.skip("phase_alignment module not importable", allow_module_level=True)


def _make_measurement(delay_gcc_ms, detected=True, coherence=0.5, phase_invert=0):
    """Create measurement dict for channel pair."""
    return {
        'delay_gcc_ms': delay_gcc_ms,
        'detected': detected,
        'coherence': coherence,
        'phase_invert': phase_invert,
    }


def _set_participating_from_measurements(controller, measurements):
    """Set _pending_participants from measurement keys for apply_corrections tests."""
    import re
    chs = set()
    for pk in measurements:
        if isinstance(pk, tuple) and len(pk) == 2:
            chs.add(pk[1])
        elif isinstance(pk, str):
            m = re.match(r'\((\d+),\s*(\d+)\)', pk)
            if m:
                chs.add(int(m.group(2)))
    controller._pending_participants = chs


@pytest.fixture
def mock_mixer():
    m = MagicMock()
    m.set_channel_delay = MagicMock()
    m.set_channel_phase_invert = MagicMock()
    m.send = MagicMock()
    m.is_connected = True
    return m


@pytest.fixture
def controller(mock_mixer):
    ctrl = PhaseAlignmentController(mixer_client=mock_mixer)
    ctrl.reference_channel = 2
    ctrl.channels_to_align = [7, 8]
    return ctrl


class TestApplyCorrectionsNormalizeToLatest:
    """Example: ref=2, ch7 delay=2ms, ch8 delay=4ms -> max=4, ref=4ms, ch7=2ms, ch8=0ms."""

    def test_example_snare_overheads(self, controller, mock_mixer):
        measurements = {
            '(2, 7)': _make_measurement(2.0),
            '(2, 8)': _make_measurement(4.0),
        }
        _set_participating_from_measurements(controller, measurements)
        success = controller.apply_corrections(measurements)
        assert success is True

        detail = controller.last_apply_detail
        assert detail[2]['applied_delay_ms'] == 4.0
        assert detail[2]['eligible_for_alignment'] is True
        assert detail[7]['applied_delay_ms'] == 2.0
        assert detail[7]['eligible_for_alignment'] is True
        assert detail[8]['applied_delay_ms'] == 0.0
        assert detail[8]['eligible_for_alignment'] is True

        assert len(controller.corrections) >= 2


class TestApplyCorrectionsEligibility:
    """Only channels with delay <= 10 ms participate (coherence not used)."""

    def test_coherence_ignored_all_eligible_by_delay(self, controller, mock_mixer):
        """Channels with low coherence are still eligible if delay <= 10 ms."""
        measurements = {
            '(2, 7)': _make_measurement(2.0, detected=False, coherence=0.1),
            '(2, 8)': _make_measurement(4.0),
        }
        _set_participating_from_measurements(controller, measurements)
        controller.apply_corrections(measurements)
        detail = controller.last_apply_detail
        assert detail[7]['eligible_for_alignment'] is True
        assert detail[7]['ignored_reason'] is None
        assert detail[7]['applied_delay_ms'] == 2.0
        assert detail[8]['eligible_for_alignment'] is True
        assert detail[2]['applied_delay_ms'] == 4.0


    def test_delay_above_10ms_excluded(self, controller, mock_mixer):
        measurements = {
            '(2, 7)': _make_measurement(2.0),
            '(2, 8)': _make_measurement(12.0),
        }
        _set_participating_from_measurements(controller, measurements)
        controller.apply_corrections(measurements)
        detail = controller.last_apply_detail
        assert detail[8]['eligible_for_alignment'] is False
        assert detail[8]['ignored_reason'] == 'delay_above_10ms'
        assert detail[8]['applied_delay_ms'] == 0.0
        assert detail[7]['eligible_for_alignment'] is True
        assert detail[2]['applied_delay_ms'] == 2.0  # only ch7 eligible


    def test_no_eligible_channels_ref_gets_zero(self, controller, mock_mixer):
        measurements = {
            '(2, 7)': _make_measurement(12.0),
            '(2, 8)': _make_measurement(15.0),
        }
        _set_participating_from_measurements(controller, measurements)
        controller.apply_corrections(measurements)
        detail = controller.last_apply_detail
        assert detail[2]['applied_delay_ms'] == 0.0
        assert detail[2]['eligible_for_alignment'] is False
        assert detail[7]['ignored_reason'] == 'delay_above_10ms'
        assert detail[8]['ignored_reason'] == 'delay_above_10ms'


class TestApplyCorrectionsEdgeCases:
    """Empty measurements, single channel, negative delay, backward compat."""

    def test_empty_measurements_returns_false(self, controller):
        result = controller.apply_corrections({})
        assert result is False
        assert controller.last_apply_detail == {}

    def test_single_eligible_channel(self, controller, mock_mixer):
        measurements = {
            '(2, 7)': _make_measurement(3.0),
        }
        _set_participating_from_measurements(controller, measurements)
        controller.apply_corrections(measurements)
        detail = controller.last_apply_detail
        assert detail[2]['applied_delay_ms'] == 3.0
        assert detail[7]['applied_delay_ms'] == 0.0
        assert detail[7]['eligible_for_alignment'] is True

    def test_negative_delay_uses_abs(self, controller, mock_mixer):
        measurements = {
            '(2, 7)': _make_measurement(-2.0),
            '(2, 8)': _make_measurement(-4.0),
        }
        _set_participating_from_measurements(controller, measurements)
        controller.apply_corrections(measurements)
        detail = controller.last_apply_detail
        assert detail[2]['applied_delay_ms'] == 4.0
        assert detail[7]['applied_delay_ms'] == 2.0
        assert detail[8]['applied_delay_ms'] == 0.0

    def test_tuple_keys(self, controller, mock_mixer):
        measurements = {
            (2, 7): _make_measurement(2.0),
            (2, 8): _make_measurement(4.0),
        }
        _set_participating_from_measurements(controller, measurements)
        controller.apply_corrections(measurements)
        detail = controller.last_apply_detail
        assert detail[2]['applied_delay_ms'] == 4.0
        assert detail[7]['applied_delay_ms'] == 2.0
        assert detail[8]['applied_delay_ms'] == 0.0

    def test_no_detected_in_measurement_still_eligible(self, controller, mock_mixer):
        """Measurement without 'detected' key; eligibility based only on delay."""
        measurements = {
            '(2, 7)': {
                'delay_gcc_ms': 2.0,
                'coherence': 0.5,
                'phase_invert': 0,
            },
        }
        _set_participating_from_measurements(controller, measurements)
        controller.apply_corrections(measurements)
        detail = controller.last_apply_detail
        assert detail[7]['eligible_for_alignment'] is True
        assert detail[7].get('detected', True) is True


class TestReferenceSignalDetection:
    """Reference presence detection in Phase Analyze window."""

    def test_detected_by_coherence_threshold(self, controller):
        controller._candidate_channels = [7]
        controller.reference_channel = 2
        controller.reference_coherence_min = 0.10
        controller._process_measurement_frame({
            '(2, 7)': {
                'coherence': 0.12,
                'gcc_peak_value': 0.01,
                'spectral_overlap_mid_high': 0.01,
            }
        })
        assert controller._reference_hits.get(7, 0) == 1

    def test_not_detected_below_coherence_threshold(self, controller):
        controller._candidate_channels = [7]
        controller.reference_channel = 2
        controller.reference_coherence_min = 0.10
        controller._process_measurement_frame({
            '(2, 7)': {
                'coherence': 0.08,
                'gcc_peak_value': 0.90,
                'spectral_overlap_mid_high': 0.90,
            }
        })
        assert controller._reference_hits.get(7, 0) == 0

    def test_nan_values_are_guarded(self, controller):
        controller._candidate_channels = [7]
        controller.reference_channel = 2
        controller.reference_coherence_min = 0.10
        controller._process_measurement_frame({
            '(2, 7)': {
                'coherence': float('nan'),
                'gcc_peak_value': float('nan'),
                'spectral_overlap_mid_high': float('nan'),
            }
        })
        assert controller._last_coherence.get(7, 0.0) == 0.0
        assert controller._last_gcc_peak.get(7, 0.0) == 0.0
        assert controller._reference_hits.get(7, 0) == 0

    def test_finalize_uses_coherence_threshold(self, controller):
        controller._candidate_channels = [7]
        controller.reference_channel = 2
        controller.reference_coherence_min = 0.10
        controller._last_coherence = {7: 0.09}
        controller._last_gcc_peak = {7: 0.25}
        controller._last_spectral_overlap = {7: 0.35}
        controller._locked_participants = set()
        controller.is_active = True
        controller.stop_analysis = MagicMock()
        controller._analysis_complete_callback = None
        controller.analyzer = MagicMock()
        controller.analyzer.get_measurements.return_value = {
            '(2, 7)': _make_measurement(2.0),
        }
        controller._finalize_analysis_window()
        assert 7 not in controller.channels_to_align
        assert controller.last_apply_detail[7]['ignored_reason'] == 'not_detected'

    def test_excluded_preset_is_skipped(self, controller):
        controller.reference_channel = 2
        controller.reference_exclude_presets = {'playback'}
        controller.channel_presets = {7: 'snare', 12: 'playback'}
        controller._candidate_channels = [7, 12]
        controller._process_measurement_frame({
            '(2, 7)': {
                'coherence': 0.55,
                'gcc_peak_value': 0.22,
                'spectral_overlap_mid_high': 0.30,
            },
            '(2, 12)': {
                'coherence': 0.80,
                'gcc_peak_value': 0.30,
                'spectral_overlap_mid_high': 0.40,
            }
        })
        assert controller._reference_hits.get(7, 0) == 1
        assert controller._reference_hits.get(12, 0) == 0
