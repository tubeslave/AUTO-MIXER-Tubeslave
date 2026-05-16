from unittest.mock import MagicMock

from lufs_gain_staging import AnalysisState, SafeGainCalibrator


def _make_ready_calibrator():
    mixer = MagicMock()
    mixer.get_channel_gain.return_value = 1.0

    calibrator = SafeGainCalibrator(mixer_client=mixer, config={})
    calibrator.channel_mapping = {1: 7}
    calibrator.suggestions = {
        1: {
            "suggested_gain_db": 2.5,
        }
    }
    calibrator.state = AnalysisState.READY
    return calibrator, mixer


def test_apply_corrections_uses_live_apply_adapter_when_present():
    calibrator, mixer = _make_ready_calibrator()
    calibrator.live_apply_adapter = MagicMock(
        return_value={"status": "applied", "effective_value": 3.5}
    )

    applied = calibrator.apply_corrections([1])

    assert applied is True
    mixer.set_channel_gain.assert_not_called()
    calibrator.live_apply_adapter.assert_called_once()
    summary = calibrator.get_last_apply_summary()
    assert summary["applied_count"] == 1
    assert summary["dry_run_count"] == 0
    assert summary["channel_results"][0]["mixer_channel"] == 7


def test_apply_corrections_records_dry_run_without_direct_write():
    calibrator, mixer = _make_ready_calibrator()
    calibrator.live_apply_adapter = MagicMock(
        return_value={"status": "dry_run", "requested_value": 3.5}
    )

    applied = calibrator.apply_corrections([1])

    assert applied is False
    mixer.set_channel_gain.assert_not_called()
    calibrator.live_apply_adapter.assert_called_once()
    summary = calibrator.get_last_apply_summary()
    assert summary["applied_count"] == 0
    assert summary["dry_run_count"] == 1
    assert summary["blocked_count"] == 0
