"""Safety-focused tests for the headless auto-soundcheck entry path."""

import sys

import pytest

import auto_soundcheck_engine as auto_soundcheck_module
from auto_soundcheck_engine import AutoSoundcheckEngine, ChannelInfo
from feedback_detector import FeedbackEvent


class FakeFeedbackMixer:
    def __init__(self):
        self.is_connected = True
        self.calls = []

    def set_eq_band(self, channel, band, freq, gain, q):
        self.calls.append(
            ("set_eq_band", int(channel), int(band), float(freq), float(gain), float(q))
        )
        return True


def test_auto_soundcheck_engine_defaults_to_dry_run():
    engine = AutoSoundcheckEngine()

    assert engine.auto_apply is False

    status = engine.get_status()
    assert status["live_apply_enabled"] is False
    assert status["dry_run"] is True


def test_cli_requires_explicit_live_apply_flag(monkeypatch):
    captured = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            captured["run_called"] = True

    monkeypatch.setattr(auto_soundcheck_module, "AutoSoundcheckEngine", FakeEngine)
    monkeypatch.setattr(sys, "argv", ["auto_soundcheck_engine.py"])

    auto_soundcheck_module.main()

    assert captured["auto_apply"] is False
    assert captured["run_called"] is True


def test_cli_live_apply_flag_is_fail_closed(monkeypatch):
    captured = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            captured["run_called"] = True

    monkeypatch.setattr(auto_soundcheck_module, "AutoSoundcheckEngine", FakeEngine)
    monkeypatch.setattr(sys, "argv", ["auto_soundcheck_engine.py", "--live-apply"])

    with pytest.raises(SystemExit):
        auto_soundcheck_module.main()

    assert captured == {}


def test_cli_feedback_emergency_flag_is_fail_closed(monkeypatch):
    captured = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            captured["run_called"] = True

    monkeypatch.setattr(auto_soundcheck_module, "AutoSoundcheckEngine", FakeEngine)
    monkeypatch.setattr(sys, "argv", ["auto_soundcheck_engine.py", "--allow-feedback-emergency"])

    with pytest.raises(SystemExit):
        auto_soundcheck_module.main()

    assert captured == {}


def test_auto_soundcheck_feedback_default_policy_blocks_direct_feedback_write():
    engine = AutoSoundcheckEngine(auto_apply=False, auto_discover=False)
    engine.mixer_client = FakeFeedbackMixer()
    engine.channels = {1: ChannelInfo(channel=1, name="Lead Vox", fader_db=-5.0)}

    status_before = engine.get_status()
    assert status_before["live_apply_enabled"] is False
    assert status_before["dry_run"] is True
    assert status_before["feedback_reaction_enabled"] is False
    assert status_before["feedback_reaction_policy"] == "blocked"

    engine._handle_feedback_event(
        1,
        FeedbackEvent(
            channel=1,
            frequency_hz=3150.0,
            magnitude_db=-6.5,
            action="notch",
            confidence=0.95,
        ),
    )

    status_after = engine.get_status()
    assert status_after["live_apply_enabled"] is False
    assert status_after["dry_run"] is True
    assert status_after["feedback_reaction_enabled"] is False
    assert status_after["feedback_reaction_policy"] == "blocked"
    assert engine.mixer_client.calls == []


@pytest.mark.parametrize("kwargs", [
    {"auto_apply": True},
    {"feedback_emergency_apply": True},
])
def test_direct_engine_live_write_modes_are_fail_closed(kwargs):
    with pytest.raises(ValueError, match="supervised"):
        AutoSoundcheckEngine(auto_discover=False, **kwargs)


def test_cli_rejects_conflicting_apply_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["auto_soundcheck_engine.py", "--live-apply", "--no-apply"],
    )

    with pytest.raises(SystemExit):
        auto_soundcheck_module.main()
