"""Replay-focused tests for extracted live input trim decisions."""

from __future__ import annotations

from arrangement_automation.live_input_trim_controller import (
    LiveInputTrimChannelState,
    LiveInputTrimController,
    LiveInputTrimReplaySnapshot,
)

from tests.replay_support import ReplayMixer, install_transport_blockers


class _ReplayTrimMixer(ReplayMixer):
    def set_channel_gain(self, channel: int, value: float) -> bool:
        self.gain_values[int(channel)] = float(value)
        self._record(
            "set_channel_gain",
            channel=channel,
            value=value,
            params={"value": float(value)},
            transport_kind="mutation",
        )
        return True

    def confirm_manual_write(self, operation, channel, desired_value, timeout_ms=300):
        return {
            "verification_id": f"verify-{channel}",
            "readback_status": "confirmed",
            "confirmed_value": float(self.gain_values.get(int(channel), desired_value)),
            "failure_reason": None,
            "timeout_ms": int(timeout_ms),
        }


def _live_trim_config(extra=None):
    cfg = {
        "automation": {
            "live_input_trim": {
                "run_background_loop": False,
                "analysis_only_mode": False,
                "live_apply_enabled": True,
                "confirm_live_apply": True,
                "activity_attack_hold_sec": 0.0,
                "min_main_signal_confidence": 0.1,
                "analysis_window_sec": 0.5,
                "apply_cooldown_sec": 0.0,
                "max_step_db_per_tick": 0.25,
                "max_boost_step_db": 0.15,
                "deadband_db": 0.1,
                "role_targets": {
                    "lead_vocal": {"peak_dbfs": -10.0, "main_sec": 0.5, "windows": 2},
                },
            }
        }
    }
    if extra:
        cfg["automation"]["live_input_trim"].update(extra)
    return cfg


def _trim_state(*, current_trim_db: float = 0.0) -> LiveInputTrimChannelState:
    return LiveInputTrimChannelState(
        audio_channel=1,
        mixer_channel=1,
        channel_name="Lead Vox",
        role="lead_vocal",
        role_confidence=0.99,
        current_trim_db=current_trim_db,
        target_trim_db=current_trim_db,
        state="ready_to_adjust",
    )


def _level_metrics(*, rms_db: float, peak_dbfs: float, true_peak_dbtp: float):
    return {
        "rms_db": float(rms_db),
        "peak_dbfs": float(peak_dbfs),
        "true_peak_dbtp": float(true_peak_dbtp),
    }


def test_replay_live_input_trim_decision_boundary_is_transport_free(monkeypatch):
    install_transport_blockers(monkeypatch)
    mixer = _ReplayTrimMixer(scenario="live_input_trim_decision_boundary")
    controller = LiveInputTrimController(
        mixer_client=mixer,
        config=_live_trim_config(),
    )
    state = _trim_state()
    metrics = _level_metrics(rms_db=-20.0, peak_dbfs=-2.0, true_peak_dbtp=-1.0)

    decision = controller._evaluate_trim_decision(
        state,
        metrics,
        metrics,
        now=0.0,
        ready=True,
        bleed_dominant=False,
    )

    assert mixer.incident_records() == []
    assert decision.reason == "hot_peak_guard"
    assert decision.blocked_reason == ""
    assert decision.delta_db == -0.25
    assert decision.write_plan is not None
    assert decision.replay_correlation_id
    assert decision.write_plan.replay_correlation_id == decision.replay_correlation_id


def test_replay_live_input_trim_decision_boundary_respects_cooldown(monkeypatch):
    install_transport_blockers(monkeypatch)
    controller = LiveInputTrimController(
        config=_live_trim_config({"apply_cooldown_sec": 2.0}),
    )
    state = _trim_state()
    state.last_apply_time = 0.0
    metrics = _level_metrics(rms_db=-20.0, peak_dbfs=-2.0, true_peak_dbtp=-1.0)

    decision = controller._evaluate_trim_decision(
        state,
        metrics,
        metrics,
        now=0.5,
        ready=True,
        bleed_dominant=False,
    )

    assert decision.blocked_reason == "cooldown"
    assert decision.delta_db == 0.0
    assert decision.write_plan is None


def test_replay_live_input_trim_executor_applies_precomputed_decision(monkeypatch):
    install_transport_blockers(monkeypatch)
    mixer = _ReplayTrimMixer(scenario="live_input_trim_executor")
    controller = LiveInputTrimController(
        mixer_client=mixer,
        config=_live_trim_config(),
    )
    state = _trim_state()
    metrics = _level_metrics(rms_db=-30.0, peak_dbfs=-18.0, true_peak_dbtp=-24.0)

    decision = controller._evaluate_trim_decision(
        state,
        metrics,
        metrics,
        now=0.0,
        ready=True,
        bleed_dominant=False,
    )
    executed = controller._execute_trim_decision(state, decision, now=0.0)

    records = mixer.incident_records()
    assert executed.sent is True
    assert executed.replay_correlation_id
    assert executed.readback_status == "confirmed"
    assert len(records) == 1
    assert records[0]["method"] == "set_channel_gain"
    assert records[0]["channel"] == 1
    assert state.current_trim_db == executed.target_trim_db


def test_replay_live_input_trim_dry_run_keeps_recorded_decision_transport_free(monkeypatch):
    install_transport_blockers(monkeypatch)
    source_controller = LiveInputTrimController(config=_live_trim_config())
    state = _trim_state()
    metrics = _level_metrics(rms_db=-30.0, peak_dbfs=-18.0, true_peak_dbtp=-24.0)

    decision = source_controller._evaluate_trim_decision(
        state,
        metrics,
        metrics,
        now=0.0,
        ready=True,
        bleed_dominant=False,
    )

    replay_mixer = _ReplayTrimMixer(scenario="live_input_trim_replay_dry_run")
    replay_controller = LiveInputTrimController(config=_live_trim_config())
    replayed = replay_controller.replay_decision(
        decision,
        mixer_client=replay_mixer,
        dry_run=True,
    )

    assert replay_mixer.incident_records() == []
    assert replayed["replay_correlation_id"]
    assert replayed["sent"] is False
    assert replayed["send_status"] == "not_sent"
    assert replayed["readback_status"] == "not_requested"
    assert replayed["current_trim_db"] == 0.0
    assert replayed["target_trim_db"] > replayed["current_trim_db"]


def test_replay_live_input_trim_requires_confirmation_before_real_apply(monkeypatch):
    install_transport_blockers(monkeypatch)
    source_controller = LiveInputTrimController(
        config=_live_trim_config({"confirm_live_apply": False}),
    )
    state = _trim_state()
    metrics = _level_metrics(rms_db=-30.0, peak_dbfs=-18.0, true_peak_dbtp=-24.0)

    decision = source_controller._evaluate_trim_decision(
        state,
        metrics,
        metrics,
        now=0.0,
        ready=True,
        bleed_dominant=False,
    )

    replay_mixer = _ReplayTrimMixer(scenario="live_input_trim_replay_confirm_gate")
    replay_controller = LiveInputTrimController(config=_live_trim_config())
    replayed = replay_controller.replay_decision(
        decision,
        mixer_client=replay_mixer,
        dry_run=False,
    )

    assert replay_mixer.incident_records() == []
    assert replayed["replay_correlation_id"]
    assert replayed["sent"] is False
    assert replayed["blocked_reason"] == "confirm_live_apply_required"
    assert replayed["send_status"] == "not_sent"
    assert replayed["readback_status"] == "not_requested"


def test_live_input_trim_snapshot_round_trip_preserves_checkpoint_state():
    controller = LiveInputTrimController(config=_live_trim_config())
    state = _trim_state(current_trim_db=1.25)
    state.main_signal_confidence = 0.87
    state.main_signal_sec = 1.5
    state.main_signal_windows = 3
    state.last_apply_time = 12.0
    state.last_update_time = 13.5
    controller.channels = [1]
    controller.channel_mapping = {1: 1}
    controller.states = {1: state}
    controller.started_at = 10.0
    controller.last_state = {"mode": "replay_ready", "channels": [1]}

    snapshot = controller.build_replay_snapshot(
        event_seq=7,
        replay_metadata={"checkpoint_reason": "unit_test"},
    )

    assert snapshot.version == "replay_state_snapshot/v1"
    assert snapshot.snapshot_id.startswith("replay_checkpoint::live_input_trim::")
    assert snapshot.transport_policy["dry_run_only"] is False

    restored_controller = LiveInputTrimController(config=_live_trim_config())
    restored_snapshot = restored_controller.restore_replay_snapshot(snapshot.to_dict())

    assert isinstance(restored_snapshot, LiveInputTrimReplaySnapshot)
    assert restored_snapshot.snapshot_id == snapshot.snapshot_id
    assert restored_controller.channels == [1]
    assert restored_controller.channel_mapping == {1: 1}
    assert restored_controller.states[1].current_trim_db == 1.25
    assert restored_controller.states[1].main_signal_windows == 3
    assert restored_controller.last_state == {"mode": "replay_ready", "channels": [1]}

    rebuilt_snapshot = restored_controller.build_replay_snapshot(
        event_seq=7,
        replay_metadata={"checkpoint_reason": "unit_test"},
    )
    assert rebuilt_snapshot.snapshot_id == snapshot.snapshot_id


def test_live_input_trim_snapshot_restore_reproduces_same_decision(monkeypatch):
    install_transport_blockers(monkeypatch)
    source_controller = LiveInputTrimController(config=_live_trim_config())
    source_state = _trim_state()
    source_state.main_signal_confidence = 0.95
    source_state.main_signal_sec = 1.25
    source_state.main_signal_windows = 4
    source_controller.channels = [1]
    source_controller.channel_mapping = {1: 1}
    source_controller.states = {1: source_state}
    metrics = _level_metrics(rms_db=-30.0, peak_dbfs=-18.0, true_peak_dbtp=-24.0)

    source_decision = source_controller._evaluate_trim_decision(
        source_state,
        metrics,
        metrics,
        now=5.0,
        ready=True,
        bleed_dominant=False,
    )
    snapshot = source_controller.build_replay_snapshot(
        event_seq=11,
        replay_metadata={"stage": "before_execute"},
    )

    restored_controller = LiveInputTrimController(config=_live_trim_config())
    restored_controller.restore_replay_snapshot(snapshot)
    restored_state = restored_controller.states[1]
    restored_decision = restored_controller._evaluate_trim_decision(
        restored_state,
        metrics,
        metrics,
        now=5.0,
        ready=True,
        bleed_dominant=False,
    )

    assert restored_decision.to_dict() == source_decision.to_dict()
