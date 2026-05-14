"""Replay-first regression tests for transport-safe console control flows."""

from __future__ import annotations

import pytest

from auto_soundcheck_engine import AutoSoundcheckEngine, ChannelInfo, ChannelSnapshot
from automixer.production_mix_v1.models import MixAction, MixCandidate, SafetyResult
from automixer.production_mix_v1.osc_queue import OSCCommandQueue
from feedback_detector import FeedbackEvent
import handlers.mixer_handlers as mixer_handlers
from live_apply import LiveApplyPolicy, LiveApplyRequest, LiveApplyService
from signal_metrics import ChannelMetrics

from tests.replay_support import RecordingSender, ReplayMixer, install_transport_blockers


def _safe_candidate(*, channel_id: int = 5) -> tuple[MixCandidate, SafetyResult]:
    action = MixAction(
        action_type="gain",
        target="Lead Vox",
        channel_id=channel_id,
        parameters={
            "gain_db": -1.5,
            "reason": "replay_test",
            "replay_correlation_id": f"corr::gain::{channel_id}",
        },
        rationale="Replay harness fixture action",
        stage="test_stage",
    )
    candidate = MixCandidate(
        "candidate_replay_gain",
        "test",
        [action],
        metadata={"candidate_type": "single_fix"},
    )
    safety = SafetyResult(candidate.name, passed=True, allowed_actions=[action])
    return candidate, safety


def test_replay_queue_dry_run_reconstructs_would_send_without_transport(monkeypatch: pytest.MonkeyPatch):
    install_transport_blockers(monkeypatch)
    sender = RecordingSender()
    queue = OSCCommandQueue(dry_run=True)
    candidate, safety = _safe_candidate(channel_id=5)

    queue.enqueue_candidate(candidate, safety)
    result = queue.flush(sender=sender)

    assert result.dry_run is True
    assert result.sent == []
    assert len(result.would_send) == 1
    assert result.warnings == ["dry_run_enabled_no_osc_sent"]
    assert sender.incident_records() == []
    assert result.would_send[0]["address"] == "/ch/05/gain"
    assert result.would_send[0]["dry_run_only"] is True
    assert result.would_send[0]["replay_correlation_id"] == "corr::gain::5"
    assert result.would_send[0]["args"] == [
        {
            "gain_db": -1.5,
            "reason": "replay_test",
            "replay_correlation_id": "corr::gain::5",
        }
    ]


def test_replay_queue_live_flush_records_incident_without_network(monkeypatch: pytest.MonkeyPatch):
    install_transport_blockers(monkeypatch)
    sender = RecordingSender()
    queue = OSCCommandQueue(dry_run=False)
    candidate, safety = _safe_candidate(channel_id=7)

    queue.enqueue_candidate(candidate, safety)
    result = queue.flush(sender=sender)

    records = sender.incident_records()
    assert result.dry_run is False
    assert result.would_send == []
    assert len(result.sent) == 1
    assert len(records) == 1
    assert records[0]["event_index"] == 1
    assert records[0]["address"] == "/ch/07/gain"
    assert records[0]["channel"] == 7
    assert records[0]["transport_kind"] == "mutation"
    assert result.sent[0]["replay_correlation_id"] == "corr::gain::7"
    assert records[0]["values"] == [
        {
            "gain_db": -1.5,
            "reason": "replay_test",
            "replay_correlation_id": "corr::gain::7",
        }
    ]


def test_replay_autosoundcheck_gain_recommendations_are_read_only(monkeypatch: pytest.MonkeyPatch):
    install_transport_blockers(monkeypatch)
    mixer = ReplayMixer(scenario="autosoundcheck_recommendations_read_only", channel_names={1: "Kick"})
    engine = AutoSoundcheckEngine(auto_apply=False, auto_discover=False)
    engine.mixer_client = mixer

    metrics = ChannelMetrics(channel=1)
    metrics.level.true_peak_dbtp = -18.0
    metrics.level.rms_db = -30.0
    metrics.level.crest_factor_db = 12.0
    engine.channels[1] = ChannelInfo(
        channel=1,
        name="Kick",
        preset="kick",
        recognized=True,
        has_signal=True,
        peak_db=-18.0,
        rms_db=-30.0,
        metrics=metrics,
        original_snapshot=ChannelSnapshot(channel=1, gain_db=1.0),
    )

    recommendations = engine.get_gain_recommendations()

    assert mixer.event_log == []
    rec = recommendations[1]
    assert rec["source"] == "auto_soundcheck_engine"
    assert rec["dry_run_only"] is True
    assert rec["live_apply_safe"] is False
    assert rec["metadata"]["auto_apply_enabled"] is False
    assert rec["recommended_target_trim_db"] > rec["current_trim_db"]


def test_replay_autosoundcheck_feedback_path_records_direct_mutation(monkeypatch: pytest.MonkeyPatch):
    install_transport_blockers(monkeypatch)
    mixer = ReplayMixer(scenario="autosoundcheck_feedback_direct_mutation", channel_names={1: "Lead Vox"})
    engine = AutoSoundcheckEngine(num_channels=1, auto_apply=False, auto_discover=False)
    engine.mixer_client = mixer
    engine.channels = {1: ChannelInfo(channel=1, name="Lead Vox", fader_db=-5.0)}

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

    records = mixer.incident_records()
    assert engine.auto_apply is False
    assert records == []


def test_replay_autosoundcheck_feedback_path_records_mutation_with_emergency_override(
    monkeypatch: pytest.MonkeyPatch,
):
    install_transport_blockers(monkeypatch)
    mixer = ReplayMixer(scenario="autosoundcheck_feedback_direct_mutation", channel_names={1: "Lead Vox"})
    engine = AutoSoundcheckEngine(
        num_channels=1,
        auto_apply=False,
        auto_discover=False,
        feedback_emergency_apply=True,
    )
    engine.mixer_client = mixer
    engine.channels = {1: ChannelInfo(channel=1, name="Lead Vox", fader_db=-5.0)}

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

    records = mixer.incident_records()
    assert engine.auto_apply is False
    assert len(records) == 1
    assert records[0]["method"] == "set_eq_band"
    assert records[0]["channel"] == 1
    assert records[0]["transport_kind"] == "mutation"
    assert records[0]["context"]["scenario"] == "autosoundcheck_feedback_direct_mutation"
    assert records[0]["source_file"] is not None
    assert records[0]["source_file"].endswith("backend/auto_soundcheck_engine.py")
    assert records[0]["source_function"] == "_handle_feedback_event"


def test_replay_live_apply_blocks_unconfirmed_request_without_mutation(monkeypatch: pytest.MonkeyPatch):
    install_transport_blockers(monkeypatch)
    events = []
    mixer = ReplayMixer(scenario="live_apply_unconfirmed_block")
    service = LiveApplyService(audit_sink=events.append)

    result = service.apply(
        LiveApplyRequest(
            source="manual_ui",
            operation="set_gain",
            channel=4,
            value=2.5,
            dry_run=False,
            confirm_live_apply=False,
        ),
        mixer,
    )

    assert result.accepted is False
    assert result.blocked_reason == "confirm_live_apply_required"
    assert result.send_status == "not_sent"
    assert mixer.incident_records() == []
    assert mixer.verify_calls == []
    assert events[-1]["blocked_reason"] == "confirm_live_apply_required"


def test_replay_live_apply_records_confirmed_incident_and_readback(monkeypatch: pytest.MonkeyPatch):
    install_transport_blockers(monkeypatch)
    events = []
    mixer = ReplayMixer(scenario="live_apply_confirmed_send")
    service = LiveApplyService(audit_sink=events.append)

    result = service.apply(
        LiveApplyRequest(
            source="manual_ui",
            operation="set_fader",
            channel=3,
            value=-6.0,
            dry_run=False,
            confirm_live_apply=True,
        ),
        mixer,
    )

    records = mixer.incident_records()
    assert result.accepted is True
    assert result.send_status == "sent"
    assert result.readback_status == "confirmed"
    assert len(records) == 1
    assert records[0]["method"] == "set_channel_fader"
    assert records[0]["channel"] == 3
    assert records[0]["value"] == pytest.approx(-6.0)
    assert mixer.verify_calls == [("set_fader", 3, -6.0, 300)]
    assert events[-1]["accepted"] is True
    assert events[-1]["audit_id"] == result.audit_id


class _ReplayServer:
    def __init__(self, mixer, live_apply_service):
        self.mixer_client = mixer
        self.live_apply_service = live_apply_service
        self.connection_mode = "wing"
        self.sent_messages = []

    async def send_to_client(self, websocket, message):
        self.sent_messages.append((websocket, message))


@pytest.mark.asyncio
async def test_replay_handler_timeout_payload_is_incident_reconstructable(monkeypatch: pytest.MonkeyPatch):
    install_transport_blockers(monkeypatch)
    events = []
    mixer = ReplayMixer(
        scenario="handler_readback_timeout",
        verify_result={
            "verification_id": "verify-timeout",
            "readback_status": "timeout",
            "confirmed_value": None,
            "failure_reason": "readback_timeout",
            "timeout_ms": 300,
        },
    )
    service = LiveApplyService(
        policy=LiveApplyPolicy(default_dry_run=True, require_confirm_live_apply=True),
        audit_sink=events.append,
    )
    server = _ReplayServer(mixer, service)
    handlers = mixer_handlers.register_handlers(server)

    await handlers["set_fader"](
        "ws",
        {
            "channel": 9,
            "value": -4.0,
            "dry_run": False,
            "confirm_live_apply": True,
        },
    )

    records = mixer.incident_records()
    payload = server.sent_messages[-1][1]
    assert len(records) == 1
    assert records[0]["method"] == "set_channel_fader"
    assert records[0]["channel"] == 9
    assert payload["accepted"] is False
    assert payload["blocked_reason"] == "readback_timeout"
    assert payload["send_status"] == "sent"
    assert payload["readback_status"] == "timeout"
    assert payload["message_for_user"] == "Live apply sent, but readback timed out before confirmation."
    assert payload["audit_id"] == events[-1]["audit_id"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation,handler_key,target_type",
    [
        ("set_eq", "set_eq", "channel_eq"),
        ("set_compressor", "set_compressor", "channel_compressor"),
    ],
)
async def test_replay_handler_unsupported_operation_is_blocked_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    handler_key: str,
    target_type: str,
):
    install_transport_blockers(monkeypatch)
    events = []
    mixer = ReplayMixer(scenario=f"handler_{operation}_blocked")
    service = LiveApplyService(audit_sink=events.append)
    server = _ReplayServer(mixer, service)
    handlers = mixer_handlers.register_handlers(server)

    await handlers[handler_key](
        "ws",
        {
            "channel": 11,
            "band": 2,
            "freq": 800,
            "gain": -3,
            "q": 1.0,
            "params": {"attack": 10, "release": 100},
            "dry_run": False,
            "confirm_live_apply": True,
        },
    )

    payload = server.sent_messages[-1][1]
    assert payload["accepted"] is False
    assert payload["blocked_reason"] == "unsupported_operation"
    assert payload["send_status"] == "not_sent"
    assert payload["readback_status"] == "not_requested"
    assert payload["target_type"] == target_type
    assert payload["operation"] == operation
    assert payload["message_for_user"] == "Live apply blocked: operation is not gated for console writes."
    assert payload["audit_id"] == events[-1]["audit_id"]
    assert mixer.incident_records() == []


@pytest.mark.parametrize(
    "operation,request_value",
    [("set_fader", -7.0), ("set_gain", 3.0)],
)
def test_replay_service_unsupported_or_tier_a_mutation_policy(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    request_value: float,
):
    install_transport_blockers(monkeypatch)
    events = []
    mixer = ReplayMixer(scenario="service_policy_matrix", fader_result=True, gain_result=True)
    service = LiveApplyService(audit_sink=events.append)

    dry_run_result = service.apply(
        LiveApplyRequest(
            source="manual_ui",
            operation=operation,
            channel=12,
            value=request_value,
            dry_run=True,
            confirm_live_apply=False,
        ),
        mixer,
    )

    assert dry_run_result.accepted is True
    assert dry_run_result.dry_run is True
    assert dry_run_result.send_attempted is False
    assert dry_run_result.send_status == "not_sent"
    assert dry_run_result.readback_status == "not_requested"
    assert mixer.incident_records() == []

    live_result = service.apply(
        LiveApplyRequest(
            source="manual_ui",
            operation=operation,
            channel=12,
            value=request_value,
            dry_run=False,
            confirm_live_apply=False,
        ),
        mixer,
    )

    if operation in {"set_fader", "set_gain"}:
        assert live_result.accepted is False
        assert live_result.blocked_reason == "confirm_live_apply_required"
        assert live_result.send_status == "not_sent"
        assert mixer.event_log == []
        assert mixer.verify_calls == []
    else:
        pytest.fail("Policy matrix only covers expected Tier A operations")

    assert len(events) == 2
    assert events[0]["accepted"] is True
    assert events[0]["dry_run"] is True
    assert events[0]["operation"] == operation
    assert events[1]["accepted"] is False
    assert events[1]["operation"] == operation
    assert events[1]["blocked_reason"] == "confirm_live_apply_required"
