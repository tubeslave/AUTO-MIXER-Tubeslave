"""Tests for the minimal live-apply contract and manual handler integration."""

import pytest

from live_apply import LiveApplyPolicy, LiveApplyRequest, LiveApplyService
import handlers.mixer_handlers as mixer_handlers


class FakeMixer:
    def __init__(
        self,
        *,
        fader_result=True,
        gain_result=True,
        last_send_status="sent",
        verify_result=None,
    ):
        self.fader_result = fader_result
        self.gain_result = gain_result
        self.last_send_status = last_send_status
        self.verify_result = verify_result
        self.calls = []
        self.verify_calls = []

    def set_channel_fader(self, channel, value):
        self.calls.append(("set_channel_fader", int(channel), float(value)))
        return self.fader_result

    def set_channel_gain(self, channel, value):
        self.calls.append(("set_channel_gain", int(channel), float(value)))
        return self.gain_result

    def set_eq_band(self, channel, band, freq, gain, q):
        self.calls.append(("set_eq_band", int(channel), band, freq, gain, q))
        return True

    def set_compressor(self, channel, threshold, ratio, attack, release):
        self.calls.append(
            ("set_compressor", int(channel), threshold, ratio, attack, release)
        )
        return True

    def get_last_send_status(self):
        return self.last_send_status

    def confirm_manual_write(self, operation, channel, desired_value, timeout_ms=300):
        self.verify_calls.append((operation, int(channel), float(desired_value), int(timeout_ms)))
        if self.verify_result is not None:
            return self.verify_result
        return {
            "verification_id": "verify-default",
            "readback_status": "confirmed",
            "confirmed_value": float(desired_value),
            "failure_reason": None,
            "timeout_ms": int(timeout_ms),
        }

    def get_state(self):
        return {}


class FakeShadowMixer(FakeMixer):
    def __init__(self):
        super().__init__(fader_result=False, gain_result=False, last_send_status="shadow_blocked")
        self.transport_contexts = []
        self._shadow_event = None

    def set_transport_context(self, **context):
        self.transport_contexts.append(dict(context))

    def clear_transport_context(self):
        return None

    def consume_shadow_transport_event(self):
        event = self._shadow_event
        self._shadow_event = None
        return event

    def set_channel_gain(self, channel, value):
        context = dict(self.transport_contexts[-1])
        self.calls.append(("set_channel_gain", int(channel), float(value)))
        self._shadow_event = {
            "type": "shadow_would_send",
            "audit_id": context["audit_id"],
            "replay_correlation_id": context["replay_correlation_id"],
            "source": context["source"],
            "operation": context["operation"],
            "channel": int(channel),
            "address": f"/ch/{int(channel)}/in/set/trim",
            "values": [float(value)],
            "send_status": "shadow_blocked",
            "dry_run_only": True,
        }
        return False


def test_live_apply_defaults_to_dry_run_and_audits():
    events = []
    service = LiveApplyService(audit_sink=events.append)
    mixer = FakeMixer()

    result = service.apply(
        LiveApplyRequest(source="manual_ui", operation="set_fader", channel=1, value=-6.0),
        mixer,
    )

    assert result.accepted is True
    assert result.dry_run is True
    assert result.send_attempted is False
    assert result.send_result is None
    assert result.send_status == "not_sent"
    assert result.readback_status == "not_requested"
    assert result.desired_value == -6.0
    assert result.confirmed_value is None
    assert result.verification_id is None
    assert result.audit_id
    assert mixer.calls == []
    assert mixer.verify_calls == []
    assert events[-1]["accepted"] is True
    assert events[-1]["audit_id"] == result.audit_id
    assert events[-1]["dry_run"] is True
    assert events[-1]["send_attempted"] is False


def test_live_apply_confirmation_does_not_bypass_default_dry_run():
    events = []
    service = LiveApplyService(audit_sink=events.append)
    mixer = FakeMixer()

    result = service.apply(
        LiveApplyRequest(
            source="manual_ui",
            operation="set_fader",
            channel=1,
            value=-6.0,
            confirm_live_apply=True,
        ),
        mixer,
    )

    assert result.accepted is True
    assert result.dry_run is True
    assert result.send_attempted is False
    assert result.send_status == "not_sent"
    assert mixer.calls == []
    assert mixer.verify_calls == []
    assert events[-1]["confirm_live_apply"] is True
    assert events[-1]["dry_run"] is True


def test_live_apply_dry_run_remains_allowed_under_panic_stop():
    events = []
    service = LiveApplyService(
        policy=LiveApplyPolicy(
            panic_stop_active=True,
            panic_stop_reason="operator_stop",
        ),
        audit_sink=events.append,
    )
    mixer = FakeMixer()

    result = service.apply(
        LiveApplyRequest(source="manual_ui", operation="set_fader", channel=1, value=-6.0),
        mixer,
    )

    assert result.accepted is True
    assert result.dry_run is True
    assert result.send_attempted is False
    assert result.send_status == "not_sent"
    assert result.readback_status == "not_requested"
    assert result.panic_stop_active is True
    assert result.panic_stop_scope == "manual_live_apply_only"
    assert result.panic_stop_reason == "operator_stop"
    assert mixer.calls == []
    assert mixer.verify_calls == []
    assert events[-1]["panic_stop_active"] is True
    assert events[-1]["send_attempted"] is False


def test_live_apply_blocks_without_confirm_for_real_send_and_audits():
    events = []
    service = LiveApplyService(audit_sink=events.append)
    mixer = FakeMixer()

    result = service.apply(
        LiveApplyRequest(
            source="manual_ui",
            operation="set_gain",
            channel=2,
            value=3.0,
            dry_run=False,
            confirm_live_apply=False,
        ),
        mixer,
    )

    assert result.accepted is False
    assert result.dry_run is False
    assert result.blocked_reason == "confirm_live_apply_required"
    assert result.send_attempted is False
    assert result.send_status == "not_sent"
    assert result.readback_status == "not_requested"
    assert result.verification_id is None
    assert result.audit_id
    assert mixer.calls == []
    assert mixer.verify_calls == []
    assert events[-1]["blocked_reason"] == "confirm_live_apply_required"
    assert events[-1]["audit_id"] == result.audit_id
    assert events[-1]["send_attempted"] is False


def test_live_apply_preserves_replay_correlation_id_in_blocked_and_audit_payloads():
    events = []
    service = LiveApplyService(audit_sink=events.append)
    mixer = FakeMixer()

    result = service.apply(
        LiveApplyRequest(
            source="manual_ui",
            operation="set_gain",
            channel=2,
            value=3.0,
            dry_run=False,
            confirm_live_apply=False,
            replay_correlation_id="corr::manual::gain::2",
        ),
        mixer,
    )

    assert result.replay_correlation_id == "corr::manual::gain::2"
    assert events[-1]["replay_correlation_id"] == "corr::manual::gain::2"


def test_live_apply_blocks_real_send_when_shadow_mode_intercepts_transport():
    events = []
    service = LiveApplyService(audit_sink=events.append)
    mixer = FakeShadowMixer()

    result = service.apply(
        LiveApplyRequest(
            source="manual_ui",
            operation="set_gain",
            channel=2,
            value=3.0,
            dry_run=False,
            confirm_live_apply=True,
            replay_correlation_id="corr::shadow::gain::2",
        ),
        mixer,
    )

    assert mixer.calls == [("set_channel_gain", 2, 3.0)]
    assert mixer.verify_calls == []
    assert result.accepted is False
    assert result.blocked_reason == "shadow_mode_active"
    assert result.send_attempted is True
    assert result.send_result is False
    assert result.send_status == "shadow_blocked"
    assert result.readback_status == "not_requested"
    assert result.replay_correlation_id == "corr::shadow::gain::2"
    assert events[-1]["audit_id"] == result.audit_id
    assert events[-1]["shadow_mode_active"] is True
    assert events[-1]["shadow_transport_event"]["audit_id"] == result.audit_id
    assert events[-1]["shadow_transport_event"]["replay_correlation_id"] == "corr::shadow::gain::2"
    assert mixer.transport_contexts[-1]["audit_id"] == result.audit_id
    assert mixer.transport_contexts[-1]["replay_correlation_id"] == "corr::shadow::gain::2"


def test_live_apply_blocks_real_set_fader_before_send_when_panic_stop_active():
    events = []
    service = LiveApplyService(
        policy=LiveApplyPolicy(
            default_dry_run=True,
            require_confirm_live_apply=True,
            panic_stop_active=True,
            panic_stop_reason="operator_stop",
        ),
        audit_sink=events.append,
    )
    mixer = FakeMixer()

    result = service.apply(
        LiveApplyRequest(
            source="manual_ui",
            operation="set_fader",
            channel=2,
            value=-3.0,
            dry_run=False,
            confirm_live_apply=True,
        ),
        mixer,
    )

    assert result.accepted is False
    assert result.blocked_reason == "panic_stop_active"
    assert result.send_attempted is False
    assert result.send_status == "not_sent"
    assert result.readback_status == "not_requested"
    assert result.panic_stop_active is True
    assert result.panic_stop_scope == "manual_live_apply_only"
    assert result.panic_stop_reason == "operator_stop"
    assert mixer.calls == []
    assert mixer.verify_calls == []
    assert events[-1]["panic_stop_active"] is True
    assert events[-1]["panic_stop_scope"] == "manual_live_apply_only"
    assert events[-1]["panic_stop_reason"] == "operator_stop"


def test_live_apply_blocks_real_set_gain_before_send_when_panic_stop_active():
    service = LiveApplyService(
        policy=LiveApplyPolicy(
            panic_stop_active=True,
            panic_stop_reason="operator_stop",
        ),
        audit_sink=lambda event: None,
    )
    mixer = FakeMixer()

    result = service.apply(
        LiveApplyRequest(
            source="manual_ui",
            operation="set_gain",
            channel=3,
            value=1.5,
            dry_run=False,
            confirm_live_apply=True,
        ),
        mixer,
    )

    assert result.accepted is False
    assert result.blocked_reason == "panic_stop_active"
    assert result.send_status == "not_sent"
    assert result.readback_status == "not_requested"
    assert mixer.calls == []
    assert mixer.verify_calls == []


def test_live_apply_allows_confirmed_real_send():
    events = []
    service = LiveApplyService(audit_sink=events.append)
    mixer = FakeMixer()

    result = service.apply(
        LiveApplyRequest(
            source="manual_ui",
            operation="set_fader",
            channel=4,
            value=-10.0,
            dry_run=False,
            confirm_live_apply=True,
        ),
        mixer,
    )

    assert result.accepted is True
    assert result.dry_run is False
    assert result.send_attempted is True
    assert result.send_result is True
    assert result.send_status == "sent"
    assert result.readback_status == "confirmed"
    assert result.desired_value == -10.0
    assert result.confirmed_value == -10.0
    assert result.verification_id == "verify-default"
    assert result.audit_id
    assert mixer.calls == [("set_channel_fader", 4, -10.0)]
    assert mixer.verify_calls == [("set_fader", 4, -10.0, 300)]
    assert events[-1]["accepted"] is True
    assert events[-1]["audit_id"] == result.audit_id


def test_live_apply_clamps_automatic_gain_requests_to_step_and_range():
    events = []
    service = LiveApplyService(audit_sink=events.append)
    mixer = FakeMixer()

    result = service.apply(
        LiveApplyRequest(
            source="auto_gain_safe_calibrator",
            operation="set_gain",
            channel=5,
            value=30.0,
            dry_run=False,
            confirm_live_apply=True,
            current_value=0.0,
            min_value=-18.0,
            max_value=18.0,
            max_step=3.0,
            metadata={"reason": "safe_gain"},
        ),
        mixer,
    )

    assert result.accepted is True
    assert result.dry_run is False
    assert result.desired_value == 3.0
    assert result.confirmed_value == 3.0
    assert result.guard_reasons == ("max_bound_clamped", "max_step_clamped")
    assert mixer.calls == [("set_channel_gain", 5, 3.0)]
    assert events[-1]["guard_reasons"] == ["max_bound_clamped", "max_step_clamped"]
    assert events[-1]["send_result"] is True


def test_live_apply_blocks_when_mixer_client_unavailable_and_audits():
    events = []
    service = LiveApplyService(audit_sink=events.append)

    result = service.apply(
        LiveApplyRequest(source="manual_ui", operation="set_fader", channel=8, value=-3.0),
        None,
    )

    assert result.accepted is False
    assert result.blocked_reason == "mixer_client_unavailable"
    assert result.send_attempted is False
    assert result.send_result is None
    assert result.send_status == "not_sent"
    assert result.readback_status == "not_requested"
    assert result.verification_id is None
    assert result.audit_id
    assert events[-1]["audit_id"] == result.audit_id
    assert events[-1]["blocked_reason"] == "mixer_client_unavailable"


def test_live_apply_send_failed_statuses():
    events = []
    mixer = FakeMixer(fader_result=False, last_send_status="failed")
    service = LiveApplyService(audit_sink=events.append)

    result = service.apply(
        LiveApplyRequest(
            source="manual_ui",
            operation="set_fader",
            channel=5,
            value=-9.0,
            dry_run=False,
            confirm_live_apply=True,
        ),
        mixer,
    )

    assert result.accepted is False
    assert result.send_status == "failed"
    assert result.readback_status == "not_requested"
    assert result.failure_reason == "failed"
    assert mixer.verify_calls == []


def test_live_apply_send_throttled_statuses():
    events = []
    mixer = FakeMixer(fader_result=False, last_send_status="throttled")
    service = LiveApplyService(audit_sink=events.append)

    result = service.apply(
        LiveApplyRequest(
            source="manual_ui",
            operation="set_fader",
            channel=6,
            value=-7.0,
            dry_run=False,
            confirm_live_apply=True,
        ),
        mixer,
    )

    assert result.accepted is False
    assert result.send_status == "throttled"
    assert result.failure_reason == "throttled"
    assert result.readback_status == "not_requested"


def test_live_apply_send_disconnected_statuses():
    mixer = FakeMixer(gain_result=False, last_send_status="disconnected")
    service = LiveApplyService(audit_sink=lambda event: None)

    result = service.apply(
        LiveApplyRequest(
            source="manual_ui",
            operation="set_gain",
            channel=6,
            value=1.0,
            dry_run=False,
            confirm_live_apply=True,
        ),
        mixer,
    )

    assert result.accepted is False
    assert result.send_status == "disconnected"
    assert result.failure_reason == "disconnected"
    assert result.readback_status == "not_requested"


def test_live_apply_readback_pending_status():
    mixer = FakeMixer(
        verify_result={
            "verification_id": "verify-pending",
            "readback_status": "pending",
            "confirmed_value": None,
            "failure_reason": None,
            "timeout_ms": 300,
        }
    )
    service = LiveApplyService(audit_sink=lambda event: None)

    result = service.apply(
        LiveApplyRequest(
            source="manual_ui",
            operation="set_fader",
            channel=10,
            value=-4.0,
            dry_run=False,
            confirm_live_apply=True,
        ),
        mixer,
    )

    assert result.accepted is False
    assert result.blocked_reason == "readback_pending"
    assert result.send_status == "sent"
    assert result.readback_status == "pending"
    assert result.confirmed_value is None
    assert result.verification_id == "verify-pending"


def test_live_apply_readback_timeout_status():
    mixer = FakeMixer(
        verify_result={
            "verification_id": "verify-timeout",
            "readback_status": "timeout",
            "confirmed_value": None,
            "failure_reason": "readback_timeout",
            "timeout_ms": 300,
        }
    )
    service = LiveApplyService(audit_sink=lambda event: None)

    result = service.apply(
        LiveApplyRequest(
            source="manual_ui",
            operation="set_gain",
            channel=10,
            value=2.0,
            dry_run=False,
            confirm_live_apply=True,
        ),
        mixer,
    )

    assert result.accepted is False
    assert result.blocked_reason == "readback_timeout"
    assert result.send_status == "sent"
    assert result.readback_status == "timeout"
    assert result.failure_reason == "readback_timeout"


def test_live_apply_readback_mismatch_status():
    mixer = FakeMixer(
        verify_result={
            "verification_id": "verify-mismatch",
            "readback_status": "mismatch",
            "confirmed_value": -30.0,
            "failure_reason": "readback_mismatch",
            "timeout_ms": 300,
        }
    )
    service = LiveApplyService(audit_sink=lambda event: None)

    result = service.apply(
        LiveApplyRequest(
            source="manual_ui",
            operation="set_fader",
            channel=11,
            value=-8.0,
            dry_run=False,
            confirm_live_apply=True,
        ),
        mixer,
    )

    assert result.accepted is False
    assert result.blocked_reason == "readback_mismatch"
    assert result.send_status == "sent"
    assert result.readback_status == "mismatch"
    assert result.confirmed_value == -30.0
    assert result.failure_reason == "readback_mismatch"


def test_live_apply_readback_unsupported_is_not_accepted():
    mixer = FakeMixer(
        verify_result={
            "verification_id": "verify-unsupported",
            "readback_status": "unsupported",
            "confirmed_value": None,
            "failure_reason": "readback_unsupported",
            "timeout_ms": 300,
        }
    )
    service = LiveApplyService(audit_sink=lambda event: None)

    result = service.apply(
        LiveApplyRequest(
            source="manual_ui",
            operation="set_gain",
            channel=12,
            value=1.5,
            dry_run=False,
            confirm_live_apply=True,
        ),
        mixer,
    )

    assert result.accepted is False
    assert result.blocked_reason == "readback_unsupported"
    assert result.send_status == "sent"
    assert result.readback_status == "unsupported"
    assert result.failure_reason == "readback_unsupported"


def test_live_apply_readback_disconnected_is_not_accepted():
    mixer = FakeMixer(
        verify_result={
            "verification_id": "verify-disconnected",
            "readback_status": "disconnected",
            "confirmed_value": None,
            "failure_reason": "console_disconnected",
            "timeout_ms": 300,
        }
    )
    service = LiveApplyService(audit_sink=lambda event: None)

    result = service.apply(
        LiveApplyRequest(
            source="manual_ui",
            operation="set_fader",
            channel=13,
            value=-9.0,
            dry_run=False,
            confirm_live_apply=True,
        ),
        mixer,
    )

    assert result.accepted is False
    assert result.blocked_reason == "console_disconnected"
    assert result.send_status == "sent"
    assert result.readback_status == "disconnected"
    assert result.failure_reason == "console_disconnected"


def test_live_apply_readback_query_send_failure_is_not_accepted():
    mixer = FakeMixer(
        verify_result={
            "verification_id": "verify-query-failed",
            "readback_status": "timeout",
            "confirmed_value": None,
            "failure_reason": "failed",
            "timeout_ms": 300,
        }
    )
    service = LiveApplyService(audit_sink=lambda event: None)

    result = service.apply(
        LiveApplyRequest(
            source="manual_ui",
            operation="set_gain",
            channel=14,
            value=0.5,
            dry_run=False,
            confirm_live_apply=True,
        ),
        mixer,
    )

    assert result.accepted is False
    assert result.blocked_reason == "readback_query_failed"
    assert result.send_status == "sent"
    assert result.readback_status == "timeout"
    assert result.failure_reason == "failed"


class FakeServer:
    def __init__(self, mixer, live_apply_service):
        self.mixer_client = mixer
        self.live_apply_service = live_apply_service
        self.connection_mode = "wing"
        self.sent_messages = []

    async def send_to_client(self, websocket, message):
        self.sent_messages.append((websocket, message))


@pytest.mark.asyncio
async def test_mixer_handler_set_fader_uses_dry_run_by_default():
    events = []
    mixer = FakeMixer()
    service = LiveApplyService(audit_sink=events.append)
    server = FakeServer(mixer, service)
    handlers = mixer_handlers.register_handlers(server)

    await handlers["set_fader"]("ws", {"channel": 7, "value": -5.0})

    assert mixer.calls == []
    assert events[-1]["operation"] == "set_fader"
    assert events[-1]["dry_run"] is True
    assert events[-1]["accepted"] is True
    payload = server.sent_messages[-1][1]
    assert payload["accepted"] is True
    assert payload["blocked"] is False
    assert payload["dry_run_only"] is True
    assert payload["send_result"] is None
    assert payload["send_status"] == "not_sent"
    assert payload["readback_status"] == "not_requested"
    assert payload["desired_value"] == -5.0
    assert payload["confirmed_value"] is None
    assert payload["verification_id"] is None
    assert payload["audit_id"] == events[-1]["audit_id"]
    assert payload["live_apply_source"] == "manual_ui"
    assert payload["target_type"] == "channel_fader"


@pytest.mark.asyncio
async def test_mixer_handler_payload_includes_replay_correlation_id():
    events = []
    mixer = FakeMixer()
    service = LiveApplyService(audit_sink=events.append)
    server = FakeServer(mixer, service)
    handlers = mixer_handlers.register_handlers(server)

    await handlers["set_gain"](
        "ws",
        {
            "channel": 3,
            "value": 2.5,
            "dry_run": False,
            "confirm_live_apply": True,
            "replay_correlation_id": "corr::handler::gain::3",
        },
    )

    payload = server.sent_messages[-1][1]
    assert events[-1]["replay_correlation_id"] == "corr::handler::gain::3"
    assert payload["replay_correlation_id"] == "corr::handler::gain::3"


@pytest.mark.asyncio
async def test_mixer_handler_reports_shadow_mode_block_with_audit_trace():
    events = []
    mixer = FakeShadowMixer()
    service = LiveApplyService(audit_sink=events.append)
    server = FakeServer(mixer, service)
    handlers = mixer_handlers.register_handlers(server)

    await handlers["set_gain"](
        "ws",
        {
            "channel": 3,
            "value": 2.5,
            "dry_run": False,
            "confirm_live_apply": True,
            "replay_correlation_id": "corr::handler::shadow::3",
        },
    )

    payload = server.sent_messages[-1][1]
    assert payload["accepted"] is False
    assert payload["blocked"] is True
    assert payload["blocked_reason"] == "shadow_mode_active"
    assert payload["send_status"] == "shadow_blocked"
    assert payload["shadow_mode_active"] is True
    assert payload["audit_id"] == events[-1]["audit_id"]
    assert payload["replay_correlation_id"] == "corr::handler::shadow::3"


@pytest.mark.asyncio
async def test_mixer_handler_set_gain_blocks_without_confirm():
    events = []
    mixer = FakeMixer()
    service = LiveApplyService(audit_sink=events.append)
    server = FakeServer(mixer, service)
    handlers = mixer_handlers.register_handlers(server)

    await handlers["set_gain"](
        "ws",
        {"channel": 3, "value": 2.5, "dry_run": False},
    )

    assert mixer.calls == []
    assert events[-1]["operation"] == "set_gain"
    assert events[-1]["blocked_reason"] == "confirm_live_apply_required"
    payload = server.sent_messages[-1][1]
    assert payload["accepted"] is False
    assert payload["blocked"] is True
    assert payload["blocked_reason"] == "confirm_live_apply_required"
    assert payload["dry_run_only"] is False
    assert payload["send_result"] is None
    assert payload["send_status"] == "not_sent"
    assert payload["readback_status"] == "not_requested"
    assert payload["message_for_user"] == "Live apply blocked: confirmation required."
    assert payload["target_type"] == "channel_gain"


@pytest.mark.asyncio
async def test_mixer_handler_confirmation_alone_does_not_send_live_osc():
    events = []
    mixer = FakeMixer()
    service = LiveApplyService(audit_sink=events.append)
    server = FakeServer(mixer, service)
    handlers = mixer_handlers.register_handlers(server)

    await handlers["set_fader"](
        "ws",
        {"channel": 3, "value": -4.5, "confirm_live_apply": True},
    )

    assert mixer.calls == []
    assert events[-1]["confirm_live_apply"] is True
    assert events[-1]["dry_run"] is True
    payload = server.sent_messages[-1][1]
    assert payload["accepted"] is True
    assert payload["blocked"] is False
    assert payload["dry_run_only"] is True
    assert payload["send_result"] is None
    assert payload["send_status"] == "not_sent"
    assert payload["message_for_user"] == "Dry-run only: no OSC command was sent."


@pytest.mark.asyncio
async def test_mixer_handler_blocks_live_apply_when_panic_stop_is_active():
    events = []
    mixer = FakeMixer()
    service = LiveApplyService(
        policy=LiveApplyPolicy(
            panic_stop_active=True,
            panic_stop_reason="operator_stop",
        ),
        audit_sink=events.append,
    )
    server = FakeServer(mixer, service)
    handlers = mixer_handlers.register_handlers(server)

    await handlers["set_fader"](
        "ws",
        {
            "channel": 3,
            "value": -4.0,
            "dry_run": False,
            "confirm_live_apply": True,
        },
    )

    assert mixer.calls == []
    assert mixer.verify_calls == []
    payload = server.sent_messages[-1][1]
    assert payload["accepted"] is False
    assert payload["blocked"] is True
    assert payload["blocked_reason"] == "panic_stop_active"
    assert payload["send_status"] == "not_sent"
    assert payload["readback_status"] == "not_requested"
    assert payload["panic_stop_active"] is True
    assert payload["panic_stop_scope"] == "manual_live_apply_only"
    assert payload["panic_stop_reason"] == "operator_stop"
    assert payload["message_for_user"] == "Live apply blocked: panic stop is active for manual live apply."
    assert events[-1]["panic_stop_active"] is True
    assert events[-1]["panic_stop_scope"] == "manual_live_apply_only"
    assert events[-1]["panic_stop_reason"] == "operator_stop"


@pytest.mark.asyncio
async def test_mixer_handler_set_gain_allows_confirmed_live_apply():
    events = []
    mixer = FakeMixer()
    service = LiveApplyService(audit_sink=events.append)
    server = FakeServer(mixer, service)
    handlers = mixer_handlers.register_handlers(server)

    await handlers["set_gain"](
        "ws",
        {
            "channel": 3,
            "value": 2.5,
            "dry_run": False,
            "confirm_live_apply": True,
        },
    )

    assert mixer.calls == [("set_channel_gain", 3, 2.5)]
    assert events[-1]["send_attempted"] is True
    assert events[-1]["send_result"] is True
    payload = server.sent_messages[-1][1]
    assert payload["accepted"] is True
    assert payload["blocked"] is False
    assert payload["dry_run_only"] is False
    assert payload["send_result"] is True
    assert payload["send_status"] == "sent"
    assert payload["readback_status"] == "confirmed"
    assert payload["desired_value"] == 2.5
    assert payload["confirmed_value"] == 2.5
    assert payload["verification_id"] == "verify-default"
    assert payload["message_for_user"] == "Live apply confirmed by mixer readback."
    assert payload["audit_id"] == events[-1]["audit_id"]


@pytest.mark.asyncio
async def test_mixer_handler_marks_readback_timeout_as_not_accepted():
    events = []
    mixer = FakeMixer(
        verify_result={
            "verification_id": "verify-timeout",
            "readback_status": "timeout",
            "confirmed_value": None,
            "failure_reason": "readback_timeout",
            "timeout_ms": 300,
        }
    )
    service = LiveApplyService(audit_sink=events.append)
    server = FakeServer(mixer, service)
    handlers = mixer_handlers.register_handlers(server)

    await handlers["set_fader"](
        "ws",
        {
            "channel": 4,
            "value": -6.0,
            "dry_run": False,
            "confirm_live_apply": True,
        },
    )

    payload = server.sent_messages[-1][1]
    assert payload["accepted"] is False
    assert payload["blocked"] is True
    assert payload["blocked_reason"] == "readback_timeout"
    assert payload["send_status"] == "sent"
    assert payload["readback_status"] == "timeout"
    assert payload["message_for_user"] == "Live apply sent, but readback timed out before confirmation."


@pytest.mark.asyncio
async def test_mixer_handler_marks_readback_query_failure_as_not_applied():
    events = []
    mixer = FakeMixer(
        verify_result={
            "verification_id": "verify-query-failed",
            "readback_status": "timeout",
            "confirmed_value": None,
            "failure_reason": "failed",
            "timeout_ms": 300,
        }
    )
    service = LiveApplyService(audit_sink=events.append)
    server = FakeServer(mixer, service)
    handlers = mixer_handlers.register_handlers(server)

    await handlers["set_gain"](
        "ws",
        {
            "channel": 4,
            "value": 1.0,
            "dry_run": False,
            "confirm_live_apply": True,
        },
    )

    payload = server.sent_messages[-1][1]
    assert payload["accepted"] is False
    assert payload["blocked"] is True
    assert payload["blocked_reason"] == "readback_query_failed"
    assert payload["send_status"] == "sent"
    assert payload["readback_status"] == "timeout"
    assert payload["failure_reason"] == "failed"
    assert payload["message_for_user"] == "Live apply sent, but readback query failed."


@pytest.mark.asyncio
async def test_mixer_handler_set_eq_is_blocked_until_gated_live_apply_exists():
    events = []
    mixer = FakeMixer()
    service = LiveApplyService(audit_sink=events.append)
    server = FakeServer(mixer, service)
    handlers = mixer_handlers.register_handlers(server)

    await handlers["set_eq"](
        "ws",
        {"channel": 5, "band": 2, "freq": 1200.0, "gain": -3.0, "q": 1.2},
    )

    assert mixer.calls == []
    assert events[-1]["operation"] == "set_eq"
    assert events[-1]["blocked_reason"] == "unsupported_operation"
    payload = server.sent_messages[-1][1]
    assert payload["accepted"] is False
    assert payload["blocked"] is True
    assert payload["blocked_reason"] == "unsupported_operation"
    assert payload["send_status"] == "not_sent"
    assert payload["readback_status"] == "not_requested"
    assert payload["target_type"] == "channel_eq"
    assert payload["message_for_user"] == "Live apply blocked: operation is not gated for console writes."


@pytest.mark.asyncio
async def test_mixer_handler_set_compressor_is_blocked_until_gated_live_apply_exists():
    events = []
    mixer = FakeMixer()
    service = LiveApplyService(audit_sink=events.append)
    server = FakeServer(mixer, service)
    handlers = mixer_handlers.register_handlers(server)

    await handlers["set_compressor"](
        "ws",
        {
            "channel": 6,
            "params": {"threshold": -18.0, "ratio": 4, "attack": 10, "release": 80},
        },
    )

    assert mixer.calls == []
    assert events[-1]["operation"] == "set_compressor"
    assert events[-1]["blocked_reason"] == "unsupported_operation"
    payload = server.sent_messages[-1][1]
    assert payload["accepted"] is False
    assert payload["blocked"] is True
    assert payload["blocked_reason"] == "unsupported_operation"
    assert payload["send_status"] == "not_sent"
    assert payload["readback_status"] == "not_requested"
    assert payload["target_type"] == "channel_compressor"
    assert payload["message_for_user"] == "Live apply blocked: operation is not gated for console writes."


@pytest.mark.asyncio
async def test_mixer_handler_returns_blocked_result_when_mixer_unavailable():
    events = []
    service = LiveApplyService(audit_sink=events.append)
    server = FakeServer(None, service)
    handlers = mixer_handlers.register_handlers(server)

    await handlers["set_fader"]("ws", {"channel": 9, "value": -2.0})

    assert events[-1]["blocked_reason"] == "mixer_client_unavailable"
    payload = server.sent_messages[-1][1]
    assert payload["accepted"] is False
    assert payload["blocked"] is True
    assert payload["blocked_reason"] == "mixer_client_unavailable"
    assert payload["send_result"] is None
    assert payload["send_status"] == "not_sent"
    assert payload["readback_status"] == "not_requested"
    assert payload["message_for_user"] == "Mixer unavailable: command was not sent."
