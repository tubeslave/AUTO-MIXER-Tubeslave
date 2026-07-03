"""Direct-write inventory smoke tests using a fake mixer."""

from __future__ import annotations

from collections import deque
import inspect
import re
import socket
import time
from pathlib import Path
from typing import Any

import pytest

import auto_fader as auto_fader_module
import auto_soundcheck_engine as auto_soundcheck_engine_module
import server as server_module
from arrangement_automation.controller import ArrangementAutomationController
from auto_fader import AutoFaderController, ChannelFaderState
from auto_soundcheck_engine import (
    AutoSoundcheckEngine,
    ChannelInfo,
    INSTRUMENT_COMPRESSOR,
    INSTRUMENT_EQ_PRESETS,
)
from feedback_detector import FeedbackEvent
from server import AutoMixerServer


_OSC_CHANNEL_RE = re.compile(r"/ch/(\d+)")


class FakeMixer:
    """Minimal fake mixer that records write-capable calls instead of sending OSC."""

    is_connected = True

    def __init__(
        self,
        *,
        scenario: str,
        channel_names: dict[int, str] | None = None,
        fader_result: bool = True,
        last_send_status: str = "sent",
        verify_result: dict[str, Any] | None = None,
    ):
        self.scenario = scenario
        self.channel_names = dict(channel_names or {})
        self.fader_result = bool(fader_result)
        self.last_send_status = last_send_status
        self.verify_result = verify_result
        self.fader_values: dict[int, float] = {}
        self.event_log: list[dict[str, Any]] = []
        self._seq = 0

    def get_channel_fader(self, channel: int) -> float:
        return float(self.fader_values.get(int(channel), -5.0))

    def get_channel_name(self, channel: int) -> str:
        return self.channel_names.get(int(channel), f"Ch {int(channel)}")

    def get_all_channel_names(self, _max_channels: int) -> dict[int, str]:
        return dict(self.channel_names)

    def set_channel_fader(self, channel: int, value: float) -> bool:
        if self.fader_result:
            self.fader_values[int(channel)] = float(value)
        self._record(
            "set_channel_fader",
            channel=channel,
            value=value,
            params={"value": float(value)},
            transport_kind="mutation",
        )
        return self.fader_result

    def set_fader(self, channel: int, value: float) -> bool:
        self._record(
            "set_fader",
            channel=channel,
            value=value,
            params={"value": float(value)},
            transport_kind="mutation",
        )
        return True

    def set_gain(self, channel: int, value: float) -> bool:
        self._record(
            "set_gain",
            channel=channel,
            value=value,
            params={"value": float(value)},
            transport_kind="mutation",
        )
        return True

    def set_eq(self, channel: int, **params: Any) -> bool:
        self._record(
            "set_eq",
            channel=channel,
            value=None,
            params=params,
            transport_kind="mutation",
        )
        return True

    def set_eq_band(
        self,
        channel: int,
        band: int,
        freq: float,
        gain: float,
        q: float,
    ) -> bool:
        self._record(
            "set_eq_band",
            channel=channel,
            value=gain,
            params={
                "band": int(band),
                "freq": float(freq),
                "gain": float(gain),
                "q": float(q),
            },
            transport_kind="mutation",
        )
        return True

    def set_compressor(self, channel: int, **params: Any) -> bool:
        self._record(
            "set_compressor",
            channel=channel,
            value=None,
            params=params,
            transport_kind="mutation",
        )
        return True

    def set_send_level(self, channel: int, send_bus: int, value: float) -> bool:
        self._record(
            "set_send_level",
            channel=channel,
            value=value,
            params={"send_bus": int(send_bus), "value": float(value)},
            transport_kind="mutation",
        )
        return True

    def set_eq_on(self, channel: int, value: int) -> bool:
        self._record(
            "set_eq_on",
            channel=channel,
            value=int(value),
            params={"value": int(value)},
            transport_kind="mutation",
        )
        return True

    def set_compressor_on(self, channel: int, value: int) -> bool:
        self._record(
            "set_compressor_on",
            channel=channel,
            value=int(value),
            params={"value": int(value)},
            transport_kind="mutation",
        )
        return True

    def set_gate_on(self, channel: int, value: int) -> bool:
        self._record(
            "set_gate_on",
            channel=channel,
            value=int(value),
            params={"value": int(value)},
            transport_kind="mutation",
        )
        return True

    def set_low_cut(self, channel: int, *, enabled: int) -> bool:
        self._record(
            "set_low_cut",
            channel=channel,
            value=int(enabled),
            params={"enabled": int(enabled)},
            transport_kind="mutation",
        )
        return True

    def set_high_cut(self, channel: int, *, enabled: int) -> bool:
        self._record(
            "set_high_cut",
            channel=channel,
            value=int(enabled),
            params={"enabled": int(enabled)},
            transport_kind="mutation",
        )
        return True

    def set_pan(self, channel: int, value: float) -> bool:
        self._record(
            "set_pan",
            channel=channel,
            value=value,
            params={"value": float(value)},
            transport_kind="mutation",
        )
        return True

    def set_mute(self, channel: int, value: bool) -> bool:
        self._record(
            "set_mute",
            channel=channel,
            value=bool(value),
            params={"value": bool(value)},
            transport_kind="mutation",
        )
        return True

    def mute(self, channel: int) -> bool:
        return self.set_mute(channel, True)

    def unmute(self, channel: int) -> bool:
        return self.set_mute(channel, False)

    def reset_channel_processing(self, channel: int) -> bool:
        self._record(
            "reset_channel_processing",
            channel=channel,
            value=None,
            params={},
            transport_kind="mutation",
        )
        return True

    def send(self, address: str, *values: Any) -> bool:
        transport_kind = "mutation" if values else "query"
        value = values[0] if len(values) == 1 else list(values) if values else None
        self._record(
            "send",
            channel=self._channel_from_address(address),
            value=value,
            params={"address": address, "values": list(values)},
            transport_kind=transport_kind,
        )
        return True

    def get_last_send_status(self) -> str:
        return self.last_send_status

    def confirm_manual_write(
        self,
        operation: str,
        channel: int,
        desired_value: float,
        timeout_ms: int = 300,
    ) -> dict[str, Any]:
        if self.verify_result is not None:
            return dict(self.verify_result)
        return {
            "verification_id": f"verify-{operation}-{int(channel)}",
            "readback_status": "confirmed",
            "confirmed_value": float(desired_value),
            "failure_reason": None,
            "timeout_ms": int(timeout_ms),
        }

    def mutation_events(self) -> list[dict[str, Any]]:
        return [event for event in self.event_log if event["transport_kind"] == "mutation"]

    def _record(
        self,
        method: str,
        *,
        channel: int | None,
        value: Any,
        params: dict[str, Any],
        transport_kind: str,
    ) -> None:
        self._seq += 1
        source_file, source_function = self._source_location()
        self.event_log.append(
            {
                "seq": self._seq,
                "method": method,
                "channel": None if channel is None else int(channel),
                "value": value,
                "params": params,
                "transport_kind": transport_kind,
                "source_file": source_file,
                "source_function": source_function,
                "context": {"scenario": self.scenario},
            }
        )

    @staticmethod
    def _channel_from_address(address: str) -> int | None:
        match = _OSC_CHANNEL_RE.search(address or "")
        return int(match.group(1)) if match else None

    @staticmethod
    def _source_location() -> tuple[str | None, str | None]:
        interesting = {
            "controller.py",
            "auto_fader.py",
            "auto_soundcheck_engine.py",
            "server.py",
        }
        for frame_info in inspect.stack()[2:]:
            filename = Path(frame_info.filename).name
            if filename in interesting:
                return frame_info.filename, frame_info.function
        return None, None


@pytest.fixture(autouse=True)
def transport_blocking_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail fast if a test accidentally reaches real socket/pythonosc transport."""

    def _blocked(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Real transport is blocked in direct-write inventory tests")

    monkeypatch.setattr(socket.socket, "sendto", _blocked)

    try:
        from pythonosc import udp_client
    except ImportError:
        return

    for owner, attr in (
        (udp_client.UDPClient, "send"),
        (udp_client.UDPClient, "send_message"),
        (udp_client.SimpleUDPClient, "send_message"),
    ):
        if hasattr(owner, attr):
            monkeypatch.setattr(owner, attr, _blocked)


def _arrangement_config(
    *,
    live_apply_enabled: bool,
    analysis_only_mode: bool,
    confirm_live_apply: bool | None = None,
) -> dict[str, Any]:
    cfg = {
        "automation": {
            "arrangement_automation": {
                "run_background_loop": False,
                "section_hold_time_sec": 0.0,
                "activity_attack_hold_sec": 0.0,
                "analysis_only_mode": analysis_only_mode,
                "live_apply_enabled": live_apply_enabled,
                "min_confidence_for_live_apply": 0.4,
            }
        }
    }
    if confirm_live_apply is not None:
        cfg["automation"]["arrangement_automation"]["confirm_live_apply"] = bool(confirm_live_apply)
    elif live_apply_enabled and not analysis_only_mode:
        cfg["automation"]["arrangement_automation"]["confirm_live_apply"] = True
    return cfg


def _metrics() -> dict[int, dict[str, float]]:
    return {1: {"rms_db": -20.0, "peak_db": -8.0, "lufs": -21.0}}


def test_arrangement_automation_inventory_analysis_only_records_zero_mutation_writes():
    mixer = FakeMixer(
        scenario="arrangement_analysis_only_tick",
        channel_names={1: "Lead Vox"},
    )
    controller = ArrangementAutomationController(
        mixer_client=mixer,
        config=_arrangement_config(
            live_apply_enabled=False,
            analysis_only_mode=True,
        ),
    )
    controller.start([1], channel_names={1: "Lead Vox"})

    state = controller.process_once(metrics_by_channel=_metrics(), timestamp=0.0)
    controller.stop()

    assert mixer.event_log == []
    assert mixer.mutation_events() == []
    assert state["safe_offsets"][1] == pytest.approx(0.25)
    assert state["applied_offsets"] == {}


def test_arrangement_automation_inventory_live_enabled_records_fake_mixer_fader_write():
    mixer = FakeMixer(
        scenario="arrangement_live_enabled_tick",
        channel_names={1: "Lead Vox"},
    )
    controller = ArrangementAutomationController(
        mixer_client=mixer,
        config=_arrangement_config(
            live_apply_enabled=True,
            analysis_only_mode=False,
        ),
    )
    controller.start([1], channel_names={1: "Lead Vox"})

    state = controller.process_once(metrics_by_channel=_metrics(), timestamp=0.0)
    controller.stop()

    assert len(mixer.event_log) == 1
    event = mixer.event_log[0]
    assert event["seq"] == 1
    assert event["method"] == "set_channel_fader"
    assert event["channel"] == 1
    assert event["value"] == pytest.approx(-4.75)
    assert event["params"]["value"] == pytest.approx(-4.75)
    assert event["transport_kind"] == "mutation"
    assert event["context"]["scenario"] == "arrangement_live_enabled_tick"
    assert event["source_file"] is not None
    assert event["source_file"].endswith("backend/arrangement_automation/controller.py")
    assert event["source_function"] == "_apply_safe_offsets"
    assert state["applied_offsets"][1] == pytest.approx(-4.75)


def test_arrangement_automation_inventory_live_enabled_enforces_fader_ceiling():
    mixer = FakeMixer(
        scenario="arrangement_live_enabled_tick_ceiling",
        channel_names={1: "Lead Vox"},
    )
    controller = ArrangementAutomationController(
        mixer_client=mixer,
        config={
            "automation": {
                "arrangement_automation": {
                    "run_background_loop": False,
                    "section_hold_time_sec": 0.0,
                    "activity_attack_hold_sec": 0.0,
                    "analysis_only_mode": False,
                    "live_apply_enabled": True,
                    "fader_ceiling_db": -4.6,
                    "min_confidence_for_live_apply": 0.4,
                }
            }
        },
    )
    controller.start([1], channel_names={1: "Lead Vox"})

    state = controller.process_once(metrics_by_channel=_metrics(), timestamp=0.0)
    controller.stop()

    assert len(mixer.event_log) == 1
    event = mixer.event_log[0]
    assert event["value"] == pytest.approx(-4.6)
    assert state["applied_offsets"][1] == pytest.approx(-4.6)


def test_arrangement_automation_inventory_live_enabled_without_confirmation_records_zero_writes():
    mixer = FakeMixer(
        scenario="arrangement_live_enabled_no_confirmation",
        channel_names={1: "Lead Vox"},
    )
    controller = ArrangementAutomationController(
        mixer_client=mixer,
        config=_arrangement_config(
            live_apply_enabled=True,
            analysis_only_mode=False,
            confirm_live_apply=False,
        ),
    )
    controller.start([1], channel_names={1: "Lead Vox"})

    state = controller.process_once(metrics_by_channel=_metrics(), timestamp=0.0)
    controller.stop()

    assert mixer.event_log == []
    assert mixer.mutation_events() == []
    assert state["safe_offsets"][1] == pytest.approx(0.25)
    assert state["applied_offsets"] == {}


def test_arrangement_automation_replay_safe_offset_plan_applies_through_live_apply_gate():
    record_mixer = FakeMixer(
        scenario="arrangement_replay_record",
        channel_names={1: "Lead Vox"},
    )
    recorder = ArrangementAutomationController(
        mixer_client=record_mixer,
        config=_arrangement_config(
            live_apply_enabled=True,
            analysis_only_mode=False,
        ),
    )
    recorder.start([1], channel_names={1: "Lead Vox"})
    recorded_state = recorder.process_once(metrics_by_channel=_metrics(), timestamp=0.0)
    recorder.stop()

    replay_mixer = FakeMixer(
        scenario="arrangement_replay_playback",
        channel_names={1: "Lead Vox"},
    )
    replayer = ArrangementAutomationController(
        mixer_client=replay_mixer,
        config=_arrangement_config(
            live_apply_enabled=True,
            analysis_only_mode=False,
        ),
    )
    replayer.base_fader_positions = recorder.base_fader_positions.copy()
    replayer.channels = list(recorder.channels)
    replayer.channel_mapping = recorder.channel_mapping.copy()
    replayer.enabled = recorder.enabled
    replayer.live_apply_enabled = recorder.live_apply_enabled
    replayer.analysis_only_mode = recorder.analysis_only_mode
    replayer.confirm_live_apply = recorder.confirm_live_apply

    applied = replayer._apply_safe_offsets(
        recorded_state["safe_offsets"],
        recorded_state["blocked_offsets"],
    )
    replayer.last_state = replayer._empty_state()

    assert applied == recorded_state["applied_offsets"]
    assert replay_mixer.event_log
    assert replay_mixer.event_log[0]["value"] == pytest.approx(recorded_state["applied_offsets"][1])
    assert replay_mixer.event_log[0]["source_function"] == "_apply_safe_offsets"


def test_auto_fader_replay_auto_balance_plan_reapplies_through_live_apply_gate():
    replay_mixer = FakeMixer(
        scenario="auto_fader_replay_playback",
        channel_names={1: "Lead Vox"},
    )
    controller = AutoFaderController(
        mixer_client=replay_mixer,
        config={"automation": {"auto_fader": {}}},
    )
    state = ChannelFaderState(
        channel_id=1,
        mixer_channel=1,
        sample_rate=48000,
        instrument_type="lead_vocal",
    )
    state.current_fader = -5.0
    controller.channels = {1: state}

    controller.auto_balance_result = {
        1: {
            "correction": 0.0,
            "integrated_lufs": -22.0,
            "target_lufs": -22.0,
            "locked": False,
        }
    }
    controller.auto_balance_pass = 1

    controller.apply_auto_balance()

    assert replay_mixer.event_log
    event = replay_mixer.event_log[0]
    assert event["method"] == "set_channel_fader"
    assert event["value"] == pytest.approx(-5.0)
    assert event["source_function"] == "_apply_fader_target"


def test_auto_fader_inventory_apply_auto_balance_records_fake_mixer_fader_write():
    mixer = FakeMixer(
        scenario="auto_fader_apply_auto_balance",
        channel_names={1: "Lead Vox"},
    )
    controller = AutoFaderController(
        mixer_client=mixer,
        config={"automation": {"auto_fader": {}}},
    )

    state = ChannelFaderState(
        channel_id=1,
        mixer_channel=1,
        sample_rate=48000,
        instrument_type="lead_vocal",
    )
    state.current_fader = -5.0
    controller.channels = {1: state}
    controller.auto_balance_result = {
        1: {
            "correction": 1.0,
            "integrated_lufs": -23.0,
            "target_lufs": -22.0,
            "locked": False,
        }
    }

    applied = controller.apply_auto_balance()

    assert applied is True

    mutation_events = mixer.mutation_events()
    assert [event["method"] for event in mutation_events] == ["set_channel_fader"]

    event = mutation_events[0]
    assert event["channel"] == 1
    assert event["value"] == pytest.approx(-4.0)
    assert event["params"]["value"] == pytest.approx(-4.0)
    assert event["transport_kind"] == "mutation"
    assert event["context"]["scenario"] == "auto_fader_apply_auto_balance"
    assert event["source_file"] is not None
    assert event["source_file"].endswith("backend/auto_fader.py")
    assert event["source_function"] == "_apply_fader_target"
    assert mixer.get_channel_fader(1) == pytest.approx(-4.0)


def test_auto_fader_inventory_apply_auto_balance_enforces_min_max_fader_clip():
    mixer = FakeMixer(
        scenario="auto_fader_apply_auto_balance_clip",
        channel_names={1: "Lead Vox"},
    )
    controller = AutoFaderController(
        mixer_client=mixer,
        config={"automation": {"auto_fader": {}}},
    )

    state = ChannelFaderState(
        channel_id=1,
        mixer_channel=1,
        sample_rate=48000,
        instrument_type="lead_vocal",
    )
    state.current_fader = -55.0
    controller.channels = {1: state}
    controller.auto_balance_result = {
        1: {
            "correction": -20.0,
            "integrated_lufs": -23.0,
            "target_lufs": -22.0,
            "locked": False,
        }
    }

    applied = controller.apply_auto_balance()

    assert applied is True
    event = mixer.mutation_events()[0]
    assert event["value"] == pytest.approx(-25.0)
    assert state.current_fader == pytest.approx(-25.0)


def test_auto_fader_inventory_apply_auto_balance_without_confirmation_records_zero_writes():
    mixer = FakeMixer(
        scenario="auto_fader_apply_auto_balance_no_confirmation",
        channel_names={1: "Lead Vox"},
    )
    controller = AutoFaderController(
        mixer_client=mixer,
        config={"automation": {"auto_fader": {}}},
        confirm_live_apply=False,
    )

    state = ChannelFaderState(
        channel_id=1,
        mixer_channel=1,
        sample_rate=48000,
        instrument_type="lead_vocal",
    )
    state.current_fader = -5.0
    controller.channels = {1: state}
    controller.auto_balance_result = {
        1: {
            "correction": 1.0,
            "integrated_lufs": -23.0,
            "target_lufs": -22.0,
            "locked": False,
        }
    }

    applied = controller.apply_auto_balance()

    assert applied is True
    assert mixer.event_log == []
    assert mixer.mutation_events() == []
    assert mixer.get_channel_fader(1) == pytest.approx(-5.0)
    assert state.current_fader == pytest.approx(-5.0)


def test_auto_fader_inventory_apply_auto_balance_with_automation_frozen_records_zero_writes():
    mixer = FakeMixer(
        scenario="auto_fader_apply_auto_balance_frozen",
        channel_names={1: "Lead Vox"},
    )
    controller = AutoFaderController(
        mixer_client=mixer,
        config={"automation": {"auto_fader": {}}},
    )

    state = ChannelFaderState(
        channel_id=1,
        mixer_channel=1,
        sample_rate=48000,
        instrument_type="lead_vocal",
    )
    state.current_fader = -5.0
    controller.channels = {1: state}
    controller.auto_balance_result = {
        1: {
            "correction": 1.0,
            "integrated_lufs": -23.0,
            "target_lufs": -22.0,
            "locked": False,
        }
    }
    controller.set_automation_frozen(True)

    applied = controller.apply_auto_balance()

    assert applied is True
    assert mixer.event_log == []
    assert mixer.mutation_events() == []
    assert mixer.get_channel_fader(1) == pytest.approx(-5.0)
    assert state.current_fader == pytest.approx(-5.0)


def test_auto_fader_inventory_apply_auto_balance_with_locked_channel_records_zero_writes():
    mixer = FakeMixer(
        scenario="auto_fader_apply_auto_balance_locked_channel",
        channel_names={1: "Lead Vox"},
    )
    controller = AutoFaderController(
        mixer_client=mixer,
        config={"automation": {"auto_fader": {}}},
    )

    state = ChannelFaderState(
        channel_id=1,
        mixer_channel=1,
        sample_rate=48000,
        instrument_type="lead_vocal",
    )
    state.current_fader = -5.0
    state.locked = True
    controller.channels = {1: state}
    controller.auto_balance_result = {
        1: {
            "correction": 1.0,
            "integrated_lufs": -23.0,
            "target_lufs": -22.0,
            "locked": False,
        }
    }

    applied = controller.apply_auto_balance()

    assert applied is True
    assert mixer.event_log == []
    assert mixer.mutation_events() == []
    assert mixer.get_channel_fader(1) == pytest.approx(-5.0)
    assert state.current_fader == pytest.approx(-5.0)


def test_auto_fader_inventory_apply_auto_balance_with_channel_freeze_until_records_zero_writes():
    mixer = FakeMixer(
        scenario="auto_fader_apply_auto_balance_channel_freeze_until",
        channel_names={1: "Lead Vox"},
    )
    controller = AutoFaderController(
        mixer_client=mixer,
        config={"automation": {"auto_fader": {}}},
    )

    state = ChannelFaderState(
        channel_id=1,
        mixer_channel=1,
        sample_rate=48000,
        instrument_type="lead_vocal",
    )
    state.current_fader = -5.0
    controller.channels = {1: state}
    controller.auto_balance_result = {
        1: {
            "correction": 1.0,
            "integrated_lufs": -23.0,
            "target_lufs": -22.0,
            "locked": False,
        }
    }
    controller.channel_freeze_until = {1: time.time() + 60.0}

    applied = controller.apply_auto_balance()

    assert applied is True
    assert mixer.event_log == []
    assert mixer.mutation_events() == []
    assert mixer.get_channel_fader(1) == pytest.approx(-5.0)
    assert state.current_fader == pytest.approx(-5.0)


def test_auto_fader_start_realtime_fader_records_query_events_without_thread_start(
    monkeypatch: pytest.MonkeyPatch,
):
    mixer = FakeMixer(
        scenario="auto_fader_start_realtime_fader_query_only",
        channel_names={1: "Lead Vox"},
    )
    controller = AutoFaderController(
        mixer_client=mixer,
        config={"automation": {"auto_fader": {}}},
    )
    controller.is_active = True
    controller.channels = {
        1: ChannelFaderState(
            channel_id=1,
            mixer_channel=1,
            sample_rate=48000,
            instrument_type="lead_vocal",
        )
    }

    thread_state: dict[str, Any] = {"started": False, "target": None, "name": None}

    class FakeThread:
        def __init__(self, *, target: Any = None, daemon: bool | None = None, name: str | None = None):
            thread_state["target"] = target
            thread_state["name"] = name
            self.daemon = daemon

        def start(self) -> None:
            thread_state["started"] = True

        def is_alive(self) -> bool:
            return False

    time_values = iter([0.0, 1.0, 1.0])

    def _fake_time() -> float:
        return next(time_values, 1.0)

    monkeypatch.setattr(auto_fader_module.threading, "Thread", FakeThread)
    monkeypatch.setattr(auto_fader_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(auto_fader_module.time, "time", _fake_time)

    started = controller.start_realtime_fader()

    assert started is True
    assert thread_state["started"] is True
    assert thread_state["target"] == controller._realtime_control_loop
    assert thread_state["name"] == "AutoFaderRealtimeLoop"
    assert controller.realtime_enabled is True

    assert mixer.mutation_events() == []
    assert [event["method"] for event in mixer.event_log] == ["send"]

    event = mixer.event_log[0]
    assert event["channel"] == 1
    assert event["value"] is None
    assert event["params"] == {"address": "/ch/1/fdr", "values": []}
    assert event["transport_kind"] == "query"
    assert event["context"]["scenario"] == "auto_fader_start_realtime_fader_query_only"
    assert event["source_file"] is not None
    assert event["source_file"].endswith("backend/auto_fader.py")
    assert event["source_function"] == "start_realtime_fader"
    assert controller.channels[1].initial_fader_db == pytest.approx(-5.0)
    assert controller.channels[1].current_fader == pytest.approx(-5.0)


def test_auto_fader_realtime_control_loop_single_iteration_records_one_mutation(
    monkeypatch: pytest.MonkeyPatch,
):
    mixer = FakeMixer(
        scenario="auto_fader_realtime_control_loop_single_iteration",
        channel_names={1: "Lead Vox"},
    )
    controller = AutoFaderController(
        mixer_client=mixer,
        config={"automation": {"auto_fader": {}}},
    )
    controller.realtime_enabled = True
    controller.bleed_service = None
    controller.automation_frozen = False
    controller.channel_freeze_until = {}
    controller.envelopes = {}
    controller._audio_buffers = {1: deque()}

    state = ChannelFaderState(
        channel_id=1,
        mixer_channel=1,
        sample_rate=48000,
        instrument_type="lead_vocal",
    )
    state.current_lufs = -20.0
    state.pre_fader_lufs = -18.0
    state.current_true_peak = -6.0
    state.is_active = True
    state.calibration_complete = True
    state.initial_fader_db = -5.0
    state.current_fader = -5.0
    state.target_fader = -5.0
    state.baseline_lufs = -24.0
    state.baseline_peak = -12.0
    controller.channels = {1: state}

    wait_calls: list[float] = []

    def _wait_once(timeout: float) -> bool:
        wait_calls.append(timeout)
        controller.realtime_enabled = False
        return False

    monkeypatch.setattr(controller._stop_event, "wait", _wait_once)

    controller._realtime_control_loop()

    assert wait_calls == [controller.update_interval]
    assert [event["method"] for event in mixer.event_log] == ["set_channel_fader"]
    assert [event["method"] for event in mixer.mutation_events()] == ["set_channel_fader"]

    event = mixer.event_log[0]
    assert event["channel"] == 1
    assert event["value"] == pytest.approx(-3.0)
    assert event["params"] == {"value": pytest.approx(-3.0)}
    assert event["transport_kind"] == "mutation"
    assert event["context"]["scenario"] == "auto_fader_realtime_control_loop_single_iteration"
    assert event["source_file"] is not None
    assert event["source_file"].endswith("backend/auto_fader.py")
    assert event["source_function"] == "_apply_fader_target"
    assert mixer.get_channel_fader(1) == pytest.approx(-3.0)
    assert state.current_fader == pytest.approx(-3.0)


def test_auto_fader_realtime_control_loop_single_iteration_without_confirmation_records_zero_writes(
    monkeypatch: pytest.MonkeyPatch,
):
    mixer = FakeMixer(
        scenario="auto_fader_realtime_control_loop_single_iteration_no_confirmation",
        channel_names={1: "Lead Vox"},
    )
    controller = AutoFaderController(
        mixer_client=mixer,
        config={"automation": {"auto_fader": {}}},
        confirm_live_apply=False,
    )
    controller.realtime_enabled = True
    controller.bleed_service = None
    controller.automation_frozen = False
    controller.channel_freeze_until = {}
    controller.envelopes = {}
    controller._audio_buffers = {1: deque()}

    state = ChannelFaderState(
        channel_id=1,
        mixer_channel=1,
        sample_rate=48000,
        instrument_type="lead_vocal",
    )
    state.current_lufs = -20.0
    state.pre_fader_lufs = -18.0
    state.current_true_peak = -6.0
    state.is_active = True
    state.calibration_complete = True
    state.initial_fader_db = -5.0
    state.current_fader = -5.0
    state.target_fader = -5.0
    state.baseline_lufs = -24.0
    state.baseline_peak = -12.0
    controller.channels = {1: state}

    wait_calls: list[float] = []

    def _wait_once(timeout: float) -> bool:
        wait_calls.append(timeout)
        controller.realtime_enabled = False
        return False

    monkeypatch.setattr(controller._stop_event, "wait", _wait_once)

    controller._realtime_control_loop()

    assert wait_calls == [controller.update_interval]
    assert mixer.event_log == []
    assert mixer.mutation_events() == []
    assert mixer.get_channel_fader(1) == pytest.approx(-5.0)
    assert state.current_fader == pytest.approx(-5.0)


def test_auto_fader_realtime_control_loop_single_iteration_with_automation_frozen_records_zero_writes(
    monkeypatch: pytest.MonkeyPatch,
):
    mixer = FakeMixer(
        scenario="auto_fader_realtime_control_loop_single_iteration_frozen",
        channel_names={1: "Lead Vox"},
    )
    controller = AutoFaderController(
        mixer_client=mixer,
        config={"automation": {"auto_fader": {}}},
    )
    controller.realtime_enabled = True
    controller.bleed_service = None
    controller.automation_frozen = True
    controller.channel_freeze_until = {}
    controller.envelopes = {}
    controller._audio_buffers = {1: deque()}

    state = ChannelFaderState(
        channel_id=1,
        mixer_channel=1,
        sample_rate=48000,
        instrument_type="lead_vocal",
    )
    state.current_lufs = -20.0
    state.pre_fader_lufs = -18.0
    state.current_true_peak = -6.0
    state.is_active = True
    state.calibration_complete = True
    state.initial_fader_db = -5.0
    state.current_fader = -5.0
    state.target_fader = -5.0
    state.baseline_lufs = -24.0
    state.baseline_peak = -12.0
    controller.channels = {1: state}

    wait_calls: list[float] = []

    def _wait_once(timeout: float) -> bool:
        wait_calls.append(timeout)
        controller.realtime_enabled = False
        return False

    monkeypatch.setattr(controller._stop_event, "wait", _wait_once)

    controller._realtime_control_loop()

    assert wait_calls == [controller.update_interval]
    assert mixer.event_log == []
    assert mixer.mutation_events() == []
    assert mixer.get_channel_fader(1) == pytest.approx(-5.0)
    assert state.current_fader == pytest.approx(-5.0)


def test_auto_fader_realtime_control_loop_single_iteration_with_channel_freeze_until_records_zero_writes(
    monkeypatch: pytest.MonkeyPatch,
):
    mixer = FakeMixer(
        scenario="auto_fader_realtime_control_loop_single_iteration_channel_frozen",
        channel_names={1: "Lead Vox"},
    )
    controller = AutoFaderController(
        mixer_client=mixer,
        config={"automation": {"auto_fader": {}}},
    )
    controller.realtime_enabled = True
    controller.bleed_service = None
    controller.automation_frozen = False
    controller.channel_freeze_until = {1: 160.0}
    controller.envelopes = {}
    controller._audio_buffers = {1: deque()}

    state = ChannelFaderState(
        channel_id=1,
        mixer_channel=1,
        sample_rate=48000,
        instrument_type="lead_vocal",
    )
    state.current_lufs = -20.0
    state.pre_fader_lufs = -18.0
    state.current_true_peak = -6.0
    state.is_active = True
    state.calibration_complete = True
    state.initial_fader_db = -5.0
    state.current_fader = -5.0
    state.target_fader = -5.0
    state.baseline_lufs = -24.0
    state.baseline_peak = -12.0
    controller.channels = {1: state}

    wait_calls: list[float] = []

    def _wait_once(timeout: float) -> bool:
        wait_calls.append(timeout)
        controller.realtime_enabled = False
        return False

    monkeypatch.setattr(controller._stop_event, "wait", _wait_once)
    monkeypatch.setattr(auto_fader_module.time, "time", lambda: 100.0)

    controller._realtime_control_loop()

    assert wait_calls == [controller.update_interval]
    assert mixer.event_log == []
    assert mixer.mutation_events() == []
    assert mixer.get_channel_fader(1) == pytest.approx(-5.0)
    assert state.current_fader == pytest.approx(-5.0)


def test_auto_soundcheck_reset_channels_records_reset_writes(
    monkeypatch: pytest.MonkeyPatch,
):
    mixer = FakeMixer(
        scenario="auto_soundcheck_reset_channels",
        channel_names={1: "Lead Vox", 2: "Guitar"},
    )
    engine = AutoSoundcheckEngine(num_channels=2, auto_apply=False, auto_discover=False)
    engine.mixer_client = mixer
    engine.channels = {
        1: ChannelInfo(channel=1, name="Lead Vox"),
        2: ChannelInfo(channel=2, name="Guitar"),
    }

    monkeypatch.setattr(auto_soundcheck_engine_module.time, "sleep", lambda _seconds: None)

    engine._reset_channels()

    assert [event["method"] for event in mixer.event_log] == [
        "reset_channel_processing",
        "reset_channel_processing",
    ]
    assert [event["channel"] for event in mixer.event_log] == [1, 2]
    assert [event["transport_kind"] for event in mixer.event_log] == [
        "mutation",
        "mutation",
    ]

    for event in mixer.event_log:
        assert event["source_file"] is not None
        assert event["source_file"].endswith("backend/auto_soundcheck_engine.py")
        assert event["source_function"] == "_reset_channels"
        assert event["context"]["scenario"] == "auto_soundcheck_reset_channels"

    assert engine.channels[1].was_reset is True
    assert engine.channels[2].was_reset is True


def test_auto_soundcheck_feedback_notch_is_blocked_by_default():
    mixer = FakeMixer(
        scenario="auto_soundcheck_feedback_notch_dry_run",
        channel_names={1: "Lead Vox"},
    )
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

    assert engine.auto_apply is False
    assert mixer.event_log == []


def test_auto_soundcheck_feedback_notch_blocks_without_override():
    mixer = FakeMixer(
        scenario="auto_soundcheck_feedback_notch_dry_run",
        channel_names={1: "Lead Vox"},
    )
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

    assert engine.auto_apply is False
    assert [event["method"] for event in mixer.event_log] == ["set_eq_band"]

    event = mixer.event_log[0]
    assert event["channel"] == 1
    assert event["value"] == pytest.approx(-6.0)
    assert event["params"] == {
        "band": 4,
        "freq": pytest.approx(3150.0),
        "gain": pytest.approx(-6.0),
        "q": pytest.approx(10.0),
    }
    assert event["transport_kind"] == "mutation"
    assert event["context"]["scenario"] == "auto_soundcheck_feedback_notch_dry_run"
    assert event["source_file"] is not None
    assert event["source_file"].endswith("backend/auto_soundcheck_engine.py")
    assert event["source_function"] == "_handle_feedback_event"


def test_auto_soundcheck_feedback_fader_reduce_is_blocked_by_default():
    mixer = FakeMixer(
        scenario="auto_soundcheck_feedback_fader_reduce_dry_run",
        channel_names={1: "Lead Vox"},
    )
    engine = AutoSoundcheckEngine(num_channels=1, auto_apply=False, auto_discover=False)
    engine.mixer_client = mixer
    info = ChannelInfo(channel=1, name="Lead Vox", fader_db=-5.0)
    engine.channels = {1: info}

    engine._handle_feedback_event(
        1,
        FeedbackEvent(
            channel=1,
            frequency_hz=2500.0,
            magnitude_db=-4.0,
            action="fader_reduce",
            confidence=0.8,
        ),
    )

    assert engine.auto_apply is False
    assert mixer.event_log == []
    assert info.fader_db == pytest.approx(-5.0)


def test_auto_soundcheck_feedback_fader_reduce_records_write_with_override():
    mixer = FakeMixer(
        scenario="auto_soundcheck_feedback_fader_reduce_dry_run",
        channel_names={1: "Lead Vox"},
    )
    engine = AutoSoundcheckEngine(
        num_channels=1,
        auto_apply=False,
        auto_discover=False,
        feedback_emergency_apply=True,
    )
    engine.mixer_client = mixer
    info = ChannelInfo(channel=1, name="Lead Vox", fader_db=-5.0)
    engine.channels = {1: info}

    engine._handle_feedback_event(
        1,
        FeedbackEvent(
            channel=1,
            frequency_hz=2500.0,
            magnitude_db=-4.0,
            action="fader_reduce",
            confidence=0.8,
        ),
    )

    assert engine.auto_apply is False
    assert [event["method"] for event in mixer.event_log] == ["set_fader"]

    event = mixer.event_log[0]
    assert event["channel"] == 1
    assert event["value"] == pytest.approx(-8.0)
    assert event["params"] == {"value": pytest.approx(-8.0)}
    assert event["transport_kind"] == "mutation"
    assert event["context"]["scenario"] == "auto_soundcheck_feedback_fader_reduce_dry_run"
    assert event["source_file"] is not None
    assert event["source_file"].endswith("backend/auto_soundcheck_engine.py")
    assert event["source_function"] == "_handle_feedback_event"
    assert info.fader_db == pytest.approx(-8.0)


@pytest.mark.asyncio
async def test_bypass_mixer_inventory_records_bulk_channel_mutations(
    monkeypatch: pytest.MonkeyPatch,
):
    server = AutoMixerServer.__new__(AutoMixerServer)
    messages: list[dict[str, Any]] = []

    async def _capture(_websocket, message):
        messages.append(message)

    server.send_to_client = _capture
    async def _fast_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(server_module.asyncio, "sleep", _fast_sleep)

    class _WingLikeMixer(FakeMixer):
        pass

    wing_like = _WingLikeMixer(scenario="bypass_mixer_inventory", channel_names={1: "Lead Vox"})
    server.mixer_client = wing_like
    monkeypatch.setattr(server_module, "WingClient", _WingLikeMixer)
    monkeypatch.setattr(server_module, "EnhancedOSCClient", type("EnhancedOSCClientStub", (), {}))

    payload = await server.bypass_mixer("ws")

    assert payload["success"] is True
    assert messages[-1]["type"] == "bypass_result"
    assert len(wing_like.event_log) == 360
    assert [event["method"] for event in wing_like.event_log[:9]] == [
        "set_channel_fader",
        "set_eq_on",
        "send",
        "set_compressor_on",
        "set_gate_on",
        "set_low_cut",
        "set_high_cut",
        "send",
        "send",
    ]
    assert [event["channel"] for event in wing_like.event_log[:9]] == [1] * 9
    assert wing_like.event_log[0]["source_file"] is not None
    assert wing_like.event_log[0]["source_file"].endswith("backend/server.py")
    assert wing_like.event_log[0]["source_function"] == "bypass_task"


def test_auto_soundcheck_apply_fader_records_fader_write_from_preset_and_gain_correction():
    mixer = FakeMixer(
        scenario="auto_soundcheck_apply_fader",
        channel_names={1: "Lead Vox"},
    )
    engine = AutoSoundcheckEngine(num_channels=1, auto_apply=False, auto_discover=False)
    engine.mixer_client = mixer

    info = ChannelInfo(
        channel=1,
        name="Lead Vox",
        preset="leadVocal",
        has_signal=True,
        gain_correction_db=-2.0,
        lufs=-20.0,
    )
    engine.channels = {1: info}

    engine._apply_fader(1, info, "leadVocal")

    assert [event["method"] for event in mixer.event_log] == ["set_fader"]
    assert mixer.event_log == mixer.mutation_events()

    event = mixer.event_log[0]
    assert event["channel"] == 1
    assert event["value"] == pytest.approx(-5.0)
    assert event["params"] == {"value": pytest.approx(-5.0)}
    assert event["transport_kind"] == "mutation"
    assert event["context"]["scenario"] == "auto_soundcheck_apply_fader"
    assert event["source_file"] is not None
    assert event["source_file"].endswith("backend/auto_soundcheck_engine.py")
    assert event["source_function"] == "_apply_fader"
    assert info.fader_db == pytest.approx(-5.0)


def test_auto_soundcheck_apply_input_gain_records_gain_write():
    mixer = FakeMixer(
        scenario="auto_soundcheck_apply_input_gain",
        channel_names={1: "Lead Vox"},
    )
    engine = AutoSoundcheckEngine(num_channels=1, auto_apply=False, auto_discover=False)
    engine.mixer_client = mixer

    info = ChannelInfo(
        channel=1,
        name="Lead Vox",
        peak_db=-18.0,
        rms_db=-30.0,
    )
    engine.channels = {1: info}

    engine._apply_input_gain(1, info, "leadVocal")

    assert [event["method"] for event in mixer.event_log] == ["set_gain"]

    event = mixer.event_log[0]
    assert event["channel"] == 1
    assert event["value"] == pytest.approx(10.0)
    assert event["params"] == {"value": pytest.approx(10.0)}
    assert event["transport_kind"] == "mutation"
    assert event["context"]["scenario"] == "auto_soundcheck_apply_input_gain"
    assert event["source_file"] is not None
    assert event["source_file"].endswith("backend/auto_soundcheck_engine.py")
    assert event["source_function"] == "_apply_input_gain"


def test_auto_soundcheck_apply_pan_records_pan_write():
    mixer = FakeMixer(
        scenario="auto_soundcheck_apply_pan",
        channel_names={1: "Guitar"},
    )
    engine = AutoSoundcheckEngine(num_channels=1, auto_apply=False, auto_discover=False)
    engine.mixer_client = mixer

    info = ChannelInfo(
        channel=1,
        name="Guitar",
        preset="electricGuitar",
        has_signal=True,
    )
    engine.channels = {1: info}

    engine._apply_pan(1, info, "electricGuitar")

    assert [event["method"] for event in mixer.event_log] == ["set_pan"]

    event = mixer.event_log[0]
    assert event["channel"] == 1
    assert event["value"] == pytest.approx(-25.0)
    assert event["params"] == {"value": pytest.approx(-25.0)}
    assert event["transport_kind"] == "mutation"
    assert event["context"]["scenario"] == "auto_soundcheck_apply_pan"
    assert event["source_file"] is not None
    assert event["source_file"].endswith("backend/auto_soundcheck_engine.py")
    assert event["source_function"] == "_apply_pan"


def test_auto_soundcheck_apply_eq_records_eq_band_writes():
    mixer = FakeMixer(
        scenario="auto_soundcheck_apply_eq",
        channel_names={1: "Lead Vox"},
    )
    engine = AutoSoundcheckEngine(num_channels=1, auto_apply=False, auto_discover=False)
    engine.mixer_client = mixer

    info = ChannelInfo(
        channel=1,
        name="Lead Vox",
    )
    engine.channels = {1: info}

    engine._apply_eq(1, "leadVocal")

    expected_bands = INSTRUMENT_EQ_PRESETS["leadVocal"][:4]

    assert len(mixer.event_log) == 4
    assert [event["method"] for event in mixer.event_log] == [
        "set_eq_band",
        "set_eq_band",
        "set_eq_band",
        "set_eq_band",
    ]
    assert [event["channel"] for event in mixer.event_log] == [1, 1, 1, 1]

    for event, (band_idx, (freq, gain, q)) in zip(
        mixer.event_log,
        enumerate(expected_bands, start=1),
    ):
        assert event["value"] == pytest.approx(gain)
        assert event["params"] == {
            "band": band_idx,
            "freq": pytest.approx(freq),
            "gain": pytest.approx(gain),
            "q": pytest.approx(q),
        }
        assert event["transport_kind"] == "mutation"
        assert event["context"]["scenario"] == "auto_soundcheck_apply_eq"
        assert event["source_file"] is not None
        assert event["source_file"].endswith("backend/auto_soundcheck_engine.py")
        assert event["source_function"] == "_apply_eq"

    assert info.eq_applied is True


def test_auto_soundcheck_apply_compressor_records_compressor_write():
    mixer = FakeMixer(
        scenario="auto_soundcheck_apply_compressor",
        channel_names={1: "Lead Vox"},
    )
    engine = AutoSoundcheckEngine(num_channels=1, auto_apply=False, auto_discover=False)
    engine.mixer_client = mixer

    info = ChannelInfo(
        channel=1,
        name="Lead Vox",
    )
    engine.channels = {1: info}

    engine._apply_compressor(1, info, "leadVocal")

    expected_threshold, expected_ratio, expected_attack, expected_release = INSTRUMENT_COMPRESSOR["leadVocal"]

    assert [event["method"] for event in mixer.event_log] == ["set_compressor"]

    event = mixer.event_log[0]
    assert event["channel"] == 1
    assert event["value"] is None
    assert event["params"] == {
        "threshold_db": pytest.approx(expected_threshold),
        "ratio": pytest.approx(expected_ratio),
        "attack_ms": pytest.approx(expected_attack),
        "release_ms": pytest.approx(expected_release),
        "makeup_db": pytest.approx(0.0),
        "enabled": True,
    }
    assert event["transport_kind"] == "mutation"
    assert event["context"]["scenario"] == "auto_soundcheck_apply_compressor"
    assert event["source_file"] is not None
    assert event["source_file"].endswith("backend/auto_soundcheck_engine.py")
    assert event["source_function"] == "_apply_compressor"
    assert info.compressor_applied is True


def test_auto_soundcheck_apply_fx_sends_records_send_level_writes():
    mixer = FakeMixer(
        scenario="auto_soundcheck_apply_fx_sends",
        channel_names={3: "Lead Vox"},
    )
    engine = AutoSoundcheckEngine(num_channels=3, auto_apply=False, auto_discover=False)
    engine.mixer_client = mixer

    info = ChannelInfo(
        channel=3,
        name="Lead Vox",
        preset="leadVocal",
        has_signal=True,
    )
    engine.channels = {3: info}

    engine._apply_fx_sends(3, info, "leadVocal")

    expected_sends = auto_soundcheck_engine_module.INSTRUMENT_FX_SENDS["leadVocal"]

    assert [event["method"] for event in mixer.event_log] == [
        "set_send_level",
        "set_send_level",
    ]
    assert [event["channel"] for event in mixer.event_log] == [3, 3]

    for event, (expected_bus, expected_level) in zip(mixer.event_log, expected_sends):
        assert event["value"] == pytest.approx(expected_level)
        assert event["params"] == {
            "send_bus": expected_bus,
            "value": pytest.approx(expected_level),
        }
        assert event["transport_kind"] == "mutation"
        assert event["context"]["scenario"] == "auto_soundcheck_apply_fx_sends"
        assert event["source_file"] is not None
        assert event["source_file"].endswith("backend/auto_soundcheck_engine.py")
        assert event["source_function"] == "_apply_fx_sends"
