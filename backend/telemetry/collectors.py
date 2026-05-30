"""Read-only telemetry collectors for channel, bus and action snapshots."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional

try:
    from .schemas import BusTelemetry, ChannelTelemetry, EngineerAction, timestamp_ms
    from .session_manager import TelemetrySessionManager
except ImportError:  # pragma: no cover - direct PYTHONPATH=backend compatibility
    from telemetry.schemas import BusTelemetry, ChannelTelemetry, EngineerAction, timestamp_ms
    from telemetry.session_manager import TelemetrySessionManager


def _numeric(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return bool(value)


class BaseTelemetryCollector(ABC):
    @abstractmethod
    def collect_once(self, manager: TelemetrySessionManager) -> Dict[str, int]:
        """Collect one read-only telemetry tick into the active session."""


class MockTelemetryCollector(BaseTelemetryCollector):
    """Deterministic collector for tests and offline development."""

    def __init__(self, *, channel_count: int = 2):
        self.channel_count = max(1, int(channel_count))

    def collect_once(self, manager: TelemetrySessionManager) -> Dict[str, int]:
        session = manager.get_current_session()
        if session is None:
            raise RuntimeError("No active telemetry session")
        now = timestamp_ms()
        counts = {"channel_telemetry": 0, "bus_telemetry": 0, "engineer_actions": 0}
        for channel_id in range(1, self.channel_count + 1):
            event = ChannelTelemetry(
                timestamp_ms=now,
                session_id=session.session_id,
                channel_id=channel_id,
                channel_name=f"Ch {channel_id}",
                detected_instrument="mock_source",
                fader_db=-10.0 - channel_id,
                trim_db=0.0,
                mute=False,
                solo=False,
                pan=0.0,
                meter_rms_dbfs=-24.0 + channel_id,
                meter_peak_dbfs=-12.0 + channel_id,
                gate_gain_reduction_db=0.0,
                comp_gain_reduction_db=0.0,
                eq_summary={"source": "mock", "bands": []},
                plugin_or_processing_snapshot={"source": "mock"},
            )
            manager.append_event("channel_telemetry", event.to_dict())
            counts["channel_telemetry"] += 1
        manager.append_event(
            "bus_telemetry",
            BusTelemetry(
                timestamp_ms=now,
                session_id=session.session_id,
                bus_name="Main LR",
                rms_dbfs=-18.0,
                peak_dbfs=-6.0,
                lufs_short=-16.0,
                limiter_gain_reduction_db=0.0,
            ).to_dict(),
        )
        counts["bus_telemetry"] += 1
        manager.append_event(
            "engineer_actions",
            EngineerAction(
                timestamp_ms=now,
                session_id=session.session_id,
                actor="test",
                target_type="scene",
                target_id="mock",
                action_type="note",
                reason="mock collector tick",
                confidence=1.0,
                safety_status="logged_only",
            ).to_dict(),
        )
        counts["engineer_actions"] += 1
        return counts


class WingTelemetryCollector(BaseTelemetryCollector):
    """Read-only collector over cached WingClient/audio_capture state."""

    def __init__(self, mixer_client: Any, audio_capture: Any = None, *, channel_count: int = 48):
        self.mixer_client = mixer_client
        self.audio_capture = audio_capture
        self.channel_count = max(1, int(channel_count))

    def collect_once(self, manager: TelemetrySessionManager) -> Dict[str, int]:
        session = manager.get_current_session()
        if session is None:
            raise RuntimeError("No active telemetry session")
        state = getattr(self.mixer_client, "state", {}) or {}
        if not isinstance(state, dict):
            state = {}
        now = timestamp_ms()
        count = 0
        for channel_id in range(1, self.channel_count + 1):
            event = self._channel_event(session.session_id, now, channel_id, state)
            manager.append_event("channel_telemetry", event.to_dict())
            count += 1
        manager.append_event(
            "bus_telemetry",
            BusTelemetry(
                timestamp_ms=now,
                session_id=session.session_id,
                bus_name="Main LR",
                rms_dbfs=self._capture_metric("get_rms", 1),
                peak_dbfs=self._capture_metric("get_peak", 1),
                lufs_short=self._capture_metric("get_lufs", 1),
                limiter_gain_reduction_db=None,
            ).to_dict(),
        )
        return {"channel_telemetry": count, "bus_telemetry": 1}

    def _state_value(self, state: Dict[str, Any], *addresses: str) -> Any:
        for address in addresses:
            if address in state:
                return state[address]
        return None

    def _capture_metric(self, method_name: str, channel_id: int) -> Optional[float]:
        capture = self.audio_capture
        if not capture or not hasattr(capture, method_name):
            return None
        try:
            return _numeric(getattr(capture, method_name)(channel_id))
        except Exception:
            return None

    def _channel_event(self, session_id: str, now: int, channel_id: int, state: Dict[str, Any]) -> ChannelTelemetry:
        name = self._state_value(state, f"/ch/{channel_id}/$name", f"/ch/{channel_id}/name") or f"Ch {channel_id}"
        return ChannelTelemetry(
            timestamp_ms=now,
            session_id=session_id,
            channel_id=channel_id,
            channel_name=str(name),
            detected_instrument=str(name),
            fader_db=_numeric(self._state_value(state, f"/ch/{channel_id}/fdr")),
            trim_db=_numeric(self._state_value(state, f"/ch/{channel_id}/in/set/trim")),
            mute=_integer_bool(self._state_value(state, f"/ch/{channel_id}/mute")),
            solo=_integer_bool(self._state_value(state, f"/ch/{channel_id}/solo")),
            pan=_numeric(self._state_value(state, f"/ch/{channel_id}/pan")),
            meter_rms_dbfs=self._capture_metric("get_rms", channel_id),
            meter_peak_dbfs=self._capture_metric("get_peak", channel_id),
            gate_gain_reduction_db=_numeric(self._state_value(state, f"/ch/{channel_id}/gate/gr")),
            comp_gain_reduction_db=_numeric(self._state_value(state, f"/ch/{channel_id}/dyn/gr")),
            eq_summary=self._eq_summary(channel_id, state),
            plugin_or_processing_snapshot=self._processing_snapshot(channel_id, state),
        )

    def _eq_summary(self, channel_id: int, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "eq_on": self._state_value(state, f"/ch/{channel_id}/eq/on"),
            "hpf": self._state_value(state, f"/ch/{channel_id}/eq/hpf"),
            "source": "cached_wing_state",
        }

    def _processing_snapshot(self, channel_id: int, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "gate_on": self._state_value(state, f"/ch/{channel_id}/gate/on"),
            "gate_threshold": self._state_value(state, f"/ch/{channel_id}/gate/thr"),
            "compressor_on": self._state_value(state, f"/ch/{channel_id}/dyn/on"),
            "compressor_threshold": self._state_value(state, f"/ch/{channel_id}/dyn/thr"),
            "source": "cached_wing_state",
        }


def duplicate_decision_trace_as_engineer_action(
    manager: TelemetrySessionManager,
    trace: Dict[str, Any],
) -> Dict[str, Any]:
    """Copy an automixer decision trace into engineer_actions.jsonl."""

    session = manager.get_current_session()
    if session is None:
        raise RuntimeError("No active telemetry session")
    action = EngineerAction(
        timestamp_ms=timestamp_ms(),
        session_id=session.session_id,
        actor="automixer",
        target_type="channel" if trace.get("target") is not None else "scene",
        target_id=str(trace.get("target") or trace.get("decision_id") or "unknown"),
        action_type="note",
        before_value=trace.get("current_value"),
        after_value=trace.get("proposed_value"),
        reason=str(trace.get("blocked_reason") or trace.get("source") or "decision_trace"),
        confidence=trace.get("confidence"),
        safety_status=str(trace.get("proof_status") or trace.get("status") or "logged_only"),
    )
    return manager.append_event("engineer_actions", action.to_dict())
