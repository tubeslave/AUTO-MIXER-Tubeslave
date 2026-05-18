"""Product-facing read-only state for the operator cockpit UI.

These builders intentionally do not connect to mixers, scan devices, or apply
changes. They translate the backend runtime into stable UI contracts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

from dante_routing_config import get_module_signal_info, get_routing_as_dict
try:
    from gain_fader_runtime import build_gain_fader_runtime_summary
except ImportError:  # pragma: no cover - compatibility for older integration branches
    def build_gain_fader_runtime_summary(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        _ = (args, kwargs)
        return {"available": False, "reason": "gain_fader_runtime_unavailable"}


DEFAULT_CHANNEL_COUNT = 24
DEFAULT_DANTE_CHANNEL_COUNT = 64
MIN_ACTIVE_DB = -90.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_call(fn: Callable[..., Any], *args: Any, default: Any = None, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return {"error": str(exc)} if default is None else default


def _round(value: Any, digits: int = 1) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return round(numeric, digits)


def _is_connected(client: Any) -> bool:
    return bool(client and getattr(client, "is_connected", False))


def _connection_summary(server: Any) -> Dict[str, Any]:
    client = getattr(server, "mixer_client", None)
    connected = _is_connected(client)
    state = _safe_call(client.get_state, default={}) if connected and hasattr(client, "get_state") else {}
    return {
        "connected": connected,
        "connection_mode": getattr(server, "connection_mode", None),
        "client_type": type(client).__name__ if client else None,
        "ip": getattr(client, "ip", None) or getattr(client, "host", None),
        "port": getattr(client, "port", None)
        or getattr(client, "send_port", None)
        or getattr(client, "osc_port", None),
        "state": state if isinstance(state, dict) else {},
    }


def _audio_capture_status(server: Any) -> Dict[str, Any]:
    capture = getattr(server, "audio_capture", None)
    if not capture:
        return {"available": False, "running": False}
    status = _safe_call(capture.get_status, default={})
    if not isinstance(status, dict):
        status = {}
    return {"available": True, **status}


def _routing_for_channel(routing: Iterable[Dict[str, Any]], channel: int) -> Dict[str, Any]:
    for item in routing:
        if int(item.get("start", 0)) <= channel <= int(item.get("end", 0)):
            return item
    return {}


def _cached_channel_name(client: Any, channel: int) -> Optional[str]:
    state = getattr(client, "state", None)
    if not isinstance(state, dict):
        return None
    for address in (f"/ch/{channel}/$name", f"/ch/{channel}/name"):
        name = state.get(address)
        if isinstance(name, str) and name.strip():
            return name.strip()
    return None


def _cached_numeric(client: Any, address: str) -> Optional[float]:
    state = getattr(client, "state", None)
    if not isinstance(state, dict):
        return None
    try:
        value = state.get(address)
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _cached_int(client: Any, address: str) -> Optional[int]:
    state = getattr(client, "state", None)
    if not isinstance(state, dict):
        return None
    try:
        value = state.get(address)
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _infer_role_and_group(name: str, routing_role: str, channel: int) -> Dict[str, str]:
    lower = name.lower()
    patterns = [
        (("kick", "bd", "бочка"), "Kick", "DRUMS"),
        (("snare", "sn", "малый"), "Snare", "DRUMS"),
        (("hat", "hh", "хэт"), "Hi-Hat", "DRUMS"),
        (("tom",), "Tom", "DRUMS"),
        (("oh", "overhead"), "Overhead", "DRUMS"),
        (("bass", "бас"), "Bass", "BASS"),
        (("vox", "vocal", "lead", "back", "вокал"), "Vocal", "VOCALS"),
        (("gtr", "guitar", "гит"), "Guitar", "GUITARS"),
        (("key", "piano", "synth", "keys"), "Keys", "KEYS"),
        (("master", "main"), "Master", "MASTER"),
    ]
    for keys, role, group in patterns:
        if any(key in lower for key in keys):
            return {"role": role, "group": group}

    role_map = {
        "master": ("Master", "MASTER"),
        "drum_bus": ("Drum Bus", "DRUMS"),
        "vocal_bus": ("Vocal Bus", "VOCALS"),
        "instrument_bus": ("Instrument Bus", "INSTRUMENTS"),
        "measurement_mic": ("Measurement Mic", "MEASUREMENT"),
        "ambient_mic": ("Ambient Mic", "AMBIENT"),
        "matrix": ("Matrix", "MATRIX"),
        "reserve": ("Reserve", "RESERVE"),
    }
    role, group = role_map.get(routing_role, ("Channel", "CHANNELS"))
    if routing_role == "channel_analysis" and 1 <= channel <= DEFAULT_CHANNEL_COUNT:
        role = f"Input {channel:02d}"
    return {"role": role, "group": group}


def _meter_for_channel(capture: Any, channel: int) -> Dict[str, Any]:
    if not capture or not getattr(capture, "running", False):
        return {"available": False}
    num_channels = int(getattr(capture, "num_channels", 0) or 0)
    if channel < 1 or channel > num_channels:
        return {"available": False}

    rms = _round(_safe_call(capture.get_rms, channel, default=None))
    peak = _round(_safe_call(capture.get_peak, channel, default=None))
    lufs = _round(_safe_call(capture.get_lufs, channel, default=None))
    dynamic = None
    if rms is not None and peak is not None:
        dynamic = round(max(0.0, peak - rms), 1)
    return {
        "available": True,
        "rms_db": rms,
        "peak_db": peak,
        "lufs": lufs,
        "dynamic_range_db": dynamic,
    }


def build_channel_inventory(server: Any, total_channels: int = DEFAULT_CHANNEL_COUNT) -> Dict[str, Any]:
    total_channels = max(1, min(int(total_channels or DEFAULT_CHANNEL_COUNT), DEFAULT_DANTE_CHANNEL_COUNT))
    client = getattr(server, "mixer_client", None)
    capture = getattr(server, "audio_capture", None)
    routing = get_routing_as_dict(DEFAULT_DANTE_CHANNEL_COUNT)
    channels: List[Dict[str, Any]] = []
    active_count = 0
    named_count = 0

    for channel in range(1, total_channels + 1):
        routing_item = _routing_for_channel(routing, channel)
        cached_name = _cached_channel_name(client, channel)
        name = cached_name or f"Ch {channel}"
        if cached_name:
            named_count += 1
        routing_role = str(routing_item.get("role") or "unknown")
        inferred = _infer_role_and_group(name, routing_role, channel)
        meter = _meter_for_channel(capture, channel)
        rms = meter.get("rms_db")
        active = bool(meter.get("available") and rms is not None and rms > MIN_ACTIVE_DB)
        if active:
            active_count += 1
        status = "active" if active else ("idle" if meter.get("available") else "unknown")

        channels.append(
            {
                "id": channel,
                "number": channel,
                "name": name,
                "source": f"In {channel:02d}",
                "dante_channel": channel,
                "routing_role": routing_role,
                "tap_point": routing_item.get("tap_point"),
                "routing_label": routing_item.get("label_short"),
                "role": inferred["role"],
                "group": inferred["group"],
                "status": status,
                "meter": meter,
                "level_db": rms,
                "peak_db": meter.get("peak_db"),
                "lufs": meter.get("lufs"),
                "dynamic_range_db": meter.get("dynamic_range_db"),
                "control_state": {
                    "fader_db": _cached_numeric(client, f"/ch/{channel}/fdr"),
                    "gain_db": _cached_numeric(client, f"/ch/{channel}/in/set/trim"),
                    "mute": _cached_int(client, f"/ch/{channel}/mute"),
                },
                "ai_recognition": {
                    "label": inferred["role"],
                    "confidence": 99 if cached_name else None,
                    "source": "cached_mixer_name" if cached_name else "routing_default",
                },
            }
        )

    return {
        "type": "channel_inventory",
        "generated_at": _utc_now(),
        "source": "runtime_cached_state",
        "available": True,
        "connection": _connection_summary(server),
        "audio_capture": _audio_capture_status(server),
        "summary": {
            "total_channels": total_channels,
            "active_channels": active_count,
            "idle_channels": total_channels - active_count if capture and getattr(capture, "running", False) else None,
            "muted_channels": None,
            "inactive_channels": total_channels - active_count if capture and getattr(capture, "running", False) else None,
            "named_channels": named_count,
            "coverage_percent": round((named_count / total_channels) * 100, 1) if total_channels else 0.0,
        },
        "channels": channels,
        "routing_scheme": routing,
    }


def _normalize_action(action: Any, index: int) -> Dict[str, Any]:
    if isinstance(action, dict):
        title = (
            action.get("title")
            or action.get("description")
            or action.get("message")
            or action.get("action")
            or f"Action {index + 1}"
        )
        return {
            "id": action.get("id") or action.get("action_id") or f"pending-{index}",
            "title": str(title),
            "target": action.get("target") or action.get("channel") or action.get("bus"),
            "kind": action.get("kind") or action.get("type") or action.get("parameter"),
            "status": action.get("status") or "pending",
            "severity": action.get("severity") or action.get("priority") or "medium",
            "confidence": action.get("confidence"),
            "created_at": action.get("created_at") or action.get("timestamp"),
            "raw": action,
        }
    return {
        "id": f"pending-{index}",
        "title": str(action),
        "target": None,
        "kind": None,
        "status": "pending",
        "severity": "medium",
        "confidence": None,
        "created_at": None,
        "raw": action,
    }


def build_decision_queue(server: Any, limit: int = 50) -> Dict[str, Any]:
    limit = max(1, min(int(limit or 50), 200))
    operator_mode = server.get_operator_mode_status()
    mixing_agent = getattr(server, "mixing_agent", None)
    proposal_queue = getattr(server, "operator_proposal_queue", None)

    queue_pending: List[Dict[str, Any]] = []
    queue_history: List[Dict[str, Any]] = []
    queue_summary: Dict[str, Any] = {}
    if proposal_queue:
        queue_pending = proposal_queue.pending_actions()
        queue_history = proposal_queue.history(limit)
        queue_summary = proposal_queue.summary()

    if not mixing_agent and not proposal_queue:
        return {
            "type": "decision_queue",
            "generated_at": _utc_now(),
            "status": "unavailable",
            "available": False,
            "reason": "mixing_agent_runtime_not_initialized",
            "operator_mode": operator_mode,
            "summary": {
                "pending_count": 0,
                "history_count": 0,
                "applied_count": 0,
                "queue_accepts_proposals": bool(operator_mode["capabilities"]["can_create_proposals"]),
                "can_apply": bool(operator_mode["capabilities"]["can_apply_to_console"]),
            },
            "pending_actions": [],
            "history": [],
            "agent_status": None,
        }

    pending_raw = _safe_call(mixing_agent.get_pending_actions, default=[]) if mixing_agent else []
    history_raw = _safe_call(mixing_agent.get_action_history, limit, default=[]) if mixing_agent else []
    agent_status = (
        _safe_call(mixing_agent.get_status, default={})
        if mixing_agent and hasattr(mixing_agent, "get_status")
        else {}
    )
    agent_pending = [_normalize_action(action, index) for index, action in enumerate(pending_raw or [])]
    agent_history = [_normalize_action(action, index) for index, action in enumerate(history_raw or [])]
    pending = queue_pending + agent_pending
    history = (queue_history + agent_history)[-limit:]
    applied_count = int(queue_summary.get("applied_count", 0)) + sum(
        1 for item in agent_history if item.get("status") in {"applied", "approved", "done"}
    )
    source = "operator_proposal_queue"
    if mixing_agent:
        source = "operator_proposal_queue+mixing_agent" if proposal_queue else "mixing_agent"

    return {
        "type": "decision_queue",
        "generated_at": _utc_now(),
        "status": "ok",
        "available": True,
        "source": source,
        "agent_runtime_available": bool(mixing_agent),
        "proposal_queue_available": bool(proposal_queue),
        "reason": None if mixing_agent else "mixing_agent_runtime_not_initialized",
        "operator_mode": operator_mode,
        "summary": {
            "pending_count": len(pending),
            "history_count": len(history),
            "applied_count": applied_count,
            "accepted_count": int(queue_summary.get("accepted_count", 0)),
            "dismissed_count": int(queue_summary.get("dismissed_count", 0)),
            "blocked_count": int(queue_summary.get("blocked_count", 0)),
            "queue_accepts_proposals": bool(operator_mode["capabilities"]["can_create_proposals"]),
            "can_apply": bool(operator_mode["capabilities"]["can_apply_to_console"]),
        },
        "pending_actions": pending,
        "history": history,
        "agent_status": agent_status if isinstance(agent_status, dict) else {},
    }


def _master_bus_metrics(server: Any) -> Dict[str, Any]:
    capture = getattr(server, "audio_capture", None)
    if not capture or not getattr(capture, "running", False):
        return {
            "available": False,
            "integrated_lufs": None,
            "short_term_lufs": None,
            "momentary_lufs": None,
            "true_peak_db": None,
            "lra_lu": None,
        }
    master_channels = [
        channel
        for channel in (49, 50)
        if channel <= int(getattr(capture, "num_channels", 0) or 0)
    ]
    if not master_channels:
        master_channels = [1]
    lufs_values = [_meter_for_channel(capture, ch).get("lufs") for ch in master_channels]
    peak_values = [_meter_for_channel(capture, ch).get("peak_db") for ch in master_channels]
    lufs_values = [value for value in lufs_values if value is not None]
    peak_values = [value for value in peak_values if value is not None]
    short_lufs = round(sum(lufs_values) / len(lufs_values), 1) if lufs_values else None
    return {
        "available": bool(lufs_values or peak_values),
        "integrated_lufs": short_lufs,
        "short_term_lufs": short_lufs,
        "momentary_lufs": short_lufs,
        "true_peak_db": max(peak_values) if peak_values else None,
        "lra_lu": None,
    }


def build_dashboard_snapshot(server: Any) -> Dict[str, Any]:
    operator_mode = server.get_operator_mode_status()
    channel_inventory = build_channel_inventory(server)
    decision_queue = build_decision_queue(server, limit=12)
    ready_for_live = _safe_call(server.get_ready_for_live_status, default={})
    pilot = _safe_call(server.get_supervised_pilot_status, default={})
    gain_fader_runtime = build_gain_fader_runtime_summary(
        connection_mode=getattr(server, "connection_mode", None),
        wing_boundary_active=_safe_call(server._is_wing_deployment_write_boundary_active, default=False),
    )
    voice_runtime = (
        _safe_call(server.get_voice_runtime_summary, default={})
        if hasattr(server, "get_voice_runtime_summary")
        else {}
    )

    return {
        "type": "dashboard_snapshot",
        "generated_at": _utc_now(),
        "operator_mode": operator_mode,
        "connection": _connection_summary(server),
        "audio_capture": _audio_capture_status(server),
        "master_bus": _master_bus_metrics(server),
        "channel_summary": channel_inventory["summary"],
        "decision_summary": decision_queue["summary"],
        "safety": {
            "ready_for_live": ready_for_live,
            "pilot_primary_state": pilot.get("primary_state"),
            "pilot_ready": pilot.get("pilot_ready"),
            "live_write_policy": operator_mode.get("live_write_policy"),
            "blocked_reasons": operator_mode.get("blocked_reasons", []),
            "manual_write_supervision": (
                _safe_call(server.get_manual_write_supervision_status, default={})
                if hasattr(server, "get_manual_write_supervision_status")
                else {}
            ),
        },
        "runtime": {
            "gain_fader": gain_fader_runtime,
            "voice": voice_runtime,
            "agent_available": bool(decision_queue.get("agent_runtime_available")),
            "proposal_queue_available": bool(decision_queue.get("proposal_queue_available")),
        },
    }


def build_connection_topology(server: Any) -> Dict[str, Any]:
    connection = _connection_summary(server)
    audio_capture = _audio_capture_status(server)
    operator_mode = server.get_operator_mode_status()
    decision_queue = build_decision_queue(server, limit=5)
    routing_scheme = get_routing_as_dict(DEFAULT_DANTE_CHANNEL_COUNT)
    module_signal_info = get_module_signal_info()

    nodes = [
        {
            "id": "director_control",
            "label": "Director Control",
            "kind": "client",
            "status": "online" if getattr(server, "connected_clients", None) else "offline",
            "detail": f"{len(getattr(server, 'connected_clients', []) or [])} websocket client(s)",
        },
        {
            "id": "automixer_core",
            "label": "Automixer Core",
            "kind": "core",
            "status": "running",
            "detail": f"mode={operator_mode['mode']}",
        },
        {
            "id": "mixer",
            "label": "Mixer",
            "kind": "mixer",
            "status": "online" if connection["connected"] else "offline",
            "detail": connection.get("connection_mode") or "not connected",
        },
        {
            "id": "audio_capture",
            "label": "Audio Capture",
            "kind": "audio",
            "status": "running" if audio_capture.get("running") else "offline",
            "detail": f"{audio_capture.get('num_channels', 0) or 0} channel(s)",
        },
        {
            "id": "decision_queue",
            "label": "Decision Queue",
            "kind": "agent",
            "status": "online" if decision_queue.get("available") else "unavailable",
            "detail": decision_queue.get("reason") or f"{decision_queue['summary']['pending_count']} pending",
        },
    ]
    links = [
        {"from": "director_control", "to": "automixer_core", "kind": "websocket", "status": "online"},
        {
            "from": "automixer_core",
            "to": "mixer",
            "kind": "osc_control",
            "status": "online" if connection["connected"] else "offline",
        },
        {
            "from": "audio_capture",
            "to": "automixer_core",
            "kind": "audio_analysis",
            "status": "online" if audio_capture.get("running") else "offline",
        },
        {
            "from": "decision_queue",
            "to": "automixer_core",
            "kind": "proposal_flow",
            "status": "online" if decision_queue.get("available") else "unavailable",
        },
    ]

    connected_devices: List[Dict[str, Any]] = []
    if connection["connected"]:
        connected_devices.append(
            {
                "name": connection.get("client_type") or "Mixer",
                "ip": connection.get("ip"),
                "port": connection.get("port"),
                "status": "online",
                "kind": connection.get("connection_mode") or "mixer",
            }
        )
    if audio_capture.get("running"):
        connected_devices.append(
            {
                "name": "Audio Capture",
                "channels": audio_capture.get("num_channels"),
                "sample_rate": audio_capture.get("sample_rate"),
                "status": "running",
                "kind": "audio",
            }
        )

    return {
        "type": "connection_topology",
        "generated_at": _utc_now(),
        "connection": connection,
        "audio_capture": audio_capture,
        "operator_mode": operator_mode,
        "nodes": nodes,
        "links": links,
        "connected_devices": connected_devices,
        "routing_scheme": routing_scheme,
        "module_signal_info": module_signal_info,
        "safety": {
            "manual_write_supervision": (
                _safe_call(server.get_manual_write_supervision_status, default={})
                if hasattr(server, "get_manual_write_supervision_status")
                else {}
            ),
            "ready_for_live": (
                _safe_call(server.get_ready_for_live_status, default={})
                if hasattr(server, "get_ready_for_live_status")
                else {}
            ),
        },
    }
