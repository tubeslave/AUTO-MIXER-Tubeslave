"""Product-facing read-only state for the operator cockpit UI.

These builders intentionally do not connect to mixers, scan devices, or apply
changes. They translate the backend runtime into stable UI contracts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from dante_routing_config import get_module_signal_info, get_routing_as_dict
try:
    from gain_fader_runtime import build_gain_fader_runtime_summary
except ImportError:  # pragma: no cover - compatibility for older integration branches
    def build_gain_fader_runtime_summary(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        _ = (args, kwargs)
        return {"available": False, "reason": "gain_fader_runtime_unavailable"}
try:
    from processing_registry import (
        build_pipeline_runtime_summary,
        build_processing_registry_snapshot,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.processing_registry import (
        build_pipeline_runtime_summary,
        build_processing_registry_snapshot,
    )
try:
    from agentic_method_foundation import (
        build_agentic_recommendations,
        build_inter_channel_context,
        feature_snapshot_from_channel_inventory,
        safety_review_recommendations,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.agentic_method_foundation import (
        build_agentic_recommendations,
        build_inter_channel_context,
        feature_snapshot_from_channel_inventory,
        safety_review_recommendations,
    )


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
        (("playback", "pb l", "pb r"), "Playback", "PLAYBACK"),
        (("gtr", "guitar", "гит"), "Guitar", "GUITARS"),
        (("vox", "vocal", "lead", "back", "вокал"), "Vocal", "VOCALS"),
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
    pilot = (
        _safe_call(server.get_supervised_pilot_status, default={})
        if hasattr(server, "get_supervised_pilot_status")
        else {}
    )
    gain_fader_runtime = build_gain_fader_runtime_summary(
        connection_mode=getattr(server, "connection_mode", None),
        wing_boundary_active=_safe_call(server._is_wing_deployment_write_boundary_active, default=False),
    )
    voice_runtime = (
        _safe_call(server.get_voice_runtime_summary, default={})
        if hasattr(server, "get_voice_runtime_summary")
        else {}
    )
    manual_write_supervision = (
        _safe_call(server.get_manual_write_supervision_status, default={})
        if hasattr(server, "get_manual_write_supervision_status")
        else {}
    )
    processing_runtime = build_pipeline_runtime_summary(operator_mode, manual_write_supervision)

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
            "manual_write_supervision": manual_write_supervision,
        },
        "runtime": {
            "gain_fader": gain_fader_runtime,
            "voice": voice_runtime,
            "processing": processing_runtime,
            "agent_available": bool(decision_queue.get("agent_runtime_available")),
            "proposal_queue_available": bool(decision_queue.get("proposal_queue_available")),
        },
    }


def build_processing_pipeline_snapshot(server: Any, include_methods: bool = True) -> Dict[str, Any]:
    """Return the selected processing methods and pipeline profiles.

    This is a read-only Product Layer contract. It exposes which methods are
    canonical, proposal-only, offline-only, lab or quarantined without importing
    DSP modules or widening the live write boundary.
    """

    operator_mode = server.get_operator_mode_status()
    manual_write_supervision = (
        _safe_call(server.get_manual_write_supervision_status, default={})
        if hasattr(server, "get_manual_write_supervision_status")
        else {}
    )
    snapshot = build_processing_registry_snapshot(include_methods=include_methods)
    snapshot["runtime"] = build_pipeline_runtime_summary(operator_mode, manual_write_supervision)
    snapshot["operator_mode"] = operator_mode
    return snapshot


def build_decision_traces(server: Any, limit: int = 50) -> Dict[str, Any]:
    traces = (
        _safe_call(server.get_automation_decision_traces, limit, default=[])
        if hasattr(server, "get_automation_decision_traces")
        else []
    )
    if not isinstance(traces, list):
        traces = []
    blocked_count = sum(1 for row in traces if isinstance(row, dict) and row.get("blocked_reason"))
    applied_count = sum(1 for row in traces if isinstance(row, dict) and row.get("status") == "applied")
    verified_count = sum(1 for row in traces if isinstance(row, dict) and row.get("proof_status") == "verified")
    return {
        "type": "decision_traces",
        "generated_at": _utc_now(),
        "status": "ok",
        "source": "automation_trace_store",
        "artifact_path": str(getattr(getattr(server, "automation_trace_store", None), "path", "")),
        "summary": {
            "trace_count": len(traces),
            "applied_count": applied_count,
            "blocked_count": blocked_count,
            "verified_count": verified_count,
        },
        "traces": traces,
    }


def build_safety_monitor(server: Any) -> Dict[str, Any]:
    operator_mode = server.get_operator_mode_status()
    dashboard = build_dashboard_snapshot(server)
    connection = dashboard.get("connection", {})
    safety = dashboard.get("safety", {})
    ready_for_live = safety.get("ready_for_live") if isinstance(safety, dict) else {}
    if not isinstance(ready_for_live, dict):
        ready_for_live = {}
    manual = (
        _safe_call(server.get_manual_write_supervision_status, default={})
        if hasattr(server, "get_manual_write_supervision_status")
        else {}
    )
    traces = build_decision_traces(server, limit=12)
    last_trace = traces["traces"][-1] if traces["traces"] else None
    quarantined_surfaces = [
        "eq",
        "compressor",
        "gate",
        "pan",
        "phase_delay",
        "fx_sends",
        "routing",
        "snapshots",
        "training_rules",
        "autonomous_apply",
    ]
    return {
        "type": "safety_monitor",
        "generated_at": _utc_now(),
        "status": "ok",
        "source": "runtime_policy",
        "operator_mode": operator_mode,
        "connection": connection,
        "manual_write_supervision": manual,
        "ready_for_live": ready_for_live,
        "live_write_policy": {
            "apply_path": operator_mode.get("apply_path"),
            "policy": operator_mode.get("live_write_policy"),
            "approved_live_write_kinds": manual.get("approved_live_write_kinds", ["fader", "gain"]),
            "requires_approval": operator_mode.get("capabilities", {}).get("requires_approval", True),
            "direct_writes_allowed": False,
            "autonomous_apply_allowed": False,
        },
        "proof": {
            "last_trace": last_trace,
            "last_pre_read": (last_trace or {}).get("pre_read"),
            "last_readback": (last_trace or {}).get("readback"),
            "last_rollback": (last_trace or {}).get("rollback"),
            "last_proof_status": (last_trace or {}).get("proof_status"),
        },
        "blocked_surfaces": quarantined_surfaces,
        "summary": {
            "ready": bool(ready_for_live.get("ready")),
            "mixer_connected": bool(connection.get("connected")),
            "emergency_stopped": bool(manual.get("emergency_stopped")),
            "reset_required": bool(manual.get("reset_required")),
            "trace_count": traces["summary"]["trace_count"],
            "blocked_trace_count": traces["summary"]["blocked_count"],
        },
    }


def _load_replay_summary(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def build_replay_runs(server: Any, limit: int = 20) -> Dict[str, Any]:
    root = Path(
        getattr(server, "_replay_artifacts_dir", None)
        or Path(__file__).resolve().parents[1] / "artifacts" / "replay"
    )
    rows: List[Dict[str, Any]] = []
    if root.exists():
        for summary_path in sorted(root.glob("*/summary.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            summary = _load_replay_summary(summary_path)
            if not summary:
                continue
            metrics = _load_replay_summary(summary_path.parent / "metrics_before_after.json") or {}
            rows.append(
                {
                    "run_id": summary_path.parent.name,
                    "status": summary.get("status", "unknown"),
                    "reason": summary.get("reason"),
                    "generated_at": summary.get("generated_at"),
                    "connection_policy": summary.get("connection_policy"),
                    "write_policy": summary.get("write_policy"),
                    "summary": summary.get("summary", {}),
                    "trace_count": metrics.get("trace_count", 0),
                    "foundation_teacher_example_count": metrics.get("foundation_teacher_example_count", 0),
                    "offline_only": metrics.get("offline_only", True),
                    "live_write": metrics.get("live_write", False),
                    "before_after_available": metrics.get("before_after_available", False),
                    "report_path": str(summary_path.parent / "report.md"),
                    "summary_path": str(summary_path),
                    "decision_traces_path": str(summary_path.parent / "decision_traces.jsonl"),
                    "metrics_path": str(summary_path.parent / "metrics_before_after.json"),
                    "foundation_teacher_examples_path": metrics.get("foundation_teacher_examples_path")
                    or str(summary_path.parent / "foundation_teacher_examples.jsonl"),
                }
            )
            if len(rows) >= limit:
                break
    return {
        "type": "replay_runs",
        "generated_at": _utc_now(),
        "status": "ok",
        "source": "offline_replay_artifacts",
        "artifact_root": str(root),
        "summary": {
            "run_count": len(rows),
            "offline_only": True,
            "live_write": False,
        },
        "runs": rows,
    }


def build_channel_intelligence(server: Any, total_channels: int = DEFAULT_CHANNEL_COUNT) -> Dict[str, Any]:
    inventory = build_channel_inventory(server, total_channels=total_channels)
    traces = build_decision_traces(server, limit=200).get("traces", [])
    feature_snapshots = [
        feature_snapshot_from_channel_inventory(channel)
        for channel in inventory.get("channels", [])
        if isinstance(channel, dict)
    ]
    context = build_inter_channel_context(feature_snapshots)
    snapshots_by_channel = {snapshot.channel_id: snapshot for snapshot in feature_snapshots}
    traces_by_channel: Dict[int, List[Dict[str, Any]]] = {}
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        target = trace.get("target")
        try:
            channel = int(target)
        except (TypeError, ValueError):
            continue
        traces_by_channel.setdefault(channel, []).append(trace)

    rows = []
    for channel in inventory.get("channels", []):
        number = int(channel.get("number") or channel.get("id") or 0)
        meter = channel.get("meter", {}) if isinstance(channel.get("meter"), dict) else {}
        channel_traces = traces_by_channel.get(number, [])
        peak = _round(channel.get("peak_db") or meter.get("peak_db"))
        lufs = _round(channel.get("lufs") or meter.get("lufs"))
        risk = "low"
        if peak is not None and peak > -3:
            risk = "peak_watch"
        if peak is not None and peak > -1:
            risk = "clip_risk"
        foundation_snapshot = snapshots_by_channel.get(number) or feature_snapshot_from_channel_inventory(channel)
        foundation_recommendations = build_agentic_recommendations(foundation_snapshot, context=context)
        foundation_review = safety_review_recommendations(foundation_recommendations)
        rows.append(
            {
                "channel": number,
                "name": channel.get("name"),
                "role": channel.get("role"),
                "group": channel.get("group"),
                "status": channel.get("status"),
                "peak_db": peak,
                "lufs": lufs,
                "recent_trace_count": len(channel_traces),
                "last_trace_status": channel_traces[-1].get("status") if channel_traces else None,
                "intelligibility_state": "shadow_only",
                "gain_ride_state": "shadow_proposal_only",
                "risk": risk,
                "live_apply_allowed": False,
                "foundation": {
                    "source": foundation_snapshot.source,
                    "features": foundation_snapshot.to_dict(),
                    "recommendation_count": len(foundation_recommendations),
                    "recommendations": [item.to_dict() for item in foundation_recommendations[:4]],
                    "safety_review": foundation_review["summary"],
                },
            }
        )
    return {
        "type": "channel_intelligence",
        "generated_at": _utc_now(),
        "status": "ok",
        "source": "runtime_inventory_traces_and_agentic_foundation",
        "summary": {
            "channel_count": len(rows),
            "shadow_only": True,
            "live_apply_allowed": False,
            "foundation_source": "agentic_foundation_v1",
            "foundation_recommendation_count": sum(
                int(row.get("foundation", {}).get("recommendation_count", 0))
                for row in rows
            ),
        },
        "channels": rows,
    }


def _agent_record(
    *,
    agent_id: str,
    name: str,
    role: str,
    status: str,
    task: str,
    confidence: Optional[float],
    tone: str,
    category: str,
    reports_to: Optional[str] = "automixer_director",
    can_write: bool = False,
    write_scope: Optional[List[str]] = None,
    blocked_reason: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "id": agent_id,
        "name": name,
        "role": role,
        "status": status,
        "task": task,
        "confidence": confidence,
        "tone": tone,
        "category": category,
        "reports_to": reports_to,
        "can_write": can_write,
        "write_scope": write_scope or [],
        "blocked_reason": blocked_reason,
    }


def build_operator_agent_roster(server: Any) -> Dict[str, Any]:
    """Return the product-facing multi-agent roster without starting agents.

    The roster is a UI/runtime contract: Director coordinates specialized roles,
    while only the console executor can ever route approved changes to the
    supervised manual gate. This function is read-only and safe offline.
    """

    operator_mode = server.get_operator_mode_status()
    connection = _connection_summary(server)
    audio_capture = _audio_capture_status(server)
    decision_queue = build_decision_queue(server, limit=10)
    queue_summary = decision_queue.get("summary", {})
    audio_running = bool(audio_capture.get("running"))
    proposal_queue_available = bool(decision_queue.get("proposal_queue_available"))
    manual_write_supervision = (
        _safe_call(server.get_manual_write_supervision_status, default={})
        if hasattr(server, "get_manual_write_supervision_status")
        else {}
    )
    can_apply = bool(operator_mode["capabilities"]["can_apply_to_console"])
    mixer_connected = bool(connection.get("connected"))
    emergency_stopped = bool(manual_write_supervision.get("emergency_stopped"))
    executor_ready = bool(can_apply and mixer_connected and not emergency_stopped)
    blocked_reason = None
    if not can_apply:
        blocked_reason = operator_mode.get("live_write_policy") or "operator_mode_blocks_live_write"
    elif not mixer_connected:
        blocked_reason = "mixer_not_connected"
    elif emergency_stopped:
        blocked_reason = "manual_write_emergency_stop_active"

    agents = [
        _agent_record(
            agent_id="automixer_director",
            name="Automixer Director",
            role="Координирует агентов и политику режима",
            status="ACTIVE",
            task=f"Mode {operator_mode['label']} · {operator_mode['live_write_policy']}",
            confidence=90,
            tone="green",
            category="director",
            reports_to=None,
        ),
        _agent_record(
            agent_id="analyzer_agent",
            name="Analyzer Agent",
            role="Читает уровни, LUFS, peak и состояние каналов",
            status="ACTIVE" if audio_running else "STANDBY",
            task="Анализ аудиопотока" if audio_running else "Ожидает audio capture или cached state",
            confidence=84 if audio_running else 55,
            tone="green" if audio_running else "yellow",
            category="analysis",
        ),
        _agent_record(
            agent_id="channel_classifier_agent",
            name="Channel Classifier Agent",
            role="Определяет роли каналов и группы",
            status="ACTIVE" if connection.get("state") else "STANDBY",
            task="Использует cached mixer names и routing defaults",
            confidence=82 if connection.get("state") else 60,
            tone="green" if connection.get("state") else "yellow",
            category="analysis",
        ),
        _agent_record(
            agent_id="spectral_agent",
            name="Spectral Agent",
            role="Следит за спектральным балансом",
            status="ACTIVE" if audio_running else "STANDBY",
            task="Мониторинг спектра" if audio_running else "Offline до появления аудиопотока",
            confidence=80 if audio_running else 50,
            tone="green" if audio_running else "yellow",
            category="analysis",
        ),
        _agent_record(
            agent_id="dynamics_agent",
            name="Dynamics Agent",
            role="Оценивает динамический диапазон и компрессию",
            status="ACTIVE" if audio_running else "STANDBY",
            task="Сравнивает RMS/peak/dynamic range" if audio_running else "Ожидает измерений",
            confidence=78 if audio_running else 50,
            tone="green" if audio_running else "yellow",
            category="analysis",
        ),
        _agent_record(
            agent_id="balance_agent",
            name="Balance Agent",
            role="Формирует предложения по fader/gain балансу",
            status="ACTIVE" if queue_summary.get("pending_count", 0) else "STANDBY",
            task=f"{queue_summary.get('pending_count', 0)} proposal(s) pending",
            confidence=86 if queue_summary.get("pending_count", 0) else 64,
            tone="green" if queue_summary.get("pending_count", 0) else "yellow",
            category="proposal",
        ),
        _agent_record(
            agent_id="safety_agent",
            name="Safety Agent",
            role="Проверяет режим, headroom и запрет опасных writes",
            status="ACTIVE",
            task=f"Allowed writes: {', '.join(operator_mode['capabilities'].get('allowed_live_write_kinds') or ['none'])}",
            confidence=96,
            tone="green",
            category="safety",
        ),
        _agent_record(
            agent_id="proposal_queue_agent",
            name="Proposal Queue Agent",
            role="Хранит предложения, принятия, отклонения и apply results",
            status="ACTIVE" if proposal_queue_available else "UNAVAILABLE",
            task=f"{queue_summary.get('pending_count', 0)} pending · {queue_summary.get('history_count', 0)} history",
            confidence=92 if proposal_queue_available else 0,
            tone="green" if proposal_queue_available else "yellow",
            category="proposal",
        ),
        _agent_record(
            agent_id="console_executor_agent",
            name="Console Executor Agent",
            role="Единственный путь к supervised manual write gate",
            status="ACTIVE" if executor_ready else "BLOCKED",
            task="Ready for approved fader/gain apply" if executor_ready else "Live apply blocked",
            confidence=100 if executor_ready else 0,
            tone="green" if executor_ready else "yellow",
            category="execution",
            can_write=executor_ready,
            write_scope=operator_mode["capabilities"].get("allowed_live_write_kinds") or [],
            blocked_reason=blocked_reason,
        ),
        _agent_record(
            agent_id="memory_learning_agent",
            name="Memory / Learning Agent",
            role="Сохраняет решения оператора как будущие правила",
            status="STANDBY",
            task="Ожидает accepted/applied history",
            confidence=70 if queue_summary.get("history_count", 0) else 45,
            tone="yellow",
            category="learning",
        ),
    ]

    active_count = sum(1 for agent in agents if agent["status"] == "ACTIVE")
    blocked_count = sum(1 for agent in agents if agent["status"] == "BLOCKED")
    standby_count = sum(1 for agent in agents if agent["status"] == "STANDBY")
    return {
        "type": "operator_agent_roster",
        "generated_at": _utc_now(),
        "status": "ok",
        "operator_mode": operator_mode,
        "connection": connection,
        "decision_queue_summary": queue_summary,
        "summary": {
            "agent_count": len(agents),
            "active_count": active_count,
            "standby_count": standby_count,
            "blocked_count": blocked_count,
            "can_live_execute": executor_ready,
            "proposal_queue_available": proposal_queue_available,
        },
        "agents": agents,
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


def build_groups_buses_snapshot(server: Any) -> Dict[str, Any]:
    channel_inventory = build_channel_inventory(server, total_channels=48)
    channels = channel_inventory.get("channels", [])
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for channel in channels:
        group = str(channel.get("group") or "CHANNELS").upper()
        grouped.setdefault(group, []).append(channel)

    buses = []
    for group, rows in sorted(grouped.items()):
        level_values = [row.get("level_db") for row in rows if row.get("level_db") is not None]
        lufs_values = [row.get("lufs") for row in rows if row.get("lufs") is not None]
        dynamic_values = [
            row.get("dynamic_range_db")
            for row in rows
            if row.get("dynamic_range_db") is not None
        ]
        confidence_values = [
            (row.get("ai_recognition") or {}).get("confidence")
            for row in rows
            if isinstance(row.get("ai_recognition"), dict)
            and (row.get("ai_recognition") or {}).get("confidence") is not None
        ]
        active_channels = sum(1 for row in rows if row.get("status") == "active")
        muted_channels = sum(
            1
            for row in rows
            if (row.get("control_state") or {}).get("mute") in {1, True, "1", "true"}
        )
        buses.append(
            {
                "name": f"{group.title().replace('_', ' ')} Bus",
                "role": "Master" if group == "MASTER" else "Group",
                "level": _round(sum(level_values) / len(level_values)) if level_values else None,
                "lufs": _round(sum(lufs_values) / len(lufs_values)) if lufs_values else None,
                "channels": len(rows),
                "active_channels": active_channels,
                "muted_channels": muted_channels,
                "dynamic": _round(sum(dynamic_values) / len(dynamic_values)) if dynamic_values else None,
                "confidence": round(sum(confidence_values) / len(confidence_values)) if confidence_values else None,
                "status": "Active" if active_channels else ("Muted" if muted_channels else "Inactive"),
                "color": "purple" if "VOCAL" in group else "yellow" if "BASS" in group else "blue" if "GUITAR" in group else "green",
                "members": [
                    f"{int(row.get('number') or row.get('id') or 0):02d} {row.get('name') or 'Channel'}"
                    for row in rows
                ],
            }
        )

    return {
        "type": "groups_buses_snapshot",
        "generated_at": _utc_now(),
        "source": "runtime_channel_inventory",
        "available": True,
        "summary": {
            "bus_count": len(buses),
            "channel_count": len(channels),
            "source": channel_inventory.get("source"),
        },
        "buses": buses,
    }


def build_memory_knowledge_snapshot(server: Any) -> Dict[str, Any]:
    roster = build_operator_agent_roster(server)
    queue = build_decision_queue(server, limit=12)
    inventory = build_channel_inventory(server, total_channels=24)
    decision_memory = (
        _safe_call(server.get_decision_memory_entries, 12, default=[])
        if hasattr(server, "get_decision_memory_entries")
        else []
    )

    memories: List[Dict[str, Any]] = []
    for entry in decision_memory[:8]:
        confidence = entry.get("confidence")
        quality = str(entry.get("evidence_quality") or "evidence")
        memories.append(
            {
                "name": str(entry.get("summary") or "Decision evidence memory"),
                "type": f"{quality.title()} Decision Evidence",
                "size": "3.2 KB",
                "confidence": round(float(confidence or 0.82) * 100),
                "updated": str(entry.get("trace_created_at") or entry.get("created_at") or "trace"),
            }
        )

    for agent in roster.get("agents", [])[:8]:
        confidence = agent.get("confidence")
        memories.append(
            {
                "name": f"{agent.get('name', 'Agent')} runtime state",
                "type": "Agent State",
                "size": f"{max(1.0, float(confidence or 50) / 24):.1f} KB",
                "confidence": round(float(confidence or 50)),
                "updated": str(agent.get("task") or "runtime"),
            }
        )

    for item in queue.get("history", [])[:5] + queue.get("pending_actions", [])[:5]:
        status = str(item.get("status") or "pending")
        memories.append(
            {
                "name": f"{item.get('target') or item.get('title') or 'Decision'} memory",
                "type": "Decision History" if status in {"applied", "accepted"} else "Decision Draft",
                "size": "2.4 KB",
                "confidence": 90 if status in {"applied", "accepted"} else 82,
                "updated": str(item.get("created_at") or "queue"),
            }
        )

    for channel in inventory.get("channels", [])[:8]:
        recognition = channel.get("ai_recognition") or {}
        memories.append(
            {
                "name": f"{channel.get('name', 'Channel')} channel profile",
                "type": f"{channel.get('group', 'CHANNELS')} Profile",
                "size": "1.6 KB",
                "confidence": round(float(recognition.get("confidence") or 72)),
                "updated": str(channel.get("status") or "inventory"),
            }
        )

    return {
        "type": "memory_knowledge_snapshot",
        "generated_at": _utc_now(),
        "source": "runtime_product_state",
        "available": True,
        "summary": {
            "memory_count": len(memories),
            "agent_records": len(roster.get("agents", [])),
            "queue_records": len(queue.get("history", [])) + len(queue.get("pending_actions", [])),
            "decision_memory_records": len(decision_memory),
        },
        "memories": memories[:32],
    }


def build_training_sessions_snapshot(server: Any) -> Dict[str, Any]:
    roster = build_operator_agent_roster(server)
    queue = build_decision_queue(server, limit=50)
    inventory = build_channel_inventory(server, total_channels=48)
    summary = queue.get("summary", {})
    agents = roster.get("agents", [])
    channels = inventory.get("channels", [])
    active_channels = inventory.get("summary", {}).get("active_channels") or 0
    pending_count = summary.get("pending_count") or 0
    history_count = summary.get("history_count") or 0
    average_confidence = 0
    confidence_values = [
        float(agent.get("confidence"))
        for agent in agents
        if agent.get("confidence") is not None
    ]
    if confidence_values:
        average_confidence = round(sum(confidence_values) / len(confidence_values))

    sessions = [
        {
            "name": "Agent roster calibration",
            "status": "Running" if agents else "Queued",
            "examples": len(agents),
            "score": f"{average_confidence or 72}%",
            "updated": "runtime",
        },
        {
            "name": "Decision queue learning set",
            "status": "Review" if pending_count else "Complete",
            "examples": int(pending_count) + int(history_count),
            "score": f"{max(80, 96 - int(pending_count) * 2)}%",
            "updated": "queue",
        },
        {
            "name": "Channel profile refresh",
            "status": "Complete" if channels else "Queued",
            "examples": len(channels),
            "score": "92%" if channels else "N/A",
            "updated": f"{active_channels} active",
        },
    ]

    return {
        "type": "training_sessions_snapshot",
        "generated_at": _utc_now(),
        "source": "runtime_product_state",
        "available": True,
        "summary": {
            "session_count": len(sessions),
            "learning_enabled": False,
        },
        "sessions": sessions,
    }


def build_rules_library_snapshot(server: Any) -> Dict[str, Any]:
    operator_mode = server.get_operator_mode_status()
    queue = build_decision_queue(server, limit=12)
    pending = queue.get("pending_actions", [])

    rules = [
        {
            "name": "No direct OSC writes from frontend",
            "category": "Safety",
            "confidence": "100%",
            "state": "Locked",
        },
        {
            "name": "Console changes require supervised gate",
            "category": "Control",
            "confidence": "100%",
            "state": "Locked",
        },
        {
            "name": f"{operator_mode.get('live_write_policy', 'read_only')} policy",
            "category": "Mode",
            "confidence": "100%",
            "state": "Active",
        },
        {
            "name": "Rollback evidence required for live validation",
            "category": "Safety",
            "confidence": "98%",
            "state": "Review" if pending else "Locked",
        },
    ]
    for item in pending[:6]:
        rules.append(
            {
                "name": f"{item.get('title', 'Proposal')} requires director review",
                "category": "Safety" if item.get("severity") in {"high", "critical"} else "Decision",
                "confidence": "94%" if item.get("severity") in {"high", "critical"} else "88%",
                "state": "Review",
            }
        )

    return {
        "type": "rules_library_snapshot",
        "generated_at": _utc_now(),
        "source": "runtime_policy_state",
        "available": True,
        "summary": {
            "rule_count": len(rules),
            "locked_count": sum(1 for rule in rules if rule["state"] == "Locked"),
            "review_count": sum(1 for rule in rules if rule["state"] == "Review"),
        },
        "rules": rules,
    }


def build_agent_communications_snapshot(server: Any) -> Dict[str, Any]:
    roster = build_operator_agent_roster(server)
    queue = build_decision_queue(server, limit=12)
    now = _utc_now()

    communications: List[Dict[str, Any]] = [
        {
            "time": now,
            "from": "Automixer Director",
            "to": "Operator",
            "topic": f"Mode policy: {roster.get('operator_mode', {}).get('live_write_policy', 'read_only')}",
            "status": "Info",
            "tone": "blue",
        }
    ]
    for item in queue.get("pending_actions", [])[:8]:
        communications.append(
            {
                "time": str(item.get("created_at") or now),
                "from": str(item.get("source") or "Decision Agent"),
                "to": "Automixer Director",
                "topic": str(item.get("title") or "Pending operator proposal"),
                "status": "Review",
                "tone": "yellow" if item.get("severity") in {"high", "critical"} else "blue",
            }
        )
    for agent in roster.get("agents", [])[:8]:
        communications.append(
            {
                "time": now,
                "from": str(agent.get("name") or "Agent"),
                "to": "Automixer Director",
                "topic": str(agent.get("task") or agent.get("role") or "Runtime state"),
                "status": "Approved" if agent.get("status") == "ACTIVE" else "Review",
                "tone": str(agent.get("tone") or "blue"),
            }
        )

    return {
        "type": "agent_communications_snapshot",
        "generated_at": now,
        "source": "runtime_agent_bus",
        "available": True,
        "summary": {
            "communication_count": len(communications),
            "agent_count": len(roster.get("agents", [])),
            "pending_count": queue.get("summary", {}).get("pending_count", 0),
        },
        "communications": communications[:30],
    }
