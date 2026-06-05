"""Product-facing read-only state for the operator cockpit UI.

These builders intentionally do not connect to mixers, scan devices, or apply
changes. They translate the backend runtime into stable UI contracts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

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
DEFAULT_BUS_COUNT = 16
DEFAULT_MAIN_COUNT = 4
DEFAULT_DCA_COUNT = 16
DEFAULT_MATRIX_COUNT = 8
DEFAULT_SEND_COUNT = 16
DEFAULT_OUTPUT_COUNT = 64
DEFAULT_OUTPUT_GROUPS = ("MOD", "CRD")
DEFAULT_FX_SLOT_COUNT = 16
DEFAULT_FX_PARAMETER_COUNT = 32
DEFAULT_DLIVE_CHANNEL_COUNT = 48
MAX_DLIVE_CHANNEL_COUNT = 128
DEFAULT_SNAPSHOT_COUNT = 200
DEFAULT_USER_PRESET_COUNT = 200
DEFAULT_CACHE_STALE_AFTER_SEC = 30.0
MIN_ACTIVE_DB = -90.0
DEFAULT_TELEMETRY_STORAGE_DIR = "data/telemetry/sessions"
DEFAULT_SUPERVISED_PROOF_REPORT_STALE_AFTER_SEC = 24 * 60 * 60
DEFAULT_SUPERVISED_PROOF_REPORT_DIRS = (
    "artifacts/no_wing_release_check",
    "artifacts/no_wing_release_check_current_cycle",
    "artifacts/supervised_proof_artifact_report",
)
SUPERVISED_PROOF_REPORT_FILENAMES = (
    "supervised_proof_artifact_report.json",
    "report.json",
)
AUTONOMY_OVERLAY_STATUSES = {
    "manual_remediation_required",
    "manual_review_ready",
    "dry_run_only",
    "no_current_evidence",
}
AUTONOMY_OVERLAY_ROW_STATUSES = {
    "manual_safety_review_candidate",
    "remediation_required",
    "dry_run_only",
    "no_current_evidence",
}
TELEMETRY_ALERT_CONTEXT_ALLOWLIST = {
    "actor",
    "action_type",
    "baseline_lufs_short",
    "current_fader_db",
    "current_lufs_short",
    "peak_dbfs",
    "previous_fader_db",
}

FX_PARAMETER_SCHEMA_BY_FAMILY: Mapping[str, Dict[int, Dict[str, Any]]] = {
    "reverb": {
        1: {"semantic_name": "decay_time", "unit": "seconds", "safe_min": 0.1, "safe_max": 8.0},
        2: {"semantic_name": "predelay", "unit": "milliseconds", "safe_min": 0.0, "safe_max": 250.0},
        3: {"semantic_name": "density", "unit": "ratio", "safe_min": 0.0, "safe_max": 1.0},
        4: {"semantic_name": "brightness", "unit": "ratio", "safe_min": 0.0, "safe_max": 1.0},
        5: {"semantic_name": "high_pass_filter", "unit": "hertz", "safe_min": 20.0, "safe_max": 1000.0},
        6: {"semantic_name": "low_pass_filter", "unit": "hertz", "safe_min": 1000.0, "safe_max": 20000.0},
    },
    "delay": {
        1: {"semantic_name": "left_delay_time", "unit": "milliseconds", "safe_min": 1.0, "safe_max": 2000.0},
        2: {"semantic_name": "right_delay_time", "unit": "milliseconds", "safe_min": 1.0, "safe_max": 2000.0},
        3: {"semantic_name": "feedback", "unit": "ratio", "safe_min": 0.0, "safe_max": 0.75},
        4: {"semantic_name": "stereo_width", "unit": "ratio", "safe_min": 0.0, "safe_max": 1.0},
        5: {"semantic_name": "high_pass_filter", "unit": "hertz", "safe_min": 20.0, "safe_max": 1000.0},
        6: {"semantic_name": "low_pass_filter", "unit": "hertz", "safe_min": 1000.0, "safe_max": 20000.0},
    },
    "modulation": {
        1: {"semantic_name": "rate", "unit": "hertz", "safe_min": 0.01, "safe_max": 10.0},
        2: {"semantic_name": "depth", "unit": "ratio", "safe_min": 0.0, "safe_max": 1.0},
        3: {"semantic_name": "left_delay_time", "unit": "milliseconds", "safe_min": 0.0, "safe_max": 80.0},
        4: {"semantic_name": "right_delay_time", "unit": "milliseconds", "safe_min": 0.0, "safe_max": 80.0},
        5: {"semantic_name": "high_pass_filter", "unit": "hertz", "safe_min": 20.0, "safe_max": 1000.0},
        6: {"semantic_name": "low_pass_filter", "unit": "hertz", "safe_min": 1000.0, "safe_max": 20000.0},
    },
}

FX_MODEL_FAMILY_TOKENS: Mapping[str, Tuple[str, ...]] = {
    "reverb": ("REV", "REVERB", "PLATE", "ROOM", "HALL", "CHAMBER", "AMBI"),
    "delay": ("DLY", "DELAY", "ECHO"),
    "modulation": ("CHORUS", "CHO", "DOUBLER", "MOD", "FLANGER", "PHASER"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _utc_epoch() -> float:
    return datetime.now(timezone.utc).timestamp()


def _coerce_epoch_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, datetime):
        timestamp = value
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.timestamp()
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric > 1_000_000_000_000:
            return numeric / 1000.0
        return numeric
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return _coerce_epoch_seconds(float(text))
        except ValueError:
            pass
        try:
            normalized = text.replace("Z", "+00:00")
            return _coerce_epoch_seconds(datetime.fromisoformat(normalized))
        except ValueError:
            return None
    return None


def _iso_from_epoch(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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


def _safe_int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bounded_int_value(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    numeric = _safe_int_value(value, default)
    return max(minimum, min(numeric, maximum))


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


def _cached_numeric_first(client: Any, *addresses: str) -> Optional[float]:
    for address in addresses:
        value = _cached_numeric(client, address)
        if value is not None:
            return value
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


def _cached_int_first(client: Any, *addresses: str) -> Optional[int]:
    for address in addresses:
        value = _cached_int(client, address)
        if value is not None:
            return value
    return None


def _cached_raw(client: Any, address: str) -> Any:
    state = getattr(client, "state", None)
    if not isinstance(state, dict):
        return None
    return state.get(address)


def _dlive_fader_raw_to_db(value: Any) -> Optional[float]:
    try:
        raw = max(0, min(0x3FFF, int(value)))
    except (TypeError, ValueError):
        return None
    if raw == 0:
        return -100.0
    if raw <= 0x2AAA:
        return round((raw / 0x2AAA) * 100.0 - 100.0, 1)
    return round(((raw - 0x2AAA) / (0x3FFF - 0x2AAA)) * 10.0, 1)


def _state_has_address(client: Any, address: str) -> bool:
    state = getattr(client, "state", None)
    return isinstance(state, dict) and address in state and state.get(address) is not None


def _address_tuple(addresses: Any) -> Tuple[str, ...]:
    if isinstance(addresses, str):
        return (addresses,)
    try:
        return tuple(str(address) for address in addresses if address)
    except TypeError:
        return ()


def _state_has_any_address(client: Any, addresses: Any) -> bool:
    return any(_state_has_address(client, address) for address in _address_tuple(addresses))


def _cached_timestamp_epoch(client: Any, address: str) -> Optional[float]:
    for attr in ("state_timestamps", "_state_timestamps", "state_updated_at", "_state_updated_at"):
        timestamps = getattr(client, attr, None)
        if not isinstance(timestamps, Mapping):
            continue
        raw = timestamps.get(address)
        if isinstance(raw, Mapping):
            raw = raw.get("updated_at") or raw.get("timestamp") or raw.get("ts")
        epoch = _coerce_epoch_seconds(raw)
        if epoch is not None:
            return epoch
    metadata = getattr(client, "state_metadata", None) or getattr(client, "_state_metadata", None)
    if isinstance(metadata, Mapping):
        row = metadata.get(address)
        if isinstance(row, Mapping):
            return _coerce_epoch_seconds(row.get("updated_at") or row.get("timestamp") or row.get("ts"))
    return None


def _cache_freshness_for_field_addresses(
    client: Any,
    field_addresses: Mapping[str, Any],
    *,
    stale_after_sec: float = DEFAULT_CACHE_STALE_AFTER_SEC,
    now_epoch: Optional[float] = None,
) -> Dict[str, Any]:
    now = _utc_epoch() if now_epoch is None else float(now_epoch)
    fields: Dict[str, Dict[str, Any]] = {}
    for field, raw_addresses in field_addresses.items():
        addresses = _address_tuple(raw_addresses)
        selected_address = next(
            (address for address in addresses if _state_has_address(client, address)),
            None,
        )
        updated_epoch = _cached_timestamp_epoch(client, selected_address) if selected_address else None
        age_sec = round(max(0.0, now - updated_epoch), 3) if updated_epoch is not None else None
        if selected_address is None:
            status = "missing"
        elif updated_epoch is None:
            status = "unknown"
        elif age_sec is not None and age_sec > stale_after_sec:
            status = "stale"
        else:
            status = "fresh"
        fields[str(field)] = {
            "status": status,
            "address": selected_address,
            "candidate_addresses": list(addresses),
            "present": selected_address is not None,
            "updated_at": _iso_from_epoch(updated_epoch),
            "age_sec": age_sec,
            "source": "state_timestamps" if updated_epoch is not None else "unavailable",
        }

    statuses = [row["status"] for row in fields.values()]
    if not statuses or all(status == "missing" for status in statuses):
        status = "missing"
    elif "stale" in statuses:
        status = "stale"
    elif "unknown" in statuses:
        status = "unknown"
    elif "missing" in statuses:
        status = "partial"
    else:
        status = "fresh"

    return {
        "status": status,
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "source": "state_timestamps" if any(row["updated_at"] for row in fields.values()) else "unavailable",
        "stale_after_sec": float(stale_after_sec),
        "field_count": len(fields),
        "present_field_count": sum(1 for row in fields.values() if row["present"]),
        "timestamped_field_count": sum(1 for row in fields.values() if row["updated_at"]),
        "fresh_field_count": sum(1 for row in fields.values() if row["status"] == "fresh"),
        "stale_field_count": sum(1 for row in fields.values() if row["status"] == "stale"),
        "unknown_field_count": sum(1 for row in fields.values() if row["status"] == "unknown"),
        "missing_field_count": sum(1 for row in fields.values() if row["status"] == "missing"),
        "missing_fields": [field for field, row in fields.items() if row["status"] == "missing"],
        "stale_fields": [field for field, row in fields.items() if row["status"] == "stale"],
        "untimestamped_present_fields": [
            field
            for field, row in fields.items()
            if row["present"] and row["updated_at"] is None
        ],
        "fields": fields,
    }


def _cache_freshness_for_addresses(
    client: Any,
    addresses: Iterable[str],
    *,
    stale_after_sec: float = DEFAULT_CACHE_STALE_AFTER_SEC,
    now_epoch: Optional[float] = None,
) -> Dict[str, Any]:
    unique_addresses = tuple(dict.fromkeys(str(address) for address in addresses if address))
    state = getattr(client, "state", None)
    present_addresses = [
        address
        for address in unique_addresses
        if isinstance(state, dict) and address in state and state.get(address) is not None
    ]
    now = _utc_epoch() if now_epoch is None else float(now_epoch)
    timestamp_rows = []
    for address in present_addresses:
        updated_epoch = _cached_timestamp_epoch(client, address)
        if updated_epoch is None:
            continue
        timestamp_rows.append({
            "address": address,
            "updated_at": _iso_from_epoch(updated_epoch),
            "age_sec": round(max(0.0, now - updated_epoch), 3),
        })
    ages = [float(row["age_sec"]) for row in timestamp_rows]
    max_age = max(ages) if ages else None
    min_age = min(ages) if ages else None
    if not present_addresses:
        status = "missing"
    elif not timestamp_rows:
        status = "unknown"
    elif max_age is not None and max_age > stale_after_sec:
        status = "stale"
    elif len(timestamp_rows) < len(present_addresses):
        status = "partial"
    else:
        status = "fresh"
    tracked = {row["address"] for row in timestamp_rows}
    return {
        "status": status,
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "source": "state_timestamps" if timestamp_rows else "unavailable",
        "stale_after_sec": float(stale_after_sec),
        "address_count": len(unique_addresses),
        "present_address_count": len(present_addresses),
        "timestamped_address_count": len(timestamp_rows),
        "missing_address_count": len(unique_addresses) - len(present_addresses),
        "untimestamped_present_addresses": [
            address
            for address in present_addresses
            if address not in tracked
        ],
        "oldest_updated_at": _iso_from_epoch(min(_coerce_epoch_seconds(row["updated_at"]) for row in timestamp_rows))
        if timestamp_rows
        else None,
        "newest_updated_at": _iso_from_epoch(max(_coerce_epoch_seconds(row["updated_at"]) for row in timestamp_rows))
        if timestamp_rows
        else None,
        "max_age_sec": round(max_age, 3) if max_age is not None else None,
        "min_age_sec": round(min_age, 3) if min_age is not None else None,
    }


def _read_coverage_status(fields: Mapping[str, bool]) -> str:
    if not fields or not any(fields.values()):
        return "missing"
    if all(fields.values()):
        return "complete"
    return "partial"


def _read_coverage_section(fields: Mapping[str, bool], source: str) -> Dict[str, Any]:
    normalized_fields = {key: bool(value) for key, value in fields.items()}
    missing_fields = [key for key, value in normalized_fields.items() if not value]
    return {
        "status": _read_coverage_status(normalized_fields),
        "source": source,
        "read_only": True,
        "mutates_console_state": False,
        "fields": normalized_fields,
        "covered_field_count": len(normalized_fields) - len(missing_fields),
        "field_count": len(normalized_fields),
        "missing_fields": missing_fields,
    }


def _coverage_source(*sources: str) -> str:
    present = tuple(dict.fromkeys(source for source in sources if source and source != "unavailable"))
    return "+".join(present) if present else "unavailable"


def _overall_read_coverage_status(statuses: Iterable[str]) -> str:
    values = tuple(statuses)
    if not values or all(status == "missing" for status in values):
        return "missing"
    if all(status == "complete" for status in values):
        return "complete"
    return "partial"


def _format_db_label(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _gate_display_label(enabled: Optional[int], threshold_db: Optional[float]) -> Optional[str]:
    if enabled == 0:
        return "G off"
    if threshold_db is not None:
        return f"G {_format_db_label(threshold_db)}"
    if enabled == 1:
        return "G on"
    return None


def _compressor_display_label(
    enabled: Optional[int],
    gain_reduction_db: Optional[float],
    threshold_db: Optional[float],
) -> Optional[str]:
    if enabled == 0:
        return "C off"
    if gain_reduction_db is not None:
        return f"C {_format_db_label(gain_reduction_db)} GR"
    if threshold_db is not None:
        return f"C thr {_format_db_label(threshold_db)}"
    if enabled == 1:
        return "C on"
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


def _source_patch_label(source_group: Any, source_channel: Any) -> Optional[str]:
    group = str(source_group or "").strip().upper()
    channel = _safe_int_value(source_channel, 0) if source_channel is not None else 0
    if group and channel > 0:
        return f"{group} {channel}"
    if group:
        return group
    if channel > 0:
        return f"Input {channel}"
    return None


def _channel_source_patch_snapshot(client: Any, channel: int) -> Dict[str, Any]:
    addresses = {
        "main_source_group": f"/ch/{channel}/in/conn/grp",
        "main_source_channel": f"/ch/{channel}/in/conn/in",
        "alt_source_group": f"/ch/{channel}/in/conn/altgrp",
        "alt_source_channel": f"/ch/{channel}/in/conn/altin",
    }
    fields = {
        field: _state_has_address(client, address)
        for field, address in addresses.items()
    }
    read_coverage = _read_coverage_section(
        fields,
        "mixer_state_cache" if any(fields.values()) else "unavailable",
    )
    read_coverage["cache_freshness"] = _cache_freshness_for_field_addresses(client, addresses)
    main_group = _cached_raw(client, addresses["main_source_group"])
    main_channel = _cached_int(client, addresses["main_source_channel"])
    alt_group = _cached_raw(client, addresses["alt_source_group"])
    alt_channel = _cached_int(client, addresses["alt_source_channel"])
    main_label = _source_patch_label(main_group, main_channel)
    alt_label = _source_patch_label(alt_group, alt_channel)
    return {
        "channel": channel,
        "main": {
            "source_group": main_group,
            "source_channel": main_channel,
            "label": main_label,
            "available": bool(main_label),
        },
        "alt": {
            "source_group": alt_group,
            "source_channel": alt_channel,
            "label": alt_label,
            "available": bool(alt_label),
        },
        "active_label": main_label,
        "input_assignment_policy": "blocked_manual_only",
        "destructive_change_marker": "source_patch_change_would_repatch_channel_input",
        "read_coverage": read_coverage,
        "cache_freshness": read_coverage["cache_freshness"],
    }


def _channel_read_coverage(
    client: Any,
    channel: int,
    *,
    cached_name: Optional[str],
    routing_item: Mapping[str, Any],
    routing_role: str,
    meter: Mapping[str, Any],
) -> Dict[str, Any]:
    identity_fields = {
        "cached_name": bool(cached_name),
        "published_name": True,
        "routing_role": routing_role != "unknown",
        "role_inference": True,
        "group_inference": True,
    }
    routing_fields = {
        "dante_channel": True,
        "routing_role": routing_role != "unknown",
        "tap_point": bool(routing_item.get("tap_point")),
        "routing_label": bool(routing_item.get("label_short")),
    }
    meter_fields = {
        "rms_db": bool(meter.get("available") and meter.get("rms_db") is not None),
        "peak_db": bool(meter.get("available") and meter.get("peak_db") is not None),
        "lufs": bool(meter.get("available") and meter.get("lufs") is not None),
        "dynamic_range_db": bool(meter.get("available") and meter.get("dynamic_range_db") is not None),
    }
    basic_control_addresses = {
        "fader_db": f"/ch/{channel}/fdr",
        "gain_db": f"/ch/{channel}/in/set/trim",
        "mute": f"/ch/{channel}/mute",
        "pan": f"/ch/{channel}/pan",
    }
    processing_addresses = {
        "gate_enabled": f"/ch/{channel}/gate/on",
        "gate_threshold_db": f"/ch/{channel}/gate/thr",
        "compressor_enabled": f"/ch/{channel}/dyn/on",
        "compressor_threshold_db": f"/ch/{channel}/dyn/thr26",
        "compressor_gain_reduction_db": f"/ch/{channel}/dyn/gr",
    }
    control_fields = {
        field: _state_has_address(client, address)
        for field, address in basic_control_addresses.items()
    }
    processing_fields = {
        "gate_enabled": _state_has_address(client, processing_addresses["gate_enabled"]),
        "gate_threshold_db": _state_has_address(client, processing_addresses["gate_threshold_db"]),
        "compressor_enabled": _state_has_address(client, processing_addresses["compressor_enabled"]),
        "compressor_threshold_db": (
            _state_has_address(client, processing_addresses["compressor_threshold_db"])
            or _state_has_address(client, f"/ch/{channel}/dyn/thr")
        ),
        "compressor_gain_reduction_db": (
            _state_has_address(client, processing_addresses["compressor_gain_reduction_db"])
            or _state_has_address(client, f"/ch/{channel}/dyn/$gr")
        ),
    }
    source_patch_fields = {
        "main_source_group": _state_has_address(client, f"/ch/{channel}/in/conn/grp"),
        "main_source_channel": _state_has_address(client, f"/ch/{channel}/in/conn/in"),
        "alt_source_group": _state_has_address(client, f"/ch/{channel}/in/conn/altgrp"),
        "alt_source_channel": _state_has_address(client, f"/ch/{channel}/in/conn/altin"),
    }

    sections = {
        "identity": _read_coverage_section(
            identity_fields,
            _coverage_source(
                "mixer_state_cache" if cached_name else "",
                "static_routing" if routing_role != "unknown" else "routing_default",
            ),
        ),
        "routing": _read_coverage_section(
            routing_fields,
            "static_routing" if routing_item else "routing_default",
        ),
        "metering": _read_coverage_section(
            meter_fields,
            "audio_capture" if meter.get("available") else "unavailable",
        ),
        "basic_controls": _read_coverage_section(
            control_fields,
            "mixer_state_cache" if any(control_fields.values()) else "unavailable",
        ),
        "source_patch": _read_coverage_section(
            source_patch_fields,
            "mixer_state_cache" if any(source_patch_fields.values()) else "unavailable",
        ),
        "processing": _read_coverage_section(
            processing_fields,
            "mixer_state_cache" if any(processing_fields.values()) else "unavailable",
        ),
    }
    section_statuses = {name: section["status"] for name, section in sections.items()}
    return {
        "status": _overall_read_coverage_status(section_statuses.values()),
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "sections": sections,
        "section_statuses": section_statuses,
        "missing_sections": [name for name, status in section_statuses.items() if status == "missing"],
        "partial_sections": [name for name, status in section_statuses.items() if status == "partial"],
        "source_kinds": sorted(
            {
                source
                for section in sections.values()
                for source in str(section.get("source") or "").split("+")
                if source and source != "unavailable"
            }
        ),
    }


def _channel_inventory_read_coverage(channels: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    channel_rows = tuple(channels)
    channel_status_counts: Dict[str, int] = {"complete": 0, "partial": 0, "missing": 0}
    section_status_counts: Dict[str, Dict[str, int]] = {}
    source_kinds = set()
    for channel in channel_rows:
        coverage = channel.get("read_coverage") if isinstance(channel, dict) else {}
        status = str(coverage.get("status") or "missing") if isinstance(coverage, dict) else "missing"
        channel_status_counts[status] = channel_status_counts.get(status, 0) + 1
        if isinstance(coverage, dict):
            for source in coverage.get("source_kinds", ()):
                source_kinds.add(str(source))
            sections = coverage.get("sections", {})
            if isinstance(sections, dict):
                for section_name, section in sections.items():
                    section_status_counts.setdefault(section_name, {"complete": 0, "partial": 0, "missing": 0})
                    section_status = (
                        str(section.get("status") or "missing")
                        if isinstance(section, dict)
                        else "missing"
                    )
                    section_status_counts[section_name][section_status] = (
                        section_status_counts[section_name].get(section_status, 0) + 1
                    )

    statuses = [
        status
        for status, count in channel_status_counts.items()
        for _ in range(count)
    ]
    return {
        "status": _overall_read_coverage_status(statuses),
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "channel_status_counts": channel_status_counts,
        "section_status_counts": section_status_counts,
        "channels_with_complete_basic_controls": section_status_counts.get("basic_controls", {}).get("complete", 0),
        "channels_with_complete_processing": section_status_counts.get("processing", {}).get("complete", 0),
        "source_kinds": sorted(source_kinds),
    }


def _channel_inventory_cache_addresses(total_channels: int) -> List[str]:
    addresses: List[str] = []
    for channel in range(1, total_channels + 1):
        addresses.extend([
            f"/ch/{channel}/$name",
            f"/ch/{channel}/name",
            f"/ch/{channel}/fdr",
            f"/ch/{channel}/in/set/trim",
            f"/ch/{channel}/mute",
            f"/ch/{channel}/pan",
            f"/ch/{channel}/in/conn/grp",
            f"/ch/{channel}/in/conn/in",
            f"/ch/{channel}/in/conn/altgrp",
            f"/ch/{channel}/in/conn/altin",
            f"/ch/{channel}/gate/on",
            f"/ch/{channel}/gate/thr",
            f"/ch/{channel}/dyn/on",
            f"/ch/{channel}/dyn/thr26",
            f"/ch/{channel}/dyn/thr",
            f"/ch/{channel}/dyn/gr",
            f"/ch/{channel}/dyn/$gr",
        ])
    return addresses


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
        source_patch = _channel_source_patch_snapshot(client, channel)
        fallback_source = f"In {channel:02d}"
        source_label = source_patch.get("active_label") or fallback_source
        pan = _cached_numeric(client, f"/ch/{channel}/pan")
        gate_enabled = _cached_int(client, f"/ch/{channel}/gate/on")
        gate_threshold_db = _cached_numeric(client, f"/ch/{channel}/gate/thr")
        compressor_enabled = _cached_int(client, f"/ch/{channel}/dyn/on")
        compressor_threshold_db = _cached_numeric_first(
            client,
            f"/ch/{channel}/dyn/thr26",
            f"/ch/{channel}/dyn/thr",
        )
        compressor_gain_reduction_db = _cached_numeric_first(
            client,
            f"/ch/{channel}/dyn/gr",
            f"/ch/{channel}/dyn/$gr",
        )
        read_coverage = _channel_read_coverage(
            client,
            channel,
            cached_name=cached_name,
            routing_item=routing_item,
            routing_role=routing_role,
            meter=meter,
        )

        channels.append(
            {
                "id": channel,
                "number": channel,
                "name": name,
                "source": source_label,
                "fallback_source": fallback_source,
                "source_patch": source_patch,
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
                "read_coverage": read_coverage,
                "control_state": {
                    "fader_db": _cached_numeric(client, f"/ch/{channel}/fdr"),
                    "gain_db": _cached_numeric(client, f"/ch/{channel}/in/set/trim"),
                    "mute": _cached_int(client, f"/ch/{channel}/mute"),
                    "pan": pan,
                    "gate_enabled": gate_enabled,
                    "gate_threshold_db": gate_threshold_db,
                    "gate_label": _gate_display_label(gate_enabled, gate_threshold_db),
                    "compressor_enabled": compressor_enabled,
                    "compressor_threshold_db": compressor_threshold_db,
                    "compressor_gain_reduction_db": compressor_gain_reduction_db,
                    "comp_label": _compressor_display_label(
                        compressor_enabled,
                        compressor_gain_reduction_db,
                        compressor_threshold_db,
                    ),
                },
                "ai_recognition": {
                    "label": inferred["role"],
                    "confidence": 99 if cached_name else None,
                    "source": "cached_mixer_name" if cached_name else "routing_default",
                },
            }
        )

    inventory_read_coverage = _channel_inventory_read_coverage(channels)
    cache_freshness = _cache_freshness_for_addresses(
        client,
        _channel_inventory_cache_addresses(total_channels),
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
            "read_coverage_status": inventory_read_coverage["status"],
            "cache_freshness_status": cache_freshness["status"],
        },
        "read_coverage": inventory_read_coverage,
        "cache_freshness": cache_freshness,
        "channels": channels,
        "routing_scheme": routing,
    }


def _processing_section_coverage(
    fields: Mapping[str, bool],
    *,
    client: Any = None,
    field_addresses: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    coverage = _read_coverage_section(
        fields,
        "mixer_state_cache" if any(fields.values()) else "unavailable",
    )
    if client is not None and field_addresses is not None:
        coverage["cache_freshness"] = _cache_freshness_for_field_addresses(client, field_addresses)
    return coverage


def _eq_band_snapshot(client: Any, channel: int, band: int) -> Dict[str, Any]:
    addresses = {
        "frequency_hz": f"/ch/{channel}/eq/{band}f",
        "gain_db": f"/ch/{channel}/eq/{band}g",
        "q": f"/ch/{channel}/eq/{band}q",
    }
    fields = {field: _state_has_any_address(client, address) for field, address in addresses.items()}
    read_coverage = _processing_section_coverage(
        fields,
        client=client,
        field_addresses=addresses,
    )
    return {
        "band": band,
        "frequency_hz": _cached_numeric(client, addresses["frequency_hz"]),
        "gain_db": _cached_numeric(client, addresses["gain_db"]),
        "q": _cached_numeric(client, addresses["q"]),
        "read_coverage": read_coverage,
        "cache_freshness": read_coverage["cache_freshness"],
    }


def _channel_processing_snapshot_row(channel: Mapping[str, Any], client: Any) -> Dict[str, Any]:
    number = int(channel.get("number") or channel.get("id") or 0)
    hpf_addresses = {
        "enabled": f"/ch/{number}/flt/lc",
        "frequency_hz": f"/ch/{number}/flt/lcf",
    }
    hpf_fields = {
        field: _state_has_any_address(client, addresses)
        for field, addresses in hpf_addresses.items()
    }
    eq_bands = [_eq_band_snapshot(client, number, band) for band in range(1, 5)]
    eq_addresses: Dict[str, Any] = {"enabled": f"/ch/{number}/eq/on"}
    for band in eq_bands:
        band_number = band["band"]
        eq_addresses[f"band_{band_number}_frequency_hz"] = f"/ch/{number}/eq/{band_number}f"
        eq_addresses[f"band_{band_number}_gain_db"] = f"/ch/{number}/eq/{band_number}g"
        eq_addresses[f"band_{band_number}_q"] = f"/ch/{number}/eq/{band_number}q"
    eq_fields = {
        field: _state_has_any_address(client, addresses)
        for field, addresses in eq_addresses.items()
    }

    compressor_addresses = {
        "enabled": f"/ch/{number}/dyn/on",
        "model": f"/ch/{number}/dyn/mdl",
        "threshold_db": (f"/ch/{number}/dyn/thr26", f"/ch/{number}/dyn/thr"),
        "ratio": f"/ch/{number}/dyn/ratio",
        "attack_ms": f"/ch/{number}/dyn/att",
        "release_ms": f"/ch/{number}/dyn/rel",
        "makeup_db": f"/ch/{number}/dyn/gain",
        "mix_pct": f"/ch/{number}/dyn/mix",
        "detector": f"/ch/{number}/dyn/det",
        "gain_reduction_db": (f"/ch/{number}/dyn/gr", f"/ch/{number}/dyn/$gr"),
    }
    compressor_fields = {
        field: _state_has_any_address(client, addresses)
        for field, addresses in compressor_addresses.items()
    }
    gate_addresses = {
        "enabled": f"/ch/{number}/gate/on",
        "model": f"/ch/{number}/gate/mdl",
        "threshold_db": f"/ch/{number}/gate/thr",
        "range_db": f"/ch/{number}/gate/range",
        "attack_ms": f"/ch/{number}/gate/att",
        "hold_ms": f"/ch/{number}/gate/hld",
        "release_ms": f"/ch/{number}/gate/rel",
        "accent": f"/ch/{number}/gate/acc",
        "ratio": f"/ch/{number}/gate/ratio",
    }
    gate_fields = {
        field: _state_has_any_address(client, addresses)
        for field, addresses in gate_addresses.items()
    }
    sections = {
        "hpf": _processing_section_coverage(
            hpf_fields,
            client=client,
            field_addresses=hpf_addresses,
        ),
        "eq": _processing_section_coverage(
            eq_fields,
            client=client,
            field_addresses=eq_addresses,
        ),
        "compressor": _processing_section_coverage(
            compressor_fields,
            client=client,
            field_addresses=compressor_addresses,
        ),
        "gate": _processing_section_coverage(
            gate_fields,
            client=client,
            field_addresses=gate_addresses,
        ),
    }
    section_statuses = {name: section["status"] for name, section in sections.items()}

    return {
        "channel": number,
        "name": channel.get("name"),
        "role": channel.get("role"),
        "group": channel.get("group"),
        "source": channel.get("source"),
        "hpf": {
            "enabled": _cached_int(client, f"/ch/{number}/flt/lc"),
            "frequency_hz": _cached_numeric(client, f"/ch/{number}/flt/lcf"),
            "read_coverage": sections["hpf"],
        },
        "eq": {
            "enabled": _cached_int(client, f"/ch/{number}/eq/on"),
            "bands": eq_bands,
            "read_coverage": sections["eq"],
        },
        "compressor": {
            "enabled": _cached_int(client, f"/ch/{number}/dyn/on"),
            "model": _cached_raw(client, f"/ch/{number}/dyn/mdl"),
            "threshold_db": _cached_numeric_first(
                client,
                f"/ch/{number}/dyn/thr26",
                f"/ch/{number}/dyn/thr",
            ),
            "ratio": _cached_raw(client, f"/ch/{number}/dyn/ratio"),
            "attack_ms": _cached_numeric(client, f"/ch/{number}/dyn/att"),
            "release_ms": _cached_numeric(client, f"/ch/{number}/dyn/rel"),
            "makeup_db": _cached_numeric(client, f"/ch/{number}/dyn/gain"),
            "mix_pct": _cached_numeric(client, f"/ch/{number}/dyn/mix"),
            "detector": _cached_raw(client, f"/ch/{number}/dyn/det"),
            "gain_reduction_db": _cached_numeric_first(
                client,
                f"/ch/{number}/dyn/gr",
                f"/ch/{number}/dyn/$gr",
            ),
            "read_coverage": sections["compressor"],
        },
        "gate": {
            "enabled": _cached_int(client, f"/ch/{number}/gate/on"),
            "model": _cached_raw(client, f"/ch/{number}/gate/mdl"),
            "threshold_db": _cached_numeric(client, f"/ch/{number}/gate/thr"),
            "range_db": _cached_numeric(client, f"/ch/{number}/gate/range"),
            "attack_ms": _cached_numeric(client, f"/ch/{number}/gate/att"),
            "hold_ms": _cached_numeric(client, f"/ch/{number}/gate/hld"),
            "release_ms": _cached_numeric(client, f"/ch/{number}/gate/rel"),
            "accent": _cached_numeric(client, f"/ch/{number}/gate/acc"),
            "ratio": _cached_raw(client, f"/ch/{number}/gate/ratio"),
            "read_coverage": sections["gate"],
        },
        "read_coverage": {
            "status": _overall_read_coverage_status(section_statuses.values()),
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "sections": sections,
            "section_statuses": section_statuses,
            "missing_sections": [name for name, status in section_statuses.items() if status == "missing"],
            "partial_sections": [name for name, status in section_statuses.items() if status == "partial"],
            "source_kinds": ["mixer_state_cache"]
            if any(section["status"] != "missing" for section in sections.values())
            else [],
        },
    }


def _channel_processing_coverage_summary(channels: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = tuple(channels)
    channel_status_counts: Dict[str, int] = {"complete": 0, "partial": 0, "missing": 0}
    section_status_counts: Dict[str, Dict[str, int]] = {}
    for row in rows:
        coverage = row.get("read_coverage") if isinstance(row, dict) else {}
        status = str(coverage.get("status") or "missing") if isinstance(coverage, dict) else "missing"
        channel_status_counts[status] = channel_status_counts.get(status, 0) + 1
        sections = coverage.get("sections", {}) if isinstance(coverage, dict) else {}
        if isinstance(sections, dict):
            for section_name, section in sections.items():
                section_status_counts.setdefault(section_name, {"complete": 0, "partial": 0, "missing": 0})
                section_status = (
                    str(section.get("status") or "missing")
                    if isinstance(section, dict)
                    else "missing"
                )
                section_status_counts[section_name][section_status] = (
                    section_status_counts[section_name].get(section_status, 0) + 1
                )

    statuses = [
        status
        for status, count in channel_status_counts.items()
        for _ in range(count)
    ]
    return {
        "status": _overall_read_coverage_status(statuses),
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "channel_status_counts": channel_status_counts,
        "section_status_counts": section_status_counts,
        "channels_with_any_processing_cache": sum(
            1
            for row in rows
            if (row.get("read_coverage") or {}).get("status") != "missing"
        ),
        "channels_with_complete_processing_cache": channel_status_counts.get("complete", 0),
    }


def _channel_processing_cache_addresses(total_channels: int) -> List[str]:
    addresses: List[str] = []
    for channel in range(1, total_channels + 1):
        addresses.extend([
            f"/ch/{channel}/flt/lc",
            f"/ch/{channel}/flt/lcf",
            f"/ch/{channel}/eq/on",
            f"/ch/{channel}/dyn/on",
            f"/ch/{channel}/dyn/mdl",
            f"/ch/{channel}/dyn/thr26",
            f"/ch/{channel}/dyn/thr",
            f"/ch/{channel}/dyn/ratio",
            f"/ch/{channel}/dyn/att",
            f"/ch/{channel}/dyn/rel",
            f"/ch/{channel}/dyn/gain",
            f"/ch/{channel}/dyn/mix",
            f"/ch/{channel}/dyn/det",
            f"/ch/{channel}/dyn/gr",
            f"/ch/{channel}/dyn/$gr",
            f"/ch/{channel}/gate/on",
            f"/ch/{channel}/gate/mdl",
            f"/ch/{channel}/gate/thr",
            f"/ch/{channel}/gate/range",
            f"/ch/{channel}/gate/att",
            f"/ch/{channel}/gate/hld",
            f"/ch/{channel}/gate/rel",
            f"/ch/{channel}/gate/acc",
            f"/ch/{channel}/gate/ratio",
        ])
        for band in range(1, 5):
            addresses.extend([
                f"/ch/{channel}/eq/{band}f",
                f"/ch/{channel}/eq/{band}g",
                f"/ch/{channel}/eq/{band}q",
            ])
    return addresses


def build_channel_processing_snapshot(
    server: Any,
    total_channels: int = DEFAULT_CHANNEL_COUNT,
) -> Dict[str, Any]:
    """Return cached channel processing state without querying or mutating a console."""

    inventory = build_channel_inventory(server, total_channels=total_channels)
    client = getattr(server, "mixer_client", None)
    channels = [
        _channel_processing_snapshot_row(channel, client)
        for channel in inventory.get("channels", [])
        if isinstance(channel, dict)
    ]
    read_coverage = _channel_processing_coverage_summary(channels)
    cache_freshness = _cache_freshness_for_addresses(
        client,
        _channel_processing_cache_addresses(len(channels)),
    )
    return {
        "type": "channel_processing_snapshot",
        "generated_at": _utc_now(),
        "source": "runtime_cached_state",
        "available": True,
        "connection": inventory.get("connection", _connection_summary(server)),
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "channel_count": len(channels),
            "read_coverage_status": read_coverage["status"],
            "channels_with_any_processing_cache": read_coverage["channels_with_any_processing_cache"],
            "channels_with_complete_processing_cache": read_coverage["channels_with_complete_processing_cache"],
            "cache_freshness_status": cache_freshness["status"],
            "live_apply_allowed": False,
            "claim": "cached processing read coverage is exposed but not complete autonomous proof",
        },
        "read_coverage": read_coverage,
        "cache_freshness": cache_freshness,
        "channels": channels,
    }


def _console_surface_section_coverage(fields: Mapping[str, bool]) -> Dict[str, Any]:
    return _read_coverage_section(
        fields,
        "mixer_state_cache" if any(fields.values()) else "unavailable",
    )


def _console_surface_section_coverage_for_addresses(
    client: Any,
    field_addresses: Mapping[str, Any],
) -> Dict[str, Any]:
    fields = {
        field: _state_has_any_address(client, addresses)
        for field, addresses in field_addresses.items()
    }
    coverage = _console_surface_section_coverage(fields)
    coverage["cache_freshness"] = _cache_freshness_for_field_addresses(client, field_addresses)
    return coverage


def _console_eq_band_snapshot(client: Any, prefix: str, band: int) -> Dict[str, Any]:
    addresses = {
        "frequency_hz": f"{prefix}/eq/{band}f",
        "gain_db": f"{prefix}/eq/{band}g",
        "q": f"{prefix}/eq/{band}q",
    }
    fields = {field: _state_has_address(client, address) for field, address in addresses.items()}
    return {
        "band": band,
        "frequency_hz": _cached_numeric(client, addresses["frequency_hz"]),
        "gain_db": _cached_numeric(client, addresses["gain_db"]),
        "q": _cached_numeric(client, addresses["q"]),
        "read_coverage": _console_surface_section_coverage(fields),
    }


def _console_eq_section(client: Any, prefix: str, band_count: int) -> Dict[str, Any]:
    bands = [_console_eq_band_snapshot(client, prefix, band) for band in range(1, band_count + 1)]
    fields = {"enabled": _state_has_address(client, f"{prefix}/eq/on")}
    for band in bands:
        coverage = band.get("read_coverage", {})
        band_fields = coverage.get("fields", {}) if isinstance(coverage, dict) else {}
        fields[f"band_{band['band']}_frequency_hz"] = bool(band_fields.get("frequency_hz"))
        fields[f"band_{band['band']}_gain_db"] = bool(band_fields.get("gain_db"))
        fields[f"band_{band['band']}_q"] = bool(band_fields.get("q"))
    return {
        "enabled": _cached_int(client, f"{prefix}/eq/on"),
        "bands": bands,
        "read_coverage": _console_surface_section_coverage(fields),
    }


def _section_status_payload(sections: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    section_statuses = {
        name: str(section.get("status") or "missing")
        for name, section in sections.items()
    }
    return {
        "status": _overall_read_coverage_status(section_statuses.values()),
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "sections": dict(sections),
        "section_statuses": section_statuses,
        "missing_sections": [name for name, status in section_statuses.items() if status == "missing"],
        "partial_sections": [name for name, status in section_statuses.items() if status == "partial"],
        "source_kinds": ["mixer_state_cache"]
        if any(status != "missing" for status in section_statuses.values())
        else [],
    }


def _parse_cached_dca_assignments(client: Any, tags: Any) -> List[int]:
    parser = getattr(client, "_parse_dca_assignments", None)
    if callable(parser):
        parsed = _safe_call(parser, tags, default=[])
        if isinstance(parsed, list):
            return [int(item) for item in parsed if isinstance(item, int)]
    return []


def _bus_surface_row(client: Any, bus: int) -> Dict[str, Any]:
    prefix = f"/bus/{bus}"
    name = _cached_raw(client, f"{prefix}/$name") or _cached_raw(client, f"{prefix}/name")
    tags = _cached_raw(client, f"{prefix}/tags")
    eq = _console_eq_section(client, prefix, 8)
    identity_fields = {
        "name": _state_has_address(client, f"{prefix}/$name") or _state_has_address(client, f"{prefix}/name"),
        "tags": _state_has_address(client, f"{prefix}/tags"),
    }
    basic_fields = {
        "fader_db": _state_has_address(client, f"{prefix}/fdr"),
        "mute": _state_has_address(client, f"{prefix}/mute"),
        "pan": _state_has_address(client, f"{prefix}/pan"),
        "width": _state_has_address(client, f"{prefix}/wid"),
        "mono": _state_has_address(client, f"{prefix}/busmono"),
    }
    dynamics_fields = {
        "enabled": _state_has_address(client, f"{prefix}/dyn/on"),
        "model": _state_has_address(client, f"{prefix}/dyn/mdl"),
        "threshold_db": _state_has_address(client, f"{prefix}/dyn/thr37"),
        "ratio": _state_has_address(client, f"{prefix}/dyn/ratio"),
        "knee": _state_has_address(client, f"{prefix}/dyn/knee"),
        "detector": _state_has_address(client, f"{prefix}/dyn/det"),
        "attack_ms": _state_has_address(client, f"{prefix}/dyn/att"),
        "release_ms": _state_has_address(client, f"{prefix}/dyn/rel"),
        "makeup_db": _state_has_address(client, f"{prefix}/dyn/gain"),
        "mix_pct": _state_has_address(client, f"{prefix}/dyn/mix"),
    }
    main_send_fields = {
        "on": _state_has_address(client, f"{prefix}/main/1/on"),
        "level_db": _state_has_address(client, f"{prefix}/main/1/lvl"),
        "pre": _state_has_address(client, f"{prefix}/main/1/pre"),
    }
    assignment_fields = {
        "tags": _state_has_address(client, f"{prefix}/tags"),
        "dca_assignments": bool(_parse_cached_dca_assignments(client, tags)),
    }
    sections = {
        "identity": _console_surface_section_coverage(identity_fields),
        "basic_controls": _console_surface_section_coverage(basic_fields),
        "eq": eq["read_coverage"],
        "dynamics": _console_surface_section_coverage(dynamics_fields),
        "main_send": _console_surface_section_coverage(main_send_fields),
        "assignments": _console_surface_section_coverage(assignment_fields),
    }
    return {
        "bus": bus,
        "name": str(name).strip() if isinstance(name, str) and name.strip() else f"Bus {bus}",
        "fader_db": _cached_numeric(client, f"{prefix}/fdr"),
        "mute": _cached_int(client, f"{prefix}/mute"),
        "pan": _cached_numeric(client, f"{prefix}/pan"),
        "width": _cached_numeric(client, f"{prefix}/wid"),
        "mono": _cached_int(client, f"{prefix}/busmono"),
        "eq": eq,
        "dynamics": {
            "enabled": _cached_int(client, f"{prefix}/dyn/on"),
            "model": _cached_raw(client, f"{prefix}/dyn/mdl"),
            "threshold_db": _cached_numeric(client, f"{prefix}/dyn/thr37"),
            "ratio": _cached_raw(client, f"{prefix}/dyn/ratio"),
            "knee": _cached_numeric(client, f"{prefix}/dyn/knee"),
            "detector": _cached_raw(client, f"{prefix}/dyn/det"),
            "attack_ms": _cached_numeric(client, f"{prefix}/dyn/att"),
            "release_ms": _cached_numeric(client, f"{prefix}/dyn/rel"),
            "makeup_db": _cached_numeric(client, f"{prefix}/dyn/gain"),
            "mix_pct": _cached_numeric(client, f"{prefix}/dyn/mix"),
            "read_coverage": sections["dynamics"],
        },
        "main_send": {
            "on": _cached_int(client, f"{prefix}/main/1/on"),
            "level_db": _cached_numeric(client, f"{prefix}/main/1/lvl"),
            "pre": _cached_int(client, f"{prefix}/main/1/pre"),
            "read_coverage": sections["main_send"],
        },
        "tags": tags,
        "dca_assignments": _parse_cached_dca_assignments(client, tags),
        "read_coverage": _section_status_payload(sections),
    }


def _main_surface_row(client: Any, main: int) -> Dict[str, Any]:
    prefix = f"/main/{main}"
    eq = _console_eq_section(client, prefix, 8)
    basic_fields = {
        "fader_db": _state_has_address(client, f"{prefix}/fdr"),
        "mute": _state_has_address(client, f"{prefix}/mute"),
        "pan": _state_has_address(client, f"{prefix}/pan"),
    }
    sections = {
        "basic_controls": _console_surface_section_coverage(basic_fields),
        "eq": eq["read_coverage"],
    }
    return {
        "main": main,
        "name": f"Main {main}",
        "fader_db": _cached_numeric(client, f"{prefix}/fdr"),
        "mute": _cached_int(client, f"{prefix}/mute"),
        "pan": _cached_numeric(client, f"{prefix}/pan"),
        "eq": eq,
        "read_coverage": _section_status_payload(sections),
    }


def _dca_surface_row(client: Any, dca: int) -> Dict[str, Any]:
    prefix = f"/dca/{dca}"
    name = _cached_raw(client, f"{prefix}/name")
    identity_fields = {
        "name": _state_has_address(client, f"{prefix}/name"),
        "color": _state_has_address(client, f"{prefix}/col"),
        "icon": _state_has_address(client, f"{prefix}/icon"),
    }
    basic_fields = {
        "fader_db": _state_has_address(client, f"{prefix}/fdr"),
        "mute": _state_has_address(client, f"{prefix}/mute"),
        "monitor": _state_has_address(client, f"{prefix}/mon"),
    }
    sections = {
        "identity": _console_surface_section_coverage(identity_fields),
        "basic_controls": _console_surface_section_coverage(basic_fields),
    }
    return {
        "dca": dca,
        "name": str(name).strip() if isinstance(name, str) and name.strip() else f"DCA {dca}",
        "fader_db": _cached_numeric(client, f"{prefix}/fdr"),
        "mute": _cached_int(client, f"{prefix}/mute"),
        "color": _cached_raw(client, f"{prefix}/col"),
        "icon": _cached_raw(client, f"{prefix}/icon"),
        "monitor": _cached_int(client, f"{prefix}/mon"),
        "read_coverage": _section_status_payload(sections),
    }


def _console_surface_coverage_summary(groups: Mapping[str, Iterable[Mapping[str, Any]]]) -> Dict[str, Any]:
    status_counts: Dict[str, int] = {"complete": 0, "partial": 0, "missing": 0}
    surface_status_counts: Dict[str, Dict[str, int]] = {}
    section_status_counts: Dict[str, Dict[str, Dict[str, int]]] = {}
    for surface_name, rows_iterable in groups.items():
        rows = tuple(rows_iterable)
        surface_status_counts[surface_name] = {"complete": 0, "partial": 0, "missing": 0}
        section_status_counts[surface_name] = {}
        for row in rows:
            coverage = row.get("read_coverage") if isinstance(row, dict) else {}
            status = str(coverage.get("status") or "missing") if isinstance(coverage, dict) else "missing"
            status_counts[status] = status_counts.get(status, 0) + 1
            surface_status_counts[surface_name][status] = surface_status_counts[surface_name].get(status, 0) + 1
            sections = coverage.get("sections", {}) if isinstance(coverage, dict) else {}
            if not isinstance(sections, dict):
                continue
            for section_name, section in sections.items():
                section_status_counts[surface_name].setdefault(section_name, {"complete": 0, "partial": 0, "missing": 0})
                section_status = (
                    str(section.get("status") or "missing")
                    if isinstance(section, dict)
                    else "missing"
                )
                section_status_counts[surface_name][section_name][section_status] = (
                    section_status_counts[surface_name][section_name].get(section_status, 0) + 1
                )

    statuses = [
        status
        for status, count in status_counts.items()
        for _ in range(count)
    ]
    return {
        "status": _overall_read_coverage_status(statuses),
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "surface_status_counts": surface_status_counts,
        "section_status_counts": section_status_counts,
        "row_status_counts": status_counts,
    }


def _bus_main_dca_cache_addresses(
    *,
    bus_count: int,
    main_count: int,
    dca_count: int,
) -> List[str]:
    addresses: List[str] = []
    for bus in range(1, bus_count + 1):
        prefix = f"/bus/{bus}"
        addresses.extend([
            f"{prefix}/$name",
            f"{prefix}/name",
            f"{prefix}/tags",
            f"{prefix}/fdr",
            f"{prefix}/mute",
            f"{prefix}/pan",
            f"{prefix}/wid",
            f"{prefix}/busmono",
            f"{prefix}/eq/on",
        ])
        for band in range(1, 9):
            addresses.extend([
                f"{prefix}/eq/{band}f",
                f"{prefix}/eq/{band}g",
                f"{prefix}/eq/{band}q",
            ])
        addresses.extend([
            f"{prefix}/dyn/on",
            f"{prefix}/dyn/mdl",
            f"{prefix}/dyn/thr37",
            f"{prefix}/dyn/ratio",
            f"{prefix}/dyn/knee",
            f"{prefix}/dyn/det",
            f"{prefix}/dyn/att",
            f"{prefix}/dyn/rel",
            f"{prefix}/dyn/gain",
            f"{prefix}/dyn/mix",
            f"{prefix}/main/1/on",
            f"{prefix}/main/1/lvl",
            f"{prefix}/main/1/pre",
        ])
    for main in range(1, main_count + 1):
        prefix = f"/main/{main}"
        addresses.extend([
            f"{prefix}/fdr",
            f"{prefix}/mute",
            f"{prefix}/pan",
            f"{prefix}/eq/on",
        ])
        for band in range(1, 9):
            addresses.extend([
                f"{prefix}/eq/{band}f",
                f"{prefix}/eq/{band}g",
                f"{prefix}/eq/{band}q",
            ])
    for dca in range(1, dca_count + 1):
        prefix = f"/dca/{dca}"
        addresses.extend([
            f"{prefix}/name",
            f"{prefix}/col",
            f"{prefix}/icon",
            f"{prefix}/fdr",
            f"{prefix}/mute",
            f"{prefix}/mon",
        ])
    return addresses


def build_bus_main_dca_snapshot(
    server: Any,
    *,
    bus_count: int = DEFAULT_BUS_COUNT,
    main_count: int = DEFAULT_MAIN_COUNT,
    dca_count: int = DEFAULT_DCA_COUNT,
) -> Dict[str, Any]:
    """Return cached bus, main, and DCA console state without querying hardware."""

    client = getattr(server, "mixer_client", None)
    bus_count = _bounded_int_value(bus_count, default=DEFAULT_BUS_COUNT, minimum=1, maximum=32)
    main_count = _bounded_int_value(main_count, default=DEFAULT_MAIN_COUNT, minimum=1, maximum=8)
    dca_count = _bounded_int_value(dca_count, default=DEFAULT_DCA_COUNT, minimum=1, maximum=32)
    buses = [_bus_surface_row(client, bus) for bus in range(1, bus_count + 1)]
    mains = [_main_surface_row(client, main) for main in range(1, main_count + 1)]
    dcas = [_dca_surface_row(client, dca) for dca in range(1, dca_count + 1)]
    read_coverage = _console_surface_coverage_summary({
        "buses": buses,
        "mains": mains,
        "dcas": dcas,
    })
    cache_freshness = _cache_freshness_for_addresses(
        client,
        _bus_main_dca_cache_addresses(
            bus_count=bus_count,
            main_count=main_count,
            dca_count=dca_count,
        ),
    )
    return {
        "type": "bus_main_dca_snapshot",
        "generated_at": _utc_now(),
        "source": "runtime_cached_state",
        "available": True,
        "connection": _connection_summary(server),
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "bus_count": len(buses),
            "main_count": len(mains),
            "dca_count": len(dcas),
            "read_coverage_status": read_coverage["status"],
            "cache_freshness_status": cache_freshness["status"],
            "live_apply_allowed": False,
            "claim": "cached bus/main/DCA read coverage is exposed but not complete autonomous proof",
        },
        "read_coverage": read_coverage,
        "cache_freshness": cache_freshness,
        "buses": buses,
        "mains": mains,
        "dcas": dcas,
    }


def _console_insert_snapshot(client: Any, *, target_type: str, target: int, position: str) -> Dict[str, Any]:
    target_prefix = {
        "matrix": "mtx",
        "mtx": "mtx",
        "bus": "bus",
        "main": "main",
        "channel": "ch",
        "ch": "ch",
    }.get(str(target_type).lower(), str(target_type).lower())
    position_key = str(position).lower()
    insert_suffix = "postins" if position_key == "post" else "preins"
    base = f"/{target_prefix}/{int(target)}/{insert_suffix}"
    fields = {
        "on": _state_has_address(client, f"{base}/on"),
        "slot": _state_has_address(client, f"{base}/ins"),
        "status": _state_has_address(client, f"{base}/$stat"),
    }
    if position_key == "post":
        fields["mode"] = _state_has_address(client, f"{base}/mode")
        fields["weight"] = _state_has_address(client, f"{base}/w")
    return {
        "position": position_key,
        "on": _cached_int(client, f"{base}/on"),
        "slot": _cached_raw(client, f"{base}/ins"),
        "status": _cached_raw(client, f"{base}/$stat"),
        "mode": _cached_raw(client, f"{base}/mode") if position_key == "post" else None,
        "weight": _cached_numeric(client, f"{base}/w") if position_key == "post" else None,
        "read_coverage": _console_surface_section_coverage(fields),
    }


def _matrix_surface_row(client: Any, matrix: int) -> Dict[str, Any]:
    prefix = f"/mtx/{matrix}"
    eq = _console_eq_section(client, prefix, 8)
    pre_insert = _console_insert_snapshot(client, target_type="matrix", target=matrix, position="pre")
    post_insert = _console_insert_snapshot(client, target_type="matrix", target=matrix, position="post")
    basic_fields = {
        "fader_db": _state_has_address(client, f"{prefix}/fdr"),
        "mute": _state_has_address(client, f"{prefix}/mute"),
        "pan": _state_has_address(client, f"{prefix}/pan"),
    }
    insert_fields = {
        "pre_insert": pre_insert["read_coverage"]["status"] == "complete",
        "post_insert": post_insert["read_coverage"]["status"] == "complete",
    }
    dynamics_addresses = {
        "enabled": f"{prefix}/dyn/on",
        "model": f"{prefix}/dyn/mdl",
        "threshold_db": f"{prefix}/dyn/thr37",
        "ratio": f"{prefix}/dyn/ratio",
        "knee": f"{prefix}/dyn/knee",
        "detector": f"{prefix}/dyn/det",
        "attack_ms": f"{prefix}/dyn/att",
        "release_ms": f"{prefix}/dyn/rel",
        "makeup_db": f"{prefix}/dyn/gain",
        "mix_pct": f"{prefix}/dyn/mix",
    }
    dynamics_fields = {
        field: _state_has_address(client, address)
        for field, address in dynamics_addresses.items()
    }
    dynamics_coverage = _console_surface_section_coverage(dynamics_fields)
    dynamics_coverage["cache_freshness"] = _cache_freshness_for_field_addresses(
        client,
        dynamics_addresses,
    )
    sections = {
        "basic_controls": _console_surface_section_coverage(basic_fields),
        "eq": eq["read_coverage"],
        "dynamics": dynamics_coverage,
        "inserts": _console_surface_section_coverage(insert_fields),
    }
    return {
        "matrix": matrix,
        "name": f"Matrix {matrix}",
        "fader_db": _cached_numeric(client, f"{prefix}/fdr"),
        "mute": _cached_int(client, f"{prefix}/mute"),
        "pan": _cached_numeric(client, f"{prefix}/pan"),
        "eq": eq,
        "dynamics": {
            "enabled": _cached_int(client, dynamics_addresses["enabled"]),
            "model": _cached_raw(client, dynamics_addresses["model"]),
            "threshold_db": _cached_numeric(client, dynamics_addresses["threshold_db"]),
            "ratio": _cached_raw(client, dynamics_addresses["ratio"]),
            "knee": _cached_numeric(client, dynamics_addresses["knee"]),
            "detector": _cached_raw(client, dynamics_addresses["detector"]),
            "attack_ms": _cached_numeric(client, dynamics_addresses["attack_ms"]),
            "release_ms": _cached_numeric(client, dynamics_addresses["release_ms"]),
            "makeup_db": _cached_numeric(client, dynamics_addresses["makeup_db"]),
            "mix_pct": _cached_numeric(client, dynamics_addresses["mix_pct"]),
            "read_coverage": sections["dynamics"],
        },
        "inserts": {
            "pre_insert": pre_insert,
            "post_insert": post_insert,
            "read_coverage": sections["inserts"],
        },
        "read_coverage": _section_status_payload(sections),
    }


def _matrix_cache_addresses(matrix_count: int) -> List[str]:
    addresses: List[str] = []
    for matrix in range(1, matrix_count + 1):
        prefix = f"/mtx/{matrix}"
        addresses.extend([
            f"{prefix}/fdr",
            f"{prefix}/mute",
            f"{prefix}/pan",
            f"{prefix}/eq/on",
        ])
        for band in range(1, 9):
            addresses.extend([
                f"{prefix}/eq/{band}f",
                f"{prefix}/eq/{band}g",
                f"{prefix}/eq/{band}q",
            ])
        addresses.extend([
            f"{prefix}/dyn/on",
            f"{prefix}/dyn/mdl",
            f"{prefix}/dyn/thr37",
            f"{prefix}/dyn/ratio",
            f"{prefix}/dyn/knee",
            f"{prefix}/dyn/det",
            f"{prefix}/dyn/att",
            f"{prefix}/dyn/rel",
            f"{prefix}/dyn/gain",
            f"{prefix}/dyn/mix",
            f"{prefix}/preins/on",
            f"{prefix}/preins/ins",
            f"{prefix}/preins/$stat",
            f"{prefix}/postins/on",
            f"{prefix}/postins/ins",
            f"{prefix}/postins/$stat",
            f"{prefix}/postins/mode",
            f"{prefix}/postins/w",
        ])
    return addresses


def build_matrix_snapshot(
    server: Any,
    *,
    matrix_count: int = DEFAULT_MATRIX_COUNT,
) -> Dict[str, Any]:
    """Return cached matrix state without querying or mutating a console."""

    client = getattr(server, "mixer_client", None)
    matrix_count = _bounded_int_value(matrix_count, default=DEFAULT_MATRIX_COUNT, minimum=1, maximum=16)
    matrices = [_matrix_surface_row(client, matrix) for matrix in range(1, matrix_count + 1)]
    read_coverage = _console_surface_coverage_summary({"matrices": matrices})
    cache_freshness = _cache_freshness_for_addresses(
        client,
        _matrix_cache_addresses(matrix_count),
    )
    return {
        "type": "matrix_snapshot",
        "generated_at": _utc_now(),
        "source": "runtime_cached_state",
        "available": True,
        "connection": _connection_summary(server),
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "matrix_count": len(matrices),
            "read_coverage_status": read_coverage["status"],
            "cache_freshness_status": cache_freshness["status"],
            "live_apply_allowed": False,
            "claim": "cached matrix read coverage is exposed but not complete autonomous proof",
        },
        "read_coverage": read_coverage,
        "cache_freshness": cache_freshness,
        "matrices": matrices,
    }


def _fx_parameter_family(model: Any) -> Optional[str]:
    text = str(model or "").strip().upper()
    if not text:
        return None
    for family, tokens in FX_MODEL_FAMILY_TOKENS.items():
        if any(token in text for token in tokens):
            return family
    return None


def _fx_parameter_schema_hint(model: Any, parameter_index: int) -> Dict[str, Any]:
    family = _fx_parameter_family(model)
    mapped_schema = (
        FX_PARAMETER_SCHEMA_BY_FAMILY.get(family, {}).get(parameter_index)
        if family
        else None
    )
    if mapped_schema:
        status = "mapped"
        semantic_name = str(mapped_schema["semantic_name"])
        unit = str(mapped_schema["unit"])
        safe_min = mapped_schema.get("safe_min")
        safe_max = mapped_schema.get("safe_max")
    else:
        status = "generic_numeric"
        semantic_name = f"parameter_{parameter_index}"
        unit = "raw"
        safe_min = None
        safe_max = None
    return {
        "schema_status": status,
        "schema_family": family or "unknown",
        "schema_source": "automixer_generic_fx_parameter_hints",
        "semantic_name": semantic_name,
        "unit": unit,
        "safe_range": {
            "min": safe_min,
            "max": safe_max,
        },
    }


def _cached_fx_parameter_rows(
    client: Any,
    fx_slot: int,
    parameter_count: int,
    *,
    model: Any,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for parameter in range(1, parameter_count + 1):
        address = f"/fx/{fx_slot}/{parameter}"
        if not _state_has_address(client, address):
            continue
        schema_hint = _fx_parameter_schema_hint(model, parameter)
        rows.append({
            "parameter_index": parameter,
            "value": _cached_raw(client, address),
            **schema_hint,
            "read_coverage": _read_coverage_section({"value": True}, "mixer_state_cache"),
        })
    return rows


def _fx_slot_parameter_schema_summary(model: Any, parameters: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = tuple(parameters)
    status_counts = {
        "mapped": sum(1 for row in rows if row.get("schema_status") == "mapped"),
        "generic_numeric": sum(1 for row in rows if row.get("schema_status") == "generic_numeric"),
    }
    if not rows:
        status = "missing"
    elif status_counts["generic_numeric"]:
        status = "partial"
    else:
        status = "mapped"
    return {
        "status": status,
        "schema_family": _fx_parameter_family(model) or "unknown",
        "schema_source": "automixer_generic_fx_parameter_hints",
        "mapped_parameter_count": status_counts["mapped"],
        "generic_parameter_count": status_counts["generic_numeric"],
        "cached_parameter_count": len(rows),
        "physical_model_schema_verified": False,
        "live_write_authority": False,
    }


def _fx_slot_row(client: Any, fx_slot: int, *, parameter_count: int) -> Dict[str, Any]:
    model = _cached_raw(client, f"/fx/{fx_slot}/mdl")
    parameters = _cached_fx_parameter_rows(client, fx_slot, parameter_count, model=model)
    parameter_schema = _fx_slot_parameter_schema_summary(model, parameters)
    fields = {
        "model": _state_has_address(client, f"/fx/{fx_slot}/mdl"),
        "on": _state_has_address(client, f"/fx/{fx_slot}/on"),
        "mix": _state_has_address(client, f"/fx/{fx_slot}/fxmix"),
        "source": _state_has_address(client, f"/fx/{fx_slot}/$esrc"),
        "mode": _state_has_address(client, f"/fx/{fx_slot}/$emode"),
        "assigned_channel": _state_has_address(client, f"/fx/{fx_slot}/$a_chn"),
        "assigned_position": _state_has_address(client, f"/fx/{fx_slot}/$a_pos"),
        "parameters": bool(parameters),
    }
    on = _cached_int(client, f"/fx/{fx_slot}/on")
    mix = _cached_numeric(client, f"/fx/{fx_slot}/fxmix")
    return {
        "slot": f"FX{fx_slot}",
        "slot_number": fx_slot,
        "model": model,
        "on": on,
        "mix": mix,
        "source": _cached_raw(client, f"/fx/{fx_slot}/$esrc"),
        "mode": _cached_raw(client, f"/fx/{fx_slot}/$emode"),
        "assigned_channel": _cached_int(client, f"/fx/{fx_slot}/$a_chn"),
        "assigned_position": _cached_raw(client, f"/fx/{fx_slot}/$a_pos"),
        "active": bool(on == 1 and mix is not None and mix > 0),
        "cached_parameter_count": len(parameters),
        "parameter_schema": parameter_schema,
        "parameters": parameters,
        "fx_model_change_policy": "blocked_manual_only",
        "fx_bypass_policy": "blocked_manual_only",
        "fx_mix_policy": "blocked_manual_only",
        "destructive_change_marker": "fx_slot_change_would_modify_processing_chain",
        "read_coverage": _console_surface_section_coverage(fields),
    }


def _active_insert_slot(slot: Any) -> Optional[str]:
    text = str(slot or "").strip().upper()
    if not text or text in {"NONE", "OFF", "0"}:
        return None
    return text


def _fx_insert_row(client: Any, *, target_type: str, target: int, position: str) -> Dict[str, Any]:
    insert_state = _console_insert_snapshot(
        client,
        target_type=target_type,
        target=target,
        position=position,
    )
    slot = _active_insert_slot(insert_state.get("slot"))
    row = {
        "target_type": target_type,
        "target": target,
        "position": position,
        "on": insert_state.get("on"),
        "slot": insert_state.get("slot"),
        "status": insert_state.get("status"),
        "mode": insert_state.get("mode"),
        "weight": insert_state.get("weight"),
        "linked_fx_slot": slot,
        "active": bool(insert_state.get("on") == 1 and slot),
        "insert_assignment_policy": "blocked_manual_only",
        "destructive_change_marker": "insert_assignment_would_repatch_processing_chain",
        "read_coverage": insert_state.get("read_coverage", _console_surface_section_coverage({})),
    }
    return row


def _fx_insert_coverage_summary(groups: Mapping[str, Iterable[Mapping[str, Any]]]) -> Dict[str, Any]:
    status_counts: Dict[str, int] = {"complete": 0, "partial": 0, "missing": 0}
    surface_status_counts: Dict[str, Dict[str, int]] = {}
    for surface_name, rows_iterable in groups.items():
        rows = tuple(rows_iterable)
        surface_status_counts[surface_name] = {"complete": 0, "partial": 0, "missing": 0}
        for row in rows:
            coverage = row.get("read_coverage") if isinstance(row, dict) else {}
            status = str(coverage.get("status") or "missing") if isinstance(coverage, dict) else "missing"
            status_counts[status] = status_counts.get(status, 0) + 1
            surface_status_counts[surface_name][status] = surface_status_counts[surface_name].get(status, 0) + 1
    statuses = [
        status
        for status, count in status_counts.items()
        for _ in range(count)
    ]
    return {
        "status": _overall_read_coverage_status(statuses),
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "surface_status_counts": surface_status_counts,
        "row_status_counts": status_counts,
    }


def _fx_parameter_schema_summary(fx_slots: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = tuple(fx_slots)
    mapped = 0
    generic = 0
    missing = 0
    for row in rows:
        schema = row.get("parameter_schema", {}) if isinstance(row, Mapping) else {}
        if not isinstance(schema, Mapping):
            missing += 1
            continue
        mapped += int(schema.get("mapped_parameter_count") or 0)
        generic += int(schema.get("generic_parameter_count") or 0)
        if schema.get("status") == "missing":
            missing += 1
    if mapped == 0 and generic == 0:
        status = "missing"
    elif generic:
        status = "partial"
    else:
        status = "mapped"
    return {
        "status": status,
        "mapped_parameter_count": mapped,
        "generic_parameter_count": generic,
        "slot_missing_schema_count": missing,
        "schema_source": "automixer_generic_fx_parameter_hints",
        "physical_model_schema_verified": False,
        "live_write_authority": False,
    }


def _insert_cache_addresses(target_type: str, target: int, position: str) -> List[str]:
    target_prefix = {
        "matrix": "mtx",
        "mtx": "mtx",
        "bus": "bus",
        "main": "main",
        "channel": "ch",
        "ch": "ch",
    }.get(str(target_type).lower(), str(target_type).lower())
    position_key = str(position).lower()
    insert_suffix = "postins" if position_key == "post" else "preins"
    base = f"/{target_prefix}/{int(target)}/{insert_suffix}"
    addresses = [
        f"{base}/on",
        f"{base}/ins",
        f"{base}/$stat",
    ]
    if position_key == "post":
        addresses.extend([f"{base}/mode", f"{base}/w"])
    return addresses


def _fx_inserts_cache_addresses(
    *,
    fx_slot_count: int,
    parameter_count: int,
    insert_targets: Iterable[Tuple[str, int]],
) -> List[str]:
    addresses: List[str] = []
    for fx_slot in range(1, fx_slot_count + 1):
        addresses.extend([
            f"/fx/{fx_slot}/mdl",
            f"/fx/{fx_slot}/on",
            f"/fx/{fx_slot}/fxmix",
            f"/fx/{fx_slot}/$esrc",
            f"/fx/{fx_slot}/$emode",
            f"/fx/{fx_slot}/$a_chn",
            f"/fx/{fx_slot}/$a_pos",
        ])
        for parameter in range(1, parameter_count + 1):
            addresses.append(f"/fx/{fx_slot}/{parameter}")
    for target_type, target in insert_targets:
        for position in ("pre", "post"):
            addresses.extend(_insert_cache_addresses(target_type, target, position))
    return addresses


def build_fx_inserts_snapshot(
    server: Any,
    *,
    fx_slot_count: int = DEFAULT_FX_SLOT_COUNT,
    total_channels: int = DEFAULT_CHANNEL_COUNT,
    bus_count: int = DEFAULT_BUS_COUNT,
    main_count: int = DEFAULT_MAIN_COUNT,
    matrix_count: int = DEFAULT_MATRIX_COUNT,
    parameter_count: int = DEFAULT_FX_PARAMETER_COUNT,
) -> Dict[str, Any]:
    """Return cached FX rack and insert state without querying or mutating hardware."""

    client = getattr(server, "mixer_client", None)
    fx_slot_count = _bounded_int_value(fx_slot_count, default=DEFAULT_FX_SLOT_COUNT, minimum=1, maximum=32)
    total_channels = _bounded_int_value(total_channels, default=DEFAULT_CHANNEL_COUNT, minimum=1, maximum=DEFAULT_DANTE_CHANNEL_COUNT)
    bus_count = _bounded_int_value(bus_count, default=DEFAULT_BUS_COUNT, minimum=1, maximum=32)
    main_count = _bounded_int_value(main_count, default=DEFAULT_MAIN_COUNT, minimum=1, maximum=8)
    matrix_count = _bounded_int_value(matrix_count, default=DEFAULT_MATRIX_COUNT, minimum=1, maximum=16)
    parameter_count = _bounded_int_value(parameter_count, default=DEFAULT_FX_PARAMETER_COUNT, minimum=1, maximum=64)

    fx_slots = [
        _fx_slot_row(client, slot, parameter_count=parameter_count)
        for slot in range(1, fx_slot_count + 1)
    ]
    insert_targets = [
        ("channel", target)
        for target in range(1, total_channels + 1)
    ] + [
        ("bus", target)
        for target in range(1, bus_count + 1)
    ] + [
        ("main", target)
        for target in range(1, main_count + 1)
    ] + [
        ("matrix", target)
        for target in range(1, matrix_count + 1)
    ]
    inserts = [
        _fx_insert_row(client, target_type=target_type, target=target, position=position)
        for target_type, target in insert_targets
        for position in ("pre", "post")
    ]
    read_coverage = _fx_insert_coverage_summary({
        "fx_slots": fx_slots,
        "inserts": inserts,
    })
    parameter_schema = _fx_parameter_schema_summary(fx_slots)
    cache_freshness = _cache_freshness_for_addresses(
        client,
        _fx_inserts_cache_addresses(
            fx_slot_count=fx_slot_count,
            parameter_count=parameter_count,
            insert_targets=insert_targets,
        ),
    )
    return {
        "type": "fx_inserts_snapshot",
        "generated_at": _utc_now(),
        "source": "runtime_cached_state",
        "available": True,
        "connection": _connection_summary(server),
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "full_autonomous_ready": False,
            "fx_apply_allowed": False,
            "insert_assignment_allowed": False,
        },
        "summary": {
            "fx_slot_count": len(fx_slots),
            "insert_count": len(inserts),
            "active_fx_slot_count": sum(1 for row in fx_slots if row.get("active")),
            "active_insert_count": sum(1 for row in inserts if row.get("active")),
            "cached_parameter_count": sum(int(row.get("cached_parameter_count") or 0) for row in fx_slots),
            "read_coverage_status": read_coverage["status"],
            "parameter_schema_status": parameter_schema["status"],
            "cache_freshness_status": cache_freshness["status"],
            "live_apply_allowed": False,
            "claim": "cached FX and insert read coverage is exposed but FX and insert apply remain blocked",
        },
        "read_coverage": read_coverage,
        "parameter_schema": parameter_schema,
        "cache_freshness": cache_freshness,
        "fx_slots": fx_slots,
        "inserts": inserts,
    }


def _dlive_midi_base_channel(client: Any) -> int:
    return _bounded_int_value(
        getattr(client, "midi_base_channel", 0),
        default=0,
        minimum=0,
        maximum=15,
    )


def _dlive_cached_name(client: Any, channel: int) -> Optional[str]:
    channel_names = getattr(client, "_channel_names", None)
    if isinstance(channel_names, Mapping):
        raw_name = channel_names.get(channel)
        if isinstance(raw_name, str) and raw_name.strip():
            return raw_name.strip()
    raw_state_name = _cached_raw(client, f"name/{channel}")
    if isinstance(raw_state_name, str) and raw_state_name.strip():
        return raw_state_name.strip()
    return None


def _dlive_cached_mute(client: Any, channel: int, midi_base_channel: int) -> Optional[bool]:
    cached = _cached_raw(client, f"mute/{channel}")
    if isinstance(cached, bool):
        return cached
    if cached is not None:
        try:
            return bool(int(cached))
        except (TypeError, ValueError):
            text = str(cached).strip().lower()
            if text in {"true", "on", "muted", "yes"}:
                return True
            if text in {"false", "off", "unmuted", "no"}:
                return False
    note_velocity = _cached_int(client, f"note/{midi_base_channel}/{channel - 1}")
    if note_velocity is not None:
        return note_velocity >= 0x40
    return None


def _dlive_channel_row(client: Any, channel: int, midi_base_channel: int) -> Dict[str, Any]:
    cached_name = _dlive_cached_name(client, channel)
    fader_raw = _cached_int(client, f"fader/{channel}")
    note_velocity = _cached_int(client, f"note/{midi_base_channel}/{channel - 1}")
    mute = _dlive_cached_mute(client, channel, midi_base_channel)
    sections = {
        "identity": _read_coverage_section(
            {"cached_name": bool(cached_name)},
            "midi_tcp_state" if cached_name else "unavailable",
        ),
        "fader": _read_coverage_section(
            {
                "raw_14bit_fader": fader_raw is not None,
                "fader_db_conversion": fader_raw is not None,
            },
            "midi_tcp_state" if fader_raw is not None else "unavailable",
        ),
        "mute": _read_coverage_section(
            {
                "tracked_mute": mute is not None,
                "raw_note_velocity": note_velocity is not None,
            },
            "midi_tcp_state" if mute is not None or note_velocity is not None else "unavailable",
        ),
    }
    read_coverage = _section_status_payload(sections)
    read_coverage["source_kinds"] = ["midi_tcp_state"] if read_coverage["status"] != "missing" else []
    return {
        "channel": channel,
        "name": cached_name or f"dLive Ch {channel}",
        "name_source": "midi_tcp_state" if cached_name else "fallback_label",
        "fader_raw_14bit": fader_raw,
        "fader_db": _dlive_fader_raw_to_db(fader_raw),
        "mute": mute,
        "raw_note_velocity": note_velocity,
        "midi_base_channel": midi_base_channel,
        "read_coverage": read_coverage,
        "fader_change_policy": "blocked_manual_only",
        "mute_change_policy": "blocked_manual_only",
    }


def _dlive_state_key_counts(client: Any) -> Dict[str, int]:
    state = getattr(client, "state", None)
    if not isinstance(state, dict):
        return {
            "raw_state_keys": 0,
            "cached_names": 0,
            "cached_faders": 0,
            "cached_mutes": 0,
            "raw_note_events": 0,
            "raw_cc_events": 0,
            "raw_pitchbend_events": 0,
        }
    return {
        "raw_state_keys": len(state),
        "cached_names": sum(1 for key in state if str(key).startswith("name/")),
        "cached_faders": sum(1 for key in state if str(key).startswith("fader/")),
        "cached_mutes": sum(1 for key in state if str(key).startswith("mute/")),
        "raw_note_events": sum(1 for key in state if str(key).startswith("note/")),
        "raw_cc_events": sum(1 for key in state if str(key).startswith("cc/")),
        "raw_pitchbend_events": sum(1 for key in state if str(key).startswith("pitchbend/")),
    }


def _dlive_state_coverage_summary(channels: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    channel_rows = tuple(channels)
    channel_status_counts: Dict[str, int] = {"complete": 0, "partial": 0, "missing": 0}
    section_status_counts: Dict[str, Dict[str, int]] = {}
    for row in channel_rows:
        coverage = row.get("read_coverage") if isinstance(row, dict) else {}
        status = str(coverage.get("status") or "missing") if isinstance(coverage, dict) else "missing"
        channel_status_counts[status] = channel_status_counts.get(status, 0) + 1
        sections = coverage.get("sections", {}) if isinstance(coverage, dict) else {}
        if not isinstance(sections, dict):
            continue
        for section_name, section in sections.items():
            section_status_counts.setdefault(section_name, {"complete": 0, "partial": 0, "missing": 0})
            section_status = (
                str(section.get("status") or "missing")
                if isinstance(section, dict)
                else "missing"
            )
            section_status_counts[section_name][section_status] = (
                section_status_counts[section_name].get(section_status, 0) + 1
            )
    statuses = [
        status
        for status, count in channel_status_counts.items()
        for _ in range(count)
    ]
    return {
        "status": _overall_read_coverage_status(statuses),
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "channel_status_counts": channel_status_counts,
        "section_status_counts": section_status_counts,
        "channels_with_cached_name": section_status_counts.get("identity", {}).get("complete", 0),
        "channels_with_cached_fader": section_status_counts.get("fader", {}).get("complete", 0),
        "channels_with_cached_mute": sum(
            1
            for row in channel_rows
            if (row.get("read_coverage") or {}).get("sections", {}).get("mute", {}).get("status") != "missing"
        ),
    }


def _dlive_cache_addresses(total_channels: int, midi_base_channel: int) -> List[str]:
    addresses: List[str] = []
    for channel in range(1, total_channels + 1):
        addresses.extend([
            f"name/{channel}",
            f"fader/{channel}",
            f"mute/{channel}",
            f"note/{midi_base_channel}/{channel - 1}",
        ])
    return addresses


def build_dlive_state_snapshot(
    server: Any,
    *,
    total_channels: int = DEFAULT_DLIVE_CHANNEL_COUNT,
) -> Dict[str, Any]:
    """Return cached dLive MIDI/TCP state without querying or mutating hardware."""

    client = getattr(server, "mixer_client", None)
    total_channels = _bounded_int_value(
        total_channels,
        default=DEFAULT_DLIVE_CHANNEL_COUNT,
        minimum=1,
        maximum=MAX_DLIVE_CHANNEL_COUNT,
    )
    midi_base_channel = _dlive_midi_base_channel(client)
    channels = [
        _dlive_channel_row(client, channel, midi_base_channel)
        for channel in range(1, total_channels + 1)
    ]
    read_coverage = _dlive_state_coverage_summary(channels)
    state_key_counts = _dlive_state_key_counts(client)
    cache_freshness = _cache_freshness_for_addresses(
        client,
        _dlive_cache_addresses(total_channels, midi_base_channel),
    )
    return {
        "type": "dlive_state_snapshot",
        "generated_at": _utc_now(),
        "source": "midi_tcp_cached_state",
        "available": True,
        "connection": _connection_summary(server),
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "full_autonomous_ready": False,
            "dlive_apply_allowed": False,
            "midi_send_allowed": False,
        },
        "summary": {
            "channel_count": len(channels),
            "midi_base_channel": midi_base_channel,
            "read_coverage_status": read_coverage["status"],
            "channels_with_cached_name": read_coverage["channels_with_cached_name"],
            "channels_with_cached_fader": read_coverage["channels_with_cached_fader"],
            "channels_with_cached_mute": read_coverage["channels_with_cached_mute"],
            "raw_state_key_count": state_key_counts["raw_state_keys"],
            "cache_freshness_status": cache_freshness["status"],
            "wing_parity_status": "partial",
            "live_apply_allowed": False,
            "claim": "cached dLive MIDI/TCP state is exposed but WING parity and autonomous proof remain incomplete",
        },
        "read_coverage": read_coverage,
        "cache_freshness": cache_freshness,
        "state_key_counts": state_key_counts,
        "known_gaps": [
            "dLive EQ, dynamics, routing, scenes, and FX are not normalized into this cached DTO",
            "preamp gain pitchbend cache is transport-channel scoped, not safely per-input-channel proof",
            "physical readback fixtures and stale-cache timestamps are still required before parity claims",
        ],
        "channels": channels,
    }


def _cached_send_active(on: Any, level_db: Any) -> bool:
    try:
        enabled = int(on) == 1
    except (TypeError, ValueError):
        enabled = str(on).strip().lower() in {"true", "on", "yes"}
    try:
        level = float(level_db)
    except (TypeError, ValueError):
        level = -144.0
    return enabled and level > -143.0


def _channel_bus_send_row(client: Any, channel: int, send: int) -> Dict[str, Any]:
    base = f"/ch/{channel}/send/{send}"
    field_addresses = {
        "on": f"{base}/on",
        "level_db": f"{base}/lvl",
        "pre_on": f"{base}/pon",
        "mode": f"{base}/mode",
        "plink": f"{base}/plink",
        "pan": f"{base}/pan",
    }
    on = _cached_int(client, f"{base}/on")
    level_db = _cached_numeric(client, f"{base}/lvl")
    return {
        "channel": channel,
        "send": send,
        "target_bus": send,
        "on": on,
        "level_db": level_db,
        "pre_on": _cached_int(client, f"{base}/pon"),
        "mode": _cached_raw(client, f"{base}/mode"),
        "plink": _cached_int(client, f"{base}/plink"),
        "pan": _cached_numeric(client, f"{base}/pan"),
        "active": _cached_send_active(on, level_db),
        "read_coverage": _console_surface_section_coverage_for_addresses(client, field_addresses),
    }


def _channel_main_send_row(client: Any, channel: int, main: int) -> Dict[str, Any]:
    base = f"/ch/{channel}/main/{main}"
    field_addresses = {
        "on": f"{base}/on",
        "level_db": f"{base}/lvl",
        "pre": f"{base}/pre",
    }
    on = _cached_int(client, f"{base}/on")
    level_db = _cached_numeric(client, f"{base}/lvl")
    return {
        "channel": channel,
        "main": main,
        "on": on,
        "level_db": level_db,
        "pre": _cached_int(client, f"{base}/pre"),
        "active": _cached_send_active(on, level_db),
        "read_coverage": _console_surface_section_coverage_for_addresses(client, field_addresses),
    }


def _channel_matrix_send_row(client: Any, channel: int, matrix: int) -> Dict[str, Any]:
    send_key = f"MX{matrix}"
    base = f"/ch/{channel}/send/{send_key}"
    field_addresses = {
        "on": f"{base}/on",
        "level_db": f"{base}/lvl",
    }
    on = _cached_int(client, f"{base}/on")
    level_db = _cached_numeric(client, f"{base}/lvl")
    return {
        "channel": channel,
        "send": send_key,
        "target_matrix": matrix,
        "on": on,
        "level_db": level_db,
        "active": _cached_send_active(on, level_db),
        "destructive_change_marker": "matrix_send_write_would_change_console_routing",
        "read_coverage": _console_surface_section_coverage_for_addresses(client, field_addresses),
    }


def _output_route_row(client: Any, output_group: str, output_number: int) -> Dict[str, Any]:
    base = f"/io/out/{output_group}/{output_number}"
    field_addresses = {
        "source_group": f"{base}/grp",
        "source_channel": f"{base}/in",
    }
    source_group = _cached_raw(client, f"{base}/grp")
    source_channel = _cached_int(client, f"{base}/in")
    return {
        "output_group": output_group,
        "output_number": output_number,
        "source_group": source_group,
        "source_channel": source_channel,
        "destructive_change_marker": "output_routing_write_would_repatch_console",
        "read_coverage": _console_surface_section_coverage_for_addresses(client, field_addresses),
    }


def _static_dante_route_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    fields = {
        "start": row.get("start") is not None,
        "end": row.get("end") is not None,
        "role": bool(row.get("role")),
        "tap_point": bool(row.get("tap_point")),
        "label": bool(row.get("label_short") or row.get("label_full")),
    }
    return {
        "start": row.get("start"),
        "end": row.get("end"),
        "role": row.get("role"),
        "tap_point": row.get("tap_point"),
        "label_short": row.get("label_short"),
        "label_full": row.get("label_full"),
        "required": bool(row.get("required")),
        "wing_routing_hint": row.get("wing_routing_hint"),
        "used_by": row.get("used_by", []),
        "read_coverage": _read_coverage_section(fields, "static_config"),
    }


def _routing_send_coverage_summary(groups: Mapping[str, Iterable[Mapping[str, Any]]]) -> Dict[str, Any]:
    status_counts: Dict[str, int] = {"complete": 0, "partial": 0, "missing": 0}
    surface_status_counts: Dict[str, Dict[str, int]] = {}
    for surface_name, rows_iterable in groups.items():
        rows = tuple(rows_iterable)
        surface_status_counts[surface_name] = {"complete": 0, "partial": 0, "missing": 0}
        for row in rows:
            coverage = row.get("read_coverage") if isinstance(row, dict) else {}
            status = str(coverage.get("status") or "missing") if isinstance(coverage, dict) else "missing"
            status_counts[status] = status_counts.get(status, 0) + 1
            surface_status_counts[surface_name][status] = surface_status_counts[surface_name].get(status, 0) + 1
    statuses = [
        status
        for status, count in status_counts.items()
        for _ in range(count)
    ]
    return {
        "status": _overall_read_coverage_status(statuses),
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "surface_status_counts": surface_status_counts,
        "row_status_counts": status_counts,
    }


def _normalise_output_groups(output_groups: Any) -> List[str]:
    if isinstance(output_groups, str):
        candidates = [output_groups]
    elif isinstance(output_groups, Iterable):
        candidates = [str(item) for item in output_groups]
    else:
        candidates = list(DEFAULT_OUTPUT_GROUPS)
    groups = []
    for item in candidates:
        group = str(item).strip().upper()
        if group and group not in groups:
            groups.append(group)
    return groups or list(DEFAULT_OUTPUT_GROUPS)


def _routing_sends_cache_addresses(
    *,
    total_channels: int,
    send_count: int,
    main_count: int,
    matrix_send_count: int,
    output_groups: Iterable[str],
    output_count: int,
) -> List[str]:
    addresses: List[str] = []
    for channel in range(1, total_channels + 1):
        for send in range(1, send_count + 1):
            base = f"/ch/{channel}/send/{send}"
            addresses.extend([
                f"{base}/on",
                f"{base}/lvl",
                f"{base}/pon",
                f"{base}/mode",
                f"{base}/plink",
                f"{base}/pan",
            ])
        for main in range(1, main_count + 1):
            base = f"/ch/{channel}/main/{main}"
            addresses.extend([f"{base}/on", f"{base}/lvl", f"{base}/pre"])
        for matrix in range(1, matrix_send_count + 1):
            base = f"/ch/{channel}/send/MX{matrix}"
            addresses.extend([f"{base}/on", f"{base}/lvl"])
    for group in output_groups:
        for output_number in range(1, output_count + 1):
            base = f"/io/out/{group}/{output_number}"
            addresses.extend([f"{base}/grp", f"{base}/in"])
    return addresses


def build_routing_sends_snapshot(
    server: Any,
    *,
    total_channels: int = DEFAULT_CHANNEL_COUNT,
    send_count: int = DEFAULT_SEND_COUNT,
    main_count: int = 1,
    matrix_send_count: int = DEFAULT_MATRIX_COUNT,
    output_groups: Any = DEFAULT_OUTPUT_GROUPS,
    output_count: int = DEFAULT_OUTPUT_COUNT,
) -> Dict[str, Any]:
    """Return static routing intent and cached send/output-route state without querying hardware."""

    client = getattr(server, "mixer_client", None)
    total_channels = _bounded_int_value(total_channels, default=DEFAULT_CHANNEL_COUNT, minimum=1, maximum=DEFAULT_DANTE_CHANNEL_COUNT)
    send_count = _bounded_int_value(send_count, default=DEFAULT_SEND_COUNT, minimum=1, maximum=32)
    main_count = _bounded_int_value(main_count, default=1, minimum=1, maximum=4)
    matrix_send_count = _bounded_int_value(matrix_send_count, default=DEFAULT_MATRIX_COUNT, minimum=1, maximum=16)
    output_count = _bounded_int_value(output_count, default=DEFAULT_OUTPUT_COUNT, minimum=1, maximum=128)
    groups = _normalise_output_groups(output_groups)
    static_dante_routes = [
        _static_dante_route_row(row)
        for row in get_routing_as_dict(DEFAULT_DANTE_CHANNEL_COUNT)
    ]
    channel_bus_sends = [
        _channel_bus_send_row(client, channel, send)
        for channel in range(1, total_channels + 1)
        for send in range(1, send_count + 1)
    ]
    channel_main_sends = [
        _channel_main_send_row(client, channel, main)
        for channel in range(1, total_channels + 1)
        for main in range(1, main_count + 1)
    ]
    channel_matrix_sends = [
        _channel_matrix_send_row(client, channel, matrix)
        for channel in range(1, total_channels + 1)
        for matrix in range(1, matrix_send_count + 1)
    ]
    output_routes = [
        _output_route_row(client, group, output_number)
        for group in groups
        for output_number in range(1, output_count + 1)
    ]
    read_coverage = _routing_send_coverage_summary({
        "static_dante_routes": static_dante_routes,
        "channel_bus_sends": channel_bus_sends,
        "channel_main_sends": channel_main_sends,
        "channel_matrix_sends": channel_matrix_sends,
        "output_routes": output_routes,
    })
    cache_freshness = _cache_freshness_for_addresses(
        client,
        _routing_sends_cache_addresses(
            total_channels=total_channels,
            send_count=send_count,
            main_count=main_count,
            matrix_send_count=matrix_send_count,
            output_groups=groups,
            output_count=output_count,
        ),
    )
    return {
        "type": "routing_sends_snapshot",
        "generated_at": _utc_now(),
        "source": "static_config_and_runtime_cached_state",
        "available": True,
        "connection": _connection_summary(server),
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "static_dante_route_count": len(static_dante_routes),
            "channel_bus_send_count": len(channel_bus_sends),
            "channel_main_send_count": len(channel_main_sends),
            "channel_matrix_send_count": len(channel_matrix_sends),
            "output_route_count": len(output_routes),
            "active_bus_send_count": sum(1 for row in channel_bus_sends if row.get("active")),
            "active_main_send_count": sum(1 for row in channel_main_sends if row.get("active")),
            "active_matrix_send_count": sum(1 for row in channel_matrix_sends if row.get("active")),
            "read_coverage_status": read_coverage["status"],
            "cache_freshness_status": cache_freshness["status"],
            "live_apply_allowed": False,
            "claim": "static routing and cached send read coverage are exposed but not complete autonomous proof",
        },
        "read_coverage": read_coverage,
        "cache_freshness": cache_freshness,
        "static_dante_routes": static_dante_routes,
        "channel_bus_sends": channel_bus_sends,
        "channel_main_sends": channel_main_sends,
        "channel_matrix_sends": channel_matrix_sends,
        "output_routes": output_routes,
    }


def _coerce_snapshot_index(value: Any) -> Optional[int]:
    if isinstance(value, str):
        value = value.strip().replace("scene:", "").replace("snapshot:", "")
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    return index if index > 0 else None


def _normalize_snapshot_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _snapshot_name_from_catalog_row(row: Any) -> Optional[str]:
    if isinstance(row, Mapping):
        for key in ("name", "$name", "snapshot_name", "scene_name", "label"):
            name = _normalize_snapshot_name(row.get(key))
            if name:
                return name
        return None
    return _normalize_snapshot_name(row)


def _cached_snapshot_catalog(client: Any) -> Dict[int, str]:
    state = getattr(client, "state", None)
    if not isinstance(state, dict):
        return {}
    catalog: Dict[int, str] = {}
    for key in ("snapshot_catalog", "snapshots", "/$ctl/lib/snapshots", "/snapshots"):
        raw = state.get(key)
        if isinstance(raw, Mapping):
            for raw_index, row in raw.items():
                index = _coerce_snapshot_index(raw_index)
                name = _snapshot_name_from_catalog_row(row)
                if index is not None and name:
                    catalog[index] = name
        elif isinstance(raw, (list, tuple)):
            for offset, row in enumerate(raw, start=1):
                index = offset
                name = _snapshot_name_from_catalog_row(row)
                if isinstance(row, Mapping):
                    index = _coerce_snapshot_index(row.get("index") or row.get("snapshot_index")) or offset
                if name:
                    catalog[index] = name
    return catalog


def _cached_snapshot_name(
    client: Any,
    index: int,
    *,
    catalog: Mapping[int, str],
    active_index: Optional[int],
    active_name: Optional[str],
) -> Optional[str]:
    if index in catalog:
        return catalog[index]
    for address in (
        f"/$ctl/lib/{index}/$name",
        f"/$ctl/lib/{index}/name",
        f"/$ctl/lib/$name/{index}",
        f"/snap/{index}/$name",
        f"/snap/{index}/name",
        f"/scene/{index}/name",
    ):
        name = _normalize_snapshot_name(_cached_raw(client, address))
        if name:
            return name
    cursor_index = _cached_int(client, "/$ctl/lib/$actionidx")
    if cursor_index == index:
        name = _normalize_snapshot_name(_cached_raw(client, "/$ctl/lib/$name"))
        if name:
            return name
    if active_index == index:
        return active_name
    return None


def _snapshot_scene_row(
    client: Any,
    index: int,
    *,
    catalog: Mapping[int, str],
    active_index: Optional[int],
    active_name: Optional[str],
) -> Dict[str, Any]:
    name = _cached_snapshot_name(
        client,
        index,
        catalog=catalog,
        active_index=active_index,
        active_name=active_name,
    )
    active_known = active_index is not None
    fields = {
        "snapshot_name": bool(name),
        "active_marker": active_known,
    }
    active = bool(active_known and active_index == index)
    return {
        "snapshot_index": index,
        "snapshot_name": name,
        "active": active,
        "artifact_type": "snapshot_scene",
        "scene_load_policy": "blocked_manual_only",
        "scene_save_policy": "blocked_manual_only",
        "destructive_change_marker": "scene_recall_would_overwrite_console_state",
        "destructive_persistence_marker": "scene_save_would_overwrite_or_create_console_state",
        "read_coverage": _read_coverage_section(
            fields,
            "mixer_state_cache" if any(fields.values()) else "unavailable",
        ),
    }


def _snapshot_scene_coverage_summary(rows: Iterable[Mapping[str, Any]], active_fields: Mapping[str, bool]) -> Dict[str, Any]:
    status_counts: Dict[str, int] = {"complete": 0, "partial": 0, "missing": 0}
    row_list = tuple(rows)
    for row in row_list:
        coverage = row.get("read_coverage") if isinstance(row, dict) else {}
        status = str(coverage.get("status") or "missing") if isinstance(coverage, dict) else "missing"
        status_counts[status] = status_counts.get(status, 0) + 1
    statuses = [
        status
        for status, count in status_counts.items()
        for _ in range(count)
    ]
    active_section = _read_coverage_section(
        active_fields,
        "mixer_state_cache" if any(active_fields.values()) else "unavailable",
    )
    status = _overall_read_coverage_status([*statuses, active_section["status"]])
    return {
        "status": status,
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "surface_status_counts": {
            "snapshot_slots": status_counts,
            "active_scene": {
                "complete": 1 if active_section["status"] == "complete" else 0,
                "partial": 1 if active_section["status"] == "partial" else 0,
                "missing": 1 if active_section["status"] == "missing" else 0,
            },
        },
        "row_status_counts": status_counts,
        "sections": {"active_scene": active_section},
    }


def _snapshot_scene_cache_addresses(snapshot_count: int) -> Tuple[str, ...]:
    addresses = [
        "snapshot_catalog",
        "snapshots",
        "/$ctl/lib/snapshots",
        "/snapshots",
        "/$ctl/lib/$actidx",
        "/$ctl/lib/$activeidx",
        "/$ctl/lib/$active",
        "/$ctl/lib/$actname",
        "/$ctl/lib/$actshow",
    ]
    for index in range(1, snapshot_count + 1):
        addresses.extend(
            (
                f"/$ctl/lib/{index}/$name",
                f"/$ctl/lib/{index}/name",
                f"/$ctl/lib/$name/{index}",
                f"/snap/{index}/$name",
                f"/snap/{index}/name",
                f"/scene/{index}/name",
            )
        )
    return tuple(addresses)


def build_snapshots_scenes_snapshot(
    server: Any,
    *,
    snapshot_count: int = DEFAULT_SNAPSHOT_COUNT,
) -> Dict[str, Any]:
    """Return cached snapshot/scene catalog state without querying or loading scenes."""

    client = getattr(server, "mixer_client", None)
    snapshot_count = _bounded_int_value(snapshot_count, default=DEFAULT_SNAPSHOT_COUNT, minimum=1, maximum=500)
    catalog = _cached_snapshot_catalog(client)
    active_index = _cached_int_first(client, "/$ctl/lib/$actidx", "/$ctl/lib/$activeidx")
    active_name = _normalize_snapshot_name(
        _cached_raw(client, "/$ctl/lib/$active")
        or _cached_raw(client, "/$ctl/lib/$actname")
    )
    current_show = _normalize_snapshot_name(_cached_raw(client, "/$ctl/lib/$actshow"))
    rows = [
        _snapshot_scene_row(
            client,
            index,
            catalog=catalog,
            active_index=active_index,
            active_name=active_name,
        )
        for index in range(1, snapshot_count + 1)
    ]
    active_fields = {
        "active_snapshot_index": active_index is not None,
        "active_snapshot_name": bool(active_name),
        "current_show": bool(current_show),
    }
    read_coverage = _snapshot_scene_coverage_summary(rows, active_fields)
    cache_freshness = _cache_freshness_for_addresses(
        client,
        _snapshot_scene_cache_addresses(snapshot_count),
    )
    return {
        "type": "snapshots_scenes_snapshot",
        "generated_at": _utc_now(),
        "source": "runtime_cached_state",
        "available": True,
        "connection": _connection_summary(server),
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "full_autonomous_ready": False,
            "scene_load_allowed": False,
            "scene_save_allowed": False,
        },
        "summary": {
            "snapshot_count": len(rows),
            "named_snapshot_count": sum(1 for row in rows if row.get("snapshot_name")),
            "active_snapshot_index": active_index,
            "active_snapshot_name": active_name,
            "current_show": current_show,
            "read_coverage_status": read_coverage["status"],
            "cache_freshness_status": cache_freshness["status"],
            "live_apply_allowed": False,
            "claim": "cached snapshot/scene catalog read coverage is exposed but scene load/save remains blocked",
        },
        "cache_freshness": cache_freshness,
        "active_scene": {
            "active_snapshot_index": active_index,
            "active_snapshot_name": active_name,
            "current_show": current_show,
            "read_coverage": read_coverage["sections"]["active_scene"],
        },
        "read_coverage": read_coverage,
        "snapshots": rows,
    }


def _coerce_preset_index(value: Any) -> Optional[int]:
    if isinstance(value, str):
        value = value.strip().replace("preset:", "").replace("user_preset:", "")
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    return index if index > 0 else None


def _preset_name_from_catalog_row(row: Any) -> Optional[str]:
    if isinstance(row, Mapping):
        for key in ("name", "$name", "preset_name", "label"):
            name = _normalize_snapshot_name(row.get(key))
            if name:
                return name
        return None
    return _normalize_snapshot_name(row)


def _cached_user_preset_catalog(client: Any) -> Dict[int, Dict[str, Any]]:
    state = getattr(client, "state", None)
    if not isinstance(state, dict):
        return {}
    catalog: Dict[int, Dict[str, Any]] = {}
    for key in ("user_preset_catalog", "user_presets", "presets", "/presets", "/$ctl/lib/presets"):
        raw = state.get(key)
        if isinstance(raw, Mapping):
            for raw_index, row in raw.items():
                index = _coerce_preset_index(raw_index)
                name = _preset_name_from_catalog_row(row)
                if index is None or not name:
                    continue
                metadata = dict(row) if isinstance(row, Mapping) else {}
                metadata["preset_name"] = name
                catalog[index] = metadata
        elif isinstance(raw, (list, tuple)):
            for offset, row in enumerate(raw, start=1):
                index = offset
                name = _preset_name_from_catalog_row(row)
                metadata = dict(row) if isinstance(row, Mapping) else {}
                if isinstance(row, Mapping):
                    index = _coerce_preset_index(row.get("index") or row.get("preset_index")) or offset
                if not name:
                    continue
                metadata["preset_name"] = name
                catalog[index] = metadata
    return catalog


def _cached_user_preset_row(client: Any, index: int, *, catalog: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    row = dict(catalog.get(index) or {})
    name = _normalize_snapshot_name(row.get("preset_name"))
    if not name:
        for address in (
            f"/preset/{index}/$name",
            f"/preset/{index}/name",
            f"/user/preset/{index}/name",
            f"/$ctl/lib/preset/{index}/$name",
        ):
            name = _normalize_snapshot_name(_cached_raw(client, address))
            if name:
                break
    preset_scope = _normalize_snapshot_name(
        row.get("scope")
        or row.get("preset_scope")
        or row.get("target_scope")
        or _cached_raw(client, f"/preset/{index}/scope")
    )
    target_type = _normalize_snapshot_name(
        row.get("target_type")
        or row.get("type")
        or row.get("artifact_type")
        or _cached_raw(client, f"/preset/{index}/type")
    )
    target_label = _normalize_snapshot_name(
        row.get("target")
        or row.get("target_label")
        or row.get("channel")
        or _cached_raw(client, f"/preset/{index}/target")
    )
    fields = {
        "preset_name": bool(name),
        "preset_scope": bool(preset_scope),
        "target_type": bool(target_type),
        "target_label": bool(target_label),
    }
    return {
        "preset_index": index,
        "preset_name": name,
        "preset_scope": preset_scope,
        "target_type": target_type,
        "target_label": target_label,
        "preset_recall_policy": "blocked_manual_only",
        "preset_store_policy": "blocked_manual_only",
        "destructive_recall_marker": "preset_recall_would_change_console_parameters",
        "destructive_store_marker": "preset_store_would_persist_console_parameters",
        "read_coverage": _read_coverage_section(
            fields,
            "mixer_state_cache" if any(fields.values()) else "unavailable",
        ),
    }


def _user_preset_coverage_summary(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    status_counts: Dict[str, int] = {"complete": 0, "partial": 0, "missing": 0}
    for row in rows:
        coverage = row.get("read_coverage") if isinstance(row, dict) else {}
        status = str(coverage.get("status") or "missing") if isinstance(coverage, dict) else "missing"
        status_counts[status] = status_counts.get(status, 0) + 1
    statuses = [
        status
        for status, count in status_counts.items()
        for _ in range(count)
    ]
    return {
        "status": _overall_read_coverage_status(statuses),
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "surface_status_counts": {"user_preset_slots": status_counts},
        "row_status_counts": status_counts,
    }


def _user_preset_cache_addresses(preset_count: int) -> Tuple[str, ...]:
    addresses = [
        "user_preset_catalog",
        "user_presets",
        "presets",
        "/presets",
        "/$ctl/lib/presets",
    ]
    for index in range(1, preset_count + 1):
        addresses.extend(
            (
                f"/preset/{index}/$name",
                f"/preset/{index}/name",
                f"/user/preset/{index}/name",
                f"/$ctl/lib/preset/{index}/$name",
                f"/preset/{index}/scope",
                f"/preset/{index}/type",
                f"/preset/{index}/target",
            )
        )
    return tuple(addresses)


def build_user_presets_snapshot(
    server: Any,
    *,
    preset_count: int = DEFAULT_USER_PRESET_COUNT,
) -> Dict[str, Any]:
    """Return cached user preset catalog state without recalling or storing presets."""

    client = getattr(server, "mixer_client", None)
    preset_count = _bounded_int_value(preset_count, default=DEFAULT_USER_PRESET_COUNT, minimum=1, maximum=500)
    catalog = _cached_user_preset_catalog(client)
    rows = [
        _cached_user_preset_row(client, index, catalog=catalog)
        for index in range(1, preset_count + 1)
    ]
    read_coverage = _user_preset_coverage_summary(rows)
    cache_freshness = _cache_freshness_for_addresses(
        client,
        _user_preset_cache_addresses(preset_count),
    )
    return {
        "type": "user_presets_snapshot",
        "generated_at": _utc_now(),
        "source": "runtime_cached_state",
        "available": True,
        "connection": _connection_summary(server),
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "full_autonomous_ready": False,
            "preset_recall_allowed": False,
            "preset_store_allowed": False,
        },
        "summary": {
            "preset_count": len(rows),
            "named_preset_count": sum(1 for row in rows if row.get("preset_name")),
            "read_coverage_status": read_coverage["status"],
            "cache_freshness_status": cache_freshness["status"],
            "live_apply_allowed": False,
            "claim": "cached user preset catalog read coverage is exposed but preset recall/store remains blocked",
        },
        "cache_freshness": cache_freshness,
        "read_coverage": read_coverage,
        "presets": rows,
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
    proof_evidence = build_supervised_proof_evidence_snapshot(server)
    snapshot["runtime"] = build_pipeline_runtime_summary(operator_mode, manual_write_supervision)
    snapshot["operator_mode"] = operator_mode
    snapshot["supervised_proof_evidence"] = proof_evidence
    snapshot["autonomy_evidence_overlay"] = proof_evidence["autonomy_evidence_overlay"]
    return snapshot


def _telemetry_storage_dir(server: Any) -> Path:
    config = getattr(server, "config", {})
    telemetry = config.get("telemetry", {}) if isinstance(config, dict) else {}
    return Path(telemetry.get("storage_dir") or DEFAULT_TELEMETRY_STORAGE_DIR)


def _read_json_file(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _configured_path_values(config: Dict[str, Any], section_names: Iterable[str], key_names: Iterable[str]) -> List[str]:
    values: List[str] = []
    for section_name in section_names:
        section = config.get(section_name)
        if not isinstance(section, dict):
            continue
        for key_name in key_names:
            raw_value = section.get(key_name)
            if isinstance(raw_value, str) and raw_value.strip():
                values.append(raw_value.strip())
            elif isinstance(raw_value, list):
                values.extend(str(item).strip() for item in raw_value if str(item).strip())
    return values


def _supervised_proof_report_roots(server: Any) -> List[Path]:
    config = getattr(server, "config", {})
    if not isinstance(config, dict):
        config = {}
    configured = _configured_path_values(
        config,
        ("supervised_proof_evidence", "proof_evidence"),
        ("report_path", "report_paths", "artifact_root", "artifact_roots", "storage_dir", "storage_dirs"),
    )
    paths = configured or list(DEFAULT_SUPERVISED_PROOF_REPORT_DIRS)
    return [Path(path) for path in paths]


def _supervised_proof_report_stale_after_sec(server: Any) -> int:
    config = getattr(server, "config", {})
    if not isinstance(config, dict):
        config = {}
    for section_name in ("supervised_proof_evidence", "proof_evidence"):
        section = config.get(section_name)
        if isinstance(section, dict) and "stale_after_sec" in section:
            return max(1, _safe_int_value(section.get("stale_after_sec"), DEFAULT_SUPERVISED_PROOF_REPORT_STALE_AFTER_SEC))
    return DEFAULT_SUPERVISED_PROOF_REPORT_STALE_AFTER_SEC


def _supervised_proof_report_candidates(roots: Iterable[Path]) -> List[Path]:
    candidates: List[Path] = []
    for root in roots:
        path = Path(root)
        if path.is_file():
            candidates.append(path)
            continue
        if not path.is_dir():
            continue
        for filename in SUPERVISED_PROOF_REPORT_FILENAMES:
            candidates.extend(candidate for candidate in path.rglob(filename) if candidate.is_file())
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates


def _string_list_value(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]


def _int_mapping(value: Any) -> Dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(key): _safe_int_value(raw_value) for key, raw_value in value.items()}


def _proof_report_freshness(
    report: Dict[str, Any],
    *,
    report_path: Path,
    generated_at: str,
    stale_after_sec: int,
) -> Dict[str, Any]:
    report_epoch = _coerce_epoch_seconds(report.get("generated_at"))
    source = "report_generated_at"
    if report_epoch is None:
        try:
            report_epoch = report_path.stat().st_mtime
            source = "file_mtime"
        except OSError:
            report_epoch = None
            source = "missing_timestamp"
    now_epoch = _coerce_epoch_seconds(generated_at) or _utc_epoch()
    age_sec = max(0.0, now_epoch - report_epoch) if report_epoch is not None else None
    stale = True if age_sec is None else age_sec > stale_after_sec
    return {
        "latest_report_timestamp_source": source,
        "latest_report_age_sec": round(age_sec, 3) if age_sec is not None else None,
        "stale_after_sec": stale_after_sec,
        "latest_report_stale": stale,
    }


def _read_surface_proof_quality_from_summary(report_path: Path) -> Dict[str, Any]:
    summary_path = report_path.parent / "summary.json"
    summary = _read_json_file(summary_path)
    read_surface = (
        summary.get("read_surface_fixture_evidence")
        if isinstance(summary.get("read_surface_fixture_evidence"), dict)
        else {}
    )
    available = bool(read_surface)
    physical_missing = _string_list_value(read_surface.get("physical_fixture_missing_surface_ids"))
    physical_gap_count = _safe_int_value(read_surface.get("physical_gap_surface_count"), len(physical_missing))
    source_full_autonomous_ready = read_surface.get("full_autonomous_ready") is True
    return {
        "available": available,
        "summary_path": str(summary_path) if summary_path.exists() else None,
        "status": str(read_surface.get("status") or "missing"),
        "snapshot_status": read_surface.get("snapshot_status"),
        "surface_count": _safe_int_value(read_surface.get("surface_count")),
        "surface_with_fixture_count": _safe_int_value(read_surface.get("surface_with_fixture_count")),
        "declared_fixture_path_count": _safe_int_value(read_surface.get("declared_fixture_path_count")),
        "declared_test_name_count": _safe_int_value(read_surface.get("declared_test_name_count")),
        "physical_gap_surface_count": physical_gap_count,
        "summary_recomputed_from_rows": read_surface.get("summary_recomputed_from_rows") is True,
        "fixture_declaration_validation": read_surface.get("fixture_declaration_validation") is True,
        "contract_gap_consistency_enforced": read_surface.get("contract_gap_consistency_enforced") is True,
        "read_only": True,
        "live_write": False,
        "full_autonomous_ready": False,
        "source_full_autonomous_ready_claimed": source_full_autonomous_ready,
        "physical_fixture_missing_surface_ids": physical_missing,
        "violations": _string_list_value(read_surface.get("violations")),
    }


def _empty_autonomy_evidence_overlay(*, generated_at: str, report_path: Optional[Path] = None) -> Dict[str, Any]:
    rows = []
    for value_type, target_id in (
        ("fader", "fader_supervised_promotion"),
        ("gain", "gain_trim_supervised_promotion"),
    ):
        rows.append(
            {
                "id": f"{value_type}_autonomy_evidence_overlay",
                "target_id": target_id,
                "value_type": value_type,
                "status": "no_current_evidence",
                "safety_review_candidate": False,
                "supervised_live_allowed_static": value_type == "fader",
                "autonomous_ready": False,
                "proof_artifact_count": 0,
                "physical_candidate_count": 0,
                "dry_run_ready_count": 0,
                "blocked_artifact_count": 0,
                "remediation_action_count": 0,
                "physical_candidate_proof_ids": [],
                "dry_run_ready_proof_ids": [],
                "blocked_proof_ids": [],
                "remediation_action_ids": [],
                "blockers": ["proof_artifact_missing"],
                "next_required_actions": ["Collect supervised proof artifacts before promotion review."],
            }
        )
    return {
        "type": "autonomy_evidence_overlay",
        "generated_at": generated_at,
        "source": "supervised_proof_artifact_report",
        "source_report_available": False,
        "source_report_path": str(report_path) if report_path else None,
        "status": "no_current_evidence",
        "safety": {
            "read_only": True,
            "live_write": False,
            "live_widening_allowed": False,
            "automatic_safety_update_allowed": False,
            "full_autonomous_ready": False,
            "operator_required": True,
        },
        "summary": {
            "target_count": len(rows),
            "status_counts": {
                "dry_run_only": 0,
                "manual_safety_review_candidate": 0,
                "no_current_evidence": len(rows),
                "remediation_required": 0,
            },
            "manual_safety_review_candidate_target_ids": [],
            "remediation_required_target_ids": [],
            "dry_run_only_target_ids": [],
            "no_current_evidence_target_ids": [row["target_id"] for row in rows],
            "autonomous_ready_target_ids": [],
            "global_blockers": [
                "proof_report_missing",
                "manual_safety_review_required_before_promotion",
                "full_autonomous_mode_remains_blocked",
            ],
        },
        "rows": rows,
    }


def _normalized_autonomy_overlay_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    value_type = str(row.get("value_type") or "")
    if value_type not in {"fader", "gain"}:
        return None
    target_id = str(
        row.get("target_id")
        or ("fader_supervised_promotion" if value_type == "fader" else "gain_trim_supervised_promotion")
    )
    status = str(row.get("status") or "no_current_evidence")
    if status not in AUTONOMY_OVERLAY_ROW_STATUSES:
        status = "no_current_evidence"
    blockers = _string_list_value(row.get("blockers"))
    if status == "manual_safety_review_candidate" and "automatic_promotion_forbidden" not in blockers:
        blockers.append("automatic_promotion_forbidden")
    if row.get("autonomous_ready") is not False:
        blockers.append("source_autonomous_ready_normalized_false")
    return {
        "id": str(row.get("id") or f"{value_type}_autonomy_evidence_overlay"),
        "target_id": target_id,
        "value_type": value_type,
        "status": status,
        "safety_review_candidate": bool(row.get("safety_review_candidate")),
        "supervised_live_allowed_static": bool(row.get("supervised_live_allowed_static")),
        "autonomous_ready": False,
        "proof_artifact_count": _safe_int_value(row.get("proof_artifact_count")),
        "physical_candidate_count": _safe_int_value(row.get("physical_candidate_count")),
        "dry_run_ready_count": _safe_int_value(row.get("dry_run_ready_count")),
        "blocked_artifact_count": _safe_int_value(row.get("blocked_artifact_count")),
        "remediation_action_count": _safe_int_value(row.get("remediation_action_count")),
        "physical_candidate_proof_ids": _string_list_value(row.get("physical_candidate_proof_ids")),
        "dry_run_ready_proof_ids": _string_list_value(row.get("dry_run_ready_proof_ids")),
        "blocked_proof_ids": _string_list_value(row.get("blocked_proof_ids")),
        "remediation_action_ids": _string_list_value(row.get("remediation_action_ids")),
        "blockers": blockers,
        "next_required_actions": _string_list_value(row.get("next_required_actions")),
    }


def _normalize_autonomy_evidence_overlay(
    overlay: Dict[str, Any],
    *,
    generated_at: str,
    report_path: Path,
) -> Dict[str, Any]:
    summary = overlay.get("summary") if isinstance(overlay.get("summary"), dict) else {}
    source_rows = overlay.get("rows") if isinstance(overlay.get("rows"), list) else []
    rows = [
        normalized
        for normalized in (
            _normalized_autonomy_overlay_row(row)
            for row in source_rows
            if isinstance(row, dict)
        )
        if normalized is not None
    ]
    present_value_types = {row["value_type"] for row in rows}
    if present_value_types != {"fader", "gain"}:
        fallback = _empty_autonomy_evidence_overlay(generated_at=generated_at, report_path=report_path)
        for row in fallback["rows"]:
            if row["value_type"] not in present_value_types:
                rows.append(row)
    status = str(overlay.get("status") or "no_current_evidence")
    if status not in AUTONOMY_OVERLAY_STATUSES:
        status = "no_current_evidence"
    global_blockers = _string_list_value(summary.get("global_blockers"))
    for blocker in (
        "overlay_does_not_modify_safety_rows",
        "manual_safety_review_required_before_promotion",
        "full_autonomous_mode_remains_blocked",
    ):
        if blocker not in global_blockers:
            global_blockers.append(blocker)
    return {
        "type": "autonomy_evidence_overlay",
        "generated_at": str(overlay.get("generated_at") or generated_at),
        "source": "supervised_proof_artifact_report",
        "source_report_available": True,
        "source_report_path": str(report_path),
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "live_widening_allowed": False,
            "automatic_safety_update_allowed": False,
            "full_autonomous_ready": False,
            "operator_required": True,
        },
        "summary": {
            "target_count": len(rows),
            "status_counts": _int_mapping(summary.get("status_counts")),
            "manual_safety_review_candidate_target_ids": _string_list_value(
                summary.get("manual_safety_review_candidate_target_ids")
            ),
            "remediation_required_target_ids": _string_list_value(summary.get("remediation_required_target_ids")),
            "dry_run_only_target_ids": _string_list_value(summary.get("dry_run_only_target_ids")),
            "no_current_evidence_target_ids": _string_list_value(summary.get("no_current_evidence_target_ids")),
            "autonomous_ready_target_ids": [],
            "global_blockers": global_blockers,
        },
        "rows": rows,
    }


def build_supervised_proof_evidence_snapshot(server: Any) -> Dict[str, Any]:
    """Return latest supervised proof report evidence without generating artifacts."""

    generated_at = _utc_now()
    roots = _supervised_proof_report_roots(server)
    stale_after_sec = _supervised_proof_report_stale_after_sec(server)
    candidates = _supervised_proof_report_candidates(roots)
    base = {
        "type": "supervised_proof_evidence_snapshot",
        "generated_at": generated_at,
        "source": "supervised_proof_artifact_report",
        "storage_roots": [str(path) for path in roots],
        "safety": {
            "read_only": True,
            "live_write": False,
            "generates_artifacts": False,
            "live_widening_allowed": False,
            "automatic_safety_update_allowed": False,
            "full_autonomous_ready": False,
        },
    }
    if not candidates:
        overlay = _empty_autonomy_evidence_overlay(generated_at=generated_at)
        return {
            **base,
            "available": False,
            "report_path": None,
            "summary": {
                "report_count": 0,
                "latest_report_status": "missing",
                "latest_report_age_sec": None,
                "latest_report_stale": True,
                "latest_report_timestamp_source": "missing_report",
                "stale_after_sec": stale_after_sec,
                "overlay_status": overlay["status"],
                "overlay_row_count": len(overlay["rows"]),
            },
            "autonomy_evidence_overlay": overlay,
        }

    report_path = candidates[0]
    report = _read_json_file(report_path)
    overlay_payload = report.get("autonomy_evidence_overlay")
    overlay = _normalize_autonomy_evidence_overlay(
        overlay_payload if isinstance(overlay_payload, dict) else {},
        generated_at=generated_at,
        report_path=report_path,
    )
    report_summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    read_surface_proof_quality = _read_surface_proof_quality_from_summary(report_path)
    freshness = _proof_report_freshness(
        report,
        report_path=report_path,
        generated_at=generated_at,
        stale_after_sec=stale_after_sec,
    )
    return {
        **base,
        "available": bool(report),
        "report_path": str(report_path),
        "summary": {
            "report_count": len(candidates),
            "latest_report_status": str(report.get("status") or "unavailable"),
            "latest_report_generated_at": report.get("generated_at"),
            **freshness,
            "artifact_count": _safe_int_value(report_summary.get("artifact_count")),
            "invalid_summary_count": _safe_int_value(report_summary.get("invalid_summary_count")),
            "overlay_status": overlay["status"],
            "overlay_row_count": len(overlay["rows"]),
            "read_surface_proof_quality": read_surface_proof_quality,
        },
        "autonomy_evidence_overlay": overlay,
    }


def _read_jsonl_file(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except OSError:
        return []
    return rows


def _severity_counts(alerts: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"critical": 0, "warning": 0, "info": 0}
    for alert in alerts:
        severity = str(alert.get("severity") or "info").lower()
        counts[severity if severity in counts else "info"] += 1
    return counts


def _telemetry_alert_session_safety(raw_safety: Any) -> Dict[str, Any]:
    safety = raw_safety if isinstance(raw_safety, dict) else {}
    source_live_write = bool(safety.get("live_write", False))
    return {
        "read_only": True,
        "live_write": False,
        "generates_artifacts": False,
        "source_live_write": source_live_write,
        "source_safety_normalized": source_live_write or safety.get("read_only") is False,
    }


def _telemetry_alert_context(raw_context: Any) -> Dict[str, Any]:
    if not isinstance(raw_context, dict):
        return {}
    context: Dict[str, Any] = {}
    for key in sorted(TELEMETRY_ALERT_CONTEXT_ALLOWLIST):
        if key not in raw_context:
            continue
        value = raw_context[key]
        if isinstance(value, (str, int, float, bool)) or value is None:
            context[key] = value
    return context


def _telemetry_alert_row(row: Dict[str, Any], fallback_session_id: str) -> Dict[str, Any]:
    severity = str(row.get("severity") or "info").lower()
    if severity not in {"critical", "warning", "info"}:
        severity = "info"
    return {
        "timestamp_ms": _safe_int_value(row.get("timestamp_ms")),
        "session_id": str(row.get("session_id") or fallback_session_id),
        "rule": str(row.get("rule") or "unknown"),
        "severity": severity,
        "target_type": str(row.get("target_type") or "unknown"),
        "target_id": str(row.get("target_id") or "unknown"),
        "message": str(row.get("message") or "Telemetry alert requires review."),
        "value": _round(row.get("value"), 3),
        "threshold": _round(row.get("threshold"), 3),
        "window_ms": _safe_int_value(row.get("window_ms")) if row.get("window_ms") is not None else None,
        "context": _telemetry_alert_context(row.get("context")),
    }


def build_telemetry_alerts_snapshot(
    server: Any,
    *,
    limit: int = 50,
    session_limit: int = 8,
) -> Dict[str, Any]:
    """Return the latest generated streaming alerts without creating artifacts.

    Streaming alert generation is an explicit local CLI/tooling action. This
    Product Layer snapshot only reads existing files so the cockpit can display
    telemetry risk evidence without creating a mixer or OSC write path.
    """

    limit = _bounded_int_value(limit, default=50, minimum=1, maximum=200)
    session_limit = _bounded_int_value(session_limit, default=8, minimum=1, maximum=32)
    storage_dir = _telemetry_storage_dir(server)
    if not storage_dir.exists():
        return {
            "type": "telemetry_alerts_snapshot",
            "generated_at": _utc_now(),
            "source": "telemetry_alert_artifacts",
            "available": False,
            "storage_dir": str(storage_dir),
            "safety": {
                "read_only": True,
                "live_write": False,
                "generates_artifacts": False,
            },
            "summary": {
                "session_count": 0,
                "artifact_session_count": 0,
                "alert_count": 0,
                "critical_count": 0,
                "warning_count": 0,
                "latest_session_id": None,
            },
            "sessions": [],
            "alerts": [],
        }

    session_dirs = [
        path
        for path in storage_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    session_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    sessions: List[Dict[str, Any]] = []
    alerts: List[Dict[str, Any]] = []
    for session_dir in session_dirs:
        alerts_dir = session_dir / "alerts"
        summary_path = alerts_dir / "summary.json"
        alerts_path = alerts_dir / "alerts.jsonl"
        if not summary_path.exists() and not alerts_path.exists():
            continue

        summary = _read_json_file(summary_path) if summary_path.exists() else {}
        session_id = str(summary.get("session_id") or session_dir.name)
        rows = [
            _telemetry_alert_row(row, session_id)
            for row in _read_jsonl_file(alerts_path)
        ]
        alerts.extend(rows)
        sessions.append(
            {
                "session_id": session_id,
                "generated_at": summary.get("generated_at"),
                "event_count": summary.get("event_count"),
                "alert_count": _safe_int_value(summary.get("alert_count"), len(rows)),
                "dashboard_row_count": summary.get("dashboard_row_count"),
                "summary_path": str(summary_path) if summary_path.exists() else None,
                "alerts_path": str(alerts_path) if alerts_path.exists() else None,
                "dashboard_path": str(alerts_dir / "dashboard.csv"),
                "safety": _telemetry_alert_session_safety(summary.get("safety")),
            }
        )
        if len(sessions) >= session_limit:
            break

    alerts.sort(key=lambda row: _safe_int_value(row.get("timestamp_ms")), reverse=True)
    severity = _severity_counts(alerts)
    latest_session_id = sessions[0]["session_id"] if sessions else None
    artifact_available = bool(sessions)
    return {
        "type": "telemetry_alerts_snapshot",
        "generated_at": _utc_now(),
        "source": "telemetry_alert_artifacts",
        "available": artifact_available,
        "storage_dir": str(storage_dir),
        "safety": {
            "read_only": True,
            "live_write": False,
            "generates_artifacts": False,
        },
        "summary": {
            "session_count": len(session_dirs),
            "artifact_session_count": len(sessions),
            "alert_count": len(alerts),
            "critical_count": severity["critical"],
            "warning_count": severity["warning"],
            "latest_session_id": latest_session_id,
        },
        "sessions": sessions,
        "alerts": alerts[:limit],
    }


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
