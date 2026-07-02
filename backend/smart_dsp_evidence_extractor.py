"""Offline audio evidence extraction for smart DSP strategy selection.

This module is transport-free. It accepts already captured or synthetic audio
arrays, derives compact evidence, and builds SmartDSPAnalysisInput objects for
the proposal-only smart DSP selector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from smart_dsp_strategy_engine import (
        HIGH_FEEDBACK_RISK,
        MAX_MONO_SUM_LOSS_DB,
        MIN_SOURCE_IDENTITY_CONFIDENCE,
        MIN_PHASE_CORRELATION_FOR_WIDTH,
        MIN_TRUE_PEAK_HEADROOM_DB,
        SOURCE_IDENTITY_MISMATCH_BLOCK_THRESHOLD,
        SmartDSPAnalysisInput,
        SmartDSPRecommendation,
        WING_DYNAMICS_THRESHOLD_PATH_SUFFIX_BY_SURFACE,
        WING_FX_MODEL_PARAMETER_BINDINGS_BY_MODEL,
        WING_FX_MODEL_READBACK_PATH_SUFFIX,
        WING_GAIN_WRITE_PATH_SUFFIXES,
        analyze_smart_dsp_strategy,
        classify_musical_role,
        classify_source_family,
        normalize_mode,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.smart_dsp_strategy_engine import (
        HIGH_FEEDBACK_RISK,
        MAX_MONO_SUM_LOSS_DB,
        MIN_SOURCE_IDENTITY_CONFIDENCE,
        MIN_PHASE_CORRELATION_FOR_WIDTH,
        MIN_TRUE_PEAK_HEADROOM_DB,
        SOURCE_IDENTITY_MISMATCH_BLOCK_THRESHOLD,
        SmartDSPAnalysisInput,
        SmartDSPRecommendation,
        WING_DYNAMICS_THRESHOLD_PATH_SUFFIX_BY_SURFACE,
        WING_FX_MODEL_PARAMETER_BINDINGS_BY_MODEL,
        WING_FX_MODEL_READBACK_PATH_SUFFIX,
        WING_GAIN_WRITE_PATH_SUFFIXES,
        analyze_smart_dsp_strategy,
        classify_musical_role,
        classify_source_family,
        normalize_mode,
    )


SMART_DSP_EVIDENCE_EXTRACTOR_VERSION = "smart_dsp_audio_evidence_extractor_v1"
SMART_DSP_WING_READONLY_PREFLIGHT_STATE_METHOD_ID = (
    "smart_dsp_wing_readonly_preflight_current_state"
)
DEFAULT_SAMPLE_RATE = 48000
EPSILON = 1e-12

SPECTRAL_BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("sub", 20.0, 60.0),
    ("bass", 60.0, 250.0),
    ("low_mid", 250.0, 500.0),
    ("mid", 500.0, 2000.0),
    ("presence", 2000.0, 5000.0),
    ("sibilance", 5000.0, 10000.0),
    ("air", 10000.0, 18000.0),
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _clip01(value: Any) -> float:
    return _clip(_safe_float(value), 0.0, 1.0)


def _dbfs(linear: float, floor_db: float = -120.0) -> float:
    if linear <= EPSILON:
        return floor_db
    return max(floor_db, 20.0 * math.log10(float(linear)))


def _power_db(power: float, floor_db: float = -160.0) -> float:
    if power <= EPSILON:
        return floor_db
    return max(floor_db, 10.0 * math.log10(float(power)))


def _ordered_unique(values: Iterable[str]) -> Tuple[str, ...]:
    ordered = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return tuple(ordered)


def _bool_or_none(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        if numeric == 0.0:
            return False
        if numeric == 1.0:
            return True
        return None
    if isinstance(value, str):
        token = value.strip().lower().replace("-", "_").replace(" ", "_")
        if token in {"true", "1", "on", "yes", "enabled", "operator_owned", "locked"}:
            return True
        if token in {"false", "0", "off", "no", "disabled", "unlocked"}:
            return False
    return None


def _mapping_first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _resolve_bool_state(
    explicit: Optional[bool],
    state: Mapping[str, Any],
    keys: Sequence[str],
    *,
    default: bool = False,
) -> bool:
    if explicit is not None:
        return bool(explicit)
    resolved = _bool_or_none(_mapping_first_present(state, keys))
    return default if resolved is None else resolved


def _resolve_optional_bool_state(
    explicit: Optional[bool],
    state: Mapping[str, Any],
    keys: Sequence[str],
) -> Optional[bool]:
    if explicit is not None:
        return bool(explicit)
    return _bool_or_none(_mapping_first_present(state, keys))


def _resolve_string_state(
    explicit: str,
    state: Mapping[str, Any],
    keys: Sequence[str],
) -> str:
    if str(explicit or "").strip():
        return str(explicit)
    resolved = _mapping_first_present(state, keys)
    return "" if resolved is None else str(resolved)


def _resolve_optional_float_state(
    explicit: Optional[float],
    state: Mapping[str, Any],
    keys: Sequence[str],
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> Optional[float]:
    raw = explicit if explicit is not None else _mapping_first_present(state, keys)
    if raw is None:
        return None
    numeric = _readback_float(raw)
    if numeric is None:
        return None
    if minimum is not None:
        numeric = max(float(minimum), numeric)
    if maximum is not None:
        numeric = min(float(maximum), numeric)
    return numeric


def _json_object_from_path(path: str | Path) -> Mapping[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, Mapping) else {}


def _jsonl_objects_from_path(path: str | Path) -> Tuple[Mapping[str, Any], ...]:
    rows: List[Mapping[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, Mapping):
                rows.append(payload)
    return tuple(rows)


def _readonly_preflight_safety_violations(summary: Mapping[str, Any]) -> Tuple[str, ...]:
    checks = (
        ("read_only", True),
        ("live_write", False),
        ("write_request_sent", False),
        ("proposal_request_sent", False),
        ("live_authority_granted", False),
        ("readiness_closure_allowed", False),
    )
    violations: List[str] = []
    for key, expected in checks:
        if key in summary and summary.get(key) is not expected:
            violations.append(f"{key}_not_{str(expected).lower()}")
    return tuple(violations)


def _readonly_preflight_summary_readbacks(
    summary: Optional[Mapping[str, Any]],
) -> Tuple[Mapping[str, Any], ...]:
    if not isinstance(summary, Mapping):
        return ()
    readbacks = summary.get("readbacks")
    if not isinstance(readbacks, list):
        return ()
    return tuple(row for row in readbacks if isinstance(row, Mapping))


def _readonly_preflight_event_readbacks(
    events: Optional[Iterable[Mapping[str, Any]]],
) -> Tuple[Mapping[str, Any], ...]:
    if events is None:
        return ()
    readbacks: List[Mapping[str, Any]] = []
    for event in events:
        if not isinstance(event, Mapping) or event.get("event_type") != "recv":
            continue
        payload = event.get("payload")
        if isinstance(payload, Mapping) and payload.get("type") == "mixer_parameter_value":
            readbacks.append(payload)
    return tuple(readbacks)


def _physical_readonly_readbacks(
    readbacks: Iterable[Mapping[str, Any]],
) -> Tuple[Mapping[str, Any], ...]:
    rows: List[Mapping[str, Any]] = []
    for row in readbacks:
        if row.get("type") != "mixer_parameter_value":
            continue
        if row.get("source") != "physical_readback":
            continue
        if row.get("status") != "verified_from_physical_readback":
            continue
        rows.append(row)
    return tuple(rows)


def _target_channel_from_target_id(target_id: str) -> Optional[int]:
    token = str(target_id or "").strip().lower().replace("/", ":")
    parts = [part for part in token.split(":") if part]
    if len(parts) >= 2 and parts[0] in {"ch", "channel"}:
        try:
            channel = int(parts[1])
        except ValueError:
            return None
        return channel if channel > 0 else None
    return None


def _readback_channel(row: Mapping[str, Any]) -> Optional[int]:
    channel = row.get("channel")
    try:
        numeric = int(channel)
    except (TypeError, ValueError):
        return None
    return numeric if numeric > 0 else None


def _readback_bool(value: Any) -> Optional[bool]:
    return _bool_or_none(value)


def _readback_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def build_current_wing_state_from_readonly_preflight(
    *,
    summary: Optional[Mapping[str, Any]] = None,
    events: Optional[Iterable[Mapping[str, Any]]] = None,
    summary_path: str | Path | None = None,
    events_path: str | Path | None = None,
    target_id: str = "",
) -> Dict[str, Any]:
    """Build SmartDSP ``current_wing_state`` from read-only preflight artifacts.

    The function only consumes already captured JSON/JSONL artifacts. It never
    opens a mixer connection, never queries WING, and never grants live apply.
    Unsafe artifact flags fail closed by returning metadata without DSP state.
    """

    loaded_summary = dict(summary or {})
    if summary_path is not None:
        loaded_summary.update(_json_object_from_path(summary_path))
    loaded_events: Tuple[Mapping[str, Any], ...] = tuple(events or ())
    if events_path is not None:
        loaded_events = (*loaded_events, *_jsonl_objects_from_path(events_path))

    violations = list(_readonly_preflight_safety_violations(loaded_summary))
    state: Dict[str, Any] = {
        "method_id": SMART_DSP_WING_READONLY_PREFLIGHT_STATE_METHOD_ID,
        "current_wing_state_source": "wing_readonly_preflight_artifact",
        "current_wing_state_read_only": True,
        "current_wing_state_live_write": False,
        "current_wing_state_live_authority_granted": False,
        "current_wing_state_readiness_closure_allowed": False,
        "current_wing_state_run_id": str(loaded_summary.get("run_id") or ""),
        "current_wing_state_connection_mode": str(
            loaded_summary.get("connection_mode") or ""
        ),
        "current_wing_state_target_id": str(target_id or ""),
        "current_wing_state_violations": violations,
        "current_wing_state_readback_available": False,
        "current_wing_state_readback_count": 0,
        "current_wing_state_physical_readback_count": 0,
        "current_wing_state_channels": [],
        "current_wing_state_value_types": [],
        "current_wing_state_rollback_required": True,
        "current_wing_state_rollback_plan_available": False,
        "current_wing_state_live_apply_capable": False,
    }
    if violations:
        state["current_wing_state_rejected"] = True
        return state

    target_channel = _target_channel_from_target_id(target_id)
    readbacks = (
        *_readonly_preflight_summary_readbacks(loaded_summary),
        *_readonly_preflight_event_readbacks(loaded_events),
    )
    physical_readbacks = _physical_readonly_readbacks(readbacks)
    channels: List[int] = []
    value_types: List[str] = []
    raw_by_value_type: Dict[str, Any] = {}
    for row in physical_readbacks:
        channel = _readback_channel(row)
        if target_channel is not None and channel != target_channel:
            continue
        value_type = str(row.get("value_type") or "").strip()
        if not value_type:
            continue
        value = row.get("value")
        if channel is not None:
            channels.append(channel)
        value_types.append(value_type)
        raw_by_value_type[value_type] = value
        if value_type == "fader":
            numeric = _readback_float(value)
            if numeric is not None:
                state["current_wing_fader_db"] = round(numeric, 3)
                state["wing_readonly_fader_db"] = round(numeric, 3)
            state["current_wing_fader_readback_available"] = True
        elif value_type == "gain":
            numeric = _readback_float(value)
            if numeric is not None:
                state["current_wing_input_trim_db"] = round(numeric, 3)
                state["wing_readonly_input_trim_db"] = round(numeric, 3)
            state["current_wing_gain_readback_available"] = True
        elif value_type == "pan":
            numeric = _readback_float(value)
            if numeric is not None:
                clamped = _clip(numeric, -100.0, 100.0)
                state["current_wing_pan_percent"] = round(clamped, 3)
                state["wing_readonly_pan_percent"] = round(clamped, 3)
            state["current_wing_pan_readback_available"] = True
        elif value_type == "eq_on":
            state["current_wing_eq_enabled"] = _readback_bool(value)
            state["current_wing_eq_readback_available"] = True
        elif value_type == "eq_mix":
            numeric = _readback_float(value)
            if numeric is not None:
                state["wing_readonly_eq_mix_percent"] = round(numeric, 3)
            state["current_wing_eq_readback_available"] = True
        elif value_type == "eq_low_gain":
            numeric = _readback_float(value)
            if numeric is not None:
                state["wing_readonly_eq_low_gain_db"] = round(numeric, 3)
            state["current_wing_eq_readback_available"] = True
        elif value_type == "eq_high_gain":
            numeric = _readback_float(value)
            if numeric is not None:
                state["wing_readonly_eq_high_gain_db"] = round(numeric, 3)
            state["current_wing_eq_readback_available"] = True
        elif value_type == "dyn_on":
            state["current_wing_dynamics_enabled"] = _readback_bool(value)
            state["current_wing_dynamics_readback_available"] = True
        elif value_type == "dyn_threshold":
            numeric = _readback_float(value)
            if numeric is not None:
                state["wing_readonly_dynamics_threshold_dbfs"] = round(numeric, 3)
            state["current_wing_dynamics_readback_available"] = True
        elif value_type == "gate_model":
            state["current_wing_gate_model"] = str(value or "")
            state["current_wing_gate_slot_readback_available"] = True
        elif value_type == "gate_on":
            state["current_wing_gate_enabled"] = _readback_bool(value)
            state["current_wing_gate_slot_readback_available"] = True
        elif value_type == "gate_threshold":
            numeric = _readback_float(value)
            if numeric is not None:
                state["wing_readonly_gate_threshold_dbfs"] = round(numeric, 3)
        elif value_type == "gate_ratio":
            state["wing_readonly_gate_ratio"] = str(value or "")
        elif value_type == "gate_attack":
            numeric = _readback_float(value)
            if numeric is not None:
                state["wing_readonly_gate_attack_ms"] = round(numeric, 3)
        elif value_type == "gate_release":
            numeric = _readback_float(value)
            if numeric is not None:
                state["wing_readonly_gate_release_ms"] = round(numeric, 3)

    state["current_wing_state_readback_count"] = len(readbacks)
    state["current_wing_state_physical_readback_count"] = len(physical_readbacks)
    state["current_wing_state_channels"] = sorted(set(channels))
    state["current_wing_state_value_types"] = list(_ordered_unique(value_types))
    state["current_wing_state_raw_readback_by_value_type"] = raw_by_value_type
    state["current_wing_state_readback_available"] = bool(value_types)
    if (
        state.get("current_wing_fader_readback_available") is True
        or state.get("current_wing_gain_readback_available") is True
        or state.get("current_wing_pan_readback_available") is True
    ):
        state["current_wing_state_basic_controls_readback_available"] = True
    if state.get("current_wing_eq_readback_available") is True:
        state["current_wing_state_eq_readback_available"] = True
    if state.get("current_wing_dynamics_readback_available") is True:
        state["current_wing_state_dynamics_readback_available"] = True
    if state.get("current_wing_gate_slot_readback_available") is True:
        state["current_wing_state_gate_slot_readback_available"] = True
    return state


def _flatten_strings(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        flattened: List[str] = []
        for item in value.values():
            flattened.extend(_flatten_strings(item))
        return tuple(flattened)
    if isinstance(value, Iterable):
        flattened = []
        for item in value:
            flattened.extend(_flatten_strings(item))
        return tuple(flattened)
    return ()


_OPERATOR_OWNED_PATH_KEYS = (
    "operator_owned_osc_paths",
    "operator_owned_paths",
    "operator_locked_osc_paths",
    "locked_osc_paths",
    "manual_osc_paths",
    "manual_paths",
)

_WING_EQ_FILTER_OPERATOR_OWNED_PATH_SUFFIXES = (
    "flt/lc",
    "flt/lcf",
    "flt/lcs",
    "flt/hc",
    "flt/hcf",
    "flt/hcs",
    "eq/lg",
    "eq/lf",
    "eq/lq",
    "eq/leq",
)

_WING_EQ_OPERATOR_OWNED_PATH_SUFFIXES = (
    "eq/on",
    "eq/mdl",
    "eq/mix",
    "eq/lg",
    "eq/lf",
    "eq/lq",
    "eq/leq",
    "eq/{band}f",
    "eq/{band}g",
    "eq/{band}q",
    "eq/hg",
    "eq/hf",
    "eq/hq",
    "eq/heq",
)

_WING_DYNAMICS_OPERATOR_OWNED_PATH_SUFFIXES = _ordered_unique(
    (
        "dyn/mdl",
        "dyn/on",
        "dyn/thr",
        "dyn/thr21",
        *(
            f"dyn/{suffix}"
            for suffix in WING_DYNAMICS_THRESHOLD_PATH_SUFFIX_BY_SURFACE.values()
        ),
        "dyn/ratio",
        "dyn/att",
        "dyn/hld",
        "dyn/rel",
        "dyn/knee",
        "dyn/gain",
        "dyn/mix",
        "dyn/det",
        "dyn/env",
        "dyn/auto",
        "dynxo/type",
        "dynxo/depth",
        "dynxo/f",
        "dynsc/type",
        "dynsc/f",
        "dynsc/q",
        "dynsc/src",
        "dynsc/tap",
        "dyn/*",
    )
)

_WING_FX_LEVEL_OPERATOR_OWNED_PATH_SUFFIXES = (
    "send/{send}/lvl",
    "fxmix",
)

_WING_FX_MODEL_OPERATOR_OWNED_PATH_SUFFIXES = _ordered_unique(
    (
        WING_FX_MODEL_READBACK_PATH_SUFFIX,
        *(
            str(binding[0])
            for model_bindings in WING_FX_MODEL_PARAMETER_BINDINGS_BY_MODEL.values()
            for binding in model_bindings.values()
        ),
    )
)

_WING_GATE_SLOT_OPERATOR_OWNED_PATH_SUFFIXES = (
    "gate/mdl",
    "gate/on",
    "gate/thr",
    "gate/thr21",
    "gate/range",
    "gate/att",
    "gate/hld",
    "gate/rel",
    "gate/acc",
    "gate/ratio",
    "gate/filt",
    "gate/g",
    "gate/f",
    "gate/q",
    "gate/mode",
    "gate/hysteresis",
    "gate/hyst",
    "gatesc/type",
    "gatesc/f",
    "gatesc/q",
    "gatesc/src",
    "gatesc/tap",
)

_WING_PAN_WIDTH_OPERATOR_OWNED_PATH_SUFFIXES = ("pan", "wid")
_WING_MUTE_OPERATOR_OWNED_PATH_SUFFIXES = ("mute",)


def _wing_target_path_prefixes(target_id: str) -> Tuple[str, ...]:
    token = str(target_id or "").strip().lower().replace(" ", "")
    if token in {"main", "master", "main_lr", "lr"}:
        return ("/main/1", "/main/{main}")
    if ":" not in token:
        return ()
    surface, raw_index = token.split(":", 1)
    surface_aliases = {
        "ch": "ch",
        "channel": "ch",
        "aux": "aux",
        "bus": "bus",
        "main": "main",
        "master": "main",
        "main_lr": "main",
        "lr": "main",
        "mtx": "mtx",
        "matrix": "mtx",
        "fx": "fx",
    }
    surface = surface_aliases.get(surface, surface)
    if surface == "main" and raw_index in {"", "lr", "main", "master", "main_lr"}:
        return ("/main/1", "/main/{main}")
    try:
        index = int(raw_index)
    except ValueError:
        return ()
    placeholders = {
        "ch": "channel",
        "aux": "aux",
        "bus": "bus",
        "main": "main",
        "mtx": "matrix",
        "fx": "fx",
    }
    placeholder = placeholders.get(surface)
    concrete = f"/{surface}/{index}"
    if not placeholder:
        return (concrete,)
    return (concrete, f"/{surface}/{{{placeholder}}}")


def _wing_path_segments_match_suffix(
    path_segments: Sequence[str],
    suffix_segments: Sequence[str],
) -> bool:
    if len(path_segments) < len(suffix_segments):
        return False
    tail = path_segments[-len(suffix_segments) :]
    for path_segment, suffix_segment in zip(tail, suffix_segments):
        if suffix_segment == "*":
            continue
        if "{" in suffix_segment and "}" in suffix_segment:
            placeholder_start = suffix_segment.find("{")
            placeholder_end = suffix_segment.find("}", placeholder_start)
            prefix = suffix_segment[:placeholder_start]
            suffix = suffix_segment[placeholder_end + 1 :]
            inner = path_segment[len(prefix) : len(path_segment) - len(suffix) if suffix else None]
            if (
                placeholder_end <= placeholder_start + 1
                or not path_segment.startswith(prefix)
                or (suffix and not path_segment.endswith(suffix))
                or not inner
            ):
                return False
            continue
        if path_segment != suffix_segment:
            return False
    return True


def _wing_path_matches_suffix(path: str, suffix: str) -> bool:
    path_segments = [segment for segment in path.strip("/").split("/") if segment]
    suffix_segments = [segment for segment in str(suffix).strip("/").split("/") if segment]
    if not path_segments or not suffix_segments:
        return False
    return _wing_path_segments_match_suffix(path_segments, suffix_segments)


def _wing_path_is_suffix_only(path: str, suffix: str) -> bool:
    path_segments = [segment for segment in path.strip("/").split("/") if segment]
    suffix_segments = [segment for segment in str(suffix).strip("/").split("/") if segment]
    if len(path_segments) != len(suffix_segments):
        return False
    return _wing_path_segments_match_suffix(path_segments, suffix_segments)


def _wing_path_has_target_prefix(path: str, prefixes: Sequence[str]) -> bool:
    normalized = f"/{path.strip('/')}"
    for raw_prefix in prefixes:
        prefix = f"/{str(raw_prefix).strip('/')}"
        if normalized == prefix or normalized.startswith(f"{prefix}/"):
            return True
    return False


def _current_wing_operator_owned_from_state_paths(
    state: Mapping[str, Any],
    *,
    target_id: str,
    suffixes: Sequence[str],
) -> Optional[bool]:
    path_values: List[str] = []
    for key in _OPERATOR_OWNED_PATH_KEYS:
        path_values.extend(_flatten_strings(state.get(key)))
    if not path_values:
        return None

    prefixes = _wing_target_path_prefixes(target_id)
    for raw_path in path_values:
        path = str(raw_path or "").strip().lower().rstrip("/")
        if not path:
            continue
        if any(_wing_path_is_suffix_only(path, suffix) for suffix in suffixes):
            return True
        if prefixes and _wing_path_has_target_prefix(path, prefixes) and any(
            _wing_path_matches_suffix(path, suffix) for suffix in suffixes
        ):
            return True
    return None


def _current_wing_operator_owned_from_state(
    state: Mapping[str, Any],
    *,
    direct_keys: Sequence[str],
    suffixes: Sequence[str],
    target_ids: Sequence[str],
) -> Optional[bool]:
    direct = _bool_or_none(_mapping_first_present(state, direct_keys))
    if direct is not None:
        return direct
    for target_id in target_ids:
        if not str(target_id or "").strip():
            continue
        inferred = _current_wing_operator_owned_from_state_paths(
            state,
            target_id=target_id,
            suffixes=suffixes,
        )
        if inferred is True:
            return True
    return None


def _current_wing_eq_filter_operator_owned_from_state(
    state: Mapping[str, Any],
    *,
    target_id: str,
) -> Optional[bool]:
    return _current_wing_operator_owned_from_state(
        state,
        direct_keys=(
            "current_wing_eq_filter_operator_owned",
            "wing_eq_filter_operator_owned",
            "eq_filter_operator_owned",
            "operator_owned_eq_filter",
            "current_wing_filter_operator_owned",
        ),
        suffixes=_WING_EQ_FILTER_OPERATOR_OWNED_PATH_SUFFIXES,
        target_ids=(target_id,),
    )


def _current_wing_eq_operator_owned_from_state(
    state: Mapping[str, Any],
    *,
    target_id: str,
) -> Optional[bool]:
    return _current_wing_operator_owned_from_state(
        state,
        direct_keys=(
            "current_wing_eq_operator_owned",
            "wing_eq_operator_owned",
            "eq_operator_owned",
        ),
        suffixes=_WING_EQ_OPERATOR_OWNED_PATH_SUFFIXES,
        target_ids=(target_id,),
    )


def _current_wing_dynamics_operator_owned_from_state(
    state: Mapping[str, Any],
    *,
    target_id: str,
) -> Optional[bool]:
    return _current_wing_operator_owned_from_state(
        state,
        direct_keys=(
            "current_wing_dynamics_operator_owned",
            "wing_dynamics_operator_owned",
            "dynamics_operator_owned",
            "compressor_operator_owned",
        ),
        suffixes=_WING_DYNAMICS_OPERATOR_OWNED_PATH_SUFFIXES,
        target_ids=(target_id,),
    )


def _current_wing_fx_level_operator_owned_from_state(
    state: Mapping[str, Any],
    *,
    target_id: str,
    fx_return_target_id: str,
) -> Optional[bool]:
    return _current_wing_operator_owned_from_state(
        state,
        direct_keys=(
            "current_wing_fx_level_operator_owned",
            "wing_fx_level_operator_owned",
            "fx_level_operator_owned",
            "fx_send_or_return_level_operator_owned",
        ),
        suffixes=_WING_FX_LEVEL_OPERATOR_OWNED_PATH_SUFFIXES,
        target_ids=(target_id, fx_return_target_id),
    )


def _current_wing_fx_operator_owned_from_state(
    state: Mapping[str, Any],
    *,
    target_id: str,
) -> Optional[bool]:
    return _current_wing_operator_owned_from_state(
        state,
        direct_keys=(
            "current_wing_fx_operator_owned",
            "wing_fx_operator_owned",
            "fx_operator_owned",
        ),
        suffixes=_WING_FX_MODEL_OPERATOR_OWNED_PATH_SUFFIXES,
        target_ids=(target_id,),
    )


def _current_wing_gate_slot_operator_owned_from_state(
    state: Mapping[str, Any],
    *,
    target_id: str,
) -> Optional[bool]:
    return _current_wing_operator_owned_from_state(
        state,
        direct_keys=(
            "current_wing_gate_slot_operator_owned",
            "wing_gate_slot_operator_owned",
            "gate_slot_operator_owned",
        ),
        suffixes=_WING_GATE_SLOT_OPERATOR_OWNED_PATH_SUFFIXES,
        target_ids=(target_id,),
    )


def _current_wing_pan_width_operator_owned_from_state(
    state: Mapping[str, Any],
    *,
    target_id: str,
) -> Optional[bool]:
    return _current_wing_operator_owned_from_state(
        state,
        direct_keys=(
            "current_wing_pan_width_operator_owned",
            "wing_pan_width_operator_owned",
            "pan_width_operator_owned",
            "pan_operator_owned",
            "width_operator_owned",
        ),
        suffixes=_WING_PAN_WIDTH_OPERATOR_OWNED_PATH_SUFFIXES,
        target_ids=(target_id,),
    )


def _current_wing_mute_operator_owned_from_state(
    state: Mapping[str, Any],
    *,
    target_id: str,
) -> Optional[bool]:
    return _current_wing_operator_owned_from_state(
        state,
        direct_keys=(
            "current_wing_mute_operator_owned",
            "wing_mute_operator_owned",
            "mute_operator_owned",
        ),
        suffixes=_WING_MUTE_OPERATOR_OWNED_PATH_SUFFIXES,
        target_ids=(target_id,),
    )


def _current_wing_gain_operator_owned_from_state(
    state: Mapping[str, Any],
    *,
    target_id: str,
) -> Optional[bool]:
    return _current_wing_operator_owned_from_state(
        state,
        direct_keys=(
            "current_wing_gain_operator_owned",
            "wing_gain_operator_owned",
            "gain_operator_owned",
            "fader_operator_owned",
            "trim_operator_owned",
        ),
        suffixes=WING_GAIN_WRITE_PATH_SUFFIXES,
        target_ids=(target_id,),
    )


def _read_only_freshness(
    source: str = SMART_DSP_EVIDENCE_EXTRACTOR_VERSION,
    *,
    field_count: int = 12,
) -> Dict[str, Any]:
    measured_field_count = max(0, int(field_count))
    return {
        "status": "fresh",
        "source": source,
        "field_count": measured_field_count,
        "fresh_field_count": measured_field_count,
        "stale_field_count": 0,
        "missing_field_count": 0,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }


def _current_wing_state_rejected(state: Mapping[str, Any]) -> bool:
    return _resolve_bool_state(
        None,
        state,
        ("current_wing_state_rejected", "wing_state_rejected"),
    )


def _current_wing_state_readback_available(state: Mapping[str, Any]) -> bool:
    if _current_wing_state_rejected(state):
        return False
    for key in (
        "current_wing_state_readback_available",
        "current_wing_state_gate_slot_readback_available",
        "current_wing_gate_slot_readback_available",
        "current_wing_eq_model_readback_available",
        "current_wing_dynamics_model_readback_available",
        "current_wing_fx_model_readback_available",
    ):
        if _bool_or_none(state.get(key)) is True:
            return True
    return _safe_float(state.get("current_wing_state_physical_readback_count")) > 0.0


def _current_wing_state_rollback_plan_available(state: Mapping[str, Any]) -> bool:
    if _current_wing_state_rejected(state):
        return False
    return _resolve_bool_state(
        None,
        state,
        (
            "current_wing_state_rollback_plan_available",
            "rollback_plan_available",
            "wing_rollback_plan_available",
        ),
    )


def _freshness_from_current_wing_state(
    state: Mapping[str, Any],
    *,
    fallback_field_count: int,
) -> Optional[Dict[str, Any]]:
    if state.get("current_wing_state_source") != "wing_readonly_preflight_artifact":
        return None

    field_count = int(
        max(
            _safe_float(state.get("current_wing_state_physical_readback_count")),
            _safe_float(state.get("current_wing_state_readback_count")),
            float(max(0, fallback_field_count)),
        )
    )
    freshness = _read_only_freshness(
        "wing_readonly_preflight_artifact",
        field_count=field_count,
    )
    if _current_wing_state_rejected(state):
        freshness["status"] = "rejected"
        freshness["stale_field_count"] = field_count
        freshness["fresh_field_count"] = 0
        freshness["missing_field_count"] = field_count
    elif not _current_wing_state_readback_available(state):
        freshness["status"] = "missing_readback"
        freshness["stale_field_count"] = 0
        freshness["fresh_field_count"] = 0
        freshness["missing_field_count"] = field_count
    return freshness


def _resolve_ayaic_target_lufs(source_role: str, source_family: str) -> float:
    text = str(source_role or "").strip().lower().replace("-", "_").replace(" ", "_")
    if source_family == "master":
        return -18.0
    if source_family == "fx_return":
        return -35.0
    if source_family in {"keys", "playback"}:
        return -30.0
    if "backing" in text or "bg" in text or "double" in text:
        return -27.0
    if source_family == "vocal" and ("lead" in text or "main" in text):
        return -22.0
    if source_family in {"kick", "snare", "tom", "drum", "bass", "guitar"}:
        return -25.0
    return -25.0


def _source_text(source_role: str, source_label: str = "") -> str:
    return f"{source_role or ''} {source_label or ''}".strip().lower().replace("-", "_").replace(" ", "_")


def _is_generic_source_role(source_role: str) -> bool:
    text = str(source_role or "").strip().lower().replace("-", "_").replace(" ", "_")
    return text in {"", "unknown", "source", "channel", "ch", "input", "mic", "line", "instrument"}


def _is_group_or_stem_source(source_role: str, source_label: str = "", target_id: str = "") -> bool:
    text = _source_text(f"{target_id} {source_role}", source_label)
    group_tokens = ("bus", "group", "subgroup", "stem", "dca", "aux")
    master_tokens = ("master", "main_lr", "main_out", "main_pa", "pa_output", "reference", "capture")
    return any(token in text for token in group_tokens) and not any(
        token in text for token in master_tokens
    )


def _is_cymbal_or_overhead_source(source_role: str, source_label: str = "") -> bool:
    text = _source_text(source_role, source_label)
    return any(
        token in text
        for token in ("overhead", "cymbal", "hat", "ride", "crash", "oh")
    )


def _infer_source_identity(
    *,
    source_role: str,
    source_label: str,
    declared_family: str,
    band_share: Mapping[str, float],
    spectral_centroid_hz: float,
    dominant_frequency_hz: float,
    transient: Mapping[str, float],
    crest_factor_db: float,
    lra_proxy_db: float,
    tail_persistence: float,
    channel_count: int,
) -> Dict[str, Any]:
    combined_family = classify_source_family(_source_text(source_role, source_label))
    if declared_family != "unknown" and not _is_generic_source_role(source_role):
        return {
            "effective_source_role": str(source_role),
            "effective_source_family": declared_family,
            "declared_source_family": declared_family,
            "inferred_source_role": str(source_role),
            "inferred_source_family": declared_family,
            "source_identity_inference_confidence": 0.95,
            "source_identity_inference_reason": "declared_source_role",
            "source_identity_inference_used": False,
        }
    if combined_family != "unknown":
        return {
            "effective_source_role": str(source_label or source_role or combined_family),
            "effective_source_family": combined_family,
            "declared_source_family": declared_family,
            "inferred_source_role": str(source_label or source_role or combined_family),
            "inferred_source_family": combined_family,
            "source_identity_inference_confidence": 0.86,
            "source_identity_inference_reason": "source_label_token_match",
            "source_identity_inference_used": True,
        }

    sub_bass = float(band_share.get("sub", 0.0)) + float(band_share.get("bass", 0.0))
    low_mid = float(band_share.get("low_mid", 0.0))
    mid = float(band_share.get("mid", 0.0))
    presence = float(band_share.get("presence", 0.0))
    sibilance = float(band_share.get("sibilance", 0.0))
    air = float(band_share.get("air", 0.0))
    transient_density = _safe_float(transient.get("transient_density"))
    transient_strength = _safe_float(transient.get("transient_strength"))

    candidates: List[Tuple[str, str, float, str]] = []
    if sub_bass >= 0.55 and 45.0 <= dominant_frequency_hz <= 130.0:
        kick_score = 0.46 + 0.28 * transient_strength + 0.18 * _clip01(crest_factor_db / 18.0)
        bass_score = 0.48 + 0.26 * (1.0 - _clip01(transient_density / 4.0)) + 0.16 * sub_bass
        if transient_density >= 0.45 or crest_factor_db >= 11.0:
            candidates.append(("kick_inferred", "kick", kick_score, "low_end_transient_dominant"))
        candidates.append(("bass_inferred", "bass", bass_score, "sustained_low_end_dominant"))
    if sibilance >= 0.08 or (presence + sibilance) >= 0.28:
        vocal_score = 0.52 + 0.22 * _clip01((presence + sibilance) * 2.0) + 0.14 * _clip01(lra_proxy_db / 18.0)
        candidates.append(("lead_vocal_inferred", "vocal", vocal_score, "presence_sibilance_vocal_signature"))
    if 80.0 <= dominant_frequency_hz <= 280.0 and transient_density >= 0.6 and tail_persistence >= 0.15:
        candidates.append(("tom_inferred", "tom", 0.58 + 0.18 * transient_strength, "drum_tone_tail_transients"))
    if channel_count >= 2 and spectral_centroid_hz >= 1200.0 and tail_persistence >= 0.45:
        candidates.append(("fx_return_inferred", "fx_return", 0.58 + 0.2 * tail_persistence, "stereo_tail_fx_return_signature"))
    if channel_count >= 2 and sub_bass >= 0.25 and mid >= 0.08 and presence >= 0.04:
        candidates.append(("master_inferred", "master", 0.58 + 0.12 * _clip01((sub_bass + mid + presence) * 1.5), "wideband_stereo_mix_signature"))
    if not candidates and (mid + presence) >= 0.45:
        candidates.append(("keys_inferred", "keys", 0.56 + 0.12 * _clip01(mid + presence), "midrange_harmonic_support_signature"))

    if not candidates:
        return {
            "effective_source_role": str(source_role),
            "effective_source_family": "unknown",
            "declared_source_family": declared_family,
            "inferred_source_role": "",
            "inferred_source_family": "",
            "source_identity_inference_confidence": 0.0,
            "source_identity_inference_reason": "insufficient_source_identity_evidence",
            "source_identity_inference_used": False,
        }

    inferred_role, inferred_family, confidence, reason = max(candidates, key=lambda item: item[2])
    confidence = round(_clip(confidence, 0.0, 0.92), 3)
    return {
        "effective_source_role": inferred_role,
        "effective_source_family": inferred_family,
        "declared_source_family": declared_family,
        "inferred_source_role": inferred_role,
        "inferred_source_family": inferred_family,
        "source_identity_inference_confidence": confidence,
        "source_identity_inference_reason": reason,
        "source_identity_inference_used": confidence >= 0.6,
    }


def _acoustic_source_identity_probe(
    *,
    band_share: Mapping[str, float],
    spectral_centroid_hz: float,
    dominant_frequency_hz: float,
    transient: Mapping[str, float],
    crest_factor_db: float,
    lra_proxy_db: float,
    tail_persistence: float,
    channel_count: int,
) -> Dict[str, Any]:
    sub_bass = float(band_share.get("sub", 0.0)) + float(band_share.get("bass", 0.0))
    low_mid = float(band_share.get("low_mid", 0.0))
    mid = float(band_share.get("mid", 0.0))
    presence = float(band_share.get("presence", 0.0))
    sibilance = float(band_share.get("sibilance", 0.0))
    transient_density = _safe_float(transient.get("transient_density"))
    transient_strength = _safe_float(transient.get("transient_strength"))

    candidates: List[Tuple[str, float, str]] = []
    if sub_bass >= 0.55 and 45.0 <= dominant_frequency_hz <= 130.0:
        kick_score = 0.46 + 0.28 * transient_strength + 0.18 * _clip01(crest_factor_db / 18.0)
        bass_score = 0.48 + 0.26 * (1.0 - _clip01(transient_density / 4.0)) + 0.16 * sub_bass
        if transient_density >= 0.45 or crest_factor_db >= 11.0:
            candidates.append(("kick", kick_score, "low_end_transient_dominant"))
        candidates.append(("bass", bass_score, "sustained_low_end_dominant"))
    if sibilance >= 0.08 or (presence + sibilance) >= 0.28:
        vocal_score = 0.52 + 0.22 * _clip01((presence + sibilance) * 2.0)
        vocal_score += 0.14 * _clip01(lra_proxy_db / 18.0)
        candidates.append(("vocal", vocal_score, "presence_sibilance_vocal_signature"))
    if 80.0 <= dominant_frequency_hz <= 280.0 and transient_density >= 0.6 and tail_persistence >= 0.15:
        candidates.append(("tom", 0.58 + 0.18 * transient_strength, "drum_tone_tail_transients"))
    if channel_count >= 2 and spectral_centroid_hz >= 1200.0 and tail_persistence >= 0.45:
        candidates.append(("fx_return", 0.58 + 0.2 * tail_persistence, "stereo_tail_fx_return_signature"))
    if channel_count >= 2 and sub_bass >= 0.25 and mid >= 0.08 and presence >= 0.04:
        mix_score = 0.58 + 0.12 * _clip01((sub_bass + mid + presence) * 1.5)
        candidates.append(("master", mix_score, "wideband_stereo_mix_signature"))
    if not candidates and (mid + presence) >= 0.45:
        candidates.append(("keys", 0.56 + 0.12 * _clip01(mid + presence), "midrange_harmonic_support_signature"))

    if not candidates:
        return {"family": "", "confidence": 0.0, "reason": "insufficient_audio_identity_evidence"}

    family, confidence, reason = max(candidates, key=lambda item: item[1])
    return {
        "family": family,
        "confidence": round(_clip(confidence, 0.0, 0.92), 3),
        "reason": reason,
    }


def _source_family_mismatch_score(
    *,
    source_role: str,
    source_label: str,
    target_id: str,
    declared_family: str,
    acoustic_probe: Mapping[str, Any],
) -> Dict[str, Any]:
    declared = str(declared_family or "unknown")
    acoustic_family = str(acoustic_probe.get("family") or "")
    acoustic_confidence = _clip01(acoustic_probe.get("confidence"))
    if (
        declared == "unknown"
        or _is_generic_source_role(source_role)
        or not acoustic_family
        or acoustic_confidence < SOURCE_IDENTITY_MISMATCH_BLOCK_THRESHOLD
    ):
        return {"score": 0.0, "family": "", "reason": ""}
    if _is_group_or_stem_source(source_role, source_label, target_id) or declared in {
        "master",
        "fx_return",
    }:
        return {"score": 0.0, "family": "", "reason": ""}

    compatible_families = (
        {"kick", "snare", "tom", "drum"},
        {"keys", "playback"},
    )
    if declared == acoustic_family or any(
        declared in family_set and acoustic_family in family_set
        for family_set in compatible_families
    ):
        return {"score": 0.0, "family": "", "reason": ""}

    low_end_or_drum = {"kick", "bass", "tom", "drum"}
    hard_conflict = (
        declared in {"vocal", "guitar", "keys", "playback"}
        and acoustic_family in low_end_or_drum
    ) or (
        declared in low_end_or_drum
        and acoustic_family in {"vocal", "fx_return", "master"}
    )
    if (
        declared == "drum"
        and acoustic_family == "vocal"
        and _is_cymbal_or_overhead_source(source_role, source_label)
        and str(acoustic_probe.get("reason") or "") == "presence_sibilance_vocal_signature"
    ):
        hard_conflict = False
    if not hard_conflict:
        return {"score": 0.0, "family": "", "reason": ""}

    severe_pair = (
        declared == "vocal" and acoustic_family in low_end_or_drum
    ) or (
        acoustic_family == "vocal" and declared in low_end_or_drum
    )
    score = _clip(acoustic_confidence + (0.08 if severe_pair else 0.0), 0.0, 0.98)
    return {
        "score": round(score, 3),
        "family": acoustic_family,
        "reason": (
            f"declared_{declared}_conflicts_with_audio_{acoustic_family}:"
            f"{acoustic_probe.get('reason') or 'audio_signature'}"
        ),
    }


def _as_audio_array(samples: Any) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float64)
    if array.size == 0:
        return np.zeros(1, dtype=np.float64)
    if array.ndim > 2:
        raise ValueError("smart DSP evidence accepts mono or two-dimensional audio arrays")
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(array, -8.0, 8.0)


def _mono_and_channel_count(samples: Any) -> Tuple[np.ndarray, int]:
    array = _as_audio_array(samples)
    if array.ndim == 1:
        return array.astype(np.float64), 1
    if array.shape[0] == 0:
        return np.zeros(1, dtype=np.float64), int(max(array.shape[1], 1))
    channels = int(array.shape[1])
    return np.mean(array, axis=1).astype(np.float64), channels


def _level_signal(samples: Any) -> np.ndarray:
    array = _as_audio_array(samples)
    if array.ndim == 1:
        return array.astype(np.float64)
    if array.shape[0] == 0:
        return np.zeros(1, dtype=np.float64)
    return np.sqrt(np.mean(array.astype(np.float64) ** 2, axis=1))


def _true_peak_dbfs(samples: Any, oversample: int = 4) -> float:
    array = _as_audio_array(samples)
    if array.ndim == 1:
        channels = [array]
    else:
        channels = [array[:, channel] for channel in range(array.shape[1])]
    peaks = []
    for channel in channels:
        if channel.size <= 1:
            peaks.append(float(np.max(np.abs(channel))) if channel.size else 0.0)
            continue
        count = int(channel.size)
        source_x = np.arange(count, dtype=np.float64)
        interp_x = np.linspace(0.0, float(count - 1), max(count * oversample, count))
        interpolated = np.interp(interp_x, source_x, channel)
        peaks.append(float(np.max(np.abs(interpolated))))
    return round(_dbfs(max(peaks) if peaks else 0.0), 2)


def _frame_rms_dbfs(
    mono: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 100.0,
    hop_ms: float = 50.0,
) -> np.ndarray:
    frame = max(32, int(sample_rate * frame_ms / 1000.0))
    hop = max(16, int(sample_rate * hop_ms / 1000.0))
    if mono.size < frame:
        rms = math.sqrt(float(np.mean(mono * mono)) + EPSILON)
        return np.asarray([_dbfs(rms)], dtype=np.float64)
    values = []
    for start in range(0, mono.size - frame + 1, hop):
        window = mono[start : start + frame]
        values.append(_dbfs(math.sqrt(float(np.mean(window * window)) + EPSILON)))
    return np.asarray(values or [-120.0], dtype=np.float64)


def _spectral_profile(mono: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    if mono.size == 0:
        mono = np.zeros(1, dtype=np.float64)
    max_samples = int(max(sample_rate * 8, 2048))
    segment = mono[:max_samples]
    if segment.size < 2048:
        segment = np.pad(segment, (0, 2048 - segment.size))
    segment = segment - float(np.mean(segment))
    window = np.hanning(segment.size)
    spectrum = np.abs(np.fft.rfft(segment * window)) ** 2
    freqs = np.fft.rfftfreq(segment.size, 1.0 / float(sample_rate))
    valid = (freqs >= 20.0) & (freqs <= min(20000.0, sample_rate / 2.0))
    total_power = float(np.sum(spectrum[valid]) + EPSILON)

    band_power: Dict[str, float] = {}
    band_share: Dict[str, float] = {}
    band_dbfs: Dict[str, float] = {}
    for name, low_hz, high_hz in SPECTRAL_BANDS:
        mask = (freqs >= low_hz) & (freqs < min(high_hz, sample_rate / 2.0))
        power = float(np.sum(spectrum[mask]))
        band_power[name] = power
        band_share[name] = round(power / total_power, 6)
        band_dbfs[name] = round(_power_db(power / max(segment.size * segment.size, 1)), 2)

    if total_power > EPSILON:
        centroid_hz = float(np.sum(freqs[valid] * spectrum[valid]) / total_power)
    else:
        centroid_hz = 0.0

    dominant_mask = valid & (freqs <= 12000.0)
    dominant_frequency_hz = 0.0
    if np.any(dominant_mask):
        indices = np.where(dominant_mask)[0]
        dominant_index = int(indices[int(np.argmax(spectrum[dominant_mask]))])
        dominant_frequency_hz = float(freqs[dominant_index])

    return {
        "band_power": band_power,
        "band_share": band_share,
        "band_energy_dbfs": band_dbfs,
        "spectral_centroid_hz": round(centroid_hz, 1),
        "dominant_frequency_hz": round(dominant_frequency_hz, 1),
        "frequency_resolution_hz": round(float(freqs[1] - freqs[0]) if freqs.size > 1 else 0.0, 3),
    }


def _band_ratio_db(band_share: Mapping[str, float], numerator: Sequence[str], denominator: Sequence[str]) -> float:
    num = sum(float(band_share.get(key, 0.0)) for key in numerator) + EPSILON
    den = sum(float(band_share.get(key, 0.0)) for key in denominator) + EPSILON
    return 10.0 * math.log10(num / den)


def _presence_lift_evidence(
    *,
    source_family: str,
    spectral_centroid_hz: float,
    presence_air_share: float,
    high_ratio_db: float,
    presence_deficit_score: float,
) -> Dict[str, Any]:
    if _clip01(presence_deficit_score) < 0.55:
        return {
            "presence_lift_frequency_hz": None,
            "presence_lift_bandwidth_hz": 0.0,
            "presence_lift_parameter_source": "",
        }

    centroid_hz = _safe_float(spectral_centroid_hz)
    deficit_depth = _clip((0.18 - _safe_float(presence_air_share)) / 0.18, 0.0, 1.0)
    ratio_depth = _clip(max(0.0, -_safe_float(high_ratio_db)) / 18.0, 0.0, 1.0)
    if source_family == "vocal":
        frequency_hz = _clip(
            centroid_hz * 2.9 + 260.0 * deficit_depth + 180.0 * ratio_depth,
            3000.0,
            6200.0,
        )
        bandwidth_hz = _clip(
            2800.0 + 1900.0 * deficit_depth + 900.0 * ratio_depth,
            2200.0,
            5800.0,
        )
    elif source_family == "guitar":
        frequency_hz = _clip(
            centroid_hz * 2.35 + 180.0 * deficit_depth + 120.0 * ratio_depth,
            2200.0,
            5200.0,
        )
        bandwidth_hz = _clip(
            1100.0 + 1200.0 * deficit_depth + 500.0 * ratio_depth,
            900.0,
            3600.0,
        )
    else:
        frequency_hz = _clip(
            centroid_hz * 2.6 + 220.0 * deficit_depth + 140.0 * ratio_depth,
            2400.0,
            6200.0,
        )
        bandwidth_hz = _clip(
            1300.0 + 1400.0 * deficit_depth + 650.0 * ratio_depth,
            1000.0,
            4200.0,
        )

    return {
        "presence_lift_frequency_hz": round(frequency_hz, 1),
        "presence_lift_bandwidth_hz": round(bandwidth_hz, 1),
        "presence_lift_parameter_source": "measured_presence_air_deficit",
    }


def _peak_frequency_in_band(mono: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> Optional[float]:
    if mono.size < 32:
        return None
    segment = mono[: int(max(sample_rate * 4, 2048))]
    if segment.size < 2048:
        segment = np.pad(segment, (0, 2048 - segment.size))
    segment = segment - float(np.mean(segment))
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(segment.size))) ** 2
    freqs = np.fft.rfftfreq(segment.size, 1.0 / float(sample_rate))
    mask = (freqs >= low_hz) & (freqs <= min(high_hz, sample_rate / 2.0))
    if not np.any(mask):
        return None
    indices = np.where(mask)[0]
    peak_index = int(indices[int(np.argmax(spectrum[mask]))])
    return round(float(freqs[peak_index]), 1)


def _peak_band_details_in_band(
    mono: np.ndarray,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
) -> Dict[str, Optional[float]]:
    if mono.size < 32:
        return {"peak_frequency_hz": None, "bandwidth_hz": None}
    segment = mono[: int(max(sample_rate * 4, 2048))]
    if segment.size < 2048:
        segment = np.pad(segment, (0, 2048 - segment.size))
    segment = segment - float(np.mean(segment))
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(segment.size))) ** 2
    freqs = np.fft.rfftfreq(segment.size, 1.0 / float(sample_rate))
    mask = (freqs >= low_hz) & (freqs <= min(high_hz, sample_rate / 2.0))
    if not np.any(mask):
        return {"peak_frequency_hz": None, "bandwidth_hz": None}

    band_freqs = freqs[mask]
    band_power = spectrum[mask]
    total_power = float(np.sum(band_power))
    peak_power = float(np.max(band_power))
    if total_power <= EPSILON or peak_power <= EPSILON:
        return {"peak_frequency_hz": None, "bandwidth_hz": None}

    peak_frequency = round(float(band_freqs[int(np.argmax(band_power))]), 1)
    power_threshold = peak_power * (10.0 ** (-18.0 / 10.0))
    significant_freqs = band_freqs[band_power >= power_threshold]
    if significant_freqs.size >= 2:
        bandwidth = float(significant_freqs[-1] - significant_freqs[0])
    else:
        resolution = float(band_freqs[1] - band_freqs[0]) if band_freqs.size > 1 else 0.0
        bandwidth = max(resolution, 0.0)
    return {
        "peak_frequency_hz": peak_frequency,
        "bandwidth_hz": round(float(bandwidth), 1),
    }


def _low_band_burst_evidence(mono: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    empty = {
        "plosive_risk_score": 0.0,
        "plosive_frequency_hz": None,
        "plosive_event_peak_dbfs": 0.0,
        "plosive_event_contrast_db": 0.0,
        "plosive_event_rise_time_ms": 0.0,
        "plosive_event_recovery_ms": 0.0,
    }
    if mono.size < 2048:
        return dict(empty)
    frame = max(512, int(sample_rate * 0.04))
    hop = max(256, int(sample_rate * 0.02))
    if mono.size < frame:
        return dict(empty)

    ratio_values: List[float] = []
    share_values: List[float] = []
    low_power_values: List[float] = []
    frequency_values: List[float] = []
    for start in range(0, mono.size - frame + 1, hop):
        segment = mono[start : start + frame] - float(np.mean(mono[start : start + frame]))
        spectrum = np.abs(np.fft.rfft(segment * np.hanning(frame))) ** 2
        freqs = np.fft.rfftfreq(frame, 1.0 / float(sample_rate))
        low_mask = (freqs >= 45.0) & (freqs <= min(160.0, sample_rate / 2.0))
        body_mask = (freqs >= 500.0) & (freqs <= min(5000.0, sample_rate / 2.0))
        full_mask = (freqs >= 45.0) & (freqs <= min(12000.0, sample_rate / 2.0))
        if not np.any(low_mask) or not np.any(full_mask):
            continue
        low_power = float(np.sum(spectrum[low_mask]))
        body_power = float(np.sum(spectrum[body_mask]))
        full_power = float(np.sum(spectrum[full_mask]))
        low_power_values.append(low_power)
        ratio_values.append(10.0 * math.log10((low_power + EPSILON) / (body_power + EPSILON)))
        share_values.append(low_power / (full_power + EPSILON))
        low_indices = np.where(low_mask)[0]
        peak_index = int(low_indices[int(np.argmax(spectrum[low_mask]))])
        frequency_values.append(float(freqs[peak_index]))

    if not ratio_values or not share_values:
        return dict(empty)

    ratios = np.asarray(ratio_values, dtype=np.float64)
    shares = np.asarray(share_values, dtype=np.float64)
    low_powers = np.asarray(low_power_values, dtype=np.float64)
    low_dbfs_frames = np.asarray(
        [_power_db(float(value) / max(frame * frame, 1)) for value in low_powers],
        dtype=np.float64,
    )
    peak_ratio_db = float(np.percentile(ratios, 95))
    median_ratio_db = float(np.percentile(ratios, 50))
    peak_share = float(np.percentile(shares, 95))
    active_low_frame_share = float(np.mean(shares > 0.35))
    burst_contrast_db = max(0.0, peak_ratio_db - median_ratio_db)
    low_peak_dbfs = float(np.max(low_dbfs_frames))
    median_low_dbfs = float(np.percentile(low_dbfs_frames, 50))
    event_contrast_db = max(0.0, low_peak_dbfs - median_low_dbfs)
    peak_frame_index = int(np.argmax(low_powers))
    frequency_hz = round(float(frequency_values[peak_frame_index]), 1)
    if burst_contrast_db < 10.0 or low_peak_dbfs < -42.0 or frequency_hz > 140.0:
        return dict(empty)
    pre_peak = low_dbfs_frames[: peak_frame_index + 1]
    rise_candidates = np.where(pre_peak <= median_low_dbfs + 3.0)[0]
    if rise_candidates.size:
        rise_start = int(rise_candidates[-1])
    else:
        rise_start = max(0, peak_frame_index - 1)
    rise_time_ms = max(20.0, (peak_frame_index - rise_start) * (hop / sample_rate) * 1000.0)
    post_peak = low_dbfs_frames[peak_frame_index:]
    recovery_candidates = np.where(post_peak <= median_low_dbfs + 3.0)[0]
    if recovery_candidates.size:
        recovery_ms = max(20.0, int(recovery_candidates[0]) * (hop / sample_rate) * 1000.0)
    else:
        recovery_ms = max(20.0, (post_peak.size - 1) * (hop / sample_rate) * 1000.0)
    score = _clip01(
        0.16
        + max(0.0, peak_ratio_db - 1.0) / 14.0
        + burst_contrast_db / 18.0
        + max(0.0, peak_share - 0.35) * 1.1
        + max(0.0, low_peak_dbfs + 44.0) / 24.0
        - max(0.0, active_low_frame_share - 0.45) * 0.55
    )
    return {
        "plosive_risk_score": round(score, 3),
        "plosive_frequency_hz": frequency_hz if score >= 0.25 else None,
        "plosive_event_peak_dbfs": round(_clip(low_peak_dbfs, -120.0, 12.0), 1),
        "plosive_event_contrast_db": round(_clip(event_contrast_db, 0.0, 60.0), 1),
        "plosive_event_rise_time_ms": round(_clip(rise_time_ms, 0.0, 500.0), 1),
        "plosive_event_recovery_ms": round(_clip(recovery_ms, 0.0, 900.0), 1),
    }


def _low_end_resonance_evidence(
    mono: np.ndarray,
    sample_rate: int,
    *,
    source_family: str,
) -> Dict[str, float]:
    empty = {
        "low_end_resonance_frequency_hz": 0.0,
        "low_end_resonance_bandwidth_hz": 0.0,
        "low_end_resonance_peak_dbfs": 0.0,
        "low_end_resonance_sustain_ms": 0.0,
    }
    if source_family not in {"bass", "kick"} or mono.size < 2048:
        return dict(empty)
    high_hz = 190.0 if source_family == "kick" else 180.0
    band = _peak_band_details_in_band(mono, sample_rate, 45.0, high_hz)
    frequency = float(band["peak_frequency_hz"] or 0.0)
    bandwidth = float(band["bandwidth_hz"] or 0.0)
    if frequency <= 0.0:
        return dict(empty)

    low_band = _band_limited_signal(mono, sample_rate, 45.0, high_hz)
    frame_rms = _frame_rms_dbfs(low_band, sample_rate, frame_ms=40.0, hop_ms=20.0)
    active = frame_rms[frame_rms > -100.0]
    if active.size < 3:
        return dict(empty)
    peak_dbfs = float(np.max(active))
    body_dbfs = float(np.percentile(active, 35))
    sustain_threshold = min(max(body_dbfs + 3.0, peak_dbfs - 18.0), peak_dbfs - 1.0)
    sustained_frames = np.where(frame_rms >= sustain_threshold)[0]
    sustain_ms = float(sustained_frames.size * 20.0) if sustained_frames.size else 0.0
    return {
        "low_end_resonance_frequency_hz": round(_clip(frequency, 45.0, high_hz), 1),
        "low_end_resonance_bandwidth_hz": round(_clip(bandwidth, 0.0, 180.0), 1),
        "low_end_resonance_peak_dbfs": round(_clip(peak_dbfs, -120.0, 12.0), 1),
        "low_end_resonance_sustain_ms": round(_clip(sustain_ms, 0.0, 2000.0), 1),
    }


def _spectral_flatness(mono: np.ndarray, sample_rate: int) -> float:
    if mono.size == 0:
        return 0.0
    segment = mono.astype(np.float64)
    if segment.size < 2048:
        segment = np.pad(segment, (0, 2048 - segment.size))
    segment = segment - float(np.mean(segment))
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(segment.size))) ** 2
    freqs = np.fft.rfftfreq(segment.size, 1.0 / float(sample_rate))
    valid = (freqs >= 120.0) & (freqs <= min(12000.0, sample_rate / 2.0))
    power = spectrum[valid] + EPSILON
    if power.size == 0:
        return 0.0
    return round(float(np.exp(np.mean(np.log(power))) / np.mean(power)), 6)


def _vocal_phrase_noise_floor_evidence(
    mono: np.ndarray,
    sample_rate: int,
    frame_rms_dbfs: np.ndarray,
) -> Dict[str, Any]:
    if frame_rms_dbfs.size < 8:
        return {
            "vocal_noise_floor_score": 0.0,
            "phrase_gap_noise_floor_dbfs": None,
            "phrase_gap_noise_flatness": 0.0,
            "phrase_gap_contrast_db": 0.0,
            "phrase_gap_median_ms": 0.0,
            "phrase_gap_coverage_ratio": 0.0,
        }

    floor_dbfs = float(np.percentile(frame_rms_dbfs, 10))
    phrase_dbfs = float(np.percentile(frame_rms_dbfs, 90))
    contrast_db = max(0.0, phrase_dbfs - floor_dbfs)
    frame = max(32, int(sample_rate * 100.0 / 1000.0))
    hop = max(16, int(sample_rate * 50.0 / 1000.0))
    quiet_threshold_dbfs = floor_dbfs + 2.5
    quiet_segments: List[np.ndarray] = []
    quiet_flags: List[bool] = []
    for start in range(0, mono.size - frame + 1, hop):
        window = mono[start : start + frame]
        rms = math.sqrt(float(np.mean(window * window)) + EPSILON)
        is_quiet = _dbfs(rms) <= quiet_threshold_dbfs
        quiet_flags.append(is_quiet)
        if is_quiet and len(quiet_segments) < 24:
            quiet_segments.append(window)
    if not quiet_segments:
        quiet_mono = mono[: min(mono.size, frame)]
    else:
        quiet_mono = np.concatenate(quiet_segments)
    quiet_profile = _spectral_profile(quiet_mono, sample_rate)
    quiet_share = quiet_profile["band_share"]
    flatness = _spectral_flatness(quiet_mono, sample_rate)
    quiet_coverage = (
        float(sum(1 for value in quiet_flags if value)) / float(len(quiet_flags))
        if quiet_flags
        else 0.0
    )
    quiet_run_lengths = []
    current_run = 0
    for is_quiet in quiet_flags:
        if is_quiet:
            current_run += 1
            continue
        if current_run:
            quiet_run_lengths.append(current_run)
            current_run = 0
    if current_run:
        quiet_run_lengths.append(current_run)
    phrase_gap_median_ms = (
        float(np.median(quiet_run_lengths)) * float(hop) * 1000.0 / float(sample_rate)
        if quiet_run_lengths
        else 0.0
    )
    broad_high_share = (
        float(quiet_share.get("presence", 0.0))
        + float(quiet_share.get("sibilance", 0.0))
        + float(quiet_share.get("air", 0.0))
    )
    flatness_score = _clip01((flatness - 0.04) / 0.22)
    broadband_score = _clip01((broad_high_share - 0.08) / 0.42)
    floor_score = _clip01((floor_dbfs + 66.0) / 18.0)
    contrast_score = _clip01((contrast_db - 8.0) / 18.0)
    score = _clip01(
        0.36 * floor_score
        + 0.24 * contrast_score
        + 0.28 * flatness_score
        + 0.12 * broadband_score
    )
    if floor_dbfs <= -68.0 or contrast_db < 6.0 or flatness < 0.08:
        score = 0.0
    return {
        "vocal_noise_floor_score": round(score, 3),
        "phrase_gap_noise_floor_dbfs": round(floor_dbfs, 2),
        "phrase_gap_noise_flatness": round(flatness, 6),
        "phrase_gap_contrast_db": round(contrast_db, 2),
        "phrase_gap_median_ms": round(_clip(phrase_gap_median_ms, 0.0, 2000.0), 1),
        "phrase_gap_coverage_ratio": round(_clip01(quiet_coverage), 3),
    }


def _line_noise_evidence(mono: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    if mono.size < 2048:
        return {
            "line_noise_score": 0.0,
            "line_noise_frequency_hz": None,
            "line_noise_prominence_db": 0.0,
            "line_noise_harmonic_count": 0,
        }
    segment_size = min(mono.size, int(max(sample_rate * 2, 4096)))
    segment = mono[:segment_size] - float(np.mean(mono[:segment_size]))
    if float(np.max(np.abs(segment))) <= EPSILON:
        return {
            "line_noise_score": 0.0,
            "line_noise_frequency_hz": None,
            "line_noise_prominence_db": 0.0,
            "line_noise_harmonic_count": 0,
        }
    window = np.hanning(segment.size)
    spectrum = np.abs(np.fft.rfft(segment * window)) ** 2
    freqs = np.fft.rfftfreq(segment.size, 1.0 / float(sample_rate))
    power_db = np.array([_power_db(value) for value in spectrum], dtype=np.float64)

    best: Dict[str, Any] = {
        "frequency_hz": None,
        "base_hz": 0.0,
        "prominence_db": 0.0,
        "harmonic_count": 0,
        "harmonics": [],
    }
    for base_hz in (50.0, 60.0):
        harmonics: List[Dict[str, float]] = []
        for multiplier in (1, 2, 3):
            target_hz = base_hz * multiplier
            if target_hz >= min(260.0, sample_rate / 2.0):
                continue
            index = int(np.argmin(np.abs(freqs - target_hz)))
            if index <= 1 or index >= power_db.size - 1:
                continue
            local_mask = (
                (freqs >= target_hz - 14.0)
                & (freqs <= target_hz + 14.0)
                & ((freqs < target_hz - 2.5) | (freqs > target_hz + 2.5))
            )
            if not np.any(local_mask):
                continue
            local_floor = float(np.percentile(power_db[local_mask], 75))
            prominence_db = max(0.0, float(power_db[index]) - local_floor)
            if prominence_db >= 12.0:
                harmonics.append(
                    {
                        "frequency_hz": round(float(freqs[index]), 1),
                        "prominence_db": round(prominence_db, 2),
                    }
                )
        if not harmonics:
            continue
        primary_prominence = float(harmonics[0]["prominence_db"])
        average_prominence = float(np.mean([item["prominence_db"] for item in harmonics]))
        score = _clip01(
            0.42 * _clip01((primary_prominence - 12.0) / 18.0)
            + 0.28 * _clip01((average_prominence - 10.0) / 18.0)
            + 0.3 * _clip01((len(harmonics) - 1.0) / 2.0)
        )
        if score > _safe_float(best.get("score")):
            best = {
                "frequency_hz": harmonics[0]["frequency_hz"],
                "base_hz": base_hz,
                "prominence_db": round(primary_prominence, 2),
                "harmonic_count": len(harmonics),
                "harmonics": harmonics,
                "score": score,
            }

    if best["frequency_hz"] is None:
        return {
            "line_noise_score": 0.0,
            "line_noise_frequency_hz": None,
            "line_noise_prominence_db": 0.0,
            "line_noise_harmonic_count": 0,
        }
    return {
        "line_noise_score": round(_clip01(best["score"]), 3),
        "line_noise_frequency_hz": best["frequency_hz"],
        "line_noise_prominence_db": best["prominence_db"],
        "line_noise_harmonic_count": int(best["harmonic_count"]),
        "line_noise_harmonics": best["harmonics"],
    }


def _transient_metrics(mono: np.ndarray, sample_rate: int) -> Dict[str, float]:
    frame_rms = _frame_rms_dbfs(mono, sample_rate, frame_ms=20.0, hop_ms=10.0)
    if frame_rms.size < 2:
        return {
            "transient_density": 0.0,
            "transient_strength": 0.0,
            "transient_peak_dbfs": 0.0,
            "transient_rise_time_ms": 0.0,
            "transient_decay_ms": 0.0,
        }
    rises = np.diff(frame_rms)
    transient_rises = rises[rises > 6.0]
    duration_sec = max(mono.size / float(sample_rate), 0.001)
    density = float(transient_rises.size / duration_sec)
    strength = float(np.mean(transient_rises)) if transient_rises.size else 0.0
    envelope = _frame_rms_dbfs(mono, sample_rate, frame_ms=5.0, hop_ms=2.5)
    active = envelope[envelope > -90.0]
    if active.size >= 3:
        floor_dbfs = float(np.percentile(active, 10))
        peak_index = int(np.argmax(envelope))
        peak_dbfs = float(envelope[peak_index])
        onset_threshold = max(floor_dbfs + 6.0, peak_dbfs - 24.0)
        pre_peak = envelope[: peak_index + 1]
        onset_candidates = np.where(pre_peak <= onset_threshold)[0]
        if onset_candidates.size:
            onset_index = int(onset_candidates[-1])
        else:
            onset_index = max(0, peak_index - 1)
        rise_time_ms = max(2.5, (peak_index - onset_index) * 2.5)
        decay_threshold = max(floor_dbfs + 6.0, peak_dbfs - 18.0)
        post_peak = envelope[peak_index:]
        decay_candidates = np.where(post_peak <= decay_threshold)[0]
        if decay_candidates.size:
            decay_ms = max(2.5, int(decay_candidates[0]) * 2.5)
        else:
            decay_ms = max(2.5, (post_peak.size - 1) * 2.5)
    else:
        peak_dbfs = 0.0
        rise_time_ms = 0.0
        decay_ms = 0.0
    return {
        "transient_density": round(_clip(density, 0.0, 20.0), 3),
        "transient_strength": round(_clip(strength / 20.0, 0.0, 1.0), 3),
        "transient_peak_dbfs": round(_clip(peak_dbfs, -120.0, 12.0), 1),
        "transient_rise_time_ms": round(_clip(rise_time_ms, 0.0, 500.0), 1),
        "transient_decay_ms": round(_clip(decay_ms, 0.0, 1200.0), 1),
    }


def _dynamics_envelope_evidence(mono: np.ndarray, sample_rate: int) -> Dict[str, float]:
    frame_rms = _frame_rms_dbfs(mono, sample_rate, frame_ms=40.0, hop_ms=20.0)
    active = frame_rms[frame_rms > -80.0]
    if active.size < 4:
        return {
            "dynamics_peak_to_median_db": 0.0,
            "dynamics_rise_time_ms": 0.0,
            "dynamics_recovery_ms": 0.0,
        }
    median_dbfs = float(np.percentile(active, 25))
    peak_index = int(np.argmax(frame_rms))
    peak_dbfs = float(frame_rms[peak_index])
    peak_to_median = max(0.0, peak_dbfs - median_dbfs)
    pre_peak = frame_rms[: peak_index + 1]
    rise_start_candidates = np.where(pre_peak <= median_dbfs + 3.0)[0]
    if rise_start_candidates.size:
        rise_start = int(rise_start_candidates[-1])
    else:
        rise_start = max(0, peak_index - 1)
    rise_time_ms = max(20.0, (peak_index - rise_start) * 20.0)
    post_peak = frame_rms[peak_index:]
    recovery_candidates = np.where(post_peak <= median_dbfs + 3.0)[0]
    if recovery_candidates.size:
        recovery_ms = int(recovery_candidates[0]) * 20.0
    else:
        recovery_ms = max(20.0, (post_peak.size - 1) * 20.0)
    return {
        "dynamics_peak_to_median_db": round(_clip(peak_to_median, 0.0, 60.0), 1),
        "dynamics_rise_time_ms": round(_clip(rise_time_ms, 0.0, 1000.0), 1),
        "dynamics_recovery_ms": round(_clip(recovery_ms, 0.0, 1500.0), 1),
    }


def _band_limited_signal(
    mono: np.ndarray,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    if mono.size < 32:
        return np.asarray(mono, dtype=np.float64)
    signal = np.asarray(mono, dtype=np.float64) - float(np.mean(mono))
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, 1.0 / float(sample_rate))
    mask = (freqs >= low_hz) & (freqs <= min(high_hz, sample_rate / 2.0))
    spectrum[~mask] = 0.0
    return np.fft.irfft(spectrum, n=signal.size)


def _low_mid_bloom_evidence(mono: np.ndarray, sample_rate: int) -> Dict[str, float]:
    low_mid = _band_limited_signal(mono, sample_rate, 160.0, 520.0)
    frame_rms = _frame_rms_dbfs(low_mid, sample_rate, frame_ms=40.0, hop_ms=20.0)
    active = frame_rms[frame_rms > -100.0]
    if active.size < 4:
        return {
            "low_mid_bloom_frequency_hz": None,
            "low_mid_bloom_db": 0.0,
            "low_mid_bloom_peak_dbfs": 0.0,
            "low_mid_bloom_rise_time_ms": 0.0,
            "low_mid_bloom_recovery_ms": 0.0,
        }
    body_dbfs = float(np.percentile(active, 25))
    peak_index = int(np.argmax(frame_rms))
    peak_dbfs = float(frame_rms[peak_index])
    bloom_db = max(0.0, peak_dbfs - body_dbfs)
    pre_peak = frame_rms[: peak_index + 1]
    rise_candidates = np.where(pre_peak <= body_dbfs + 3.0)[0]
    if rise_candidates.size:
        rise_start = int(rise_candidates[-1])
    else:
        rise_start = max(0, peak_index - 1)
    rise_time_ms = max(20.0, (peak_index - rise_start) * 20.0)
    post_peak = frame_rms[peak_index:]
    recovery_candidates = np.where(post_peak <= body_dbfs + 3.0)[0]
    if recovery_candidates.size:
        recovery_ms = max(20.0, int(recovery_candidates[0]) * 20.0)
    else:
        recovery_ms = max(20.0, (post_peak.size - 1) * 20.0)
    frame_samples = max(64, int(round(sample_rate * 0.04)))
    hop_samples = max(1, int(round(sample_rate * 0.02)))
    start = max(0, peak_index * hop_samples)
    stop = min(low_mid.size, start + frame_samples)
    peak_segment = low_mid[start:stop] if stop > start else low_mid
    bloom_frequency_hz = _peak_frequency_in_band(
        peak_segment,
        sample_rate,
        160.0,
        520.0,
    )
    return {
        "low_mid_bloom_frequency_hz": bloom_frequency_hz,
        "low_mid_bloom_db": round(_clip(bloom_db, 0.0, 60.0), 1),
        "low_mid_bloom_peak_dbfs": round(_clip(peak_dbfs, -120.0, 12.0), 1),
        "low_mid_bloom_rise_time_ms": round(_clip(rise_time_ms, 0.0, 1000.0), 1),
        "low_mid_bloom_recovery_ms": round(_clip(recovery_ms, 0.0, 1500.0), 1),
    }


def _sibilance_event_evidence(mono: np.ndarray, sample_rate: int) -> Dict[str, float]:
    sibilance_band = _band_limited_signal(mono, sample_rate, 5000.0, 10000.0)
    frame_rms = _frame_rms_dbfs(sibilance_band, sample_rate, frame_ms=10.0, hop_ms=5.0)
    active = frame_rms[frame_rms > -100.0]
    if active.size < 4:
        return {
            "sibilance_event_peak_dbfs": 0.0,
            "sibilance_event_contrast_db": 0.0,
            "sibilance_event_rise_time_ms": 0.0,
            "sibilance_event_recovery_ms": 0.0,
        }
    body_dbfs = float(np.percentile(active, 30))
    peak_index = int(np.argmax(frame_rms))
    peak_dbfs = float(frame_rms[peak_index])
    contrast_db = max(0.0, peak_dbfs - body_dbfs)
    pre_peak = frame_rms[: peak_index + 1]
    rise_candidates = np.where(pre_peak <= body_dbfs + 3.0)[0]
    if rise_candidates.size:
        rise_start = int(rise_candidates[-1])
    else:
        rise_start = max(0, peak_index - 1)
    rise_time_ms = max(5.0, (peak_index - rise_start) * 5.0)
    post_peak = frame_rms[peak_index:]
    recovery_candidates = np.where(post_peak <= body_dbfs + 3.0)[0]
    if recovery_candidates.size:
        recovery_ms = max(5.0, int(recovery_candidates[0]) * 5.0)
    else:
        recovery_ms = max(5.0, (post_peak.size - 1) * 5.0)
    return {
        "sibilance_event_peak_dbfs": round(_clip(peak_dbfs, -120.0, 12.0), 1),
        "sibilance_event_contrast_db": round(_clip(contrast_db, 0.0, 60.0), 1),
        "sibilance_event_rise_time_ms": round(_clip(rise_time_ms, 0.0, 300.0), 1),
        "sibilance_event_recovery_ms": round(_clip(recovery_ms, 0.0, 600.0), 1),
    }


def _tail_persistence_score(frame_rms_dbfs: np.ndarray) -> float:
    if frame_rms_dbfs.size < 6:
        return 0.0
    peak = float(np.max(frame_rms_dbfs))
    if peak <= -90.0:
        return 0.0
    late = frame_rms_dbfs[max(0, int(frame_rms_dbfs.size * 0.66)) :]
    if late.size == 0:
        return 0.0
    late_level = float(np.percentile(late, 75))
    return round(_clip01((late_level - peak + 18.0) / 18.0), 3)


def _drum_gate_hit_evidence(
    mono: np.ndarray,
    sample_rate: int,
    noise_floor_dbfs: float,
) -> Dict[str, float]:
    frame_rms = _frame_rms_dbfs(mono, sample_rate, frame_ms=20.0, hop_ms=10.0)
    if frame_rms.size < 3:
        return {
            "gate_hit_peak_dbfs": 0.0,
            "gate_hit_to_bleed_margin_db": 0.0,
            "gate_tail_decay_ms": 0.0,
        }
    peak_index = int(np.argmax(frame_rms))
    peak_dbfs = float(frame_rms[peak_index])
    if peak_dbfs <= -90.0:
        return {
            "gate_hit_peak_dbfs": 0.0,
            "gate_hit_to_bleed_margin_db": 0.0,
            "gate_tail_decay_ms": 0.0,
        }
    bleed_floor = max(_safe_float(noise_floor_dbfs, -80.0), float(np.percentile(frame_rms, 20)))
    hit_to_bleed_margin_db = max(0.0, peak_dbfs - bleed_floor)
    close_threshold = max(bleed_floor + 6.0, peak_dbfs - 24.0)
    tail = frame_rms[peak_index:]
    above_threshold = np.where(tail >= close_threshold)[0]
    tail_decay_ms = 0.0
    if above_threshold.size:
        tail_decay_ms = (int(above_threshold[-1]) + 1) * 10.0
    return {
        "gate_hit_peak_dbfs": round(peak_dbfs, 1),
        "gate_hit_to_bleed_margin_db": round(_clip(hit_to_bleed_margin_db, 0.0, 80.0), 1),
        "gate_tail_decay_ms": round(_clip(tail_decay_ms, 0.0, 800.0), 1),
    }


def _fx_tail_decay_seconds(frame_rms_dbfs: np.ndarray, *, hop_ms: float = 50.0) -> float:
    if frame_rms_dbfs.size < 4:
        return 0.0
    peak_index = int(np.argmax(frame_rms_dbfs))
    peak = float(frame_rms_dbfs[peak_index])
    if peak <= -90.0:
        return 0.0
    tail = frame_rms_dbfs[peak_index:]
    if tail.size < 3:
        return 0.0
    threshold = max(peak - 24.0, -80.0)
    above_threshold = np.where(tail >= threshold)[0]
    if above_threshold.size == 0:
        return 0.0
    decay_seconds = (int(above_threshold[-1]) + 1) * hop_ms / 1000.0
    return round(_clip(decay_seconds, 0.0, 8.0), 2)


def _fx_delay_repeat_ms(mono: np.ndarray, sample_rate: int) -> float:
    frame_rms = _frame_rms_dbfs(mono, sample_rate, frame_ms=20.0, hop_ms=10.0)
    if frame_rms.size < 8:
        return 0.0
    peak = float(np.max(frame_rms))
    if peak <= -90.0:
        return 0.0
    threshold = max(peak - 16.0, -72.0)
    peak_indices: List[int] = []
    min_peak_gap_frames = 4
    for index in range(1, frame_rms.size - 1):
        value = float(frame_rms[index])
        if (
            value < threshold
            or value < float(frame_rms[index - 1])
            or value < float(frame_rms[index + 1])
        ):
            continue
        if peak_indices and index - peak_indices[-1] < min_peak_gap_frames:
            if value > float(frame_rms[peak_indices[-1]]):
                peak_indices[-1] = index
            continue
        peak_indices.append(index)
    if len(peak_indices) < 2:
        return 0.0

    histogram: Dict[float, float] = {}
    for left_pos, left_index in enumerate(peak_indices):
        left_level = float(frame_rms[left_index])
        for right_index in peak_indices[left_pos + 1 :]:
            delta_ms = (right_index - left_index) * 10.0
            if delta_ms < 60.0:
                continue
            if delta_ms > 450.0:
                break
            bin_ms = round(delta_ms / 10.0) * 10.0
            right_level = float(frame_rms[right_index])
            histogram[bin_ms] = histogram.get(bin_ms, 0.0) + 1.0 + _clip01(
                (min(left_level, right_level) - threshold) / 16.0
            )
    if not histogram:
        return 0.0
    best_ms, best_score = max(
        histogram.items(),
        key=lambda item: (item[1], -abs(item[0] - 140.0), -item[0]),
    )
    if best_score < 1.0:
        return 0.0
    return round(_clip(best_ms, 60.0, 450.0), 1)


def _phase_metrics(samples: Any) -> Dict[str, float]:
    array = _as_audio_array(samples)
    if array.ndim != 2 or array.shape[1] < 2:
        return {"phase_correlation": 1.0, "mono_sum_delta_db": 0.0}
    left = array[:, 0].astype(np.float64)
    right = array[:, 1].astype(np.float64)
    left = left - float(np.mean(left))
    right = right - float(np.mean(right))
    left_rms = math.sqrt(float(np.mean(left * left)) + EPSILON)
    right_rms = math.sqrt(float(np.mean(right * right)) + EPSILON)
    if left_rms <= EPSILON or right_rms <= EPSILON:
        return {"phase_correlation": 1.0, "mono_sum_delta_db": 0.0}
    corr = float(np.mean(left * right) / max(left_rms * right_rms, EPSILON))
    mono = 0.5 * (left + right)
    mono_rms = math.sqrt(float(np.mean(mono * mono)) + EPSILON)
    stereo_ref_rms = 0.5 * (left_rms + right_rms)
    mono_delta = 20.0 * math.log10(max(mono_rms, EPSILON) / max(stereo_ref_rms, EPSILON))
    return {
        "phase_correlation": round(_clip(corr, -1.0, 1.0), 3),
        "mono_sum_delta_db": round(_clip(mono_delta, -80.0, 12.0), 2),
    }


def _stereo_balance_metrics(
    samples: Any,
    *,
    current_pan_percent: float = 0.0,
) -> Dict[str, float]:
    array = _as_audio_array(samples)
    if array.ndim != 2 or array.shape[1] < 2:
        return {
            "stereo_balance_db": 0.0,
            "stereo_balance_confidence": 0.0,
            "measured_pan_target_percent": 0.0,
            "measured_pan_delta_percent": 0.0,
        }

    left = array[:, 0].astype(np.float64)
    right = array[:, 1].astype(np.float64)
    left = left - float(np.mean(left))
    right = right - float(np.mean(right))
    left_rms = math.sqrt(float(np.mean(left * left)) + EPSILON)
    right_rms = math.sqrt(float(np.mean(right * right)) + EPSILON)
    if left_rms <= EPSILON or right_rms <= EPSILON:
        return {
            "stereo_balance_db": 0.0,
            "stereo_balance_confidence": 0.0,
            "measured_pan_target_percent": 0.0,
            "measured_pan_delta_percent": 0.0,
        }

    balance_db = 20.0 * math.log10(max(right_rms, EPSILON) / max(left_rms, EPSILON))
    imbalance_db = abs(balance_db)
    confidence = _clip01((imbalance_db - 1.0) / 8.0)
    target_pan = _clip(-5.0 * balance_db, -35.0, 35.0)
    if confidence < 0.15:
        target_pan = 0.0
    current_pan = _clip(_safe_float(current_pan_percent), -100.0, 100.0)
    return {
        "stereo_balance_db": round(_clip(balance_db, -24.0, 24.0), 2),
        "stereo_balance_confidence": round(confidence, 3),
        "measured_pan_target_percent": round(target_pan, 1),
        "measured_pan_delta_percent": round(target_pan - current_pan, 1),
    }


def _masking_evidence(
    target_mono: np.ndarray,
    reference_samples: Any,
    sample_rate: int,
) -> Dict[str, Any]:
    reference_mono, _ = _mono_and_channel_count(reference_samples)
    size = min(target_mono.size, reference_mono.size, int(max(sample_rate * 4, 2048)))
    if size < 256:
        return {
            "masking_score": 0.0,
            "masking_frequency_hz": None,
            "masking_bandwidth_hz": 0.0,
            "masking_overlap_peak_share": 0.0,
            "arrangement_overlap_score": 0.0,
        }
    target = target_mono[:size] - float(np.mean(target_mono[:size]))
    reference = reference_mono[:size] - float(np.mean(reference_mono[:size]))
    window = np.hanning(size)
    target_power = np.abs(np.fft.rfft(target * window)) ** 2
    reference_power = np.abs(np.fft.rfft(reference * window)) ** 2
    freqs = np.fft.rfftfreq(size, 1.0 / float(sample_rate))
    mask = (freqs >= 45.0) & (freqs <= 6000.0)
    if not np.any(mask):
        return {
            "masking_score": 0.0,
            "masking_frequency_hz": None,
            "masking_bandwidth_hz": 0.0,
            "masking_overlap_peak_share": 0.0,
            "arrangement_overlap_score": 0.0,
        }
    target_norm = target_power / max(float(np.sum(target_power[mask])), EPSILON)
    reference_norm = reference_power / max(float(np.sum(reference_power[mask])), EPSILON)
    overlap = np.minimum(target_norm[mask], reference_norm[mask])
    overlap_score = float(np.sum(overlap))
    masked_indices = np.where(mask)[0]
    masked_freqs = freqs[mask]
    if overlap.size:
        peak_index = int(masked_indices[int(np.argmax(overlap))])
        frequency = round(float(freqs[peak_index]), 1)
        peak_overlap = float(overlap[int(np.argmax(overlap))])
        peak_share = peak_overlap / max(overlap_score, EPSILON)
        half_peak_indices = np.where(overlap >= peak_overlap * 0.5)[0]
        resolution_hz = (
            float(masked_freqs[1] - masked_freqs[0])
            if masked_freqs.size > 1
            else 0.0
        )
        bandwidth_hz = float(max(half_peak_indices.size, 1) * max(resolution_hz, 1.0))
    else:
        frequency = None
        peak_share = 0.0
        bandwidth_hz = 0.0
    low_register_overlap = float(
        np.sum(overlap[(masked_freqs >= 45.0) & (masked_freqs <= 140.0)])
    )
    low_register_share = low_register_overlap / max(overlap_score, EPSILON)
    masking_score = _clip01(overlap_score * 3.0)
    arrangement_overlap_score = 0.0
    if frequency is not None and 45.0 <= frequency <= 140.0:
        arrangement_overlap_score = _clip01(
            masking_score * (0.55 + 0.45 * _clip01(low_register_share))
        )
    return {
        "masking_score": round(masking_score, 3),
        "masking_frequency_hz": frequency,
        "masking_bandwidth_hz": round(_clip(bandwidth_hz, 0.0, 2400.0), 1),
        "masking_overlap_peak_share": round(_clip01(peak_share), 3),
        "arrangement_overlap_score": round(arrangement_overlap_score, 3),
    }


def _feedback_ring_evidence(
    mono: np.ndarray,
    sample_rate: int,
    *,
    source_family: str,
    high_ratio_db: float,
    sibilance_score: float,
    monitor_feedback_risk: float,
) -> Dict[str, Any]:
    if mono.size < 2048:
        return {"frequency_hz": None, "confidence": 0.0, "source": "insufficient_audio"}
    segment_size = min(mono.size, int(max(sample_rate * 2, 4096)))
    segment = mono[:segment_size] - float(np.mean(mono[:segment_size]))
    if float(np.max(np.abs(segment))) <= EPSILON:
        return {"frequency_hz": None, "confidence": 0.0, "source": "silent_audio"}
    window = np.hanning(segment.size)
    power = np.abs(np.fft.rfft(segment * window)) ** 2
    freqs = np.fft.rfftfreq(segment.size, 1.0 / float(sample_rate))
    min_hz = 450.0 if source_family in {"snare", "tom", "drum"} else 1200.0
    mask = (freqs >= min_hz) & (freqs <= min(9000.0, sample_rate / 2.0))
    if not np.any(mask):
        return {"frequency_hz": None, "confidence": 0.0, "source": "no_feedback_band"}
    band_power = power[mask]
    band_freqs = freqs[mask]
    if band_power.size < 8:
        return {"frequency_hz": None, "confidence": 0.0, "source": "feedback_band_too_small"}
    power_db = np.array([_power_db(value) for value in band_power], dtype=np.float64)
    peak_local_index = int(np.argmax(power_db))
    peak_frequency_hz = round(float(band_freqs[peak_local_index]), 1)
    peak_db = float(power_db[peak_local_index])
    floor_db = float(np.percentile(power_db, 75))
    prominence_db = peak_db - floor_db
    within_6db = np.where(power_db >= peak_db - 6.0)[0]
    resolution_hz = float(band_freqs[1] - band_freqs[0]) if band_freqs.size > 1 else 0.0
    width_hz = float(max(within_6db.size, 1) * max(resolution_hz, 1.0))
    narrowness_score = _clip01(1.0 - width_hz / 900.0)
    confidence = _clip01((prominence_db - 10.0) / 18.0 + 0.25 * narrowness_score)
    instrument_tonal_family = source_family in {"guitar", "keys", "playback"}
    risk_context = (
        _clip01(monitor_feedback_risk) >= 0.3
        or (high_ratio_db >= 6.0 and not instrument_tonal_family)
    )
    sibilance_only_peak = (
        source_family == "vocal"
        and peak_frequency_hz >= 5000.0
        and sibilance_score >= 0.45
        and _clip01(monitor_feedback_risk) < 0.3
    )
    instrument_tone_only_peak = (
        instrument_tonal_family
        and _clip01(monitor_feedback_risk) < 0.3
    )
    if confidence < 0.45 or not risk_context or sibilance_only_peak or instrument_tone_only_peak:
        return {
            "frequency_hz": peak_frequency_hz,
            "confidence": round(confidence, 3),
            "source": "narrow_peak_without_feedback_context",
        }
    return {
        "frequency_hz": peak_frequency_hz,
        "confidence": round(confidence, 3),
        "source": "measured_feedback_ring_peak",
    }


def _drum_ring_resonance_evidence(
    mono: np.ndarray,
    sample_rate: int,
    *,
    source_family: str,
    tail_persistence: float,
    transient: Mapping[str, float],
) -> Dict[str, Any]:
    empty = {
        "score": 0.0,
        "frequency_hz": None,
        "decay_score": 0.0,
        "bandwidth_hz": 0.0,
        "peak_dbfs": 0.0,
        "decay_ms": 0.0,
    }
    if source_family not in {"snare", "tom"} or mono.size < 2048:
        return empty
    segment_size = min(mono.size, int(max(sample_rate * 4, 4096)))
    segment = mono[:segment_size] - float(np.mean(mono[:segment_size]))
    if float(np.max(np.abs(segment))) <= EPSILON:
        return empty

    window = np.hanning(segment.size)
    power = np.abs(np.fft.rfft(segment * window)) ** 2
    freqs = np.fft.rfftfreq(segment.size, 1.0 / float(sample_rate))
    low_hz = 90.0 if source_family == "tom" else 140.0
    high_hz = 760.0 if source_family == "tom" else 900.0
    mask = (freqs >= low_hz) & (freqs <= min(high_hz, sample_rate / 2.0))
    full_mask = (freqs >= 60.0) & (freqs <= min(6000.0, sample_rate / 2.0))
    if not np.any(mask) or not np.any(full_mask):
        return empty

    band_power = power[mask]
    band_freqs = freqs[mask]
    if band_power.size < 8:
        return empty
    power_db = np.array([_power_db(value) for value in band_power], dtype=np.float64)
    peak_local_index = int(np.argmax(power_db))
    peak_frequency_hz = round(float(band_freqs[peak_local_index]), 1)
    peak_db = float(power_db[peak_local_index])
    floor_db = float(np.percentile(power_db, 65))
    prominence_db = max(0.0, peak_db - floor_db)
    within_6db = np.where(power_db >= peak_db - 6.0)[0]
    resolution_hz = float(band_freqs[1] - band_freqs[0]) if band_freqs.size > 1 else 0.0
    width_hz = float(max(within_6db.size, 1) * max(resolution_hz, 1.0))
    narrowness_score = _clip01(1.0 - width_hz / 420.0)
    ring_band_share = float(np.sum(band_power) / max(np.sum(power[full_mask]), EPSILON))
    decay_score = _clip01(tail_persistence)
    ring_low_hz = max(low_hz, peak_frequency_hz - max(width_hz * 1.5, 35.0))
    ring_high_hz = min(high_hz, peak_frequency_hz + max(width_hz * 1.5, 35.0))
    ring_signal = _band_limited_signal(mono, sample_rate, ring_low_hz, ring_high_hz)
    frame_rms = _frame_rms_dbfs(ring_signal, sample_rate, frame_ms=30.0, hop_ms=10.0)
    active_frames = frame_rms[frame_rms > -100.0]
    ring_peak_dbfs = float(np.max(active_frames)) if active_frames.size else 0.0
    if active_frames.size:
        body_dbfs = float(np.percentile(active_frames, 35))
        decay_threshold_dbfs = min(
            max(body_dbfs + 3.0, ring_peak_dbfs - 18.0),
            ring_peak_dbfs - 1.0,
        )
        above_decay = np.where(frame_rms >= decay_threshold_dbfs)[0]
        decay_ms = float(above_decay.size * 10.0) if above_decay.size else 0.0
    else:
        decay_ms = 0.0
    transient_density = _safe_float(transient.get("transient_density"))
    transient_strength = _safe_float(transient.get("transient_strength"))
    transient_penalty = 0.0
    if transient_strength >= 0.65 and transient_density >= 2.5 and decay_score < 0.55:
        transient_penalty = 0.22
    score = _clip01(
        0.14
        + max(0.0, prominence_db - 7.0) / 18.0
        + 0.26 * narrowness_score
        + 0.2 * decay_score
        + 0.14 * _clip01(ring_band_share * 3.0)
        - transient_penalty
    )
    return {
        "score": round(score, 3),
        "frequency_hz": peak_frequency_hz if score >= 0.25 else None,
        "decay_score": round(decay_score, 3),
        "bandwidth_hz": round(_clip(width_hz, 0.0, 400.0), 1),
        "peak_dbfs": round(_clip(ring_peak_dbfs, -120.0, 12.0), 1),
        "decay_ms": round(_clip(decay_ms, 0.0, 1500.0), 1),
    }


def _drum_harshness_event_evidence(
    mono: np.ndarray,
    sample_rate: int,
    *,
    source_family: str,
    source_role: str,
    source_label: str,
) -> Dict[str, Any]:
    empty = {
        "frequency_hz": None,
        "bandwidth_hz": 0.0,
        "event_peak_dbfs": 0.0,
        "event_contrast_db": 0.0,
        "event_recovery_ms": 0.0,
    }
    if source_family not in {"drum", "snare", "tom"} or mono.size < 2048:
        return empty

    text = _source_text(source_role, source_label)
    if any(token in text for token in ("overhead", "cymbal", "hat", "ride", "crash", "oh")):
        low_hz, high_hz = 5200.0, 11000.0
    elif source_family == "snare":
        low_hz, high_hz = 3200.0, 8200.0
    elif source_family == "tom":
        low_hz, high_hz = 2400.0, 6500.0
    else:
        low_hz, high_hz = 3600.0, 9500.0

    segment_size = min(mono.size, int(max(sample_rate * 4, 4096)))
    segment = mono[:segment_size] - float(np.mean(mono[:segment_size]))
    if float(np.max(np.abs(segment))) <= EPSILON:
        return empty

    window = np.hanning(segment.size)
    power = np.abs(np.fft.rfft(segment * window)) ** 2
    freqs = np.fft.rfftfreq(segment.size, 1.0 / float(sample_rate))
    mask = (freqs >= low_hz) & (freqs <= min(high_hz, sample_rate / 2.0))
    if not np.any(mask):
        return empty

    band_power = power[mask]
    band_freqs = freqs[mask]
    if band_power.size < 8:
        return empty
    power_db = np.array([_power_db(value) for value in band_power], dtype=np.float64)
    peak_local_index = int(np.argmax(power_db))
    peak_frequency_hz = round(float(band_freqs[peak_local_index]), 1)
    peak_db = float(power_db[peak_local_index])
    within_6db = np.where(power_db >= peak_db - 6.0)[0]
    resolution_hz = float(band_freqs[1] - band_freqs[0]) if band_freqs.size > 1 else 0.0
    bandwidth_hz = float(max(within_6db.size, 1) * max(resolution_hz, 1.0))

    event_low_hz = max(low_hz, peak_frequency_hz - max(bandwidth_hz * 1.2, 600.0))
    event_high_hz = min(high_hz, peak_frequency_hz + max(bandwidth_hz * 1.2, 600.0))
    harsh_signal = _band_limited_signal(mono, sample_rate, event_low_hz, event_high_hz)
    frame_rms = _frame_rms_dbfs(harsh_signal, sample_rate, frame_ms=20.0, hop_ms=10.0)
    active_frames = frame_rms[frame_rms > -100.0]
    if active_frames.size < 2:
        return {
            "frequency_hz": peak_frequency_hz,
            "bandwidth_hz": round(_clip(bandwidth_hz, 0.0, 7000.0), 1),
            "event_peak_dbfs": 0.0,
            "event_contrast_db": 0.0,
            "event_recovery_ms": 0.0,
        }

    event_peak_dbfs = float(np.max(active_frames))
    body_dbfs = float(np.percentile(active_frames, 35))
    event_contrast_db = max(0.0, event_peak_dbfs - body_dbfs)
    peak_frame_index = int(np.argmax(frame_rms))
    post_peak = frame_rms[peak_frame_index:]
    recovery_candidates = np.where(post_peak <= body_dbfs + 3.0)[0]
    if recovery_candidates.size:
        recovery_ms = float(recovery_candidates[0] * 10.0)
    else:
        recovery_ms = float(max(10, post_peak.size - 1) * 10.0)

    return {
        "frequency_hz": peak_frequency_hz,
        "bandwidth_hz": round(_clip(bandwidth_hz, 0.0, 7000.0), 1),
        "event_peak_dbfs": round(_clip(event_peak_dbfs, -120.0, 12.0), 1),
        "event_contrast_db": round(_clip(event_contrast_db, 0.0, 48.0), 1),
        "event_recovery_ms": round(_clip(recovery_ms, 0.0, 1000.0), 1),
    }


def _instrument_harshness_event_evidence(
    mono: np.ndarray,
    sample_rate: int,
    *,
    source_family: str,
    source_role: str,
    source_label: str,
) -> Dict[str, Any]:
    empty = {
        "frequency_hz": None,
        "bandwidth_hz": 0.0,
        "event_peak_dbfs": 0.0,
        "event_contrast_db": 0.0,
        "event_recovery_ms": 0.0,
    }
    if source_family not in {"guitar", "keys", "playback"} or mono.size < 2048:
        return empty

    band_by_family = {
        "guitar": (1800.0, 6500.0),
        "keys": (2200.0, 7500.0),
        "playback": (1800.0, 9000.0),
    }
    low_hz, high_hz = band_by_family.get(source_family, (1800.0, 9000.0))
    text = _source_text(source_role, source_label)
    if source_family == "guitar" and any(token in text for token in ("acoustic", "nylon")):
        high_hz = min(high_hz, 5800.0)

    segment_size = min(mono.size, int(max(sample_rate * 4, 4096)))
    segment = mono[:segment_size] - float(np.mean(mono[:segment_size]))
    if float(np.max(np.abs(segment))) <= EPSILON:
        return empty

    window = np.hanning(segment.size)
    power = np.abs(np.fft.rfft(segment * window)) ** 2
    freqs = np.fft.rfftfreq(segment.size, 1.0 / float(sample_rate))
    mask = (freqs >= low_hz) & (freqs <= min(high_hz, sample_rate / 2.0))
    if not np.any(mask):
        return empty

    band_power = power[mask]
    band_freqs = freqs[mask]
    if band_power.size < 8:
        return empty
    power_db = np.array([_power_db(value) for value in band_power], dtype=np.float64)
    peak_local_index = int(np.argmax(power_db))
    peak_frequency_hz = round(float(band_freqs[peak_local_index]), 1)
    peak_db = float(power_db[peak_local_index])
    within_6db = np.where(power_db >= peak_db - 6.0)[0]
    resolution_hz = float(band_freqs[1] - band_freqs[0]) if band_freqs.size > 1 else 0.0
    bandwidth_hz = float(max(within_6db.size, 1) * max(resolution_hz, 1.0))

    event_low_hz = max(low_hz, peak_frequency_hz - max(bandwidth_hz * 1.2, 550.0))
    event_high_hz = min(high_hz, peak_frequency_hz + max(bandwidth_hz * 1.2, 550.0))
    harsh_signal = _band_limited_signal(mono, sample_rate, event_low_hz, event_high_hz)
    frame_rms = _frame_rms_dbfs(harsh_signal, sample_rate, frame_ms=20.0, hop_ms=10.0)
    active_frames = frame_rms[frame_rms > -100.0]
    if active_frames.size < 2:
        return {
            "frequency_hz": peak_frequency_hz,
            "bandwidth_hz": round(_clip(bandwidth_hz, 0.0, 6000.0), 1),
            "event_peak_dbfs": 0.0,
            "event_contrast_db": 0.0,
            "event_recovery_ms": 0.0,
        }

    event_peak_dbfs = float(np.max(active_frames))
    body_dbfs = float(np.percentile(active_frames, 35))
    event_contrast_db = max(0.0, event_peak_dbfs - body_dbfs)
    peak_frame_index = int(np.argmax(frame_rms))
    post_peak = frame_rms[peak_frame_index:]
    recovery_candidates = np.where(post_peak <= body_dbfs + 3.0)[0]
    if recovery_candidates.size:
        recovery_ms = float(recovery_candidates[0] * 10.0)
    else:
        recovery_ms = float(max(10, post_peak.size - 1) * 10.0)

    return {
        "frequency_hz": peak_frequency_hz,
        "bandwidth_hz": round(_clip(bandwidth_hz, 0.0, 6000.0), 1),
        "event_peak_dbfs": round(_clip(event_peak_dbfs, -120.0, 12.0), 1),
        "event_contrast_db": round(_clip(event_contrast_db, 0.0, 48.0), 1),
        "event_recovery_ms": round(_clip(recovery_ms, 0.0, 1200.0), 1),
    }


@dataclass(frozen=True)
class SmartDSPAudioEvidence:
    target_id: str
    source_role: str
    source_label: str
    musical_role: str
    declared_source_role: str
    declared_source_family: str
    inferred_source_role: str
    inferred_source_family: str
    source_identity_inference_confidence: float
    source_identity_inference_reason: str
    source_identity_inference_used: bool
    source_identity_mismatch_score: float
    source_identity_mismatch_reason: str
    source_identity_mismatch_family: str
    mode: str
    sample_rate: int
    duration_sec: float
    channel_count: int
    peak_dbfs: float
    true_peak_dbfs: float
    true_peak_headroom_db: float
    rms_dbfs: float
    estimated_lufs: float
    crest_factor_db: float
    lra_proxy_db: float
    noise_floor_dbfs: float
    spectral_centroid_hz: float
    dominant_frequency_hz: float
    band_energy_dbfs: Dict[str, float]
    band_energy_share: Dict[str, float]
    low_end_excess_score: float
    spectral_tilt_score: float
    low_end_resonance_frequency_hz: Optional[float]
    low_end_resonance_bandwidth_hz: float
    low_end_resonance_peak_dbfs: float
    low_end_resonance_sustain_ms: float
    presence_deficit_score: float
    presence_lift_frequency_hz: Optional[float]
    presence_lift_bandwidth_hz: float
    presence_lift_parameter_source: str
    muddiness_score: float
    mud_frequency_hz: Optional[float]
    low_mid_bloom_frequency_hz: Optional[float]
    low_mid_bloom_db: float
    low_mid_bloom_peak_dbfs: float
    low_mid_bloom_rise_time_ms: float
    low_mid_bloom_recovery_ms: float
    plosive_risk_score: float
    plosive_frequency_hz: Optional[float]
    plosive_event_peak_dbfs: float
    plosive_event_contrast_db: float
    plosive_event_rise_time_ms: float
    plosive_event_recovery_ms: float
    harshness_score: float
    presence_harshness_score: float
    instrument_harsh_frequency_hz: Optional[float]
    instrument_harsh_bandwidth_hz: float
    instrument_harsh_event_peak_dbfs: float
    instrument_harsh_event_contrast_db: float
    instrument_harsh_event_recovery_ms: float
    drum_harsh_frequency_hz: Optional[float]
    drum_harsh_bandwidth_hz: float
    drum_harsh_event_peak_dbfs: float
    drum_harsh_event_contrast_db: float
    drum_harsh_event_recovery_ms: float
    sibilance_score: float
    sibilance_frequency_hz: Optional[float]
    sibilance_bandwidth_hz: Optional[float]
    sibilance_event_peak_dbfs: float
    sibilance_event_contrast_db: float
    sibilance_event_rise_time_ms: float
    sibilance_event_recovery_ms: float
    dynamics_variance_score: float
    dynamics_peak_to_median_db: float
    dynamics_rise_time_ms: float
    dynamics_recovery_ms: float
    vocal_noise_floor_score: float
    phrase_gap_noise_floor_dbfs: Optional[float]
    phrase_gap_noise_flatness: float
    phrase_gap_contrast_db: float
    phrase_gap_median_ms: float
    phrase_gap_coverage_ratio: float
    line_noise_score: float
    line_noise_frequency_hz: Optional[float]
    line_noise_prominence_db: float
    line_noise_harmonic_count: int
    line_noise_harmonics: Tuple[Dict[str, float], ...]
    drum_ring_resonance_score: float
    drum_ring_frequency_hz: Optional[float]
    drum_ring_decay_score: float
    drum_ring_bandwidth_hz: float
    drum_ring_peak_dbfs: float
    drum_ring_decay_ms: float
    transient_density: float
    transient_strength: float
    transient_peak_dbfs: float
    transient_rise_time_ms: float
    transient_decay_ms: float
    bleed_risk: float
    tail_risk: float
    chatter_risk: float
    gate_hit_peak_dbfs: float
    gate_hit_to_bleed_margin_db: float
    gate_tail_decay_ms: float
    phase_correlation: float
    mono_sum_delta_db: float
    stereo_balance_db: float
    stereo_balance_confidence: float
    measured_pan_target_percent: float
    measured_pan_delta_percent: float
    feedback_risk: float
    feedback_frequency_hz: Optional[float]
    feedback_frequency_confidence: float
    feedback_frequency_source: str
    masking_conflict_role: str
    masking_conflict_target_id: str
    masking_frequency_hz: Optional[float]
    masking_score: float
    masking_bandwidth_hz: float
    masking_overlap_peak_share: float
    arrangement_overlap_score: float
    fx_wetness_score: float
    fx_decay_risk_score: float
    fx_delay_smear_score: float
    fx_tail_decay_seconds: float
    fx_delay_repeat_ms: float
    fx_return_headroom_db: float
    fx_return_target_id: str
    monitor_feedback_risk: float
    master_true_peak_db: float
    master_integrated_lufs: float
    master_presence_frequency_hz: Optional[float]
    master_presence_bandwidth_hz: float
    master_low_end_frequency_hz: Optional[float]
    master_low_end_bandwidth_hz: float
    target_plane_lufs: float
    level_plane_error_lu: float
    summed_bus_evidence_available: bool
    summed_bus_role: str
    summed_bus_target_id: str
    summed_bus_true_peak_headroom_db: float
    summed_bus_estimated_lufs: float
    summed_bus_target_plane_lufs: float
    summed_bus_level_plane_error_lu: float
    evidence_confidence: float
    problem_hints: Tuple[str, ...]
    extraction_warnings: Tuple[str, ...]
    read_only: bool = True
    live_write: bool = False
    transport_free: bool = True
    version: str = SMART_DSP_EVIDENCE_EXTRACTOR_VERSION

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["problem_hints"] = list(self.problem_hints)
        payload["extraction_warnings"] = list(self.extraction_warnings)
        return payload


def extract_smart_dsp_audio_evidence(
    samples: Any,
    *,
    target_id: str,
    source_role: str,
    source_label: str = "",
    mode: str = "soundcheck",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    reference_samples: Any = None,
    reference_role: str = "",
    reference_target_id: str = "",
    summed_bus_samples: Any = None,
    summed_bus_role: str = "",
    summed_bus_target_id: str = "",
    summed_bus_target_plane_lufs: Optional[float] = None,
    desired_stereo_width_request: float = 0.0,
    current_pan_percent: float = 0.0,
    fx_return_headroom_db: Optional[float] = None,
    fx_return_target_id: str = "",
    monitor_feedback_risk: float = 0.0,
    target_plane_lufs: Optional[float] = None,
    extra_problem_hints: Iterable[str] = (),
) -> SmartDSPAudioEvidence:
    sample_rate = int(max(8000, sample_rate))
    mono, channel_count = _mono_and_channel_count(samples)
    level_signal = _level_signal(samples)
    duration_sec = round(mono.size / float(sample_rate), 3)
    warnings = []
    if duration_sec < 0.5:
        warnings.append("audio_window_shorter_than_500ms")

    peak_dbfs = round(_dbfs(float(np.max(np.abs(_as_audio_array(samples))))), 2)
    true_peak = _true_peak_dbfs(samples)
    rms = math.sqrt(float(np.mean(level_signal * level_signal)) + EPSILON)
    rms_dbfs = round(_dbfs(rms), 2)
    frame_rms = _frame_rms_dbfs(level_signal, sample_rate)
    active_frames = frame_rms[frame_rms > -80.0]
    if active_frames.size:
        noise_floor_dbfs = round(float(np.percentile(active_frames, 10)), 2)
        lra_proxy_db = round(float(np.percentile(active_frames, 95) - np.percentile(active_frames, 10)), 2)
    else:
        noise_floor_dbfs = -120.0
        lra_proxy_db = 0.0
    estimated_lufs = round(rms_dbfs - 0.7, 2)
    crest_factor_db = round(_clip(peak_dbfs - rms_dbfs, 0.0, 80.0), 2)
    true_peak_headroom_db = round(max(0.0, -1.0 - true_peak), 2)

    declared_source_role = str(source_role)
    declared_source_family = classify_source_family(source_role)
    spectral = _spectral_profile(mono, sample_rate)
    share = spectral["band_share"]
    transient = _transient_metrics(mono, sample_rate)
    tail_persistence = _tail_persistence_score(frame_rms)
    inference = _infer_source_identity(
        source_role=declared_source_role,
        source_label=str(source_label),
        declared_family=declared_source_family,
        band_share=share,
        spectral_centroid_hz=float(spectral["spectral_centroid_hz"]),
        dominant_frequency_hz=float(spectral["dominant_frequency_hz"]),
        transient=transient,
        crest_factor_db=crest_factor_db,
        lra_proxy_db=lra_proxy_db,
        tail_persistence=tail_persistence,
        channel_count=channel_count,
    )
    acoustic_probe = _acoustic_source_identity_probe(
        band_share=share,
        spectral_centroid_hz=float(spectral["spectral_centroid_hz"]),
        dominant_frequency_hz=float(spectral["dominant_frequency_hz"]),
        transient=transient,
        crest_factor_db=crest_factor_db,
        lra_proxy_db=lra_proxy_db,
        tail_persistence=tail_persistence,
        channel_count=channel_count,
    )
    identity_mismatch = _source_family_mismatch_score(
        source_role=declared_source_role,
        source_label=str(source_label),
        target_id=str(target_id),
        declared_family=declared_source_family,
        acoustic_probe=acoustic_probe,
    )
    source_family = str(inference["effective_source_family"] or declared_source_family or "unknown")
    effective_source_role = str(inference["effective_source_role"] or declared_source_role)
    musical_role = classify_musical_role(effective_source_role, str(source_label), source_family)
    resolved_target_plane_lufs = round(
        _clip(
            _safe_float(
                target_plane_lufs,
                _resolve_ayaic_target_lufs(effective_source_role, source_family),
            ),
            -40.0,
            -16.0,
        ),
        1,
    )
    level_plane_error_lu = round(estimated_lufs - resolved_target_plane_lufs, 2)
    summed_bus_evidence_available = summed_bus_samples is not None
    summed_bus_role_text = str(summed_bus_role or "summed_bus")
    summed_bus_target_id_text = str(summed_bus_target_id or "").strip()
    summed_bus_true_peak_headroom_db = 6.0
    summed_bus_estimated_lufs = -25.0
    summed_bus_target_lufs = _clip(
        _safe_float(
            summed_bus_target_plane_lufs,
            _resolve_ayaic_target_lufs(
                summed_bus_role_text,
                classify_source_family(summed_bus_role_text),
            ),
        ),
        -40.0,
        -16.0,
    )
    summed_bus_level_plane_error_lu = 0.0
    if summed_bus_evidence_available:
        summed_bus_level_signal = _level_signal(summed_bus_samples)
        summed_bus_true_peak = _true_peak_dbfs(summed_bus_samples)
        summed_bus_rms = math.sqrt(
            float(np.mean(summed_bus_level_signal * summed_bus_level_signal)) + EPSILON
        )
        summed_bus_estimated_lufs = round(_dbfs(summed_bus_rms) - 0.7, 2)
        summed_bus_true_peak_headroom_db = round(max(0.0, -1.0 - summed_bus_true_peak), 2)
        summed_bus_level_plane_error_lu = round(
            summed_bus_estimated_lufs - summed_bus_target_lufs,
            2,
        )
    mud_numerator_bands = (
        ("low_mid",)
        if source_family in {"kick", "tom", "bass"}
        else ("bass", "low_mid")
    )
    low_mid_ratio_db = _band_ratio_db(
        share,
        mud_numerator_bands,
        ("mid", "presence", "sibilance"),
    )
    high_ratio_db = _band_ratio_db(share, ("presence", "sibilance"), ("mid", "low_mid"))
    low_end_ratio_db = _band_ratio_db(share, ("sub", "bass"), ("mid", "presence"))
    low_end_excess_score = round(_clip01(0.25 + (low_end_ratio_db + 3.0) / 16.0), 3)
    spectral_tilt_score = round(_clip01(0.25 + (low_mid_ratio_db + 3.0) / 18.0), 3)
    low_end_resonance = _low_end_resonance_evidence(
        mono,
        sample_rate,
        source_family=source_family,
    )
    presence_air_share = float(share.get("presence", 0.0)) + float(share.get("air", 0.0))
    presence_deficit_score = round(
        _clip01(
            0.12
            + max(0.0, 0.18 - presence_air_share) * 3.1
            + max(0.0, -high_ratio_db) / 16.0
        ),
        3,
    )
    presence_lift = _presence_lift_evidence(
        source_family=source_family,
        spectral_centroid_hz=float(spectral["spectral_centroid_hz"]),
        presence_air_share=presence_air_share,
        high_ratio_db=high_ratio_db,
        presence_deficit_score=presence_deficit_score,
    )
    bass_mud_weight = 0.2 if source_family in {"kick", "tom", "bass"} else 0.9
    low_mid_mud_weight = 1.7 if source_family in {"kick", "tom"} else 2.4
    muddiness_score = round(
        _clip01(
            0.12
            + low_mid_mud_weight * float(share.get("low_mid", 0.0))
            + bass_mud_weight * float(share.get("bass", 0.0))
            + max(0.0, low_mid_ratio_db) / 16.0
        ),
        3,
    )
    if source_family in {"kick", "tom"} and float(share.get("low_mid", 0.0)) < 0.08:
        muddiness_score = min(muddiness_score, 0.39)
    mud_frequency_hz = (
        _peak_frequency_in_band(mono, sample_rate, 180.0, 520.0)
        if muddiness_score >= 0.45
        else None
    )
    low_mid_bloom = _low_mid_bloom_evidence(mono, sample_rate)
    harshness_score = round(
        _clip01(
            0.08
            + 2.2 * float(share.get("presence", 0.0))
            + 0.8 * float(share.get("sibilance", 0.0))
            + max(0.0, high_ratio_db) / 18.0
        ),
        3,
    )
    presence_harshness_score = round(
        _clip01(
            0.04
            + 3.0 * float(share.get("presence", 0.0))
            + max(0.0, _band_ratio_db(share, ("presence",), ("mid", "low_mid")))
            / 16.0
        ),
        3,
    )
    instrument_harsh = _instrument_harshness_event_evidence(
        mono,
        sample_rate,
        source_family=source_family,
        source_role=effective_source_role,
        source_label=str(source_label),
    )
    instrument_harsh_frequency_hz = (
        instrument_harsh["frequency_hz"]
        if source_family in {"guitar", "keys", "playback"}
        and max(harshness_score, presence_harshness_score) >= 0.45
        else None
    )
    instrument_harsh_bandwidth_hz = (
        round(float(instrument_harsh["bandwidth_hz"]), 1)
        if instrument_harsh_frequency_hz is not None
        else 0.0
    )
    instrument_harsh_event_peak_dbfs = (
        round(float(instrument_harsh["event_peak_dbfs"]), 1)
        if instrument_harsh_frequency_hz is not None
        else 0.0
    )
    instrument_harsh_event_contrast_db = (
        round(float(instrument_harsh["event_contrast_db"]), 1)
        if instrument_harsh_frequency_hz is not None
        else 0.0
    )
    instrument_harsh_event_recovery_ms = (
        round(float(instrument_harsh["event_recovery_ms"]), 1)
        if instrument_harsh_frequency_hz is not None
        else 0.0
    )
    drum_harsh = _drum_harshness_event_evidence(
        mono,
        sample_rate,
        source_family=source_family,
        source_role=effective_source_role,
        source_label=str(source_label),
    )
    drum_harsh_frequency_hz = (
        drum_harsh["frequency_hz"]
        if source_family in {"drum", "snare", "tom"}
        and max(harshness_score, presence_harshness_score) >= 0.45
        else None
    )
    drum_harsh_bandwidth_hz = (
        round(float(drum_harsh["bandwidth_hz"]), 1)
        if drum_harsh_frequency_hz is not None
        else 0.0
    )
    drum_harsh_event_peak_dbfs = (
        round(float(drum_harsh["event_peak_dbfs"]), 1)
        if drum_harsh_frequency_hz is not None
        else 0.0
    )
    drum_harsh_event_contrast_db = (
        round(float(drum_harsh["event_contrast_db"]), 1)
        if drum_harsh_frequency_hz is not None
        else 0.0
    )
    drum_harsh_event_recovery_ms = (
        round(float(drum_harsh["event_recovery_ms"]), 1)
        if drum_harsh_frequency_hz is not None
        else 0.0
    )
    sibilance_score = round(
        _clip01(
            0.04
            + 4.0 * float(share.get("sibilance", 0.0))
            + max(0.0, _band_ratio_db(share, ("sibilance",), ("mid", "presence"))) / 12.0
        ),
        3,
    )
    master_presence_band = (
        _peak_band_details_in_band(mono, sample_rate, 2200.0, 5200.0)
        if source_family == "master" and presence_harshness_score >= 0.45
        else {"peak_frequency_hz": None, "bandwidth_hz": None}
    )
    master_presence_frequency_hz = master_presence_band["peak_frequency_hz"]
    master_presence_bandwidth_hz = (
        round(float(master_presence_band["bandwidth_hz"]), 1)
        if master_presence_frequency_hz is not None
        and master_presence_band["bandwidth_hz"] is not None
        else 0.0
    )
    master_low_end_band = (
        _peak_band_details_in_band(mono, sample_rate, 55.0, 180.0)
        if source_family == "master" and low_end_excess_score >= 0.55
        else {"peak_frequency_hz": None, "bandwidth_hz": None}
    )
    master_low_end_frequency_hz = master_low_end_band["peak_frequency_hz"]
    master_low_end_bandwidth_hz = (
        round(float(master_low_end_band["bandwidth_hz"]), 1)
        if master_low_end_frequency_hz is not None
        and master_low_end_band["bandwidth_hz"] is not None
        else 0.0
    )
    sibilance_frequency = None
    sibilance_bandwidth_hz = None
    sibilance_event = {
        "sibilance_event_peak_dbfs": 0.0,
        "sibilance_event_contrast_db": 0.0,
        "sibilance_event_rise_time_ms": 0.0,
        "sibilance_event_recovery_ms": 0.0,
    }
    if sibilance_score >= 0.25:
        sibilance_band = _peak_band_details_in_band(mono, sample_rate, 5000.0, 10000.0)
        sibilance_frequency = sibilance_band["peak_frequency_hz"]
        sibilance_bandwidth_hz = sibilance_band["bandwidth_hz"]
        sibilance_event = _sibilance_event_evidence(mono, sample_rate)
    dynamics_variance_score = round(
        _clip01((lra_proxy_db - 7.0) / 20.0 + max(0.0, crest_factor_db - 14.0) / 18.0),
        3,
    )
    dynamics_envelope = _dynamics_envelope_evidence(mono, sample_rate)

    phase = _phase_metrics(samples)
    low_band_burst = _low_band_burst_evidence(mono, sample_rate)
    plosive_risk_score = (
        round(_clip01(low_band_burst["plosive_risk_score"]), 3)
        if source_family == "vocal"
        else 0.0
    )
    plosive_frequency_hz = (
        low_band_burst["plosive_frequency_hz"]
        if source_family == "vocal" and plosive_risk_score >= 0.25
        else None
    )
    plosive_event_peak_dbfs = (
        float(low_band_burst["plosive_event_peak_dbfs"])
        if source_family == "vocal" and plosive_risk_score >= 0.25
        else 0.0
    )
    plosive_event_contrast_db = (
        float(low_band_burst["plosive_event_contrast_db"])
        if source_family == "vocal" and plosive_risk_score >= 0.25
        else 0.0
    )
    plosive_event_rise_time_ms = (
        float(low_band_burst["plosive_event_rise_time_ms"])
        if source_family == "vocal" and plosive_risk_score >= 0.25
        else 0.0
    )
    plosive_event_recovery_ms = (
        float(low_band_burst["plosive_event_recovery_ms"])
        if source_family == "vocal" and plosive_risk_score >= 0.25
        else 0.0
    )
    phrase_noise = _vocal_phrase_noise_floor_evidence(mono, sample_rate, frame_rms)
    vocal_noise_floor_score = (
        round(_clip01(phrase_noise["vocal_noise_floor_score"]), 3)
        if source_family == "vocal"
        else 0.0
    )
    phrase_gap_noise_floor_dbfs = (
        phrase_noise["phrase_gap_noise_floor_dbfs"]
        if source_family == "vocal" and vocal_noise_floor_score >= 0.25
        else None
    )
    phrase_gap_noise_flatness = (
        float(phrase_noise["phrase_gap_noise_flatness"])
        if source_family == "vocal"
        else 0.0
    )
    phrase_gap_contrast_db = (
        float(phrase_noise["phrase_gap_contrast_db"])
        if source_family == "vocal"
        else 0.0
    )
    phrase_gap_median_ms = (
        float(phrase_noise["phrase_gap_median_ms"])
        if source_family == "vocal"
        else 0.0
    )
    phrase_gap_coverage_ratio = (
        float(phrase_noise["phrase_gap_coverage_ratio"])
        if source_family == "vocal"
        else 0.0
    )
    line_noise = _line_noise_evidence(mono, sample_rate)
    line_noise_supported_family = source_family in {"vocal", "guitar", "keys", "playback"}
    line_noise_score = (
        round(_clip01(line_noise["line_noise_score"]), 3)
        if line_noise_supported_family
        else 0.0
    )
    line_noise_frequency_hz = (
        line_noise["line_noise_frequency_hz"]
        if line_noise_supported_family and line_noise_score >= 0.25
        else None
    )
    line_noise_prominence_db = (
        float(line_noise["line_noise_prominence_db"])
        if line_noise_supported_family
        else 0.0
    )
    line_noise_harmonic_count = (
        int(line_noise["line_noise_harmonic_count"])
        if line_noise_supported_family
        else 0
    )
    line_noise_harmonics = (
        tuple(dict(item) for item in line_noise.get("line_noise_harmonics", ()))
        if line_noise_supported_family
        else ()
    )

    bleed_risk = 0.0
    tail_risk = 0.0
    chatter_risk = 0.0
    gate_hit = {
        "gate_hit_peak_dbfs": 0.0,
        "gate_hit_to_bleed_margin_db": 0.0,
        "gate_tail_decay_ms": 0.0,
    }
    if source_family in {"kick", "snare", "tom", "drum"}:
        bleed_risk = _clip01((noise_floor_dbfs + 66.0) / 20.0)
        tail_risk = _clip01((noise_floor_dbfs + 62.0) / 22.0 + lra_proxy_db / 60.0)
        chatter_risk = _clip01((transient["transient_density"] - 5.0) / 10.0 + bleed_risk * 0.35)
        gate_hit = _drum_gate_hit_evidence(mono, sample_rate, noise_floor_dbfs)

    feedback_ring = _feedback_ring_evidence(
        mono,
        sample_rate,
        source_family=source_family,
        high_ratio_db=high_ratio_db,
        sibilance_score=sibilance_score,
        monitor_feedback_risk=monitor_feedback_risk,
    )
    feedback_frequency_confidence = _clip01(feedback_ring.get("confidence"))
    feedback_ring_risk = (
        feedback_frequency_confidence * 0.42
        if feedback_ring.get("source") == "measured_feedback_ring_peak"
        else 0.0
    )
    high_ratio_feedback_component = (
        max(0.0, high_ratio_db) / 24.0
        if source_family not in {"guitar", "keys", "playback"}
        or _clip01(_safe_float(monitor_feedback_risk)) >= 0.3
        else 0.0
    )
    feedback_risk = _clip01(
        high_ratio_feedback_component
        + (0.18 if source_family == "vocal" and sibilance_score > 0.65 else 0.0)
        + _clip01(_safe_float(monitor_feedback_risk)) * 0.35
        + feedback_ring_risk
    )
    drum_ring = _drum_ring_resonance_evidence(
        mono,
        sample_rate,
        source_family=source_family,
        tail_persistence=tail_persistence,
        transient=transient,
    )
    drum_ring_resonance_score = round(_clip01(drum_ring["score"]), 3)
    drum_ring_frequency_hz = drum_ring["frequency_hz"]
    drum_ring_decay_score = round(_clip01(drum_ring["decay_score"]), 3)
    drum_ring_bandwidth_hz = round(float(drum_ring["bandwidth_hz"]), 1)
    drum_ring_peak_dbfs = round(float(drum_ring["peak_dbfs"]), 1)
    drum_ring_decay_ms = round(float(drum_ring["decay_ms"]), 1)
    stereo_balance = _stereo_balance_metrics(
        samples,
        current_pan_percent=current_pan_percent,
    )

    masking_score = 0.0
    masking_frequency_hz = None
    masking_conflict_role = ""
    masking_conflict_target_id = ""
    arrangement_overlap_score = 0.0
    masking = {
        "masking_bandwidth_hz": 0.0,
        "masking_overlap_peak_share": 0.0,
    }
    if reference_samples is not None and reference_role:
        masking = _masking_evidence(mono, reference_samples, sample_rate)
        masking_score = float(masking["masking_score"])
        masking_frequency_hz = masking["masking_frequency_hz"]
        arrangement_overlap_score = float(masking.get("arrangement_overlap_score", 0.0))
        if arrangement_overlap_score > 0.0:
            reference_mono, _ = _mono_and_channel_count(reference_samples)
            reference_transient = _transient_metrics(reference_mono, sample_rate)
            source_groove_factor = max(
                _clip01(transient["transient_strength"]),
                _clip01(transient["transient_density"] / 4.0),
            )
            reference_groove_factor = max(
                _clip01(reference_transient["transient_strength"]),
                _clip01(reference_transient["transient_density"] / 4.0),
            )
            low_end_pair = (
                source_family in {"kick", "bass"}
                and classify_source_family(str(reference_role)) in {"kick", "bass"}
            )
            if low_end_pair:
                arrangement_overlap_score = _clip01(
                    arrangement_overlap_score
                    * (1.0 - 0.75 * max(source_groove_factor, reference_groove_factor))
                )
            else:
                arrangement_overlap_score = 0.0
        if masking_score >= 0.45:
            masking_conflict_role = str(reference_role)
            masking_conflict_target_id = str(reference_target_id or "").strip()

    fx_wetness_score = 0.0
    fx_decay_risk_score = 0.0
    fx_delay_smear_score = 0.0
    fx_tail_decay_seconds = 0.0
    fx_delay_repeat_ms = 0.0
    if source_family == "fx_return":
        fx_wetness_score = _clip01(
            0.35
            + float(share.get("air", 0.0)) * 1.2
            + float(share.get("sibilance", 0.0)) * 0.8
            + max(0.0, lra_proxy_db - 10.0) / 28.0
        )
        fx_decay_risk_score = _clip01(
            0.18 + tail_persistence * 0.72 + max(0.0, lra_proxy_db - 8.0) / 36.0
        )
        fx_delay_smear_score = _clip01(
            0.12
            + max(0.0, transient["transient_density"] - 1.2) / 6.0
            + tail_persistence * 0.35
        )
        fx_tail_decay_seconds = _fx_tail_decay_seconds(frame_rms)
        fx_delay_repeat_ms = _fx_delay_repeat_ms(mono, sample_rate)

    master_true_peak_db = true_peak if source_family == "master" else -6.0
    problem_hints = list(extra_problem_hints)
    if identity_mismatch["score"] >= SOURCE_IDENTITY_MISMATCH_BLOCK_THRESHOLD:
        problem_hints.append("source_identity_mismatch")
    if muddiness_score >= 0.45:
        problem_hints.append("muddy")
    if source_family == "vocal" and sibilance_score >= 0.45:
        problem_hints.append("sibilant")
    if harshness_score >= 0.55:
        problem_hints.append("harsh")
    if (
        source_family in {"vocal", "guitar", "keys"}
        and presence_deficit_score >= 0.65
        and muddiness_score < 0.45
        and harshness_score < 0.55
    ):
        problem_hints.append("dull_presence")
    if source_family == "vocal" and plosive_risk_score >= 0.55:
        problem_hints.append("plosive_control")
    if source_family == "vocal" and vocal_noise_floor_score >= 0.55:
        problem_hints.append("vocal_noise_floor")
    if line_noise_supported_family and line_noise_score >= 0.55:
        problem_hints.append("line_noise")
    if (
        source_family in {"kick", "bass"}
        and low_end_excess_score >= 0.62
        and abs(level_plane_error_lu) < 3.5
        and not masking_conflict_role
        and not (
            source_family == "kick"
            and (
                transient["transient_strength"] >= 0.55
                or transient["transient_density"] >= 3.0
                or crest_factor_db >= 20.0
            )
        )
    ):
        problem_hints.append("low_end_resonance")
    if (
        source_family in {"snare", "tom"}
        and drum_ring_resonance_score >= 0.55
        and drum_ring_frequency_hz is not None
        and abs(level_plane_error_lu) < 3.0
        and not (
            source_family == "tom"
            and (bleed_risk >= 0.35 or noise_floor_dbfs > -58.0)
        )
        and not (
            transient["transient_strength"] >= 0.65
            and transient["transient_density"] >= 2.5
            and drum_ring_decay_score < 0.55
        )
        and not (
            feedback_risk >= HIGH_FEEDBACK_RISK
            and feedback_ring.get("source") == "measured_feedback_ring_peak"
        )
    ):
        problem_hints.append("drum_ring_resonance")
    if source_family in {"vocal", "bass", "guitar", "keys"} and dynamics_variance_score >= 0.45:
        problem_hints.append("dynamic_control")
    if source_family in {"kick", "snare", "drum"} and (
        transient["transient_strength"] >= 0.45
        or transient["transient_density"] >= 3.0
        or crest_factor_db >= 18.0
    ):
        problem_hints.append("transient_control")
    if _is_group_or_stem_source(effective_source_role, str(source_label), str(target_id)) and (
        abs(level_plane_error_lu) >= 1.5
        or true_peak_headroom_db < 2.0
        or low_end_excess_score >= 0.65
        or spectral_tilt_score >= 0.65
        or dynamics_variance_score >= 0.55
        or lra_proxy_db >= 10.0
    ):
        problem_hints.append("bus_group_impact")
    if channel_count >= 2 and (
        phase["phase_correlation"] < MIN_PHASE_CORRELATION_FOR_WIDTH
        or phase["mono_sum_delta_db"] <= MAX_MONO_SUM_LOSS_DB
    ):
        problem_hints.append("phase_mono_risk")
    if source_family != "master" and abs(level_plane_error_lu) >= 2.0:
        problem_hints.append("level_plane_balance")
    if masking_conflict_role:
        problem_hints.append("masking")
    if arrangement_overlap_score >= 0.68:
        problem_hints.append("arrangement_overlap")
    if source_family in {"kick", "snare", "tom", "drum"} and (bleed_risk >= 0.45 or noise_floor_dbfs > -58.0):
        problem_hints.append("noisy_drum")
    if fx_wetness_score >= 0.45:
        problem_hints.append("fx_intelligibility")
    if source_family == "fx_return" and (
        fx_decay_risk_score >= 0.45 or fx_delay_smear_score >= 0.45
    ):
        problem_hints.append("fx_intelligibility")
    if abs(_safe_float(desired_stereo_width_request)) > 1.0:
        problem_hints.append("pan_width")
    if (
        channel_count >= 2
        and source_family not in {"vocal", "kick", "snare", "bass", "master"}
        and stereo_balance["stereo_balance_confidence"] >= 0.35
        and abs(stereo_balance["measured_pan_delta_percent"]) >= 5.0
    ):
        problem_hints.append("pan_width")
    if source_family == "master" or master_true_peak_db > -1.5:
        problem_hints.append("master_overload")

    duration_factor = _clip01(duration_sec / 2.0)
    level_factor = _clip01((rms_dbfs + 70.0) / 45.0)
    identity_hint_factor = 0.72 if source_family != "unknown" else 0.45
    evidence_confidence = round(
        _clip(
            0.2 + 0.34 * duration_factor + 0.28 * level_factor + 0.18 * identity_hint_factor,
            0.0,
            0.92,
        ),
        3,
    )
    if warnings:
        evidence_confidence = round(max(0.0, evidence_confidence - 0.12), 3)

    return SmartDSPAudioEvidence(
        target_id=str(target_id),
        source_role=effective_source_role,
        source_label=str(source_label),
        musical_role=musical_role,
        declared_source_role=declared_source_role,
        declared_source_family=declared_source_family,
        inferred_source_role=str(inference["inferred_source_role"]),
        inferred_source_family=str(inference["inferred_source_family"]),
        source_identity_inference_confidence=round(
            float(inference["source_identity_inference_confidence"]),
            3,
        ),
        source_identity_inference_reason=str(inference["source_identity_inference_reason"]),
        source_identity_inference_used=bool(inference["source_identity_inference_used"]),
        source_identity_mismatch_score=round(float(identity_mismatch["score"]), 3),
        source_identity_mismatch_reason=str(identity_mismatch["reason"]),
        source_identity_mismatch_family=str(identity_mismatch["family"]),
        mode=normalize_mode(mode),
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        channel_count=channel_count,
        peak_dbfs=peak_dbfs,
        true_peak_dbfs=true_peak,
        true_peak_headroom_db=true_peak_headroom_db,
        rms_dbfs=rms_dbfs,
        estimated_lufs=estimated_lufs,
        crest_factor_db=crest_factor_db,
        lra_proxy_db=lra_proxy_db,
        noise_floor_dbfs=noise_floor_dbfs,
        spectral_centroid_hz=float(spectral["spectral_centroid_hz"]),
        dominant_frequency_hz=float(spectral["dominant_frequency_hz"]),
        band_energy_dbfs=dict(spectral["band_energy_dbfs"]),
        band_energy_share=dict(share),
        low_end_excess_score=low_end_excess_score,
        spectral_tilt_score=spectral_tilt_score,
        low_end_resonance_frequency_hz=(
            low_end_resonance["low_end_resonance_frequency_hz"] or None
        ),
        low_end_resonance_bandwidth_hz=low_end_resonance[
            "low_end_resonance_bandwidth_hz"
        ],
        low_end_resonance_peak_dbfs=low_end_resonance[
            "low_end_resonance_peak_dbfs"
        ],
        low_end_resonance_sustain_ms=low_end_resonance[
            "low_end_resonance_sustain_ms"
        ],
        presence_deficit_score=presence_deficit_score,
        presence_lift_frequency_hz=presence_lift["presence_lift_frequency_hz"],
        presence_lift_bandwidth_hz=presence_lift["presence_lift_bandwidth_hz"],
        presence_lift_parameter_source=presence_lift[
            "presence_lift_parameter_source"
        ],
        muddiness_score=muddiness_score,
        mud_frequency_hz=mud_frequency_hz,
        low_mid_bloom_frequency_hz=low_mid_bloom["low_mid_bloom_frequency_hz"],
        low_mid_bloom_db=low_mid_bloom["low_mid_bloom_db"],
        low_mid_bloom_peak_dbfs=low_mid_bloom["low_mid_bloom_peak_dbfs"],
        low_mid_bloom_rise_time_ms=low_mid_bloom["low_mid_bloom_rise_time_ms"],
        low_mid_bloom_recovery_ms=low_mid_bloom["low_mid_bloom_recovery_ms"],
        plosive_risk_score=plosive_risk_score,
        plosive_frequency_hz=plosive_frequency_hz,
        plosive_event_peak_dbfs=round(float(plosive_event_peak_dbfs), 1),
        plosive_event_contrast_db=round(float(plosive_event_contrast_db), 1),
        plosive_event_rise_time_ms=round(float(plosive_event_rise_time_ms), 1),
        plosive_event_recovery_ms=round(float(plosive_event_recovery_ms), 1),
        harshness_score=harshness_score,
        presence_harshness_score=presence_harshness_score,
        instrument_harsh_frequency_hz=instrument_harsh_frequency_hz,
        instrument_harsh_bandwidth_hz=instrument_harsh_bandwidth_hz,
        instrument_harsh_event_peak_dbfs=instrument_harsh_event_peak_dbfs,
        instrument_harsh_event_contrast_db=instrument_harsh_event_contrast_db,
        instrument_harsh_event_recovery_ms=instrument_harsh_event_recovery_ms,
        drum_harsh_frequency_hz=drum_harsh_frequency_hz,
        drum_harsh_bandwidth_hz=drum_harsh_bandwidth_hz,
        drum_harsh_event_peak_dbfs=drum_harsh_event_peak_dbfs,
        drum_harsh_event_contrast_db=drum_harsh_event_contrast_db,
        drum_harsh_event_recovery_ms=drum_harsh_event_recovery_ms,
        sibilance_score=sibilance_score,
        sibilance_frequency_hz=sibilance_frequency,
        sibilance_bandwidth_hz=sibilance_bandwidth_hz,
        sibilance_event_peak_dbfs=sibilance_event["sibilance_event_peak_dbfs"],
        sibilance_event_contrast_db=sibilance_event["sibilance_event_contrast_db"],
        sibilance_event_rise_time_ms=sibilance_event[
            "sibilance_event_rise_time_ms"
        ],
        sibilance_event_recovery_ms=sibilance_event[
            "sibilance_event_recovery_ms"
        ],
        dynamics_variance_score=dynamics_variance_score,
        dynamics_peak_to_median_db=dynamics_envelope["dynamics_peak_to_median_db"],
        dynamics_rise_time_ms=dynamics_envelope["dynamics_rise_time_ms"],
        dynamics_recovery_ms=dynamics_envelope["dynamics_recovery_ms"],
        vocal_noise_floor_score=vocal_noise_floor_score,
        phrase_gap_noise_floor_dbfs=phrase_gap_noise_floor_dbfs,
        phrase_gap_noise_flatness=round(float(phrase_gap_noise_flatness), 6),
        phrase_gap_contrast_db=round(float(phrase_gap_contrast_db), 2),
        phrase_gap_median_ms=round(float(phrase_gap_median_ms), 1),
        phrase_gap_coverage_ratio=round(float(phrase_gap_coverage_ratio), 3),
        line_noise_score=line_noise_score,
        line_noise_frequency_hz=line_noise_frequency_hz,
        line_noise_prominence_db=round(float(line_noise_prominence_db), 2),
        line_noise_harmonic_count=int(line_noise_harmonic_count),
        line_noise_harmonics=line_noise_harmonics,
        drum_ring_resonance_score=drum_ring_resonance_score,
        drum_ring_frequency_hz=drum_ring_frequency_hz,
        drum_ring_decay_score=drum_ring_decay_score,
        drum_ring_bandwidth_hz=drum_ring_bandwidth_hz,
        drum_ring_peak_dbfs=drum_ring_peak_dbfs,
        drum_ring_decay_ms=drum_ring_decay_ms,
        transient_density=transient["transient_density"],
        transient_strength=transient["transient_strength"],
        transient_peak_dbfs=transient["transient_peak_dbfs"],
        transient_rise_time_ms=transient["transient_rise_time_ms"],
        transient_decay_ms=transient["transient_decay_ms"],
        bleed_risk=round(float(bleed_risk), 3),
        tail_risk=round(float(tail_risk), 3),
        chatter_risk=round(float(chatter_risk), 3),
        gate_hit_peak_dbfs=gate_hit["gate_hit_peak_dbfs"],
        gate_hit_to_bleed_margin_db=gate_hit["gate_hit_to_bleed_margin_db"],
        gate_tail_decay_ms=gate_hit["gate_tail_decay_ms"],
        phase_correlation=phase["phase_correlation"],
        mono_sum_delta_db=phase["mono_sum_delta_db"],
        stereo_balance_db=stereo_balance["stereo_balance_db"],
        stereo_balance_confidence=stereo_balance["stereo_balance_confidence"],
        measured_pan_target_percent=stereo_balance["measured_pan_target_percent"],
        measured_pan_delta_percent=stereo_balance["measured_pan_delta_percent"],
        feedback_risk=round(float(feedback_risk), 3),
        feedback_frequency_hz=feedback_ring["frequency_hz"],
        feedback_frequency_confidence=round(float(feedback_frequency_confidence), 3),
        feedback_frequency_source=str(feedback_ring["source"]),
        masking_conflict_role=masking_conflict_role,
        masking_conflict_target_id=masking_conflict_target_id,
        masking_frequency_hz=masking_frequency_hz,
        masking_score=round(float(masking_score), 3),
        masking_bandwidth_hz=round(
            float(masking.get("masking_bandwidth_hz", 0.0)),
            1,
        ),
        masking_overlap_peak_share=round(
            float(masking.get("masking_overlap_peak_share", 0.0)),
            3,
        ),
        arrangement_overlap_score=round(float(arrangement_overlap_score), 3),
        fx_wetness_score=round(float(fx_wetness_score), 3),
        fx_decay_risk_score=round(float(fx_decay_risk_score), 3),
        fx_delay_smear_score=round(float(fx_delay_smear_score), 3),
        fx_tail_decay_seconds=round(float(fx_tail_decay_seconds), 2),
        fx_delay_repeat_ms=round(float(fx_delay_repeat_ms), 1),
        fx_return_headroom_db=round(
            _safe_float(fx_return_headroom_db, true_peak_headroom_db),
            2,
        ),
        fx_return_target_id=str(fx_return_target_id or "").strip(),
        monitor_feedback_risk=round(_clip01(monitor_feedback_risk), 3),
        master_true_peak_db=round(float(master_true_peak_db), 2),
        master_integrated_lufs=estimated_lufs if source_family == "master" else -23.0,
        master_presence_frequency_hz=master_presence_frequency_hz,
        master_presence_bandwidth_hz=master_presence_bandwidth_hz,
        master_low_end_frequency_hz=master_low_end_frequency_hz,
        master_low_end_bandwidth_hz=master_low_end_bandwidth_hz,
        target_plane_lufs=resolved_target_plane_lufs,
        level_plane_error_lu=level_plane_error_lu,
        summed_bus_evidence_available=bool(summed_bus_evidence_available),
        summed_bus_role=summed_bus_role_text if summed_bus_evidence_available else "",
        summed_bus_target_id=(
            summed_bus_target_id_text if summed_bus_evidence_available else ""
        ),
        summed_bus_true_peak_headroom_db=round(float(summed_bus_true_peak_headroom_db), 2),
        summed_bus_estimated_lufs=round(float(summed_bus_estimated_lufs), 2),
        summed_bus_target_plane_lufs=round(float(summed_bus_target_lufs), 1),
        summed_bus_level_plane_error_lu=round(float(summed_bus_level_plane_error_lu), 2),
        evidence_confidence=evidence_confidence,
        problem_hints=_ordered_unique(problem_hints),
        extraction_warnings=tuple(warnings),
    )


def build_smart_dsp_analysis_input_from_audio(
    samples: Any,
    *,
    target_id: str,
    source_role: str,
    source_label: str = "",
    mode: str = "soundcheck",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    source_identity_confirmed: bool = False,
    source_identity_confidence: float = 0.0,
    current_state_cache_freshness: Optional[Mapping[str, Any]] = None,
    current_state_readback_available: bool = False,
    current_wing_state: Optional[Mapping[str, Any]] = None,
    current_wing_eq_model: str = "",
    current_wing_eq_enabled: Optional[bool] = None,
    current_wing_eq_model_readback_available: Optional[bool] = None,
    current_wing_eq_operator_owned: Optional[bool] = None,
    current_wing_eq_filter_operator_owned: Optional[bool] = None,
    current_wing_dynamics_model: str = "",
    current_wing_dynamics_enabled: Optional[bool] = None,
    current_wing_dynamics_model_readback_available: Optional[bool] = None,
    current_wing_dynamics_operator_owned: Optional[bool] = None,
    current_wing_gate_model: str = "",
    current_wing_gate_enabled: Optional[bool] = None,
    current_wing_gate_slot_readback_available: Optional[bool] = None,
    current_wing_gate_slot_operator_owned: Optional[bool] = None,
    current_wing_fx_model: str = "",
    current_wing_fx_model_readback_available: Optional[bool] = None,
    current_wing_fx_operator_owned: Optional[bool] = None,
    current_wing_fx_level_operator_owned: Optional[bool] = None,
    current_wing_pan_width_operator_owned: Optional[bool] = None,
    current_wing_mute_operator_owned: Optional[bool] = None,
    current_wing_gain_operator_owned: Optional[bool] = None,
    current_wing_fader_db: Optional[float] = None,
    current_wing_input_trim_db: Optional[float] = None,
    current_wing_pan_percent: Optional[float] = None,
    rollback_plan_available: bool = False,
    operator_intent: str = "",
    reference_samples: Any = None,
    reference_role: str = "",
    reference_target_id: str = "",
    summed_bus_samples: Any = None,
    summed_bus_role: str = "",
    summed_bus_target_id: str = "",
    summed_bus_target_plane_lufs: Optional[float] = None,
    desired_stereo_width_request: float = 0.0,
    fx_return_headroom_db: Optional[float] = None,
    fx_return_target_id: str = "",
    monitor_feedback_risk: float = 0.0,
    target_plane_lufs: Optional[float] = None,
    extra_problem_hints: Iterable[str] = (),
    current_pan_percent: float = 0.0,
    allow_inferred_source_identity: bool = False,
    active_performance: bool = False,
) -> SmartDSPAnalysisInput:
    wing_state = dict(current_wing_state or {})
    resolved_current_wing_pan_percent = _resolve_optional_float_state(
        current_wing_pan_percent,
        wing_state,
        (
            "current_wing_pan_percent",
            "wing_readonly_pan_percent",
            "wing_pan_percent",
            "pan_percent",
            "pan",
        ),
        minimum=-100.0,
        maximum=100.0,
    )
    resolved_current_pan_percent = (
        resolved_current_wing_pan_percent
        if resolved_current_wing_pan_percent is not None
        else _safe_float(current_pan_percent)
    )
    evidence = extract_smart_dsp_audio_evidence(
        samples,
        target_id=target_id,
        source_role=source_role,
        source_label=source_label,
        mode=mode,
        sample_rate=sample_rate,
        reference_samples=reference_samples,
        reference_role=reference_role,
        reference_target_id=reference_target_id,
        summed_bus_samples=summed_bus_samples,
        summed_bus_role=summed_bus_role,
        summed_bus_target_id=summed_bus_target_id,
        summed_bus_target_plane_lufs=summed_bus_target_plane_lufs,
        desired_stereo_width_request=desired_stereo_width_request,
        current_pan_percent=resolved_current_pan_percent,
        fx_return_headroom_db=fx_return_headroom_db,
        fx_return_target_id=fx_return_target_id,
        monitor_feedback_risk=monitor_feedback_risk,
        target_plane_lufs=target_plane_lufs,
        extra_problem_hints=extra_problem_hints,
    )
    source_identity_confidence = _clip01(source_identity_confidence)
    if source_identity_confirmed and source_identity_confidence <= 0.0:
        source_identity_confidence = 0.72
    inference_confidence = _clip01(evidence.source_identity_inference_confidence)
    inferred_identity_confirmed = (
        bool(allow_inferred_source_identity)
        and evidence.source_identity_inference_used
        and inference_confidence >= MIN_SOURCE_IDENTITY_CONFIDENCE
    )
    if inferred_identity_confirmed and not source_identity_confirmed:
        source_identity_confirmed = True
        source_identity_confidence = max(source_identity_confidence, inference_confidence)

    evidence_field_count = len(evidence.to_dict())
    freshness = dict(
        current_state_cache_freshness
        or _freshness_from_current_wing_state(
            wing_state,
            fallback_field_count=evidence_field_count,
        )
        or _read_only_freshness(field_count=evidence_field_count)
    )
    wing_state_rejected = _current_wing_state_rejected(wing_state)
    resolved_current_state_readback_available = bool(
        current_state_readback_available
        or _current_wing_state_readback_available(wing_state)
    ) and not wing_state_rejected
    resolved_rollback_plan_available = bool(
        rollback_plan_available
        or _current_wing_state_rollback_plan_available(wing_state)
    ) and not wing_state_rejected
    resolved_eq_filter_operator_owned = _resolve_bool_state(
        current_wing_eq_filter_operator_owned,
        {
            **wing_state,
            "current_wing_eq_filter_operator_owned": (
                _current_wing_eq_filter_operator_owned_from_state(
                    wing_state,
                    target_id=evidence.target_id,
                )
            ),
        },
        ("current_wing_eq_filter_operator_owned",),
    )
    resolved_eq_operator_owned = _resolve_bool_state(
        current_wing_eq_operator_owned,
        {
            **wing_state,
            "current_wing_eq_operator_owned": (
                _current_wing_eq_operator_owned_from_state(
                    wing_state,
                    target_id=evidence.target_id,
                )
            ),
        },
        ("current_wing_eq_operator_owned",),
    )
    resolved_dynamics_operator_owned = _resolve_bool_state(
        current_wing_dynamics_operator_owned,
        {
            **wing_state,
            "current_wing_dynamics_operator_owned": (
                _current_wing_dynamics_operator_owned_from_state(
                    wing_state,
                    target_id=evidence.target_id,
                )
            ),
        },
        ("current_wing_dynamics_operator_owned",),
    )
    resolved_fx_level_operator_owned = _resolve_bool_state(
        current_wing_fx_level_operator_owned,
        {
            **wing_state,
            "current_wing_fx_level_operator_owned": (
                _current_wing_fx_level_operator_owned_from_state(
                    wing_state,
                    target_id=evidence.target_id,
                    fx_return_target_id=str(fx_return_target_id or "").strip(),
                )
            ),
        },
        ("current_wing_fx_level_operator_owned",),
    )
    resolved_fx_operator_owned = _resolve_bool_state(
        current_wing_fx_operator_owned,
        {
            **wing_state,
            "current_wing_fx_operator_owned": (
                _current_wing_fx_operator_owned_from_state(
                    wing_state,
                    target_id=str(fx_return_target_id or evidence.target_id).strip(),
                )
            ),
        },
        ("current_wing_fx_operator_owned",),
    )
    resolved_gate_slot_operator_owned = _resolve_bool_state(
        current_wing_gate_slot_operator_owned,
        {
            **wing_state,
            "current_wing_gate_slot_operator_owned": (
                _current_wing_gate_slot_operator_owned_from_state(
                    wing_state,
                    target_id=evidence.target_id,
                )
            ),
        },
        ("current_wing_gate_slot_operator_owned",),
    )
    resolved_pan_width_operator_owned = _resolve_bool_state(
        current_wing_pan_width_operator_owned,
        {
            **wing_state,
            "current_wing_pan_width_operator_owned": (
                _current_wing_pan_width_operator_owned_from_state(
                    wing_state,
                    target_id=evidence.target_id,
                )
            ),
        },
        ("current_wing_pan_width_operator_owned",),
    )
    resolved_mute_operator_owned = _resolve_bool_state(
        current_wing_mute_operator_owned,
        {
            **wing_state,
            "current_wing_mute_operator_owned": (
                _current_wing_mute_operator_owned_from_state(
                    wing_state,
                    target_id=evidence.target_id,
                )
            ),
        },
        ("current_wing_mute_operator_owned",),
    )
    resolved_gain_operator_owned = _resolve_bool_state(
        current_wing_gain_operator_owned,
        {
            **wing_state,
            "current_wing_gain_operator_owned": (
                _current_wing_gain_operator_owned_from_state(
                    wing_state,
                    target_id=evidence.target_id,
                )
            ),
        },
        ("current_wing_gain_operator_owned",),
    )
    resolved_current_wing_fader_db = _resolve_optional_float_state(
        current_wing_fader_db,
        wing_state,
        ("current_wing_fader_db", "wing_readonly_fader_db", "fader_db", "fader"),
    )
    resolved_current_wing_input_trim_db = _resolve_optional_float_state(
        current_wing_input_trim_db,
        wing_state,
        (
            "current_wing_input_trim_db",
            "wing_readonly_input_trim_db",
            "input_trim_db",
            "trim_db",
            "gain_db",
            "gain",
        ),
    )
    resolved_control_readback_available = bool(
        _resolve_bool_state(
            None,
            wing_state,
            (
                "current_wing_state_basic_controls_readback_available",
                "current_wing_fader_readback_available",
                "current_wing_gain_readback_available",
                "current_wing_pan_readback_available",
            ),
        )
        or resolved_current_wing_fader_db is not None
        or resolved_current_wing_input_trim_db is not None
        or resolved_current_wing_pan_percent is not None
    )

    return SmartDSPAnalysisInput(
        target_id=evidence.target_id,
        source_role=evidence.source_role,
        source_label=evidence.source_label,
        musical_role=evidence.musical_role,
        declared_source_role=evidence.declared_source_role,
        declared_source_family=evidence.declared_source_family,
        inferred_source_role=evidence.inferred_source_role,
        inferred_source_family=evidence.inferred_source_family,
        source_identity_inference_confidence=evidence.source_identity_inference_confidence,
        source_identity_inference_reason=evidence.source_identity_inference_reason,
        source_identity_inference_used=evidence.source_identity_inference_used,
        source_identity_mismatch_score=evidence.source_identity_mismatch_score,
        source_identity_mismatch_reason=evidence.source_identity_mismatch_reason,
        source_identity_mismatch_family=evidence.source_identity_mismatch_family,
        mode=evidence.mode,
        problem_hints=evidence.problem_hints,
        operator_intent=str(operator_intent),
        source_identity_confirmed=bool(source_identity_confirmed),
        source_identity_confidence=source_identity_confidence,
        evidence_confidence=evidence.evidence_confidence,
        current_state_cache_freshness=freshness,
        current_state_readback_available=resolved_current_state_readback_available,
        current_wing_eq_model=_resolve_string_state(
            current_wing_eq_model,
            wing_state,
            ("current_wing_eq_model", "wing_eq_model", "eq_model"),
        ),
        current_wing_eq_enabled=_resolve_optional_bool_state(
            current_wing_eq_enabled,
            wing_state,
            ("current_wing_eq_enabled", "wing_eq_enabled", "eq_enabled", "eq_on"),
        ),
        current_wing_eq_model_readback_available=_resolve_bool_state(
            current_wing_eq_model_readback_available,
            wing_state,
            (
                "current_wing_eq_model_readback_available",
                "wing_eq_model_readback_available",
                "eq_model_readback_available",
            ),
        ),
        current_wing_eq_operator_owned=resolved_eq_operator_owned,
        current_wing_eq_filter_operator_owned=resolved_eq_filter_operator_owned,
        current_wing_dynamics_model=_resolve_string_state(
            current_wing_dynamics_model,
            wing_state,
            (
                "current_wing_dynamics_model",
                "wing_dynamics_model",
                "dynamics_model",
                "compressor_model",
            ),
        ),
        current_wing_dynamics_enabled=_resolve_optional_bool_state(
            current_wing_dynamics_enabled,
            wing_state,
            (
                "current_wing_dynamics_enabled",
                "wing_dynamics_enabled",
                "dynamics_enabled",
                "compressor_enabled",
                "dyn_on",
            ),
        ),
        current_wing_dynamics_model_readback_available=_resolve_bool_state(
            current_wing_dynamics_model_readback_available,
            wing_state,
            (
                "current_wing_dynamics_model_readback_available",
                "wing_dynamics_model_readback_available",
                "dynamics_model_readback_available",
                "compressor_model_readback_available",
            ),
        ),
        current_wing_dynamics_operator_owned=resolved_dynamics_operator_owned,
        current_wing_gate_model=_resolve_string_state(
            current_wing_gate_model,
            wing_state,
            ("current_wing_gate_model", "wing_gate_model", "gate_model"),
        ),
        current_wing_gate_enabled=_resolve_optional_bool_state(
            current_wing_gate_enabled,
            wing_state,
            (
                "current_wing_gate_enabled",
                "wing_gate_enabled",
                "gate_enabled",
                "gate_on",
            ),
        ),
        current_wing_gate_slot_readback_available=_resolve_bool_state(
            current_wing_gate_slot_readback_available,
            wing_state,
            (
                "current_wing_gate_slot_readback_available",
                "wing_gate_slot_readback_available",
                "gate_slot_readback_available",
            ),
        ),
        current_wing_gate_slot_operator_owned=resolved_gate_slot_operator_owned,
        current_wing_fx_model=_resolve_string_state(
            current_wing_fx_model,
            wing_state,
            ("current_wing_fx_model", "wing_fx_model", "fx_model"),
        ),
        current_wing_fx_model_readback_available=_resolve_bool_state(
            current_wing_fx_model_readback_available,
            wing_state,
            (
                "current_wing_fx_model_readback_available",
                "wing_fx_model_readback_available",
                "fx_model_readback_available",
            ),
        ),
        current_wing_fx_operator_owned=resolved_fx_operator_owned,
        current_wing_fx_level_operator_owned=resolved_fx_level_operator_owned,
        current_wing_pan_width_operator_owned=resolved_pan_width_operator_owned,
        current_wing_mute_operator_owned=resolved_mute_operator_owned,
        current_wing_gain_operator_owned=resolved_gain_operator_owned,
        current_wing_fader_db=resolved_current_wing_fader_db,
        current_wing_input_trim_db=resolved_current_wing_input_trim_db,
        current_wing_pan_percent=resolved_current_wing_pan_percent,
        current_wing_control_readback_available=resolved_control_readback_available,
        rollback_plan_available=resolved_rollback_plan_available,
        rms_dbfs=evidence.rms_dbfs,
        estimated_lufs=evidence.estimated_lufs,
        lra_proxy_db=evidence.lra_proxy_db,
        true_peak_headroom_db=max(evidence.true_peak_headroom_db, 0.0),
        feedback_risk=evidence.feedback_risk,
        feedback_frequency_hz=evidence.feedback_frequency_hz,
        feedback_frequency_confidence=evidence.feedback_frequency_confidence,
        feedback_frequency_source=evidence.feedback_frequency_source,
        level_plane_error_lu=evidence.level_plane_error_lu,
        summed_bus_evidence_available=evidence.summed_bus_evidence_available,
        summed_bus_role=evidence.summed_bus_role,
        summed_bus_target_id=evidence.summed_bus_target_id,
        summed_bus_true_peak_headroom_db=evidence.summed_bus_true_peak_headroom_db,
        summed_bus_estimated_lufs=evidence.summed_bus_estimated_lufs,
        summed_bus_target_plane_lufs=evidence.summed_bus_target_plane_lufs,
        summed_bus_level_plane_error_lu=evidence.summed_bus_level_plane_error_lu,
        masking_conflict_role=evidence.masking_conflict_role,
        masking_conflict_target_id=evidence.masking_conflict_target_id,
        masking_frequency_hz=evidence.masking_frequency_hz,
        masking_score=evidence.masking_score,
        masking_bandwidth_hz=evidence.masking_bandwidth_hz,
        masking_overlap_peak_share=evidence.masking_overlap_peak_share,
        arrangement_overlap_score=evidence.arrangement_overlap_score,
        muddiness_score=evidence.muddiness_score,
        mud_frequency_hz=_safe_float(evidence.mud_frequency_hz),
        low_mid_bloom_frequency_hz=evidence.low_mid_bloom_frequency_hz,
        low_mid_bloom_db=evidence.low_mid_bloom_db,
        low_mid_bloom_peak_dbfs=evidence.low_mid_bloom_peak_dbfs,
        low_mid_bloom_rise_time_ms=evidence.low_mid_bloom_rise_time_ms,
        low_mid_bloom_recovery_ms=evidence.low_mid_bloom_recovery_ms,
        plosive_risk_score=evidence.plosive_risk_score,
        plosive_frequency_hz=evidence.plosive_frequency_hz,
        plosive_event_peak_dbfs=evidence.plosive_event_peak_dbfs,
        plosive_event_contrast_db=evidence.plosive_event_contrast_db,
        plosive_event_rise_time_ms=evidence.plosive_event_rise_time_ms,
        plosive_event_recovery_ms=evidence.plosive_event_recovery_ms,
        harshness_score=evidence.harshness_score,
        presence_harshness_score=evidence.presence_harshness_score,
        instrument_harsh_frequency_hz=evidence.instrument_harsh_frequency_hz,
        instrument_harsh_bandwidth_hz=evidence.instrument_harsh_bandwidth_hz,
        instrument_harsh_event_peak_dbfs=evidence.instrument_harsh_event_peak_dbfs,
        instrument_harsh_event_contrast_db=evidence.instrument_harsh_event_contrast_db,
        instrument_harsh_event_recovery_ms=evidence.instrument_harsh_event_recovery_ms,
        drum_harsh_frequency_hz=evidence.drum_harsh_frequency_hz,
        drum_harsh_bandwidth_hz=evidence.drum_harsh_bandwidth_hz,
        drum_harsh_event_peak_dbfs=evidence.drum_harsh_event_peak_dbfs,
        drum_harsh_event_contrast_db=evidence.drum_harsh_event_contrast_db,
        drum_harsh_event_recovery_ms=evidence.drum_harsh_event_recovery_ms,
        sibilance_score=evidence.sibilance_score,
        sibilance_frequency_hz=evidence.sibilance_frequency_hz,
        sibilance_bandwidth_hz=_safe_float(evidence.sibilance_bandwidth_hz),
        sibilance_event_peak_dbfs=evidence.sibilance_event_peak_dbfs,
        sibilance_event_contrast_db=evidence.sibilance_event_contrast_db,
        sibilance_event_rise_time_ms=evidence.sibilance_event_rise_time_ms,
        sibilance_event_recovery_ms=evidence.sibilance_event_recovery_ms,
        presence_deficit_score=evidence.presence_deficit_score,
        presence_lift_frequency_hz=evidence.presence_lift_frequency_hz,
        presence_lift_bandwidth_hz=evidence.presence_lift_bandwidth_hz,
        presence_lift_parameter_source=evidence.presence_lift_parameter_source,
        dynamics_variance_score=evidence.dynamics_variance_score,
        dynamics_peak_to_median_db=evidence.dynamics_peak_to_median_db,
        dynamics_rise_time_ms=evidence.dynamics_rise_time_ms,
        dynamics_recovery_ms=evidence.dynamics_recovery_ms,
        vocal_noise_floor_score=evidence.vocal_noise_floor_score,
        phrase_gap_noise_floor_dbfs=evidence.phrase_gap_noise_floor_dbfs,
        phrase_gap_noise_flatness=evidence.phrase_gap_noise_flatness,
        phrase_gap_contrast_db=evidence.phrase_gap_contrast_db,
        phrase_gap_median_ms=evidence.phrase_gap_median_ms,
        phrase_gap_coverage_ratio=evidence.phrase_gap_coverage_ratio,
        line_noise_score=evidence.line_noise_score,
        line_noise_frequency_hz=evidence.line_noise_frequency_hz,
        line_noise_prominence_db=evidence.line_noise_prominence_db,
        line_noise_harmonic_count=evidence.line_noise_harmonic_count,
        line_noise_harmonics=evidence.line_noise_harmonics,
        drum_ring_resonance_score=evidence.drum_ring_resonance_score,
        drum_ring_frequency_hz=evidence.drum_ring_frequency_hz,
        drum_ring_decay_score=evidence.drum_ring_decay_score,
        drum_ring_bandwidth_hz=evidence.drum_ring_bandwidth_hz,
        drum_ring_peak_dbfs=evidence.drum_ring_peak_dbfs,
        drum_ring_decay_ms=evidence.drum_ring_decay_ms,
        crest_factor_db=evidence.crest_factor_db,
        noise_floor_dbfs=evidence.noise_floor_dbfs,
        bleed_risk=evidence.bleed_risk,
        tail_risk=evidence.tail_risk,
        chatter_risk=evidence.chatter_risk,
        gate_hit_peak_dbfs=evidence.gate_hit_peak_dbfs,
        gate_hit_to_bleed_margin_db=evidence.gate_hit_to_bleed_margin_db,
        gate_tail_decay_ms=evidence.gate_tail_decay_ms,
        transient_density=evidence.transient_density,
        transient_strength=evidence.transient_strength,
        transient_peak_dbfs=evidence.transient_peak_dbfs,
        transient_rise_time_ms=evidence.transient_rise_time_ms,
        transient_decay_ms=evidence.transient_decay_ms,
        fx_wetness_score=evidence.fx_wetness_score,
        fx_decay_risk_score=evidence.fx_decay_risk_score,
        fx_delay_smear_score=evidence.fx_delay_smear_score,
        fx_tail_decay_seconds=evidence.fx_tail_decay_seconds,
        fx_delay_repeat_ms=evidence.fx_delay_repeat_ms,
        fx_return_headroom_db=evidence.fx_return_headroom_db,
        fx_return_target_id=evidence.fx_return_target_id,
        monitor_feedback_risk=evidence.monitor_feedback_risk,
        phase_correlation=evidence.phase_correlation,
        mono_sum_delta_db=evidence.mono_sum_delta_db,
        stereo_balance_db=evidence.stereo_balance_db,
        stereo_balance_confidence=evidence.stereo_balance_confidence,
        measured_pan_target_percent=evidence.measured_pan_target_percent,
        measured_pan_delta_percent=evidence.measured_pan_delta_percent,
        stereo_width_request=_safe_float(desired_stereo_width_request),
        current_pan_percent=_safe_float(resolved_current_pan_percent),
        master_true_peak_db=evidence.master_true_peak_db,
        master_integrated_lufs=evidence.master_integrated_lufs,
        master_presence_frequency_hz=evidence.master_presence_frequency_hz,
        master_presence_bandwidth_hz=evidence.master_presence_bandwidth_hz,
        master_low_end_frequency_hz=evidence.master_low_end_frequency_hz,
        master_low_end_bandwidth_hz=evidence.master_low_end_bandwidth_hz,
        spectral_tilt_score=evidence.spectral_tilt_score,
        low_end_excess_score=evidence.low_end_excess_score,
        low_end_resonance_frequency_hz=evidence.low_end_resonance_frequency_hz,
        low_end_resonance_bandwidth_hz=evidence.low_end_resonance_bandwidth_hz,
        low_end_resonance_peak_dbfs=evidence.low_end_resonance_peak_dbfs,
        low_end_resonance_sustain_ms=evidence.low_end_resonance_sustain_ms,
        active_performance=bool(active_performance),
    )


def analyze_smart_dsp_audio(
    samples: Any,
    **kwargs: Any,
) -> SmartDSPRecommendation:
    """Extract offline audio evidence and run the smart DSP strategy selector."""

    analysis = build_smart_dsp_analysis_input_from_audio(samples, **kwargs)
    recommendation = analyze_smart_dsp_strategy(analysis)
    recommendation.evidence["audio_evidence"] = extract_smart_dsp_audio_evidence(
        samples,
        target_id=kwargs["target_id"],
        source_role=kwargs["source_role"],
        source_label=kwargs.get("source_label", ""),
        mode=kwargs.get("mode", "soundcheck"),
        sample_rate=kwargs.get("sample_rate", DEFAULT_SAMPLE_RATE),
        reference_samples=kwargs.get("reference_samples"),
        reference_role=kwargs.get("reference_role", ""),
        reference_target_id=kwargs.get("reference_target_id", ""),
        summed_bus_samples=kwargs.get("summed_bus_samples"),
        summed_bus_role=kwargs.get("summed_bus_role", ""),
        summed_bus_target_id=kwargs.get("summed_bus_target_id", ""),
        summed_bus_target_plane_lufs=kwargs.get("summed_bus_target_plane_lufs"),
        desired_stereo_width_request=kwargs.get("desired_stereo_width_request", 0.0),
        fx_return_headroom_db=kwargs.get("fx_return_headroom_db"),
        fx_return_target_id=kwargs.get("fx_return_target_id", ""),
        monitor_feedback_risk=kwargs.get("monitor_feedback_risk", 0.0),
        target_plane_lufs=kwargs.get("target_plane_lufs"),
        extra_problem_hints=kwargs.get("extra_problem_hints", ()),
    ).to_dict()
    recommendation.evidence["audio_evidence_source"] = SMART_DSP_EVIDENCE_EXTRACTOR_VERSION
    recommendation.evidence["true_peak_headroom_minimum_db"] = MIN_TRUE_PEAK_HEADROOM_DB
    return recommendation
