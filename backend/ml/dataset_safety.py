"""
Safety filters for training corpora (live sound).

Reject or flag examples that encourage unsafe levels before ML training.
See docs/AGENT_TRAINING_DATA.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

TRAINING_EVENT_SCHEMA_VERSION = "1.0"

@dataclass(frozen=True)
class SafetyLimits:
    """Default ceilings aligned with project safety rules (.cursorrules)."""

    max_fader_dbfs: float = 0.0
    true_peak_limit_dbtp: float = -1.0
    max_gain_trim_db: float = 12.0
    min_gain_trim_db: float = -24.0


def _get_nested(
    d: Mapping[str, Any],
    path: Tuple[str, ...],
    default: Any = None,
) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def validate_training_event_v1(
    event: Mapping[str, Any],
    limits: Optional[SafetyLimits] = None,
) -> Tuple[bool, List[str]]:
    """
    Validate a single mix training event dict (v1).

    Returns (ok, list_of_reason_strings). Empty list means ok.
    """
    limits = limits or SafetyLimits()
    errors: List[str] = []

    if event.get("schema_version") != TRAINING_EVENT_SCHEMA_VERSION:
        errors.append(
            f"schema_version must be '{TRAINING_EVENT_SCHEMA_VERSION}'"
        )

    for key in ("event_id", "recorded_at", "source"):
        if key not in event or not event[key]:
            errors.append(f"missing required field: {key}")

    override = bool(
        _get_nested(event, ("safety", "explicit_override_positive_fader"), False)
    )

    # Target fader in operator.parameters_final or automation.parameters
    for path in (
        ("operator", "parameters_final", "fader_dbfs"),
        ("operator", "parameters_final", "fader_db"),
        ("automation", "parameters", "target_fader_dbfs"),
    ):
        val = _get_nested(event, path)
        if val is not None and isinstance(val, (int, float)):
            if val > limits.max_fader_dbfs and not override:
                errors.append(
                    f"fader target {val} dBFS exceeds ceiling {limits.max_fader_dbfs}"
                )

    params = _get_nested(event, ("automation", "parameters"))
    if isinstance(params, Mapping):
        delta_db = params.get("delta_db")
        if isinstance(delta_db, (int, float)) and delta_db > 0:
            tp = _get_nested(event, ("observation", "true_peak_dbtp"))
            if isinstance(tp, (int, float)) and tp > limits.true_peak_limit_dbtp:
                errors.append(
                    "positive delta_db requires true_peak_dbtp within limit: "
                    f"{tp} > {limits.true_peak_limit_dbtp}"
                )
        trim_max = limits.max_gain_trim_db
        trim_min = limits.min_gain_trim_db
        if "trim_db" in params:
            t = params["trim_db"]
            if isinstance(t, (int, float)) and (t > trim_max or t < trim_min):
                errors.append(
                    f"trim_db {t} outside [{trim_min}, {trim_max}]"
                )
        if "gain_db" in params:
            g = params["gain_db"]
            if isinstance(g, (int, float)) and (g > trim_max or g < trim_min):
                errors.append(
                    f"gain_db {g} outside [{trim_min}, {trim_max}]"
                )

    return (len(errors) == 0, errors)


def redact_training_event(
    event: MutableMapping[str, Any],
    strip_keys: Optional[Iterable[str]] = None,
) -> MutableMapping[str, Any]:
    """
    Remove known sensitive keys in-place (copy first if you need immutability).

    Extend strip_keys for site-specific PII fields.
    """
    default_strip = {
        "artist_name",
        "venue_name",
        "venue_address",
        "operator_name",
        "client_email",
        "ip_address",
    }
    keys = set(strip_keys) if strip_keys is not None else default_strip
    for k in list(event.keys()):
        if k in keys:
            del event[k]
    return event


def filter_events_for_training(
    events: Iterable[Mapping[str, Any]],
    limits: Optional[SafetyLimits] = None,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, List[str]]]]:
    """
    Keep only events that pass validate_training_event_v1.

    Returns (accepted_as_dicts, rejected_list of (event_id or 'unknown', errors)).
    """
    accepted: List[Dict[str, Any]] = []
    rejected: List[Tuple[str, List[str]]] = []
    limits = limits or SafetyLimits()
    for raw in events:
        ev = dict(raw)
        ok, errs = validate_training_event_v1(ev, limits=limits)
        eid = ev.get("event_id", "unknown")
        if ok:
            accepted.append(ev)
        else:
            rejected.append((str(eid), errs))
    return accepted, rejected
