"""Shadow-only gain ride candidates for operator evidence.

This module never sends mixer commands and never creates applyable proposals.
It produces conservative decision-trace payloads that can be reviewed by the
operator cockpit and replay tooling.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

try:
    from operator_mode_policy import WING_DEPLOYMENT_LIVE_WRITE_KINDS
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_mode_policy import WING_DEPLOYMENT_LIVE_WRITE_KINDS


MAX_SHADOW_STEP_DB = 0.5
VOCAL_LOW_LUFS = -26.0
VOCAL_HIGH_LUFS = -19.0
PEAK_HEADROOM_FOR_UPWARD_RIDE_DB = -6.0
PEAK_WATCH_DB = -3.0
OSCILLATION_WINDOW = 4
OSCILLATION_DIRECTION_CHANGES = 2


SHADOW_PROMOTION_LIVE_WRITE_KINDS = tuple(WING_DEPLOYMENT_LIVE_WRITE_KINDS)


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _round(value: Any, digits: int = 1) -> Optional[float]:
    numeric = _coerce_float(value)
    return round(numeric, digits) if numeric is not None else None


def _is_vocal_channel(channel: Dict[str, Any]) -> bool:
    role = str(channel.get("role") or "").lower()
    group = str(channel.get("group") or "").lower()
    name = str(channel.get("name") or "").lower()
    return role == "vocal" or group == "vocals" or "vocal" in name or "vox" in name


def _trace_direction(trace: Dict[str, Any]) -> int:
    delta = _coerce_float(trace.get("delta"))
    if delta is None:
        return 0
    if delta > 0:
        return 1
    if delta < 0:
        return -1
    return 0


def _recent_trace_directions(channel: int, recent_traces: Iterable[Dict[str, Any]]) -> List[int]:
    directions: List[int] = []
    for trace in recent_traces:
        if not isinstance(trace, dict):
            continue
        if str(trace.get("action_type") or "") != "gain_ride_shadow":
            continue
        target = _coerce_int(trace.get("target"))
        if target != channel:
            continue
        direction = _trace_direction(trace)
        if direction:
            directions.append(direction)
    return directions[-OSCILLATION_WINDOW:]


def _oscillation_risk(directions: List[int], next_direction: int) -> str:
    if not next_direction:
        return "low"
    sequence = [*directions, next_direction]
    changes = sum(
        1
        for previous, current in zip(sequence, sequence[1:])
        if previous and current and previous != current
    )
    if changes >= OSCILLATION_DIRECTION_CHANGES:
        return "blocked"
    if changes == OSCILLATION_DIRECTION_CHANGES - 1:
        return "medium"
    return "low"


def _pumping_risk(channel: Dict[str, Any], delta_db: float) -> str:
    dynamic = _coerce_float(channel.get("dynamic_range_db") or channel.get("dynamic"))
    peak = _coerce_float(channel.get("peak_db"))
    if peak is not None and peak >= PEAK_WATCH_DB and delta_db > 0:
        return "high"
    if dynamic is not None and dynamic < 5.0:
        return "medium"
    return "low"


def _confidence(*, lufs: Optional[float], peak: Optional[float], pumping_risk: str, oscillation_risk: str) -> float:
    value = 0.76
    if lufs is not None:
        value += 0.06
    if peak is not None:
        value += 0.04
    if pumping_risk == "medium":
        value -= 0.12
    elif pumping_risk == "high":
        value -= 0.24
    if oscillation_risk == "medium":
        value -= 0.14
    elif oscillation_risk == "blocked":
        value = 0.35
    return round(max(0.1, min(0.95, value)), 2)


def _select_delta(channel: Dict[str, Any]) -> Optional[float]:
    lufs = _coerce_float(channel.get("lufs"))
    peak = _coerce_float(channel.get("peak_db"))
    if peak is not None and peak >= PEAK_WATCH_DB:
        return -MAX_SHADOW_STEP_DB
    if lufs is None:
        return None
    if lufs <= VOCAL_LOW_LUFS and (peak is None or peak <= PEAK_HEADROOM_FOR_UPWARD_RIDE_DB):
        return MAX_SHADOW_STEP_DB
    if lufs >= VOCAL_HIGH_LUFS:
        return -MAX_SHADOW_STEP_DB
    return None


def build_shadow_gain_ride_candidates(
    channels: Iterable[Dict[str, Any]],
    *,
    recent_traces: Optional[Iterable[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    traces = list(recent_traces or [])
    candidates: List[Dict[str, Any]] = []
    for channel in channels:
        if not isinstance(channel, dict):
            continue
        if str(channel.get("status") or "").lower() != "active":
            continue
        if not _is_vocal_channel(channel):
            continue
        channel_id = _coerce_int(channel.get("id") or channel.get("number") or channel.get("channel"))
        if channel_id is None:
            continue
        delta_db = _select_delta(channel)
        if delta_db is None:
            continue

        lufs = _coerce_float(channel.get("lufs"))
        peak = _coerce_float(channel.get("peak_db"))
        directions = _recent_trace_directions(channel_id, traces)
        next_direction = 1 if delta_db > 0 else -1
        oscillation = _oscillation_risk(directions, next_direction)
        pumping = _pumping_risk(channel, delta_db)
        blocked_reason = None
        status = "shadow_proposed"
        if oscillation == "blocked":
            blocked_reason = "shadow_gain_ride_oscillation_risk"
            status = "blocked"
        elif pumping == "high":
            blocked_reason = "shadow_gain_ride_pumping_risk"
            status = "blocked"

        direction_label = "shadow_up" if delta_db > 0 else "shadow_down"
        candidates.append(
            {
                "id": f"shadow_gain_ride:{channel_id}",
                "channel": channel_id,
                "name": str(channel.get("name") or f"Ch {channel_id}"),
                "role": str(channel.get("role") or "Vocal"),
                "group": str(channel.get("group") or "VOCALS"),
                "action_type": "gain_ride_shadow",
                "status": status,
                "delta_db": round(delta_db, 1),
                "direction": direction_label,
                "current_value": _round(channel.get("control_state", {}).get("fader_db")),
                "proposed_value": None,
                "rollback_value": _round(channel.get("control_state", {}).get("fader_db")),
                "live_apply_allowed": False,
                "blocked_reason": blocked_reason,
                "safety_limits": {
                    "max_step_db": MAX_SHADOW_STEP_DB,
                    "live_promotion_allowed": False,
                    "requires_future_hardware_proof": True,
                    "allowed_live_write_kinds": list(SHADOW_PROMOTION_LIVE_WRITE_KINDS),
                },
                "safety_evidence": [
                    {"label": "Mode", "value": "shadow_only"},
                    {"label": "Direction", "value": direction_label},
                    {"label": "LUFS", "value": _round(lufs), "low": VOCAL_LOW_LUFS, "high": VOCAL_HIGH_LUFS},
                    {"label": "Peak", "value": _round(peak), "upward_limit": PEAK_HEADROOM_FOR_UPWARD_RIDE_DB},
                    {"label": "Rollback", "value": _round(channel.get("control_state", {}).get("fader_db"))},
                ],
                "scoring": {
                    "intelligibility_proxy": "improved" if delta_db > 0 and not blocked_reason else "protected" if delta_db < 0 and not blocked_reason else "blocked",
                    "pumping_risk": pumping,
                    "oscillation_risk": oscillation,
                    "confidence": _confidence(lufs=lufs, peak=peak, pumping_risk=pumping, oscillation_risk=oscillation),
                },
            }
        )
    return candidates


def shadow_gain_ride_trace(candidate: Dict[str, Any], *, mode: str = "shadow") -> Dict[str, Any]:
    return {
        "source": "shadow_gain_ride_v0",
        "mode": mode,
        "target": candidate.get("channel"),
        "action_type": "gain_ride_shadow",
        "status": candidate.get("status", "shadow_proposed"),
        "current_value": candidate.get("current_value"),
        "proposed_value": candidate.get("proposed_value"),
        "delta": candidate.get("delta_db"),
        "safety_limits": candidate.get("safety_limits", {}),
        "pre_read": {
            "status": "not_required_shadow_only",
            "source": "channel_inventory_snapshot",
            "value": candidate.get("current_value"),
        },
        "readback": {
            "status": "not_available_shadow_only",
            "source": "not_live_apply",
            "value": None,
        },
        "rollback": {
            "available": True,
            "value": candidate.get("rollback_value"),
            "status": "not_required_shadow_only",
        },
        "proof_status": "shadow_only",
        "blocked_reason": candidate.get("blocked_reason"),
        "metadata": {
            "candidate_id": candidate.get("id"),
            "name": candidate.get("name"),
            "live_apply_allowed": False,
            "safety_evidence": candidate.get("safety_evidence", []),
            "scoring": candidate.get("scoring", {}),
        },
    }
