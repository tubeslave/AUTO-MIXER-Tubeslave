"""Read-only operator analysis that can feed the proposal queue.

This is the first product-facing producer for the mode workflow. It only uses
runtime snapshots and cached mixer state; it does not query or write the mixer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from operator_product_state import build_channel_inventory


PEAK_RISK_THRESHOLD_DB = -3.0
PEAK_TARGET_DB = -6.0
MAX_FADER_REDUCTION_DB = 6.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _round(value: Any, digits: int = 1) -> Optional[float]:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _build_peak_proposal(channel: Dict[str, Any]) -> Dict[str, Any]:
    channel_id = int(channel["id"])
    peak_db = float(channel["peak_db"])
    fader_db = channel.get("control_state", {}).get("fader_db")
    reduction_db = min(MAX_FADER_REDUCTION_DB, max(0.5, peak_db - PEAK_TARGET_DB))
    requested_change = None
    target_fader = None
    if fader_db is not None:
        target_fader = round(float(fader_db) - reduction_db, 1)
        requested_change = {
            "value_type": "fader",
            "channel": channel_id,
            "value": target_fader,
            "current_value": _round(fader_db),
            "delta_db": round(-reduction_db, 1),
        }

    return {
        "id": f"analysis:peak_risk:{channel_id}",
        "title": (
            f"{channel['name']}: peak {peak_db:.1f} dBFS"
            + (f", propose fader {target_fader:.1f} dB" if target_fader is not None else "")
        ),
        "target": channel_id,
        "channel": channel_id,
        "kind": "fader" if requested_change else "analysis",
        "severity": "critical" if peak_db >= -1.0 else "high",
        "confidence": 0.9,
        "source": "operator_analysis:peak_risk",
        "reason": "Peak is above the operator analysis safety threshold.",
        "requested_change": requested_change,
        "raw": {
            "rule": "peak_risk",
            "peak_db": _round(peak_db),
            "threshold_db": PEAK_RISK_THRESHOLD_DB,
            "target_peak_db": PEAK_TARGET_DB,
            "current_fader_db": _round(fader_db),
            "reduction_db": round(reduction_db, 1),
        },
    }


def _observe_channel(channel: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if channel.get("status") != "active":
        return None
    peak_db = channel.get("peak_db")
    if peak_db is None:
        return None
    if float(peak_db) >= PEAK_RISK_THRESHOLD_DB:
        proposal = _build_peak_proposal(channel)
        return {
            "rule": "peak_risk",
            "severity": proposal["severity"],
            "channel": channel["id"],
            "channel_name": channel["name"],
            "peak_db": _round(peak_db),
            "message": proposal["title"],
            "proposal": proposal,
        }
    return None


def build_operator_analysis_report(
    server: Any,
    *,
    total_channels: int = 24,
    create_proposals: Optional[bool] = None,
) -> Dict[str, Any]:
    operator_mode = server.get_operator_mode_status()
    inventory = build_channel_inventory(server, total_channels=total_channels)
    audio_running = bool(inventory.get("audio_capture", {}).get("running"))
    should_create = (
        bool(operator_mode["capabilities"]["can_create_proposals"])
        if create_proposals is None
        else bool(create_proposals)
    )
    should_create = should_create and bool(operator_mode["capabilities"]["can_create_proposals"])

    observations: List[Dict[str, Any]] = []
    created: List[Dict[str, Any]] = []
    blocked: List[Dict[str, Any]] = []

    if audio_running:
        for channel in inventory["channels"]:
            observation = _observe_channel(channel)
            if not observation:
                continue
            observations.append({key: value for key, value in observation.items() if key != "proposal"})
            if should_create:
                result = server.operator_proposal_queue.create(observation["proposal"], operator_mode)
                if result.get("success"):
                    created.append(result["proposal"])
                else:
                    blocked.append(result)

    status = "ok"
    reason = None
    if not audio_running:
        status = "no_audio"
        reason = "audio_capture_not_running"
    elif not observations:
        status = "clear"
        reason = "no_operator_analysis_observations"

    return {
        "type": "operator_analysis_report",
        "generated_at": _utc_now(),
        "status": status,
        "reason": reason,
        "operator_mode": operator_mode,
        "audio_running": audio_running,
        "create_proposals": should_create,
        "observation_count": len(observations),
        "created_count": len(created),
        "blocked_count": len(blocked),
        "observations": observations,
        "proposals_created": created,
        "proposals_blocked": blocked,
        "channel_summary": inventory.get("summary", {}),
    }
