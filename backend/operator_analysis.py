"""Read-only operator analysis that can feed the proposal queue.

This is the first product-facing producer for the mode workflow. It only uses
runtime snapshots and cached mixer state; it does not query or write the mixer.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from operator_product_state import build_channel_inventory


PEAK_RISK_THRESHOLD_DB = -3.0
PEAK_TARGET_DB = -6.0
GAIN_TRIM_PEAK_THRESHOLD_DB = -1.0
GAIN_TRIM_PEAK_TARGET_DB = -3.0
MAX_FADER_REDUCTION_DB = 6.0
MAX_GAIN_REDUCTION_DB = 3.0
VOCAL_LUFS_TARGET_DB = -22.0
VOCAL_LUFS_MARGIN_DB = 2.0
STRONG_SOURCE_LUFS_TARGET_DB = -25.0
STRONG_SOURCE_LUFS_MARGIN_DB = 2.0
ROLE_LEVEL_RULES = {
    "Vocal": {
        "rule": "vocal_level_high",
        "label": "vocal",
        "target_lufs": VOCAL_LUFS_TARGET_DB,
        "margin_db": VOCAL_LUFS_MARGIN_DB,
    },
    "Kick": {
        "rule": "kick_level_high",
        "label": "kick",
        "target_lufs": STRONG_SOURCE_LUFS_TARGET_DB,
        "margin_db": STRONG_SOURCE_LUFS_MARGIN_DB,
    },
    "Snare": {
        "rule": "snare_level_high",
        "label": "snare",
        "target_lufs": STRONG_SOURCE_LUFS_TARGET_DB,
        "margin_db": STRONG_SOURCE_LUFS_MARGIN_DB,
    },
    "Bass": {
        "rule": "bass_level_high",
        "label": "bass",
        "target_lufs": STRONG_SOURCE_LUFS_TARGET_DB,
        "margin_db": STRONG_SOURCE_LUFS_MARGIN_DB,
    },
    "Guitar": {
        "rule": "guitar_level_high",
        "label": "guitar",
        "target_lufs": STRONG_SOURCE_LUFS_TARGET_DB,
        "margin_db": STRONG_SOURCE_LUFS_MARGIN_DB,
    },
}
STEREO_SIDE_PATTERN = re.compile(r"^(?P<base>.+?)[\s_\-/]+(?P<side>L|R|LEFT|RIGHT)$", re.IGNORECASE)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _round(value: Any, digits: int = 1) -> Optional[float]:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _fader_reduction_change(channel: Dict[str, Any], reduction_db: float) -> Optional[Dict[str, Any]]:
    fader_db = channel.get("control_state", {}).get("fader_db")
    if fader_db is None:
        return None
    return {
        "value_type": "fader",
        "channel": int(channel["id"]),
        "value": round(float(fader_db) - reduction_db, 1),
        "current_value": _round(fader_db),
        "delta_db": round(-reduction_db, 1),
    }


def _gain_reduction_change(channel: Dict[str, Any], reduction_db: float) -> Optional[Dict[str, Any]]:
    gain_db = channel.get("control_state", {}).get("gain_db")
    if gain_db is None:
        return None
    current_value = _round(gain_db)
    proposed_value = round(float(gain_db) - reduction_db, 1)
    return {
        "value_type": "gain",
        "channel": int(channel["id"]),
        "value": proposed_value,
        "current_value": current_value,
        "delta_db": round(-reduction_db, 1),
        "old_value": current_value,
        "proposed_value": proposed_value,
        "rollback_value": current_value,
        "readback_required": True,
    }


def _add_change_safety_evidence(
    change: Optional[Dict[str, Any]],
    evidence: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not change:
        return change
    current_value = change.get("current_value")
    change.setdefault("old_value", current_value)
    change.setdefault("proposed_value", change.get("value"))
    change.setdefault("rollback_value", current_value)
    change["safety_evidence"] = evidence
    return change


def _build_gain_trim_proposal(channel: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    channel_id = int(channel["id"])
    peak_db = float(channel["peak_db"])
    reduction_db = min(MAX_GAIN_REDUCTION_DB, max(0.5, peak_db - GAIN_TRIM_PEAK_TARGET_DB))
    requested_change = _gain_reduction_change(channel, reduction_db)
    if not requested_change:
        return None
    target_gain = requested_change["value"]
    _add_change_safety_evidence(
        requested_change,
        [
            {"label": "Peak", "value": _round(peak_db), "limit": GAIN_TRIM_PEAK_THRESHOLD_DB},
            {"label": "Direction", "value": "gain_down_only"},
            {"label": "Rollback", "value": requested_change["rollback_value"]},
        ],
    )

    return {
        "id": f"analysis:gain_trim_down:{channel_id}",
        "title": f"{channel['name']}: peak {peak_db:.1f} dBFS, propose input trim {target_gain:.1f} dB",
        "target": channel_id,
        "channel": channel_id,
        "kind": "gain",
        "severity": "critical",
        "confidence": 0.88,
        "source": "operator_analysis:gain_trim_down",
        "reason": (
            "Input peak is near clipping; the safe proposal only reduces input trim "
            "and requires supervised approval plus readback."
        ),
        "requested_change": requested_change,
        "raw": {
            "rule": "gain_trim_down",
            "peak_db": _round(peak_db),
            "threshold_db": GAIN_TRIM_PEAK_THRESHOLD_DB,
            "target_peak_db": GAIN_TRIM_PEAK_TARGET_DB,
            "current_gain_db": requested_change.get("current_value"),
            "reduction_db": round(reduction_db, 1),
            "rollback_value": requested_change.get("rollback_value"),
            "safety_evidence": requested_change.get("safety_evidence", []),
        },
    }


def _build_peak_proposal(channel: Dict[str, Any]) -> Dict[str, Any]:
    channel_id = int(channel["id"])
    peak_db = float(channel["peak_db"])
    reduction_db = min(MAX_FADER_REDUCTION_DB, max(0.5, peak_db - PEAK_TARGET_DB))
    requested_change = _fader_reduction_change(channel, reduction_db)
    target_fader = requested_change["value"] if requested_change else None
    _add_change_safety_evidence(
        requested_change,
        [
            {"label": "Peak", "value": _round(peak_db), "limit": PEAK_RISK_THRESHOLD_DB},
            {"label": "Direction", "value": "fader_down_only"},
            {"label": "Rollback", "value": requested_change.get("rollback_value") if requested_change else None},
        ],
    )

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
            "current_fader_db": requested_change.get("current_value") if requested_change else None,
            "reduction_db": round(reduction_db, 1),
        },
    }


def _is_vocal_channel(channel: Dict[str, Any]) -> bool:
    return channel.get("role") == "Vocal" or channel.get("group") == "VOCALS"


def _level_rule_for_channel(channel: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    role = str(channel.get("role") or "")
    if role in ROLE_LEVEL_RULES:
        return ROLE_LEVEL_RULES[role]
    if _is_vocal_channel(channel):
        return ROLE_LEVEL_RULES["Vocal"]
    return None


def _level_reduction_db(channel: Dict[str, Any], rule: Dict[str, Any]) -> float:
    lufs = float(channel["lufs"])
    target_lufs = float(rule["target_lufs"])
    over_target_db = max(0.0, lufs - target_lufs)
    return min(MAX_FADER_REDUCTION_DB, max(0.5, over_target_db))


def _build_level_proposal(channel: Dict[str, Any], rule: Dict[str, Any]) -> Dict[str, Any]:
    channel_id = int(channel["id"])
    lufs = float(channel["lufs"])
    target_lufs = float(rule["target_lufs"])
    margin_db = float(rule["margin_db"])
    label = str(rule["label"])
    rule_name = str(rule["rule"])
    over_target_db = max(0.0, lufs - target_lufs)
    reduction_db = _level_reduction_db(channel, rule)
    requested_change = _fader_reduction_change(channel, reduction_db)
    target_fader = requested_change["value"] if requested_change else None
    _add_change_safety_evidence(
        requested_change,
        [
            {"label": "LUFS", "value": _round(lufs), "target": target_lufs},
            {"label": "Direction", "value": "fader_down_only"},
            {"label": "Rollback", "value": requested_change.get("rollback_value") if requested_change else None},
        ],
    )

    return {
        "id": f"analysis:{rule_name}:{channel_id}",
        "title": (
            f"{channel['name']}: {label} level {lufs:.1f} LUFS"
            + (f", propose fader {target_fader:.1f} dB" if target_fader is not None else "")
        ),
        "target": channel_id,
        "channel": channel_id,
        "kind": "fader" if requested_change else "analysis",
        "severity": "high" if over_target_db >= 6.0 else "medium",
        "confidence": 0.82,
        "source": f"operator_analysis:{rule_name}",
        "reason": (
            f"{label.title()} loudness is above the conservative live target; "
            "the safe proposal only reduces fader level."
        ),
        "requested_change": requested_change,
        "raw": {
            "rule": rule_name,
            "lufs": _round(lufs),
            "target_lufs": target_lufs,
            "margin_db": margin_db,
            "current_fader_db": requested_change.get("current_value") if requested_change else None,
            "reduction_db": round(reduction_db, 1),
        },
    }


def _parse_stereo_side(name: str) -> Optional[Tuple[str, str]]:
    match = STEREO_SIDE_PATTERN.match(str(name or "").strip())
    if not match:
        return None
    base = re.sub(r"\s+", " ", match.group("base")).strip().lower()
    side = match.group("side").upper()
    return base, "L" if side.startswith("L") else "R"


def _stereo_pair_key(channel: Dict[str, Any], rule: Dict[str, Any]) -> Optional[Tuple[str, str, str, str]]:
    parsed = _parse_stereo_side(str(channel.get("name") or ""))
    if not parsed:
        return None
    base, side = parsed
    return (str(channel.get("group") or ""), str(channel.get("role") or ""), str(rule["rule"]), base), side


def _build_stereo_level_observation(left: Dict[str, Any], right: Dict[str, Any], rule: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    threshold = float(rule["target_lufs"]) + float(rule["margin_db"])
    if max(float(left["lufs"]), float(right["lufs"])) < threshold:
        return None

    rule_name = str(rule["rule"])
    stereo_rule_name = f"stereo_{rule_name}"
    label = str(rule["label"])
    reduction_db = max(_level_reduction_db(left, rule), _level_reduction_db(right, rule))
    left_change = _fader_reduction_change(left, reduction_db)
    right_change = _fader_reduction_change(right, reduction_db)
    _add_change_safety_evidence(
        left_change,
        [
            {"label": "LUFS L", "value": _round(left["lufs"]), "target": float(rule["target_lufs"])},
            {"label": "Direction", "value": "linked_fader_down_only"},
        ],
    )
    _add_change_safety_evidence(
        right_change,
        [
            {"label": "LUFS R", "value": _round(right["lufs"]), "target": float(rule["target_lufs"])},
            {"label": "Direction", "value": "linked_fader_down_only"},
        ],
    )
    requested_changes = [left_change, right_change] if left_change and right_change else []
    left_id = int(left["id"])
    right_id = int(right["id"])
    title = (
        f"{left['name']} / {right['name']}: stereo {label} level "
        f"L {_round(left['lufs'])} / R {_round(right['lufs'])} LUFS"
        + (f", propose linked faders {requested_changes[0]['value']:.1f}/{requested_changes[1]['value']:.1f} dB" if requested_changes else "")
    )
    proposal = {
        "id": f"analysis:{stereo_rule_name}:{left_id}_{right_id}",
        "title": title,
        "target": [left_id, right_id],
        "channels": [left_id, right_id],
        "kind": "linked_fader",
        "severity": "high" if reduction_db >= 6.0 else "medium",
        "confidence": 0.84,
        "source": f"operator_analysis:{stereo_rule_name}",
        "reason": (
            f"Stereo {label} pair is above the conservative live target. "
            "This linked proposal preserves the current L/R fader relationship and requires supervised approval."
        ),
        "requested_change": None,
        "requested_changes": requested_changes,
        "raw": {
            "rule": stereo_rule_name,
            "base_rule": rule_name,
            "pair": "stereo_lr",
            "channels": [left_id, right_id],
            "channel_names": [left["name"], right["name"]],
            "lufs": {"left": _round(left["lufs"]), "right": _round(right["lufs"])},
            "target_lufs": float(rule["target_lufs"]),
            "margin_db": float(rule["margin_db"]),
            "reduction_db": round(reduction_db, 1),
            "apply_status": "linked_supervised_apply_supported",
            "apply_requires": "supervised_operator_approval",
        },
    }
    return {
        "rule": stereo_rule_name,
        "severity": proposal["severity"],
        "channels": [left_id, right_id],
        "channel_names": [left["name"], right["name"]],
        "lufs": proposal["raw"]["lufs"],
        "message": proposal["title"],
        "proposal": proposal,
    }


def _build_stereo_level_observations(channels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pairs: Dict[Tuple[str, str, str, str], Dict[str, Dict[str, Any]]] = {}
    rules: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for channel in channels:
        if channel.get("status") != "active" or channel.get("lufs") is None:
            continue
        rule = _level_rule_for_channel(channel)
        if not rule:
            continue
        keyed = _stereo_pair_key(channel, rule)
        if not keyed:
            continue
        key, side = keyed
        pairs.setdefault(key, {})[side] = channel
        rules[key] = rule

    observations = []
    for key, sides in pairs.items():
        if "L" not in sides or "R" not in sides:
            continue
        observation = _build_stereo_level_observation(sides["L"], sides["R"], rules[key])
        if observation:
            observations.append(observation)
    return observations


def _observe_channel(channel: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if channel.get("status") != "active":
        return None
    peak_db = channel.get("peak_db")
    if peak_db is not None and float(peak_db) >= PEAK_RISK_THRESHOLD_DB:
        proposal = None
        if float(peak_db) >= GAIN_TRIM_PEAK_THRESHOLD_DB:
            proposal = _build_gain_trim_proposal(channel)
        if proposal is None:
            proposal = _build_peak_proposal(channel)
        return {
            "rule": proposal["raw"]["rule"],
            "severity": proposal["severity"],
            "channel": channel["id"],
            "channel_name": channel["name"],
            "peak_db": _round(peak_db),
            "message": proposal["title"],
            "proposal": proposal,
        }
    lufs = channel.get("lufs")
    rule = _level_rule_for_channel(channel)
    if lufs is not None and rule:
        threshold = float(rule["target_lufs"]) + float(rule["margin_db"])
        if float(lufs) >= threshold:
            proposal = _build_level_proposal(channel, rule)
            return {
                "rule": proposal["raw"]["rule"],
                "severity": proposal["severity"],
                "channel": channel["id"],
                "channel_name": channel["name"],
                "lufs": _round(lufs),
                "message": proposal["title"],
                "proposal": proposal,
            }
    return None


def build_operator_analysis_report(
    server: Any,
    *,
    total_channels: int = 24,
    create_proposals: Optional[bool] = None,
    include_proposal_previews: bool = False,
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
    previewed: List[Dict[str, Any]] = []
    created: List[Dict[str, Any]] = []
    blocked: List[Dict[str, Any]] = []

    if audio_running:
        paired_channels = set()
        for observation in _build_stereo_level_observations(inventory["channels"]):
            paired_channels.update(int(channel) for channel in observation.get("channels", []))
            observations.append({key: value for key, value in observation.items() if key != "proposal"})
            if include_proposal_previews:
                previewed.append(observation["proposal"])
            if should_create:
                result = server.operator_proposal_queue.create(observation["proposal"], operator_mode)
                if result.get("success"):
                    created.append(result["proposal"])
                else:
                    blocked.append(result)

        for channel in inventory["channels"]:
            if int(channel["id"]) in paired_channels:
                continue
            observation = _observe_channel(channel)
            if not observation:
                continue
            observations.append({key: value for key, value in observation.items() if key != "proposal"})
            if include_proposal_previews:
                previewed.append(observation["proposal"])
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
        "proposals_preview": previewed,
        "proposals_created": created,
        "proposals_blocked": blocked,
        "channel_summary": inventory.get("summary", {}),
    }
