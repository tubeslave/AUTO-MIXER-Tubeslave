"""Read-only operator analysis that can feed the proposal queue.

This is the first product-facing producer for the mode workflow. It only uses
runtime snapshots and cached mixer state; it does not query or write the mixer.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from operator_product_state import build_channel_inventory
from shadow_gain_rides import build_shadow_gain_ride_candidates, shadow_gain_ride_trace


PEAK_RISK_THRESHOLD_DB = -3.0
PEAK_TARGET_DB = -6.0
GAIN_TRIM_PEAK_THRESHOLD_DB = -1.0
GAIN_TRIM_PEAK_TARGET_DB = -3.0
MAX_FADER_REDUCTION_DB = 6.0
MAX_GAIN_REDUCTION_DB = 3.0
CACHE_FRESHNESS_STALE_AFTER_SEC = 30.0
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


def _coerce_epoch_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, datetime):
        timestamp = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return timestamp.timestamp()
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric / 1000.0 if numeric > 1_000_000_000_000 else numeric
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return _coerce_epoch_seconds(float(text))
        except ValueError:
            pass
        try:
            return _coerce_epoch_seconds(datetime.fromisoformat(text.replace("Z", "+00:00")))
        except ValueError:
            return None
    return None


def _iso_from_epoch(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _cached_timestamp_epoch(client: Any, address: str) -> Optional[float]:
    for attr in ("state_timestamps", "_state_timestamps", "state_updated_at", "_state_updated_at"):
        timestamps = getattr(client, attr, None)
        if not isinstance(timestamps, dict):
            continue
        raw = timestamps.get(address)
        if isinstance(raw, dict):
            raw = raw.get("updated_at") or raw.get("timestamp") or raw.get("ts")
        epoch = _coerce_epoch_seconds(raw)
        if epoch is not None:
            return epoch
    return None


def _change_cache_address(change: Dict[str, Any]) -> Optional[str]:
    channel = change.get("channel")
    value_type = str(change.get("value_type") or "").strip().lower()
    if channel is None:
        return None
    try:
        channel_id = int(channel)
    except (TypeError, ValueError):
        return None
    if value_type == "fader":
        return f"/ch/{channel_id}/fdr"
    if value_type == "gain":
        return f"/ch/{channel_id}/in/set/trim"
    return None


def _change_cache_freshness(client: Any, change: Dict[str, Any]) -> Dict[str, Any]:
    address = _change_cache_address(change)
    state = getattr(client, "state", None)
    present = bool(address and isinstance(state, dict) and address in state and state.get(address) is not None)
    updated_epoch = _cached_timestamp_epoch(client, address) if address else None
    now = datetime.now(timezone.utc).timestamp()
    age_sec = round(max(0.0, now - updated_epoch), 3) if updated_epoch is not None else None
    if not address:
        status = "unsupported"
    elif not present:
        status = "missing"
    elif updated_epoch is None:
        status = "unknown"
    elif age_sec is not None and age_sec > CACHE_FRESHNESS_STALE_AFTER_SEC:
        status = "stale"
    else:
        status = "fresh"
    return {
        "status": status,
        "address": address,
        "value_type": str(change.get("value_type") or "").strip().lower() or None,
        "channel": change.get("channel"),
        "source": "state_timestamps" if updated_epoch is not None else "unavailable",
        "updated_at": _iso_from_epoch(updated_epoch),
        "age_sec": age_sec,
        "stale_after_sec": CACHE_FRESHNESS_STALE_AFTER_SEC,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }


def _append_cache_freshness_evidence(change: Dict[str, Any], evidence: Dict[str, Any]) -> None:
    change["cache_freshness"] = evidence
    change.setdefault("safety_evidence", []).append({
        "label": "Cache freshness",
        "value": evidence["status"],
        "address": evidence.get("address"),
        "age_sec": evidence.get("age_sec"),
        "stale_after_sec": evidence.get("stale_after_sec"),
    })


def _apply_cache_freshness_gate(proposal: Dict[str, Any], client: Any) -> Dict[str, Any]:
    raw = proposal.setdefault("raw", {})
    single_change = proposal.get("requested_change")
    linked_changes = proposal.get("requested_changes") or []
    changes = linked_changes if linked_changes else ([single_change] if isinstance(single_change, dict) else [])
    if not changes:
        return proposal

    freshness_rows = [_change_cache_freshness(client, change) for change in changes]
    blockers = [
        f"cache_freshness_{row['status']}:{row.get('address') or 'unknown'}"
        for row in freshness_rows
        if row["status"] != "fresh"
    ]
    raw["cache_freshness"] = freshness_rows
    if not blockers:
        for change, freshness in zip(changes, freshness_rows):
            _append_cache_freshness_evidence(change, freshness)
        raw["apply_status"] = raw.get("apply_status", "cache_freshness_verified")
        proposal["cache_freshness"] = freshness_rows
        proposal["apply_status"] = raw["apply_status"]
        return proposal

    raw["apply_status"] = "blocked_cache_freshness_not_fresh"
    raw["apply_blockers"] = sorted(blockers)
    raw["blocked_requested_changes"] = [dict(change) for change in changes]
    proposal["cache_freshness"] = freshness_rows
    proposal["apply_status"] = raw["apply_status"]
    proposal["apply_blockers"] = raw["apply_blockers"]
    proposal["blocked_requested_changes"] = raw["blocked_requested_changes"]
    proposal["kind"] = "analysis"
    proposal["reason"] = (
        f"{proposal.get('reason', '').rstrip()} "
        "Current cached console value is not fresh enough for an applyable proposal."
    ).strip()
    proposal["title"] = f"{proposal.get('title', 'Operator analysis')}: cache freshness not verified"
    if linked_changes:
        proposal["requested_changes"] = []
    else:
        proposal["requested_change"] = None
    return proposal


def _gate_observation_proposal(observation: Dict[str, Any], client: Any) -> Dict[str, Any]:
    proposal = observation.get("proposal")
    if isinstance(proposal, dict):
        _apply_cache_freshness_gate(proposal, client)
        observation["message"] = proposal.get("title", observation.get("message"))
    return observation


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
    cache_freshness_blocked = 0
    shadow_gain_rides: List[Dict[str, Any]] = []
    shadow_traces_recorded = 0
    client = getattr(server, "mixer_client", None)

    if audio_running:
        recent_traces = (
            server.get_automation_decision_traces(200)
            if hasattr(server, "get_automation_decision_traces")
            else []
        )
        shadow_gain_rides = build_shadow_gain_ride_candidates(
            inventory["channels"],
            recent_traces=recent_traces,
        )
        if should_create and shadow_gain_rides and hasattr(server, "record_automation_decision_trace"):
            mode = str(operator_mode.get("mode") or "assist")
            for candidate in shadow_gain_rides:
                server.record_automation_decision_trace(**shadow_gain_ride_trace(candidate, mode=mode))
                shadow_traces_recorded += 1

        paired_channels = set()
        for observation in _build_stereo_level_observations(inventory["channels"]):
            observation = _gate_observation_proposal(observation, client)
            if (observation.get("proposal") or {}).get("raw", {}).get("apply_status") == "blocked_cache_freshness_not_fresh":
                cache_freshness_blocked += 1
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
            observation = _gate_observation_proposal(observation, client)
            if (observation.get("proposal") or {}).get("raw", {}).get("apply_status") == "blocked_cache_freshness_not_fresh":
                cache_freshness_blocked += 1
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
        "cache_freshness_blocked_count": cache_freshness_blocked,
        "shadow_gain_ride_count": len(shadow_gain_rides),
        "shadow_gain_ride_trace_count": shadow_traces_recorded,
        "observations": observations,
        "shadow_gain_rides": shadow_gain_rides,
        "proposals_preview": previewed,
        "proposals_created": created,
        "proposals_blocked": blocked,
        "channel_summary": inventory.get("summary", {}),
    }
