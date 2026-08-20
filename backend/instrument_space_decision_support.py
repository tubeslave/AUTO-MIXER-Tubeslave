"""Deterministic decision evidence derived from read-only Instrument Space.

This module deliberately stops before correction synthesis.  It identifies
coarse spatial/spectral/level competition, recommends which existing analyzer
should run next, and can annotate an already-built operator proposal.  It has
no mixer, queue, transport, or action-executor dependency.
"""

from __future__ import annotations

import copy
import math
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "automixer.instrument_space_decision_support.v1"
MAX_OBJECTS = 64
MAX_EMITTED_RELATIONS = 128
PAN_REVIEW_DISTANCE = 0.22
FREQUENCY_REVIEW_OCTAVES = 0.75
LEVEL_REVIEW_DELTA_DB = 8.0
WATCH_SCORE = 0.45
REVIEW_SCORE = 0.68
DRIFT_THRESHOLDS = {
    "x_pan": 0.12,
    "y_frequency": 0.08,
    "z_level": 0.08,
}
PROPOSAL_EVIDENCE_MAX_AGE_SEC = 30.0

AUTHORITY_DENIAL = {
    "read_only": True,
    "live_write": False,
    "queue_mutation": False,
    "can_import_to_queue": False,
    "queue_import_allowed": False,
    "can_apply_to_console": False,
    "live_apply_allowed": False,
    "live_authority_granted": False,
    "readiness_closure_allowed": False,
    "mutates_mixer": False,
}


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _utc_age_seconds(value: Any) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        return None
    age = (datetime.now(timezone.utc) - timestamp.astimezone(timezone.utc)).total_seconds()
    return age if age >= -2.0 else None


def _contract_is_safe(payload: Mapping[str, Any]) -> bool:
    return all(
        payload.get(field) is expected
        for field, expected in AUTHORITY_DENIAL.items()
    )


def _object_values(obj: Mapping[str, Any]) -> Dict[str, Any]:
    coordinates = _mapping(obj.get("coordinates"))
    evidence = _mapping(obj.get("evidence"))
    pan = _mapping(evidence.get("pan"))
    frequency = _mapping(evidence.get("frequency"))
    level = _mapping(evidence.get("level"))
    recognition = _mapping(evidence.get("recognition"))

    centroid_hz = _finite(frequency.get("value"))
    level_db = _finite(level.get("value"))
    confidence_percent = _finite(recognition.get("confidence_percent"))
    frequency_freshness = str(frequency.get("freshness_status") or "unknown")
    frequency_usable = (
        frequency.get("available") is True
        and centroid_hz is not None
        and centroid_hz > 0.0
        and frequency_freshness in {"fresh", "not_applicable"}
    )
    return {
        "id": str(obj.get("id") or ""),
        "channel_id": int(_finite(obj.get("channel_id")) or 0),
        "label": str(obj.get("label") or obj.get("id") or "Unknown"),
        "role": str(obj.get("role") or "Unknown"),
        "group": str(obj.get("group") or "CHANNELS"),
        "status": str(obj.get("status") or "unknown"),
        "x_pan": _finite(coordinates.get("x_pan")),
        "y_frequency": _finite(coordinates.get("y_frequency")) if frequency_usable else None,
        "z_level": _finite(coordinates.get("z_level")),
        "centroid_hz": centroid_hz if frequency_usable else None,
        "level_db": level_db if level.get("available") is True else None,
        "confidence": (
            _clip01(confidence_percent / 100.0)
            if confidence_percent is not None
            else None
        ),
        "pan_source": str(pan.get("source") or "unavailable"),
        "frequency_source": str(frequency.get("source") or "unavailable"),
        "frequency_age_sec": _finite(frequency.get("age_sec")),
        "level_source": str(level.get("source") or "unavailable"),
        "recognition_source": str(recognition.get("source") or "unavailable"),
        "frequency_freshness": frequency_freshness,
        "read_coverage_status": str(evidence.get("read_coverage_status") or "unknown"),
    }


def _pair_relation(first: Mapping[str, Any], second: Mapping[str, Any]) -> Dict[str, Any]:
    pan_distance = None
    if first["x_pan"] is not None and second["x_pan"] is not None:
        pan_distance = abs(float(first["x_pan"]) - float(second["x_pan"]))

    octave_distance = None
    if first["centroid_hz"] is not None and second["centroid_hz"] is not None:
        octave_distance = abs(
            math.log2(float(first["centroid_hz"]) / float(second["centroid_hz"]))
        )

    level_delta_db = None
    if first["level_db"] is not None and second["level_db"] is not None:
        level_delta_db = abs(float(first["level_db"]) - float(second["level_db"]))

    components: Dict[str, Optional[float]] = {
        "pan_proximity": (
            _clip01(1.0 - pan_distance / PAN_REVIEW_DISTANCE)
            if pan_distance is not None
            else None
        ),
        "spectral_proximity": (
            _clip01(1.0 - octave_distance / FREQUENCY_REVIEW_OCTAVES)
            if octave_distance is not None
            else None
        ),
        "level_competition": (
            _clip01(1.0 - level_delta_db / LEVEL_REVIEW_DELTA_DB)
            if level_delta_db is not None
            else None
        ),
        "identity_confidence": (
            min(float(first["confidence"]), float(second["confidence"]))
            if first["confidence"] is not None and second["confidence"] is not None
            else None
        ),
    }
    weights = {
        "pan_proximity": 0.30,
        "spectral_proximity": 0.40,
        "level_competition": 0.20,
        "identity_confidence": 0.10,
    }
    present = [(name, value) for name, value in components.items() if value is not None]
    present_weight = sum(weights[name] for name, _ in present)
    score = (
        sum(float(value) * weights[name] for name, value in present) / present_weight
        if present_weight
        else 0.0
    )

    blockers: List[str] = []
    if pan_distance is None:
        blockers.append("pan_evidence_missing")
    if octave_distance is None:
        blockers.append("fresh_spectral_evidence_missing")
    if level_delta_db is None:
        blockers.append("level_evidence_missing")
    if first["confidence"] is None or second["confidence"] is None:
        blockers.append("source_identity_confidence_missing")

    evidence_axis_count = sum(
        value is not None for value in (pan_distance, octave_distance, level_delta_db)
    )
    if evidence_axis_count < 2:
        status = "insufficient_evidence"
    elif score >= REVIEW_SCORE:
        status = "review_required"
    elif score >= WATCH_SCORE:
        status = "watch"
    else:
        status = "clear"

    analyzers: List[str] = []
    if components["level_competition"] is not None and components["level_competition"] >= 0.55:
        analyzers.append("bus_group_balance_proposal")
    if components["pan_proximity"] is not None and components["pan_proximity"] >= 0.55:
        analyzers.append("pan_spatial_proposal")
    if components["spectral_proximity"] is not None and components["spectral_proximity"] >= 0.55:
        analyzers.extend(("frequency_separation_proposal", "cross_adaptive_eq"))

    channel_ids = sorted((int(first["channel_id"]), int(second["channel_id"])))
    return {
        "id": f"instrument-space:pair:{channel_ids[0]}:{channel_ids[1]}",
        "type": "instrument_space_pair_relation",
        "status": status,
        "channel_ids": channel_ids,
        "object_ids": [str(first["id"]), str(second["id"])],
        "labels": [str(first["label"]), str(second["label"])],
        "groups": [str(first["group"]), str(second["group"])],
        "score": round(score, 4),
        "measurements": {
            "pan_distance": round(pan_distance, 4) if pan_distance is not None else None,
            "centroid_distance_octaves": (
                round(octave_distance, 4) if octave_distance is not None else None
            ),
            "level_delta_db": round(level_delta_db, 2) if level_delta_db is not None else None,
        },
        "components": {
            name: round(value, 4) if value is not None else None
            for name, value in components.items()
        },
        "recommended_next_analyzers": list(dict.fromkeys(analyzers)),
        "blockers": blockers,
        "evidence_provenance": {
            "channels": [
                {
                    "channel_id": item["channel_id"],
                    "object_id": item["id"],
                    "pan_source": item["pan_source"],
                    "frequency_source": item["frequency_source"],
                    "frequency_freshness": item["frequency_freshness"],
                    "frequency_age_sec": item["frequency_age_sec"],
                    "level_source": item["level_source"],
                    "recognition_source": item["recognition_source"],
                    "recognition_confidence": item["confidence"],
                    "read_coverage_status": item["read_coverage_status"],
                }
                for item in (first, second)
            ],
        },
        "explanation": (
            "Coarse pan, centroid and level evidence indicates competing mix space; "
            "run the listed analyzers before choosing a correction."
            if status == "review_required"
            else "Instrument Space relation is advisory and does not select a console value."
        ),
        **AUTHORITY_DENIAL,
    }


def _build_drift(
    current_objects: Sequence[Mapping[str, Any]],
    previous_scene: Optional[Mapping[str, Any]],
    *,
    current_generated_at: Any = None,
    current_source_identity: str = "",
) -> Dict[str, Any]:
    if not isinstance(previous_scene, Mapping):
        return {
            "status": "baseline_required",
            "changes": [],
            "reason": "previous_instrument_space_snapshot_not_provided",
            "baseline_generated_at": None,
            "current_generated_at": current_generated_at,
            **AUTHORITY_DENIAL,
        }
    if not _contract_is_safe(previous_scene):
        return {
            "status": "invalid",
            "changes": [],
            "reason": "previous_instrument_space_contract_not_read_only",
            "baseline_generated_at": None,
            "current_generated_at": current_generated_at,
            **AUTHORITY_DENIAL,
        }

    previous_source_context = _mapping(previous_scene.get("source_context"))
    previous_source_identity = str(previous_source_context.get("identity") or "")
    if (
        current_source_identity
        and previous_source_identity
        and previous_source_identity != current_source_identity
    ):
        return {
            "status": "invalid",
            "changes": [],
            "reason": "instrument_space_source_identity_changed",
            "baseline_generated_at": previous_scene.get("generated_at"),
            "current_generated_at": current_generated_at,
            **AUTHORITY_DENIAL,
        }

    previous_by_id = {
        str(item.get("id") or ""): _object_values(item)
        for item in previous_scene.get("objects") or []
        if isinstance(item, Mapping)
    }
    changes = []
    for current in current_objects:
        previous = previous_by_id.get(str(current["id"]))
        if not previous:
            continue
        deltas = {}
        crossed = []
        for axis, threshold in DRIFT_THRESHOLDS.items():
            current_value = current.get(axis)
            previous_value = previous.get(axis)
            if current_value is None or previous_value is None:
                deltas[axis] = None
                continue
            delta = round(float(current_value) - float(previous_value), 4)
            deltas[axis] = delta
            if abs(delta) >= threshold:
                crossed.append(axis)
        if crossed:
            changes.append(
                {
                    "id": f"instrument-space:drift:{current['channel_id']}",
                    "channel_id": current["channel_id"],
                    "label": current["label"],
                    "changed_axes": crossed,
                    "deltas": deltas,
                    "baseline_coordinates": {
                        axis: previous.get(axis) for axis in DRIFT_THRESHOLDS
                    },
                    "current_coordinates": {
                        axis: current.get(axis) for axis in DRIFT_THRESHOLDS
                    },
                    "status": "review_required",
                    **AUTHORITY_DENIAL,
                }
            )
    return {
        "status": "review_required" if changes else "stable",
        "changes": changes,
        "thresholds": dict(DRIFT_THRESHOLDS),
        "reason": None,
        "baseline_generated_at": previous_scene.get("generated_at"),
        "current_generated_at": current_generated_at,
        **AUTHORITY_DENIAL,
    }


def build_instrument_space_decision_support(
    scene: Mapping[str, Any],
    *,
    previous_scene: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build relation and drift evidence from a safe Instrument Space scene."""

    base = {
        "type": "instrument_space_decision_support",
        "schema_version": SCHEMA_VERSION,
        "source_schema_version": str(scene.get("schema_version") or ""),
        "source_generated_at": scene.get("generated_at"),
        "source_context": copy.deepcopy(dict(_mapping(scene.get("source_context")))),
        "mode": "advisory_only",
        "correction_policy": (
            "select_existing_analyzer_then_existing_proposal_and_safety_pipeline"
        ),
        "centroid_is_triage_not_eq_proof": True,
        "console_scope": ["wing", "dlive"],
        **AUTHORITY_DENIAL,
    }
    if not _contract_is_safe(scene):
        return {
            **base,
            "status": "invalid",
            "reason": "instrument_space_contract_not_read_only",
            "relations": [],
            "review_candidates": [],
            "proposal_annotation_index": {},
            "drift": _build_drift(
                [],
                previous_scene,
                current_generated_at=scene.get("generated_at"),
                current_source_identity=str(
                    _mapping(scene.get("source_context")).get("identity") or ""
                ),
            ),
            "summary": {
                "relation_count": 0,
                "review_required_count": 0,
                "watch_count": 0,
                "insufficient_evidence_count": 0,
            },
        }
    if str(scene.get("schema_version") or "") != "automixer.instrument_space.v1":
        return {
            **base,
            "status": "invalid",
            "reason": "instrument_space_schema_version_rejected",
            "relations": [],
            "review_candidates": [],
            "proposal_annotation_index": {},
            "drift": _build_drift(
                [],
                previous_scene,
                current_generated_at=scene.get("generated_at"),
                current_source_identity=str(
                    _mapping(scene.get("source_context")).get("identity") or ""
                ),
            ),
            "summary": {
                "relation_count": 0,
                "review_required_count": 0,
                "watch_count": 0,
                "insufficient_evidence_count": 0,
            },
        }

    objects = [
        _object_values(item)
        for item in list(scene.get("objects") or [])[:MAX_OBJECTS]
        if isinstance(item, Mapping)
    ]
    objects = [item for item in objects if item["channel_id"] > 0]
    relation_objects = [
        item
        for item in objects
        if item["status"] == "active"
        and sum(item[axis] is not None for axis in ("x_pan", "y_frequency", "z_level")) >= 2
    ]
    evaluated_relations = [
        _pair_relation(first, second)
        for index, first in enumerate(relation_objects)
        for second in relation_objects[index + 1 :]
    ]
    evaluated_relations.sort(key=lambda row: (-float(row["score"]), row["channel_ids"]))
    relations = [
        relation for relation in evaluated_relations if relation["status"] != "clear"
    ][:MAX_EMITTED_RELATIONS]
    candidates = [
        relation for relation in relations if relation["status"] in {"review_required", "watch"}
    ]
    annotation_index: Dict[str, List[Dict[str, Any]]] = {}
    for relation in candidates:
        compact = {
            "relation_id": relation["id"],
            "status": relation["status"],
            "score": relation["score"],
            "peer_channel_ids": relation["channel_ids"],
            "labels": relation["labels"],
            "measurements": relation["measurements"],
            "components": relation["components"],
            "recommended_next_analyzers": relation["recommended_next_analyzers"],
            "blockers": relation["blockers"],
            "evidence_provenance": relation["evidence_provenance"],
            "explanation": relation["explanation"],
            "source_generated_at": scene.get("generated_at"),
        }
        for channel_id in relation["channel_ids"]:
            annotation_index.setdefault(str(channel_id), []).append(compact)

    review_count = sum(row["status"] == "review_required" for row in evaluated_relations)
    watch_count = sum(row["status"] == "watch" for row in evaluated_relations)
    insufficient_count = sum(
        row["status"] == "insufficient_evidence" for row in evaluated_relations
    )
    return {
        **base,
        "status": "review_required" if review_count else ("watch" if watch_count else "clear"),
        "reason": None,
        "relations": relations,
        "review_candidates": candidates,
        "proposal_annotation_index": annotation_index,
        "drift": _build_drift(
            objects,
            previous_scene,
            current_generated_at=scene.get("generated_at"),
            current_source_identity=str(
                _mapping(scene.get("source_context")).get("identity") or ""
            ),
        ),
        "summary": {
            "object_count": len(objects),
            "relation_object_count": len(relation_objects),
            "relation_count": len(evaluated_relations),
            "emitted_relation_count": len(relations),
            "relations_truncated": len(relations) < sum(
                row["status"] != "clear" for row in evaluated_relations
            ),
            "review_required_count": review_count,
            "watch_count": watch_count,
            "insufficient_evidence_count": insufficient_count,
            "annotated_channel_count": len(annotation_index),
        },
    }


def annotate_operator_proposal_with_instrument_space(
    proposal: Mapping[str, Any],
    decision_support: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Attach evidence without changing a proposed value or its authority."""

    annotated = copy.deepcopy(dict(proposal))
    if not isinstance(decision_support, Mapping):
        return annotated
    contract_safe = _contract_is_safe(decision_support) and str(
        decision_support.get("schema_version") or ""
    ) == SCHEMA_VERSION
    source_age_sec = _utc_age_seconds(decision_support.get("source_generated_at"))
    fresh = source_age_sec is not None and source_age_sec <= PROPOSAL_EVIDENCE_MAX_AGE_SEC
    safe = contract_safe and fresh
    channel_id = int(_finite(annotated.get("channel") or annotated.get("target")) or 0)
    index = _mapping(decision_support.get("proposal_annotation_index"))
    matches = index.get(str(channel_id)) if safe and channel_id > 0 else None
    matches = [dict(item) for item in matches or [] if isinstance(item, Mapping)]
    annotation = {
        "accepted": safe,
        "status": (
            "review_required"
            if any(item.get("status") == "review_required" for item in matches)
            else ("watch" if matches else "clear")
        ) if safe else "rejected",
        "reason": (
            None
            if safe
            else (
                "instrument_space_decision_support_contract_rejected"
                if not contract_safe
                else "instrument_space_decision_support_stale_or_undated"
            )
        ),
        "source_generated_at": decision_support.get("source_generated_at"),
        "source_age_sec": round(source_age_sec, 3) if source_age_sec is not None else None,
        "max_source_age_sec": PROPOSAL_EVIDENCE_MAX_AGE_SEC,
        "channel_id": channel_id or None,
        "relations": matches,
        "recommended_next_analyzers": list(
            dict.fromkeys(
                analyzer
                for item in matches
                for analyzer in item.get("recommended_next_analyzers") or []
            )
        ),
        "proposal_value_changed": False,
        "transaction_authority_changed": False,
        **AUTHORITY_DENIAL,
    }
    raw = annotated.get("raw")
    raw = dict(raw) if isinstance(raw, Mapping) else {}
    raw["instrument_space_decision_support"] = annotation
    annotated["raw"] = raw
    annotated["instrument_space_review"] = annotation
    return annotated


def annotate_operator_proposals_with_instrument_space(
    proposals: Iterable[Mapping[str, Any]],
    decision_support: Optional[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        annotate_operator_proposal_with_instrument_space(proposal, decision_support)
        for proposal in proposals
    ]
