"""Read-only operator review queue for physical readback artifact intake.

The queue aggregates the per-surface artifact-intake snapshots into one
operator-facing review status. It does not close fixture gaps, query hardware,
execute tests, or grant live authority.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, Mapping, Tuple

try:
    from bus_group_main_dca_physical_readback_artifact_intake import (
        build_bus_group_main_dca_physical_readback_artifact_intake,
    )
    from channel_basic_controls_physical_readback_artifact_intake import (
        build_channel_basic_controls_physical_readback_artifact_intake,
    )
    from channel_processing_physical_readback_artifact_intake import (
        build_channel_processing_physical_readback_artifact_intake,
    )
    from dlive_state_physical_readback_artifact_intake import (
        build_dlive_state_physical_readback_artifact_intake,
    )
    from fx_slots_inserts_physical_readback_artifact_intake import (
        build_fx_slots_inserts_physical_readback_artifact_intake,
    )
    from matrix_state_physical_readback_artifact_intake import (
        build_matrix_state_physical_readback_artifact_intake,
    )
    from raw_mixer_state_physical_readback_artifact_intake import (
        build_raw_mixer_state_physical_readback_artifact_intake,
    )
    from routing_sends_physical_readback_artifact_intake import (
        build_routing_sends_physical_readback_artifact_intake,
    )
    from snapshots_scenes_physical_readback_artifact_intake import (
        build_snapshots_scenes_physical_readback_artifact_intake,
    )
    from user_presets_physical_readback_artifact_intake import (
        build_user_presets_physical_readback_artifact_intake,
    )
except ImportError:  # pragma: no cover - package import compatibility
    from backend.bus_group_main_dca_physical_readback_artifact_intake import (
        build_bus_group_main_dca_physical_readback_artifact_intake,
    )
    from backend.channel_basic_controls_physical_readback_artifact_intake import (
        build_channel_basic_controls_physical_readback_artifact_intake,
    )
    from backend.channel_processing_physical_readback_artifact_intake import (
        build_channel_processing_physical_readback_artifact_intake,
    )
    from backend.dlive_state_physical_readback_artifact_intake import (
        build_dlive_state_physical_readback_artifact_intake,
    )
    from backend.fx_slots_inserts_physical_readback_artifact_intake import (
        build_fx_slots_inserts_physical_readback_artifact_intake,
    )
    from backend.matrix_state_physical_readback_artifact_intake import (
        build_matrix_state_physical_readback_artifact_intake,
    )
    from backend.raw_mixer_state_physical_readback_artifact_intake import (
        build_raw_mixer_state_physical_readback_artifact_intake,
    )
    from backend.routing_sends_physical_readback_artifact_intake import (
        build_routing_sends_physical_readback_artifact_intake,
    )
    from backend.snapshots_scenes_physical_readback_artifact_intake import (
        build_snapshots_scenes_physical_readback_artifact_intake,
    )
    from backend.user_presets_physical_readback_artifact_intake import (
        build_user_presets_physical_readback_artifact_intake,
    )


READ_SURFACE_PHYSICAL_READBACK_ARTIFACT_REVIEW_QUEUE_METHOD_ID = (
    "read_surface_physical_readback_artifact_review_queue"
)
READ_SURFACE_PHYSICAL_READBACK_ARTIFACT_REVIEW_QUEUE_STATUSES = {
    "operator_artifact_capture_required",
    "operator_review_candidates_available",
    "operator_artifact_replacement_required",
}
UNSAFE_AUTHORITY_FLAGS = (
    "live_write",
    "mutates_console_state",
    "queries_console",
    "executes_tests",
    "execution_allowed_by_intake",
    "fixture_closure_allowed_by_intake",
    "live_widening_allowed",
    "live_authority_granted",
    "full_autonomous_ready",
    "scene_load_allowed",
    "scene_save_allowed",
    "preset_recall_allowed",
    "preset_store_allowed",
    "midi_send_allowed",
    "tcp_send_allowed",
    "dlive_write_allowed",
    "dlive_apply_allowed",
)


IntakeBuilder = Callable[[], Mapping[str, Any]]


INTAKE_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "method_id": "raw_mixer_state_physical_readback_artifact_intake",
        "surface_id": "raw_mixer_state_cache",
        "remediation_action_id": "remediate_raw_mixer_state_cache",
        "required_contract_method_id": "raw_mixer_state_physical_readback_contract",
        "builder": build_raw_mixer_state_physical_readback_artifact_intake,
    },
    {
        "method_id": "channel_basic_controls_physical_readback_artifact_intake",
        "surface_id": "channel_identity_and_basic_controls",
        "remediation_action_id": "remediate_channel_identity_and_basic_controls",
        "required_contract_method_id": "channel_basic_controls_physical_readback_contract",
        "builder": build_channel_basic_controls_physical_readback_artifact_intake,
    },
    {
        "method_id": "channel_processing_physical_readback_artifact_intake",
        "surface_id": "channel_eq_dynamics_gate",
        "remediation_action_id": "remediate_channel_eq_dynamics_gate",
        "required_contract_method_id": "channel_processing_physical_readback_contract",
        "builder": build_channel_processing_physical_readback_artifact_intake,
    },
    {
        "method_id": "bus_group_main_dca_physical_readback_artifact_intake",
        "surface_id": "bus_group_main_dca_state",
        "remediation_action_id": "remediate_bus_group_main_dca_state",
        "required_contract_method_id": "bus_group_main_dca_physical_readback_contract",
        "builder": build_bus_group_main_dca_physical_readback_artifact_intake,
    },
    {
        "method_id": "matrix_state_physical_readback_artifact_intake",
        "surface_id": "matrix_state",
        "remediation_action_id": "remediate_matrix_state",
        "required_contract_method_id": "matrix_state_physical_readback_contract",
        "builder": build_matrix_state_physical_readback_artifact_intake,
    },
    {
        "method_id": "fx_slots_inserts_physical_readback_artifact_intake",
        "surface_id": "fx_slots_and_inserts",
        "remediation_action_id": "remediate_fx_slots_and_inserts",
        "required_contract_method_id": "fx_slots_inserts_physical_readback_contract",
        "builder": build_fx_slots_inserts_physical_readback_artifact_intake,
    },
    {
        "method_id": "routing_sends_physical_readback_artifact_intake",
        "surface_id": "routing_and_sends",
        "remediation_action_id": "remediate_routing_and_sends",
        "required_contract_method_id": "routing_sends_physical_readback_contract",
        "builder": build_routing_sends_physical_readback_artifact_intake,
    },
    {
        "method_id": "snapshots_scenes_physical_readback_artifact_intake",
        "surface_id": "snapshots_scenes",
        "remediation_action_id": "remediate_snapshots_scenes",
        "required_contract_method_id": "snapshots_scenes_physical_readback_contract",
        "builder": build_snapshots_scenes_physical_readback_artifact_intake,
    },
    {
        "method_id": "user_presets_physical_readback_artifact_intake",
        "surface_id": "user_presets",
        "remediation_action_id": "remediate_user_presets",
        "required_contract_method_id": "user_presets_physical_readback_contract",
        "builder": build_user_presets_physical_readback_artifact_intake,
    },
    {
        "method_id": "dlive_state_physical_readback_artifact_intake",
        "surface_id": "dlive_state",
        "remediation_action_id": "remediate_dlive_state",
        "required_contract_method_id": "dlive_state_physical_readback_contract",
        "builder": build_dlive_state_physical_readback_artifact_intake,
    },
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_int(value: Any) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def _string_list(value: Any) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str) and item)


def _artifact_source_ids(
    summary: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> Tuple[str, ...]:
    sources: list[str] = []
    sources.extend(_string_list(summary.get("artifact_source_ids")))
    latest_source = summary.get("latest_valid_candidate_source_id")
    if isinstance(latest_source, str) and latest_source:
        sources.append(latest_source)
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), list) else []
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            continue
        if artifact.get("accepted_as_physical_fixture_candidate") is not True:
            continue
        source_id = artifact.get("source_id")
        if isinstance(source_id, str) and source_id:
            sources.append(source_id)
    return tuple(dict.fromkeys(sources))


def _pick_bool(summary: Mapping[str, Any], safety: Mapping[str, Any], key: str) -> bool:
    return summary.get(key) is True or safety.get(key) is True


def _intake_snapshot_map(
    intake_snapshots: Mapping[str, Mapping[str, Any]] | None,
) -> Dict[str, Mapping[str, Any]]:
    if intake_snapshots is not None:
        return {
            method_id: _mapping(intake_snapshots.get(method_id))
            for method_id in (str(spec["method_id"]) for spec in INTAKE_SPECS)
        }
    snapshots: Dict[str, Mapping[str, Any]] = {}
    for spec in INTAKE_SPECS:
        builder = spec["builder"]
        snapshots[str(spec["method_id"])] = _mapping(builder())
    return snapshots


def _queue_item(spec: Mapping[str, Any], payload: Mapping[str, Any]) -> Dict[str, Any]:
    summary = _mapping(payload.get("summary")) or payload
    safety = _mapping(payload.get("safety")) or payload
    method_id = str(spec["method_id"])
    surface_id = summary.get("surface_id") or payload.get("surface_id")
    remediation_action_id = (
        summary.get("remediation_action_id") or payload.get("remediation_action_id")
    )
    required_contract_method_id = (
        summary.get("required_contract_method_id")
        or payload.get("required_contract_method_id")
    )
    available = bool(payload)
    target_consistency = (
        available
        and surface_id == spec["surface_id"]
        and remediation_action_id == spec["remediation_action_id"]
        and required_contract_method_id == spec["required_contract_method_id"]
    )
    artifact_count = _safe_int(summary.get("artifact_count"))
    valid_candidate_count = _safe_int(summary.get("valid_candidate_count"))
    invalid_candidate_count = _safe_int(summary.get("invalid_candidate_count"))
    unsafe_flags = tuple(
        key for key in UNSAFE_AUTHORITY_FLAGS if _pick_bool(summary, safety, key)
    )
    operator_review_required = summary.get("operator_review_required") is True
    physical_gap_remains = summary.get("physical_fixture_gap_remains") is True
    source_candidate_claimed = (
        summary.get("accepted_as_physical_fixture_candidate") is True
    )
    artifact_source_ids = _artifact_source_ids(summary, payload)
    candidate_source_id_missing = (
        source_candidate_claimed and valid_candidate_count > 0 and not artifact_source_ids
    )
    review_candidate = (
        source_candidate_claimed
        and target_consistency
        and not candidate_source_id_missing
        and valid_candidate_count > 0
        and invalid_candidate_count == 0
        and operator_review_required
        and physical_gap_remains
        and not unsafe_flags
    )
    if (
        unsafe_flags
        or invalid_candidate_count
        or candidate_source_id_missing
        or (available and not target_consistency)
    ):
        status = "operator_artifact_replacement_required"
    elif review_candidate:
        status = "operator_review_candidate_available"
    elif artifact_count == 0:
        status = "operator_artifact_missing"
    else:
        status = "operator_artifact_replacement_required"

    return {
        "method_id": method_id,
        "available": available,
        "schema_id": payload.get("schema_id") or summary.get("schema_id"),
        "intake_status": payload.get("status") or summary.get("intake_status"),
        "surface_id": surface_id if isinstance(surface_id, str) else spec["surface_id"],
        "remediation_action_id": (
            remediation_action_id
            if isinstance(remediation_action_id, str)
            else spec["remediation_action_id"]
        ),
        "required_contract_method_id": (
            required_contract_method_id
            if isinstance(required_contract_method_id, str)
            else spec["required_contract_method_id"]
        ),
        "status": status,
        "artifact_count": artifact_count,
        "valid_candidate_count": valid_candidate_count,
        "invalid_candidate_count": invalid_candidate_count,
        "accepted_as_physical_fixture_candidate": review_candidate,
        "source_accepted_as_physical_fixture_candidate_claimed": source_candidate_claimed,
        "operator_review_required": True,
        "source_operator_review_required_claimed": operator_review_required,
        "physical_fixture_gap_remains": True,
        "source_physical_fixture_gap_remains_claimed": physical_gap_remains,
        "fixture_closure_allowed_by_queue": False,
        "fixture_closure_allowed_by_intake": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
        "target_consistency": target_consistency,
        "candidate_source_id_missing": candidate_source_id_missing,
        "unsafe_authority_flags": list(unsafe_flags),
        "artifact_source_ids": list(artifact_source_ids),
        "blocked_by": list(_string_list(summary.get("blocked_by"))),
        "violations": list(_string_list(summary.get("violations"))),
    }


def build_read_surface_physical_readback_artifact_review_queue(
    *,
    intake_snapshots: Mapping[str, Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Build a read-only review queue from physical readback artifact intakes."""

    snapshots = _intake_snapshot_map(intake_snapshots)
    queue_items = tuple(
        _queue_item(spec, snapshots.get(str(spec["method_id"]), {}))
        for spec in INTAKE_SPECS
    )
    review_candidates = tuple(
        item for item in queue_items if item["accepted_as_physical_fixture_candidate"]
    )
    invalid_items = tuple(
        item
        for item in queue_items
        if item["invalid_candidate_count"] > 0
        or item["unsafe_authority_flags"]
        or item["candidate_source_id_missing"]
        or not item["target_consistency"]
    )
    missing_items = tuple(
        item for item in queue_items if item["artifact_count"] == 0
    )
    if invalid_items:
        status = "operator_artifact_replacement_required"
        blocked_by = ("operator_must_replace_invalid_or_unsafe_physical_readback_artifacts",)
    elif review_candidates:
        status = "operator_review_candidates_available"
        blocked_by = ("operator_must_review_valid_candidates_before_fixture_closure",)
    else:
        status = "operator_artifact_capture_required"
        blocked_by = ("operator_must_capture_missing_physical_readback_artifacts",)

    promotion_families = tuple(
        dict.fromkeys(
            family
            for item in queue_items
            for family in _string_list(
                _mapping(snapshots.get(item["method_id"], {})).get("summary", {}).get(
                    "promotion_families_requiring_surface"
                )
                if isinstance(
                    _mapping(snapshots.get(item["method_id"], {})).get("summary"),
                    Mapping,
                )
                else _mapping(snapshots.get(item["method_id"], {})).get(
                    "promotion_families_requiring_surface"
                )
            )
        )
    )
    artifact_count = sum(item["artifact_count"] for item in queue_items)
    valid_candidate_count = sum(item["valid_candidate_count"] for item in queue_items)
    invalid_candidate_count = sum(item["invalid_candidate_count"] for item in queue_items)
    return {
        "type": "read_surface_physical_readback_artifact_review_queue",
        "generated_at": _utc_now(),
        "method_id": READ_SURFACE_PHYSICAL_READBACK_ARTIFACT_REVIEW_QUEUE_METHOD_ID,
        "status": status,
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_queue": False,
            "fixture_closure_allowed_by_queue": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "intake_surface_count": len(queue_items),
            "available_intake_count": sum(1 for item in queue_items if item["available"]),
            "artifact_count": artifact_count,
            "valid_candidate_count": valid_candidate_count,
            "invalid_candidate_count": invalid_candidate_count,
            "review_candidate_count": len(review_candidates),
            "missing_artifact_surface_count": len(missing_items),
            "unsafe_authority_claim_count": sum(
                1 for item in queue_items if item["unsafe_authority_flags"]
            ),
            "candidate_source_id_missing_count": sum(
                1 for item in queue_items if item["candidate_source_id_missing"]
            ),
            "target_inconsistent_surface_count": sum(
                1 for item in queue_items if not item["target_consistency"]
            ),
            "review_candidate_surface_ids": [
                item["surface_id"] for item in review_candidates
            ],
            "invalid_surface_ids": [item["surface_id"] for item in invalid_items],
            "missing_artifact_surface_ids": [item["surface_id"] for item in missing_items],
            "unsafe_authority_surface_ids": [
                item["surface_id"] for item in queue_items if item["unsafe_authority_flags"]
            ],
            "candidate_source_id_missing_surface_ids": [
                item["surface_id"]
                for item in queue_items
                if item["candidate_source_id_missing"]
            ],
            "target_inconsistent_surface_ids": [
                item["surface_id"] for item in queue_items if not item["target_consistency"]
            ],
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "execution_allowed_by_queue": False,
            "fixture_closure_allowed_by_queue": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "promotion_families_requiring_surface": list(promotion_families),
            "blocked_by": list(blocked_by),
            "next_global_step": (
                "Operator replaces invalid or unsafe physical readback artifacts."
                if invalid_items
                else (
                    "Operator reviews valid physical readback candidates before "
                    "any fixture gap can be closed."
                    if review_candidates
                    else "Operator captures missing physical readback artifacts."
                )
            ),
        },
        "queue_items": list(queue_items),
    }
