"""Strict durable-WAL contract for the finite M110 maintenance reset."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping

from maintenance_dsp_reset import (
    M110_PRESERVATION_ADDRESSES,
    M110_RESET_ADDRESSES,
    M110_RESET_BUNDLE_KIND,
    M110_RESET_OPERATION_DEFINITIONS,
    compare_preservation_manifest_rows,
    m110_reset_review_preimage_document,
)


M110_WAL_CONTRACT = "automixer.m110_maintenance_wal.v1"
M110_WAL_PAYLOAD_FIELD = "wal_payload"
M110_WAL_PAYLOAD_SHA256_FIELD = "wal_payload_sha256"
M110_WAL_RESERVED_PAYLOAD_FIELDS = frozenset(
    {
        "schema",
        "timestamp",
        "transaction_id",
        "phase",
        "wal_contract",
        M110_WAL_PAYLOAD_FIELD,
        M110_WAL_PAYLOAD_SHA256_FIELD,
    }
)

M110_LIFECYCLE_RESOLVED = "resolved"
M110_LIFECYCLE_UNRESOLVED = "unresolved"
M110_LIFECYCLE_INVALID = "invalid"

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
_SUCCESS_PHASES = frozenset({"committed", "proof_restored"})
_PENDING_PHASES = (
    "verification_pending",
    "verified_state_pending_cleanup",
)
_EXPECTED_OPERATION_IDS = tuple(
    str(row["operation_id"])
    for row in M110_RESET_OPERATION_DEFINITIONS
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def freeze_wal_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Own one JSON-safe payload snapshot before opening the WAL file."""

    if type(payload) is not dict:
        raise TypeError("m110_wal_payload_builtin_dict_required")
    if M110_WAL_RESERVED_PAYLOAD_FIELDS.intersection(payload):
        raise ValueError("m110_wal_payload_reserved_field")
    frozen = json.loads(canonical_json_bytes(payload).decode("utf-8"))
    if type(frozen) is not dict:
        raise ValueError("m110_wal_payload_object_required")
    return frozen


def _valid_sha(value: Any) -> bool:
    return bool(_SHA256_RE.fullmatch(str(value or "")))


def _event_payload(event: Any) -> dict[str, Any] | None:
    if not isinstance(event, Mapping):
        return None
    payload = event.get(M110_WAL_PAYLOAD_FIELD)
    if not (
        event.get("schema") == "automixer.manual_write_wal.v1"
        and event.get("wal_contract") == M110_WAL_CONTRACT
        and event.get("scope") == "supervised_transaction"
        and event.get("maintenance_dsp_reset") is True
        and type(payload) is dict
        and _valid_sha(event.get(M110_WAL_PAYLOAD_SHA256_FIELD))
        and not M110_WAL_RESERVED_PAYLOAD_FIELDS.intersection(payload)
    ):
        return None
    try:
        payload_sha256 = canonical_sha256(payload)
    except (TypeError, ValueError, OverflowError):
        return None
    if event.get(M110_WAL_PAYLOAD_SHA256_FIELD) != payload_sha256:
        return None
    return payload


def _is_m110_marker(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    if (
        value.get("wal_contract") == M110_WAL_CONTRACT
        or value.get("maintenance_dsp_reset") is True
        or value.get("bundle_kind") == M110_RESET_BUNDLE_KIND
    ):
        return True
    expected = value.get("expected_operation_ids")
    if isinstance(expected, list) and tuple(expected) == _EXPECTED_OPERATION_IDS:
        return True
    operations = value.get("operations")
    return bool(
        isinstance(operations, list)
        and any(
            isinstance(row, Mapping)
            and row.get("family_id") == "maintenance_dsp_reset"
            for row in operations
        )
    )


def lifecycle_is_m110(lifecycle: Any) -> bool:
    if not isinstance(lifecycle, Mapping):
        return False
    phases = tuple(lifecycle.get("phases") or ())
    if any(phase in _PENDING_PHASES for phase in phases):
        return True
    for event in lifecycle.get("events") or ():
        if _is_m110_marker(event) or _is_m110_marker(
            event.get(M110_WAL_PAYLOAD_FIELD)
            if isinstance(event, Mapping)
            else None
        ):
            return True
    return False


def _lifecycle_events(lifecycle: Any) -> tuple[Mapping[str, Any], ...] | None:
    if not isinstance(lifecycle, Mapping):
        return None
    transaction_id = str(lifecycle.get("transaction_id") or "").strip()
    phases = tuple(lifecycle.get("phases") or ())
    events_value = lifecycle.get("events")
    if not transaction_id or not isinstance(events_value, list):
        return None
    events = tuple(events_value)
    if not events or len(events) != len(phases):
        return None
    for index, event in enumerate(events):
        if not (
            isinstance(event, Mapping)
            and event.get("schema") == "automixer.manual_write_wal.v1"
            and str(event.get("transaction_id") or "") == transaction_id
            and str(event.get("phase") or "") == str(phases[index])
        ):
            return None
    if lifecycle.get("prepared") != events[0]:
        return None
    if lifecycle.get("latest_event") != events[-1]:
        return None
    return events


def _binding(payload: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        payload.get(field)
        for field in (
            "scope",
            "maintenance_dsp_reset",
            "proposal_id",
            "approval_id",
            "idempotency_key",
            "review_content_digest",
            "operation_digest",
            "bundle_kind",
            "proof_receipt_id",
            "proof_run",
            "restore_after_verify",
            "capability_kind",
        )
    )


def _validate_prepared(payload: Any) -> dict[str, Any] | None:
    if type(payload) is not dict:
        return None
    proof_run = payload.get("proof_run") is True
    expected_capability = (
        "restore_only_proof"
        if proof_run
        else "maintenance_dsp_reset_commit_once"
    )
    if not (
        payload.get("scope") == "supervised_transaction"
        and payload.get("maintenance_dsp_reset") is True
        and payload.get("bundle_kind") == M110_RESET_BUNDLE_KIND
        and payload.get("restore_after_verify") is proof_run
        and payload.get("capability_kind") == expected_capability
        and bool(str(payload.get("proposal_id") or "").strip())
        and bool(str(payload.get("approval_id") or "").strip())
        and bool(str(payload.get("idempotency_key") or "").strip())
        and _valid_sha(payload.get("review_content_digest"))
        and _valid_sha(payload.get("operation_digest"))
        and tuple(payload.get("expected_operation_ids") or ())
        == _EXPECTED_OPERATION_IDS
    ):
        return None
    proof_receipt_id = str(payload.get("proof_receipt_id") or "").strip()
    if (proof_run and proof_receipt_id) or (not proof_run and not proof_receipt_id):
        return None
    reset_document = payload.get("reset_preimage")
    before_manifest = payload.get("before_manifest")
    if not (
        isinstance(reset_document, Mapping)
        and isinstance(before_manifest, Mapping)
        and reset_document.get("row_count") == 64
        and before_manifest.get("row_count") == 184
        and [
            row.get("address") if isinstance(row, Mapping) else None
            for row in (reset_document.get("rows") or ())
        ]
        == list(M110_RESET_ADDRESSES)
        and [
            row.get("address") if isinstance(row, Mapping) else None
            for row in (before_manifest.get("rows") or ())
        ]
        == list(M110_PRESERVATION_ADDRESSES)
    ):
        return None
    try:
        reviewed = m110_reset_review_preimage_document(
            {
                "reset_rows": list(reset_document["rows"]),
                "preservation_manifest": dict(before_manifest),
            }
        )
    except (TypeError, ValueError):
        return None
    if not (
        reviewed["reset_preimage"] == dict(reset_document)
        and reviewed["preservation_manifest"] == dict(before_manifest)
        and payload.get("review_preimage_sha256")
        == reviewed["review_preimage_sha256"]
        and payload.get("reset_preimage_sha256")
        == reviewed["reset_preimage_sha256"]
        and payload.get("before_manifest_sha256")
        == reviewed["preservation_manifest_sha256"]
        and payload.get("allowed_reset_surface_sha256")
        == canonical_sha256(list(M110_RESET_ADDRESSES))
    ):
        return None
    preimage = {
        str(row["address"]): int(row["value"])
        for row in reset_document["rows"]
    }
    operations = payload.get("operations")
    if not isinstance(operations, list) or len(operations) != 64:
        return None
    for expected, actual in zip(M110_RESET_OPERATION_DEFINITIONS, operations):
        if not isinstance(actual, Mapping):
            return None
        if any(
            actual.get(field) != expected[field]
            for field in (
                "operation_id",
                "family_id",
                "write_kind",
                "channel",
                "parameter",
            )
        ):
            return None
        if actual.get("writes") != [
            {
                "address": expected["address"],
                "target_value": 0,
                "tolerance": 0.0,
            }
        ]:
            return None
        if actual.get("preimage_values") != {
            expected["address"]: preimage[expected["address"]]
        }:
            return None
    return {
        "payload": payload,
        "proof_run": proof_run,
        "binding": _binding(payload),
        "preimage": preimage,
        "before_manifest": dict(before_manifest),
        "before_manifest_sha256": payload["before_manifest_sha256"],
    }


def _binding_matches(context: Mapping[str, Any], payload: Any) -> bool:
    return bool(
        isinstance(payload, Mapping)
        and _binding(payload) == context.get("binding")
    )


def _validate_pending(
    context: Mapping[str, Any],
    payload: Any,
    phase: str,
) -> bool:
    if not _binding_matches(context, payload):
        return False
    proof_run = context.get("proof_run") is True
    success_phase = "proof_restored" if proof_run else "committed"
    success_status = "proof_passed" if proof_run else "applied"
    success_reason = (
        "m110_reset_proof_applied_and_restored"
        if proof_run
        else "m110_maintenance_dsp_reset_committed"
    )
    pending_reason = {
        "verification_pending": "m110_final_verification_pending",
        "verified_state_pending_cleanup": (
            "m110_verified_state_waiting_for_guard_cleanup"
        ),
    }.get(phase)
    reset_rows = list(context["payload"]["reset_preimage"]["rows"])
    before_manifest = context["before_manifest"]
    after_manifest = _validate_manifest(reset_rows, payload.get("after_manifest"))
    if after_manifest is None:
        return False
    try:
        manifest_matches = not compare_preservation_manifest_rows(
            list(before_manifest["rows"]),
            list(after_manifest["rows"]),
        )
    except (TypeError, ValueError):
        return False
    observer = payload.get("delivered_observer_audit")
    gate = payload.get("mutation_audit")
    audits_safe = bool(
        isinstance(observer, Mapping)
        and isinstance(gate, Mapping)
        and payload.get("delivered_observer_audit_sha256")
        == canonical_sha256(observer)
        and payload.get("mutation_audit_sha256") == canonical_sha256(gate)
        and payload.get("delivered_observer_clean") is True
        and observer.get("active") is True
        and observer.get("unsubscribe_failed") is False
        and observer.get("overflow") is False
        and not observer.get("unexpected_events")
        and payload.get("no_outside_mutation_attempt_proven") is True
        and gate.get("active") is True
        and gate.get("sealed") is True
        and gate.get("remove_failed") is False
        and gate.get("boundary_state_unknown") is False
        and gate.get("overflow") is False
        and gate.get("no_outside_mutation_attempt_proven") is True
        and not gate.get("denied_attempts")
    )
    incident_fields = {
        "post_linearization_incident",
        "linearized_terminal_phase",
        "success_terminal_attempted",
        "success_terminal_durable_confirmed",
        "compensation_attempted",
        "rollback_attempted_after_linearization",
    }
    return bool(
        pending_reason
        and payload.get("result_status") == phase
        and payload.get("result_reason") == pending_reason
        and payload.get("verification_candidate_status") == success_status
        and payload.get("verification_candidate_reason") == success_reason
        and payload.get("verification_candidate_phase") == success_phase
        and payload.get("applied_operation_ids")
        == list(_EXPECTED_OPERATION_IDS)
        and payload.get("rollback_operation_ids")
        == (list(reversed(_EXPECTED_OPERATION_IDS)) if proof_run else [])
        and payload.get("physical_verified") is False
        and payload.get("audio_recheck_passed") is True
        and payload.get("proof_target_verified") is False
        and payload.get("physical_state_restored") is proof_run
        and payload.get("terminal_after_console_send") is True
        and payload.get("rollback_attempted") is proof_run
        and payload.get("rollback_succeeded") is proof_run
        and payload.get("write_state_unknown") is False
        and payload.get("rollback_state_unknown") is False
        and payload.get("manual_review_required") is False
        and payload.get("emergency_stop_activated") is False
        and payload.get("emergency_stop_activation_failed") is False
        and payload.get("emergency_stop_required") is False
        and payload.get("accepted") is False
        and payload.get("owner_review_required") is True
        and payload.get("live_authority_granted") is False
        and payload.get("readiness_closure_allowed") is False
        and payload.get("before_manifest") == before_manifest
        and payload.get("before_manifest_sha256")
        == context.get("before_manifest_sha256")
        and payload.get("after_manifest_sha256")
        == canonical_sha256(after_manifest)
        and manifest_matches
        and payload.get("preservation_mismatches") == []
        and payload.get("physical_truth_scope")
        == "not_verified_at_terminal_cut"
        and payload.get("verification_cut_epoch") is None
        and payload.get("verification_cut_reset_surface_sha256") == ""
        and payload.get("verification_cut_preservation_manifest_sha256") == ""
        and payload.get("current_state_guaranteed_at_response") is False
        and payload.get("attended_no_touch_required_during_execution") is True
        and payload.get("fresh_readback_required_before_next_stage") is True
        and payload.get("later_console_event_invalidates_readiness_freshness")
        is True
        and payload.get("cancellation_requested") is False
        and audits_safe
        and not incident_fields.intersection(payload)
    )


def _validate_manifest(
    reset_rows: list[dict[str, Any]],
    manifest: Any,
) -> dict[str, Any] | None:
    if not isinstance(manifest, Mapping):
        return None
    try:
        reviewed = m110_reset_review_preimage_document(
            {
                "reset_rows": reset_rows,
                "preservation_manifest": dict(manifest),
            }
        )
    except (TypeError, ValueError):
        return None
    if reviewed["preservation_manifest"] != dict(manifest):
        return None
    return dict(manifest)


def _validate_audits(payload: Mapping[str, Any]) -> bool:
    observer = payload.get("delivered_observer_audit")
    gate = payload.get("mutation_audit")
    return bool(
        isinstance(observer, Mapping)
        and isinstance(gate, Mapping)
        and payload.get("delivered_observer_audit_sha256")
        == canonical_sha256(observer)
        and payload.get("mutation_audit_sha256") == canonical_sha256(gate)
        and payload.get("delivered_observer_clean") is True
        and observer.get("active") is False
        and observer.get("unsubscribe_failed") is False
        and observer.get("overflow") is False
        and not observer.get("unexpected_events")
        and payload.get("no_outside_mutation_attempt_proven") is True
        and gate.get("active") is True
        and gate.get("sealed") is True
        and gate.get("remove_failed") is False
        and gate.get("boundary_state_unknown") is False
        and gate.get("overflow") is False
        and gate.get("no_outside_mutation_attempt_proven") is True
        and not gate.get("denied_attempts")
    )


def _validate_no_send_abort_audits(payload: Mapping[str, Any]) -> bool:
    """Accept a blocked outsider only when no owned transport send occurred."""

    observer = payload.get("delivered_observer_audit")
    gate = payload.get("mutation_audit")
    if not (isinstance(observer, Mapping) and isinstance(gate, Mapping)):
        return False
    denied_attempts = gate.get("denied_attempts")
    if not isinstance(denied_attempts, list):
        return False
    no_outside_attempt = not bool(denied_attempts)
    return bool(
        payload.get("delivered_observer_audit_sha256")
        == canonical_sha256(observer)
        and payload.get("mutation_audit_sha256") == canonical_sha256(gate)
        and payload.get("delivered_observer_clean") is True
        and observer.get("active") is False
        and observer.get("unsubscribe_failed") is False
        and observer.get("overflow") is False
        and not observer.get("unexpected_events")
        and payload.get("no_outside_mutation_attempt_proven")
        is no_outside_attempt
        and gate.get("active") is True
        and gate.get("sealed") is True
        and gate.get("remove_failed") is False
        and gate.get("boundary_state_unknown") is False
        and gate.get("overflow") is False
        and gate.get("no_outside_mutation_attempt_proven")
        is no_outside_attempt
    )


def _validate_success(
    context: Mapping[str, Any],
    payload: Any,
    phase: str,
) -> bool:
    if not _binding_matches(context, payload):
        return False
    proof_run = context.get("proof_run") is True
    if phase != ("proof_restored" if proof_run else "committed"):
        return False
    expected_status = "proof_passed" if proof_run else "applied"
    expected_reason = (
        "m110_reset_proof_applied_and_restored"
        if proof_run
        else "m110_maintenance_dsp_reset_committed"
    )
    reset_rows = list(context["payload"]["reset_preimage"]["rows"])
    before_manifest = context["before_manifest"]
    after_manifest = _validate_manifest(reset_rows, payload.get("after_manifest"))
    if after_manifest is None:
        return False
    try:
        manifest_matches = not compare_preservation_manifest_rows(
            list(before_manifest["rows"]),
            list(after_manifest["rows"]),
        )
    except (TypeError, ValueError):
        return False
    expected_reset_values = (
        context["preimage"]
        if proof_run
        else {address: 0 for address in M110_RESET_ADDRESSES}
    )
    expected_cut_sha = canonical_sha256(
        [
            {"address": address, "value": expected_reset_values[address]}
            for address in M110_RESET_ADDRESSES
        ]
    )
    cut_epoch = payload.get("verification_cut_epoch")
    if not (
        payload.get("result_status") == expected_status
        and payload.get("result_reason") == expected_reason
        and payload.get("verification_candidate_status") == expected_status
        and payload.get("verification_candidate_reason") == expected_reason
        and payload.get("verification_candidate_phase") == phase
        and payload.get("applied_operation_ids")
        == list(_EXPECTED_OPERATION_IDS)
        and payload.get("rollback_operation_ids")
        == (list(reversed(_EXPECTED_OPERATION_IDS)) if proof_run else [])
        and payload.get("physical_verified") is True
        and payload.get("audio_recheck_passed") is True
        and payload.get("proof_target_verified") is proof_run
        and payload.get("physical_state_restored") is proof_run
        and payload.get("terminal_after_console_send") is True
        and payload.get("rollback_attempted") is proof_run
        and payload.get("rollback_succeeded") is proof_run
        and payload.get("write_state_unknown") is False
        and payload.get("rollback_state_unknown") is False
        and payload.get("manual_review_required") is False
        and payload.get("emergency_stop_activated") is False
        and payload.get("emergency_stop_activation_failed") is False
        and payload.get("emergency_stop_required") is False
        and payload.get("accepted") is False
        and payload.get("owner_review_required") is True
        and payload.get("live_authority_granted") is False
        and payload.get("readiness_closure_allowed") is False
        and payload.get("before_manifest") == before_manifest
        and payload.get("before_manifest_sha256")
        == context.get("before_manifest_sha256")
        and payload.get("after_manifest_sha256")
        == canonical_sha256(after_manifest)
        and manifest_matches
        and payload.get("preservation_mismatches") == []
        and isinstance(cut_epoch, (int, float))
        and not isinstance(cut_epoch, bool)
        and math.isfinite(float(cut_epoch))
        and float(cut_epoch) > 0.0
        and payload.get("verification_cut_reset_surface_sha256")
        == expected_cut_sha
        and payload.get("verification_cut_preservation_manifest_sha256")
        == payload.get("after_manifest_sha256")
        and payload.get("physical_truth_scope")
        == "verified_as_of_m110_verification_cut"
        and payload.get("current_state_guaranteed_at_response") is False
        and payload.get("attended_no_touch_required_during_execution") is True
        and payload.get("fresh_readback_required_before_next_stage") is True
        and payload.get("later_console_event_invalidates_readiness_freshness")
        is True
        and payload.get("cancellation_requested") is False
        and _validate_audits(payload)
    ):
        return False
    incident_fields = {
        "post_linearization_incident",
        "linearized_terminal_phase",
        "success_terminal_attempted",
        "success_terminal_durable_confirmed",
        "compensation_attempted",
        "rollback_attempted_after_linearization",
    }
    return not incident_fields.intersection(payload)


def _validate_incident(
    context: Mapping[str, Any],
    payload: Any,
    success_phase: str,
    success_payload: Mapping[str, Any],
) -> bool:
    history_fields = (
        "verification_candidate_status",
        "verification_candidate_reason",
        "verification_candidate_phase",
        "applied_operation_ids",
        "rollback_operation_ids",
        "audio_recheck_passed",
        "physical_state_restored",
        "terminal_after_console_send",
        "rollback_attempted",
        "rollback_succeeded",
        "write_state_unknown",
        "rollback_state_unknown",
        "before_manifest",
        "before_manifest_sha256",
        "after_manifest",
        "after_manifest_sha256",
        "preservation_mismatches",
        "verification_cut_epoch",
        "verification_cut_reset_surface_sha256",
        "verification_cut_preservation_manifest_sha256",
        "physical_truth_scope",
    )
    return bool(
        _validate_manual_terminal(context, payload)
        and all(
            payload.get(field) == success_payload.get(field)
            for field in history_fields
        )
        and payload.get("post_linearization_incident") is True
        and payload.get("linearized_terminal_phase") == success_phase
        and payload.get("linearized_state_preserved_by_executor") is True
        and payload.get("success_terminal_attempted") is True
        and isinstance(
            payload.get("success_terminal_durable_confirmed"), bool
        )
        and payload.get("compensation_attempted") is False
        and payload.get("rollback_attempted_after_linearization") is False
    )


def _validate_abort(context: Mapping[str, Any], payload: Any) -> bool:
    if not _binding_matches(context, payload):
        return False
    reason = str(payload.get("result_reason") or "").strip()
    reset_rows = list(context["payload"]["reset_preimage"]["rows"])
    before_manifest = context["before_manifest"]
    after_manifest = _validate_manifest(reset_rows, payload.get("after_manifest"))
    if after_manifest is None:
        return False
    try:
        manifest_matches = not compare_preservation_manifest_rows(
            list(before_manifest["rows"]),
            list(after_manifest["rows"]),
        )
    except (TypeError, ValueError):
        return False
    incident_fields = {
        "post_linearization_incident",
        "linearized_terminal_phase",
        "success_terminal_attempted",
        "success_terminal_durable_confirmed",
        "compensation_attempted",
        "rollback_attempted_after_linearization",
    }
    return bool(
        reason
        and payload.get("result_status") == "blocked"
        and payload.get("verification_candidate_status") == "blocked"
        and payload.get("verification_candidate_reason") == reason
        and payload.get("verification_candidate_phase")
        == "aborted_before_send"
        and payload.get("terminal_after_console_send") is False
        and payload.get("applied_operation_ids") == []
        and payload.get("rollback_operation_ids") == []
        and payload.get("physical_verified") is False
        and isinstance(payload.get("audio_recheck_passed"), bool)
        and payload.get("proof_target_verified") is False
        and payload.get("physical_state_restored") is False
        and payload.get("rollback_attempted") is False
        and payload.get("rollback_succeeded") is False
        and payload.get("write_state_unknown") is False
        and payload.get("rollback_state_unknown") is False
        and payload.get("manual_review_required") is False
        and payload.get("emergency_stop_activated") is False
        and payload.get("emergency_stop_activation_failed") is False
        and payload.get("emergency_stop_required") is False
        and payload.get("accepted") is False
        and payload.get("owner_review_required") is True
        and payload.get("live_authority_granted") is False
        and payload.get("readiness_closure_allowed") is False
        and payload.get("before_manifest") == before_manifest
        and payload.get("before_manifest_sha256")
        == context.get("before_manifest_sha256")
        and payload.get("after_manifest_sha256")
        == canonical_sha256(after_manifest)
        and manifest_matches
        and payload.get("preservation_mismatches") == []
        and payload.get("physical_truth_scope")
        == "not_verified_at_terminal_cut"
        and payload.get("verification_cut_epoch") is None
        and payload.get("verification_cut_reset_surface_sha256") == ""
        and payload.get("verification_cut_preservation_manifest_sha256") == ""
        and payload.get("current_state_guaranteed_at_response") is False
        and payload.get("attended_no_touch_required_during_execution") is True
        and payload.get("fresh_readback_required_before_next_stage") is True
        and payload.get("later_console_event_invalidates_readiness_freshness")
        is True
        and isinstance(payload.get("cancellation_requested"), bool)
        and _validate_no_send_abort_audits(payload)
        and not incident_fields.intersection(payload)
    )


def _operation_prefixes_valid(payload: Mapping[str, Any]) -> bool:
    applied = payload.get("applied_operation_ids")
    rolled_back = payload.get("rollback_operation_ids")
    if not isinstance(applied, list) or not isinstance(rolled_back, list):
        return False
    applied_count = len(applied)
    rollback_count = len(rolled_back)
    return bool(
        applied == list(_EXPECTED_OPERATION_IDS[:applied_count])
        and rolled_back
        == list(reversed(_EXPECTED_OPERATION_IDS[:rollback_count]))
        and applied_count <= rollback_count <= min(64, applied_count + 1)
        and rollback_count > 0
    )


def _validate_rollback_common(
    context: Mapping[str, Any],
    payload: Any,
    *,
    observer_active: bool,
    terminal_cut: bool,
    success_candidate: bool = False,
) -> bool:
    if not _binding_matches(context, payload):
        return False
    reason = str(payload.get("verification_candidate_reason") or "").strip()
    reset_rows = list(context["payload"]["reset_preimage"]["rows"])
    before_manifest = context["before_manifest"]
    after_manifest = _validate_manifest(reset_rows, payload.get("after_manifest"))
    if after_manifest is None:
        return False
    try:
        manifest_matches = not compare_preservation_manifest_rows(
            list(before_manifest["rows"]),
            list(after_manifest["rows"]),
        )
    except (TypeError, ValueError):
        return False
    observer = payload.get("delivered_observer_audit")
    gate = payload.get("mutation_audit")
    audits_safe = bool(
        isinstance(observer, Mapping)
        and isinstance(gate, Mapping)
        and payload.get("delivered_observer_audit_sha256")
        == canonical_sha256(observer)
        and payload.get("mutation_audit_sha256") == canonical_sha256(gate)
        and payload.get("delivered_observer_clean") is True
        and observer.get("active") is observer_active
        and observer.get("unsubscribe_failed") is False
        and observer.get("overflow") is False
        and not observer.get("unexpected_events")
        and gate.get("active") is True
        and gate.get("sealed") is True
        and gate.get("remove_failed") is False
        and gate.get("boundary_state_unknown") is False
        and gate.get("overflow") is False
        and isinstance(gate.get("denied_attempts"), list)
        and isinstance(
            gate.get("no_outside_mutation_attempt_proven"),
            bool,
        )
        and payload.get("no_outside_mutation_attempt_proven")
        is gate.get("no_outside_mutation_attempt_proven")
        and gate.get("no_outside_mutation_attempt_proven")
        is (not bool(gate.get("denied_attempts")))
    )
    expected_cut_sha = canonical_sha256(
        [
            {"address": address, "value": context["preimage"][address]}
            for address in M110_RESET_ADDRESSES
        ]
    )
    cut_epoch = payload.get("verification_cut_epoch")
    cut_valid = bool(
        isinstance(cut_epoch, (int, float))
        and not isinstance(cut_epoch, bool)
        and math.isfinite(float(cut_epoch))
        and float(cut_epoch) > 0.0
        and payload.get("verification_cut_reset_surface_sha256")
        == expected_cut_sha
        and payload.get("verification_cut_preservation_manifest_sha256")
        == payload.get("after_manifest_sha256")
        and payload.get("physical_truth_scope")
        == "verified_as_of_m110_verification_cut"
    )
    no_cut = bool(
        cut_epoch is None
        and payload.get("verification_cut_reset_surface_sha256") == ""
        and payload.get("verification_cut_preservation_manifest_sha256") == ""
        and payload.get("physical_truth_scope")
        == "not_verified_at_terminal_cut"
    )
    if success_candidate:
        proof_run = context.get("proof_run") is True
        expected_candidate_status = (
            "proof_passed" if proof_run else "applied"
        )
        expected_candidate_reason = (
            "m110_reset_proof_applied_and_restored"
            if proof_run
            else "m110_maintenance_dsp_reset_committed"
        )
        expected_candidate_phase = (
            "proof_restored" if proof_run else "committed"
        )
        operation_ids_valid = bool(
            payload.get("applied_operation_ids")
            == list(_EXPECTED_OPERATION_IDS)
            and payload.get("rollback_operation_ids")
            == list(reversed(_EXPECTED_OPERATION_IDS))
        )
    else:
        expected_candidate_status = "rolled_back"
        expected_candidate_reason = reason
        expected_candidate_phase = "rolled_back"
        operation_ids_valid = _operation_prefixes_valid(payload)
    return bool(
        reason
        and payload.get("verification_candidate_status")
        == expected_candidate_status
        and payload.get("verification_candidate_reason")
        == expected_candidate_reason
        and payload.get("verification_candidate_phase")
        == expected_candidate_phase
        and operation_ids_valid
        and (
            isinstance(payload.get("physical_verified"), bool)
            if terminal_cut
            else payload.get("physical_verified") is False
        )
        and isinstance(payload.get("audio_recheck_passed"), bool)
        and payload.get("proof_target_verified") is False
        and payload.get("physical_state_restored") is True
        and payload.get("terminal_after_console_send") is True
        and payload.get("rollback_attempted") is True
        and payload.get("rollback_succeeded") is True
        and payload.get("write_state_unknown") is False
        and payload.get("rollback_state_unknown") is False
        and payload.get("manual_review_required") is False
        and payload.get("emergency_stop_activated") is False
        and payload.get("emergency_stop_activation_failed") is False
        and payload.get("emergency_stop_required") is False
        and payload.get("accepted") is False
        and payload.get("owner_review_required") is True
        and payload.get("live_authority_granted") is False
        and payload.get("readiness_closure_allowed") is False
        and payload.get("before_manifest") == before_manifest
        and payload.get("before_manifest_sha256")
        == context.get("before_manifest_sha256")
        and payload.get("after_manifest_sha256")
        == canonical_sha256(after_manifest)
        and manifest_matches
        and payload.get("preservation_mismatches") == []
        and payload.get("current_state_guaranteed_at_response") is False
        and payload.get("attended_no_touch_required_during_execution") is True
        and payload.get("fresh_readback_required_before_next_stage") is True
        and payload.get("later_console_event_invalidates_readiness_freshness")
        is True
        and isinstance(payload.get("cancellation_requested"), bool)
        and audits_safe
        and (cut_valid if terminal_cut else no_cut)
    )


def _validate_rollback_candidate(
    context: Mapping[str, Any],
    payload: Any,
) -> bool:
    return bool(
        _validate_rollback_common(
            context,
            payload,
            observer_active=True,
            terminal_cut=False,
        )
        and payload.get("result_status")
        == "verified_state_pending_cleanup"
        and payload.get("result_reason")
        == "m110_verified_state_waiting_for_guard_cleanup"
    )


def _validate_rollback_terminal(
    context: Mapping[str, Any],
    candidate: Any,
    payload: Any,
) -> bool:
    return bool(
        _validate_rollback_common(
            context,
            payload,
            observer_active=False,
            terminal_cut=True,
        )
        and payload.get("result_status") == "rolled_back"
        and payload.get("result_reason")
        == payload.get("verification_candidate_reason")
        and all(
            payload.get(field) == candidate.get(field)
            for field in (
                "verification_candidate_status",
                "verification_candidate_reason",
                "verification_candidate_phase",
                "applied_operation_ids",
                "rollback_operation_ids",
                "audio_recheck_passed",
                "no_outside_mutation_attempt_proven",
                "delivered_observer_clean",
                "write_state_unknown",
                "rollback_state_unknown",
                "physical_state_restored",
            )
        )
    )


def _validate_post_verification_compensation_candidate(
    context: Mapping[str, Any],
    payload: Any,
) -> bool:
    return bool(
        _validate_rollback_common(
            context,
            payload,
            observer_active=True,
            terminal_cut=False,
            success_candidate=True,
        )
        and payload.get("result_status")
        == "verified_state_pending_cleanup"
        and payload.get("result_reason")
        == "m110_verified_state_waiting_for_guard_cleanup"
        and payload.get("cancellation_requested") is False
    )


def _validate_post_verification_compensation_terminal(
    context: Mapping[str, Any],
    candidate: Any,
    payload: Any,
) -> bool:
    return bool(
        _validate_rollback_common(
            context,
            payload,
            observer_active=False,
            terminal_cut=True,
            success_candidate=True,
        )
        and payload.get("result_status") == "rolled_back"
        and payload.get("result_reason")
        == "m110_write_authority_expired_before_terminal"
        and payload.get("cancellation_requested") is False
        and all(
            payload.get(field) == candidate.get(field)
            for field in (
                "verification_candidate_status",
                "verification_candidate_reason",
                "verification_candidate_phase",
                "applied_operation_ids",
                "rollback_operation_ids",
                "audio_recheck_passed",
                "no_outside_mutation_attempt_proven",
                "delivered_observer_clean",
                "write_state_unknown",
                "rollback_state_unknown",
                "physical_state_restored",
            )
        )
    )


def _validate_manual_terminal(
    context: Mapping[str, Any],
    payload: Any,
) -> bool:
    if not _binding_matches(context, payload):
        return False
    reason = str(payload.get("result_reason") or "").strip()
    candidate_status = str(
        payload.get("verification_candidate_status") or ""
    ).strip()
    candidate_reason = str(
        payload.get("verification_candidate_reason") or ""
    ).strip()
    candidate_phase = str(
        payload.get("verification_candidate_phase") or ""
    ).strip()
    proof_run = context.get("proof_run") is True
    candidate_truth_valid = bool(
        reason
        and candidate_status
        and candidate_reason
        and (
            (
                candidate_phase == "aborted_before_send"
                and candidate_status == "blocked"
            )
            or (
                candidate_phase == "committed"
                and not proof_run
                and candidate_status == "applied"
                and candidate_reason
                == "m110_maintenance_dsp_reset_committed"
            )
            or (
                candidate_phase == "proof_restored"
                and proof_run
                and candidate_status == "proof_passed"
                and candidate_reason
                == "m110_reset_proof_applied_and_restored"
            )
            or (
                candidate_phase == "rolled_back"
                and candidate_status == "rolled_back"
            )
            or (
                candidate_phase == "manual_review_required"
                and candidate_status == "manual_review_required"
            )
        )
    )
    if not candidate_truth_valid:
        return False
    reset_rows = list(context["payload"]["reset_preimage"]["rows"])
    before_manifest = context["before_manifest"]
    after_value = payload.get("after_manifest")
    manifest_valid = False
    if isinstance(after_value, Mapping):
        after_manifest = _validate_manifest(reset_rows, after_value)
        if after_manifest is None:
            return False
        try:
            expected_mismatches = list(
                compare_preservation_manifest_rows(
                    list(before_manifest["rows"]),
                    list(after_manifest["rows"]),
                )
            )
        except (TypeError, ValueError):
            return False
        manifest_valid = bool(
            payload.get("after_manifest_sha256")
            == canonical_sha256(after_manifest)
            and payload.get("preservation_mismatches")
            == expected_mismatches
        )
    else:
        manifest_valid = bool(
            after_value is None
            and payload.get("after_manifest_sha256") == ""
            and isinstance(payload.get("preservation_mismatches"), list)
            and bool(payload.get("preservation_mismatches"))
        )
    observer = payload.get("delivered_observer_audit")
    gate = payload.get("mutation_audit")
    if not (isinstance(observer, Mapping) and isinstance(gate, Mapping)):
        return False
    observer_clean = bool(
        observer.get("unsubscribe_failed") is False
        and observer.get("overflow") is False
        and isinstance(observer.get("unexpected_events"), list)
        and not observer.get("unexpected_events")
    )
    audits_valid = bool(
        isinstance(observer.get("active"), bool)
        and isinstance(gate.get("active"), bool)
        and gate.get("active") is True
        and gate.get("sealed") is True
        and isinstance(gate.get("remove_failed"), bool)
        and isinstance(gate.get("boundary_state_unknown"), bool)
        and isinstance(gate.get("overflow"), bool)
        and isinstance(gate.get("denied_attempts"), list)
        and isinstance(
            gate.get("no_outside_mutation_attempt_proven"),
            bool,
        )
        and gate.get("no_outside_mutation_attempt_proven")
        is (not bool(gate.get("denied_attempts")))
        and payload.get("delivered_observer_audit_sha256")
        == canonical_sha256(observer)
        and payload.get("mutation_audit_sha256") == canonical_sha256(gate)
        and payload.get("delivered_observer_clean") is observer_clean
        and payload.get("no_outside_mutation_attempt_proven") is False
    )
    applied = payload.get("applied_operation_ids")
    rolled_back = payload.get("rollback_operation_ids")
    operation_shape_valid = bool(
        isinstance(applied, list)
        and isinstance(rolled_back, list)
        and applied == list(_EXPECTED_OPERATION_IDS[: len(applied)])
        and rolled_back
        == list(reversed(_EXPECTED_OPERATION_IDS[: len(rolled_back)]))
        and len(applied) <= 64
        and len(rolled_back) <= min(64, len(applied) + 1)
    )
    terminal_after_send = payload.get("terminal_after_console_send")
    rollback_attempted = payload.get("rollback_attempted")
    rollback_succeeded = payload.get("rollback_succeeded")
    physical_state_restored = payload.get("physical_state_restored")
    rollback_truth_valid = bool(
        isinstance(terminal_after_send, bool)
        and isinstance(rollback_attempted, bool)
        and isinstance(rollback_succeeded, bool)
        and isinstance(physical_state_restored, bool)
        and (terminal_after_send or (not applied and not rolled_back))
        and (
            bool(rolled_back)
            if rollback_attempted
            else not rolled_back
        )
        and (rollback_attempted or rollback_succeeded is False)
        and (
            not rollback_succeeded
            or (rollback_attempted and physical_state_restored)
        )
    )
    candidate_operation_truth_valid = bool(
        (
            candidate_status == "blocked"
            and applied == []
            and rolled_back == []
            and terminal_after_send is False
            and rollback_attempted is False
            and rollback_succeeded is False
        )
        or (
            candidate_status == "applied"
            and applied == list(_EXPECTED_OPERATION_IDS)
            and rolled_back
            in ([], list(reversed(_EXPECTED_OPERATION_IDS)))
        )
        or (
            candidate_status == "proof_passed"
            and applied == list(_EXPECTED_OPERATION_IDS)
            and rolled_back == list(reversed(_EXPECTED_OPERATION_IDS))
            and rollback_attempted is True
            and rollback_succeeded is True
            and physical_state_restored is True
        )
        or (
            candidate_status == "rolled_back"
            and _operation_prefixes_valid(payload)
        )
        or candidate_status == "manual_review_required"
    )
    cut_epoch = payload.get("verification_cut_epoch")
    if cut_epoch is None:
        cut_valid = bool(
            payload.get("verification_cut_reset_surface_sha256") == ""
            and payload.get("verification_cut_preservation_manifest_sha256")
            == ""
            and payload.get("physical_truth_scope")
            == "not_verified_at_terminal_cut"
        )
    else:
        if physical_state_restored:
            expected_cut_values = context["preimage"]
        elif (
            candidate_phase == "committed"
            and candidate_status == "applied"
        ):
            expected_cut_values = {
                address: 0 for address in M110_RESET_ADDRESSES
            }
        else:
            expected_cut_values = None
        expected_cut_sha = (
            canonical_sha256(
                [
                    {
                        "address": address,
                        "value": expected_cut_values[address],
                    }
                    for address in M110_RESET_ADDRESSES
                ]
            )
            if expected_cut_values is not None
            else ""
        )
        cut_valid = bool(
            isinstance(cut_epoch, (int, float))
            and not isinstance(cut_epoch, bool)
            and math.isfinite(float(cut_epoch))
            and float(cut_epoch) > 0.0
            and payload.get("verification_cut_reset_surface_sha256")
            == expected_cut_sha
            and payload.get(
                "verification_cut_preservation_manifest_sha256"
            )
            == payload.get("after_manifest_sha256")
            and payload.get("physical_truth_scope")
            == "verified_as_of_m110_verification_cut"
        )
    incident_fields = {
        "post_linearization_incident",
        "linearized_terminal_phase",
        "linearized_state_preserved_by_executor",
        "success_terminal_attempted",
        "success_terminal_durable_confirmed",
        "compensation_attempted",
        "rollback_attempted_after_linearization",
    }
    present_incident_fields = incident_fields.intersection(payload)
    if present_incident_fields:
        incident_valid = bool(
            present_incident_fields == incident_fields
            and isinstance(payload.get("post_linearization_incident"), bool)
            and isinstance(payload.get("linearized_terminal_phase"), str)
            and payload.get("linearized_state_preserved_by_executor") is True
            and isinstance(payload.get("success_terminal_attempted"), bool)
            and isinstance(
                payload.get("success_terminal_durable_confirmed"),
                bool,
            )
            and payload.get("post_linearization_incident")
            is payload.get("success_terminal_attempted")
            and payload.get("compensation_attempted") is False
            and payload.get("rollback_attempted_after_linearization") is False
            and (
                payload.get("linearized_terminal_phase")
                in _SUCCESS_PHASES
                if payload.get("success_terminal_attempted") is True
                else payload.get("linearized_terminal_phase") == ""
            )
        )
    else:
        incident_valid = True
    emergency_activated = payload.get("emergency_stop_activated")
    emergency_failed = payload.get("emergency_stop_activation_failed")
    return bool(
        payload.get("result_status") == "manual_review_required"
        and payload.get("manual_review_required") is True
        and payload.get("physical_verified") is False
        and isinstance(payload.get("audio_recheck_passed"), bool)
        and payload.get("proof_target_verified") is False
        and isinstance(payload.get("write_state_unknown"), bool)
        and isinstance(payload.get("rollback_state_unknown"), bool)
        and payload.get("emergency_stop_required") is True
        and isinstance(emergency_activated, bool)
        and isinstance(emergency_failed, bool)
        and emergency_activated is not emergency_failed
        and payload.get("accepted") is False
        and payload.get("owner_review_required") is True
        and payload.get("live_authority_granted") is False
        and payload.get("readiness_closure_allowed") is False
        and payload.get("before_manifest") == before_manifest
        and payload.get("before_manifest_sha256")
        == context.get("before_manifest_sha256")
        and payload.get("current_state_guaranteed_at_response") is False
        and payload.get("attended_no_touch_required_during_execution") is True
        and payload.get("fresh_readback_required_before_next_stage") is True
        and payload.get("later_console_event_invalidates_readiness_freshness")
        is True
        and isinstance(payload.get("cancellation_requested"), bool)
        and manifest_valid
        and audits_valid
        and operation_shape_valid
        and rollback_truth_valid
        and candidate_operation_truth_valid
        and cut_valid
        and incident_valid
    )


def _validate_pre_success_manual_payload(
    context: Mapping[str, Any],
    payload: Any,
) -> bool:
    incident_fields = {
        "post_linearization_incident",
        "linearized_terminal_phase",
        "linearized_state_preserved_by_executor",
        "success_terminal_attempted",
        "success_terminal_durable_confirmed",
        "compensation_attempted",
        "rollback_attempted_after_linearization",
    }
    if not isinstance(payload, Mapping):
        return False
    present_incident_fields = incident_fields.intersection(payload)
    explicit_pre_success_truth = bool(
        present_incident_fields == incident_fields
        and payload.get("post_linearization_incident") is False
        and payload.get("linearized_terminal_phase") == ""
        and payload.get("linearized_state_preserved_by_executor") is True
        and payload.get("success_terminal_attempted") is False
        and payload.get("success_terminal_durable_confirmed") is False
        and payload.get("compensation_attempted") is False
        and payload.get("rollback_attempted_after_linearization") is False
    )
    return bool(
        _validate_manual_terminal(context, payload)
        and (not present_incident_fields or explicit_pre_success_truth)
    )


def _validate_manual_after_terminal(
    context: Mapping[str, Any],
    terminal_payload: Mapping[str, Any],
    payload: Any,
) -> bool:
    history_fields = (
        "verification_candidate_status",
        "verification_candidate_reason",
        "verification_candidate_phase",
        "applied_operation_ids",
        "rollback_operation_ids",
        "audio_recheck_passed",
        "proof_target_verified",
        "physical_state_restored",
        "terminal_after_console_send",
        "rollback_attempted",
        "rollback_succeeded",
        "write_state_unknown",
        "rollback_state_unknown",
        "before_manifest",
        "before_manifest_sha256",
        "after_manifest",
        "after_manifest_sha256",
        "preservation_mismatches",
        "verification_cut_epoch",
        "verification_cut_reset_surface_sha256",
        "verification_cut_preservation_manifest_sha256",
        "physical_truth_scope",
    )
    return bool(
        _validate_pre_success_manual_payload(context, payload)
        and all(
            payload.get(field) == terminal_payload.get(field)
            for field in history_fields
        )
    )


def _validate_manual_after_pending(
    context: Mapping[str, Any],
    pending_payload: Mapping[str, Any],
    payload: Any,
) -> bool:
    return bool(
        _validate_pre_success_manual_payload(context, payload)
        and all(
            payload.get(field) == pending_payload.get(field)
            for field in (
                "verification_candidate_status",
                "verification_candidate_reason",
                "verification_candidate_phase",
                "applied_operation_ids",
                "before_manifest",
                "before_manifest_sha256",
            )
        )
    )


def classify_m110_lifecycle(lifecycle: Any) -> str | None:
    """Return resolved/unresolved/invalid, or None for a legacy lifecycle."""

    if not lifecycle_is_m110(lifecycle):
        return None
    events = _lifecycle_events(lifecycle)
    if events is None:
        return M110_LIFECYCLE_INVALID
    payloads = tuple(_event_payload(event) for event in events)
    if any(payload is None for payload in payloads):
        return M110_LIFECYCLE_INVALID
    phases = tuple(str(event.get("phase") or "") for event in events)
    if phases[0] != "prepared":
        return M110_LIFECYCLE_INVALID
    context = _validate_prepared(payloads[0])
    if context is None:
        return M110_LIFECYCLE_INVALID
    if phases == ("prepared", "aborted_before_send"):
        return (
            M110_LIFECYCLE_RESOLVED
            if _validate_abort(context, payloads[1])
            else M110_LIFECYCLE_INVALID
        )
    expected_success = (
        "proof_restored" if context["proof_run"] else "committed"
    )
    exact_success = (
        "prepared",
        *_PENDING_PHASES,
        expected_success,
    )
    if len(phases) < len(exact_success) and phases == exact_success[: len(phases)]:
        for index, phase in enumerate(phases[1:], start=1):
            pending_valid = _validate_pending(
                context,
                payloads[index],
                phase,
            )
            if (
                not pending_valid
                and index == 2
                and phase == "verified_state_pending_cleanup"
            ):
                pending_valid = (
                    _validate_post_verification_compensation_candidate(
                        context,
                        payloads[index],
                    )
                )
            if not pending_valid:
                return M110_LIFECYCLE_INVALID
        return M110_LIFECYCLE_UNRESOLVED
    if phases[:4] == exact_success:
        if not (
            _validate_pending(context, payloads[1], phases[1])
            and _validate_pending(context, payloads[2], phases[2])
            and _validate_success(context, payloads[3], phases[3])
        ):
            return M110_LIFECYCLE_INVALID
        if len(phases) == 4:
            return M110_LIFECYCLE_RESOLVED
        if all(phase == "manual_review_required" for phase in phases[4:]) and all(
            _validate_incident(
                context,
                payload,
                expected_success,
                payloads[3],
            )
            for payload in payloads[4:]
        ):
            return M110_LIFECYCLE_UNRESOLVED
        return M110_LIFECYCLE_INVALID
    if _SUCCESS_PHASES.intersection(phases):
        return M110_LIFECYCLE_INVALID
    if "prepared" in phases[1:] or any(
        not _binding_matches(context, payload)
        for payload in payloads[1:]
    ):
        return M110_LIFECYCLE_INVALID

    if phases == (
        "prepared",
        "verified_state_pending_cleanup",
    ):
        return (
            M110_LIFECYCLE_UNRESOLVED
            if _validate_rollback_candidate(context, payloads[1])
            else M110_LIFECYCLE_INVALID
        )
    ordinary_rollback = (
        "prepared",
        "verified_state_pending_cleanup",
        "rolled_back",
    )
    post_verification_rollback = (
        "prepared",
        "verification_pending",
        "verified_state_pending_cleanup",
        "rolled_back",
    )
    if phases == ordinary_rollback:
        if _validate_rollback_candidate(
            context,
            payloads[1],
        ) and _validate_rollback_terminal(
            context,
            payloads[1],
            payloads[2],
        ):
            return M110_LIFECYCLE_UNRESOLVED
        return M110_LIFECYCLE_INVALID
    if phases == post_verification_rollback:
        if (
            _validate_pending(
                context,
                payloads[1],
                "verification_pending",
            )
            and _validate_post_verification_compensation_candidate(
                context,
                payloads[2],
            )
            and _validate_post_verification_compensation_terminal(
                context,
                payloads[2],
                payloads[3],
            )
        ):
            return M110_LIFECYCLE_UNRESOLVED
        return M110_LIFECYCLE_INVALID

    if "manual_review_required" in phases:
        manual_index = phases.index("manual_review_required")
        if not all(
            phase == "manual_review_required"
            for phase in phases[manual_index:]
        ):
            return M110_LIFECYCLE_INVALID
        prefix = phases[:manual_index]
        prefix_valid = False
        if prefix == ("prepared",):
            prefix_valid = True
        elif prefix == ("prepared", "verification_pending"):
            prefix_valid = _validate_pending(
                context,
                payloads[1],
                "verification_pending",
            )
        elif prefix == (
            "prepared",
            "verification_pending",
            "verified_state_pending_cleanup",
        ):
            prefix_valid = bool(
                _validate_pending(
                    context,
                    payloads[1],
                    "verification_pending",
                )
                and (
                    _validate_pending(
                        context,
                        payloads[2],
                        "verified_state_pending_cleanup",
                    )
                    or _validate_rollback_candidate(
                        context,
                        payloads[2],
                    )
                )
            )
        elif prefix == (
            "prepared",
            "verified_state_pending_cleanup",
        ):
            prefix_valid = _validate_rollback_candidate(
                context,
                payloads[1],
            )
        elif prefix == ("prepared", "aborted_before_send"):
            prefix_valid = _validate_abort(context, payloads[1])
        elif prefix == ordinary_rollback:
            prefix_valid = bool(
                _validate_rollback_candidate(context, payloads[1])
                and _validate_rollback_terminal(
                    context,
                    payloads[1],
                    payloads[2],
                )
            )
        elif prefix == post_verification_rollback:
            prefix_valid = bool(
                _validate_pending(
                    context,
                    payloads[1],
                    "verification_pending",
                )
                and _validate_post_verification_compensation_candidate(
                    context,
                    payloads[2],
                )
                and _validate_post_verification_compensation_terminal(
                    context,
                    payloads[2],
                    payloads[3],
                )
            )
        completed_terminal_prefix = prefix in {
            ("prepared", "aborted_before_send"),
            ordinary_rollback,
            post_verification_rollback,
        }
        manual_payloads_valid = all(
            (
                _validate_manual_after_terminal(
                    context,
                    payloads[manual_index - 1],
                    payload,
                )
                if completed_terminal_prefix
                else (
                    _validate_manual_after_pending(
                        context,
                        payloads[manual_index - 1],
                        payload,
                    )
                    if manual_index > 1
                    else _validate_pre_success_manual_payload(
                        context,
                        payload,
                    )
                )
            )
            for payload in payloads[manual_index:]
        )
        if prefix_valid and manual_payloads_valid:
            return M110_LIFECYCLE_UNRESOLVED
        return M110_LIFECYCLE_INVALID
    return M110_LIFECYCLE_INVALID
