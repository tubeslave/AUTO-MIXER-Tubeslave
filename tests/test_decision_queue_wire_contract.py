from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest

from decision_queue_wire_contract import (
    DecisionQueueSnapshotManager,
    PIPELINE_MIGRATION_REQUIRED_HANDLER_NAMES,
    OperatorProposalSequenceStateManager,
    REVIEW_CONTRACT_VERSION,
    REVIEW_SCHEMA,
    WIRE_MAX_BYTES,
    OutboundMessageProjectionError,
    ReviewContentError,
    _freeze_trusted_apply_receipt,
    _freeze_trusted_prewrite_no_send_receipt,
    _freeze_trusted_sequence_execution_receipt,
    _freeze_verified_apply_identity,
    build_review_identity,
    build_proposal_detail_response,
    build_proposal_result_detail_response,
    build_v1_decision_queue_shim,
    canonical_json_bytes,
    collect_compact_rows,
    collect_global_hazards,
    compact_mixing_agent_row,
    compact_proposal_reference,
    compact_proposal_row,
    compact_trusted_apply_receipt_reference,
    finalize_wire_size,
    project_outbound_message,
    serialize_bounded_outbound,
    utf8_preview,
    wire_serialized_bytes,
)
from operator_proposal_queue import OperatorProposalQueue
from operator_proposal_result_store import ResultEvidenceStore


@dataclass
class FakeProposal:
    id: str = "proposal-unicode"
    title: str = "Вокал 🎙️"
    target: object = 1
    kind: str = "fader"
    severity: str = "high"
    confidence: float = 0.9
    status: str = "pending"
    source: str = "unit"
    reason: str = "Убрать маскировку"
    created_at: str = "2026-07-13T08:00:00Z"
    updated_at: str = "2026-07-13T08:00:00Z"
    expires_at: str = "2026-07-13T08:02:00Z"
    ttl_seconds: int = 120
    requested_change: dict = field(
        default_factory=lambda: {"value_type": "fader", "channel": 1, "value": -9.0}
    )
    requested_changes: list = field(default_factory=list)
    mode_at_creation: str = "supervised"
    raw: dict = field(default_factory=lambda: {"operator_authored": True})
    result: object = None
    review_contract_version: int = REVIEW_CONTRACT_VERSION
    review_content_digest: str = "sha256:" + "0" * 64
    review_content_bytes: bytes = b"{}"

    def apply_state(self):
        return {"can_apply": True, "expired": False, "apply_blockers": []}

    def to_dict(self):  # pragma: no cover - must never be called by compact row
        raise AssertionError("compact wire projection called full to_dict")


def test_review_identity_is_deterministic_unicode_json_and_covers_raw_plan():
    proposal = FakeProposal(
        raw={
            "supervised_transaction_plan": {
                "transaction_id": "tx-1",
                "operations": [{"write_kind": "fader", "target_value": -9.0}],
            }
        }
    )

    first_bytes, first_digest = build_review_identity(proposal)
    second_bytes, second_digest = build_review_identity(proposal)

    assert first_bytes == second_bytes
    assert first_digest == second_digest
    parsed = json.loads(first_bytes)
    assert parsed["schema"] == REVIEW_SCHEMA
    assert parsed["title"] == "Вокал 🎙️"
    assert parsed["raw"]["supervised_transaction_plan"]["transaction_id"] == "tx-1"

    proposal.raw["supervised_transaction_plan"]["operations"][0]["target_value"] = -3.0
    assert build_review_identity(proposal)[1] != first_digest


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object(), {1: "bad-key"}])
def test_review_canonicalization_fails_closed_for_non_json(value):
    with pytest.raises(ReviewContentError):
        canonical_json_bytes({"value": value})


def test_utf8_preview_never_splits_a_codepoint():
    assert utf8_preview("abc🎙️xyz", 7) == "abc🎙"
    assert len(utf8_preview("🎙️" * 300, 512).encode("utf-8")) <= 512


def test_universal_outbound_projector_preserves_small_native_json_and_numpy():
    payload = {
        "type": "state_update",
        "title": "Вокал 🎙️",
        "scalar": np.float32(-3.5),
        "array": np.array([[1, 2], [3, 4]], dtype=np.int16),
        "tuple": (True, None),
    }

    serialized = serialize_bounded_outbound(payload)

    assert len(serialized.encode("utf-8")) <= WIRE_MAX_BYTES
    assert json.loads(serialized) == {
        "type": "state_update",
        "title": "Вокал 🎙️",
        "scalar": -3.5,
        "array": [[1, 2], [3, 4]],
        "tuple": [True, None],
    }
    assert project_outbound_message(payload) == serialized


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "bad", "value": float("nan")},
        {"type": "bad", "value": object()},
        {"type": "bad", "value": "x" * 1_000_000},
        {"type": "bad", "value": list(range(9_000))},
        {"type": "bad", "value": 1 << 5_000},
    ],
)
def test_universal_outbound_projector_degrades_invalid_or_oversized_payloads(payload):
    serialized = project_outbound_message(payload)
    parsed = json.loads(serialized)

    assert len(serialized.encode("utf-8")) <= WIRE_MAX_BYTES
    assert parsed["type"] == "outbound_message_degraded"
    assert parsed["outbound_message_degraded"] is True
    assert parsed["original_type"] == "bad"
    assert "x" * 256 not in serialized


def test_universal_outbound_projector_degrades_cycles_depth_and_false_wire_truth():
    cyclic = {"type": "cycle"}
    cyclic["self"] = cyclic
    deep = {"type": "deep"}
    cursor = deep
    for _ in range(40):
        cursor["child"] = {}
        cursor = cursor["child"]

    for payload, reason in (
        (cyclic, "outbound_message_invalid"),
        (deep, "outbound_message_invalid"),
        (
            {
                "type": "decision_queue",
                "wire": {"max_bytes": WIRE_MAX_BYTES, "serialized_bytes": 1},
            },
            "outbound_message_wire_size_mismatch",
        ),
        (
            {"type": "decision_queue", "wire": {"max_bytes": 999}},
            "outbound_message_wire_max_mismatch",
        ),
    ):
        parsed = json.loads(project_outbound_message(payload))
        assert parsed["type"] == "outbound_message_degraded"
        assert parsed["reason"] == reason

    with pytest.raises(OutboundMessageProjectionError):
        serialize_bounded_outbound(cyclic)


def test_trusted_apply_receipt_dominates_forged_non_apply_message_type():
    proposal = FakeProposal(status="pending")
    receipt = _freeze_trusted_prewrite_no_send_receipt(
        proposal,
        reason_code="proposal_review_digest_mismatch",
    )
    serialized = project_outbound_message(
        {
            "type": "state_update",
            "raw": "must-not-cross-apply-receipt-boundary",
        },
        trusted_receipt=receipt,
    )
    parsed = json.loads(serialized)

    assert parsed["type"] == "operator_proposal_apply_blocked"
    assert parsed["no_console_send_proven"] is True
    assert "must-not-cross" not in serialized


def test_apply_receipt_never_projects_free_text_reason():
    receipt = _freeze_trusted_prewrite_no_send_receipt(
        FakeProposal(status="pending"),
        reason_code="/Users/operator/private/token-sk-live-secret",
    )

    serialized = project_outbound_message({}, trusted_receipt=receipt)
    parsed = json.loads(serialized)

    assert parsed["reason"] == "operator_apply_blocked_before_send"
    assert "private" not in serialized
    assert "token-sk" not in serialized


@pytest.mark.parametrize(
    "reason_code",
    [
        "proposal_review_contract_version_required",
        "proposal_review_digest_mismatch",
    ],
)
def test_prewrite_review_binding_failures_preserve_no_mutation_truth(reason_code):
    proposal = FakeProposal(status="pending")
    receipt = _freeze_trusted_prewrite_no_send_receipt(
        proposal,
        reason_code=reason_code,
    )

    parsed = json.loads(project_outbound_message({}, trusted_receipt=receipt))

    assert parsed["reason"] == reason_code
    assert parsed["proposal"]["review_contract_version"] == REVIEW_CONTRACT_VERSION
    assert parsed["proposal"]["review_content_digest"] == proposal.review_content_digest
    assert parsed["no_console_send_proven"] is True
    assert parsed["idempotency_reserved"] is False
    assert parsed["proposal_status_mutation"] is False
    assert parsed["result_evidence_reserved"] is False


def test_prewrite_result_evidence_capacity_exhaustion_preserves_exact_reason():
    receipt = _freeze_trusted_prewrite_no_send_receipt(
        FakeProposal(status="pending"),
        reason_code="operator_proposal_result_evidence_capacity_exhausted",
    )

    parsed = json.loads(project_outbound_message({}, trusted_receipt=receipt))

    assert parsed["reason"] == "operator_proposal_result_evidence_capacity_exhausted"
    assert parsed["no_console_send_proven"] is True
    assert parsed["idempotency_reserved"] is False
    assert parsed["proposal_status_mutation"] is False
    assert parsed["result_evidence_reserved"] is False


def test_apply_shaped_message_without_authentic_receipt_is_fixed_unknown_fallback():
    serialized = project_outbound_message(
        {
            "type": "operator_proposal_apply_result",
            "status": "applied",
            "no_console_send_proven": True,
        }
    )
    parsed = json.loads(serialized)

    assert parsed["type"] == "operator_proposal_apply_result"
    assert parsed["status"] == "apply_error"
    assert parsed["no_console_send_proven"] is None
    assert parsed["write_state_unknown"] is True


def test_known_cp02_create_response_never_emits_full_proposal_or_private_fields():
    marker = "TOPSECRET-CREATE-MARKER"
    payload = {
        "type": "operator_proposal_created",
        "status": "created",
        "success": True,
        "error": "/Users/private/token-sk-create-error",
        "detail": marker,
        "proposal": {
            "id": "created-1",
            "status": "pending",
            "review_contract_version": 2,
            "review_content_digest": "sha256:" + "1" * 64,
            "can_apply": True,
            "raw": {"token": marker, "artifact_path": "/Users/private/result"},
            "result": {"secret": marker},
            "requested_changes": [{"address": "/ch/1/fdr"}],
        },
    }

    serialized = project_outbound_message(payload)
    parsed = json.loads(serialized)

    assert parsed["proposal"]["id"] == "created-1"
    assert parsed["proposal"]["review_contract_version"] == 2
    assert "raw" not in parsed["proposal"]
    assert "result" not in parsed["proposal"]
    assert "requested_changes" not in parsed["proposal"]
    assert marker not in serialized
    assert "/Users/private" not in serialized
    assert parsed["wire_projection"]["omitted_field_count"] >= 0


def test_known_cp02_import_response_never_emits_full_proposal_or_private_fields():
    marker = "TOPSECRET_CP02_IMPORT"
    payload = {
        "type": "operator_proposals_imported",
        "status": "ok",
        "success": True,
        "results": [
            {
                "status": "created",
                "proposal": {
                    "id": "imported-1",
                    "status": "pending",
                    "review_contract_version": 2,
                    "review_content_digest": "sha256:" + "5" * 64,
                    "raw": {
                        "supervised_transaction_plan": marker,
                        "artifact_path": "/Users/private/WING",
                        "token": "SECRET_TOKEN",
                    },
                    "requested_change": {"value": -8.0},
                },
            }
        ],
    }

    serialized = project_outbound_message(payload)
    parsed = json.loads(serialized)

    assert parsed["type"] == "operator_proposals_imported"
    assert parsed["results"][0]["proposal"]["id"] == "imported-1"
    assert "raw" not in parsed["results"][0]["proposal"]
    assert "requested_change" not in parsed["results"][0]["proposal"]
    assert marker not in serialized
    assert "/Users/private/WING" not in serialized
    assert "SECRET_TOKEN" not in serialized


def test_cp02_explicit_schema_rejects_suffix_inference_and_unknown_types():
    marker = "topsecret_suffix_marker"
    imported = json.loads(
        project_outbound_message(
            {
                "type": "operator_proposals_imported",
                "status": "ok",
                "success": True,
                "blocked_reason": marker,
                "failure_reason": marker,
                "original_reason": marker,
                "internal_source": marker,
                "result_status": marker,
                "results": [
                    {
                        "status": "created",
                        "proposal": {
                            "id": "safe-id",
                            "status": "pending",
                            "review_contract_version": 2,
                            "review_content_digest": "sha256:" + "8" * 64,
                        },
                        "blocked_reason": marker,
                        "internal_source": marker,
                    }
                ],
            }
        )
    )
    unknown = json.loads(
        project_outbound_message(
            {
                "type": "operator_proposal_future_unregistered",
                "status": "ok",
                "success": True,
                "reason": marker,
                "payload": {"source": marker},
            }
        )
    )

    assert marker not in json.dumps(imported)
    assert imported["results"][0]["proposal"]["id"] == "safe-id"
    assert set(imported) == {
        "type",
        "contract_version",
        "status",
        "success",
        "reason",
        "no_console_send_proven",
        "results",
        "results_wire_counts",
        "wire_projection",
    }
    assert unknown == {
        "type": "outbound_message_degraded",
        "contract_version": 2,
        "status": "error",
        "success": False,
        "reason": "outbound_schema_not_registered",
        "original_type": "operator_proposal_future_unregistered",
        "outbound_message_degraded": True,
        "wire_projection": {
            "schema": "automixer.cp02.explicit-response.v2",
            "omitted_field_count": 4,
            "omitted_item_count": 0,
            "truncated_array_count": 0,
        },
    }


@pytest.mark.parametrize(
    "blocked_handler",
    sorted(PIPELINE_MIGRATION_REQUIRED_HANDLER_NAMES),
)
def test_cp02_pipeline_migration_guard_has_explicit_compact_schema(blocked_handler):
    payload = {
        "type": "operator_proposal_pipeline_migration_required",
        "contract_version": 2,
        "status": "migration_required",
        "success": False,
        "reason": "server_owned_sequence_state_required",
        "blocked_handler": blocked_handler,
        "no_additional_console_send_proven": True,
        "manual_review_required": True,
        "soundcheck_completion_blocked": True,
        "memory_mutation": False,
        "policy_mutation": False,
        "queue_mutation": False,
        "model_mutation": False,
        "live_authority_granted": False,
        "private_detail": "/Users/operator/private/migration.json",
    }

    parsed = json.loads(project_outbound_message(payload))

    assert parsed["type"] == "operator_proposal_pipeline_migration_required"
    assert parsed["contract_version"] == 2
    assert parsed["status"] == "migration_required"
    assert parsed["success"] is False
    assert parsed["reason"] == "server_owned_sequence_state_required"
    assert parsed["blocked_handler"] == payload["blocked_handler"]
    assert parsed["no_additional_console_send_proven"] is True
    assert parsed["manual_review_required"] is True
    assert parsed["soundcheck_completion_blocked"] is True
    assert parsed["memory_mutation"] is False
    assert parsed["policy_mutation"] is False
    assert parsed["queue_mutation"] is False
    assert parsed["model_mutation"] is False
    assert parsed["live_authority_granted"] is False
    assert "/Users/operator/private" not in json.dumps(parsed)

    legacy_audit = json.loads(
        project_outbound_message(
            {
                "type": "operator_proposal_apply_sequence_execution_audit",
                "status": "blocked",
                "success": False,
            }
        )
    )
    assert legacy_audit["type"] == "outbound_message_degraded"
    assert legacy_audit["reason"] == "outbound_schema_not_registered"


@pytest.mark.parametrize(
    "blocked_handler",
    ["unregistered_client_echo_handler", ["unhashable_client_echo_handler"]],
)
def test_cp02_pipeline_migration_guard_drops_unregistered_handler_name(
    blocked_handler,
):
    parsed = json.loads(
        project_outbound_message(
            {
                "type": "operator_proposal_pipeline_migration_required",
                "contract_version": 2,
                "status": "migration_required",
                "success": False,
                "reason": "server_owned_sequence_state_required",
                "blocked_handler": blocked_handler,
            }
        )
    )

    assert "blocked_handler" not in parsed


def test_cp02_per_type_schema_drops_known_fields_from_wrong_response_family():
    payload = {
        "type": "operator_proposal_apply_sequence_plan",
        "status": "ok",
        "success": True,
        "proposal_id": "must-not-cross-plan-top",
        "source": "safe_gain_calibrator",
        "summary": {
            "planned_apply_count": 0,
            "possible_console_send_count": 999,
            "attempted_count": 999,
            "applied_proposal_ids": ["must-not-cross-summary"],
        },
        "safety": {
            "sequence_plan_only": True,
            "sequence_executor": True,
            "requires_execute_true": True,
            "manual_review_required": False,
        },
        "planned_apply_sequence": [],
        "blocked_results": [],
        "sequence_state": {
            "schema": "automixer.operator_proposal.sequence_state.v2",
            "cursor": "plan-cross-type-0123456789abcdef",
            "stage": "sequence_plan",
            "content_digest": "sha256:" + "9" * 64,
            "created_at_epoch": 100.0,
            "expires_at_epoch": 130.0,
        },
    }

    parsed = json.loads(project_outbound_message(payload))

    assert "proposal_id" not in parsed
    assert "source" not in parsed
    assert parsed["summary"] == {"planned_apply_count": 0}
    assert parsed["safety"] == {"sequence_plan_only": True}
    assert "must-not-cross" not in json.dumps(parsed)


def test_known_cp02_response_rejects_false_input_wire_truth():
    payload = {
        "type": "operator_proposal_created",
        "status": "created",
        "success": True,
        "wire": {"max_bytes": 999, "serialized_bytes": True},
    }

    parsed = json.loads(project_outbound_message(payload))

    assert parsed["type"] == "outbound_message_degraded"
    assert parsed["outbound_message_degraded"] is True


def test_sequence_state_is_owner_digest_stage_and_expiry_bound():
    clock = [100.0]
    manager = OperatorProposalSequenceStateManager(
        ttl_seconds=10.0,
        clock=lambda: clock[0],
    )
    owner = object()
    stored = manager.store(
        owner,
        "batch_preflight",
        {"preflight_results": [{"proposal_id": "p-1", "raw": "private"}]},
    )
    assert stored["stored"] is True
    reference = stored["reference"]

    wrong_owner = manager.resolve(
        object(),
        reference,
        expected_stage="batch_preflight",
    )
    assert wrong_owner["reason"] == "sequence_state_owner_mismatch"
    wrong_digest = manager.resolve(
        owner,
        {**reference, "content_digest": "sha256:" + "0" * 64},
        expected_stage="batch_preflight",
    )
    assert wrong_digest["reason"] == "sequence_state_digest_mismatch"
    wrong_stage = manager.resolve(
        owner,
        reference,
        expected_stage="sequence_plan",
    )
    assert wrong_stage["reason"] == "sequence_state_stage_mismatch"

    resolved = manager.resolve(
        owner,
        reference,
        expected_stage="batch_preflight",
    )
    assert resolved["resolved"] is True
    assert resolved["payload"]["preflight_results"][0]["raw"] == "private"
    replay = manager.resolve(
        owner,
        reference,
        expected_stage="batch_preflight",
    )
    assert replay["reason"] == "sequence_state_not_found"

    expiring = manager.store(owner, "dry_run", {"dry_run_results": []})
    clock[0] = 111.0
    expired = manager.resolve(
        owner,
        expiring["reference"],
        expected_stage="dry_run",
    )
    assert expired["reason"] == "sequence_state_expired"


@pytest.mark.parametrize(
    "message_type,array_key",
    [
        ("operator_proposal_apply_preflight_batch", "preflight_results"),
        ("operator_proposal_apply_sequence_plan", "planned_apply_sequence"),
        ("operator_proposal_apply_sequence_dry_run", "dry_run_results"),
        (
            "operator_proposal_apply_sequence_execution_gate",
            "ready_for_executor",
        ),
    ],
)
def test_known_cp02_plan_responses_hash_authority_tokens_and_omit_full_changes(
    message_type,
    array_key,
):
    idempotency_key = "raw-idempotency-key-must-not-cross-wire"
    approval_id = "raw-approval-id-must-not-cross-wire"
    marker = "TOPSECRET-PLAN-MARKER"
    stage_by_type = {
        "operator_proposal_apply_preflight_batch": "batch_preflight",
        "operator_proposal_apply_sequence_plan": "sequence_plan",
        "operator_proposal_apply_sequence_dry_run": "dry_run",
        "operator_proposal_apply_sequence_execution_gate": "execution_gate",
    }
    sequence_state = {
        "schema": "automixer.operator_proposal.sequence_state.v2",
        "cursor": "safe-cursor-0123456789abcdef",
        "stage": stage_by_type[message_type],
        "content_digest": "sha256:" + "3" * 64,
        "created_at_epoch": 100.0,
        "expires_at_epoch": 130.0,
    }
    payload = {
        "type": message_type,
        "status": "ok",
        "success": True,
        "sequence_state": sequence_state,
        "error": "/Users/private/token-sk-plan-error",
        "note": marker,
        array_key: [
            {
                "proposal_id": "plan-1",
                "status": "ready",
                "idempotency_key": idempotency_key,
                "approval_id": approval_id,
                "apply_request": {
                    "type": "apply_operator_proposal",
                    "proposal_id": "plan-1",
                    "approved": True,
                    "idempotency_key": idempotency_key,
                    "approval_id": approval_id,
                    "expected_review_contract_version": 2,
                    "expected_review_content_digest": "sha256:" + "2" * 64,
                },
                "current_proposal": {
                    "id": "plan-1",
                    "status": "pending",
                    "review_contract_version": 2,
                    "review_content_digest": "sha256:" + "2" * 64,
                    "raw": {"secret": marker},
                },
                "normalized_changes": [
                    {
                        "address": "/Users/private/not-an-osc-address",
                        "artifact_path": "/Users/private/result.json",
                        "secret": marker,
                    }
                ],
                "normalized_change": {
                    "wing_address": "/private/osc/path",
                    "note": marker,
                },
            }
        ],
    }

    serialized = project_outbound_message(payload)
    parsed = json.loads(serialized)
    row = parsed[array_key][0]

    assert idempotency_key not in serialized
    assert approval_id not in serialized
    assert marker not in serialized
    assert "/Users/private" not in serialized
    assert "normalized_changes" not in row
    assert row["idempotency_key_digest"].startswith("sha256:")
    assert row["approval_id_digest"].startswith("sha256:")
    assert row["apply_request"]["idempotency_key_digest"].startswith("sha256:")
    assert row["apply_request"]["approval_id_digest"].startswith("sha256:")
    assert row["current_proposal"]["id"] == "plan-1"
    assert parsed["sequence_state"] == sequence_state
    assert parsed[f"{array_key}_wire_counts"] == {
        "total_count": 1,
        "included_count": 1,
        "omitted_count": 0,
    }


def test_cp02_sequence_projection_preserves_producer_scalars_and_reason_enums():
    preflight = json.loads(
        project_outbound_message(
            {
                "type": "operator_proposal_apply_preflight_batch",
                "status": "blocked",
                "success": False,
                "preflight_results": [
                    {
                        "type": "operator_proposal_apply_preflight",
                        "status": "blocked",
                        "success": False,
                        "proposal_id": "preflight-1",
                        "reason": "missing_idempotency_key",
                    }
                ],
            }
        )
    )
    assert preflight["preflight_results"][0]["type"] == (
        "operator_proposal_apply_preflight"
    )
    assert preflight["preflight_results"][0]["reason"] == (
        "missing_idempotency_key"
    )

    sequence_plan = json.loads(
        project_outbound_message(
            {
                "type": "operator_proposal_apply_sequence_plan",
                "status": "blocked",
                "success": False,
                "summary": {
                    "input_preflight_result_count": 2,
                    "ready_result_count": 1,
                    "blocked_result_count": 1,
                },
                "blocked_results": [
                    {
                        "proposal_id": "plan-1",
                        "preflight_result_index": 0,
                        "reason": "preflight_has_no_normalized_change",
                    }
                ],
            }
        )
    )
    assert sequence_plan["summary"]["blocked_result_count"] == 1
    assert sequence_plan["blocked_results"][0]["reason"] == (
        "preflight_has_no_normalized_change"
    )

    dry_run = json.loads(
        project_outbound_message(
            {
                "type": "operator_proposal_apply_sequence_dry_run",
                "status": "blocked",
                "success": False,
                "safety": {
                    "dry_run_only": True,
                    "sequence_plan_required": True,
                },
                "dry_run_results": [
                    {
                        "proposal_id": "dry-1",
                        "status": "blocked",
                        "reason": "missing_apply_request",
                    }
                ],
            }
        )
    )
    assert dry_run["safety"]["sequence_plan_required"] is True
    assert dry_run["dry_run_results"][0]["reason"] == "missing_apply_request"

    execution_gate = json.loads(
        project_outbound_message(
            {
                "type": "operator_proposal_apply_sequence_execution_gate",
                "status": "blocked",
                "success": False,
                "blocked_results": [
                    {
                        "proposal_id": "gate-1",
                        "status": "blocked",
                        "reason": "approval_id_required",
                    }
                ],
            }
        )
    )
    assert execution_gate["blocked_results"][0]["reason"] == (
        "approval_id_required"
    )

    execution_message = {
        "type": "operator_proposal_apply_sequence_execution_result",
        "status": "blocked",
        "success": False,
        "reason": "sequence_executor_blocked",
        "summary": {"input_candidate_count": 0},
        "execution_results": [],
        "blocked_results": [
            {
                "proposal_id": None,
                "candidate_index": None,
                "status": "blocked",
                "reason": "invalid_ready_for_executor_row",
                "count": 1,
            }
        ],
    }
    execution_receipt = _freeze_trusted_sequence_execution_receipt(
        execution_message,
        child_receipts=[],
        expected_candidate_count=0,
    )
    execution_result = json.loads(
        project_outbound_message(
            execution_message,
            trusted_receipt=execution_receipt,
        )
    )
    assert execution_result["blocked_results"][0]["reason"] == (
        "invalid_ready_for_executor_row"
    )


@pytest.mark.parametrize(
    "reason",
    [
        "proposal_review_content_non_finite_number",
        "proposal_review_content_non_string_key",
    ],
)
def test_cp02_create_preserves_canonical_review_failure_reason(reason):
    parsed = json.loads(
        project_outbound_message(
            {
                "type": "operator_proposal_created",
                "status": "blocked",
                "success": False,
                "reason": reason,
                "proposal_id": "create-invalid-review",
            }
        )
    )

    assert parsed["reason"] == reason


def test_cp02_proposal_and_readiness_preserve_registered_blocker_codes():
    proposal_blockers = [
        "missing_requested_change",
        "missing_channel",
        "soundcheck_physical_preflight_required",
    ]
    created = json.loads(
        project_outbound_message(
            {
                "type": "operator_proposal_created",
                "status": "created",
                "success": True,
                "proposal": {
                    "id": "blocked-proposal",
                    "status": "pending",
                    "review_contract_version": 2,
                    "review_content_digest": "sha256:" + "a" * 64,
                    "can_apply": False,
                    "apply_blockers": proposal_blockers,
                },
            }
        )
    )
    assert created["proposal"]["apply_blocker_codes"] == proposal_blockers

    imported = json.loads(
        project_outbound_message(
            {
                "type": "operator_proposals_imported",
                "status": "blocked",
                "success": False,
                "reason": "no_soundcheck_recommendations",
                "soundcheck_readiness": {
                    "schema_version": "soundcheck_mix_readiness/v2",
                    "run_id": "run-preflight-missing",
                    "global_blockers": ["soundcheck_physical_preflight_missing"],
                },
                "results": [],
            }
        )
    )
    assert imported["soundcheck_readiness"]["global_blocker_codes"] == [
        "soundcheck_physical_preflight_missing"
    ]


def test_large_known_cp02_response_preserves_exact_sequence_state_reference():
    sequence_state = {
        "schema": "automixer.operator_proposal.sequence_state.v2",
        "cursor": "large-batch-cursor-0123456789abcdef",
        "stage": "batch_preflight",
        "content_digest": "sha256:" + "4" * 64,
        "created_at_epoch": 100.0,
        "expires_at_epoch": 130.0,
    }
    payload = {
        "type": "operator_proposal_apply_preflight_batch",
        "status": "ok",
        "success": True,
        "preflight_results": [
            {
                "proposal_id": f"p-{row_index}",
                "flags_a": [True] * 64,
                "flags_b": [False] * 64,
            }
            for row_index in range(64)
        ],
        "sequence_state": sequence_state,
    }

    serialized = project_outbound_message(payload)
    parsed = json.loads(serialized)

    assert parsed["type"] == "operator_proposal_apply_preflight_batch"
    assert parsed["status"] == "ok"
    assert parsed["sequence_state"] == sequence_state
    assert parsed["wire_projection"]["omitted_field_count"] > 0
    assert len(serialized.encode("utf-8")) <= WIRE_MAX_BYTES


def test_successful_sequence_response_without_exact_state_reference_degrades():
    serialized = project_outbound_message(
        {
            "type": "operator_proposal_apply_sequence_plan",
            "status": "ok",
            "success": True,
            "sequence_state": None,
            "planned_apply_sequence": [],
        }
    )
    parsed = json.loads(serialized)

    assert parsed["type"] == "outbound_message_degraded"
    assert parsed["outbound_message_degraded"] is True


def _trusted_terminal_receipt(
    *,
    proposal_id="sequence-p-1",
    terminal_result=None,
    final_status="applied",
):
    identity = _freeze_verified_apply_identity(
        SimpleNamespace(
            proposal_id=proposal_id,
            review_contract_version=2,
            review_content_digest="sha256:" + "6" * 64,
        )
    )
    result = terminal_result or {
        "status": "applied",
        "no_console_send_proven": False,
        "physical_verified": True,
        "wal_terminal": "committed",
        "audio_recheck_required": False,
        "audio_recheck_disposition": "not_required_by_manual_write_contract",
        "write_state_unknown": False,
        "rollback_state_unknown": False,
        "manual_review_required": False,
        "emergency_stop_activated": False,
        "emergency_stop_required": False,
    }
    return _freeze_trusted_apply_receipt(
        identity,
        terminal_result=result,
        proposal_status=("applied" if final_status == "applied" else "pending"),
        final_status=final_status,
        final_success=final_status == "applied",
        result_digest="sha256:" + "7" * 64,
        result_detail_available=True,
        result_evidence_pinned=False,
        reason_code=None if final_status == "applied" else "operator_apply_reason_available_in_result_detail",
        idempotency_key="sequence-key",
        operator_idempotency_released=final_status != "applied",
    )


def _restored_proof_terminal(**overrides):
    return {
        "status": "proof_passed",
        "proof_run": True,
        "restore_after_verify": True,
        "capability_kind": "restore_only_proof",
        "proposal_id": "sequence-proof-1",
        "review_content_digest": "sha256:" + "6" * 64,
        "operation_digest": "sha256:" + "8" * 64,
        "accepted": False,
        "owner_review_required": True,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "no_console_send_proven": False,
        "physical_verified": True,
        "proof_target_verified": True,
        "physical_state_restored": True,
        "terminal_after_console_send": True,
        "rollback_attempted": True,
        "rollback_succeeded": True,
        "wal_terminal": "proof_restored",
        "audio_recheck_passed": True,
        "write_state_unknown": False,
        "rollback_state_unknown": False,
        "manual_review_required": False,
        "emergency_stop_activated": False,
        "emergency_stop_required": False,
        "applied_operation_ids": ["proof-op"],
        "rollback_operation_ids": ["proof-op"],
        **overrides,
    }


def test_restored_proof_projects_as_distinct_non_accepted_terminal():
    identity = _freeze_verified_apply_identity(
        SimpleNamespace(
            proposal_id="sequence-proof-1",
            review_contract_version=2,
            review_content_digest="sha256:" + "6" * 64,
        )
    )
    receipt = _freeze_trusted_apply_receipt(
        identity,
        terminal_result=_restored_proof_terminal(),
        proposal_status="proof_passed",
        final_status="proof_passed",
        final_success=True,
        result_digest="sha256:" + "7" * 64,
        result_detail_available=True,
        result_evidence_pinned=False,
        reason_code=None,
        idempotency_key="proof-key",
        operator_idempotency_released=False,
    )
    parsed = json.loads(project_outbound_message({}, trusted_receipt=receipt))

    assert parsed["status"] == "proof_passed"
    assert parsed["success"] is True
    assert parsed["accepted"] is False
    assert parsed["proposal_status"] == "proof_passed"
    assert parsed["apply_status"] == "proof_passed"
    assert parsed["proof_run"] is True
    assert parsed["restore_after_verify"] is True
    assert parsed["capability_kind"] == "restore_only_proof"
    assert parsed["live_authority_granted"] is False
    assert parsed["readiness_closure_allowed"] is False

    with pytest.raises(Exception, match="restored_proof_terminal_invalid"):
        _freeze_trusted_apply_receipt(
            identity,
            terminal_result=_restored_proof_terminal(
                rollback_operation_ids=[]
            ),
            proposal_status="proof_passed",
            final_status="proof_passed",
            final_success=True,
            result_digest="sha256:" + "7" * 64,
            result_detail_available=True,
            result_evidence_pinned=True,
            reason_code=None,
        )


def test_m110_committed_reset_is_known_applied_and_projects_bound_evidence():
    identity = _freeze_verified_apply_identity(
        SimpleNamespace(
            proposal_id="sequence-reset-commit-1",
            review_contract_version=2,
            review_content_digest="sha256:" + "6" * 64,
        )
    )
    terminal = {
        "status": "applied",
        "capability_kind": "maintenance_dsp_reset_commit_once",
        "operation_digest": "sha256:" + "8" * 64,
        "bundle_kind": "maintenance_dsp_reset_bundle",
        "review_preimage_sha256": "sha256:" + "9" * 64,
        "before_manifest_sha256": "sha256:" + "a" * 64,
        "after_manifest_sha256": "sha256:" + "b" * 64,
        "verification_cut_reset_surface_sha256": "sha256:" + "c" * 64,
        "verification_cut_preservation_manifest_sha256": (
            "sha256:" + "b" * 64
        ),
        "no_outside_mutation_attempt_proven": True,
        "current_state_guaranteed_at_response": False,
        "fresh_readback_required_before_next_stage": True,
        "attended_no_touch_required_during_execution": True,
        "later_console_event_invalidates_readiness_freshness": True,
        "no_console_send_proven": False,
        "physical_verified": True,
        "physical_state_restored": False,
        "terminal_after_console_send": True,
        "rollback_attempted": False,
        "rollback_succeeded": False,
        "wal_terminal": "committed",
        "audio_recheck_passed": True,
        "write_state_unknown": False,
        "rollback_state_unknown": False,
        "manual_review_required": False,
        "emergency_stop_activated": False,
        "emergency_stop_required": False,
    }
    receipt = _freeze_trusted_apply_receipt(
        identity,
        terminal_result=terminal,
        proposal_status="applied",
        final_status="applied",
        final_success=True,
        result_digest="sha256:" + "7" * 64,
        result_detail_available=True,
        result_evidence_pinned=False,
        reason_code=None,
        idempotency_key="reset-commit-key",
        operator_idempotency_released=False,
    )
    parsed = json.loads(project_outbound_message({}, trusted_receipt=receipt))

    assert receipt.success is True
    assert receipt.rollback_state_unknown is False
    assert receipt.manual_review_required is False
    assert parsed["status"] == "applied"
    assert parsed["success"] is True
    assert parsed["capability_kind"] == (
        "maintenance_dsp_reset_commit_once"
    )
    assert parsed["bundle_kind"] == "maintenance_dsp_reset_bundle"
    assert parsed["review_preimage_sha256"] == "sha256:" + "9" * 64
    assert parsed["physical_state_restored"] is False
    assert parsed["rollback_state_unknown"] is False
    assert parsed["fresh_readback_required_before_next_stage"] is True


def _sequence_execution_message(*, status="applied", forged_receipt=None):
    return {
        "type": "operator_proposal_apply_sequence_execution_result",
        "status": status,
        "success": status == "applied",
        "summary": {
            "input_candidate_count": 1,
            "attempted_count": 1,
            "applied_count": 1 if status == "applied" else 0,
            "blocked_count": 0,
            "unknown_count": 0,
            "possible_console_send_count": 1,
            "no_console_send_proven_count": 0,
            "applied_proposal_ids": ["forged-summary-id"],
            "blocked_proposal_ids": ["forged-blocked-id"],
        },
        "execution_results": [
            {
                "planned_apply_index": 0,
                "proposal_id": "sequence-p-1",
                "candidate_index": 0,
                "status": status,
                "apply_attempted": True,
                "apply_receipt": forged_receipt,
            }
        ],
        "blocked_results": [],
    }


def test_sequence_execution_wire_is_derived_from_authentic_child_receipt():
    child = _trusted_terminal_receipt()
    message = _sequence_execution_message(
        forged_receipt=compact_trusted_apply_receipt_reference(child)
    )
    receipt = _freeze_trusted_sequence_execution_receipt(
        message,
        child_receipts=[child],
        expected_candidate_count=1,
    )

    parsed = json.loads(
        project_outbound_message(message, trusted_receipt=receipt)
    )

    assert parsed["status"] == "applied"
    assert parsed["success"] is True
    assert parsed["no_console_send_proven"] is False
    assert parsed["write_state_unknown"] is False
    assert parsed["manual_review_required"] is False
    assert parsed["execution_results"][0]["apply_receipt"]["proposal_id"] == "sequence-p-1"
    assert parsed["summary"]["applied_proposal_ids"] == ["sequence-p-1"]
    assert parsed["summary"]["blocked_proposal_ids"] == []
    assert "forged" not in json.dumps(parsed)


def test_sequence_execution_safe_no_send_and_verified_rollback_remain_distinct():
    no_send_receipt = _trusted_terminal_receipt(
        terminal_result={
            "status": "blocked",
            "no_console_send_proven": True,
            "write_state_unknown": False,
            "rollback_state_unknown": False,
            "manual_review_required": False,
            "wal_terminal": "not_prepared",
        },
        final_status="apply_blocked",
    )
    rollback_receipt = _trusted_terminal_receipt(
        terminal_result={
            "status": "blocked",
            "no_console_send_proven": False,
            "physical_verified": False,
            "physical_state_restored": True,
            "rollback_attempted": True,
            "rollback_succeeded": True,
            "write_state_unknown": False,
            "rollback_state_unknown": False,
            "manual_review_required": False,
            "emergency_stop_required": False,
            "wal_terminal": "rolled_back",
        },
        final_status="apply_blocked",
    )

    projected = []
    for child in (no_send_receipt, rollback_receipt):
        message = _sequence_execution_message(
            status="blocked",
            forged_receipt=compact_trusted_apply_receipt_reference(child),
        )
        receipt = _freeze_trusted_sequence_execution_receipt(
            message,
            child_receipts=[child],
            expected_candidate_count=1,
        )
        projected.append(
            json.loads(project_outbound_message(message, trusted_receipt=receipt))
        )

    assert projected[0]["status"] == "blocked"
    assert projected[0]["no_console_send_proven"] is True
    assert projected[1]["status"] == "blocked"
    assert projected[1]["no_console_send_proven"] is False
    assert projected[1]["write_state_unknown"] is False
    assert projected[1]["rollback_state_unknown"] is False


def test_sequence_execution_without_authentic_aggregate_is_fixed_unknown():
    message = _sequence_execution_message(
        forged_receipt={
            "proposal_id": "sequence-p-1",
            "status": "applied",
            "success": True,
            "no_console_send_proven": False,
            "physical_verified": True,
            "wal_terminal": "committed",
        }
    )

    parsed = json.loads(project_outbound_message(message))

    assert parsed["type"] == "operator_proposal_apply_sequence_execution_result"
    assert parsed["status"] == "unknown"
    assert parsed["success"] is False
    assert parsed["no_console_send_proven"] is None
    assert parsed["write_state_unknown"] is True
    assert parsed["rollback_state_unknown"] is True
    assert parsed["manual_review_required"] is True
    assert parsed["emergency_stop_required"] is True
    assert parsed["response_degraded"] is True


def test_sequence_execution_receipt_never_repromotes_handler_unknown_row():
    child = _trusted_terminal_receipt()
    message = _sequence_execution_message(
        status="unknown",
        forged_receipt=None,
    )
    message["execution_results"][0].update(
        {
            "manual_review_required": True,
            "executor_emergency_stop_activated": True,
            "executor_emergency_stop_teardown_complete": True,
            "executor_emergency_stop_required": False,
        }
    )
    aggregate = _freeze_trusted_sequence_execution_receipt(
        message,
        child_receipts=[child],
        expected_candidate_count=1,
    )

    parsed = json.loads(
        project_outbound_message(message, trusted_receipt=aggregate)
    )

    assert parsed["status"] == "unknown"
    assert parsed["success"] is False
    assert parsed["no_console_send_proven"] is None
    assert parsed["manual_review_required"] is True
    assert parsed["emergency_stop_activated"] is True
    assert parsed["execution_results"][0]["status"] == "unknown"
    assert "apply_receipt" not in parsed["execution_results"][0]


def test_cp02_import_readiness_ref_preserves_blocker_and_count_truth():
    payload = {
        "type": "operator_proposals_imported",
        "status": "ok",
        "success": True,
        "full_results_endpoint": "get_decision_queue",
        "soundcheck_readiness_endpoint": "get_auto_soundcheck_status",
        "soundcheck_readiness": {
            "schema_version": "soundcheck_mix_readiness/v2",
            "run_id": "run-1",
            "plan_revision": 2,
            "plan_fingerprint": "fp-1",
            "mix_readiness": "soundcheck_incomplete",
            "safety_state": "blocked",
            "execution_status": "pending",
            "proposal_only": False,
            "baseline_verified": True,
            "logical_counts": {
                "total": 10,
                "applied": 2,
                "already_satisfied": 1,
                "replaced": 1,
                "operator_review": 1,
                "forbidden": 1,
                "pending": 4,
            },
            "physical_step_counts": {
                "planned": 8,
                "committed": 2,
                "rolled_back": 1,
            },
            "global_blockers": ["mixer_disconnected"],
            "unresolved_required_operation_ids": ["op-secret-1", "op-secret-2"],
        },
        "results": [],
    }

    parsed = json.loads(project_outbound_message(payload))
    readiness = parsed["soundcheck_readiness"]

    assert readiness["global_blocker_count"] == 1
    assert readiness["unresolved_required_count"] == 2
    assert readiness["logical_counts"]["pending"] == 4
    assert readiness["physical_step_counts"]["committed"] == 2
    assert "op-secret" not in json.dumps(parsed)
    assert parsed["full_results_endpoint"] == "get_decision_queue"
    assert parsed["soundcheck_readiness_endpoint"] == "get_auto_soundcheck_status"


@pytest.mark.parametrize(
    "reason",
    [
        "no_soundcheck_recommendations",
        "soundcheck_engine_superseded",
        "operator_proposal_queue_unavailable",
        "soundcheck_manifest_same_identity_content_mismatch",
    ],
)
def test_cp02_auto_soundcheck_import_preserves_registered_trigger_time_and_reason(reason):
    parsed = json.loads(
        project_outbound_message(
            {
                "type": "operator_proposals_imported",
                "status": "blocked",
                "success": False,
                "reason": reason,
                "trigger": "auto_soundcheck_state",
                "trigger_state": "running",
                "generated_at": "2026-07-13T14:45:00Z",
                "results": [],
            }
        )
    )

    assert parsed["reason"] == reason
    assert parsed["trigger"] == "auto_soundcheck_state"
    assert parsed["trigger_state"] == "running"
    assert parsed["generated_at"] == "2026-07-13T14:45:00Z"


def test_compact_row_never_calls_full_to_dict_and_contains_no_raw_or_result():
    row = compact_proposal_row(FakeProposal(raw={"marker": "x" * 1_000_000}))

    assert row["id"] == "proposal-unicode"
    assert row["review_contract_version"] == 2
    assert "raw" not in row
    assert "result" not in row
    assert "marker" not in json.dumps(row)


def test_mixing_agent_identity_is_stable_and_analysis_only_without_parameters():
    action = {
        "type": "set_eq",
        "channel": 2,
        "parameters": {"blob": "x" * 1_000_000},
        "reason": "masking",
        "source": "rule_engine",
        "timestamp": 1_784_000_000.0,
        "risk": "high",
    }

    first = compact_mixing_agent_row(action)
    second = compact_mixing_agent_row(action)

    assert first["id"] == second["id"]
    assert first["analysis_only"] is True
    assert first["import_required"] is True
    assert first["can_apply"] is False
    assert "parameters" not in first


def test_mixing_agent_identity_is_cycle_safe_and_refuses_hostile_mapping_access():
    parameters = {"frequency": 300.0}
    parameters["self"] = parameters
    cyclic = compact_mixing_agent_row(
        {
            "type": "set_eq",
            "channel": 2,
            "parameters": parameters,
        }
    )

    class HostileMapping(Mapping):
        def __getitem__(self, key):
            raise AssertionError("hostile mapping was accessed")

        def __iter__(self):
            raise AssertionError("hostile mapping was iterated")

        def __len__(self):
            raise AssertionError("hostile mapping length was read")

    hostile = compact_mixing_agent_row(HostileMapping())

    assert cyclic["id"].startswith("mixing:")
    assert hostile["id"].startswith("mixing:")
    assert hostile["title_preview"] == "Mixing agent action"
    assert hostile["can_apply"] is False


def test_mixing_agent_identity_budgets_large_and_deep_parameters():
    deep = {}
    cursor = deep
    for index in range(100):
        cursor["child"] = {}
        cursor["payload"] = "x" * 2_000_000
        cursor = cursor["child"]

    first = compact_mixing_agent_row({"type": "set_eq", "parameters": deep})
    second = compact_mixing_agent_row({"type": "set_eq", "parameters": deep})

    assert first["id"] == second["id"]
    assert "payload" not in json.dumps(first)


def test_mixing_agent_row_is_total_for_extreme_timestamp_and_hostile_values():
    class StringBomb:
        def __str__(self):
            raise RuntimeError("must not stringify")

    class HostileMeta(type):
        def __getattribute__(cls, name):
            if name in {"__module__", "__name__"}:
                raise RuntimeError("must not inspect hostile type")
            return super().__getattribute__(name)

    class HostileValue(metaclass=HostileMeta):
        pass

    row = compact_mixing_agent_row(
        {
            "type": "set_eq",
            "timestamp": 1e308,
            "action_identity_digest": StringBomb(),
            "parameters": {"hostile": HostileValue()},
        }
    )

    assert row["id"].startswith("mixing:")
    assert row["created_at"] is None
    assert row["can_apply"] is False


def test_mixing_agent_identity_caps_raw_mapping_entries_not_only_string_keys():
    first_parameters = {index: index for index in range(10_000)}
    first_parameters["late"] = "first"
    second_parameters = dict(first_parameters)
    second_parameters["late"] = "second"

    first = compact_mixing_agent_row(
        {"type": "set_eq", "parameters": first_parameters}
    )
    second = compact_mixing_agent_row(
        {"type": "set_eq", "parameters": second_parameters}
    )

    assert first["id"] == second["id"]


def test_mixing_agent_source_cap_reports_exact_unpageable_counts():
    class Agent:
        @staticmethod
        def get_pending_actions():
            return [
                {
                    "type": "set_fader",
                    "channel": index,
                    "parameters": {"value": -12.0},
                }
                for index in range(400)
            ]

        @staticmethod
        def get_action_history(limit):
            assert limit == 128
            return []

    server = _wire_server(OperatorProposalQueue(), mixing_agent=Agent())
    rows = collect_compact_rows(server)
    response = DecisionQueueSnapshotManager().response(
        server,
        object(),
        {"contract_version": 2, "page_size": 50},
    )

    assert len(rows) == 128
    assert rows.source_omitted_count == 272
    assert response["counts"]["total_count"] == 400
    assert response["counts"]["materialized_source_count"] == 128
    assert response["counts"]["source_omitted_count"] == 272
    assert response["counts"]["omitted_count"] == 272
    assert response["counts"]["pinned_omitted_count"] == 272
    assert response["status"] == "degraded_overflow"
    assert response["overflow"]["apply_ui_disabled"] is True


def test_fixed_point_wire_oracle_matches_transport_serialization():
    payload = finalize_wire_size(
        {
            "type": "decision_queue",
            "contract_version": 2,
            "items": [{"title_preview": "Вокал 🎙️"}],
            "wire": {},
        }
    )

    assert payload["wire"]["serialized_bytes"] == wire_serialized_bytes(payload)
    assert payload["wire"]["serialized_bytes"] < WIRE_MAX_BYTES


def _supervised_mode():
    return {
        "mode": "supervised",
        "capabilities": {
            "can_create_proposals": True,
            "can_apply_to_console": True,
            "allowed_live_write_kinds": ["fader"],
        },
    }


def _create_queue_proposal(queue, *, proposal_id="p-1", raw_extra=None):
    payload = {
        "id": proposal_id,
        "title": "Lead vocal",
        "reason": "balance",
        "requested_change": {"value_type": "fader", "channel": 1, "value": -9.0},
    }
    payload.update(raw_extra or {})
    result = queue.create(payload, _supervised_mode())
    assert result["status"] == "created"
    return queue.find(proposal_id=proposal_id)


def _wire_server(queue, *, mixing_agent=None, connected=True, mode=None):
    return SimpleNamespace(
        operator_proposal_queue=queue,
        mixing_agent=mixing_agent,
        mixer_client=SimpleNamespace(is_connected=connected),
        _wing_observation_only=False,
        _is_shutting_down=False,
        _manual_write_emergency_stopped=False,
        _system_emergency_stop_in_progress=False,
        _system_emergency_stop_errors=[],
        _manual_write_recovery_transactions={},
        _console_write_lock=None,
        _console_writer_conflict_recorded=False,
        get_operator_mode_status=lambda: mode or _supervised_mode(),
    )


def test_tombstoned_unknown_write_remains_a_global_hazard(tmp_path):
    queue = OperatorProposalQueue(
        max_history_bytes=1,
        result_store=ResultEvidenceStore(tmp_path / "results", min_free_bytes=0),
    )
    proposal = _create_queue_proposal(queue, proposal_id="unknown-tombstone")
    claim = queue.claim_verified_for_apply(
        proposal,
        expected_review_contract_version=proposal.review_contract_version,
        expected_review_content_digest=proposal.review_content_digest,
    )
    assert claim["claimed"] is True
    reservation = queue.reserve_result_evidence(proposal)
    queue.mark_apply_result(
        proposal,
        {
            "status": "blocked",
            "reason": "readback_lost_after_send",
            "no_console_send_proven": False,
            "write_state_unknown": True,
            "physical_state_restored": False,
            "manual_review_required": True,
        },
        evidence_reservation_binding=reservation["reservation_binding"],
    )
    server = _wire_server(queue)
    rows = collect_compact_rows(server)
    hazards = collect_global_hazards(server, rows)

    assert proposal.terminal_tombstone is True
    assert rows == []
    assert hazards["fatal"] is True
    assert "terminal_write_state_unknown" in hazards["codes"]
    assert "terminal_rollback_state_unknown" in hazards["codes"]
    assert "terminal_manual_review_after_send" in hazards["codes"]


def test_restarted_result_artifact_hazards_enter_fixed_global_envelope():
    queue = SimpleNamespace(
        result_persistence_failed=False,
        result_recovery_hazards=(
            {
                "code": "terminal_write_state_unknown",
                "result_digest": "sha256:" + "1" * 64,
            },
            {
                "code": "terminal_rollback_state_unknown",
                "result_digest": "sha256:" + "1" * 64,
            },
            {
                "code": "terminal_manual_review_after_send",
                "result_digest": "sha256:" + "1" * 64,
            },
            {
                "code": "operator_proposal_result_recovery_required",
                "result_digest": "sha256:" + "2" * 64,
            },
            {
                "code": "untrusted_free_text_must_not_escape",
                "result_digest": "sha256:" + "3" * 64,
            },
        ),
        iter_terminal_hazards=lambda: (),
    )
    hazards = collect_global_hazards(_wire_server(queue), [])

    assert hazards["fatal"] is True
    assert hazards["codes"] == [
        "operator_proposal_result_recovery_required",
        "terminal_manual_review_after_send",
        "terminal_rollback_state_unknown",
        "terminal_write_state_unknown",
    ]
    assert hazards["source_counts"]["terminal_result"] == 4
    assert "untrusted_free_text_must_not_escape" not in hazards["codes"]


def test_queue_binds_private_review_identity_and_status_result_do_not_change_it():
    queue = OperatorProposalQueue()
    proposal = _create_queue_proposal(queue)
    digest = proposal.review_content_digest
    content = proposal.review_content_bytes

    proposal.status = "applying"
    proposal.updated_at = "2026-07-13T08:01:00Z"
    proposal.result = {"large": "x" * 1_000_000}

    integrity = queue.validate_stored_review_integrity(proposal)
    assert integrity["valid"] is True
    assert proposal.review_content_digest == digest
    assert proposal.review_content_bytes == content


def test_queue_detects_authoritative_mutation_before_claim_without_mutation():
    queue = OperatorProposalQueue()
    proposal = _create_queue_proposal(queue)
    proposal.raw["requested_change"]["value"] = -3.0

    result = queue.claim_verified_for_apply(
        proposal,
        expected_review_contract_version=2,
        expected_review_content_digest=proposal.review_content_digest,
    )

    assert result["claimed"] is False
    assert result["reason"] == "proposal_review_content_integrity_error"
    assert proposal.status == "pending"


def test_queue_verified_claim_returns_stored_byte_snapshot():
    queue = OperatorProposalQueue()
    proposal = _create_queue_proposal(queue)

    result = queue.claim_verified_for_apply(
        proposal,
        expected_review_contract_version=2,
        expected_review_content_digest=proposal.review_content_digest,
    )

    assert result["claimed"] is True
    execution = result["execution"]
    content = execution.content_copy()
    assert content["proposal_id"] == proposal.id
    assert content["requested_change"]["value"] == -9.0
    assert proposal.status == "applying"


def test_queue_admission_rejects_oversize_without_consuming_id(monkeypatch):
    import operator_proposal_queue as queue_module

    monkeypatch.setattr(queue_module, "MAX_PROPOSAL_REVIEW_BYTES", 256)
    queue = OperatorProposalQueue()
    result = queue.create(
        {
            "id": "oversize",
            "title": "large",
            "blob": "x" * 1_000,
            "requested_change": {"value_type": "fader", "channel": 1, "value": -9.0},
        },
        _supervised_mode(),
    )

    assert result["reason"] == "proposal_review_content_too_large"
    assert queue.find(proposal_id="oversize") is None
    assert queue.proposal_id_was_seen("oversize") is False


def test_queue_pending_count_capacity_rejects_before_id_ledger(monkeypatch):
    import operator_proposal_queue as queue_module

    monkeypatch.setattr(queue_module, "MAX_PENDING_PROPOSALS", 1)
    queue = OperatorProposalQueue()
    _create_queue_proposal(queue, proposal_id="first")
    result = queue.create(
        {
            "id": "second",
            "requested_change": {"value_type": "fader", "channel": 1, "value": -10.0},
        },
        _supervised_mode(),
    )

    assert result["reason"] == "operator_proposal_pending_capacity_exceeded"
    assert queue.proposal_id_was_seen("second") is False


def test_one_megabyte_review_is_absent_from_list_and_reassembles_from_bounded_chunks():
    queue = OperatorProposalQueue()
    proposal = _create_queue_proposal(
        queue,
        raw_extra={"evidence_marker": "PRIVATE-MARKER-" + "x" * 1_000_000},
    )

    class Mixer:
        is_connected = True

    class Server:
        operator_proposal_queue = queue
        mixing_agent = None
        mixer_client = Mixer()
        _wing_observation_only = False
        _is_shutting_down = False
        _manual_write_emergency_stopped = False
        _system_emergency_stop_in_progress = False
        _system_emergency_stop_errors = []
        _manual_write_recovery_transactions = {}
        _console_write_lock = None

        @staticmethod
        def get_operator_mode_status():
            return _supervised_mode()

    manager = DecisionQueueSnapshotManager()
    response = manager.response(
        Server(),
        object(),
        {"contract_version": 2, "page_size": 50},
    )

    assert response["wire"]["serialized_bytes"] == wire_serialized_bytes(response)
    assert response["wire"]["serialized_bytes"] <= WIRE_MAX_BYTES
    assert "PRIVATE-MARKER" not in json.dumps(response)
    assert response["items"][0]["review_content_digest"] == proposal.review_content_digest

    chunks = []
    chunk_index = 0
    while True:
        detail = build_proposal_detail_response(
            queue,
            {
                "proposal_id": proposal.id,
                "expected_review_contract_version": 2,
                "expected_review_content_digest": proposal.review_content_digest,
                "chunk_index": chunk_index,
            },
        )
        assert detail["wire"]["serialized_bytes"] == wire_serialized_bytes(detail)
        assert detail["wire"]["serialized_bytes"] <= WIRE_MAX_BYTES
        chunks.append(detail["content"]["data"])
        chunk_index += 1
        if chunk_index == detail["content"]["chunk_count"]:
            break

    import base64

    reassembled = b"".join(base64.b64decode(item) for item in chunks)
    assert reassembled == proposal.review_content_bytes
    assert b"PRIVATE-MARKER" in reassembled


@pytest.mark.parametrize("chunk_index", [False, "0", 0.0])
def test_proposal_detail_rejects_coerced_chunk_index(chunk_index):
    queue = OperatorProposalQueue()
    proposal = _create_queue_proposal(queue)

    response = build_proposal_detail_response(
        queue,
        {
            "proposal_id": proposal.id,
            "expected_review_contract_version": 2,
            "expected_review_content_digest": proposal.review_content_digest,
            "chunk_index": chunk_index,
        },
    )

    assert response["reason"] == "operator_proposal_detail_chunk_index_invalid"
    assert response["no_console_send_proven"] is True


def test_result_detail_reassembles_digest_bound_chunks_with_exact_wire_budget(
    tmp_path,
    monkeypatch,
):
    import base64
    import hashlib
    from server import convert_numpy_types
    from ws_transport import serialize_message

    store = ResultEvidenceStore(
        tmp_path / "results",
        min_free_bytes=0,
    )
    queue = OperatorProposalQueue(
        result_store=store,
    )
    proposal = _create_queue_proposal(
        queue,
        proposal_id="large-result-detail",
    )
    claim = queue.claim_verified_for_apply(
        proposal,
        expected_review_contract_version=proposal.review_contract_version,
        expected_review_content_digest=proposal.review_content_digest,
    )
    assert claim["claimed"] is True
    reservation = queue.reserve_result_evidence(proposal)
    completed = queue.mark_apply_result(
        proposal,
        {
            "status": "applied",
            "success": True,
            "evidence": "🎙️" * 40_000,
        },
        evidence_reservation_binding=reservation["reservation_binding"],
    )
    digest = completed["result_digest"]

    chunks = []
    chunk_index = 0
    while True:
        detail = build_proposal_result_detail_response(
            queue,
            {
                "proposal_id": proposal.id,
                "expected_result_digest": digest,
                "chunk_index": chunk_index,
            },
        )
        assert detail["type"] == "operator_proposal_result_detail"
        assert detail["content"]["media_type"] == (
            "application/vnd.automixer.operator-proposal-result+json;version=2"
        )
        assert detail["content"]["raw_chunk_bytes"] <= 65_536
        assert detail["wire"]["serialized_bytes"] == wire_serialized_bytes(detail)
        assert detail["wire"]["serialized_bytes"] <= WIRE_MAX_BYTES
        raw_chunk = base64.b64decode(detail["content"]["data"])
        assert hashlib.sha256(raw_chunk).hexdigest() == detail["content"][
            "chunk_sha256"
        ]
        assert detail["content"]["content_sha256"] == digest.removeprefix(
            "sha256:"
        )
        assert len(
            serialize_message(detail, convert_numpy_types).encode("utf-8")
        ) == detail["wire"]["serialized_bytes"]
        chunks.append(raw_chunk)
        chunk_index += 1
        if chunk_index == detail["content"]["chunk_count"]:
            break

    reassembled = b"".join(chunks)
    assert hashlib.sha256(reassembled).hexdigest() == digest.removeprefix("sha256:")
    assert json.loads(reassembled)["evidence"] == "🎙️" * 40_000

    mismatch = build_proposal_result_detail_response(
        queue,
        {
            "proposal_id": proposal.id,
            "expected_result_digest": "sha256:" + "0" * 64,
            "chunk_index": 0,
        },
    )
    assert mismatch["reason"] == "operator_proposal_result_digest_mismatch"
    invalid_chunk = build_proposal_result_detail_response(
        queue,
        {
            "proposal_id": proposal.id,
            "expected_result_digest": digest,
            "chunk_index": chunk_index,
        },
    )
    assert invalid_chunk["reason"] == (
        "operator_proposal_result_detail_chunk_index_invalid"
    )
    for invalid_index in (True, 0.0, "0", "00"):
        invalid = build_proposal_result_detail_response(
            queue,
            {
                "proposal_id": proposal.id,
                "expected_result_digest": digest,
                "chunk_index": invalid_index,
            },
        )
        assert invalid["reason"] == (
            "operator_proposal_result_detail_chunk_index_invalid"
        )

    (store.root / f"{digest.removeprefix('sha256:')}.json").unlink()
    not_retained = build_proposal_result_detail_response(
        queue,
        {
            "proposal_id": proposal.id,
            "expected_result_digest": digest,
            "chunk_index": 0,
        },
    )
    assert not_retained["reason"] == (
        "operator_proposal_result_detail_not_retained"
    )
    not_found = build_proposal_result_detail_response(
        queue,
        {
            "proposal_id": "never-seen",
            "expected_result_digest": digest,
            "chunk_index": 0,
        },
    )
    assert not_found["reason"] == "operator_proposal_not_found"

    class HostileLoadError(Exception):
        code = "hostile-" + ("x" * 1_000_000)

    def hostile_load(*_args, **_kwargs):
        raise HostileLoadError()

    monkeypatch.setattr(queue, "load_result_evidence", hostile_load)
    hostile = build_proposal_result_detail_response(
        queue,
        {
            "proposal_id": proposal.id,
            "expected_result_digest": digest,
            "chunk_index": 0,
        },
    )
    assert hostile["reason"] == "operator_proposal_result_detail_unavailable"
    assert hostile["wire"]["serialized_bytes"] <= WIRE_MAX_BYTES


def test_compact_reference_uses_durable_detail_truth_in_both_directions(tmp_path):
    queue = OperatorProposalQueue(
        result_store=ResultEvidenceStore(
            tmp_path / "results",
            min_free_bytes=0,
        )
    )
    proposal = _create_queue_proposal(queue, proposal_id="detail-truth")
    proposal.result = {"status": "applied", "not_persisted": True}
    assert compact_proposal_reference(proposal)["result_detail_available"] is False

    proposal.result = None
    claim = queue.claim_verified_for_apply(
        proposal,
        expected_review_contract_version=proposal.review_contract_version,
        expected_review_content_digest=proposal.review_content_digest,
    )
    assert claim["claimed"] is True
    reservation = queue.reserve_result_evidence(proposal)
    queue.mark_apply_result(
        proposal,
        {"status": "applied", "success": True},
        evidence_reservation_binding=reservation["reservation_binding"],
    )
    proposal.result = None
    assert compact_proposal_reference(proposal)["result_detail_available"] is True


def test_unbound_proposal_and_every_v1_row_are_explicitly_read_only():
    legacy = FakeProposal(
        review_contract_version=None,
        review_content_digest=None,
        review_content_bytes=b"",
    )
    row = compact_proposal_row(legacy)
    assert row["can_apply"] is False
    assert row["requires_approval"] is False
    assert row["review_contract_status"] == "upgrade_required"
    assert "proposal_review_contract_upgrade_required" in row["apply_blocker_codes"]

    queue = OperatorProposalQueue()
    _create_queue_proposal(queue)
    shim = build_v1_decision_queue_shim(_wire_server(queue))
    assert shim["apply_authority"] is False
    assert shim["wire"]["serialized_bytes"] <= WIRE_MAX_BYTES
    assert shim["pending_actions"]
    assert all(item["can_apply"] is False for item in shim["pending_actions"])
    assert all(item["requires_approval"] is False for item in shim["pending_actions"])
    assert all(
        "legacy_decision_queue_read_only" in item["apply_blocker_codes"]
        for item in shim["pending_actions"]
    )


def test_mixing_agent_identity_changes_when_only_full_parameters_change():
    base = {
        "type": "set_eq",
        "channel": 2,
        "reason": "masking",
        "source": "rule_engine",
        "timestamp": 1_784_000_000.0,
        "parameters": {"frequency": 300.0, "gain": -2.0},
    }
    changed = {**base, "parameters": {"frequency": 350.0, "gain": -2.0}}
    assert compact_mixing_agent_row(base)["id"] != compact_mixing_agent_row(changed)["id"]


def test_authority_id_over_wire_limit_is_rejected_without_consuming_ledger():
    queue = OperatorProposalQueue()
    proposal_id = "🎙️" * 100
    result = queue.create(
        {
            "id": proposal_id,
            "requested_change": {"value_type": "fader", "channel": 1, "value": -9.0},
        },
        _supervised_mode(),
    )
    assert result["reason"] == "proposal_id_too_large"
    assert queue.proposal_id_was_seen(proposal_id) is False


def test_cursor_owner_expiry_reuse_and_raw_mutation_staleness(monkeypatch):
    import decision_queue_wire_contract as wire_module

    queue = OperatorProposalQueue()
    first_proposal = _create_queue_proposal(queue, proposal_id="p-1")
    for index in range(2, 53):
        _create_queue_proposal(queue, proposal_id=f"p-{index}")
    server = _wire_server(queue)
    manager = DecisionQueueSnapshotManager()
    owner = object()

    first = manager.response(server, owner, {"contract_version": 2, "page_size": 50})
    cursor = first["page"]["next_cursor"]
    revision = first["snapshot"]["revision"]
    assert cursor
    second_request = {
        "contract_version": 2,
        "cursor": cursor,
        "expected_snapshot_revision": revision,
    }
    second = manager.response(server, owner, second_request)
    repeated = manager.response(server, owner, second_request)
    assert second["page"]["next_cursor"] == repeated["page"]["next_cursor"]
    assert len(manager._cursor_index) <= 2

    wrong_owner = manager.response(server, object(), second_request)
    assert wrong_owner["reason"] == "decision_queue_cursor_owner_mismatch"

    first_proposal.raw["requested_change"]["value"] = -3.0
    stale = manager.response(server, owner, second_request)
    assert stale["reason"] == "decision_queue_cursor_stale"

    stable_queue = OperatorProposalQueue()
    for index in range(1, 52):
        _create_queue_proposal(stable_queue, proposal_id=f"e-{index}")
    stable_manager = DecisionQueueSnapshotManager()
    clock = [100.0]
    monkeypatch.setattr(wire_module.time, "time", lambda: clock[0])
    expiring = stable_manager.response(
        _wire_server(stable_queue), owner, {"contract_version": 2, "page_size": 50}
    )
    clock[0] += 31.0
    expired = stable_manager.response(
        _wire_server(stable_queue),
        owner,
        {
            "contract_version": 2,
            "cursor": expiring["page"]["next_cursor"],
            "expected_snapshot_revision": expiring["snapshot"]["revision"],
        },
    )
    assert expired["reason"] == "decision_queue_snapshot_expired"


@pytest.mark.parametrize(
    "frame",
    [
        {"contract_version": 2},
        {"contract_version": 2, "page_size": True},
        {"contract_version": 2, "page_size": "50"},
        {"contract_version": 2, "page_size": 50.0},
        {"contract_version": 2, "page_size": 49},
        {"contract_version": 2, "page_size": 200},
        {"contract_version": 2, "page_size": 50, "limit": 50},
    ],
)
def test_first_page_rejects_non_exact_page_size(frame):
    response = DecisionQueueSnapshotManager().response(
        _wire_server(OperatorProposalQueue()),
        object(),
        frame,
    )

    assert response["reason"] == "decision_queue_page_size_invalid"
    assert response["no_console_send_proven"] is True
    assert response["proposal_status_mutation"] is False
    assert response["idempotency_reserved"] is False


def test_new_owner_token_survives_pruning_and_can_page(monkeypatch):
    import decision_queue_wire_contract as wire_module

    monkeypatch.setattr(wire_module, "SNAPSHOT_MAX_GLOBAL", 1)
    queue = OperatorProposalQueue()
    for index in range(51):
        _create_queue_proposal(queue, proposal_id=f"owner-prune-{index}")
    server = _wire_server(queue)
    manager = DecisionQueueSnapshotManager()
    manager.response(server, object(), {"contract_version": 2, "page_size": 50})
    manager.response(server, object(), {"contract_version": 2, "page_size": 50})
    owner = object()

    first = manager.response(
        server,
        owner,
        {"contract_version": 2, "page_size": 50},
    )
    second = manager.response(
        server,
        owner,
        {
            "contract_version": 2,
            "cursor": first["page"]["next_cursor"],
            "expected_snapshot_revision": first["snapshot"]["revision"],
        },
    )

    assert second["status"] == "ok"
    assert second["page"]["returned_count"] == 1


def test_snapshot_manager_release_owner_removes_connection_state():
    queue = OperatorProposalQueue()
    for index in range(51):
        _create_queue_proposal(queue, proposal_id=f"owner-release-{index}")
    server = _wire_server(queue)
    manager = DecisionQueueSnapshotManager()
    owner = object()
    first = manager.response(
        server,
        owner,
        {"contract_version": 2, "page_size": 50},
    )
    token = next(
        token for connection, token in manager._owner_tokens if connection is owner
    )

    manager.release_owner(owner)

    assert all(connection is not owner for connection, _ in manager._owner_tokens)
    assert all(snapshot.owner != token for snapshot in manager._snapshots.values())
    assert first["page"]["next_cursor"] not in manager._cursor_index
    assert all(tombstone[1] != token for tombstone in manager._cursor_tombstones.values())


def test_pinned_overflow_and_global_hazard_force_all_apply_surfaces_false(monkeypatch):
    import decision_queue_wire_contract as wire_module

    queue = OperatorProposalQueue()
    proposal = _create_queue_proposal(queue)
    server = _wire_server(queue, connected=False)
    manager = DecisionQueueSnapshotManager()
    response = manager.response(server, object(), {"contract_version": 2, "page_size": 50})
    assert response["hazards"]["fatal"] is True
    assert "mixer_disconnected" in response["hazards"]["codes"]
    assert all(item["can_apply"] is False for item in response["items"])

    detail = build_proposal_detail_response(
        queue,
        {
            "proposal_id": proposal.id,
            "expected_review_contract_version": 2,
            "expected_review_content_digest": proposal.review_content_digest,
            "chunk_index": 0,
        },
        server=server,
    )
    assert detail["current_state"]["can_apply"] is False
    assert any(
        item == "global_hazard:mixer_disconnected"
        for item in detail["current_state"]["apply_blocker_codes"]
    )

    blocked_queue = OperatorProposalQueue()
    blocked = _create_queue_proposal(blocked_queue)
    blocked.status = "applying"
    monkeypatch.setattr(wire_module, "SNAPSHOT_MAX_BYTES", 16)
    overflow = DecisionQueueSnapshotManager().response(
        _wire_server(blocked_queue), object(), {"contract_version": 2, "page_size": 50}
    )
    assert overflow["status"] == "degraded_overflow"
    assert overflow["overflow"]["apply_ui_disabled"] is True
    assert overflow["counts"]["pinned_omitted_count"] == 1


def test_wire_oracle_matches_production_converter_for_unicode():
    from server import convert_numpy_types
    from ws_transport import serialize_message

    payload = finalize_wire_size(
        {"type": "decision_queue", "items": [{"title_preview": "🎙️" * 50}], "wire": {}}
    )
    production_bytes = len(
        serialize_message(payload, convert_numpy_types).encode("utf-8")
    )
    assert payload["wire"]["serialized_bytes"] == production_bytes
