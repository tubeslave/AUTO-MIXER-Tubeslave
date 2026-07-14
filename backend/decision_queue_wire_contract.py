"""Bounded WebSocket contract for operator proposal review.

This module deliberately never calls ``OperatorProposal.to_dict``.  The
internal queue may retain full evidence, while the wire list is assembled from
bounded scalar metadata and immutable review bytes are fetched in chunks.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import secrets
import time
from dataclasses import dataclass, field, fields as dataclass_fields
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ws_transport import serialize_message

try:
    import numpy as _numpy
except ImportError:  # pragma: no cover - the backend normally provides NumPy
    _numpy = None


REVIEW_CONTRACT_VERSION = 2
REVIEW_SCHEMA = "automixer.operator_proposal.review_content.v2"
DECISION_QUEUE_SCHEMA = "automixer.decision_queue.v2"
WIRE_MAX_BYTES = 131_072
WIRE_PACK_TARGET_BYTES = 126_976
OUTBOUND_MAX_DEPTH = 32
OUTBOUND_MAX_NODES = 32_768
OUTBOUND_MAX_MAPPING_ITEMS = 8_192
OUTBOUND_MAX_SEQUENCE_ITEMS = 8_192
OUTBOUND_MAX_STRING_BYTES = 65_536
OUTBOUND_MAX_KEY_BYTES = 1_024
OUTBOUND_MAX_INTEGER_BITS = 4_096
DETAIL_CHUNK_BYTES = 65_536
SNAPSHOT_TTL_SECONDS = 30.0
SNAPSHOT_MAX_PER_CONNECTION = 4
SNAPSHOT_MAX_GLOBAL = 32
SNAPSHOT_MAX_BYTES = 2_097_152
SNAPSHOT_CACHE_MAX_BYTES = 8_388_608
MIXING_AGENT_PENDING_SOURCE_CAP = 128
MIXING_AGENT_HISTORY_SOURCE_CAP = 128
MIXING_AGENT_IDENTITY_MAX_DEPTH = 6
MIXING_AGENT_IDENTITY_MAX_NODES = 256
MIXING_AGENT_IDENTITY_MAX_MAPPING_ITEMS = 32
MIXING_AGENT_IDENTITY_MAX_SEQUENCE_ITEMS = 32
MIXING_AGENT_IDENTITY_MAX_STRING_CHARS = 512

LEGACY_CLIENT_ECHO_CONSUMER_HANDLERS = frozenset(
    {
        "verify_operator_proposal_apply_sequence_policy_queue_import_execution_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_execution_audit_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_audit_policy_gate_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_memory_draft_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_memory_writer_dry_run_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_memory_writer_apply_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_memory_index_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_memory_trust_review_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_memory_trust_acceptance_ledger_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_plan_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_contract_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_acceptance_ledger_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_execution_plan_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_executor_dry_run_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_executor_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_apply_audit_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_rollback_verification_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_sequence_policy_gate_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_policy_consumer_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_ordering_preview_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_queue_ordering_plan_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_queue_importer_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_preflight_plan_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_batch_preflight_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_sequence_plan_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_dry_run_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_execution_gate_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_execution_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_execution_audit_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_audit_policy_gate_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_draft_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_writer_dry_run_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_writer_apply_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_index_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_trust_review_result",
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_trust_acceptance_ledger_result",
    }
)
POST_EXECUTOR_BLOCKED_HANDLER_NAMES = frozenset(
    {
        "audit_operator_proposal_apply_sequence_execution",
        "draft_operator_proposal_apply_sequence_policy_memory",
        "gate_operator_proposal_apply_sequence_audit_policy",
    }
)
MUTATION_CAPABLE_LEGACY_HANDLER_NAMES = frozenset(
    {
        "execute_operator_proposal_apply_sequence_policy_memory_trust_promotion",
        "import_operator_proposal_apply_sequence_policy_queue_ordering_plan",
        "write_operator_proposal_apply_sequence_policy_memory",
    }
)
PIPELINE_MIGRATION_REQUIRED_HANDLER_NAMES = frozenset(
    POST_EXECUTOR_BLOCKED_HANDLER_NAMES
    | LEGACY_CLIENT_ECHO_CONSUMER_HANDLERS
    | MUTATION_CAPABLE_LEGACY_HANDLER_NAMES
)

_DIGEST_RE_PREFIX = "sha256:"
_RESULT_DETAIL_LOAD_ERROR_CODES = {
    "operator_proposal_result_detail_not_retained",
    "operator_proposal_result_digest_invalid",
    "operator_proposal_result_digest_mismatch",
    "operator_proposal_result_digest_required",
}
_PREWRITE_APPLY_REASON_CODES = frozenset(
    {
        "approval_id_required",
        "approval_required",
        "emergency_stop_active",
        "mixer_not_connected",
        "operator_mode_blocks_live_write",
        "operator_proposal_global_hazard_active",
        "operator_proposal_result_evidence_capacity_exhausted",
        "operator_proposal_result_evidence_capacity_unavailable",
        "operator_proposal_result_evidence_contract_incomplete",
        "operator_proposal_result_recovery_required",
        "proposal_apply_claim_failed",
        "proposal_expired",
        "proposal_has_no_applyable_requested_change",
        "proposal_id_required_for_apply",
        "proposal_no_longer_applyable",
        "proposal_not_found",
        "proposal_not_pending",
        "proposal_review_content_integrity_error",
        "proposal_review_contract_upgrade_required",
        "proposal_review_contract_version_invalid",
        "proposal_review_contract_version_mismatch",
        "proposal_review_contract_version_required",
        "proposal_review_digest_invalid",
        "proposal_review_digest_mismatch",
        "proposal_review_digest_required",
        "soundcheck_correction_requires_supervised_transaction",
        "supervised_apply_path_unavailable",
        "unsupported_value_type",
    }
)
_VALIDATION_APPLY_REASON_CODES = frozenset(
    {
        "cooldown",
        "cumulative_delta_limit_exceeded",
        "delta_limit_exceeded",
        "feedback_risk",
        "headroom_insufficient",
        "invalid_channel",
        "invalid_value",
        "linked_pre_read_cache_not_fresh",
        "linked_pre_read_unavailable",
        "pre_read_cache_not_fresh",
        "pre_read_unavailable",
        "stale_readback",
        "unsupported_value_type",
        "upward_move_requires_operator_authored_proposal",
        "wing_proof_direction_requires_negative_delta",
        "wing_surface_contract_mismatch",
        "wing_surface_contract_missing",
    }
)
_CLAIM_APPLY_REASON_CODES = frozenset(
    {
        "duplicate_idempotency_key",
        "idempotency_key_mismatch",
        "missing_idempotency_key",
        "operator_approval_expired",
        "proposal_apply_claim_failed",
        "proposal_identity_changed_after_claim",
        "proposal_not_pending",
    }
)
_FINALIZER_APPLY_REASON_CODES = frozenset(
    {
        "apply_terminal_evidence_incomplete",
        "apply_terminal_physical_proof_incomplete",
        "operator_apply_result_shape_invalid",
        "operator_proposal_pre_read_cancelled",
        "operator_proposal_pre_read_exception",
        "operator_proposal_apply_cancelled",
        "operator_proposal_apply_exception",
        "operator_proposal_result_persistence_failed",
        "proposal_identity_changed_after_claim",
    }
)
_LINKED_APPLY_REASON_CODES = frozenset(
    {
        "approval_id_required",
        "approval_required",
        "invalid_channel",
        "invalid_value",
        "linked_apply_failed_before_send",
        "linked_apply_failed_manual_review_required",
        "linked_apply_failed_rolled_back",
        "linked_apply_terminal_proof_incomplete",
        "linked_operator_proposal_cancelled",
        "linked_operator_proposal_exception",
        "linked_pre_read_cache_not_fresh",
        "linked_pre_read_unavailable",
        "mixer_not_connected",
        "operator_approval_expired",
        "operator_mode_blocks_live_write",
        "supervised_apply_path_unavailable",
        "unsupported_value_type",
        "upward_move_requires_operator_authored_proposal",
        "wing_proof_direction_requires_negative_delta",
        "wing_surface_contract_mismatch",
        "wing_surface_contract_missing",
    }
)
_SUPERVISED_TRANSACTION_REASON_CODES = frozenset(
    {
        "supervised_transaction_cancelled",
        "supervised_transaction_exception",
        "supervised_transaction_path_unavailable",
    }
)
_MANUAL_WRITE_REASON_CODES = frozenset(
    {
        "mixer_write_failed",
        "pre_read_cache_not_fresh",
        "pre_read_unavailable",
        "rollback_intervening_change_or_unverified_state",
        "stale_readback",
        "unsupported_value_type",
        "wing_surface_contract_mismatch",
        "wing_surface_contract_missing",
    }
)
_TRUSTED_REASON_CODES_BY_SOURCE = {
    "claim": _CLAIM_APPLY_REASON_CODES,
    "finalizer": _FINALIZER_APPLY_REASON_CODES,
    "linked_apply": _LINKED_APPLY_REASON_CODES,
    "manual_write": _MANUAL_WRITE_REASON_CODES,
    "prewrite": _PREWRITE_APPLY_REASON_CODES,
    "supervised_transaction": _SUPERVISED_TRANSACTION_REASON_CODES,
    "validation": _VALIDATION_APPLY_REASON_CODES,
}
_TRUSTED_APPLY_REASON_CODES = frozenset().union(
    *_TRUSTED_REASON_CODES_BY_SOURCE.values()
)
_AUDIO_RECHECK_NOT_REQUIRED_DISPOSITIONS = frozenset(
    {
        "not_required_by_manual_write_contract",
        "not_required_by_linked_manual_write_contract",
    }
)


class ReviewContentError(ValueError):
    """Raised when authoritative proposal content is not canonical JSON."""


class ApplyResponseProjectionError(ValueError):
    """Raised when a compact apply response cannot be projected safely."""


class OutboundMessageProjectionError(ValueError):
    """Raised only when even the fixed outbound safety envelope is invalid."""


def _exact_optional_bool(value: Any) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def _exact_string(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None


def classify_trusted_apply_reason_code(
    value: Any,
    *,
    source: str,
    final_status: str,
) -> Optional[str]:
    """Classify a reason only inside the private producer that created it."""

    if final_status in {"applied", "proof_passed"}:
        return None
    allowed = _TRUSTED_REASON_CODES_BY_SOURCE.get(source, frozenset())
    if type(value) is str and value in allowed:
        return value
    return "operator_apply_reason_available_in_result_detail"


def terminal_result_proves_verified_apply(result: Any) -> bool:
    """Require the full durable terminal proof before any green apply truth."""

    if type(result) is not dict:
        return False
    audio_recheck_satisfied = bool(
        dict.get(result, "audio_recheck_passed") is True
        or (
            "audio_recheck_passed" not in result
            and dict.get(result, "audio_recheck_required") is False
            and dict.get(result, "audio_recheck_disposition")
            in _AUDIO_RECHECK_NOT_REQUIRED_DISPOSITIONS
        )
    )
    return bool(
        dict.get(result, "status") == "applied"
        and dict.get(result, "no_console_send_proven") is False
        and dict.get(result, "physical_verified") is True
        and dict.get(result, "wal_terminal") == "committed"
        and dict.get(result, "write_state_unknown") is False
        and dict.get(result, "rollback_state_unknown") is False
        and dict.get(result, "manual_review_required") is False
        and dict.get(result, "emergency_stop_activated") is not True
        and dict.get(result, "emergency_stop_required") is not True
        and audio_recheck_satisfied
    )


def terminal_result_proves_verified_already_satisfied_apply(
    result: Any,
) -> bool:
    """Recognize a durable two-pass commit that required no console send."""

    if type(result) is not dict:
        return False
    applied = dict.get(result, "applied_operation_ids")
    rolled_back = dict.get(result, "rollback_operation_ids")
    already_satisfied = dict.get(result, "already_satisfied_operation_ids")
    identifiers_are_exact = bool(
        type(applied) is list
        and applied == []
        and type(rolled_back) is list
        and rolled_back == []
        and type(already_satisfied) is list
        and already_satisfied
        and all(type(value) is str and value for value in already_satisfied)
        and len(already_satisfied) == len(set(already_satisfied))
    )
    return bool(
        identifiers_are_exact
        and dict.get(result, "status") == "applied"
        and dict.get(result, "success") is True
        and dict.get(result, "proof_run") is False
        and dict.get(result, "restore_after_verify") is False
        and dict.get(result, "accepted") is True
        and dict.get(result, "owner_review_required") is False
        and dict.get(result, "live_authority_granted") is False
        and dict.get(result, "readiness_closure_allowed") is False
        and dict.get(result, "no_console_send_proven") is True
        and dict.get(result, "physical_verified") is True
        and dict.get(result, "proof_target_verified") is False
        and dict.get(result, "physical_state_restored") is False
        and dict.get(result, "terminal_after_console_send") is False
        and dict.get(result, "rollback_attempted") is False
        and dict.get(result, "rollback_succeeded") is False
        and dict.get(result, "rollback_status") == "not_attempted"
        and dict.get(result, "wal_terminal") == "committed"
        and dict.get(result, "audio_recheck_passed") is True
        and dict.get(result, "write_state_unknown") is False
        and dict.get(result, "rollback_state_unknown") is False
        and dict.get(result, "manual_review_required") is False
        and dict.get(result, "emergency_stop_activated") is False
        and dict.get(result, "emergency_stop_required") is False
        and dict.get(result, "fresh_analysis_required") is False
    )


def terminal_result_proves_verified_restored_proof(result: Any) -> bool:
    """Recognize only a complete, durable, restore-only proof terminal."""

    if type(result) is not dict:
        return False
    applied = dict.get(result, "applied_operation_ids")
    rolled_back = dict.get(result, "rollback_operation_ids")
    already_satisfied = dict.get(result, "already_satisfied_operation_ids", [])
    identifier_lists_are_exact = bool(
        type(applied) is list
        and type(rolled_back) is list
        and type(already_satisfied) is list
        and all(
            type(value) is str and value
            for value in (*applied, *rolled_back, *already_satisfied)
        )
        and len(applied) == len(set(applied))
        and len(rolled_back) == len(set(rolled_back))
        and len(already_satisfied) == len(set(already_satisfied))
        and set(applied).isdisjoint(already_satisfied)
    )
    mutation_proof = bool(
        identifier_lists_are_exact
        and applied
        and rolled_back
        and set(applied) == set(rolled_back)
    )
    already_satisfied_proof = bool(
        identifier_lists_are_exact
        and not applied
        and not rolled_back
        and already_satisfied
    )
    return bool(
        (mutation_proof or already_satisfied_proof)
        and dict.get(result, "status") == "proof_passed"
        and dict.get(result, "proof_run") is True
        and dict.get(result, "restore_after_verify") is True
        and dict.get(result, "capability_kind") == "restore_only_proof"
        and is_sha256_digest(dict.get(result, "operation_digest"))
        and is_sha256_digest(dict.get(result, "review_content_digest"))
        and type(dict.get(result, "proposal_id")) is str
        and bool(dict.get(result, "proposal_id"))
        and dict.get(result, "accepted") is False
        and dict.get(result, "owner_review_required") is True
        and dict.get(result, "live_authority_granted") is False
        and dict.get(result, "readiness_closure_allowed") is False
        and dict.get(result, "no_console_send_proven")
        is already_satisfied_proof
        and dict.get(result, "physical_verified") is True
        and dict.get(result, "proof_target_verified") is True
        and dict.get(result, "physical_state_restored") is True
        and dict.get(result, "terminal_after_console_send") is mutation_proof
        and dict.get(result, "rollback_attempted") is mutation_proof
        and dict.get(result, "rollback_succeeded") is mutation_proof
        and dict.get(result, "rollback_status")
        == (
            "verified"
            if mutation_proof
            else "not_required_already_satisfied"
        )
        and dict.get(result, "wal_terminal") == "proof_restored"
        and dict.get(result, "audio_recheck_passed") is True
        and dict.get(result, "write_state_unknown") is False
        and dict.get(result, "rollback_state_unknown") is False
        and dict.get(result, "manual_review_required") is False
        and dict.get(result, "emergency_stop_activated") is False
        and dict.get(result, "emergency_stop_required") is False
    )


_TRUSTED_IDENTITY_FACTORY_MARKER = object()
_TRUSTED_RECEIPT_FACTORY_MARKER = object()
_TRUSTED_PREWRITE_FACTORY_MARKER = object()
_TRUSTED_SEQUENCE_EXECUTION_FACTORY_MARKER = object()


@dataclass(frozen=True, slots=True, init=False)
class TrustedApplyIdentity:
    """Private immutable identity captured from VerifiedProposalExecution."""

    __factory_marker: object = field(repr=False, compare=False)
    proposal_id: str
    review_contract_version: int
    review_content_digest: str

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError(
            "TrustedApplyIdentity can only be created from a verified claim"
        )

    def _is_authentic(self) -> bool:
        return (
            getattr(self, "_TrustedApplyIdentity__factory_marker", None)
            is _TRUSTED_IDENTITY_FACTORY_MARKER
        )


def _freeze_verified_apply_identity(execution: Any) -> TrustedApplyIdentity:
    proposal_id = getattr(execution, "proposal_id", None)
    review_version = getattr(execution, "review_contract_version", None)
    review_digest = getattr(execution, "review_content_digest", None)
    if (
        not isinstance(proposal_id, str)
        or not proposal_id
        or isinstance(review_version, bool)
        or not isinstance(review_version, int)
        or not is_sha256_digest(review_digest)
    ):
        raise ApplyResponseProjectionError("verified_apply_identity_invalid")
    identity = object.__new__(TrustedApplyIdentity)
    object.__setattr__(
        identity,
        "_TrustedApplyIdentity__factory_marker",
        _TRUSTED_IDENTITY_FACTORY_MARKER,
    )
    object.__setattr__(identity, "proposal_id", proposal_id)
    object.__setattr__(identity, "review_contract_version", review_version)
    object.__setattr__(identity, "review_content_digest", review_digest)
    return identity


@dataclass(frozen=True, slots=True, init=False)
class TrustedPrewriteNoSendReceipt:
    """Private capability proving a refusal occurred before any send lane."""

    __factory_marker: object = field(repr=False, compare=False)
    proposal_id: Optional[str]
    review_contract_version: Optional[int]
    review_content_digest: Optional[str]
    proposal_status: str
    reason_code: str

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError(
            "TrustedPrewriteNoSendReceipt can only be created by a refusal path"
        )

    def _is_authentic(self) -> bool:
        return (
            getattr(
                self,
                "_TrustedPrewriteNoSendReceipt__factory_marker",
                None,
            )
            is _TRUSTED_PREWRITE_FACTORY_MARKER
        )


def _freeze_trusted_prewrite_no_send_receipt(
    proposal: Any,
    *,
    reason_code: Any,
) -> TrustedPrewriteNoSendReceipt:
    receipt = object.__new__(TrustedPrewriteNoSendReceipt)
    object.__setattr__(
        receipt,
        "_TrustedPrewriteNoSendReceipt__factory_marker",
        _TRUSTED_PREWRITE_FACTORY_MARKER,
    )
    proposal_id = getattr(proposal, "id", None)
    review_version = getattr(proposal, "review_contract_version", None)
    review_digest = getattr(proposal, "review_content_digest", None)
    proposal_status = getattr(proposal, "status", None)
    object.__setattr__(
        receipt,
        "proposal_id",
        proposal_id if isinstance(proposal_id, str) else None,
    )
    object.__setattr__(
        receipt,
        "review_contract_version",
        (
            review_version
            if isinstance(review_version, int)
            and not isinstance(review_version, bool)
            else None
        ),
    )
    object.__setattr__(
        receipt,
        "review_content_digest",
        review_digest if is_sha256_digest(review_digest) else None,
    )
    object.__setattr__(
        receipt,
        "proposal_status",
        proposal_status if isinstance(proposal_status, str) else "unknown",
    )
    object.__setattr__(
        receipt,
        "reason_code",
        (
            reason_code
            if type(reason_code) is str
            and reason_code in _PREWRITE_APPLY_REASON_CODES
            else "operator_apply_blocked_before_send"
        ),
    )
    return receipt


@dataclass(frozen=True, slots=True, init=False)
class TrustedApplyReceipt:
    """Private frozen terminal truth used as the only post-finalize wire source."""

    __factory_marker: object = field(repr=False, compare=False)
    proposal_id: str
    review_contract_version: Optional[int]
    review_content_digest: Optional[str]
    proposal_status: str
    final_status: str
    apply_status: str
    success: bool
    reason_code: Optional[str]
    no_console_send_proven: Optional[bool]
    physical_verified: Optional[bool]
    physical_state_restored: Optional[bool]
    terminal_after_console_send: Optional[bool]
    write_state_unknown: bool
    rollback_attempted: bool
    rollback_succeeded: Optional[bool]
    rollback_state_unknown: bool
    manual_review_required: bool
    wal_terminal: Optional[str]
    rollback_status: Optional[str]
    audio_recheck_required: Optional[bool]
    audio_recheck_passed: Optional[bool]
    audio_recheck_disposition: Optional[str]
    applied_operation_ids: Optional[Tuple[str, ...]]
    rollback_operation_ids: Optional[Tuple[str, ...]]
    already_satisfied_operation_ids: Optional[Tuple[str, ...]]
    result_digest: Optional[str]
    result_detail_available: bool
    result_evidence_pinned: bool
    emergency_stop_activated: bool
    emergency_stop_required: bool
    emergency_stop_teardown_complete: Optional[bool]
    idempotency_key_digest: Optional[str]
    idempotency_status: Optional[str]
    idempotency_replay: bool
    operator_idempotency_released: Optional[bool]
    channel: Optional[int]
    value_type: Optional[str]
    accepted: Optional[bool]
    owner_review_required: Optional[bool]
    live_authority_granted: Optional[bool]
    readiness_closure_allowed: Optional[bool]
    proof_run: Optional[bool]
    restore_after_verify: Optional[bool]
    capability_kind: Optional[str]
    operation_digest: Optional[str]
    bundle_kind: Optional[str]
    proof_receipt_id: Optional[str]
    review_preimage_sha256: Optional[str]
    before_manifest_sha256: Optional[str]
    after_manifest_sha256: Optional[str]
    verification_cut_reset_surface_sha256: Optional[str]
    verification_cut_preservation_manifest_sha256: Optional[str]
    no_outside_mutation_attempt_proven: Optional[bool]
    current_state_guaranteed_at_response: Optional[bool]
    fresh_readback_required_before_next_stage: Optional[bool]
    attended_no_touch_required_during_execution: Optional[bool]
    later_console_event_invalidates_readiness_freshness: Optional[bool]
    proof_target_verified: Optional[bool]
    parent_commit_recorded: Optional[bool]
    parent_commit_reason: Optional[str]

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError(
            "TrustedApplyReceipt can only be created by the apply finalizer"
        )

    def _is_authentic(self) -> bool:
        return (
            getattr(self, "_TrustedApplyReceipt__factory_marker", None)
            is _TRUSTED_RECEIPT_FACTORY_MARKER
        )

    @property
    def response_type(self) -> str:
        return (
            "operator_proposal_apply_blocked"
            if self.final_status == "apply_blocked"
            else "operator_proposal_apply_result"
        )

class FinalizedApplyResult(dict):
    """Dict-compatible internal result carrying a non-wire frozen receipt."""

    __slots__ = ("__trusted_receipt",)

    def __init__(
        self,
        value: Mapping[str, Any],
        *,
        trusted_receipt: Optional[TrustedApplyReceipt],
    ) -> None:
        super().__init__(value)
        self.__trusted_receipt = trusted_receipt

    @property
    def trusted_receipt(self) -> Optional[TrustedApplyReceipt]:
        return self.__trusted_receipt


@dataclass(frozen=True, slots=True, init=False)
class TrustedSequenceExecutionReceipt:
    """Private immutable aggregate built only from authentic child receipts."""

    __factory_marker: object = field(repr=False, compare=False)
    payload_bytes: bytes = field(repr=False, compare=False)

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError(
            "TrustedSequenceExecutionReceipt can only be created by the sequence executor"
        )

    def _is_authentic(self) -> bool:
        return (
            getattr(
                self,
                "_TrustedSequenceExecutionReceipt__factory_marker",
                None,
            )
            is _TRUSTED_SEQUENCE_EXECUTION_FACTORY_MARKER
        )


def _native_json(value: Any) -> Any:
    """Return strict native JSON values or fail closed.

    NumPy scalars are accepted without importing NumPy into this low-level
    module.  Arbitrary ``item`` methods are not invoked.
    """

    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReviewContentError("proposal_review_content_non_finite_number")
        return value
    value_type = type(value)
    if value_type.__module__.split(".", 1)[0] == "numpy" and hasattr(value, "item"):
        return _native_json(value.item())
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ReviewContentError("proposal_review_content_non_string_key")
            normalized[key] = _native_json(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_native_json(item) for item in value]
    raise ReviewContentError(
        f"proposal_review_content_not_json:{value_type.__module__}.{value_type.__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            _native_json(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ReviewContentError):
            raise
        raise ReviewContentError("proposal_review_content_not_canonical_json") from exc
    return text.encode("utf-8")


def sha256_digest(data: bytes) -> str:
    return f"{_DIGEST_RE_PREFIX}{hashlib.sha256(data).hexdigest()}"


def review_content_projection(proposal: Any) -> Dict[str, Any]:
    """Build the complete apply-authoritative v2 review projection."""

    return {
        "schema": REVIEW_SCHEMA,
        "proposal_id": str(getattr(proposal, "id", "")),
        "review_contract_version": REVIEW_CONTRACT_VERSION,
        "title": getattr(proposal, "title", ""),
        "target": getattr(proposal, "target", None),
        "kind": getattr(proposal, "kind", None),
        "severity": getattr(proposal, "severity", "medium"),
        "confidence": getattr(proposal, "confidence", None),
        "source": getattr(proposal, "source", "operator_proposal_queue"),
        "reason": getattr(proposal, "reason", ""),
        "created_at": getattr(proposal, "created_at", None),
        "expires_at": getattr(proposal, "expires_at", None),
        "ttl_seconds": getattr(proposal, "ttl_seconds", None),
        "mode_at_creation": getattr(proposal, "mode_at_creation", None),
        "requested_change": getattr(proposal, "requested_change", None),
        "requested_changes": getattr(proposal, "requested_changes", None),
        "raw": getattr(proposal, "raw", None),
    }


def build_review_identity(proposal: Any) -> Tuple[bytes, str]:
    content = canonical_json_bytes(review_content_projection(proposal))
    return content, sha256_digest(content)


def is_sha256_digest(value: Any) -> bool:
    raw = str(value or "")
    return (
        len(raw) == 71
        and raw.startswith(_DIGEST_RE_PREFIX)
        and all(char in "0123456789abcdef" for char in raw[7:])
    )


def _freeze_trusted_apply_receipt(
    identity: TrustedApplyIdentity,
    *,
    terminal_result: Any,
    proposal_status: Any,
    final_status: Any,
    final_success: Any,
    result_digest: Any,
    result_detail_available: Any,
    result_evidence_pinned: Any,
    reason_code: Any,
    idempotency_key: Any = None,
    operator_idempotency_released: Any = None,
) -> TrustedApplyReceipt:
    """Internal finalizer-only factory for immutable terminal receipt truth."""

    if (
        type(identity) is not TrustedApplyIdentity
        or not identity._is_authentic()
    ):
        raise ApplyResponseProjectionError("trusted_apply_identity_invalid")
    if final_status not in {
        "applied",
        "proof_passed",
        "apply_blocked",
        "apply_error",
    }:
        raise ApplyResponseProjectionError("finalized_apply_status_invalid")
    if type(terminal_result) is not dict:
        raise ApplyResponseProjectionError("finalized_apply_result_invalid")
    detail_available = result_detail_available is True
    if detail_available and not is_sha256_digest(result_digest):
        raise ApplyResponseProjectionError("finalized_apply_result_digest_invalid")
    normalized_result_digest = result_digest if detail_available else None
    evidence_pinned = result_evidence_pinned is True
    result = terminal_result
    restored_proof = terminal_result_proves_verified_restored_proof(result)
    if final_status == "proof_passed" and not restored_proof:
        raise ApplyResponseProjectionError("restored_proof_terminal_invalid")
    apply_status = _exact_string(dict.get(result, "status")) or "unknown"
    no_send = _exact_optional_bool(dict.get(result, "no_console_send_proven"))
    physical_verified = _exact_optional_bool(dict.get(result, "physical_verified"))
    physical_restored = _exact_optional_bool(
        dict.get(result, "physical_state_restored")
    )
    terminal_after_send = _exact_optional_bool(
        dict.get(result, "terminal_after_console_send")
    )
    possible_send = no_send is not True
    write_unknown = bool(
        dict.get(result, "write_state_unknown") is True
        or (
            possible_send
            and physical_verified is not True
            and physical_restored is not True
        )
    )
    rollback_results = dict.get(result, "rollback_results")
    rollback_operation_ids = dict.get(result, "rollback_operation_ids")
    rollback_attempted = bool(
        dict.get(result, "rollback_attempted") is True
        or (type(rollback_results) is list and bool(rollback_results))
        or (
            type(rollback_operation_ids) is list
            and bool(rollback_operation_ids)
        )
    )
    explicit_rollback_succeeded = _exact_optional_bool(
        dict.get(result, "rollback_succeeded")
    )
    rollback_succeeded = (
        explicit_rollback_succeeded
        if explicit_rollback_succeeded is not None
        else (
            True
            if rollback_attempted and physical_restored is True
            else None
        )
    )
    known_contained_recovery = bool(
        dict.get(result, "safe_target_retention_attempted") is True
        and dict.get(result, "contained_target_physical_verified") is True
        and dict.get(result, "audio_failure_recovery_state_known") is True
        and physical_verified is True
        and physical_restored is False
    )
    rollback_unknown = bool(
        dict.get(result, "rollback_state_unknown") is True
        or (
            rollback_attempted
            and physical_restored is False
            and not known_contained_recovery
        )
        or (
            rollback_attempted
            and rollback_succeeded is not True
            and possible_send
            and not known_contained_recovery
        )
    )
    manual_review = bool(
        dict.get(result, "manual_review_required") is True
        or write_unknown
        or rollback_unknown
    )
    emergency_stop_activated = bool(
        dict.get(result, "emergency_stop_activated") is True
    )
    emergency_stop_required = bool(
        dict.get(result, "emergency_stop_required") is True
        or (
            (write_unknown or rollback_unknown)
            and not emergency_stop_activated
        )
    )
    if evidence_pinned and no_send is True and (
        write_unknown or rollback_unknown or manual_review
    ):
        no_send = None
    idempotency = dict.get(result, "idempotency")
    idempotency_mapping = idempotency if type(idempotency) is dict else {}
    idempotency_status = (
        _exact_string(dict.get(idempotency_mapping, "reason"))
        or _exact_string(dict.get(idempotency_mapping, "status"))
        or _exact_string(dict.get(result, "idempotency_status"))
    )
    idempotency_replay = bool(
        dict.get(result, "idempotency_replay") is True
        or dict.get(result, "replayed") is True
        or dict.get(idempotency_mapping, "replay") is True
        or dict.get(idempotency_mapping, "duplicate") is True
    )
    raw_idempotency_key = (
        idempotency_key if isinstance(idempotency_key, str) else ""
    )
    channel = dict.get(result, "channel")
    if isinstance(channel, bool) or not isinstance(channel, int):
        channel = None

    def freeze_operation_ids(value: Any) -> Optional[Tuple[str, ...]]:
        # A supervised transaction is capped at 64 operations.  Preserve the
        # exact ordered identifiers in immutable receipt truth, but refuse an
        # unbounded or non-string result rather than letting it reach the wire.
        if type(value) is not list or len(value) > 64:
            return None
        if not all(
            type(operation_id) is str
            and 0 < len(operation_id.encode("utf-8")) <= 512
            for operation_id in value
        ):
            return None
        return tuple(value)

    values = {
        "proposal_id": identity.proposal_id,
        "review_contract_version": identity.review_contract_version,
        "review_content_digest": identity.review_content_digest,
        "proposal_status": (
            proposal_status
            if isinstance(proposal_status, str)
            else "unknown"
        ),
        "final_status": final_status,
        "apply_status": apply_status,
        "success": bool(
            final_success is True
            and (
                (
                    final_status == "applied"
                    and apply_status == "applied"
                    and (
                        terminal_result_proves_verified_apply(result)
                        or terminal_result_proves_verified_already_satisfied_apply(
                            result
                        )
                    )
                )
                or (
                    final_status == "proof_passed"
                    and apply_status == "proof_passed"
                    and restored_proof
                )
            )
        ),
        "reason_code": (
            reason_code
            if (
                reason_code is None
                and final_status in {"applied", "proof_passed"}
            )
            or (
                type(reason_code) is str
                and reason_code in _TRUSTED_APPLY_REASON_CODES
            )
            or reason_code == "operator_apply_reason_available_in_result_detail"
            else "operator_apply_reason_available_in_result_detail"
        ),
        "no_console_send_proven": no_send,
        "physical_verified": physical_verified,
        "physical_state_restored": physical_restored,
        "terminal_after_console_send": terminal_after_send,
        "write_state_unknown": write_unknown,
        "rollback_attempted": rollback_attempted,
        "rollback_succeeded": rollback_succeeded,
        "rollback_state_unknown": rollback_unknown,
        "manual_review_required": manual_review,
        "wal_terminal": _exact_string(dict.get(result, "wal_terminal")),
        "rollback_status": _exact_string(dict.get(result, "rollback_status")),
        "audio_recheck_required": _exact_optional_bool(
            dict.get(result, "audio_recheck_required")
        ),
        "audio_recheck_passed": _exact_optional_bool(
            dict.get(result, "audio_recheck_passed")
        ),
        "audio_recheck_disposition": _exact_string(
            dict.get(result, "audio_recheck_disposition")
        ),
        "applied_operation_ids": freeze_operation_ids(
            dict.get(result, "applied_operation_ids")
        ),
        "rollback_operation_ids": freeze_operation_ids(
            dict.get(result, "rollback_operation_ids")
        ),
        "already_satisfied_operation_ids": freeze_operation_ids(
            dict.get(result, "already_satisfied_operation_ids")
        ),
        "result_digest": normalized_result_digest,
        "result_detail_available": detail_available,
        "result_evidence_pinned": evidence_pinned,
        "emergency_stop_activated": emergency_stop_activated,
        "emergency_stop_required": emergency_stop_required,
        "emergency_stop_teardown_complete": _exact_optional_bool(
            dict.get(result, "emergency_stop_teardown_complete")
        ),
        "idempotency_key_digest": (
            sha256_digest(raw_idempotency_key.encode("utf-8"))
            if raw_idempotency_key
            else None
        ),
        "idempotency_status": idempotency_status,
        "idempotency_replay": idempotency_replay,
        "operator_idempotency_released": _exact_optional_bool(
            operator_idempotency_released
        ),
        "channel": channel,
        "value_type": _exact_string(dict.get(result, "value_type")),
        "accepted": _exact_optional_bool(dict.get(result, "accepted")),
        "owner_review_required": _exact_optional_bool(
            dict.get(result, "owner_review_required")
        ),
        "live_authority_granted": _exact_optional_bool(
            dict.get(result, "live_authority_granted")
        ),
        "readiness_closure_allowed": _exact_optional_bool(
            dict.get(result, "readiness_closure_allowed")
        ),
        "proof_run": _exact_optional_bool(dict.get(result, "proof_run")),
        "restore_after_verify": _exact_optional_bool(
            dict.get(result, "restore_after_verify")
        ),
        "capability_kind": _exact_string(dict.get(result, "capability_kind")),
        "operation_digest": (
            dict.get(result, "operation_digest")
            if is_sha256_digest(dict.get(result, "operation_digest"))
            else None
        ),
        "bundle_kind": _exact_string(dict.get(result, "bundle_kind")),
        "proof_receipt_id": _exact_string(
            dict.get(result, "proof_receipt_id")
        ),
        "review_preimage_sha256": (
            dict.get(result, "review_preimage_sha256")
            if is_sha256_digest(dict.get(result, "review_preimage_sha256"))
            else None
        ),
        "before_manifest_sha256": (
            dict.get(result, "before_manifest_sha256")
            if is_sha256_digest(dict.get(result, "before_manifest_sha256"))
            else None
        ),
        "after_manifest_sha256": (
            dict.get(result, "after_manifest_sha256")
            if is_sha256_digest(dict.get(result, "after_manifest_sha256"))
            else None
        ),
        "verification_cut_reset_surface_sha256": (
            dict.get(result, "verification_cut_reset_surface_sha256")
            if is_sha256_digest(
                dict.get(result, "verification_cut_reset_surface_sha256")
            )
            else None
        ),
        "verification_cut_preservation_manifest_sha256": (
            dict.get(
                result,
                "verification_cut_preservation_manifest_sha256",
            )
            if is_sha256_digest(
                dict.get(
                    result,
                    "verification_cut_preservation_manifest_sha256",
                )
            )
            else None
        ),
        "no_outside_mutation_attempt_proven": _exact_optional_bool(
            dict.get(result, "no_outside_mutation_attempt_proven")
        ),
        "current_state_guaranteed_at_response": _exact_optional_bool(
            dict.get(result, "current_state_guaranteed_at_response")
        ),
        "fresh_readback_required_before_next_stage": _exact_optional_bool(
            dict.get(result, "fresh_readback_required_before_next_stage")
        ),
        "attended_no_touch_required_during_execution": _exact_optional_bool(
            dict.get(result, "attended_no_touch_required_during_execution")
        ),
        "later_console_event_invalidates_readiness_freshness": (
            _exact_optional_bool(
                dict.get(
                    result,
                    "later_console_event_invalidates_readiness_freshness",
                )
            )
        ),
        "proof_target_verified": _exact_optional_bool(
            dict.get(result, "proof_target_verified")
        ),
        # Parent journal truth does not exist when the physical terminal is
        # frozen.  It is attached only by the private post-finalize binder.
        "parent_commit_recorded": None,
        "parent_commit_reason": None,
    }
    receipt = object.__new__(TrustedApplyReceipt)
    object.__setattr__(
        receipt,
        "_TrustedApplyReceipt__factory_marker",
        _TRUSTED_RECEIPT_FACTORY_MARKER,
    )
    for name, value in values.items():
        object.__setattr__(receipt, name, value)
    return receipt


def _bind_trusted_apply_parent_commit(
    receipt: TrustedApplyReceipt,
    *,
    recorded: bool,
    reason: Any,
) -> TrustedApplyReceipt:
    """Return an authentic receipt bound to post-apply parent journal truth.

    The physical terminal is already immutable at this point.  Cloning keeps
    that truth intact while preventing either the adapter result or the raw
    outbound message from forging the later parent-commit outcome.
    """

    if type(receipt) is not TrustedApplyReceipt or not receipt._is_authentic():
        raise ApplyResponseProjectionError("trusted_apply_receipt_invalid")
    if type(recorded) is not bool:
        raise ApplyResponseProjectionError("parent_commit_recorded_invalid")
    if receipt.final_status != "applied" or receipt.capability_kind not in {
        "maintenance_dsp_reset_commit_once",
        "processor_enable_commit_once",
        "soundcheck_child_commit_once",
    }:
        raise ApplyResponseProjectionError("parent_commit_binding_not_allowed")
    if receipt.parent_commit_recorded is not None:
        raise ApplyResponseProjectionError("parent_commit_already_bound")

    safe_reason = _safe_wire_reason_code(reason if reason else None)
    clone = object.__new__(TrustedApplyReceipt)
    for item in dataclass_fields(TrustedApplyReceipt):
        name = item.name
        if name == "_TrustedApplyReceipt__factory_marker":
            object.__setattr__(
                clone,
                name,
                _TRUSTED_RECEIPT_FACTORY_MARKER,
            )
        elif name == "parent_commit_recorded":
            object.__setattr__(clone, name, recorded)
        elif name == "parent_commit_reason":
            object.__setattr__(clone, name, safe_reason)
        else:
            object.__setattr__(clone, name, getattr(receipt, name))
    return clone


def utf8_preview(value: Any, max_bytes: int) -> str:
    raw = str(value or "")[:max_bytes].encode("utf-8")
    if len(raw) <= max_bytes:
        return raw.decode("utf-8")
    clipped = raw[:max_bytes]
    while clipped:
        try:
            return clipped.decode("utf-8")
        except UnicodeDecodeError:
            clipped = clipped[:-1]
    return ""


def _bounded_outbound_native_json(value: Any) -> Any:
    """Normalize one outbound payload without unbounded traversal or coercion."""

    node_count = 0
    normalized_byte_budget = 0
    ancestors: set[int] = set()

    def account_bytes(count: int) -> None:
        nonlocal normalized_byte_budget
        normalized_byte_budget += max(0, int(count))
        if normalized_byte_budget > WIRE_MAX_BYTES:
            raise OutboundMessageProjectionError(
                "outbound_message_normalized_byte_limit"
            )

    def visit(item: Any, depth: int) -> Any:
        nonlocal node_count
        node_count += 1
        account_bytes(4)
        if node_count > OUTBOUND_MAX_NODES:
            raise OutboundMessageProjectionError("outbound_message_node_limit")
        if depth > OUTBOUND_MAX_DEPTH:
            raise OutboundMessageProjectionError("outbound_message_depth_limit")
        if item is None or type(item) is bool:
            return item
        if type(item) is int:
            if item.bit_length() > OUTBOUND_MAX_INTEGER_BITS:
                raise OutboundMessageProjectionError(
                    "outbound_message_integer_limit"
                )
            account_bytes((item.bit_length() + 2) // 3 + 2)
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise OutboundMessageProjectionError(
                    "outbound_message_non_finite_number"
                )
            account_bytes(32)
            return item
        if type(item) is str:
            try:
                encoded = item.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise OutboundMessageProjectionError(
                    "outbound_message_invalid_unicode"
                ) from exc
            if len(encoded) > OUTBOUND_MAX_STRING_BYTES:
                raise OutboundMessageProjectionError(
                    "outbound_message_string_limit"
                )
            account_bytes(len(encoded) + 2)
            return item
        if _numpy is not None and isinstance(item, _numpy.generic):
            converted = _numpy.generic.item(item)
            if converted is item:
                raise OutboundMessageProjectionError(
                    "outbound_message_numpy_scalar_invalid"
                )
            return visit(converted, depth + 1)
        if _numpy is not None and isinstance(item, _numpy.ndarray):
            if item.ndim > OUTBOUND_MAX_DEPTH:
                raise OutboundMessageProjectionError(
                    "outbound_message_numpy_depth_limit"
                )
            if item.size > OUTBOUND_MAX_NODES:
                raise OutboundMessageProjectionError(
                    "outbound_message_numpy_node_limit"
                )
            if any(
                dimension > OUTBOUND_MAX_SEQUENCE_ITEMS
                for dimension in item.shape
            ):
                raise OutboundMessageProjectionError(
                    "outbound_message_numpy_sequence_limit"
                )
            return visit(_numpy.ndarray.tolist(item), depth + 1)
        if type(item) is dict:
            if len(item) > OUTBOUND_MAX_MAPPING_ITEMS:
                raise OutboundMessageProjectionError(
                    "outbound_message_mapping_limit"
                )
            identity = id(item)
            if identity in ancestors:
                raise OutboundMessageProjectionError(
                    "outbound_message_cycle"
                )
            ancestors.add(identity)
            try:
                normalized: Dict[str, Any] = {}
                for key, child in item.items():
                    if type(key) is not str:
                        raise OutboundMessageProjectionError(
                            "outbound_message_non_string_key"
                        )
                    try:
                        key_bytes = key.encode("utf-8")
                    except UnicodeEncodeError as exc:
                        raise OutboundMessageProjectionError(
                            "outbound_message_invalid_key_unicode"
                        ) from exc
                    if len(key_bytes) > OUTBOUND_MAX_KEY_BYTES:
                        raise OutboundMessageProjectionError(
                            "outbound_message_key_limit"
                        )
                    account_bytes(len(key_bytes) + 3)
                    normalized[key] = visit(child, depth + 1)
                return normalized
            finally:
                ancestors.remove(identity)
        if type(item) in {list, tuple}:
            if len(item) > OUTBOUND_MAX_SEQUENCE_ITEMS:
                raise OutboundMessageProjectionError(
                    "outbound_message_sequence_limit"
                )
            identity = id(item)
            if identity in ancestors:
                raise OutboundMessageProjectionError(
                    "outbound_message_cycle"
                )
            ancestors.add(identity)
            try:
                return [visit(child, depth + 1) for child in item]
            finally:
                ancestors.remove(identity)
        raise OutboundMessageProjectionError(
            "outbound_message_unsupported_value"
        )

    return visit(value, 0)


def _safe_outbound_type_preview(message: Any) -> Optional[str]:
    if type(message) is not dict:
        return None
    message_type = dict.get(message, "type")
    if type(message_type) is not str:
        return None
    try:
        return utf8_preview(message_type, 128)
    except (TypeError, UnicodeError, ValueError):
        return None


_CATASTROPHIC_OUTBOUND_RESPONSE_ASCII = (
    '{"type":"outbound_message_degraded","status":"error",'
    '"success":false,"reason":"outbound_message_projection_failed",'
    '"original_type":null,"outbound_message_degraded":true,'
    '"wire":{"max_bytes":131072}}'
)


def catastrophic_outbound_response_serialized() -> str:
    """Return the fixed ASCII fallback used before a close-1011 decision."""

    serialized = _CATASTROPHIC_OUTBOUND_RESPONSE_ASCII
    try:
        encoded = serialized.encode("ascii")
        parsed = json.loads(serialized)
    except (UnicodeEncodeError, json.JSONDecodeError, TypeError) as exc:
        raise OutboundMessageProjectionError(
            "catastrophic_outbound_response_invalid"
        ) from exc
    if (
        len(encoded) > WIRE_MAX_BYTES
        or type(parsed) is not dict
        or dict.get(parsed, "type") != "outbound_message_degraded"
        or dict.get(parsed, "outbound_message_degraded") is not True
    ):
        raise OutboundMessageProjectionError(
            "catastrophic_outbound_response_invalid"
        )
    return serialized


def _degraded_outbound_response_serialized(
    message: Any,
    *,
    reason: str,
) -> str:
    payload = {
        "type": "outbound_message_degraded",
        "status": "error",
        "success": False,
        "reason": reason,
        "original_type": _safe_outbound_type_preview(message),
        "outbound_message_degraded": True,
        "wire": {"max_bytes": WIRE_MAX_BYTES},
    }
    try:
        serialized = serialize_message(payload, lambda item: item)
        if len(serialized.encode("utf-8")) <= WIRE_MAX_BYTES:
            return serialized
    except (TypeError, ValueError, UnicodeError):
        pass
    return catastrophic_outbound_response_serialized()


_CP02_COMPACT_RESPONSE_TYPES = frozenset(
    {
        "operator_proposal_created",
        "operator_proposal_accepted",
        "operator_proposal_dismissed",
        "operator_proposals_imported",
        "operator_proposal_apply_preflight",
        "operator_proposal_apply_preflight_batch",
        "operator_proposal_apply_sequence_plan",
        "operator_proposal_apply_sequence_dry_run",
        "operator_proposal_apply_sequence_execution_gate",
        "operator_proposal_apply_sequence_execution_result",
        "operator_proposal_pipeline_migration_required",
    }
)
_CP02_SEQUENCE_STATE_RESPONSE_STAGES = {
    "operator_proposal_apply_preflight_batch": "batch_preflight",
    "operator_proposal_apply_sequence_plan": "sequence_plan",
    "operator_proposal_apply_sequence_dry_run": "dry_run",
    "operator_proposal_apply_sequence_execution_gate": "execution_gate",
}
_CP02_SEQUENCE_STATE_SUCCESS_STATUSES = frozenset({"ok", "ready"})


def _safe_wire_reason_code(value: Any) -> Optional[str]:
    if value is None:
        return None
    if type(value) is not str:
        return "response_reason_available_in_detail"
    if (
        0 < len(value) <= 128
        and value.isascii()
        and all(char in "abcdefghijklmnopqrstuvwxyz0123456789_:" for char in value)
    ):
        return value
    return "response_reason_available_in_detail"


def _wire_string_digest(value: Any) -> Optional[str]:
    if type(value) is not str or not value:
        return None
    try:
        return sha256_digest(value.encode("utf-8"))
    except UnicodeEncodeError:
        return None


def _compact_wire_proposal_mapping(value: Any) -> Optional[Dict[str, Any]]:
    if type(value) is not dict:
        return None
    proposal_id = dict.get(value, "id") or dict.get(value, "proposal_id")
    blockers = dict.get(value, "apply_blocker_codes")
    if type(blockers) is not list:
        blockers = dict.get(value, "apply_blockers")
    compact_blockers = []
    if type(blockers) is list:
        for blocker in blockers[:8]:
            compact_blockers.append(_safe_wire_reason_code(blocker))
    review_digest = dict.get(value, "review_content_digest")
    result_digest = dict.get(value, "result_digest")
    return {
        "id": utf8_preview(proposal_id, 256),
        "review_contract_version": (
            dict.get(value, "review_contract_version")
            if type(dict.get(value, "review_contract_version")) is int
            else None
        ),
        "review_content_digest": (
            review_digest if is_sha256_digest(review_digest) else None
        ),
        "status": utf8_preview(dict.get(value, "status"), 64),
        "expires_at": (
            utf8_preview(dict.get(value, "expires_at"), 64)
            if dict.get(value, "expires_at") is not None
            else None
        ),
        "can_apply": dict.get(value, "can_apply") is True,
        "apply_blocker_codes": compact_blockers,
        "apply_blocker_count": (
            dict.get(value, "apply_blocker_count")
            if type(dict.get(value, "apply_blocker_count")) is int
            else len(blockers) if type(blockers) is list else 0
        ),
        "result_digest": result_digest if is_sha256_digest(result_digest) else None,
        "result_detail_available": (
            dict.get(value, "result_detail_available") is True
        ),
        "result_evidence_pinned": dict.get(value, "result_evidence_pinned") is True,
    }


def _compact_wire_apply_request(value: Any) -> Optional[Dict[str, Any]]:
    if type(value) is not dict:
        return None
    review_digest = dict.get(value, "expected_review_content_digest")
    return {
        "type": (
            "apply_operator_proposal"
            if dict.get(value, "type") == "apply_operator_proposal"
            else None
        ),
        "proposal_id": utf8_preview(dict.get(value, "proposal_id"), 256),
        "approved": dict.get(value, "approved") is True,
        "expected_review_contract_version": (
            dict.get(value, "expected_review_contract_version")
            if type(dict.get(value, "expected_review_contract_version")) is int
            else None
        ),
        "expected_review_content_digest": (
            review_digest if is_sha256_digest(review_digest) else None
        ),
        "approval_id_digest": _wire_string_digest(dict.get(value, "approval_id")),
        "idempotency_key_digest": _wire_string_digest(
            dict.get(value, "idempotency_key")
        ),
    }


def _compact_sequence_state_reference(
    value: Any,
    *,
    expected_stage: str,
) -> Optional[Dict[str, Any]]:
    """Validate the authority cursor outside the lossy compact-value budget."""

    if type(value) is not dict:
        return None
    cursor = dict.get(value, "cursor")
    content_digest = dict.get(value, "content_digest")
    created_at_epoch = dict.get(value, "created_at_epoch")
    expires_at_epoch = dict.get(value, "expires_at_epoch")
    if (
        dict.get(value, "schema")
        != "automixer.operator_proposal.sequence_state.v2"
        or dict.get(value, "stage") != expected_stage
        or type(cursor) is not str
        or not 16 <= len(cursor) <= 256
        or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in cursor
        )
        or not is_sha256_digest(content_digest)
        or type(created_at_epoch) not in {int, float}
        or type(expires_at_epoch) not in {int, float}
        or not math.isfinite(float(created_at_epoch))
        or not math.isfinite(float(expires_at_epoch))
        or float(expires_at_epoch) <= float(created_at_epoch)
    ):
        return None
    return {
        "schema": "automixer.operator_proposal.sequence_state.v2",
        "cursor": cursor,
        "stage": expected_stage,
        "content_digest": content_digest,
        "created_at_epoch": created_at_epoch,
        "expires_at_epoch": expires_at_epoch,
    }


_SEQUENCE_STATE_PUBLIC_REASON_CODES = frozenset(
    {
        "sequence_state_stage_invalid",
        "sequence_state_shape_invalid",
        "sequence_state_entry_too_large",
        "sequence_state_owner_capacity_unavailable",
        "sequence_state_capacity_unavailable",
        "sequence_state_required",
        "sequence_state_cursor_required",
        "sequence_state_not_found",
        "sequence_state_expired",
        "sequence_state_owner_mismatch",
        "sequence_state_stage_mismatch",
        "sequence_state_digest_mismatch",
        "sequence_state_integrity_error",
        "sequence_state_invalid",
        "sequence_state_payload_invalid",
        "sequence_state_unavailable",
    }
)
_CP02_REASON_CODES_BY_TYPE = {
    "operator_proposal_created": frozenset(
        {
            "operator_mode_blocks_proposal_creation",
            "operator_proposal_id_ledger_capacity_exceeded",
            "operator_proposal_pending_byte_capacity_exceeded",
            "operator_proposal_pending_capacity_exceeded",
            "proposal_expired_fresh_analysis_required",
            "proposal_id_already_consumed",
            "proposal_id_already_used",
            "proposal_id_too_large",
            "proposal_review_content_non_finite_number",
            "proposal_review_content_non_string_key",
            "proposal_review_content_not_canonical_json",
            "proposal_review_content_too_large",
        }
    ),
    "operator_proposal_accepted": frozenset(
        {"proposal_not_found", "proposal_not_pending"}
    ),
    "operator_proposal_dismissed": frozenset(
        {"proposal_not_found", "proposal_not_pending"}
    ),
    "operator_proposals_imported": frozenset(
        {
            "invalid_soundcheck_manifest_identity",
            "no_soundcheck_recommendation_bundle",
            "no_soundcheck_recommendations",
            "no_actionable_safe_gain_suggestions",
            "no_safe_gain_suggestions",
            "operator_proposal_queue_unavailable",
            "safe_gain_suggestions_unavailable",
            "soundcheck_engine_superseded",
            "soundcheck_manifest_content_digest_mismatch",
            "soundcheck_manifest_revision_collision",
            "soundcheck_manifest_revision_not_monotonic",
            "soundcheck_manifest_revision_without_content_change",
            "soundcheck_manifest_same_identity_content_mismatch",
            "soundcheck_other_run_apply_in_progress",
            "soundcheck_parent_journal_recovery_failed",
            "soundcheck_parent_journal_unavailable",
            "soundcheck_previous_revision_apply_in_progress",
            "soundcheck_recommendation_bundle_unavailable",
            "soundcheck_run_superseded",
            "untrusted_supervised_soundcheck_manifest",
            "unsupported_soundcheck_manifest_schema",
        }
    ),
    "operator_proposal_apply_preflight": frozenset(
        set(_PREWRITE_APPLY_REASON_CODES) | set(_VALIDATION_APPLY_REASON_CODES)
    ),
    "operator_proposal_apply_preflight_batch": _SEQUENCE_STATE_PUBLIC_REASON_CODES,
    "operator_proposal_apply_sequence_plan": frozenset(
        _SEQUENCE_STATE_PUBLIC_REASON_CODES | {
            "invalid_preflight_result",
            "missing_preflight_results",
        }
    ),
    "operator_proposal_apply_sequence_dry_run": frozenset(
        _SEQUENCE_STATE_PUBLIC_REASON_CODES | {
            "invalid_planned_step",
            "missing_planned_apply_sequence",
        }
    ),
    "operator_proposal_apply_sequence_execution_gate": frozenset(
        _SEQUENCE_STATE_PUBLIC_REASON_CODES | {
            "invalid_dry_run_result",
            "missing_dry_run_results",
            "operator_ack_required",
        }
    ),
    "operator_proposal_apply_sequence_execution_result": frozenset(
        _SEQUENCE_STATE_PUBLIC_REASON_CODES | {
            "approved_true_required",
            "missing_ready_for_executor",
            "operator_ack_required",
            "operator_mode_blocks_live_write",
            "sequence_execution_response_projection_failed",
            "sequence_executor_blocked",
            "sequence_executor_stopped_on_block",
            "sequence_executor_stopped_on_unknown",
        }
    ),
    "operator_proposal_pipeline_migration_required": frozenset(
        {"server_owned_sequence_state_required"}
    ),
}
_CP02_ALL_ROW_REASON_CODES = frozenset(
    set().union(*_CP02_REASON_CODES_BY_TYPE.values())
    | set(_PREWRITE_APPLY_REASON_CODES)
    | set(_VALIDATION_APPLY_REASON_CODES)
    | set(_CLAIM_APPLY_REASON_CODES)
    | set(_FINALIZER_APPLY_REASON_CODES)
    | {
        "apply_receipt_missing_or_identity_mismatch",
        "apply_request_approval_id_required",
        "apply_request_approval_required",
        "apply_request_proposal_id_mismatch",
        "apply_request_type_mismatch",
        "apply_safely_blocked",
        "apply_terminal_state_unknown",
        "duplicate_idempotency_key_in_sequence",
        "execute_true_required",
        "gate_row_execution_authority_forged",
        "gate_row_missing_separate_executor_requirement",
        "gate_row_not_ready_for_executor",
        "invalid_ready_for_executor_row",
        "missing_apply_request",
        "missing_channel",
        "missing_gate_proposal_id",
        "missing_idempotency_key",
        "missing_planned_changes",
        "missing_requested_change",
        "no_ready_for_executor_rows",
        "planned_step_execution_authority_forged",
        "planned_step_live_apply_entrypoint_mismatch",
        "planned_step_missing_operator_apply_requirement",
        "preflight_has_no_normalized_change",
        "soundcheck_physical_preflight_missing",
        "soundcheck_physical_preflight_required",
    }
)
_CP02_SEQUENCE_ROW_REASON_CODES = frozenset(
    {
        "approval_id_required",
        "invalid_ready_for_executor_row",
        "missing_apply_request",
        "missing_idempotency_key",
        "preflight_has_no_normalized_change",
    }
)
_CP02_STATUS_BY_TYPE = {
    "operator_proposal_created": frozenset({"created", "exists", "blocked"}),
    "operator_proposal_accepted": frozenset({"accepted", "not_found", "blocked"}),
    "operator_proposal_dismissed": frozenset({"dismissed", "not_found", "blocked"}),
    "operator_proposals_imported": frozenset(
        {"ok", "partial", "blocked", "unavailable", "compacted"}
    ),
    "operator_proposal_apply_preflight": frozenset({"ready", "blocked"}),
    "operator_proposal_apply_preflight_batch": frozenset({"ok", "blocked"}),
    "operator_proposal_apply_sequence_plan": frozenset({"ok", "blocked"}),
    "operator_proposal_apply_sequence_dry_run": frozenset({"ok", "blocked"}),
    "operator_proposal_apply_sequence_execution_gate": frozenset(
        {"ready", "blocked"}
    ),
    "operator_proposal_apply_sequence_execution_result": frozenset(
        {"applied", "partial", "blocked", "unknown"}
    ),
    "operator_proposal_pipeline_migration_required": frozenset(
        {"migration_required"}
    ),
}
_CP02_SUCCESS_STATUS_BY_TYPE = {
    "operator_proposal_created": frozenset({"created", "exists"}),
    "operator_proposal_accepted": frozenset({"accepted"}),
    "operator_proposal_dismissed": frozenset({"dismissed"}),
    "operator_proposals_imported": frozenset({"ok", "partial", "compacted"}),
    "operator_proposal_apply_preflight": frozenset({"ready"}),
    "operator_proposal_apply_preflight_batch": frozenset({"ok"}),
    "operator_proposal_apply_sequence_plan": frozenset({"ok"}),
    "operator_proposal_apply_sequence_dry_run": frozenset({"ok"}),
    "operator_proposal_apply_sequence_execution_gate": frozenset({"ready"}),
    "operator_proposal_apply_sequence_execution_result": frozenset({"applied"}),
    "operator_proposal_pipeline_migration_required": frozenset(),
}
_CP02_SEQUENCE_ARRAY_BY_TYPE = {
    "operator_proposal_apply_preflight_batch": (
        "preflight_results",
        "blocked_results",
    ),
    "operator_proposal_apply_sequence_plan": (
        "planned_apply_sequence",
        "blocked_results",
    ),
    "operator_proposal_apply_sequence_dry_run": (
        "dry_run_results",
        "blocked_results",
    ),
    "operator_proposal_apply_sequence_execution_gate": (
        "ready_for_executor",
        "blocked_results",
    ),
    "operator_proposal_apply_sequence_execution_result": (
        "execution_results",
        "blocked_results",
    ),
}
_CP02_TOP_FIELDS_BY_TYPE = {
    "operator_proposal_created": frozenset(
        {
            "proposal_id",
            "proposal",
            "operator_mode",
            "review_content_bytes",
            "max_review_content_bytes",
            "queue_mutation",
            "proposal_status_mutation",
        }
    ),
    "operator_proposal_accepted": frozenset(
        {"proposal_id", "proposal", "queue_mutation", "proposal_status_mutation"}
    ),
    "operator_proposal_dismissed": frozenset(
        {"proposal_id", "proposal", "queue_mutation", "proposal_status_mutation"}
    ),
    "operator_proposals_imported": frozenset(
        {
            "complete",
            "source",
            "operator_mode",
            "imported_count",
            "created_count",
            "existing_count",
            "blocked_count",
            "generated_at",
            "recommendation_derivation_complete",
            "trigger",
            "trigger_state",
            "results",
            "result_count",
            "results_omitted_count",
            "results_compacted",
            "results_counts_only",
            "full_results_endpoint",
            "soundcheck_readiness_endpoint",
            "soundcheck_readiness_omitted",
            "soundcheck_readiness",
        }
    ),
    "operator_proposal_apply_preflight": frozenset(
        {
            "ready_to_apply",
            "proposal_id",
            "index",
            "batch",
            "proposal",
            "operator_mode",
            "idempotency",
            "normalized_change_count",
            "safety",
        }
    ),
    "operator_proposal_apply_preflight_batch": frozenset(
        {"generated_at", "summary", "safety", "preflight_results", "sequence_state"}
    ),
    "operator_proposal_apply_sequence_plan": frozenset(
        {
            "generated_at",
            "operator_mode",
            "summary",
            "safety",
            "planned_apply_sequence",
            "blocked_results",
            "sequence_state",
        }
    ),
    "operator_proposal_apply_sequence_dry_run": frozenset(
        {"generated_at", "summary", "safety", "dry_run_results", "sequence_state"}
    ),
    "operator_proposal_apply_sequence_execution_gate": frozenset(
        {
            "generated_at",
            "operator_mode",
            "summary",
            "safety",
            "ready_for_executor",
            "blocked_results",
            "sequence_state",
        }
    ),
    "operator_proposal_apply_sequence_execution_result": frozenset(
        {
            "generated_at",
            "operator_mode",
            "summary",
            "safety",
            "execution_results",
            "blocked_results",
            "no_console_send_proven",
            "write_state_unknown",
            "rollback_state_unknown",
            "manual_review_required",
            "emergency_stop_activated",
            "emergency_stop_required",
            "emergency_stop_teardown_complete",
            "response_degraded",
        }
    ),
    "operator_proposal_pipeline_migration_required": frozenset(
        {
            "blocked_handler",
            "no_additional_console_send_proven",
            "manual_review_required",
            "soundcheck_completion_blocked",
            "memory_mutation",
            "policy_mutation",
            "queue_mutation",
            "model_mutation",
            "live_authority_granted",
        }
    ),
}
_CP02_SUMMARY_FIELDS_BY_TYPE = {
    "operator_proposal_apply_preflight_batch": frozenset(
        {
            "selected_count", "preflight_result_count", "ready_count", "blocked_count",
            "skipped_analysis_only_count", "not_found_count", "ready_proposal_ids",
            "blocked_proposal_ids", "skipped_analysis_only_ids", "not_found_ids",
            "approved", "idempotency_key_prefix", "live_write", "apply_attempted",
            "queue_mutation", "idempotency_reserved",
        }
    ),
    "operator_proposal_apply_sequence_plan": frozenset(
        {
            "input_preflight_result_count", "ready_result_count", "blocked_result_count",
            "planned_apply_count", "plan_blocked_count", "planned_apply_proposal_ids",
            "blocked_proposal_ids", "operator_mode_allows_apply_now",
            "execution_allowed_by_plan", "live_write", "apply_attempted", "queue_mutation",
            "idempotency_reserved", "manual_console_write_called", "mutates_console_state",
            "queries_console", "live_widening_allowed", "queue_freshness_checked",
        }
    ),
    "operator_proposal_apply_sequence_dry_run": frozenset(
        {
            "input_step_count", "result_count", "would_apply_count", "blocked_count",
            "would_apply_proposal_ids", "blocked_proposal_ids", "execution_allowed_by_dry_run",
            "live_write", "apply_attempted", "queue_mutation", "proposal_status_mutation",
            "idempotency_reserved", "manual_console_write_called", "mutates_console_state",
            "queries_console", "live_widening_allowed", "queue_freshness_checked",
        }
    ),
    "operator_proposal_apply_sequence_execution_gate": frozenset(
        {
            "input_dry_run_result_count", "ready_for_executor_count", "blocked_count",
            "ready_for_executor_proposal_ids", "blocked_proposal_ids",
            "operator_mode_allows_apply_now", "operator_ack_received", "execution_allowed_by_gate",
            "requires_separate_executor", "live_write", "apply_attempted", "queue_mutation",
            "proposal_status_mutation", "idempotency_reserved", "manual_console_write_called",
            "mutates_console_state", "queries_console", "live_widening_allowed",
            "queue_freshness_checked",
        }
    ),
    "operator_proposal_apply_sequence_execution_result": frozenset(
        {
            "input_candidate_count", "attempted_count", "applied_count", "blocked_count",
            "unknown_count", "possible_console_send_count", "no_console_send_proven_count",
            "applied_proposal_ids", "blocked_proposal_ids", "operator_mode_allows_apply_now",
            "operator_ack_received", "execute_requested", "approved", "requires_gate_ready",
            "uses_existing_apply_operator_proposal", "sequence_executor_called_apply_operator_proposal",
            "live_apply_entrypoint", "live_write", "direct_live_write", "apply_attempted",
            "queue_mutation", "proposal_status_mutation", "idempotency_reserved",
            "idempotency_reserved_by_executor_directly",
            "idempotency_reservation_delegated_to_apply_operator_proposal",
            "manual_console_write_called_by_apply_operator_proposal",
            "manual_console_write_called_by_sequence_executor", "direct_manual_console_write_called",
            "mutates_console_state", "queries_console", "live_widening_allowed", "stopped_on_block",
            "manual_review_required", "queue_freshness_checked",
        }
    ),
}
_CP02_SAFETY_FIELDS_BY_TYPE = {
    "operator_proposal_apply_preflight": frozenset(
        {
            "preflight_only", "live_write", "apply_attempted", "queue_mutation",
            "proposal_status_mutation", "idempotency_reserved", "manual_console_write_called",
            "allowed_live_write_kinds", "live_apply_entrypoint",
        }
    ),
    "operator_proposal_apply_preflight_batch": frozenset(
        {
            "batch_preflight_only", "live_write", "apply_attempted", "queue_mutation",
            "proposal_status_mutation", "idempotency_reserved", "manual_console_write_called",
            "live_apply_entrypoint",
        }
    ),
    "operator_proposal_apply_sequence_plan": frozenset(
        {
            "sequence_plan_only", "preflight_results_required", "current_queue_proposal_required",
            "preflight_result_is_not_apply_authority", "requires_operator_apply",
            "execution_allowed_by_plan", "live_write", "apply_attempted", "queue_mutation",
            "proposal_status_mutation", "idempotency_reserved", "manual_console_write_called",
            "mutates_console_state", "queries_console", "live_widening_allowed",
            "live_apply_entrypoint", "allowed_live_write_kinds",
        }
    ),
    "operator_proposal_apply_sequence_dry_run": frozenset(
        {
            "dry_run_only", "sequence_plan_required", "current_queue_proposal_required",
            "does_not_call_apply_operator_proposal", "execution_allowed_by_dry_run",
            "requires_operator_apply", "live_write", "apply_attempted", "queue_mutation",
            "proposal_status_mutation", "idempotency_reserved", "manual_console_write_called",
            "mutates_console_state", "queries_console", "live_widening_allowed",
            "live_apply_entrypoint", "allowed_live_write_kinds",
        }
    ),
    "operator_proposal_apply_sequence_execution_gate": frozenset(
        {
            "execution_gate_only", "dry_run_results_required", "operator_ack_required",
            "current_queue_proposal_required", "does_not_call_apply_operator_proposal",
            "execution_allowed_by_gate", "requires_separate_executor", "live_write",
            "apply_attempted", "queue_mutation", "proposal_status_mutation",
            "idempotency_reserved", "manual_console_write_called", "mutates_console_state",
            "queries_console", "live_widening_allowed", "live_apply_entrypoint",
            "allowed_live_write_kinds",
        }
    ),
    "operator_proposal_apply_sequence_execution_result": frozenset(
        {
            "sequence_executor", "execution_gate_payload_required", "ready_for_executor_required",
            "requires_execute_true", "requires_operator_ack", "requires_approved_true",
            "requires_gate_ready", "uses_existing_apply_operator_proposal", "live_apply_entrypoint",
            "allowed_live_write_kinds", "live_write", "direct_live_write", "apply_attempted",
            "queue_mutation_by_executor_directly", "proposal_status_mutation_by_executor_directly",
            "idempotency_reserved_by_executor_directly",
            "idempotency_reservation_delegated_to_apply_operator_proposal",
            "does_not_call_manual_console_write_directly",
            "manual_console_write_called_by_sequence_executor", "direct_manual_console_write_called",
            "manual_console_write_called_by_apply_operator_proposal", "mutates_console_state",
            "manual_review_required", "queries_console", "live_widening_allowed",
        }
    ),
}
_CP02_ROW_FIELDS_BY_ARRAY = {
    "results": frozenset({"status", "success", "reason", "proposal_id", "proposal"}),
    "preflight_results": frozenset(
        {
            "type", "status", "success", "reason", "ready_to_apply", "proposal_id",
            "index", "batch", "proposal", "operator_mode", "idempotency", "safety",
            "change_count", "approval_id_digest", "idempotency_key_digest", "apply_request",
            "current_proposal",
        }
    ),
    "planned_apply_sequence": frozenset(
        {
            "planned_apply_index", "proposal_id", "approval_required", "requires_operator_apply",
            "live_apply_entrypoint", "execution_allowed_by_plan", "current_proposal",
            "change_count", "apply_request", "idempotency_key_digest", "approval_id_digest",
        }
    ),
    "dry_run_results": frozenset(
        {
            "planned_apply_index", "proposal_id", "status", "reason", "count",
            "current_proposal", "apply_request", "change_count", "idempotency_key_digest",
            "approval_id_digest",
        }
    ),
    "ready_for_executor": frozenset(
        {
            "planned_apply_index", "proposal_id", "dry_run_result_index", "status",
            "execution_allowed_by_gate", "requires_separate_executor", "current_proposal",
            "apply_request", "change_count", "idempotency_key_digest", "approval_id_digest",
        }
    ),
    "execution_results": frozenset(
        {
            "planned_apply_index", "proposal_id", "candidate_index", "status",
            "apply_attempted", "reason", "manual_review_required", "no_console_send_proven",
            "executor_emergency_stop_activated", "executor_emergency_stop_teardown_complete",
            "executor_emergency_stop_required", "apply_receipt",
        }
    ),
    "blocked_results": frozenset(
        {
            "planned_apply_index", "proposal_id", "candidate_index", "preflight_result_index",
            "dry_run_result_index", "status", "reason", "count", "preflight_status",
            "current_proposal",
        }
    ),
}
_CP02_SUMMARY_INTEGER_FIELDS = frozenset(
    {
        "input_candidate_count",
        "input_dry_run_result_count",
        "input_preflight_result_count",
        "input_step_count",
        "attempted_count",
        "applied_count",
        "blocked_count",
        "blocked_result_count",
        "no_console_send_proven_count",
        "not_found_count",
        "plan_blocked_count",
        "planned_apply_count",
        "preflight_result_count",
        "possible_console_send_count",
        "ready_count",
        "ready_for_executor_count",
        "ready_result_count",
        "result_count",
        "selected_count",
        "skipped_analysis_only_count",
        "unknown_count",
        "would_apply_count",
    }
)
_CP02_SUMMARY_BOOLEAN_FIELDS = frozenset(
    {
        "approved",
        "apply_attempted",
        "direct_live_write",
        "direct_manual_console_write_called",
        "execute_requested",
        "execution_allowed_by_dry_run",
        "execution_allowed_by_gate",
        "execution_allowed_by_plan",
        "idempotency_reserved",
        "idempotency_reserved_by_executor_directly",
        "idempotency_reservation_delegated_to_apply_operator_proposal",
        "live_widening_allowed",
        "live_write",
        "manual_console_write_called",
        "manual_console_write_called_by_apply_operator_proposal",
        "manual_console_write_called_by_sequence_executor",
        "manual_review_required",
        "mutates_console_state",
        "operator_ack_received",
        "operator_mode_allows_apply_now",
        "proposal_status_mutation",
        "queries_console",
        "queue_freshness_checked",
        "queue_mutation",
        "requires_gate_ready",
        "requires_separate_executor",
        "sequence_executor_called_apply_operator_proposal",
        "stopped_on_block",
        "uses_existing_apply_operator_proposal",
    }
)
_CP02_SUMMARY_ID_ARRAY_FIELDS = frozenset(
    {
        "applied_proposal_ids",
        "blocked_proposal_ids",
        "not_found_ids",
        "planned_apply_proposal_ids",
        "ready_for_executor_proposal_ids",
        "ready_proposal_ids",
        "skipped_analysis_only_ids",
        "would_apply_proposal_ids",
    }
)
_CP02_SAFETY_BOOLEAN_FIELDS = frozenset(
    {
        "apply_attempted",
        "batch_preflight_only",
        "current_queue_proposal_required",
        "direct_live_write",
        "direct_manual_console_write_called",
        "does_not_call_apply_operator_proposal",
        "does_not_call_manual_console_write_directly",
        "dry_run_only",
        "dry_run_results_required",
        "execution_allowed_by_dry_run",
        "execution_allowed_by_gate",
        "execution_allowed_by_plan",
        "execution_gate_only",
        "execution_gate_payload_required",
        "idempotency_reserved",
        "idempotency_reserved_by_executor_directly",
        "idempotency_reservation_delegated_to_apply_operator_proposal",
        "live_widening_allowed",
        "live_write",
        "manual_console_write_called",
        "manual_console_write_called_by_apply_operator_proposal",
        "manual_console_write_called_by_sequence_executor",
        "manual_review_required",
        "mutates_console_state",
        "operator_ack_required",
        "preflight_only",
        "preflight_result_is_not_apply_authority",
        "preflight_results_required",
        "proposal_status_mutation",
        "proposal_status_mutation_by_executor_directly",
        "queries_console",
        "queue_mutation",
        "queue_mutation_by_executor_directly",
        "ready_for_executor_required",
        "requires_approved_true",
        "requires_execute_true",
        "requires_gate_ready",
        "requires_operator_ack",
        "requires_operator_apply",
        "requires_separate_executor",
        "sequence_executor",
        "sequence_plan_required",
        "sequence_plan_only",
        "uses_existing_apply_operator_proposal",
    }
)
_CP02_SEQUENCE_ROW_SCALAR_FIELDS = frozenset(
    {
        "planned_apply_index",
        "proposal_id",
        "candidate_index",
        "preflight_result_index",
        "dry_run_result_index",
        "status",
        "type",
        "approval_required",
        "requires_operator_apply",
        "live_apply_entrypoint",
        "execution_allowed_by_plan",
        "execution_allowed_by_dry_run",
        "execution_allowed_by_gate",
        "requires_separate_executor",
        "apply_attempted",
        "reason",
        "manual_review_required",
        "no_console_send_proven",
        "executor_emergency_stop_activated",
        "executor_emergency_stop_teardown_complete",
        "executor_emergency_stop_required",
        "count",
        "preflight_status",
        "ready_to_apply",
        "batch",
        "proposal_id",
        "index",
        "success",
    }
)
_CP02_TERMINAL_RECEIPT_FIELDS = frozenset(
    {
        "proposal_id",
        "review_contract_version",
        "review_content_digest",
        "proposal_status",
        "status",
        "apply_status",
        "success",
        "no_console_send_proven",
        "physical_verified",
        "physical_state_restored",
        "terminal_after_console_send",
        "write_state_unknown",
        "rollback_attempted",
        "rollback_succeeded",
        "rollback_state_unknown",
        "manual_review_required",
        "wal_terminal",
        "rollback_status",
        "audio_recheck_required",
        "audio_recheck_passed",
        "audio_recheck_disposition",
        "result_digest",
        "result_detail_available",
        "result_evidence_pinned",
        "emergency_stop_activated",
        "emergency_stop_required",
        "emergency_stop_teardown_complete",
        "idempotency_key_digest",
        "idempotency_status",
        "idempotency_replay",
        "operator_idempotency_released",
        "channel",
        "value_type",
    }
)


def _cp02_registered_reason(value: Any, allowed: Iterable[str]) -> Optional[str]:
    if value is None:
        return None
    if type(value) is str and value in allowed:
        return value
    return "response_reason_available_in_detail"


def _cp02_id(value: Any, max_bytes: int = 256) -> Optional[str]:
    if type(value) is not str or not value:
        return None
    return utf8_preview(value, max_bytes)


def _cp02_finite_number(value: Any) -> Optional[float]:
    if type(value) not in {int, float} or type(value) is bool:
        return None
    return value if math.isfinite(float(value)) else None


def _cp02_iso_timestamp(value: Any) -> Optional[str]:
    if type(value) is not str or not 1 <= len(value) <= 64:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return value


def _cp02_compact_proposal(value: Any) -> Optional[Dict[str, Any]]:
    compact = _compact_wire_proposal_mapping(value)
    if compact is None:
        return None
    compact["apply_blocker_codes"] = [
        _cp02_registered_reason(item, _CP02_ALL_ROW_REASON_CODES)
        for item in compact.get("apply_blocker_codes", [])[:8]
    ]
    return compact


def _cp02_compact_operator_mode(value: Any) -> Optional[Dict[str, Any]]:
    if type(value) is not dict:
        return None
    capabilities = dict.get(value, "capabilities")
    compact_capabilities = None
    if type(capabilities) is dict:
        allowed = dict.get(capabilities, "allowed_live_write_kinds")
        compact_allowed = []
        if type(allowed) is list:
            compact_allowed = [
                item
                for item in allowed[:16]
                if type(item) is str
                and item in {"fader", "gain", "pan", "eq", "gate", "compressor", "send"}
            ]
        compact_capabilities = {
            "can_create_proposals": dict.get(capabilities, "can_create_proposals") is True,
            "can_apply_to_console": dict.get(capabilities, "can_apply_to_console") is True,
            "requires_approval": dict.get(capabilities, "requires_approval") is True,
            "can_autonomous_apply": dict.get(capabilities, "can_autonomous_apply") is True,
            "allowed_live_write_kinds": compact_allowed,
        }
    return {
        "mode": _cp02_id(dict.get(value, "mode"), 64),
        "connection_mode": _cp02_id(dict.get(value, "connection_mode"), 64),
        "wing_deployment": dict.get(value, "wing_deployment") is True,
        "live_write_policy": _cp02_id(dict.get(value, "live_write_policy"), 128),
        "apply_path": _cp02_id(dict.get(value, "apply_path"), 128),
        "capabilities": compact_capabilities,
    }


def _cp02_compact_idempotency(value: Any) -> Optional[Dict[str, Any]]:
    if type(value) is not dict:
        return None
    raw_key = dict.get(value, "idempotency_key") or dict.get(value, "key")
    return {
        "required": dict.get(value, "required") is True,
        "provided": dict.get(value, "provided") is True,
        "reserved": dict.get(value, "reserved") is True,
        "would_reserve_on_apply": dict.get(value, "would_reserve_on_apply") is True,
        "duplicate": dict.get(value, "duplicate") is True,
        "key_digest": _wire_string_digest(raw_key),
    }


def _cp02_compact_safety(
    value: Any,
    allowed_fields: Iterable[str],
) -> Optional[Dict[str, Any]]:
    if type(value) is not dict:
        return None
    allowed = frozenset(allowed_fields)
    result = {
        key: dict.get(value, key) is True
        for key in (_CP02_SAFETY_BOOLEAN_FIELDS & allowed)
        if key in value
    }
    entrypoint = dict.get(value, "live_apply_entrypoint")
    if "live_apply_entrypoint" in allowed and "live_apply_entrypoint" in value:
        result["live_apply_entrypoint"] = (
            "apply_operator_proposal"
            if entrypoint == "apply_operator_proposal"
            else None
        )
    allowed = dict.get(value, "allowed_live_write_kinds")
    if "allowed_live_write_kinds" in frozenset(allowed_fields) and type(allowed) is list:
        result["allowed_live_write_kinds"] = [
            item
            for item in allowed[:16]
            if type(item) is str
            and item in {"fader", "gain", "pan", "eq", "gate", "compressor", "send"}
        ]
    return result


def _cp02_compact_summary(
    value: Any,
    allowed_fields: Iterable[str],
) -> Optional[Dict[str, Any]]:
    if type(value) is not dict:
        return None
    allowed = frozenset(allowed_fields)
    result: Dict[str, Any] = {}
    for key in (_CP02_SUMMARY_INTEGER_FIELDS & allowed):
        raw = dict.get(value, key)
        if type(raw) is int and type(raw) is not bool and raw >= 0:
            result[key] = raw
    for key in (_CP02_SUMMARY_BOOLEAN_FIELDS & allowed):
        if key in value:
            result[key] = dict.get(value, key) is True
    for key in (_CP02_SUMMARY_ID_ARRAY_FIELDS & allowed):
        raw = dict.get(value, key)
        if type(raw) is list:
            projected = [
                item for item in (_cp02_id(item) for item in raw[:64]) if item
            ]
            result[key] = projected
            result[f"{key}_wire_counts"] = {
                "total_count": len(raw),
                "included_count": len(projected),
                "omitted_count": len(raw) - len(projected),
            }
    prefix = dict.get(value, "idempotency_key_prefix")
    if "idempotency_key_prefix" in allowed and prefix is not None:
        result["idempotency_key_prefix_digest"] = _wire_string_digest(prefix)
    if (
        "live_apply_entrypoint" in allowed
        and dict.get(value, "live_apply_entrypoint") == "apply_operator_proposal"
    ):
        result["live_apply_entrypoint"] = "apply_operator_proposal"
    return result


def _cp02_compact_apply_receipt(value: Any) -> Optional[Dict[str, Any]]:
    if type(value) is not dict:
        return None
    result: Dict[str, Any] = {}
    for key in _CP02_TERMINAL_RECEIPT_FIELDS:
        raw = dict.get(value, key)
        if raw is None or type(raw) in {bool, int}:
            result[key] = raw
        elif type(raw) is str:
            if key in {"review_content_digest", "result_digest"}:
                result[key] = raw if is_sha256_digest(raw) else None
            elif key == "reason":
                result[key] = _cp02_registered_reason(raw, _CP02_ALL_ROW_REASON_CODES)
            else:
                result[key] = utf8_preview(raw, 256)
    return result


def _cp02_compact_sequence_row(
    value: Any,
    *,
    allowed_fields: Iterable[str],
    reason_codes: Iterable[str],
) -> Optional[Dict[str, Any]]:
    if type(value) is not dict:
        return None
    allowed = frozenset(allowed_fields)
    result: Dict[str, Any] = {}
    for key in (_CP02_SEQUENCE_ROW_SCALAR_FIELDS & allowed):
        raw = dict.get(value, key)
        if key == "reason" and raw is not None:
            result[key] = _cp02_registered_reason(raw, reason_codes)
        elif type(raw) in {bool, int} or raw is None:
            result[key] = raw
        elif type(raw) is str:
            result[key] = utf8_preview(raw, 256)
    if "idempotency_key_digest" in allowed and "idempotency_key" in value:
        result["idempotency_key_digest"] = _wire_string_digest(
            dict.get(value, "idempotency_key")
        )
    if "approval_id_digest" in allowed and "approval_id" in value:
        result["approval_id_digest"] = _wire_string_digest(
            dict.get(value, "approval_id")
        )
    if "apply_request" in allowed:
        result["apply_request"] = _compact_wire_apply_request(
            dict.get(value, "apply_request")
        )
    if "current_proposal" in allowed:
        result["current_proposal"] = _cp02_compact_proposal(
            dict.get(value, "current_proposal")
        )
    if "proposal" in allowed:
        result["proposal"] = _cp02_compact_proposal(dict.get(value, "proposal"))
    if "operator_mode" in allowed:
        result["operator_mode"] = _cp02_compact_operator_mode(
            dict.get(value, "operator_mode")
        )
    if "idempotency" in allowed:
        result["idempotency"] = _cp02_compact_idempotency(
            dict.get(value, "idempotency")
        )
    if "safety" in allowed:
        result["safety"] = _cp02_compact_safety(
            dict.get(value, "safety"),
            _CP02_SAFETY_FIELDS_BY_TYPE["operator_proposal_apply_preflight"],
        )
    if "apply_receipt" in allowed:
        result["apply_receipt"] = _cp02_compact_apply_receipt(
            dict.get(value, "apply_receipt")
        )
    normalized = dict.get(value, "normalized_changes")
    if "change_count" in allowed and type(normalized) is list:
        result["change_count"] = len(normalized)
    elif "change_count" in allowed and type(dict.get(value, "normalized_change")) is dict:
        result["change_count"] = 1
    return {key: child for key, child in result.items() if child is not None}


def _cp02_sequence_row_omitted_field_count(
    value: Any,
    allowed_fields: Iterable[str],
) -> int:
    if type(value) is not dict:
        return 0
    allowed = frozenset(allowed_fields)
    registered = set(allowed) | {
        "idempotency_key",
        "approval_id",
        "normalized_changes",
        "normalized_change",
    }
    return max(0, len(value) - sum(1 for key in registered if key in value))


def _cp02_project_array(
    value: Any,
    *,
    message_type: str,
    array_key: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    if type(value) is not list:
        return [], {"total_count": 0, "included_count": 0, "omitted_count": 0}
    rows = []
    allowed_fields = _CP02_ROW_FIELDS_BY_ARRAY[array_key]
    reason_codes = set(_CP02_REASON_CODES_BY_TYPE[message_type])
    reason_codes.update(_CP02_SEQUENCE_ROW_REASON_CODES)
    if array_key == "results":
        reason_codes.update(_CP02_REASON_CODES_BY_TYPE["operator_proposal_created"])
    elif array_key == "preflight_results":
        reason_codes.update(_PREWRITE_APPLY_REASON_CODES)
        reason_codes.update(_VALIDATION_APPLY_REASON_CODES)
    elif array_key == "execution_results":
        reason_codes.update(
            {
                "apply_receipt_missing_or_identity_mismatch",
                "apply_safely_blocked",
                "apply_terminal_state_unknown",
            }
        )
    for item in value[:64]:
        compact = _cp02_compact_sequence_row(
            item,
            allowed_fields=allowed_fields,
            reason_codes=reason_codes,
        )
        if compact is not None:
            rows.append(compact)
    return rows, {
        "total_count": len(value),
        "included_count": len(rows),
        "omitted_count": len(value) - len(rows),
    }


def _cp02_compact_soundcheck_readiness(value: Any) -> Optional[Dict[str, Any]]:
    if type(value) is not dict:
        return None
    result = {
        "schema_version": _cp02_id(dict.get(value, "schema_version"), 128),
        "run_id": _cp02_id(dict.get(value, "run_id")),
        "plan_revision": dict.get(value, "plan_revision") if type(dict.get(value, "plan_revision")) is int else None,
        "plan_fingerprint": _cp02_id(dict.get(value, "plan_fingerprint")),
        "mix_readiness": _cp02_id(dict.get(value, "mix_readiness"), 64),
        "safety_state": _cp02_id(dict.get(value, "safety_state"), 64),
        "execution_status": _cp02_id(dict.get(value, "execution_status"), 64),
        "proposal_only": dict.get(value, "proposal_only") is True,
        "baseline_verified": dict.get(value, "baseline_verified") is True,
        "readiness_evidence_complete": dict.get(value, "readiness_evidence_complete") is True,
        "safe_core_stage_coverage_complete": dict.get(value, "safe_core_stage_coverage_complete") is True,
        "show_mix_stage_coverage_complete": dict.get(value, "show_mix_stage_coverage_complete") is True,
    }
    blockers = dict.get(value, "global_blockers")
    if type(blockers) is list:
        result["global_blocker_codes"] = [
            _cp02_registered_reason(item, _CP02_ALL_ROW_REASON_CODES)
            for item in blockers[:16]
        ]
        result["global_blocker_count"] = len(blockers)
    else:
        result["global_blocker_codes"] = []
        result["global_blocker_count"] = 0
    unresolved = dict.get(value, "unresolved_required_operation_ids")
    result["unresolved_required_count"] = (
        len(unresolved) if type(unresolved) is list else 0
    )
    count_keys_by_group = {
        "logical_counts": (
            "total",
            "applied",
            "already_satisfied",
            "replaced",
            "operator_review",
            "forbidden",
            "pending",
        ),
        "physical_step_counts": ("planned", "committed", "rolled_back"),
    }
    for key, count_keys in count_keys_by_group.items():
        raw_counts = dict.get(value, key)
        if type(raw_counts) is dict:
            result[key] = {
                count_key: count_value
                for count_key in count_keys
                if type((count_value := dict.get(raw_counts, count_key))) is int
                and type(count_value) is not bool
                and count_value >= 0
            }
    return result


def project_known_cp02_response(message: Any) -> Any:
    """Construct registered CP02 wire objects without suffix inference."""

    if type(message) is not dict:
        return message
    message_type = dict.get(message, "type")
    if message_type not in _CP02_COMPACT_RESPONSE_TYPES:
        if type(message_type) is str and message_type.startswith(
            ("operator_proposal_", "operator_proposals_")
        ) and message_type not in {
            "operator_proposal_detail",
            "operator_proposal_detail_error",
            "operator_proposal_result_detail",
            "operator_proposal_result_detail_error",
        }:
            return {
                "type": "outbound_message_degraded",
                "contract_version": 2,
                "status": "error",
                "success": False,
                "reason": "outbound_schema_not_registered",
                "original_type": utf8_preview(message_type, 128),
                "outbound_message_degraded": True,
                "wire_projection": {
                    "schema": "automixer.cp02.explicit-response.v2",
                    "omitted_field_count": max(0, len(message) - 1),
                    "omitted_item_count": 0,
                    "truncated_array_count": 0,
                },
            }
        return message
    expected_sequence_stage = _CP02_SEQUENCE_STATE_RESPONSE_STAGES.get(message_type)
    raw_sequence_state = dict.get(message, "sequence_state")
    sequence_state = None
    if expected_sequence_stage is not None and raw_sequence_state is not None:
        sequence_state = _compact_sequence_state_reference(
            raw_sequence_state,
            expected_stage=expected_sequence_stage,
        )
        if sequence_state is None:
            raise OutboundMessageProjectionError(
                "cp02_sequence_state_reference_invalid"
            )
    if (
        expected_sequence_stage is not None
        and dict.get(message, "status") in _CP02_SEQUENCE_STATE_SUCCESS_STATUSES
        and sequence_state is None
    ):
        raise OutboundMessageProjectionError("cp02_sequence_state_reference_required")
    if "wire" in message:
        raise OutboundMessageProjectionError("cp02_input_wire_truth_invalid")
    allowed_statuses = _CP02_STATUS_BY_TYPE[message_type]
    allowed_top = _CP02_TOP_FIELDS_BY_TYPE[message_type]
    raw_status = dict.get(message, "status")
    status = raw_status if type(raw_status) is str and raw_status in allowed_statuses else "blocked"
    projected: Dict[str, Any] = {
        "type": message_type,
        "contract_version": 2,
        "status": status,
        "success": bool(
            dict.get(message, "success") is True
            and status in _CP02_SUCCESS_STATUS_BY_TYPE[message_type]
        ),
        "reason": _cp02_registered_reason(
            dict.get(message, "reason"),
            _CP02_REASON_CODES_BY_TYPE[message_type],
        ),
        "no_console_send_proven": True,
    }
    if "queue_mutation" in allowed_top:
        mutation_status = {
            "operator_proposal_created": "created",
            "operator_proposal_accepted": "accepted",
            "operator_proposal_dismissed": "dismissed",
        }.get(message_type)
        projected["queue_mutation"] = status == mutation_status
    if "proposal_status_mutation" in allowed_top:
        projected["proposal_status_mutation"] = projected.get(
            "queue_mutation",
            False,
        )
    proposal = _cp02_compact_proposal(dict.get(message, "proposal"))
    if "proposal" in allowed_top and proposal is not None:
        projected["proposal"] = proposal
    proposal_id = _cp02_id(dict.get(message, "proposal_id"))
    if proposal_id is None and proposal is not None:
        proposal_id = proposal.get("id")
    if "proposal_id" in allowed_top and proposal_id is not None:
        projected["proposal_id"] = proposal_id
    operator_mode = _cp02_compact_operator_mode(dict.get(message, "operator_mode"))
    if "operator_mode" in allowed_top and operator_mode is not None:
        projected["operator_mode"] = operator_mode
    for key in (
        "complete",
        "ready_to_apply",
        "batch",
        "recommendation_derivation_complete",
        "results_compacted",
        "results_counts_only",
        "soundcheck_readiness_omitted",
    ):
        if key in allowed_top and key in message:
            projected[key] = dict.get(message, key) is True
    for key in (
        "imported_count",
        "created_count",
        "existing_count",
        "blocked_count",
        "result_count",
        "results_omitted_count",
        "review_content_bytes",
        "max_review_content_bytes",
        "index",
    ):
        raw = dict.get(message, key)
        if key in allowed_top and type(raw) is int and type(raw) is not bool and raw >= 0:
            projected[key] = raw
    generated_at = _cp02_finite_number(dict.get(message, "generated_at"))
    if generated_at is None and message_type == "operator_proposals_imported":
        generated_at = _cp02_iso_timestamp(dict.get(message, "generated_at"))
    if "generated_at" in allowed_top and generated_at is not None:
        projected["generated_at"] = generated_at
    source = dict.get(message, "source")
    if "source" in allowed_top and source in {"safe_gain_calibrator", "soundcheck_recommendation_bundle"}:
        projected["source"] = source
    trigger = dict.get(message, "trigger")
    if "trigger" in allowed_top and trigger in {
        "manual",
        "automatic",
        "auto_soundcheck_state",
    }:
        projected["trigger"] = trigger
    trigger_state = dict.get(message, "trigger_state")
    if "trigger_state" in allowed_top and trigger_state in {
        "running",
        "stopped",
        "completed",
        "pending",
        "blocked",
    }:
        projected["trigger_state"] = trigger_state
    if message_type == "operator_proposal_pipeline_migration_required":
        blocked_handler = dict.get(message, "blocked_handler")
        if (
            type(blocked_handler) is str
            and blocked_handler in PIPELINE_MIGRATION_REQUIRED_HANDLER_NAMES
        ):
            projected["blocked_handler"] = blocked_handler
        for key in (
            "no_additional_console_send_proven",
            "manual_review_required",
            "soundcheck_completion_blocked",
            "memory_mutation",
            "policy_mutation",
            "model_mutation",
            "live_authority_granted",
        ):
            projected[key] = dict.get(message, key) is True
    if message_type == "operator_proposals_imported":
        rows, counts = _cp02_project_array(
            dict.get(message, "results"),
            message_type=message_type,
            array_key="results",
        )
        projected["results"] = rows
        projected["results_wire_counts"] = counts
        readiness = _cp02_compact_soundcheck_readiness(
            dict.get(message, "soundcheck_readiness")
        )
        if readiness is not None:
            projected["soundcheck_readiness"] = readiness
        for key, fixed in (
            ("full_results_endpoint", "get_decision_queue"),
            ("soundcheck_readiness_endpoint", "get_auto_soundcheck_status"),
        ):
            if key in message:
                projected[key] = fixed
    if message_type == "operator_proposal_apply_preflight":
        projected["idempotency"] = _cp02_compact_idempotency(
            dict.get(message, "idempotency")
        )
        projected["safety"] = _cp02_compact_safety(
            dict.get(message, "safety"),
            _CP02_SAFETY_FIELDS_BY_TYPE[message_type],
        )
        normalized = dict.get(message, "normalized_changes")
        projected["normalized_change_count"] = (
            len(normalized)
            if type(normalized) is list
            else 1 if type(dict.get(message, "normalized_change")) is dict else 0
        )
    summary = _cp02_compact_summary(
        dict.get(message, "summary"),
        _CP02_SUMMARY_FIELDS_BY_TYPE.get(message_type, ()),
    )
    if summary is not None:
        projected["summary"] = summary
    safety = _cp02_compact_safety(
        dict.get(message, "safety"),
        _CP02_SAFETY_FIELDS_BY_TYPE.get(message_type, ()),
    )
    if safety is not None:
        projected["safety"] = safety
    for array_key in _CP02_SEQUENCE_ARRAY_BY_TYPE.get(message_type, ()):
        rows, counts = _cp02_project_array(
            dict.get(message, array_key),
            message_type=message_type,
            array_key=array_key,
        )
        projected[array_key] = rows
        projected[f"{array_key}_wire_counts"] = counts
    if sequence_state is not None:
        projected["sequence_state"] = sequence_state
    recognized_top = {"type", "status", "success", "reason", "wire"} | set(allowed_top)
    if message_type == "operator_proposal_apply_preflight":
        recognized_top.update({"normalized_changes", "normalized_change"})
    omitted_items = sum(
        counts["omitted_count"]
        for key in _CP02_SEQUENCE_ARRAY_BY_TYPE.get(message_type, ())
        for counts in [projected.get(f"{key}_wire_counts", {})]
        if type(counts) is dict
    )
    nested_omitted_fields = sum(
        _cp02_sequence_row_omitted_field_count(
            item,
            _CP02_ROW_FIELDS_BY_ARRAY[key],
        )
        for key in _CP02_SEQUENCE_ARRAY_BY_TYPE.get(message_type, ())
        for raw in [dict.get(message, key)]
        if type(raw) is list
        for item in raw[:64]
    )
    projected["wire_projection"] = {
        "schema": "automixer.cp02.explicit-response.v2",
        "omitted_field_count": (
            sum(1 for key in message if key not in recognized_top)
            + nested_omitted_fields
        ),
        "omitted_item_count": omitted_items,
        "truncated_array_count": sum(
            1
            for key in _CP02_SEQUENCE_ARRAY_BY_TYPE.get(message_type, ())
            if projected.get(f"{key}_wire_counts", {}).get("omitted_count", 0) > 0
        ),
    }
    return projected


def _sequence_receipt_ref_proves_applied(value: Any) -> bool:
    if type(value) is not dict:
        return False
    return bool(
        dict.get(value, "status") == "applied"
        and dict.get(value, "proposal_status") == "applied"
        and dict.get(value, "apply_status") == "applied"
        and dict.get(value, "success") is True
        and dict.get(value, "no_console_send_proven") is False
        and dict.get(value, "physical_verified") is True
        and dict.get(value, "wal_terminal") == "committed"
        and dict.get(value, "write_state_unknown") is False
        and dict.get(value, "rollback_state_unknown") is False
        and dict.get(value, "manual_review_required") is False
        and dict.get(value, "emergency_stop_activated") is False
        and dict.get(value, "emergency_stop_required") is False
        and dict.get(value, "result_detail_available") is True
        and is_sha256_digest(dict.get(value, "result_digest"))
        and (
            dict.get(value, "audio_recheck_passed") is True
            or (
                dict.get(value, "audio_recheck_required") is False
                and dict.get(value, "audio_recheck_disposition")
                in _AUDIO_RECHECK_NOT_REQUIRED_DISPOSITIONS
            )
        )
    )


def _sequence_receipt_ref_proves_safe_block(value: Any) -> bool:
    if type(value) is not dict:
        return False
    return bool(
        (
            dict.get(value, "no_console_send_proven") is True
            and dict.get(value, "write_state_unknown") is False
            and dict.get(value, "rollback_state_unknown") is False
            and dict.get(value, "manual_review_required") is False
        )
        or (
            dict.get(value, "physical_state_restored") is True
            and dict.get(value, "rollback_attempted") is True
            and dict.get(value, "rollback_succeeded") is True
            and dict.get(value, "write_state_unknown") is False
            and dict.get(value, "rollback_state_unknown") is False
            and dict.get(value, "manual_review_required") is False
            and dict.get(value, "emergency_stop_required") is False
        )
    )


def _sequence_receipt_ref_proves_known_contained_recovery(value: Any) -> bool:
    """Recognize a failed apply whose safer physical target is still known."""

    if type(value) is not dict:
        return False
    return bool(
        dict.get(value, "status") == "apply_blocked"
        and dict.get(value, "success") is False
        and dict.get(value, "no_console_send_proven") is False
        and dict.get(value, "physical_verified") is True
        and dict.get(value, "physical_state_restored") is False
        and dict.get(value, "write_state_unknown") is False
        and dict.get(value, "rollback_state_unknown") is False
        and dict.get(value, "manual_review_required") is True
        and dict.get(value, "wal_terminal") == "manual_review_required"
        and dict.get(value, "rollback_status")
        == "known_safe_target_retained"
        and dict.get(value, "emergency_stop_activated") is True
        and dict.get(value, "emergency_stop_required") is False
        and dict.get(value, "result_detail_available") is True
        and is_sha256_digest(dict.get(value, "result_digest"))
    )


def _sequence_execution_unknown_payload(message: Any) -> Dict[str, Any]:
    summary = dict.get(message, "summary") if type(message) is dict else None

    def safe_count(key: str) -> int:
        value = dict.get(summary, key) if type(summary) is dict else None
        return value if type(value) is int and type(value) is not bool and value >= 0 else 0

    return {
        "type": "operator_proposal_apply_sequence_execution_result",
        "contract_version": 2,
        "status": "unknown",
        "success": False,
        "reason": "sequence_execution_response_projection_failed",
        "no_console_send_proven": None,
        "write_state_unknown": True,
        "rollback_state_unknown": True,
        "manual_review_required": True,
        "emergency_stop_activated": None,
        "emergency_stop_required": True,
        "emergency_stop_teardown_complete": None,
        "response_degraded": True,
        "summary": {
            "attempted_count": safe_count("attempted_count"),
            "applied_count": safe_count("applied_count"),
            "unknown_count": max(1, safe_count("unknown_count")),
            "possible_console_send_count": max(
                1,
                safe_count("possible_console_send_count"),
            ),
        },
        "execution_results": [],
        "blocked_results": [],
        "wire_projection": {
            "schema": "automixer.cp02.sequence-execution-receipt.v2",
            "omitted_field_count": 0,
            "omitted_item_count": 0,
            "truncated_array_count": 0,
        },
    }


def _freeze_trusted_sequence_execution_receipt(
    message: Any,
    *,
    child_receipts: Sequence[Any],
    expected_candidate_count: int,
) -> TrustedSequenceExecutionReceipt:
    """Bind aggregate executor truth to authentic finalized child receipts."""

    if (
        type(message) is not dict
        or dict.get(message, "type")
        != "operator_proposal_apply_sequence_execution_result"
        or type(child_receipts) not in {list, tuple}
        or type(expected_candidate_count) is not int
        or type(expected_candidate_count) is bool
        or expected_candidate_count < 0
        or expected_candidate_count > 64
    ):
        raise ApplyResponseProjectionError("sequence_execution_receipt_input_invalid")
    projected = project_known_cp02_response(message)
    if type(projected) is not dict:
        raise ApplyResponseProjectionError("sequence_execution_projection_invalid")
    raw_rows = dict.get(message, "execution_results")
    if type(raw_rows) is not list or len(raw_rows) > 64:
        raise ApplyResponseProjectionError("sequence_execution_rows_invalid")
    references = [
        compact_trusted_apply_receipt_reference(receipt)
        for receipt in child_receipts
    ]
    receipt_set_complete = bool(
        len(raw_rows) == len(references)
        and all(type(reference) is dict for reference in references)
    )
    execution_rows: List[Dict[str, Any]] = []
    applied_count = 0
    safely_blocked_count = 0
    known_contained_review_count = 0
    unknown_count = 0
    no_send_count = 0
    possible_send_count = 0
    for index, raw_row in enumerate(raw_rows):
        row = _cp02_compact_sequence_row(
            raw_row,
            allowed_fields=_CP02_ROW_FIELDS_BY_ARRAY["execution_results"],
            reason_codes=(
                set(
                    _CP02_REASON_CODES_BY_TYPE[
                        "operator_proposal_apply_sequence_execution_result"
                    ]
                )
                | {
                    "apply_receipt_missing_or_identity_mismatch",
                    "apply_safely_blocked",
                    "apply_known_contained_manual_review",
                    "apply_terminal_state_unknown",
                }
            ),
        ) or {}
        reference = references[index] if index < len(references) else None
        raw_apply_receipt = (
            dict.get(raw_row, "apply_receipt")
            if type(raw_row) is dict
            else None
        )
        identity_matches = bool(
            type(reference) is dict
            and reference.get("proposal_id") == row.get("proposal_id")
            and type(raw_apply_receipt) is dict
            and raw_apply_receipt.get("proposal_id")
            == reference.get("proposal_id")
            and raw_apply_receipt.get("review_contract_version")
            == reference.get("review_contract_version")
            and raw_apply_receipt.get("review_content_digest")
            == reference.get("review_content_digest")
        )
        authentic_reference = reference if identity_matches else None
        if authentic_reference is not None:
            row["apply_receipt"] = authentic_reference
        else:
            row.pop("apply_receipt", None)
        handler_status = dict.get(raw_row, "status") if type(raw_row) is dict else None
        handler_terminal_unknown = bool(
            handler_status == "unknown"
            or (type(raw_row) is dict and dict.get(raw_row, "manual_review_required") is True)
            or (
                type(raw_row) is dict
                and dict.get(raw_row, "executor_emergency_stop_activated") is True
            )
            or (
                type(raw_row) is dict
                and dict.get(raw_row, "executor_emergency_stop_required") is True
            )
        )
        if (
            not handler_terminal_unknown
            and handler_status == "applied"
            and authentic_reference is not None
            and _sequence_receipt_ref_proves_applied(authentic_reference)
        ):
            row_status = "applied"
            row_reason = None
            applied_count += 1
        elif (
            not handler_terminal_unknown
            and handler_status == "blocked"
            and authentic_reference is not None
            and _sequence_receipt_ref_proves_safe_block(authentic_reference)
        ):
            row_status = "blocked"
            row_reason = "apply_safely_blocked"
            safely_blocked_count += 1
        elif (
            handler_status in {"blocked", "unknown"}
            and authentic_reference is not None
            and _sequence_receipt_ref_proves_known_contained_recovery(
                authentic_reference
            )
        ):
            row_status = "blocked"
            row_reason = "apply_known_contained_manual_review"
            known_contained_review_count += 1
        else:
            row_status = "unknown"
            row_reason = "apply_receipt_missing_or_identity_mismatch"
            unknown_count += 1
        row["status"] = row_status
        if row_reason is None:
            row.pop("reason", None)
        else:
            row["reason"] = row_reason
        row["apply_attempted"] = True
        if authentic_reference is None or authentic_reference.get(
            "no_console_send_proven"
        ) is not True:
            possible_send_count += 1
        else:
            no_send_count += 1
        execution_rows.append(row)
    if len(references) != len(raw_rows):
        unknown_count += abs(len(references) - len(raw_rows)) or 1
        possible_send_count += abs(len(references) - len(raw_rows)) or 1
    all_expected_applied = bool(
        expected_candidate_count > 0
        and receipt_set_complete
        and len(execution_rows) == expected_candidate_count
        and applied_count == expected_candidate_count
        and unknown_count == 0
        and safely_blocked_count == 0
    )
    if not execution_rows:
        final_status = "blocked"
        final_reason = _cp02_registered_reason(
            dict.get(message, "reason"),
            _CP02_REASON_CODES_BY_TYPE[
                "operator_proposal_apply_sequence_execution_result"
            ],
        )
        no_send: Optional[bool] = True
    elif all_expected_applied:
        final_status = "applied"
        final_reason = None
        no_send = False
    elif unknown_count:
        final_status = "partial" if applied_count else "unknown"
        final_reason = "sequence_executor_stopped_on_unknown"
        no_send = None
    else:
        final_status = "partial" if applied_count else "blocked"
        final_reason = "sequence_executor_stopped_on_block"
        no_send = True if possible_send_count == 0 else False
    aggregate_unknown = unknown_count > 0
    aggregate_manual_review = bool(
        aggregate_unknown or known_contained_review_count
    )
    emergency_activated = any(
        (
            type(reference) is dict
            and reference.get("emergency_stop_activated") is True
        )
        or (
            type(raw_row) is dict
            and dict.get(raw_row, "executor_emergency_stop_activated") is True
        )
        for reference, raw_row in zip(references, raw_rows)
    )
    emergency_teardown_values = [
        reference.get("emergency_stop_teardown_complete")
        for reference in references
        if type(reference) is dict
        and reference.get("emergency_stop_teardown_complete") is not None
    ]
    emergency_teardown_values.extend(
        dict.get(raw_row, "executor_emergency_stop_teardown_complete")
        for raw_row in raw_rows
        if type(raw_row) is dict
        and dict.get(raw_row, "executor_emergency_stop_teardown_complete")
        is not None
    )
    emergency_required = bool(
        aggregate_unknown
        or any(
            type(reference) is dict
            and reference.get("emergency_stop_required") is True
            for reference in references
        )
        or any(
            type(raw_row) is dict
            and dict.get(raw_row, "executor_emergency_stop_required") is True
            for raw_row in raw_rows
        )
    )
    summary = projected.get("summary")
    if type(summary) is not dict:
        summary = {}
    summary.update(
        {
            "input_candidate_count": expected_candidate_count,
            "attempted_count": len(execution_rows),
            "applied_count": applied_count,
            "blocked_count": safely_blocked_count
            + len(projected.get("blocked_results") or []),
            "unknown_count": unknown_count,
            "possible_console_send_count": possible_send_count,
            "no_console_send_proven_count": no_send_count,
            "applied_proposal_ids": [
                row["proposal_id"]
                for row in execution_rows
                if row.get("status") == "applied" and row.get("proposal_id")
            ],
            "blocked_proposal_ids": [
                row["proposal_id"]
                for row in [
                    *execution_rows,
                    *(projected.get("blocked_results") or []),
                ]
                if row.get("status") == "blocked" and row.get("proposal_id")
            ],
            "apply_attempted": bool(execution_rows),
            "live_write": possible_send_count > 0,
            "direct_live_write": False,
            "queue_mutation": bool(execution_rows),
            "proposal_status_mutation": bool(execution_rows),
            "idempotency_reserved": bool(execution_rows),
            "idempotency_reserved_by_executor_directly": False,
            "idempotency_reservation_delegated_to_apply_operator_proposal": bool(
                execution_rows
            ),
            "manual_console_write_called_by_apply_operator_proposal": (
                possible_send_count > 0
            ),
            "manual_console_write_called_by_sequence_executor": False,
            "direct_manual_console_write_called": False,
            "mutates_console_state": possible_send_count > 0,
            "stopped_on_block": not all_expected_applied and bool(execution_rows),
            "manual_review_required": aggregate_manual_review,
        }
    )
    safety = projected.get("safety")
    if type(safety) is not dict:
        safety = {}
    safety.update(
        {
            "sequence_executor": True,
            "execution_gate_payload_required": True,
            "ready_for_executor_required": True,
            "requires_execute_true": True,
            "requires_operator_ack": True,
            "requires_approved_true": True,
            "requires_gate_ready": True,
            "uses_existing_apply_operator_proposal": True,
            "live_apply_entrypoint": "apply_operator_proposal",
            "live_write": possible_send_count > 0,
            "direct_live_write": False,
            "apply_attempted": bool(execution_rows),
            "queue_mutation_by_executor_directly": False,
            "proposal_status_mutation_by_executor_directly": False,
            "idempotency_reserved_by_executor_directly": False,
            "idempotency_reservation_delegated_to_apply_operator_proposal": bool(
                execution_rows
            ),
            "does_not_call_manual_console_write_directly": True,
            "manual_console_write_called_by_sequence_executor": False,
            "direct_manual_console_write_called": False,
            "manual_console_write_called_by_apply_operator_proposal": (
                possible_send_count > 0
            ),
            "mutates_console_state": possible_send_count > 0,
            "manual_review_required": aggregate_manual_review,
            "queries_console": False,
            "live_widening_allowed": False,
        }
    )
    projected.update(
        {
            "status": final_status,
            "success": all_expected_applied,
            "reason": final_reason,
            "no_console_send_proven": no_send,
            "write_state_unknown": aggregate_unknown,
            "rollback_state_unknown": aggregate_unknown,
            "manual_review_required": aggregate_manual_review,
            "emergency_stop_activated": emergency_activated,
            "emergency_stop_required": emergency_required,
            "emergency_stop_teardown_complete": (
                all(value is True for value in emergency_teardown_values)
                if emergency_teardown_values
                else None
            ),
            "response_degraded": False,
            "summary": summary,
            "safety": safety,
            "execution_results": execution_rows,
        }
    )
    projected["wire_projection"] = {
        "schema": "automixer.cp02.sequence-execution-receipt.v2",
        "omitted_field_count": projected.get("wire_projection", {}).get(
            "omitted_field_count",
            0,
        ),
        "omitted_item_count": projected.get("wire_projection", {}).get(
            "omitted_item_count",
            0,
        ),
        "truncated_array_count": projected.get("wire_projection", {}).get(
            "truncated_array_count",
            0,
        ),
    }
    payload_bytes = canonical_json_bytes(projected)
    if len(payload_bytes) > WIRE_MAX_BYTES:
        raise ApplyResponseProjectionError("sequence_execution_receipt_too_large")
    receipt = object.__new__(TrustedSequenceExecutionReceipt)
    object.__setattr__(
        receipt,
        "_TrustedSequenceExecutionReceipt__factory_marker",
        _TRUSTED_SEQUENCE_EXECUTION_FACTORY_MARKER,
    )
    object.__setattr__(receipt, "payload_bytes", payload_bytes)
    return receipt


def project_sequence_execution_response(
    message: Any,
    trusted_receipt: Any,
) -> str:
    if (
        type(trusted_receipt) is TrustedSequenceExecutionReceipt
        and trusted_receipt._is_authentic()
    ):
        try:
            payload = json.loads(trusted_receipt.payload_bytes.decode("utf-8"))
            normalized = _bounded_outbound_native_json(payload)
            serialized = serialize_message(normalized, lambda item: item)
            if len(serialized.encode("utf-8")) <= WIRE_MAX_BYTES:
                return serialized
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return serialize_message(
        _sequence_execution_unknown_payload(message),
        lambda item: item,
    )


def catastrophic_sequence_execution_response_serialized() -> str:
    return serialize_message(
        _sequence_execution_unknown_payload({}),
        lambda item: item,
    )


def serialize_bounded_outbound(message: Any) -> str:
    """Strictly normalize and serialize one exact in-budget object or raise."""

    normalized = _bounded_outbound_native_json(
        project_known_cp02_response(message)
    )
    if type(normalized) is not dict:
        raise OutboundMessageProjectionError(
            "outbound_message_top_level_not_object"
        )
    serialized = serialize_message(normalized, lambda item: item)
    serialized_bytes = len(serialized.encode("utf-8"))
    if serialized_bytes > WIRE_MAX_BYTES:
        raise OutboundMessageProjectionError("outbound_message_too_large")
    wire = normalized.get("wire")
    if type(wire) is dict:
        declared_max = wire.get("max_bytes")
        if declared_max is not None and (
            type(declared_max) is not int
            or declared_max != WIRE_MAX_BYTES
        ):
            raise OutboundMessageProjectionError(
                "outbound_message_wire_max_mismatch"
            )
        declared_bytes = wire.get("serialized_bytes")
        if declared_bytes is not None and (
            type(declared_bytes) is not int
            or declared_bytes != serialized_bytes
        ):
            raise OutboundMessageProjectionError(
                "outbound_message_wire_size_mismatch"
            )
    return serialized


def project_outbound_message(
    message: Any,
    *,
    trusted_receipt: Any = None,
) -> str:
    """Project every UI WebSocket response to one exact bounded JSON string."""

    message_type = dict.get(message, "type") if type(message) is dict else None
    if message_type == "operator_proposal_apply_sequence_execution_result":
        return project_sequence_execution_response(message, trusted_receipt)
    if trusted_receipt is not None or message_type in {
        "operator_proposal_apply_result",
        "operator_proposal_apply_blocked",
    }:
        return project_operator_apply_response(message, trusted_receipt)
    try:
        return serialize_bounded_outbound(message)
    except (
        OutboundMessageProjectionError,
        RecursionError,
        TypeError,
        ValueError,
        UnicodeError,
    ) as exc:
        reason = str(exc)
        if reason not in {
            "outbound_message_too_large",
            "outbound_message_wire_max_mismatch",
            "outbound_message_wire_size_mismatch",
        }:
            reason = "outbound_message_invalid"
        return _degraded_outbound_response_serialized(
            message,
            reason=reason,
        )


@dataclass(frozen=True)
class _SequenceStateEntry:
    owner: Any
    stage: str
    content: bytes
    content_digest: str
    created_at_epoch: float
    expires_at_epoch: float


class OperatorProposalSequenceStateManager:
    """Short-lived owner-bound storage for full sequence authority state."""

    _ALLOWED_STAGES = frozenset(
        {"batch_preflight", "sequence_plan", "dry_run", "execution_gate"}
    )

    def __init__(
        self,
        *,
        ttl_seconds: float = 30.0,
        max_entry_bytes: int = 4_194_304,
        max_total_bytes: int = 16_777_216,
        max_entries: int = 32,
        max_entries_per_owner: int = 4,
        clock=time.time,
    ) -> None:
        self.ttl_seconds = float(ttl_seconds)
        self.max_entry_bytes = int(max_entry_bytes)
        self.max_total_bytes = int(max_total_bytes)
        self.max_entries = int(max_entries)
        self.max_entries_per_owner = int(max_entries_per_owner)
        self._clock = clock
        self._entries: Dict[str, _SequenceStateEntry] = {}
        self._total_bytes = 0

    def _drop(self, cursor: str) -> None:
        entry = self._entries.pop(cursor, None)
        if entry is not None:
            self._total_bytes = max(0, self._total_bytes - len(entry.content))

    def _prune_expired(self, now: float) -> None:
        for cursor, entry in tuple(self._entries.items()):
            if now >= entry.expires_at_epoch:
                self._drop(cursor)

    def store(self, owner: Any, stage: str, payload: Any) -> Dict[str, Any]:
        if stage not in self._ALLOWED_STAGES:
            return {"stored": False, "reason": "sequence_state_stage_invalid"}
        try:
            content = canonical_json_bytes(payload)
        except Exception:
            return {"stored": False, "reason": "sequence_state_shape_invalid"}
        if len(content) > self.max_entry_bytes:
            return {"stored": False, "reason": "sequence_state_entry_too_large"}
        now = float(self._clock())
        self._prune_expired(now)
        owner_count = sum(
            1 for entry in self._entries.values() if entry.owner is owner
        )
        if owner_count >= self.max_entries_per_owner:
            return {
                "stored": False,
                "reason": "sequence_state_owner_capacity_unavailable",
            }
        if (
            len(self._entries) >= self.max_entries
            or self._total_bytes + len(content) > self.max_total_bytes
        ):
            return {"stored": False, "reason": "sequence_state_capacity_unavailable"}
        cursor = secrets.token_urlsafe(24)
        while cursor in self._entries:
            cursor = secrets.token_urlsafe(24)
        digest = sha256_digest(content)
        expires_at = now + self.ttl_seconds
        self._entries[cursor] = _SequenceStateEntry(
            owner=owner,
            stage=stage,
            content=content,
            content_digest=digest,
            created_at_epoch=now,
            expires_at_epoch=expires_at,
        )
        self._total_bytes += len(content)
        return {
            "stored": True,
            "reference": {
                "schema": "automixer.operator_proposal.sequence_state.v2",
                "cursor": cursor,
                "stage": stage,
                "content_digest": digest,
                "created_at_epoch": now,
                "expires_at_epoch": expires_at,
            },
        }

    def resolve(
        self,
        owner: Any,
        reference: Any,
        *,
        expected_stage: str,
        consume: bool = True,
    ) -> Dict[str, Any]:
        if type(reference) is not dict:
            return {"resolved": False, "reason": "sequence_state_required"}
        cursor = dict.get(reference, "cursor")
        content_digest = dict.get(reference, "content_digest")
        if type(cursor) is not str or not cursor:
            return {"resolved": False, "reason": "sequence_state_cursor_required"}
        entry = self._entries.get(cursor)
        if entry is None:
            return {"resolved": False, "reason": "sequence_state_not_found"}
        now = float(self._clock())
        if now >= entry.expires_at_epoch:
            self._drop(cursor)
            return {"resolved": False, "reason": "sequence_state_expired"}
        if entry.owner is not owner:
            return {"resolved": False, "reason": "sequence_state_owner_mismatch"}
        if entry.stage != expected_stage:
            return {"resolved": False, "reason": "sequence_state_stage_mismatch"}
        if content_digest != entry.content_digest:
            return {"resolved": False, "reason": "sequence_state_digest_mismatch"}
        try:
            payload = json.loads(entry.content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._drop(cursor)
            return {"resolved": False, "reason": "sequence_state_integrity_error"}
        if consume:
            self._drop(cursor)
        return {
            "resolved": True,
            "payload": payload,
            "content_digest": entry.content_digest,
        }

    def release_owner(self, owner: Any) -> None:
        for cursor, entry in tuple(self._entries.items()):
            if entry.owner is owner:
                self._drop(cursor)


def _iso_or_none(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    return utf8_preview(value, 64)


def _display_target(value: Any) -> str:
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return utf8_preview(value, 256)
    try:
        return utf8_preview(
            json.dumps(_native_json(value), ensure_ascii=False, sort_keys=True),
            256,
        )
    except (TypeError, ValueError):
        return "unavailable"


def compact_proposal_row(proposal: Any) -> Dict[str, Any]:
    apply_state = proposal.apply_state()
    blockers = [utf8_preview(item, 128) for item in apply_state["apply_blockers"]]
    status = str(getattr(proposal, "status", "unknown") or "unknown")
    severity = str(getattr(proposal, "severity", "medium") or "medium").lower()
    stored_version = getattr(proposal, "review_contract_version", None)
    stored_digest = getattr(proposal, "review_content_digest", None)
    review_status = "bound" if stored_version == 2 and is_sha256_digest(stored_digest) else "upgrade_required"
    if review_status == "bound":
        try:
            current_bytes, current_digest = build_review_identity(proposal)
            review_integrity = bool(
                current_digest == stored_digest
                and current_bytes == getattr(proposal, "review_content_bytes", b"")
            )
        except ReviewContentError:
            review_integrity = False
        if not review_integrity:
            review_status = "integrity_error"
    can_apply = bool(apply_state["can_apply"]) and review_status == "bound"
    if review_status != "bound" and "proposal_review_contract_upgrade_required" not in blockers:
        blockers.append(
            "proposal_review_content_integrity_error"
            if review_status == "integrity_error"
            else "proposal_review_contract_upgrade_required"
        )
    blocker_count = len(blockers)
    blockers = blockers[:8]
    pending = status in {"pending", "applying"}
    result = getattr(proposal, "result", None)
    terminal_hazard = bool(
        isinstance(result, dict)
        and (
            result.get("write_state_unknown") is True
            or result.get("manual_review_required") is True
            or result.get("rollback_state_unknown") is True
            or (
                result.get("physical_state_restored") is False
                and not (
                    terminal_result_proves_verified_apply(result)
                    or terminal_result_proves_verified_already_satisfied_apply(
                        result
                    )
                )
            )
        )
    )
    if terminal_hazard:
        hazard_scope = "global"
        priority = "global_hazard"
    elif pending and not can_apply:
        hazard_scope = "proposal_local"
        priority = "blocked"
    elif pending and severity in {"critical", "high"}:
        hazard_scope = "none"
        priority = "high"
    elif pending:
        hazard_scope = "none"
        priority = "ordinary"
    else:
        hazard_scope = "none"
        priority = "history"
    return {
        "id": utf8_preview(getattr(proposal, "id", ""), 256),
        "entity_kind": "operator_proposal",
        "queue_section": "pending" if pending else "history",
        "title_preview": utf8_preview(getattr(proposal, "title", ""), 512),
        "reason_preview": utf8_preview(getattr(proposal, "reason", ""), 1_024),
        "display_target": _display_target(getattr(proposal, "target", None)),
        "kind": utf8_preview(getattr(proposal, "kind", ""), 128),
        "severity": utf8_preview(severity, 32),
        "status": utf8_preview(status, 32),
        "source": utf8_preview(getattr(proposal, "source", ""), 128),
        "created_at": _iso_or_none(getattr(proposal, "created_at", None)),
        "updated_at": _iso_or_none(getattr(proposal, "updated_at", None)),
        "expires_at": _iso_or_none(getattr(proposal, "expires_at", None)),
        "can_apply": can_apply,
        "requires_approval": can_apply,
        "apply_blocker_codes": blockers,
        "apply_blocker_count": blocker_count,
        "linked_change_count": len(getattr(proposal, "requested_changes", None) or []),
        "review_contract_version": stored_version if isinstance(stored_version, int) else None,
        "review_contract_status": review_status,
        "review_content_digest": stored_digest if is_sha256_digest(stored_digest) else None,
        "detail_available": bool(getattr(proposal, "review_content_bytes", None)),
        "hazard_scope": hazard_scope,
        "priority_class": priority,
        "analysis_only": False,
        "import_required": False,
    }


def _safe_action_text(value: Any, max_bytes: int) -> str:
    if type(value) is str:
        return utf8_preview(value, max_bytes)
    if type(value) is bool or value is None:
        return ""
    if type(value) is int:
        return utf8_preview(str(value), max_bytes)
    if type(value) is float and math.isfinite(value):
        return utf8_preview(str(value), max_bytes)
    return ""


def _bounded_mixing_agent_identity_value(value: Any) -> Any:
    """Return a deterministic, fixed-work identity projection.

    MixingAgent output is analysis-only and not an authority surface.  This
    projection therefore refuses arbitrary Mapping implementations and never
    hashes an unbounded action body merely to render the queue.
    """

    state = {"nodes": 0, "ancestors": set()}

    def visit(item: Any, depth: int) -> Any:
        state["nodes"] += 1
        if state["nodes"] > MIXING_AGENT_IDENTITY_MAX_NODES:
            return {"$truncated": "node_limit"}
        if depth > MIXING_AGENT_IDENTITY_MAX_DEPTH:
            return {"$truncated": "depth_limit"}
        if item is None or type(item) is bool:
            return item
        if type(item) is int:
            if item.bit_length() > 256:
                return {"$integer_bits": item.bit_length()}
            return item
        if type(item) is float:
            return item if math.isfinite(item) else {"$number": "non_finite"}
        if type(item) is str:
            preview = item[:MIXING_AGENT_IDENTITY_MAX_STRING_CHARS]
            return {
                "$string_preview": preview,
                "$string_chars": len(item),
                "$truncated": len(item) > len(preview),
            }
        if type(item) is dict:
            identity = id(item)
            if identity in state["ancestors"]:
                return {"$truncated": "cycle"}
            state["ancestors"].add(identity)
            try:
                projected: Dict[str, Any] = {}
                included = 0
                for raw_index, (key, child) in enumerate(item.items()):
                    if raw_index >= MIXING_AGENT_IDENTITY_MAX_MAPPING_ITEMS:
                        break
                    if type(key) is not str:
                        continue
                    projected[key[:128]] = visit(child, depth + 1)
                    included += 1
                if len(item) > included:
                    projected["$omitted_items"] = len(item) - included
                return projected
            finally:
                state["ancestors"].remove(identity)
        if type(item) in {list, tuple}:
            identity = id(item)
            if identity in state["ancestors"]:
                return {"$truncated": "cycle"}
            state["ancestors"].add(identity)
            try:
                included = min(len(item), MIXING_AGENT_IDENTITY_MAX_SEQUENCE_ITEMS)
                projected = [visit(item[index], depth + 1) for index in range(included)]
                if len(item) > included:
                    projected.append({"$omitted_items": len(item) - included})
                return projected
            finally:
                state["ancestors"].remove(identity)
        return {"$unsupported_type": "unsupported"}

    return visit(value, 0)


def compact_mixing_agent_row(action: Any, *, history: bool = False) -> Dict[str, Any]:
    data = action if type(action) is dict else {}
    producer_digest = dict.get(data, "action_identity_digest")
    if type(producer_digest) is not str or not is_sha256_digest(producer_digest):
        producer_digest = dict.get(data, "parameters_digest")
    if type(producer_digest) is not str or not is_sha256_digest(producer_digest):
        producer_digest = None
    identity_projection = {
        "history": bool(history),
        "mapping_type": "dict" if type(action) is dict else "unsupported",
        "source": _safe_action_text(dict.get(data, "source"), 128),
        "action_type": _safe_action_text(dict.get(data, "type"), 128),
        "channel": _safe_action_text(dict.get(data, "channel"), 64),
        "timestamp": _safe_action_text(dict.get(data, "timestamp"), 64),
        "risk": _safe_action_text(dict.get(data, "risk"), 32),
        "reason": _safe_action_text(dict.get(data, "reason"), 256),
        "producer_digest": producer_digest,
        "parameters": (
            None
            if producer_digest is not None
            else _bounded_mixing_agent_identity_value(dict.get(data, "parameters"))
        ),
    }
    identity = hashlib.sha256(canonical_json_bytes(identity_projection)).hexdigest()[:24]
    source = _safe_action_text(dict.get(data, "source"), 128) or "mixing_agent"
    status = "history" if history else "analysis_only"
    timestamp = dict.get(data, "timestamp")
    if type(timestamp) in {int, float} and not isinstance(timestamp, bool) and math.isfinite(float(timestamp)):
        try:
            stamp = datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat().replace("+00:00", "Z")
        except (OverflowError, OSError, ValueError):
            stamp = None
    else:
        stamp = _iso_or_none(timestamp) if type(timestamp) is str else None
    return {
        "id": f"mixing:{identity}",
        "entity_kind": "mixing_agent_action",
        "queue_section": "history" if history else "analysis_only",
        "title_preview": _safe_action_text(dict.get(data, "type"), 512) or "Mixing agent action",
        "reason_preview": _safe_action_text(dict.get(data, "reason"), 1_024),
        "display_target": _safe_action_text(dict.get(data, "channel"), 256),
        "kind": _safe_action_text(dict.get(data, "type"), 128),
        "severity": _safe_action_text(dict.get(data, "risk"), 32) or "medium",
        "status": status,
        "source": source,
        "created_at": stamp,
        "updated_at": stamp,
        "expires_at": None,
        "can_apply": False,
        "requires_approval": False,
        "apply_blocker_codes": ["mixing_agent_action_import_required"],
        "apply_blocker_count": 1,
        "linked_change_count": 0,
        "review_contract_version": None,
        "review_contract_status": "import_required",
        "review_content_digest": None,
        "detail_available": False,
        "hazard_scope": "proposal_local",
        "priority_class": "blocked" if not history else "history",
        "analysis_only": True,
        "import_required": True,
    }


def _row_sort_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    rank = {"global_hazard": 0, "blocked": 1, "high": 2, "ordinary": 3, "history": 4}
    timestamp = str(row.get("updated_at") or row.get("created_at") or "")
    # ISO timestamps have lexical order; invert their bytes for newest-first.
    inverse_time = tuple(-byte for byte in timestamp.encode("utf-8"))
    return (
        rank.get(str(row.get("priority_class")), 1),
        inverse_time,
        str(row.get("source") or ""),
        str(row.get("id") or ""),
    )


class CompactRowCollection(list):
    """Materialized rows plus exact counts discarded before compaction."""

    def __init__(
        self,
        rows: Iterable[Dict[str, Any]],
        *,
        source_omitted_by_section: Optional[Mapping[str, int]] = None,
        source_omitted_by_priority: Optional[Mapping[str, int]] = None,
        source_omitted_by_apply_state: Optional[Mapping[str, int]] = None,
    ) -> None:
        super().__init__(rows)
        self.source_omitted_by_section = dict(source_omitted_by_section or {})
        self.source_omitted_by_priority = dict(source_omitted_by_priority or {})
        self.source_omitted_by_apply_state = dict(source_omitted_by_apply_state or {})
        self.source_omitted_count = sum(self.source_omitted_by_section.values())
        self.source_pinned_omitted_count = sum(
            count
            for priority, count in self.source_omitted_by_priority.items()
            if priority in {"global_hazard", "blocked"}
        )


def collect_compact_rows(server: Any) -> CompactRowCollection:
    queue = getattr(server, "operator_proposal_queue", None)
    rows: List[Dict[str, Any]] = []
    omitted_section = {"analysis_only": 0, "history": 0}
    omitted_priority = {"blocked": 0, "history": 0}
    omitted_apply_state = {"analysis_only": 0}
    if queue is not None:
        rows.extend(compact_proposal_row(item) for item in queue.iter_pending())
        rows.extend(compact_proposal_row(item) for item in queue.iter_history())
    agent = getattr(server, "mixing_agent", None)
    if agent is not None:
        try:
            pending_source = agent.get_pending_actions() or []
        except Exception:
            pending_source = []
        try:
            history_source = agent.get_action_history(MIXING_AGENT_HISTORY_SOURCE_CAP) or []
        except Exception:
            history_source = []
        for source, cap, history in (
            (pending_source, MIXING_AGENT_PENDING_SOURCE_CAP, False),
            (history_source, MIXING_AGENT_HISTORY_SOURCE_CAP, True),
        ):
            if type(source) not in {list, tuple}:
                continue
            included_source = source[:cap]
            valid_items = [item for item in included_source if type(item) is dict]
            rows.extend(
                compact_mixing_agent_row(item, history=history)
                for item in valid_items
            )
            omitted = len(source) - len(valid_items)
            section = "history" if history else "analysis_only"
            priority = "history" if history else "blocked"
            omitted_section[section] += omitted
            omitted_priority[priority] += omitted
            omitted_apply_state["analysis_only"] += omitted
    return CompactRowCollection(
        sorted(rows, key=_row_sort_key),
        source_omitted_by_section=omitted_section,
        source_omitted_by_priority=omitted_priority,
        source_omitted_by_apply_state=omitted_apply_state,
    )


_HAZARD_SOURCES = ("runtime", "manual_write", "terminal_result")


def collect_global_hazards(server: Any, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    hazards: List[Tuple[str, str, str]] = []

    def add(condition: Any, code: str, source: str, entity: str = "server") -> None:
        if condition:
            hazards.append((code, source, hashlib.sha256(entity.encode("utf-8")).hexdigest()))

    mode = server.get_operator_mode_status()
    capabilities = mode.get("capabilities", {}) if isinstance(mode, dict) else {}
    mixer = getattr(server, "mixer_client", None)
    add(getattr(server, "_is_shutting_down", False), "server_shutdown_active", "runtime")
    add(mixer is None or not getattr(mixer, "is_connected", False), "mixer_disconnected", "runtime")
    add(getattr(server, "_wing_observation_only", False), "wing_observation_only", "runtime")
    add(not capabilities.get("can_apply_to_console", False), "operator_mode_blocks_apply", "runtime")
    add(getattr(server, "_manual_write_emergency_stopped", False), "emergency_stop_active", "manual_write")
    add(getattr(server, "_system_emergency_stop_in_progress", False), "emergency_stop_in_progress", "manual_write")
    add(
        bool(getattr(server, "_system_emergency_stop_errors", None)),
        "emergency_stop_teardown_incomplete",
        "manual_write",
    )
    add(
        bool(getattr(server, "_manual_write_recovery_transactions", None)),
        "manual_write_wal_recovery_required",
        "manual_write",
    )
    lock = getattr(server, "_console_write_lock", None)
    add(bool(lock is not None and getattr(lock, "locked", lambda: False)()), "console_writer_busy", "manual_write")
    add(
        getattr(server, "_console_writer_conflict_recorded", False),
        "console_writer_conflict_recorded",
        "manual_write",
    )
    queue = getattr(server, "operator_proposal_queue", None)
    add(
        bool(getattr(queue, "result_persistence_failed", False)),
        "operator_proposal_result_persistence_failed",
        "terminal_result",
    )
    recovered_result_codes = frozenset(
        {
            "terminal_write_state_unknown",
            "terminal_rollback_state_unknown",
            "terminal_manual_review_after_send",
            "operator_proposal_result_recovery_required",
            "operator_proposal_result_persistence_failed",
        }
    )
    recovered_result_hazards = (
        getattr(queue, "result_recovery_hazards", ())
        if queue is not None
        else ()
    )
    for recovered in recovered_result_hazards:
        if not isinstance(recovered, Mapping):
            continue
        code = str(recovered.get("code") or "")
        if code not in recovered_result_codes:
            continue
        entity = str(
            recovered.get("result_digest")
            or recovered.get("review_content_digest")
            or recovered.get("proposal_id")
            or "retained_result"
        )
        add(True, code, "terminal_result", entity)
    terminal_items = (
        queue.iter_terminal_hazards()
        if queue is not None and hasattr(queue, "iter_terminal_hazards")
        else ()
    )
    for proposal in terminal_items:
        result = getattr(proposal, "result", None)
        if not isinstance(result, dict):
            continue
        entity = str(
            getattr(proposal, "review_content_digest", None)
            or getattr(proposal, "result_digest", None)
            or getattr(proposal, "id", None)
            or "proposal"
        )
        add(result.get("write_state_unknown") is True, "terminal_write_state_unknown", "terminal_result", entity)
        add(
            result.get("rollback_state_unknown") is True
            or (
                result.get("physical_state_restored") is False
                and not (
                    terminal_result_proves_verified_apply(result)
                    or terminal_result_proves_verified_already_satisfied_apply(
                        result
                    )
                )
            ),
            "terminal_rollback_state_unknown",
            "terminal_result",
            entity,
        )
        add(result.get("manual_review_required") is True, "terminal_manual_review_after_send", "terminal_result", entity)
        add(
            str(result.get("reason") or "")
            == "operator_proposal_result_persistence_failed",
            "operator_proposal_result_persistence_failed",
            "terminal_result",
            entity,
        )

    hazards = sorted(set(hazards))
    codes = [item[0] for item in hazards]
    source_counts = {source: 0 for source in _HAZARD_SOURCES}
    for _code, source, _entity in hazards:
        source_counts[source] += 1
    digest_bytes = canonical_json_bytes(
        [{"code": code, "source": source, "entity_reference_digest": entity} for code, source, entity in hazards]
    )
    return {
        "fatal": bool(hazards),
        "apply_ui_disabled": bool(hazards),
        "count": len(hazards),
        "codes": codes[:16],
        "overflow_count": max(0, len(codes) - 16),
        "source_counts": source_counts,
        "digest": sha256_digest(digest_bytes),
    }


def wire_serialized_bytes(payload: Mapping[str, Any]) -> int:
    return len(serialize_message(dict(payload), _native_json).encode("utf-8"))


def finalize_wire_size(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload.setdefault("wire", {})["max_bytes"] = WIRE_MAX_BYTES
    previous = None
    for _ in range(8):
        size = wire_serialized_bytes(payload)
        payload["wire"]["serialized_bytes"] = size
        if size == previous and wire_serialized_bytes(payload) == size:
            break
        previous = size
    actual = wire_serialized_bytes(payload)
    payload["wire"]["serialized_bytes"] = actual
    if wire_serialized_bytes(payload) != actual:
        payload["wire"]["serialized_bytes"] = wire_serialized_bytes(payload)
    return payload


_CATASTROPHIC_APPLY_RESPONSE_ASCII = (
    '{"type":"operator_proposal_apply_result","contract_version":2,'
    '"status":"apply_error","success":false,'
    '"reason":"operator_apply_response_projection_failed",'
    '"proposal_id":null,"proposal_status":"unknown",'
    '"apply_status":"unknown","no_console_send_proven":null,'
    '"physical_verified":null,"physical_state_restored":null,'
    '"write_state_unknown":true,"rollback_state_unknown":true,'
    '"manual_review_required":true,"emergency_stop_activated":null,'
    '"emergency_stop_required":true,'
    '"emergency_stop_teardown_complete":null,"response_degraded":true}'
)


def catastrophic_apply_response_serialized() -> str:
    """Return the fixed last-resort ASCII receipt, or fail for close-1011."""

    serialized = _CATASTROPHIC_APPLY_RESPONSE_ASCII
    try:
        encoded = serialized.encode("ascii")
        parsed = json.loads(serialized)
    except (UnicodeEncodeError, json.JSONDecodeError, TypeError) as exc:
        raise ApplyResponseProjectionError(
            "catastrophic_apply_response_invalid"
        ) from exc
    if (
        len(encoded) > WIRE_MAX_BYTES
        or type(parsed) is not dict
        or dict.get(parsed, "type") != "operator_proposal_apply_result"
        or dict.get(parsed, "no_console_send_proven") is not None
        or dict.get(parsed, "write_state_unknown") is not True
        or dict.get(parsed, "rollback_state_unknown") is not True
        or dict.get(parsed, "manual_review_required") is not True
        or dict.get(parsed, "emergency_stop_activated") is not None
        or dict.get(parsed, "emergency_stop_required") is not True
    ):
        raise ApplyResponseProjectionError(
            "catastrophic_apply_response_invalid"
        )
    return serialized


def _receipt_projection(receipt: TrustedApplyReceipt) -> Dict[str, Any]:
    proposal_id = utf8_preview(receipt.proposal_id, 256)
    proposal_status = utf8_preview(receipt.proposal_status, 64)
    apply_status = utf8_preview(receipt.apply_status, 64)
    payload = {
        "type": receipt.response_type,
        "contract_version": 2,
        "status": receipt.final_status,
        "success": receipt.success,
        "reason": receipt.reason_code,
        "proposal_id": proposal_id,
        "proposal_status": proposal_status,
        "proposal": {
            "id": proposal_id,
            "review_contract_version": receipt.review_contract_version,
            "review_content_digest": receipt.review_content_digest,
            "status": proposal_status,
            "result_digest": receipt.result_digest,
            "result_detail_available": receipt.result_detail_available,
            "result_evidence_pinned": receipt.result_evidence_pinned,
        },
        "apply_status": apply_status,
        "no_console_send_proven": receipt.no_console_send_proven,
        "physical_verified": receipt.physical_verified,
        "physical_state_restored": receipt.physical_state_restored,
        "terminal_after_console_send": receipt.terminal_after_console_send,
        "write_state_unknown": receipt.write_state_unknown,
        "rollback_attempted": receipt.rollback_attempted,
        "rollback_succeeded": receipt.rollback_succeeded,
        "rollback_state_unknown": receipt.rollback_state_unknown,
        "manual_review_required": receipt.manual_review_required,
        "wal_terminal": (
            utf8_preview(receipt.wal_terminal, 64)
            if receipt.wal_terminal is not None
            else None
        ),
        "rollback_status": (
            utf8_preview(receipt.rollback_status, 64)
            if receipt.rollback_status is not None
            else None
        ),
        "audio_recheck_required": receipt.audio_recheck_required,
        "audio_recheck_passed": receipt.audio_recheck_passed,
        "audio_recheck_disposition": (
            utf8_preview(receipt.audio_recheck_disposition, 128)
            if receipt.audio_recheck_disposition is not None
            else None
        ),
        "result_digest": receipt.result_digest,
        "result_detail_available": receipt.result_detail_available,
        "result_evidence_pinned": receipt.result_evidence_pinned,
        "emergency_stop_activated": receipt.emergency_stop_activated,
        "emergency_stop_required": receipt.emergency_stop_required,
        "emergency_stop_teardown_complete": (
            receipt.emergency_stop_teardown_complete
        ),
        "idempotency": {
            "key_digest": receipt.idempotency_key_digest,
            "status": (
                utf8_preview(receipt.idempotency_status, 128)
                if receipt.idempotency_status is not None
                else None
            ),
            "replay": receipt.idempotency_replay,
            "released": receipt.operator_idempotency_released,
        },
        "channel": receipt.channel,
        "value_type": (
            utf8_preview(receipt.value_type, 128)
            if receipt.value_type is not None
            else None
        ),
        "response_degraded": False,
        "wire": {},
    }
    m110_capability_terminal = bool(
        receipt.final_status in {"proof_passed", "applied"}
        and receipt.capability_kind
        in {
            "restore_only_proof",
            "maintenance_dsp_reset_commit_once",
            "processor_enable_commit_once",
            "soundcheck_child_commit_once",
        }
    )
    if m110_capability_terminal:
        payload.update(
            {
                "accepted": receipt.accepted,
                "owner_review_required": receipt.owner_review_required,
                "live_authority_granted": receipt.live_authority_granted,
                "readiness_closure_allowed": (
                    receipt.readiness_closure_allowed
                ),
                "proof_run": receipt.proof_run,
                "restore_after_verify": receipt.restore_after_verify,
                "capability_kind": (
                    utf8_preview(receipt.capability_kind, 64)
                    if receipt.capability_kind is not None
                    else None
                ),
                "operation_digest": receipt.operation_digest,
                "bundle_kind": (
                    utf8_preview(receipt.bundle_kind, 64)
                    if receipt.bundle_kind is not None
                    else None
                ),
                "proof_receipt_id": (
                    utf8_preview(receipt.proof_receipt_id, 96)
                    if receipt.proof_receipt_id is not None
                    else None
                ),
                "review_preimage_sha256": receipt.review_preimage_sha256,
                "before_manifest_sha256": receipt.before_manifest_sha256,
                "after_manifest_sha256": receipt.after_manifest_sha256,
                "verification_cut_reset_surface_sha256": (
                    receipt.verification_cut_reset_surface_sha256
                ),
                "verification_cut_preservation_manifest_sha256": (
                    receipt.verification_cut_preservation_manifest_sha256
                ),
                "no_outside_mutation_attempt_proven": (
                    receipt.no_outside_mutation_attempt_proven
                ),
                "current_state_guaranteed_at_response": (
                    receipt.current_state_guaranteed_at_response
                ),
                "fresh_readback_required_before_next_stage": (
                    receipt.fresh_readback_required_before_next_stage
                ),
                "attended_no_touch_required_during_execution": (
                    receipt.attended_no_touch_required_during_execution
                ),
                "later_console_event_invalidates_readiness_freshness": (
                    receipt.later_console_event_invalidates_readiness_freshness
                ),
                "proof_target_verified": receipt.proof_target_verified,
                "applied_operation_ids": (
                    list(receipt.applied_operation_ids)
                    if receipt.applied_operation_ids is not None
                    else None
                ),
                "rollback_operation_ids": (
                    list(receipt.rollback_operation_ids)
                    if receipt.rollback_operation_ids is not None
                    else None
                ),
                "already_satisfied_operation_ids": (
                    list(receipt.already_satisfied_operation_ids)
                    if receipt.already_satisfied_operation_ids is not None
                    else None
                ),
                "parent_commit_recorded": receipt.parent_commit_recorded,
                "parent_commit_reason": receipt.parent_commit_reason,
            }
        )
    return payload


def _prewrite_no_send_projection(
    receipt: TrustedPrewriteNoSendReceipt,
) -> Dict[str, Any]:
    proposal_id = (
        utf8_preview(receipt.proposal_id, 256)
        if receipt.proposal_id is not None
        else None
    )
    proposal_status = utf8_preview(receipt.proposal_status, 64)
    return {
        "type": "operator_proposal_apply_blocked",
        "contract_version": 2,
        "status": "apply_blocked",
        "success": False,
        "reason": receipt.reason_code,
        "proposal_id": proposal_id,
        "proposal_status": proposal_status,
        "proposal": {
            "id": proposal_id,
            "review_contract_version": receipt.review_contract_version,
            "review_content_digest": receipt.review_content_digest,
            "status": proposal_status,
            "result_digest": None,
            "result_detail_available": False,
            "result_evidence_pinned": False,
        },
        "apply_status": "blocked",
        "idempotency_reserved": False,
        "proposal_status_mutation": False,
        "result_evidence_reserved": False,
        "no_console_send_proven": True,
        "physical_verified": False,
        "physical_state_restored": None,
        "terminal_after_console_send": False,
        "write_state_unknown": False,
        "rollback_attempted": False,
        "rollback_succeeded": None,
        "rollback_state_unknown": False,
        "manual_review_required": False,
        "wal_terminal": None,
        "rollback_status": None,
        "audio_recheck_required": None,
        "audio_recheck_passed": None,
        "audio_recheck_disposition": None,
        "result_digest": None,
        "result_detail_available": False,
        "result_evidence_pinned": False,
        "emergency_stop_activated": False,
        "emergency_stop_required": False,
        "emergency_stop_teardown_complete": None,
        "idempotency": {
            "key_digest": None,
            "status": None,
            "replay": False,
            "released": None,
        },
        "channel": None,
        "value_type": None,
        "response_degraded": True,
        "wire": {},
    }


def project_operator_apply_response(
    message: Any,
    trusted_receipt: Any,
) -> str:
    """Project and serialize one bounded apply response exactly once."""

    if (
        type(trusted_receipt) is TrustedApplyReceipt
        and trusted_receipt._is_authentic()
    ):
        payload = _receipt_projection(trusted_receipt)
    elif (
        type(trusted_receipt) is TrustedPrewriteNoSendReceipt
        and trusted_receipt._is_authentic()
    ):
        payload = _prewrite_no_send_projection(trusted_receipt)
    else:
        return catastrophic_apply_response_serialized()
    finalize_wire_size(payload)
    serialized = serialize_message(payload, _native_json)
    encoded_bytes = len(serialized.encode("utf-8"))
    if (
        encoded_bytes > WIRE_MAX_BYTES
        or payload["wire"].get("serialized_bytes") != encoded_bytes
    ):
        raise ApplyResponseProjectionError(
            "operator_apply_response_wire_budget_invariant_breached"
        )
    return serialized


def compact_trusted_apply_receipt_reference(
    trusted_receipt: Any,
) -> Optional[Dict[str, Any]]:
    """Return bounded scalar truth from an authentic finalized apply receipt.

    Sequence executors use this projection instead of rereading mutable queue
    rows after a child apply.  A missing or forged receipt has no reference at
    all; callers must stop and report unknown/manual state.
    """

    if (
        type(trusted_receipt) is TrustedApplyReceipt
        and trusted_receipt._is_authentic()
    ):
        payload = _receipt_projection(trusted_receipt)
    elif (
        type(trusted_receipt) is TrustedPrewriteNoSendReceipt
        and trusted_receipt._is_authentic()
    ):
        payload = _prewrite_no_send_projection(trusted_receipt)
    else:
        return None
    proposal = payload.get("proposal")
    proposal_ref = proposal if type(proposal) is dict else {}
    idempotency = payload.get("idempotency")
    idempotency_ref = idempotency if type(idempotency) is dict else {}
    return {
        "proposal_id": payload.get("proposal_id"),
        "review_contract_version": proposal_ref.get("review_contract_version"),
        "review_content_digest": proposal_ref.get("review_content_digest"),
        "proposal_status": payload.get("proposal_status"),
        "status": payload.get("status"),
        "apply_status": payload.get("apply_status"),
        "success": payload.get("success") is True,
        "no_console_send_proven": payload.get("no_console_send_proven"),
        "physical_verified": payload.get("physical_verified"),
        "physical_state_restored": payload.get("physical_state_restored"),
        "terminal_after_console_send": payload.get("terminal_after_console_send"),
        "write_state_unknown": payload.get("write_state_unknown") is True,
        "rollback_attempted": payload.get("rollback_attempted") is True,
        "rollback_succeeded": payload.get("rollback_succeeded"),
        "rollback_state_unknown": payload.get("rollback_state_unknown") is True,
        "manual_review_required": payload.get("manual_review_required") is True,
        "wal_terminal": payload.get("wal_terminal"),
        "rollback_status": payload.get("rollback_status"),
        "audio_recheck_required": payload.get("audio_recheck_required"),
        "audio_recheck_passed": payload.get("audio_recheck_passed"),
        "audio_recheck_disposition": payload.get("audio_recheck_disposition"),
        "result_digest": payload.get("result_digest"),
        "result_detail_available": payload.get("result_detail_available") is True,
        "result_evidence_pinned": payload.get("result_evidence_pinned") is True,
        "emergency_stop_activated": payload.get("emergency_stop_activated"),
        "emergency_stop_required": payload.get("emergency_stop_required") is True,
        "emergency_stop_teardown_complete": payload.get(
            "emergency_stop_teardown_complete"
        ),
        "idempotency_key_digest": idempotency_ref.get("key_digest"),
        "idempotency_status": idempotency_ref.get("status"),
        "idempotency_replay": idempotency_ref.get("replay") is True,
        "operator_idempotency_released": idempotency_ref.get("released"),
        "channel": payload.get("channel"),
        "value_type": payload.get("value_type"),
        "applied_operation_ids": payload.get("applied_operation_ids"),
        "rollback_operation_ids": payload.get("rollback_operation_ids"),
        "already_satisfied_operation_ids": payload.get(
            "already_satisfied_operation_ids"
        ),
        "parent_commit_recorded": payload.get("parent_commit_recorded"),
        "parent_commit_reason": payload.get("parent_commit_reason"),
    }


def _fixed_counts(rows: Sequence[Mapping[str, Any]], included_count: int, pinned_included: int) -> Dict[str, Any]:
    section = {key: 0 for key in ("pending", "history", "analysis_only")}
    priority = {key: 0 for key in ("global_hazard", "blocked", "high", "ordinary", "history")}
    apply_state = {key: 0 for key in ("applyable", "blocked", "analysis_only")}
    for row in rows:
        section[str(row["queue_section"])] += 1
        priority[str(row["priority_class"])] += 1
        state = "analysis_only" if row["analysis_only"] else "applyable" if row["can_apply"] else "blocked"
        apply_state[state] += 1
    source_omitted = int(getattr(rows, "source_omitted_count", 0) or 0)
    source_pinned_omitted = int(
        getattr(rows, "source_pinned_omitted_count", 0) or 0
    )
    for key, count in getattr(rows, "source_omitted_by_section", {}).items():
        if key in section:
            section[key] += int(count or 0)
    for key, count in getattr(rows, "source_omitted_by_priority", {}).items():
        if key in priority:
            priority[key] += int(count or 0)
    for key, count in getattr(rows, "source_omitted_by_apply_state", {}).items():
        if key in apply_state:
            apply_state[key] += int(count or 0)
    materialized_pinned = sum(
        1 for row in rows if row["priority_class"] in {"global_hazard", "blocked"}
    )
    pinned_total = materialized_pinned + source_pinned_omitted
    total_count = len(rows) + source_omitted
    return {
        "total_count": total_count,
        "materialized_source_count": len(rows),
        "source_omitted_count": source_omitted,
        "included_count": included_count,
        "omitted_count": total_count - included_count,
        "pinned_total_count": pinned_total,
        "pinned_included_count": pinned_included,
        "pinned_omitted_count": pinned_total - pinned_included,
        "by_section": section,
        "by_priority": priority,
        "by_apply_state": apply_state,
    }


@dataclass
class _Snapshot:
    id: str
    owner: str
    rows: List[Dict[str, Any]]
    revision: str
    hazards: Dict[str, Any]
    counts: Dict[str, Any]
    created_at_epoch: float
    expires_at_epoch: float
    page_size: int
    bytes: int


def _utc_from_epoch(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _snapshot_revision(rows: Sequence[Mapping[str, Any]], hazards: Mapping[str, Any], page_size: int) -> str:
    projection = {
        "contract_version": 2,
        "page_size": page_size,
        "rows": [
            {
                "id": row["id"],
                "entity_kind": row["entity_kind"],
                "review_contract_version": row["review_contract_version"],
                "review_content_digest": row["review_content_digest"],
                "status": row["status"],
                "can_apply": row["can_apply"],
                "expires_at": row["expires_at"],
                "priority_class": row["priority_class"],
            }
            for row in rows
        ],
        "source_omitted_count": int(
            getattr(rows, "source_omitted_count", 0) or 0
        ),
        "source_pinned_omitted_count": int(
            getattr(rows, "source_pinned_omitted_count", 0) or 0
        ),
        "source_omitted_by_section": dict(
            getattr(rows, "source_omitted_by_section", {})
        ),
        "hazard_digest": hazards["digest"],
    }
    return sha256_digest(canonical_json_bytes(projection))


def decision_queue_error(reason: str) -> Dict[str, Any]:
    return finalize_wire_size({
        "type": "decision_queue_error",
        "contract_version": 2,
        "status": "error",
        "success": False,
        "reason": reason,
        "no_console_send_proven": True,
        "proposal_status_mutation": False,
        "idempotency_reserved": False,
        "wire": {},
    })


class DecisionQueueSnapshotManager:
    """Small connection-owned cache of frozen compact queue snapshots."""

    def __init__(self) -> None:
        self._snapshots: Dict[str, _Snapshot] = {}
        self._cursor_index: Dict[str, Tuple[str, int]] = {}
        self._cursor_by_offset: Dict[Tuple[str, int], str] = {}
        self._cursor_tombstones: Dict[str, Tuple[str, str, float]] = {}
        self._owner_tokens: List[Tuple[Any, str]] = []

    def _owner_token(self, websocket: Any) -> str:
        for connection, token in self._owner_tokens:
            if connection is websocket:
                return token
        token = secrets.token_hex(16)
        self._owner_tokens.append((websocket, token))
        if len(self._owner_tokens) > SNAPSHOT_MAX_GLOBAL * 2:
            active = {token}
            active.update(item.owner for item in self._snapshots.values())
            active.update(item[1] for item in self._cursor_tombstones.values())
            self._owner_tokens = [
                item for item in self._owner_tokens if item[1] in active
            ][-(SNAPSHOT_MAX_GLOBAL * 2) :]
        return token

    def release_owner(self, websocket: Any) -> None:
        token = next(
            (
                token
                for connection, token in self._owner_tokens
                if connection is websocket
            ),
            None,
        )
        if token is None:
            return
        for snapshot_id, snapshot in tuple(self._snapshots.items()):
            if snapshot.owner == token:
                self._remove(
                    snapshot_id,
                    "decision_queue_snapshot_owner_released",
                )
        for cursor, tombstone in tuple(self._cursor_tombstones.items()):
            if tombstone[1] == token:
                self._cursor_tombstones.pop(cursor, None)
        self._owner_tokens = [
            item for item in self._owner_tokens if item[0] is not websocket
        ]

    def _purge(self, now: float) -> None:
        expired = [key for key, item in self._snapshots.items() if item.expires_at_epoch <= now]
        for snapshot_id in expired:
            self._remove(snapshot_id, "decision_queue_snapshot_expired", now)
        for token, tombstone in list(self._cursor_tombstones.items()):
            if tombstone[2] <= now:
                self._cursor_tombstones.pop(token, None)

    def _remove(self, snapshot_id: str, reason: str, now: Optional[float] = None) -> None:
        snapshot = self._snapshots.pop(snapshot_id, None)
        owner = snapshot.owner if snapshot is not None else ""
        expires = (now if now is not None else time.time()) + SNAPSHOT_TTL_SECONDS
        for token, binding in list(self._cursor_index.items()):
            if binding[0] == snapshot_id:
                self._cursor_index.pop(token, None)
                self._cursor_tombstones[token] = (reason, owner, expires)
        for key, token in list(self._cursor_by_offset.items()):
            if key[0] == snapshot_id:
                self._cursor_by_offset.pop(key, None)
        while len(self._cursor_tombstones) > 256:
            self._cursor_tombstones.pop(next(iter(self._cursor_tombstones)))

    def _enforce_limits(self, owner: str) -> None:
        owner_items = sorted(
            (item for item in self._snapshots.values() if item.owner == owner),
            key=lambda item: item.created_at_epoch,
        )
        while len(owner_items) > SNAPSHOT_MAX_PER_CONNECTION:
            self._remove(owner_items.pop(0).id, "decision_queue_snapshot_evicted")
        all_items = sorted(self._snapshots.values(), key=lambda item: item.created_at_epoch)
        while len(all_items) > SNAPSHOT_MAX_GLOBAL or sum(item.bytes for item in all_items) > SNAPSHOT_CACHE_MAX_BYTES:
            self._remove(all_items.pop(0).id, "decision_queue_snapshot_evicted")

    def _materialize(self, server: Any, owner: str, page_size: int, now: float) -> _Snapshot:
        all_rows = collect_compact_rows(server)
        hazards = collect_global_hazards(server, all_rows)
        included: List[Dict[str, Any]] = []
        used = wire_serialized_bytes({"rows": []})
        for row in all_rows:
            candidate_bytes = wire_serialized_bytes({"rows": [*included, row]})
            if candidate_bytes > SNAPSHOT_MAX_BYTES:
                continue
            included.append(row)
            used = candidate_bytes
        pinned_included = sum(1 for row in included if row["priority_class"] in {"global_hazard", "blocked"})
        counts = _fixed_counts(all_rows, len(included), pinned_included)
        revision = _snapshot_revision(all_rows, hazards, page_size)
        snapshot = _Snapshot(
            id=secrets.token_hex(16),
            owner=owner,
            rows=included,
            revision=revision,
            hazards=hazards,
            counts=counts,
            created_at_epoch=now,
            expires_at_epoch=now + SNAPSHOT_TTL_SECONDS,
            page_size=page_size,
            bytes=used,
        )
        self._snapshots[snapshot.id] = snapshot
        self._enforce_limits(owner)
        return snapshot

    def _current_revision(self, server: Any, page_size: int) -> str:
        rows = collect_compact_rows(server)
        return _snapshot_revision(rows, collect_global_hazards(server, rows), page_size)

    def response(self, server: Any, websocket: Any, request: Mapping[str, Any]) -> Dict[str, Any]:
        now = time.time()
        self._purge(now)
        owner = self._owner_token(websocket)
        cursor = str(request.get("cursor") or "").strip()
        if cursor:
            binding = self._cursor_index.get(cursor)
            if binding is None:
                tombstone = self._cursor_tombstones.get(cursor)
                if tombstone is not None:
                    if tombstone[1] != owner:
                        return decision_queue_error("decision_queue_cursor_owner_mismatch")
                    return decision_queue_error(tombstone[0])
                return decision_queue_error("decision_queue_cursor_invalid")
            snapshot = self._snapshots.get(binding[0])
            if snapshot is None:
                return decision_queue_error("decision_queue_snapshot_evicted")
            if snapshot.owner != owner:
                return decision_queue_error("decision_queue_cursor_owner_mismatch")
            if "page_size" in request or "limit" in request:
                return decision_queue_error("decision_queue_cursor_request_conflict")
            expected = str(request.get("expected_snapshot_revision") or "")
            if not expected:
                return decision_queue_error("decision_queue_snapshot_revision_required")
            if expected != snapshot.revision:
                return decision_queue_error("decision_queue_snapshot_revision_mismatch")
            if self._current_revision(server, snapshot.page_size) != snapshot.revision:
                return decision_queue_error("decision_queue_cursor_stale")
            offset = binding[1]
        else:
            page_size = request.get("page_size")
            if (
                "limit" in request
                or type(page_size) is not int
                or page_size != 50
            ):
                return decision_queue_error("decision_queue_page_size_invalid")
            snapshot = self._materialize(server, owner, page_size, now)
            offset = 0

        pinned_omitted = snapshot.counts["pinned_omitted_count"]
        degraded = pinned_omitted > 0
        available = snapshot.rows[offset : offset + snapshot.page_size]
        items: List[Dict[str, Any]] = []
        for candidate in available:
            item = dict(candidate)
            if degraded or snapshot.hazards["fatal"]:
                item["can_apply"] = False
                item["requires_approval"] = False
            items.append(item)

        def build_payload(page_items: List[Dict[str, Any]]) -> Dict[str, Any]:
            next_offset = offset + len(page_items)
            has_more = next_offset < len(snapshot.rows)
            cursor_key = (snapshot.id, next_offset)
            next_cursor = self._cursor_by_offset.get(cursor_key) if has_more else None
            if has_more and next_cursor is None:
                next_cursor = secrets.token_urlsafe(24)
            payload = {
                "type": "decision_queue",
                "contract_version": 2,
                "wire_schema": DECISION_QUEUE_SCHEMA,
                "status": "degraded_overflow" if degraded else "ok",
                "success": True,
                "snapshot": {
                    "id": snapshot.id,
                    "revision": snapshot.revision,
                    "created_at": _utc_from_epoch(snapshot.created_at_epoch),
                    "expires_at": _utc_from_epoch(snapshot.expires_at_epoch),
                    "fully_pageable": snapshot.counts["omitted_count"] == 0,
                },
                "page": {
                    "page_size": snapshot.page_size,
                    "returned_count": len(page_items),
                    "remaining_count": max(0, len(snapshot.rows) - next_offset),
                    "next_cursor": next_cursor,
                    "has_more": has_more,
                },
                "counts": snapshot.counts,
                "items": page_items,
                "overflow": {
                    "counts_only": snapshot.counts["omitted_count"] > 0,
                    "reason": "pinned_metadata_budget" if degraded else "snapshot_materialization_budget" if snapshot.counts["omitted_count"] else None,
                    "unpageable_count": snapshot.counts["omitted_count"],
                    "pinned_unpageable_count": pinned_omitted,
                    "apply_ui_disabled": bool(degraded or snapshot.hazards["apply_ui_disabled"]),
                    "hazard_digest": snapshot.hazards["digest"],
                },
                "hazards": snapshot.hazards,
                "wire": {},
            }
            finalize_wire_size(payload)
            payload["_candidate_next_cursor"] = next_cursor
            return payload

        payload = build_payload(items)
        while payload["wire"]["serialized_bytes"] > WIRE_PACK_TARGET_BYTES and items:
            items.pop()
            payload = build_payload(items)
        next_cursor = payload.pop("_candidate_next_cursor")
        finalize_wire_size(payload)
        if payload["wire"]["serialized_bytes"] > WIRE_MAX_BYTES:
            return decision_queue_error("decision_queue_response_budget_invariant_breached")
        if next_cursor:
            self._cursor_index[next_cursor] = (snapshot.id, offset + len(items))
            self._cursor_by_offset[(snapshot.id, offset + len(items))] = next_cursor
        return payload


def build_v1_decision_queue_shim(server: Any) -> Dict[str, Any]:
    rows = []
    for source_row in collect_compact_rows(server):
        row = dict(source_row)
        blockers = list(row.get("apply_blocker_codes") or [])
        if "legacy_decision_queue_read_only" not in blockers:
            blockers.append("legacy_decision_queue_read_only")
        row["apply_blocker_codes"] = blockers[:8]
        row["apply_blocker_count"] = int(row.get("apply_blocker_count") or 0) + 1
        row["can_apply"] = False
        row["requires_approval"] = False
        row["review_contract_status"] = "legacy_read_only_migration_required"
        rows.append(row)
    pending = [row for row in rows if row["queue_section"] != "history"]
    history = [row for row in rows if row["queue_section"] == "history"]
    payload = {
        "type": "decision_queue",
        "contract_version": 1,
        "status": "ok",
        "success": True,
        "deprecated": True,
        "migration_required": True,
        "apply_authority": False,
        "summary": {
            "pending_count": len(pending),
            "history_count": len(history),
            "returned_pending_count": 0,
            "returned_history_count": 0,
        },
        "pending_actions": [],
        "history": [],
        "wire": {},
    }
    for destination, rows_to_pack, count_key in (
        (payload["pending_actions"], pending, "returned_pending_count"),
        (payload["history"], history, "returned_history_count"),
    ):
        for row in rows_to_pack:
            destination.append(row)
            payload["summary"][count_key] += 1
            finalize_wire_size(payload)
            if payload["wire"]["serialized_bytes"] > WIRE_PACK_TARGET_BYTES:
                destination.pop()
                payload["summary"][count_key] -= 1
                break
    finalize_wire_size(payload)
    return payload


def detail_error(reason: str, proposal_id: Optional[str] = None) -> Dict[str, Any]:
    return finalize_wire_size({
        "type": "operator_proposal_detail_error",
        "contract_version": 2,
        "status": "error",
        "success": False,
        "reason": reason,
        "proposal_id": proposal_id,
        "no_console_send_proven": True,
        "proposal_status_mutation": False,
        "idempotency_reserved": False,
        "wire": {},
    })


def build_proposal_detail_response(
    queue: Any,
    request: Mapping[str, Any],
    *,
    server: Any = None,
) -> Dict[str, Any]:
    proposal_id = str(request.get("proposal_id") or "").strip()
    if not proposal_id:
        return detail_error("proposal_id_required")
    expected_version = request.get("expected_review_contract_version")
    if expected_version is None:
        return detail_error("proposal_review_contract_version_required", proposal_id)
    if isinstance(expected_version, bool) or not isinstance(expected_version, int):
        return detail_error("proposal_review_contract_version_invalid", proposal_id)
    proposal = queue.find(proposal_id=proposal_id)
    if proposal is None:
        if queue.proposal_id_was_seen(proposal_id):
            return detail_error("operator_proposal_detail_not_retained", proposal_id)
        return detail_error("operator_proposal_not_found", proposal_id)
    if expected_version != proposal.review_contract_version:
        return detail_error("proposal_review_contract_version_mismatch", proposal_id)
    expected_digest = str(request.get("expected_review_content_digest") or "")
    if not expected_digest:
        return detail_error("operator_proposal_detail_digest_required", proposal_id)
    if expected_digest != proposal.review_content_digest:
        return detail_error("operator_proposal_detail_digest_mismatch", proposal_id)
    integrity = queue.validate_stored_review_integrity(proposal)
    if not integrity["valid"]:
        return detail_error("proposal_review_content_integrity_error", proposal_id)
    chunk_index = request.get("chunk_index", 0)
    if type(chunk_index) is not int:
        return detail_error("operator_proposal_detail_chunk_index_invalid", proposal_id)
    content = proposal.review_content_bytes
    chunk_count = max(1, (len(content) + DETAIL_CHUNK_BYTES - 1) // DETAIL_CHUNK_BYTES)
    if chunk_index < 0 or chunk_index >= chunk_count:
        return detail_error("operator_proposal_detail_chunk_index_invalid", proposal_id)
    chunk = content[chunk_index * DETAIL_CHUNK_BYTES : (chunk_index + 1) * DETAIL_CHUNK_BYTES]
    state = proposal.apply_state()
    global_hazards = None
    if server is not None:
        rows = collect_compact_rows(server)
        global_hazards = collect_global_hazards(server, rows)
    detail_blockers = list(state["apply_blockers"])
    if global_hazards and global_hazards["fatal"]:
        detail_blockers.extend(
            f"global_hazard:{code}" for code in global_hazards["codes"]
        )
    payload = {
        "type": "operator_proposal_detail",
        "contract_version": 2,
        "status": "ok",
        "success": True,
        "proposal_id": proposal.id,
        "review_contract_version": proposal.review_contract_version,
        "review_content_digest": proposal.review_content_digest,
        "current_state": {
            "status": proposal.status,
            "updated_at": proposal.updated_at,
            "expired": state["expired"],
            "can_apply": bool(
                state["can_apply"]
                and not (global_hazards and global_hazards["fatal"])
            ),
            "apply_blocker_codes": [utf8_preview(item, 128) for item in detail_blockers[:8]],
            "apply_blocker_count": len(detail_blockers),
        },
        "content": {
            "encoding": "base64",
            "media_type": "application/vnd.automixer.operator-proposal-review+json;version=2",
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "raw_chunk_bytes": len(chunk),
            "total_raw_bytes": len(content),
            "chunk_sha256": hashlib.sha256(chunk).hexdigest(),
            "content_sha256": hashlib.sha256(content).hexdigest(),
            "data": base64.b64encode(chunk).decode("ascii"),
        },
        "wire": {},
    }
    finalize_wire_size(payload)
    if payload["wire"]["serialized_bytes"] > WIRE_MAX_BYTES:
        return detail_error("operator_proposal_detail_response_budget_invariant_breached", proposal_id)
    return payload


def result_detail_error(
    reason: str,
    proposal_id: Optional[str] = None,
) -> Dict[str, Any]:
    return finalize_wire_size({
        "type": "operator_proposal_result_detail_error",
        "contract_version": 2,
        "status": "error",
        "success": False,
        "reason": reason,
        "proposal_id": proposal_id,
        "no_console_send_proven": True,
        "proposal_status_mutation": False,
        "idempotency_reserved": False,
        "wire": {},
    })


def build_proposal_result_detail_response(
    queue: Any,
    request: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return one authenticated chunk of digest-bound terminal result bytes."""

    proposal_id = str(request.get("proposal_id") or "").strip()
    if not proposal_id:
        return result_detail_error("proposal_id_required")
    expected_digest = str(request.get("expected_result_digest") or "")
    if not expected_digest:
        return result_detail_error(
            "operator_proposal_result_digest_required",
            proposal_id,
        )
    if not is_sha256_digest(expected_digest):
        return result_detail_error(
            "operator_proposal_result_digest_invalid",
            proposal_id,
        )
    proposal = queue.find(proposal_id=proposal_id)
    if proposal is None:
        if queue.proposal_id_was_seen(proposal_id):
            return result_detail_error(
                "operator_proposal_result_detail_not_retained",
                proposal_id,
            )
        return result_detail_error("operator_proposal_not_found", proposal_id)
    if expected_digest != str(getattr(proposal, "result_digest", "") or ""):
        return result_detail_error(
            "operator_proposal_result_digest_mismatch",
            proposal_id,
        )
    try:
        content = queue.load_result_evidence(
            proposal,
            expected_result_digest=expected_digest,
        )
    except Exception as exc:
        candidate = getattr(exc, "code", None)
        reason = (
            candidate
            if candidate in _RESULT_DETAIL_LOAD_ERROR_CODES
            else "operator_proposal_result_detail_unavailable"
        )
        return result_detail_error(reason, proposal_id)
    chunk_index = request.get("chunk_index", 0)
    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int):
        return result_detail_error(
            "operator_proposal_result_detail_chunk_index_invalid",
            proposal_id,
        )
    chunk_count = max(
        1,
        (len(content) + DETAIL_CHUNK_BYTES - 1) // DETAIL_CHUNK_BYTES,
    )
    if chunk_index < 0 or chunk_index >= chunk_count:
        return result_detail_error(
            "operator_proposal_result_detail_chunk_index_invalid",
            proposal_id,
        )
    chunk = content[
        chunk_index * DETAIL_CHUNK_BYTES : (chunk_index + 1) * DETAIL_CHUNK_BYTES
    ]
    payload = {
        "type": "operator_proposal_result_detail",
        "contract_version": 2,
        "status": "ok",
        "success": True,
        "proposal_id": proposal_id,
        "result_digest": expected_digest,
        "current_state": {
            "proposal_status": str(getattr(proposal, "status", "unknown")),
            "updated_at": _iso_or_none(getattr(proposal, "updated_at", None)),
            "result_detail_available": bool(
                getattr(proposal, "result_detail_available", False)
            ),
            "result_bytes": int(getattr(proposal, "result_bytes", 0) or 0),
            "result_evidence_pinned": bool(
                getattr(proposal, "result_evidence_pinned", False)
            ),
        },
        "content": {
            "encoding": "base64",
            "media_type": (
                "application/vnd.automixer.operator-proposal-result+json;"
                "version=2"
            ),
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "raw_chunk_bytes": len(chunk),
            "total_raw_bytes": len(content),
            "chunk_sha256": hashlib.sha256(chunk).hexdigest(),
            "content_sha256": hashlib.sha256(content).hexdigest(),
            "data": base64.b64encode(chunk).decode("ascii"),
        },
        "wire": {},
    }
    finalize_wire_size(payload)
    if payload["wire"]["serialized_bytes"] > WIRE_MAX_BYTES:
        return result_detail_error(
            "operator_proposal_result_detail_response_budget_invariant_breached",
            proposal_id,
        )
    return payload


def compact_proposal_reference(proposal: Any) -> Dict[str, Any]:
    row = compact_proposal_row(proposal)
    return {
        "id": row["id"],
        "review_contract_version": row["review_contract_version"],
        "review_content_digest": row["review_content_digest"],
        "status": row["status"],
        "expires_at": row["expires_at"],
        "can_apply": row["can_apply"],
        "apply_blocker_codes": row["apply_blocker_codes"],
        "apply_blocker_count": row["apply_blocker_count"],
        "result_detail_available": bool(
            getattr(proposal, "result_detail_available", False)
        ),
        "hazard_scope": row["hazard_scope"],
    }
