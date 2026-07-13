"""Read-only operator cockpit state handlers."""

import asyncio
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from operator_mode_policy import CURRENT_PROVEN_LIVE_WRITE_KINDS
from operator_proposal_queue import soundcheck_proposal_apply_blocker
from decision_queue_wire_contract import (
    DecisionQueueSnapshotManager,
    FinalizedApplyResult,
    LEGACY_CLIENT_ECHO_CONSUMER_HANDLERS,
    MUTATION_CAPABLE_LEGACY_HANDLER_NAMES,
    OperatorProposalSequenceStateManager,
    POST_EXECUTOR_BLOCKED_HANDLER_NAMES,
    _freeze_trusted_apply_receipt,
    _freeze_trusted_prewrite_no_send_receipt,
    _freeze_trusted_sequence_execution_receipt,
    _freeze_verified_apply_identity,
    build_proposal_detail_response,
    build_proposal_result_detail_response,
    build_v1_decision_queue_shim,
    collect_compact_rows,
    collect_global_hazards,
    compact_trusted_apply_receipt_reference,
    compact_proposal_reference,
    canonical_json_bytes,
    classify_trusted_apply_reason_code,
    decision_queue_error,
    result_detail_error,
    sha256_digest,
    terminal_result_proves_verified_apply,
    terminal_result_proves_verified_restored_proof,
)
from supervised_proof_capability import validate_m110_capability_apply_request
from operator_product_state import (
    build_agent_communications_snapshot,
    build_bus_main_dca_snapshot,
    build_channel_intelligence,
    build_channel_processing_snapshot,
    build_operator_agent_roster,
    build_processing_pipeline_snapshot,
    build_channel_inventory,
    build_connection_topology,
    build_dashboard_snapshot,
    build_decision_queue,
    build_decision_traces,
    build_dlive_state_snapshot,
    build_fx_inserts_snapshot,
    build_groups_buses_snapshot,
    build_matrix_snapshot,
    build_memory_knowledge_snapshot,
    build_replay_runs,
    build_routing_sends_snapshot,
    build_rules_library_snapshot,
    build_safety_monitor,
    build_snapshots_scenes_snapshot,
    build_telemetry_alerts_snapshot,
    build_training_sessions_snapshot,
    build_user_presets_snapshot,
)
from operator_analysis import build_operator_analysis_report
from processing_pipeline_preview import build_main_pipeline_preview
from source_identity_proposal_queue_importer import (
    build_source_identity_proposal_queue_importer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_index import (
    build_operator_proposal_apply_sequence_policy_memory_index,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_review import (
    build_operator_proposal_apply_sequence_policy_memory_trust_review,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_plan_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_plan_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_draft_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_draft_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger import (
    build_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_plan import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_plan,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_contract import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_executor import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview,
)
from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan import (
    build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan,
)


VERIFIED_MANUAL_READBACK_STATUSES = {"verified_from_cache", "verified_from_physical_readback"}
APPLY_DELTA_LIMITS_DB = {"fader": 1.0, "gain": 0.5}
CUMULATIVE_DELTA_LIMITS_DB = {"fader": 3.0, "gain": 1.5}
CUMULATIVE_DELTA_WINDOW_SEC = 300.0
PROVEN_LIVE_WRITE_KIND_SET = set(CURRENT_PROVEN_LIVE_WRITE_KINDS)
WING_OPERATOR_PROOF_RUNNER_SOURCE = "wing_operator_proposal_validation_runner"
WING_PROOF_SURFACE_CONTRACT_FIELDS = (
    "wing_surface",
    "wing_address",
    "proof_direction",
)
WING_PROOF_SURFACE_CONTRACT_BY_VALUE_TYPE = {
    "fader": {
        "wing_surface": "channel_fader",
        "address_template": "/ch/{channel}/fdr",
        "proof_direction": "negative_db_move_only",
    },
    "gain": {
        "wing_surface": "input_trim",
        "address_template": "/ch/{channel}/in/set/trim",
        "proof_direction": "negative_input_trim_move_only",
    },
}


@dataclass(frozen=True)
class FrozenApplyRequestFacts:
    """Scalar request facts captured before a proposal can be claimed."""

    approved: bool
    approval_id: str
    approval_token: str
    idempotency_key: str
    expected_review_contract_version: object
    expected_review_content_digest: str


def manual_write_result_verified(result):
    readback_status = result.get("readback_status")
    proof_status = result.get("proof_status")
    if proof_status is not None and proof_status != "verified":
        return False
    return readback_status in VERIFIED_MANUAL_READBACK_STATUSES


def register_handlers(server):
    sequence_state_manager = getattr(
        server,
        "_operator_proposal_sequence_state_manager",
        None,
    )
    if sequence_state_manager is None:
        sequence_state_manager = OperatorProposalSequenceStateManager()
        server._operator_proposal_sequence_state_manager = sequence_state_manager

    def proposal_id_or_index(data):
        return data.get("proposal_id") or data.get("id"), data.get("index")

    def resolve_sequence_stage_payload(
        websocket,
        data,
        *,
        expected_stage,
    ):
        reference = data.get("sequence_state")
        if reference is None:
            return None, None
        resolved = sequence_state_manager.resolve(
            websocket,
            reference,
            expected_stage=expected_stage,
            consume=True,
        )
        if not resolved.get("resolved"):
            return None, resolved.get("reason") or "sequence_state_invalid"
        payload = resolved.get("payload")
        if not isinstance(payload, dict):
            return None, "sequence_state_payload_invalid"
        return payload, None

    def bind_sequence_stage_payload(websocket, payload, *, stage):
        stored = sequence_state_manager.store(websocket, stage, payload)
        if not stored.get("stored"):
            return None, stored.get("reason") or "sequence_state_unavailable"
        return {**payload, "sequence_state": stored["reference"]}, None

    def allowed_live_write_kinds_for_safety_surface(operator_mode=None):
        status = (
            operator_mode
            if isinstance(operator_mode, dict)
            else server.get_operator_mode_status()
        )
        if status.get("wing_deployment"):
            return list(
                status.get("capabilities", {}).get("allowed_live_write_kinds") or []
            )
        return list(CURRENT_PROVEN_LIVE_WRITE_KINDS)

    def live_write_kind_set_for_safety_surface(operator_mode=None):
        return set(allowed_live_write_kinds_for_safety_surface(operator_mode))

    def coerce_float(value):
        if isinstance(value, bool):
            return None
        try:
            if value is None:
                return None
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    def coerce_channel(value):
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            parsed = value
        elif isinstance(value, float):
            if not math.isfinite(value) or not value.is_integer():
                return None
            parsed = int(value)
        elif isinstance(value, str):
            raw = value.strip()
            if not raw or not raw.lstrip("+-").isdigit():
                return None
            parsed = int(raw)
        else:
            return None
        return parsed if 1 <= parsed <= 48 else None

    def is_operator_authored(proposal, approval_token):
        validator = getattr(server, "validate_operator_approval_token", None)
        if not callable(validator):
            return False
        try:
            return validator(
                token=approval_token,
                proposal_id=getattr(proposal, "id", None),
                requested_change=(
                    getattr(proposal, "requested_change", None) or {}
                ),
            ) is True
        except Exception:
            return False

    def expected_wing_surface_contract(normalized_change):
        value_type = str(normalized_change.get("value_type") or "").strip().lower()
        contract = WING_PROOF_SURFACE_CONTRACT_BY_VALUE_TYPE.get(value_type)
        if not contract:
            return None
        channel = normalized_change.get("channel")
        address = contract["address_template"].format(channel=channel)
        return {
            "wing_surface": contract["wing_surface"],
            "wing_address": address,
            "proof_direction": contract["proof_direction"],
            "address_template": contract["address_template"],
        }

    def observed_wing_surface_contract(change, proposal, *, linked):
        sources = []
        if isinstance(change, dict):
            sources.append(("requested_change", change))
            raw_change = change.get("raw")
            if isinstance(raw_change, dict):
                sources.append(("requested_change.raw", raw_change))
        if not linked and proposal is not None:
            raw_proposal = getattr(proposal, "raw", {})
            if isinstance(raw_proposal, dict):
                sources.append(("proposal.raw", raw_proposal))

        observed = {}
        for field in WING_PROOF_SURFACE_CONTRACT_FIELDS:
            values = []
            for source_name, source in sources:
                if field not in source:
                    continue
                values.append({
                    "source": source_name,
                    "value": str(source.get(field) or "").strip(),
                })
            if values:
                observed[field] = values
        return observed

    def validate_wing_surface_contract(change, normalized_change, proposal, *, linked):
        expected = expected_wing_surface_contract(normalized_change)
        if expected is None:
            return None
        observed = observed_wing_surface_contract(change, proposal, linked=linked)
        proposal_source = str(getattr(proposal, "source", "") or "").strip()
        contract_required = proposal_source == WING_OPERATOR_PROOF_RUNNER_SOURCE or bool(observed)
        if not contract_required:
            for field in WING_PROOF_SURFACE_CONTRACT_FIELDS:
                normalized_change[field] = expected[field]
            return None

        missing = [
            field
            for field in WING_PROOF_SURFACE_CONTRACT_FIELDS
            if field not in observed
        ]
        base_error = {
            "channel": normalized_change["channel"],
            "value_type": normalized_change["value_type"],
            "address": normalized_change["address"],
            "expected_wing_surface_contract": expected,
            "observed_wing_surface_contract": observed,
            "wing_surface_contract_required": contract_required,
            "wing_surface_contract_source": proposal_source or "requested_change",
        }
        if missing:
            return {
                **base_error,
                "reason": "wing_surface_contract_missing",
                "missing_contract_fields": missing,
            }

        mismatches = []
        for field, observations in observed.items():
            expected_value = expected[field]
            for item in observations:
                if item["value"] != expected_value:
                    mismatches.append({
                        "field": field,
                        "source": item["source"],
                        "expected": expected_value,
                        "actual": item["value"],
                    })
        if mismatches:
            return {
                **base_error,
                "reason": "wing_surface_contract_mismatch",
                "mismatched_contract_fields": mismatches,
            }

        for field in WING_PROOF_SURFACE_CONTRACT_FIELDS:
            normalized_change[field] = expected[field]
        return None

    def wing_proof_direction_blocker(change, normalized_change, proposal, *, linked):
        expected = expected_wing_surface_contract(normalized_change)
        if expected is None:
            return None
        observed = observed_wing_surface_contract(change, proposal, linked=linked)
        proposal_source = str(getattr(proposal, "source", "") or "").strip()
        proof_direction_enforced = (
            proposal_source == WING_OPERATOR_PROOF_RUNNER_SOURCE or bool(observed)
        )
        if not proof_direction_enforced:
            return None

        proof_direction = str(
            normalized_change.get("proof_direction") or expected["proof_direction"]
        ).strip()
        if proof_direction not in {
            "negative_db_move_only",
            "negative_input_trim_move_only",
        }:
            return None
        delta = coerce_float(normalized_change.get("delta_db"))
        if delta is None or delta < 0:
            return None
        return {
            "reason": "wing_proof_direction_requires_negative_delta",
            "channel": normalized_change["channel"],
            "value_type": normalized_change["value_type"],
            "delta_db": normalized_change["delta_db"],
            "proof_direction": proof_direction,
            "expected_wing_surface_contract": expected,
        }

    def operator_apply_delta_history():
        history = getattr(server, "_operator_apply_delta_history", None)
        if not isinstance(history, dict):
            history = {}
            setattr(server, "_operator_apply_delta_history", history)
        return history

    def cumulative_delta_blocker(normalized_change, *, update_history=True):
        value_type = normalized_change.get("value_type")
        limit = CUMULATIVE_DELTA_LIMITS_DB.get(value_type)
        if limit is None:
            return None
        delta = coerce_float(normalized_change.get("delta_db"))
        if delta is None:
            return None
        key = (int(normalized_change["channel"]), value_type)
        now = time.monotonic()
        if update_history:
            history = operator_apply_delta_history()
        else:
            history = getattr(server, "_operator_apply_delta_history", {})
            if not isinstance(history, dict):
                history = {}
        entries = [
            entry
            for entry in history.get(key, [])
            if now - float(entry.get("ts", 0.0)) <= CUMULATIVE_DELTA_WINDOW_SEC
        ]
        if update_history:
            history[key] = entries
        used = round(sum(abs(float(entry.get("delta_db", 0.0))) for entry in entries), 3)
        requested = round(abs(float(delta)), 3)
        if used + requested > limit:
            return {
                "reason": "cumulative_delta_limit_exceeded",
                "channel": normalized_change["channel"],
                "value_type": value_type,
                "delta_db": normalized_change["delta_db"],
                "cumulative_delta_db": used,
                "requested_delta_db": requested,
                "max_cumulative_delta_db": limit,
                "window_sec": CUMULATIVE_DELTA_WINDOW_SEC,
            }
        return None

    def remember_applied_delta(normalized_change, apply_result):
        if apply_result.get("status") != "applied":
            return
        if not manual_write_result_verified(apply_result):
            return
        if not apply_result.get("rollback_available"):
            return
        delta = coerce_float(normalized_change.get("delta_db"))
        if delta is None:
            return
        key = (int(normalized_change["channel"]), normalized_change["value_type"])
        history = operator_apply_delta_history()
        history.setdefault(key, []).append({
            "ts": time.monotonic(),
            "delta_db": delta,
            "proposal_id": apply_result.get("proposal_id"),
            "approval_id": apply_result.get("approval_id"),
        })

    def trace_payload_for_result(
        *,
        proposal=None,
        operator_mode=None,
        requested_change=None,
        normalized_change=None,
        apply_result=None,
        idempotency_key=None,
        status="blocked",
        blocked_reason=None,
        metadata=None,
    ):
        change = normalized_change or requested_change or {}
        result = apply_result or {}
        value_type = str(result.get("value_type") or change.get("value_type") or getattr(proposal, "kind", "") or "").strip().lower()
        target = result.get("channel", change.get("channel", getattr(proposal, "target", None)))
        current_value = result.get("old_value", change.get("old_value", change.get("current_value")))
        proposed_value = result.get("effective_value", result.get("requested_value", change.get("effective_value", change.get("value"))))
        delta = result.get("delta_db", change.get("delta_db"))
        if delta is None:
            current_float = coerce_float(current_value)
            proposed_float = coerce_float(proposed_value)
            if current_float is not None and proposed_float is not None:
                delta = round(proposed_float - current_float, 3)
        mode = (operator_mode or {}).get("mode") if isinstance(operator_mode, dict) else getattr(server, "operator_mode", None)
        readback = None
        if result.get("readback_status") or result.get("readback_value") is not None:
            readback = {
                "status": result.get("readback_status"),
                "source": result.get("readback_source"),
                "value": result.get("readback_value"),
            }
            cache_freshness = result.get("readback_cache_freshness")
            if isinstance(cache_freshness, dict):
                readback["cache_freshness"] = cache_freshness
        rollback = None
        if result.get("rollback_available") is not None or result.get("rollback_status") is not None:
            rollback = {
                "available": result.get("rollback_available"),
                "value": result.get("rollback_value"),
                "status": result.get("rollback_status"),
                "readback_status": result.get("rollback_readback_status"),
                "proof_status": result.get("rollback_proof_status"),
            }
        pre_read = None
        pre_read_status = result.get("pre_read_status", change.get("pre_read_status"))
        pre_read_value = result.get("pre_read_value", change.get("pre_read_value", current_value))
        if pre_read_status or pre_read_value is not None:
            pre_read = {
                "status": pre_read_status,
                "source": result.get("pre_read_source", change.get("pre_read_source")),
                "value": pre_read_value,
            }
            cache_freshness = (
                result.get("pre_read_cache_freshness")
                or result.get("cache_freshness")
                or change.get("pre_read_cache_freshness")
                or change.get("cache_freshness")
            )
            if isinstance(cache_freshness, dict):
                pre_read["cache_freshness"] = cache_freshness
        return {
            "idempotency_key": idempotency_key,
            "proposal_id": getattr(proposal, "id", None),
            "source": getattr(proposal, "source", "operator_proposal_queue") if proposal else "operator_apply",
            "mode": mode,
            "target": target,
            "action_type": value_type or "proposal",
            "status": status,
            "current_value": current_value,
            "proposed_value": proposed_value,
            "delta": delta,
            "safety_limits": {
                "allowed_live_write_kinds": allowed_live_write_kinds_for_safety_surface(),
                "max_delta_db": APPLY_DELTA_LIMITS_DB.get(value_type),
                "upward_auto_generated_moves_allowed": False,
            },
            "pre_read": pre_read,
            "readback": readback,
            "rollback": rollback,
            "proof_status": result.get("proof_status", "blocked" if blocked_reason else "unverified"),
            "blocked_reason": blocked_reason,
            "metadata": metadata or {},
        }

    def record_apply_trace(**kwargs):
        recorder = getattr(server, "record_automation_decision_trace", None)
        if callable(recorder):
            recorder(**trace_payload_for_result(**kwargs))

    async def validate_change_before_apply(
        change,
        *,
        proposal,
        operator_authored,
        operator_mode,
        linked=False,
        update_delta_history=True,
    ):
        normalized, error = normalize_linked_change(change)
        if error:
            return None, {
                "reason": error,
                "requested_change": change,
            }
        surface_error = validate_wing_surface_contract(
            change,
            normalized,
            proposal,
            linked=linked,
        )
        if surface_error:
            return normalized, surface_error
        read_value = None
        read_helper = getattr(server, "_manual_write_read_value", None)
        if callable(read_helper):
            read_value = await read_helper(
                channel=normalized["channel"],
                value_type=normalized["value_type"],
                address=normalized["address"],
                prefer_physical=True,
            )
            old_value = read_value.get("value")
        else:
            state = getattr(getattr(server, "mixer_client", None), "state", {})
            old_value = state.get(normalized["address"]) if isinstance(state, dict) else None
        old_float = coerce_float(old_value)
        if old_float is None:
            return None, {
                "reason": "linked_pre_read_unavailable" if linked else "pre_read_unavailable",
                "channel": normalized["channel"],
                "value_type": normalized["value_type"],
                "address": normalized["address"],
                "pre_read_status": (read_value or {}).get("status"),
                "pre_read_source": (read_value or {}).get("source"),
            }
        cache_freshness_blocker = getattr(server, "_cache_freshness_blocks_pre_read", None)
        if callable(cache_freshness_blocker) and cache_freshness_blocker(read_value):
            return None, {
                "reason": "linked_pre_read_cache_not_fresh" if linked else "pre_read_cache_not_fresh",
                "channel": normalized["channel"],
                "value_type": normalized["value_type"],
                "address": normalized["address"],
                "pre_read_status": (read_value or {}).get("status"),
                "pre_read_source": (read_value or {}).get("source"),
                "pre_read_value": old_value,
                "cache_freshness": (read_value or {}).get("cache_freshness"),
            }
        normalized["old_value"] = old_value
        normalized["pre_read_value"] = old_value
        normalized["pre_read_status"] = (read_value or {}).get("status")
        normalized["pre_read_source"] = (read_value or {}).get("source")
        normalized["pre_read_cache_freshness"] = (read_value or {}).get("cache_freshness")
        normalized["delta_db"] = round(float(normalized["effective_value"]) - old_float, 3)
        max_delta = APPLY_DELTA_LIMITS_DB.get(normalized["value_type"])
        if max_delta is not None and abs(normalized["delta_db"]) > max_delta:
            return normalized, {
                "reason": "delta_limit_exceeded",
                "channel": normalized["channel"],
                "value_type": normalized["value_type"],
                "delta_db": normalized["delta_db"],
                "max_delta_db": max_delta,
            }
        if normalized["delta_db"] > 0 and not operator_authored:
            return normalized, {
                "reason": "upward_move_requires_operator_authored_proposal",
                "channel": normalized["channel"],
                "value_type": normalized["value_type"],
                "delta_db": normalized["delta_db"],
            }
        direction_error = wing_proof_direction_blocker(
            change,
            normalized,
            proposal,
            linked=linked,
        )
        if direction_error:
            return normalized, direction_error
        cumulative_error = cumulative_delta_blocker(
            normalized,
            update_history=update_delta_history,
        )
        if cumulative_error:
            return normalized, cumulative_error
        return normalized, None

    async def handle_get_dashboard_snapshot(websocket, data):
        await server.send_to_client(websocket, build_dashboard_snapshot(server))

    async def handle_get_channel_inventory(websocket, data):
        total_channels = data.get("total_channels", 24)
        await server.send_to_client(
            websocket,
            build_channel_inventory(server, total_channels=total_channels),
        )

    async def handle_get_channel_processing_snapshot(websocket, data):
        total_channels = data.get("total_channels", 24)
        await server.send_to_client(
            websocket,
            build_channel_processing_snapshot(server, total_channels=total_channels),
        )

    async def handle_get_decision_queue(websocket, data):
        contract_version = data.get("contract_version", 1)
        if contract_version == 1:
            payload = build_v1_decision_queue_shim(server)
        elif contract_version == 2:
            manager = getattr(server, "_decision_queue_snapshot_manager", None)
            if manager is None:
                manager = DecisionQueueSnapshotManager()
                server._decision_queue_snapshot_manager = manager
            payload = manager.response(server, websocket, data)
        else:
            payload = decision_queue_error(
                "decision_queue_contract_version_unsupported"
            )
        await server.send_to_client(websocket, payload)

    async def handle_get_operator_proposal_detail(websocket, data):
        if data.get("contract_version") != 2:
            payload = build_proposal_detail_response(
                server.operator_proposal_queue,
                {**data, "expected_review_contract_version": None},
                server=server,
            )
        else:
            payload = build_proposal_detail_response(
                server.operator_proposal_queue,
                data,
                server=server,
            )
        await server.send_to_client(websocket, payload)

    async def handle_get_operator_proposal_result_detail(websocket, data):
        if data.get("contract_version") != 2:
            payload = result_detail_error(
                "operator_proposal_result_contract_version_required",
                str(data.get("proposal_id") or "").strip() or None,
            )
        else:
            payload = build_proposal_result_detail_response(
                server.operator_proposal_queue,
                data,
            )
        await server.send_to_client(websocket, payload)

    async def handle_get_decision_traces(websocket, data):
        limit = data.get("limit", 50)
        await server.send_to_client(websocket, build_decision_traces(server, limit=limit))

    async def handle_get_safety_monitor(websocket, data):
        del data
        await server.send_to_client(websocket, build_safety_monitor(server))

    async def handle_get_replay_runs(websocket, data):
        limit = data.get("limit", 20)
        await server.send_to_client(websocket, build_replay_runs(server, limit=limit))

    async def handle_start_replay_run(websocket, data):
        try:
            from agentic_method_foundation import build_offline_teacher_examples_from_channel_inventory
            from offline_validation_harness import compact_stamp, run_offline_validation, write_offline_validation_artifacts
        except ImportError:  # pragma: no cover - package import compatibility
            from backend.agentic_method_foundation import build_offline_teacher_examples_from_channel_inventory
            from backend.offline_validation_harness import compact_stamp, run_offline_validation, write_offline_validation_artifacts

        total_channels = int(data.get("total_channels", 16) or 16)
        run_id = str(data.get("run_id") or "").strip() or f"{compact_stamp()}_offline_operator_validation"
        result = run_offline_validation(total_channels=total_channels)
        inventory = build_channel_inventory(server, total_channels=total_channels)
        teacher_examples = build_offline_teacher_examples_from_channel_inventory(
            inventory.get("channels", []),
            run_id=run_id,
            reference_metrics={
                "offline_validation_status": result.get("status"),
                "offline_validation_summary": result.get("summary", {}),
            },
        )
        paths = write_offline_validation_artifacts(
            result,
            artifact_root=getattr(server, "_replay_artifacts_dir", "artifacts/replay"),
            run_id=run_id,
            teacher_examples=teacher_examples,
        )
        recorder = getattr(server, "record_automation_decision_trace", None)
        if callable(recorder):
            recorder(
                source="offline_replay_v1",
                mode="offline",
                target=paths.get("run_id"),
                action_type="replay_run",
                status=result.get("status"),
                safety_limits={
                    "offline_only": True,
                    "live_write": False,
                    "mixer_transport_allowed": False,
                    "allowed_live_write_kinds": [],
                },
                proof_status="offline",
                blocked_reason=None if result.get("status") in {"passed", "clear"} else result.get("reason"),
                metadata={
                    "run_id": paths.get("run_id"),
                    "artifact_paths": paths,
                    "summary": result.get("summary", {}),
                    "foundation_teacher_example_count": len(teacher_examples),
                    "connection_policy": result.get("connection_policy"),
                    "write_policy": result.get("write_policy"),
                    "forbidden_messages": result.get("forbidden_messages", []),
                    "offline_only": True,
                    "live_write": False,
                },
            )
        payload = {
            "type": "replay_run_started",
            "status": result.get("status"),
            "success": result.get("status") in {"passed", "clear"},
            "offline_only": True,
            "live_write": False,
            "run_id": (paths.get("run_dir") or "").split("/")[-1],
            "result": result,
            "artifacts": paths,
            "foundation_teacher_example_count": len(teacher_examples),
        }
        await server.send_to_client(websocket, payload)

    async def handle_get_channel_intelligence(websocket, data):
        await server.send_to_client(
            websocket,
            build_channel_intelligence(server, total_channels=data.get("total_channels", 48)),
        )

    async def handle_get_connection_topology(websocket, data):
        await server.send_to_client(websocket, build_connection_topology(server))

    async def handle_get_operator_agent_roster(websocket, data):
        del data
        await server.send_to_client(websocket, build_operator_agent_roster(server))

    async def handle_get_groups_buses_snapshot(websocket, data):
        del data
        await server.send_to_client(websocket, build_groups_buses_snapshot(server))

    async def handle_get_bus_main_dca_snapshot(websocket, data):
        await server.send_to_client(
            websocket,
            build_bus_main_dca_snapshot(
                server,
                bus_count=data.get("bus_count", 16),
                main_count=data.get("main_count", 4),
                dca_count=data.get("dca_count", 16),
            ),
        )

    async def handle_get_matrix_snapshot(websocket, data):
        await server.send_to_client(
            websocket,
            build_matrix_snapshot(
                server,
                matrix_count=data.get("matrix_count", 8),
            ),
        )

    async def handle_get_routing_sends_snapshot(websocket, data):
        await server.send_to_client(
            websocket,
            build_routing_sends_snapshot(
                server,
                total_channels=data.get("total_channels", 24),
                send_count=data.get("send_count", 16),
                main_count=data.get("main_count", 1),
                matrix_send_count=data.get("matrix_send_count", 8),
                output_groups=data.get("output_groups", ("MOD", "CRD")),
                output_count=data.get("output_count", 64),
            ),
        )

    async def handle_get_fx_inserts_snapshot(websocket, data):
        await server.send_to_client(
            websocket,
            build_fx_inserts_snapshot(
                server,
                fx_slot_count=data.get("fx_slot_count", 16),
                total_channels=data.get("total_channels", 24),
                bus_count=data.get("bus_count", 16),
                main_count=data.get("main_count", 4),
                matrix_count=data.get("matrix_count", 8),
                parameter_count=data.get("parameter_count", 32),
            ),
        )

    async def handle_get_dlive_state_snapshot(websocket, data):
        await server.send_to_client(
            websocket,
            build_dlive_state_snapshot(
                server,
                total_channels=data.get("total_channels", 48),
            ),
        )

    async def handle_get_snapshots_scenes_snapshot(websocket, data):
        await server.send_to_client(
            websocket,
            build_snapshots_scenes_snapshot(
                server,
                snapshot_count=data.get("snapshot_count", 200),
            ),
        )

    async def handle_get_user_presets_snapshot(websocket, data):
        await server.send_to_client(
            websocket,
            build_user_presets_snapshot(
                server,
                preset_count=data.get("preset_count", 200),
            ),
        )

    async def handle_get_memory_knowledge_snapshot(websocket, data):
        del data
        await server.send_to_client(websocket, build_memory_knowledge_snapshot(server))

    async def handle_get_training_sessions_snapshot(websocket, data):
        del data
        await server.send_to_client(websocket, build_training_sessions_snapshot(server))

    async def handle_get_telemetry_alerts_snapshot(websocket, data):
        await server.send_to_client(
            websocket,
            build_telemetry_alerts_snapshot(
                server,
                limit=data.get("limit", 50),
                session_limit=data.get("session_limit", 8),
            ),
        )

    async def handle_get_rules_library_snapshot(websocket, data):
        del data
        await server.send_to_client(websocket, build_rules_library_snapshot(server))

    async def handle_get_agent_communications_snapshot(websocket, data):
        del data
        await server.send_to_client(websocket, build_agent_communications_snapshot(server))

    async def handle_get_processing_pipeline_snapshot(websocket, data):
        await server.send_to_client(
            websocket,
            build_processing_pipeline_snapshot(
                server,
                include_methods=bool(data.get("include_methods", True)),
            ),
        )

    async def handle_get_main_pipeline_preview(websocket, data):
        await server.send_to_client(
            websocket,
            build_main_pipeline_preview(
                server,
                total_channels=data.get("total_channels", 24),
                include_methods=bool(data.get("include_methods", False)),
            ),
        )

    async def handle_get_operator_mode(websocket, data):
        del data
        await server.send_to_client(websocket, server.get_operator_mode_status())

    async def handle_set_operator_mode(websocket, data):
        await server.send_to_client(
            websocket,
            server.set_operator_mode(
                data.get("mode"),
                reason=data.get("reason", "websocket_operator_mode_change"),
            ),
        )

    async def handle_run_operator_analysis(websocket, data):
        await server.send_to_client(
            websocket,
            build_operator_analysis_report(
                server,
                total_channels=data.get("total_channels", 24),
                create_proposals=data.get("create_proposals"),
            ),
        )

    async def handle_import_safe_gain_suggestions(websocket, data):
        del data
        await server.send_to_client(
            websocket,
            server.import_safe_gain_suggestions_to_operator_queue(),
        )

    async def handle_import_soundcheck_recommendations(websocket, data):
        await server.send_to_client(
            websocket,
            server.import_soundcheck_recommendations_to_operator_queue(data.get("bundle")),
        )

    async def handle_import_source_identity_queue_ordering_plan(websocket, data):
        pipeline_preview = None
        queue_ordering_plan = data.get("queue_ordering_plan")
        proposal_payloads = data.get("proposal_payloads")
        if not isinstance(queue_ordering_plan, dict) or proposal_payloads is None:
            pipeline_preview = build_main_pipeline_preview(
                server,
                total_channels=data.get("total_channels", 24),
                include_methods=False,
            )
            queue_ordering_plan = pipeline_preview.get(
                "source_identity_proposal_queue_ordering_plan"
            )
            proposal_payloads = pipeline_preview.get("proposals", [])

        apply_requested = data.get("apply") is True
        operator_ack = data.get("operator_ack") is True
        result = build_source_identity_proposal_queue_importer(
            queue_ordering_plan=queue_ordering_plan,
            proposal_payloads=proposal_payloads,
            target_queue=(
                server.operator_proposal_queue
                if apply_requested and operator_ack
                else None
            ),
            operator_mode=server.get_operator_mode_status(),
            apply=apply_requested,
            operator_ack=operator_ack,
        )
        result["handler"] = "import_source_identity_queue_ordering_plan"
        result["pipeline_preview_generated"] = pipeline_preview is not None
        await server.send_to_client(websocket, result)

    async def handle_import_operator_proposal_apply_sequence_policy_queue_ordering_plan(
        websocket,
        data,
    ):
        pipeline_preview = None
        queue_ordering_plan = data.get("queue_ordering_plan")
        proposal_payloads = data.get("proposal_payloads")
        if not isinstance(queue_ordering_plan, dict) or proposal_payloads is None:
            pipeline_preview = build_main_pipeline_preview(
                server,
                total_channels=data.get("total_channels", 24),
                include_methods=False,
            )
            queue_ordering_plan = pipeline_preview.get(
                "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan"
            )
            proposal_payloads = pipeline_preview.get("proposals", [])

        apply_requested = data.get("apply") is True
        operator_ack = data.get("operator_ack") is True
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer(
                queue_ordering_plan=queue_ordering_plan,
                proposal_payloads=proposal_payloads,
                target_queue=(
                    server.operator_proposal_queue
                    if apply_requested and operator_ack
                    else None
                ),
                operator_mode=server.get_operator_mode_status(),
                apply=apply_requested,
                operator_ack=operator_ack,
            )
        )
        result["handler"] = (
            "import_operator_proposal_apply_sequence_policy_queue_ordering_plan"
        )
        result["pipeline_preview_generated"] = pipeline_preview is not None
        await server.send_to_client(websocket, result)

    async def handle_plan_operator_proposal_apply_sequence_policy_queue_import_preflight(
        websocket,
        data,
    ):
        pending_actions = server.operator_proposal_queue.pending_actions()
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan(
                pending_actions=pending_actions,
                idempotency_key_prefix=data.get(
                    "idempotency_key_prefix",
                    "sequence-policy-queue-import",
                ),
                limit=data.get("limit", 20),
            )
        )
        result["handler"] = (
            "plan_operator_proposal_apply_sequence_policy_queue_import_preflight"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_preflight_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer(
                queue_import_preflight_plan=(
                    data.get("queue_import_preflight_plan")
                    or data.get("preflight_plan")
                    or data.get("plan")
                ),
                batch_preflight_result=(
                    data.get("batch_preflight_result")
                    or data.get("preflight_batch")
                    or data.get("batch")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_preflight_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_sequence_plan_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer(
                preflight_result_consumer=(
                    data.get("preflight_result_consumer")
                    or data.get("queue_import_preflight_result_consumer")
                    or data.get("consumer")
                ),
                sequence_plan_result=(
                    data.get("sequence_plan_result")
                    or data.get("sequence_plan")
                    or data.get("plan")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_sequence_plan_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_dry_run_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer(
                sequence_plan_result_consumer=(
                    data.get("sequence_plan_result_consumer")
                    or data.get("queue_import_sequence_plan_result_consumer")
                    or data.get("consumer")
                ),
                dry_run_result=(
                    data.get("dry_run_result")
                    or data.get("dry_run_payload")
                    or data.get("dry_run")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_dry_run_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_execution_gate_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer(
                dry_run_result_consumer=(
                    data.get("dry_run_result_consumer")
                    or data.get("queue_import_dry_run_result_consumer")
                    or data.get("consumer")
                ),
                execution_gate_result=(
                    data.get("execution_gate_result")
                    or data.get("execution_gate_payload")
                    or data.get("gate")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_execution_gate_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_execution_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer(
                execution_gate_result_consumer=(
                    data.get("execution_gate_result_consumer")
                    or data.get("queue_import_execution_gate_result_consumer")
                    or data.get("consumer")
                ),
                execution_result=(
                    data.get("execution_result")
                    or data.get("execution_result_payload")
                    or data.get("execution")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_execution_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_execution_audit_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer(
                execution_result_consumer=(
                    data.get("execution_result_consumer")
                    or data.get("queue_import_execution_result_consumer")
                    or data.get("consumer")
                ),
                execution_audit_result=(
                    data.get("execution_audit_result")
                    or data.get("execution_audit_payload")
                    or data.get("audit")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_execution_audit_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_audit_policy_gate_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer(
                execution_audit_result_consumer=(
                    data.get("execution_audit_result_consumer")
                    or data.get("queue_import_execution_audit_result_consumer")
                    or data.get("consumer")
                ),
                audit_policy_gate_result=(
                    data.get("audit_policy_gate_result")
                    or data.get("policy_gate_payload")
                    or data.get("policy_gate")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_audit_policy_gate_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_memory_draft_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer(
                audit_policy_gate_result_consumer=(
                    data.get("audit_policy_gate_result_consumer")
                    or data.get("queue_import_audit_policy_gate_result_consumer")
                    or data.get("consumer")
                ),
                memory_draft_result=(
                    data.get("memory_draft_result")
                    or data.get("memory_draft_payload")
                    or data.get("draft")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_memory_draft_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_memory_writer_dry_run_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer(
                memory_draft_result_consumer=(
                    data.get("memory_draft_result_consumer")
                    or data.get("queue_import_memory_draft_result_consumer")
                    or data.get("consumer")
                ),
                memory_writer_dry_run_result=(
                    data.get("memory_writer_dry_run_result")
                    or data.get("memory_writer_result")
                    or data.get("writer")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_memory_writer_dry_run_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_memory_writer_apply_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer(
                memory_writer_dry_run_result_consumer=(
                    data.get("memory_writer_dry_run_result_consumer")
                    or data.get("queue_import_memory_writer_dry_run_result_consumer")
                    or data.get("consumer")
                ),
                memory_writer_apply_result=(
                    data.get("memory_writer_apply_result")
                    or data.get("memory_writer_result")
                    or data.get("writer")
                ),
                root_path=data.get("root_path") or data.get("root"),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_memory_writer_apply_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_memory_index_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer(
                memory_writer_apply_result_consumer=(
                    data.get("memory_writer_apply_result_consumer")
                    or data.get("queue_import_memory_writer_apply_result_consumer")
                    or data.get("consumer")
                ),
                memory_index_result=(
                    data.get("memory_index_result")
                    or data.get("memory_index")
                    or data.get("index")
                ),
                current_sequence_review_freshness=(
                    data.get("current_sequence_review_freshness")
                    or data.get("sequence_review_freshness")
                    or data.get("freshness")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_memory_index_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_memory_trust_review_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer(
                memory_index_result_consumer=(
                    data.get("memory_index_result_consumer")
                    or data.get("queue_import_memory_index_result_consumer")
                    or data.get("consumer")
                ),
                memory_trust_review_result=(
                    data.get("memory_trust_review_result")
                    or data.get("memory_trust_review")
                    or data.get("review")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_memory_trust_review_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_memory_trust_acceptance_ledger_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer(
                memory_trust_review_result_consumer=(
                    data.get("memory_trust_review_result_consumer")
                    or data.get("queue_import_memory_trust_review_result_consumer")
                    or data.get("consumer")
                ),
                memory_trust_acceptance_ledger_result=(
                    data.get("memory_trust_acceptance_ledger_result")
                    or data.get("acceptance_ledger_result")
                    or data.get("acceptance_ledger")
                    or data.get("ledger")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_memory_trust_acceptance_ledger_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_plan_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer(
                memory_trust_acceptance_ledger_result_consumer=(
                    data.get("memory_trust_acceptance_ledger_result_consumer")
                    or data.get("promotion_memory_trust_acceptance_ledger_result_consumer")
                    or data.get("queue_import_memory_trust_acceptance_ledger_result_consumer")
                    or data.get("queue_import_promotion_memory_trust_acceptance_ledger_result_consumer")
                    or data.get("consumer")
                ),
                promotion_plan_result=(
                    data.get("promotion_plan_result")
                    or data.get("promotion_plan")
                    or data.get("plan")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_plan_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_contract_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer(
                promotion_plan_result_consumer=(
                    data.get("promotion_plan_result_consumer")
                    or data.get("queue_import_promotion_plan_result_consumer")
                    or data.get("consumer")
                ),
                promotion_contract_result=(
                    data.get("promotion_contract_result")
                    or data.get("promotion_contract")
                    or data.get("contract")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_contract_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_acceptance_ledger_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer(
                promotion_contract_result_consumer=(
                    data.get("promotion_contract_result_consumer")
                    or data.get("promotion_memory_contract_result_consumer")
                    or data.get("queue_import_promotion_contract_result_consumer")
                    or data.get("queue_import_promotion_memory_contract_result_consumer")
                    or data.get("consumer")
                ),
                promotion_acceptance_ledger_result=(
                    data.get("promotion_acceptance_ledger_result")
                    or data.get("promotion_acceptance_ledger")
                    or data.get("acceptance_ledger")
                    or data.get("ledger")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_acceptance_ledger_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_execution_plan_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer(
                promotion_acceptance_ledger_result_consumer=(
                    data.get("promotion_acceptance_ledger_result_consumer")
                    or data.get("promotion_memory_acceptance_ledger_result_consumer")
                    or data.get("queue_import_promotion_acceptance_ledger_result_consumer")
                    or data.get(
                        "queue_import_promotion_memory_acceptance_ledger_result_consumer"
                    )
                    or data.get("consumer")
                ),
                promotion_execution_plan_result=(
                    data.get("promotion_execution_plan_result")
                    or data.get("promotion_execution_plan")
                    or data.get("execution_plan")
                    or data.get("plan")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_execution_plan_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_executor_dry_run_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer(
                promotion_execution_plan_result_consumer=(
                    data.get("promotion_execution_plan_result_consumer")
                    or data.get("promotion_memory_execution_plan_result_consumer")
                    or data.get("queue_import_promotion_execution_plan_result_consumer")
                    or data.get(
                        "queue_import_promotion_memory_execution_plan_result_consumer"
                    )
                    or data.get("consumer")
                ),
                promotion_executor_dry_run_result=(
                    data.get("promotion_executor_dry_run_result")
                    or data.get("promotion_executor_dry_run")
                    or data.get("executor_dry_run")
                    or data.get("dry_run")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_executor_dry_run_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_executor_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer(
                promotion_executor_dry_run_result_consumer=(
                    data.get("promotion_executor_dry_run_result_consumer")
                    or data.get("promotion_memory_executor_dry_run_result_consumer")
                    or data.get("queue_import_promotion_executor_dry_run_result_consumer")
                    or data.get(
                        "queue_import_promotion_memory_executor_dry_run_result_consumer"
                    )
                    or data.get("consumer")
                ),
                promotion_executor_result=(
                    data.get("promotion_executor_result")
                    or data.get("promotion_executor")
                    or data.get("executor")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_executor_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_apply_audit_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer(
                promotion_executor_result_consumer=(
                    data.get("promotion_executor_result_consumer")
                    or data.get("promotion_memory_executor_result_consumer")
                    or data.get("queue_import_promotion_executor_result_consumer")
                    or data.get(
                        "queue_import_promotion_memory_executor_result_consumer"
                    )
                    or data.get("consumer")
                ),
                promotion_apply_audit_result=(
                    data.get("promotion_apply_audit_result")
                    or data.get("promotion_apply_audit")
                    or data.get("apply_audit")
                    or data.get("audit")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_apply_audit_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_rollback_verification_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer(
                promotion_apply_audit_result_consumer=(
                    data.get("promotion_apply_audit_result_consumer")
                    or data.get("promotion_memory_apply_audit_result_consumer")
                    or data.get("queue_import_promotion_apply_audit_result_consumer")
                    or data.get(
                        "queue_import_promotion_memory_apply_audit_result_consumer"
                    )
                    or data.get("consumer")
                ),
                promotion_rollback_verification_result=(
                    data.get("promotion_rollback_verification_result")
                    or data.get("promotion_rollback_verification")
                    or data.get("rollback_verification")
                    or data.get("rollback")
                    or data.get("verification")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_rollback_verification_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_sequence_policy_gate_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer(
                promotion_rollback_verification_result_consumer=(
                    data.get("promotion_rollback_verification_result_consumer")
                    or data.get(
                        "promotion_memory_rollback_verification_result_consumer"
                    )
                    or data.get("queue_import_promotion_rollback_verification_result_consumer")
                    or data.get(
                        "queue_import_promotion_memory_rollback_verification_result_consumer"
                    )
                    or data.get("consumer")
                ),
                promotion_sequence_policy_gate_result=(
                    data.get("promotion_sequence_policy_gate_result")
                    or data.get("promotion_sequence_policy_gate")
                    or data.get("sequence_policy_gate")
                    or data.get("policy_gate")
                    or data.get("gate")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_sequence_policy_gate_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_policy_consumer_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer(
                promotion_sequence_policy_gate_result_consumer=(
                    data.get("promotion_sequence_policy_gate_result_consumer")
                    or data.get("queue_import_promotion_sequence_policy_gate_result_consumer")
                    or data.get("promotion_memory_sequence_policy_gate_result_consumer")
                    or data.get("queue_import_promotion_memory_sequence_policy_gate_result_consumer")
                    or data.get("consumer")
                ),
                promotion_policy_consumer_result=(
                    data.get("promotion_policy_consumer_result")
                    or data.get("promotion_policy_consumer")
                    or data.get("policy_consumer")
                    or data.get("result")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_policy_consumer_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_ordering_preview_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer(
                promotion_policy_consumer_result_consumer=(
                    data.get("promotion_policy_consumer_result_consumer")
                    or data.get("queue_import_promotion_policy_consumer_result_consumer")
                    or data.get("promotion_memory_policy_consumer_result_consumer")
                    or data.get("queue_import_promotion_memory_policy_consumer_result_consumer")
                    or data.get("consumer")
                ),
                promotion_ordering_preview_result=(
                    data.get("promotion_ordering_preview_result")
                    or data.get("promotion_ordering_preview")
                    or data.get("ordering_preview")
                    or data.get("preview")
                    or data.get("result")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_ordering_preview_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_queue_ordering_plan_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer(
                promotion_ordering_preview_result_consumer=(
                    data.get("promotion_ordering_preview_result_consumer")
                    or data.get("queue_import_promotion_ordering_preview_result_consumer")
                    or data.get("promotion_memory_ordering_preview_result_consumer")
                    or data.get("queue_import_promotion_memory_ordering_preview_result_consumer")
                    or data.get("consumer")
                ),
                promotion_queue_ordering_plan_result=(
                    data.get("promotion_queue_ordering_plan_result")
                    or data.get("promotion_queue_ordering_plan")
                    or data.get("queue_ordering_plan")
                    or data.get("plan")
                    or data.get("result")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_queue_ordering_plan_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_queue_importer_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer(
                promotion_queue_ordering_plan_result_consumer=(
                    data.get("promotion_queue_ordering_plan_result_consumer")
                    or data.get("queue_import_promotion_queue_ordering_plan_result_consumer")
                    or data.get("promotion_memory_queue_ordering_plan_result_consumer")
                    or data.get("queue_import_promotion_memory_queue_ordering_plan_result_consumer")
                    or data.get("consumer")
                ),
                promotion_queue_importer_result=(
                    data.get("promotion_queue_importer_result")
                    or data.get("queue_importer_result")
                    or data.get("queue_importer")
                    or data.get("importer")
                    or data.get("result")
                ),
                idempotency_key_prefix=data.get(
                    "idempotency_key_prefix",
                    "sequence-policy-queue-import",
                ),
                limit=data.get("limit", 20),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_queue_importer_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_preflight_plan_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer(
                promotion_queue_importer_result_consumer=(
                    data.get("promotion_queue_importer_result_consumer")
                    or data.get("queue_import_promotion_queue_importer_result_consumer")
                    or data.get("promotion_memory_queue_importer_result_consumer")
                    or data.get("queue_import_promotion_memory_queue_importer_result_consumer")
                    or data.get("consumer")
                ),
                promotion_queue_import_preflight_plan_result=(
                    data.get("promotion_queue_import_preflight_plan_result")
                    or data.get("queue_import_preflight_plan")
                    or data.get("preflight_plan")
                    or data.get("plan")
                    or data.get("result")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_preflight_plan_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_batch_preflight_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer(
                promotion_preflight_plan_result_consumer=(
                    data.get("promotion_preflight_plan_result_consumer")
                    or data.get("queue_import_promotion_preflight_plan_result_consumer")
                    or data.get("promotion_memory_preflight_plan_result_consumer")
                    or data.get("queue_import_promotion_memory_preflight_plan_result_consumer")
                    or data.get("consumer")
                ),
                promotion_batch_preflight_result=(
                    data.get("promotion_batch_preflight_result")
                    or data.get("batch_preflight_result")
                    or data.get("preflight_result")
                    or data.get("result")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_batch_preflight_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_sequence_plan_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_plan_result_consumer(
                promotion_batch_preflight_result_consumer=(
                    data.get("promotion_batch_preflight_result_consumer")
                    or data.get("queue_import_promotion_batch_preflight_result_consumer")
                    or data.get("consumer")
                ),
                promotion_sequence_plan_result=(
                    data.get("promotion_sequence_plan_result")
                    or data.get("sequence_plan_result")
                    or data.get("sequence_plan")
                    or data.get("plan")
                    or data.get("result")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_sequence_plan_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_dry_run_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer(
                promotion_sequence_plan_result_consumer=(
                    data.get("promotion_sequence_plan_result_consumer")
                    or data.get("queue_import_promotion_sequence_plan_result_consumer")
                    or data.get("consumer")
                ),
                promotion_dry_run_result=(
                    data.get("promotion_dry_run_result")
                    or data.get("dry_run_result")
                    or data.get("dry_run")
                    or data.get("result")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_dry_run_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_execution_gate_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer(
                promotion_dry_run_result_consumer=(
                    data.get("promotion_dry_run_result_consumer")
                    or data.get("queue_import_promotion_dry_run_result_consumer")
                    or data.get("consumer")
                ),
                promotion_execution_gate_result=(
                    data.get("promotion_execution_gate_result")
                    or data.get("execution_gate_result")
                    or data.get("execution_gate")
                    or data.get("gate")
                    or data.get("result")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_execution_gate_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_execution_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_result_consumer(
                promotion_execution_gate_result_consumer=(
                    data.get("promotion_execution_gate_result_consumer")
                    or data.get("queue_import_promotion_execution_gate_result_consumer")
                    or data.get("consumer")
                ),
                promotion_execution_result=(
                    data.get("promotion_execution_result")
                    or data.get("execution_result")
                    or data.get("execution_result_payload")
                    or data.get("execution")
                    or data.get("result")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_execution_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_execution_audit_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer(
                promotion_execution_result_consumer=(
                    data.get("promotion_execution_result_consumer")
                    or data.get("queue_import_promotion_execution_result_consumer")
                    or data.get("consumer")
                ),
                promotion_execution_audit_result=(
                    data.get("promotion_execution_audit_result")
                    or data.get("execution_audit_result")
                    or data.get("execution_audit_payload")
                    or data.get("audit")
                    or data.get("result")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_execution_audit_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_audit_policy_gate_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer(
                promotion_execution_audit_result_consumer=(
                    data.get("promotion_execution_audit_result_consumer")
                    or data.get("queue_import_promotion_execution_audit_result_consumer")
                    or data.get("consumer")
                ),
                promotion_audit_policy_gate_result=(
                    data.get("promotion_audit_policy_gate_result")
                    or data.get("audit_policy_gate_result")
                    or data.get("audit_policy_gate_payload")
                    or data.get("policy_gate")
                    or data.get("gate")
                    or data.get("result")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_audit_policy_gate_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_draft_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_draft_result_consumer(
                promotion_audit_policy_gate_result_consumer=(
                    data.get("promotion_audit_policy_gate_result_consumer")
                    or data.get("queue_import_promotion_audit_policy_gate_result_consumer")
                    or data.get("consumer")
                ),
                promotion_memory_draft_result=(
                    data.get("promotion_memory_draft_result")
                    or data.get("memory_draft_result")
                    or data.get("memory_draft_payload")
                    or data.get("draft")
                    or data.get("result")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_draft_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_writer_dry_run_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer(
                promotion_memory_draft_result_consumer=(
                    data.get("promotion_memory_draft_result_consumer")
                    or data.get("queue_import_promotion_memory_draft_result_consumer")
                    or data.get("consumer")
                ),
                promotion_memory_writer_dry_run_result=(
                    data.get("promotion_memory_writer_dry_run_result")
                    or data.get("memory_writer_dry_run_result")
                    or data.get("memory_writer_result")
                    or data.get("writer")
                    or data.get("result")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_writer_dry_run_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_writer_apply_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer(
                promotion_memory_writer_dry_run_result_consumer=(
                    data.get("promotion_memory_writer_dry_run_result_consumer")
                    or data.get("queue_import_promotion_memory_writer_dry_run_result_consumer")
                    or data.get("consumer")
                ),
                promotion_memory_writer_apply_result=(
                    data.get("promotion_memory_writer_apply_result")
                    or data.get("memory_writer_apply_result")
                    or data.get("memory_writer_result")
                    or data.get("writer")
                    or data.get("result")
                ),
                root_path=data.get("root_path") or data.get("root"),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_writer_apply_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_index_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer(
                promotion_memory_writer_apply_result_consumer=(
                    data.get("promotion_memory_writer_apply_result_consumer")
                    or data.get("queue_import_promotion_memory_writer_apply_result_consumer")
                    or data.get("consumer")
                ),
                promotion_memory_index_result=(
                    data.get("promotion_memory_index_result")
                    or data.get("memory_index_result")
                    or data.get("memory_index")
                    or data.get("index")
                    or data.get("result")
                ),
                current_sequence_review_freshness=(
                    data.get("current_sequence_review_freshness")
                    or data.get("sequence_review_freshness")
                    or data.get("freshness")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_index_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_trust_review_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer(
                promotion_memory_index_result_consumer=(
                    data.get("promotion_memory_index_result_consumer")
                    or data.get("queue_import_promotion_memory_index_result_consumer")
                    or data.get("consumer")
                ),
                promotion_memory_trust_review_result=(
                    data.get("promotion_memory_trust_review_result")
                    or data.get("memory_trust_review_result")
                    or data.get("memory_trust_review")
                    or data.get("review")
                    or data.get("result")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_trust_review_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_trust_acceptance_ledger_result(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer(
                promotion_memory_trust_review_result_consumer=(
                    data.get("promotion_memory_trust_review_result_consumer")
                    or data.get("queue_import_promotion_memory_trust_review_result_consumer")
                    or data.get("consumer")
                ),
                promotion_memory_trust_acceptance_ledger_result=(
                    data.get("promotion_memory_trust_acceptance_ledger_result")
                    or data.get("memory_trust_acceptance_ledger_result")
                    or data.get("acceptance_ledger_result")
                    or data.get("acceptance_ledger")
                    or data.get("ledger")
                    or data.get("result")
                ),
            )
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_trust_acceptance_ledger_result"
        )
        await server.send_to_client(websocket, result)

    async def handle_create_operator_proposal(websocket, data):
        payload = dict(data.get("proposal") or data)
        payload.pop("type", None)
        explicit_proposal_id = str(
            payload.get("id") or payload.get("proposal_id") or ""
        ).strip()
        durable_ids = getattr(server, "_durable_operator_proposal_ids", set())
        if explicit_proposal_id and explicit_proposal_id in durable_ids:
            result = {
                "type": "operator_proposal_created",
                "status": "blocked",
                "success": False,
                "reason": "proposal_id_already_consumed",
                "proposal_id": explicit_proposal_id,
                "proposal": None,
                "operator_mode": server.get_operator_mode_status(),
            }
        else:
            result = server.operator_proposal_queue.create(
                payload,
                server.get_operator_mode_status(),
            )
        proposal = result.get("proposal") if isinstance(result, dict) else None
        if isinstance(proposal, dict):
            requested_change = proposal.get("requested_change") or {}
            recorder = getattr(server, "record_automation_decision_trace", None)
            if callable(recorder):
                recorder(
                    proposal_id=proposal.get("id"),
                    source=proposal.get("source") or "operator_proposal_queue",
                    mode=proposal.get("mode_at_creation"),
                    target=proposal.get("target"),
                    action_type=proposal.get("kind") or requested_change.get("value_type") or "proposal",
                    status="proposal_created" if result.get("status") in {"created", "exists"} else str(result.get("status")),
                    proposed_value=requested_change.get("value"),
                    delta=requested_change.get("delta_db"),
                    safety_limits={
                        "allowed_live_write_kinds": allowed_live_write_kinds_for_safety_surface(),
                        "proposal_only_if_unsupported": True,
                    },
                    proof_status="pending",
                    blocked_reason=result.get("reason") if not result.get("success") else None,
                    metadata={"result_status": result.get("status")},
                )
        await server.send_to_client(websocket, result)

    async def handle_accept_operator_proposal(websocket, data):
        proposal_id, index = proposal_id_or_index(data)
        result = server.operator_proposal_queue.accept(proposal_id=proposal_id, index=index)
        completed = result.get("proposal") if isinstance(result, dict) else None
        revoke = getattr(server, "revoke_m110_capability_for_proposal", None)
        if (
            result.get("success") is True
            and isinstance(completed, dict)
            and callable(revoke)
        ):
            revoke(
                completed.get("id"),
                reason="proposal_accepted_terminal",
            )
        await server.send_to_client(websocket, {"type": "operator_proposal_accepted", **result})

    async def handle_dismiss_operator_proposal(websocket, data):
        proposal_id, index = proposal_id_or_index(data)
        result = server.operator_proposal_queue.dismiss(
            proposal_id=proposal_id,
            index=index,
            reason=data.get("reason", "dismissed_by_operator"),
        )
        completed = result.get("proposal") if isinstance(result, dict) else None
        revoke = getattr(server, "revoke_m110_capability_for_proposal", None)
        if (
            result.get("success") is True
            and isinstance(completed, dict)
            and callable(revoke)
        ):
            revoke(
                completed.get("id"),
                reason="proposal_dismissed_terminal",
            )
        await server.send_to_client(websocket, {"type": "operator_proposal_dismissed", **result})

    class ApplyEvidenceLifecycle:
        def __init__(self, reservation_binding):
            self._reservation_binding = reservation_binding
            self._trusted_identity = None
            self._state = "active"

        @property
        def trusted_identity(self):
            return self._trusted_identity

        def bind_verified_execution(self, execution):
            if self._state != "active" or self._trusted_identity is not None:
                raise RuntimeError(
                    "operator_proposal_verified_identity_bind_state_invalid"
                )
            self._trusted_identity = _freeze_verified_apply_identity(execution)
            return self._trusted_identity

        @property
        def reservation_binding(self):
            return (
                self._reservation_binding
                if self._state == "active"
                else None
            )

        @property
        def state(self):
            return self._state

        def begin_finalize(self):
            if (
                self._state != "active"
                or self._reservation_binding is None
                or self._trusted_identity is None
            ):
                raise RuntimeError(
                    "operator_proposal_result_evidence_reservation_already_consumed"
                )
            self._state = "finalizing"
            return self._reservation_binding

        def complete_finalize(self):
            if self._state != "finalizing":
                raise RuntimeError(
                    "operator_proposal_result_evidence_finalize_state_invalid"
                )
            self._state = "consumed"

        def abandon_finalize(self):
            self._state = "consumed"

        def release(self):
            if self._state != "active" or self._reservation_binding is None:
                return False
            server.operator_proposal_queue.release_result_evidence(
                self._reservation_binding
            )
            self._state = "consumed"
            return True

    def release_apply_evidence_safely(evidence_lifecycle):
        try:
            return evidence_lifecycle.release()
        except Exception:
            server.operator_proposal_queue.latch_result_persistence_failure()
            return False

    async def activate_apply_emergency_stop(reason):
        emergency_stop = getattr(server, "activate_system_emergency_stop", None)
        if not callable(emergency_stop):
            return False, None
        try:
            outcome = await emergency_stop(reason)
        except asyncio.CancelledError:
            raise
        except Exception:
            return False, None
        teardown_complete = (
            outcome.get("teardown_complete")
            if type(outcome) is dict
            and isinstance(outcome.get("teardown_complete"), bool)
            else None
        )
        activated = bool(
            getattr(server, "_manual_write_emergency_stopped", False) is True
            or (
                type(outcome) is dict
                and outcome.get("status")
                in {"stopping", "stopped", "stop_incomplete"}
            )
        )
        return activated, teardown_complete

    def finalized_failure_receipt(
        identity,
        terminal_result,
        *,
        proposal_status,
        idempotency_key,
        result_evidence_pinned,
    ):
        reason_code = classify_trusted_apply_reason_code(
            terminal_result.get("reason"),
            source="finalizer",
            final_status="apply_error",
        )
        return _freeze_trusted_apply_receipt(
            identity,
            terminal_result=terminal_result,
            proposal_status=proposal_status,
            final_status="apply_error",
            final_success=False,
            result_digest=None,
            result_detail_available=False,
            result_evidence_pinned=result_evidence_pinned,
            reason_code=reason_code,
            idempotency_key=idempotency_key,
            operator_idempotency_released=None,
        )

    async def finalize_apply_result(
        proposal,
        apply_result,
        *,
        evidence_lifecycle,
        idempotency_key,
        reason_source,
        expected_operation_digest=None,
    ):
        identity = evidence_lifecycle.trusted_identity
        reservation_binding = evidence_lifecycle.begin_finalize()
        terminal_candidate = None
        effective_reason_source = reason_source
        if type(apply_result) is dict:
            try:
                frozen_input_bytes = canonical_json_bytes(apply_result)
                terminal_candidate = json.loads(
                    frozen_input_bytes.decode("utf-8")
                )
            except Exception:
                effective_reason_source = "finalizer"
                terminal_candidate = {
                    "status": "error",
                    "reason": "operator_apply_result_shape_invalid",
                    "no_console_send_proven": False,
                    "write_state_unknown": True,
                    "rollback_state_unknown": True,
                    "manual_review_required": True,
                }
        if type(terminal_candidate) is not dict:
            effective_reason_source = "finalizer"
            terminal_candidate = {
                "status": "error",
                "reason": "operator_apply_result_shape_invalid",
                "no_console_send_proven": False,
            }
        identity_matches = bool(
            getattr(proposal, "id", None) == identity.proposal_id
            and getattr(proposal, "review_contract_version", None)
            == identity.review_contract_version
            and getattr(proposal, "review_content_digest", None)
            == identity.review_content_digest
        )
        if not identity_matches:
            effective_reason_source = "finalizer"
            terminal_candidate.update(
                {
                    "status": "error",
                    "success": False,
                    "reason": "proposal_identity_changed_after_claim",
                }
            )
        no_console_send_proven = (
            terminal_candidate.get("no_console_send_proven") is True
        )
        possible_console_send = not no_console_send_proven
        physical_verified = terminal_candidate.get("physical_verified") is True
        physical_restored = (
            terminal_candidate.get("physical_state_restored") is True
        )
        write_state_unknown = bool(
            terminal_candidate.get("write_state_unknown") is True
            or (
                possible_console_send
                and not physical_verified
                and not physical_restored
            )
        )
        rollback_state_unknown = bool(
            terminal_candidate.get("rollback_state_unknown") is True
            or terminal_candidate.get("physical_state_restored") is False
        )
        emergency_required = bool(
            possible_console_send
            and (write_state_unknown or rollback_state_unknown)
        )
        if possible_console_send:
            terminal_candidate["no_console_send_proven"] = False
        if write_state_unknown:
            terminal_candidate.update(
                {
                    "write_state_unknown": True,
                    "manual_review_required": True,
                }
            )
        if rollback_state_unknown:
            terminal_candidate.update(
                {
                    "rollback_state_unknown": True,
                    "manual_review_required": True,
                }
            )
        if (
            possible_console_send
            and terminal_candidate.get("status") == "applied"
            and not physical_verified
        ):
            effective_reason_source = "finalizer"
            terminal_candidate.update(
                {
                    "status": "error",
                    "success": False,
                    "reason": "apply_terminal_physical_proof_incomplete",
                }
            )
        elif (
            possible_console_send
            and terminal_candidate.get("status") == "applied"
            and not terminal_result_proves_verified_apply(terminal_candidate)
        ):
            effective_reason_source = "finalizer"
            terminal_candidate.update(
                {
                    "status": "error",
                    "success": False,
                    "reason": "apply_terminal_evidence_incomplete",
                    "manual_review_required": True,
                    "fresh_analysis_required": True,
                }
            )
        elif terminal_candidate.get("status") == "proof_passed" and not (
            terminal_result_proves_verified_restored_proof(
                terminal_candidate
            )
            and terminal_candidate.get("proposal_id") == identity.proposal_id
            and terminal_candidate.get("review_content_digest")
            == identity.review_content_digest
            and isinstance(expected_operation_digest, str)
            and terminal_candidate.get("operation_digest")
            == expected_operation_digest
        ):
            effective_reason_source = "finalizer"
            terminal_candidate.update(
                {
                    "status": "error",
                    "success": False,
                    "reason": "proof_terminal_identity_incomplete",
                    "manual_review_required": True,
                    "fresh_analysis_required": True,
                }
            )
        if emergency_required:
            activated, teardown_complete = await activate_apply_emergency_stop(
                "operator_proposal_terminal_state_unknown:"
                f"{identity.proposal_id}"
            )
            terminal_candidate["emergency_stop_activated"] = activated
            terminal_candidate["emergency_stop_required"] = bool(
                not activated or teardown_complete is not True
            )
            terminal_candidate["emergency_stop_teardown_complete"] = (
                teardown_complete
            )
        elif terminal_candidate.get("emergency_stop_activated") is True:
            terminal_candidate["emergency_stop_activated"] = bool(
                getattr(server, "_manual_write_emergency_stopped", False) is True
            )

        identity_matches_after_await = bool(
            getattr(proposal, "id", None) == identity.proposal_id
            and getattr(proposal, "review_contract_version", None)
            == identity.review_contract_version
            and getattr(proposal, "review_content_digest", None)
            == identity.review_content_digest
        )
        if not identity_matches_after_await:
            effective_reason_source = "finalizer"
            terminal_candidate.update(
                {
                    "status": "error",
                    "success": False,
                    "reason": "proposal_identity_changed_after_claim",
                    "manual_review_required": possible_console_send,
                }
            )

        try:
            result = server.operator_proposal_queue.mark_apply_result(
                proposal,
                terminal_candidate,
                evidence_reservation_binding=reservation_binding,
                expected_proposal_id=identity.proposal_id,
                expected_review_contract_version=(
                    identity.review_contract_version
                ),
                expected_review_content_digest=(
                    identity.review_content_digest
                ),
            )
        except Exception as exc:
            proposal_status_after_exception = getattr(
                proposal,
                "status",
                "unknown",
            )
            if not isinstance(proposal_status_after_exception, str):
                proposal_status_after_exception = "unknown"
            server.operator_proposal_queue.latch_result_persistence_failure()
            if no_console_send_proven:
                try:
                    server.operator_proposal_queue.release_result_evidence(
                        reservation_binding
                    )
                except Exception:
                    pass
            evidence_lifecycle.abandon_finalize()
            activated = terminal_candidate.get("emergency_stop_activated") is True
            teardown_complete = terminal_candidate.get(
                "emergency_stop_teardown_complete"
            )
            if possible_console_send and not activated:
                activated, teardown_complete = await activate_apply_emergency_stop(
                    "operator_proposal_result_finalize_exception_after_send:"
                    f"{identity.proposal_id}"
                )
            failure_result = {
                **terminal_candidate,
                "status": "error",
                "reason": "operator_proposal_result_persistence_failed",
                "error_type": type(exc).__name__,
                "no_console_send_proven": (
                    True if no_console_send_proven else False
                ),
                "write_state_unknown": possible_console_send,
                "physical_state_restored": (
                    None
                ),
                "manual_review_required": possible_console_send,
                "emergency_stop_activated": (
                    activated if possible_console_send else False
                ),
                "emergency_stop_required": bool(
                    possible_console_send
                    and (not activated or teardown_complete is not True)
                ),
                "emergency_stop_teardown_complete": (
                    teardown_complete if possible_console_send else None
                ),
                "result_persistence_failed": True,
            }
            receipt = finalized_failure_receipt(
                identity,
                failure_result,
                proposal_status=proposal_status_after_exception,
                idempotency_key=idempotency_key,
                result_evidence_pinned=possible_console_send,
            )
            return FinalizedApplyResult(
                {
                    "status": "apply_error",
                    "success": False,
                    "reason": "operator_proposal_result_persistence_failed",
                    "proposal": {
                        "id": identity.proposal_id,
                        "review_contract_version": (
                            identity.review_contract_version
                        ),
                        "review_content_digest": identity.review_content_digest,
                        "status": "apply_error",
                    },
                    "apply_result": failure_result,
                    "result_digest": None,
                    "result_detail_available": False,
                },
                trusted_receipt=receipt,
            )

        evidence_lifecycle.complete_finalize()
        persisted_result = result.get("apply_result")
        result_digest = result.get("result_digest")
        frozen_result = None
        persistence_verified = False
        if type(persisted_result) is dict and result.get(
            "result_detail_available"
        ) is True:
            try:
                result_bytes = canonical_json_bytes(persisted_result)
                persistence_verified = bool(
                    sha256_digest(result_bytes) == result_digest
                )
                if persistence_verified:
                    frozen_result = json.loads(result_bytes.decode("utf-8"))
            except Exception:
                persistence_verified = False
        if not persistence_verified or type(frozen_result) is not dict:
            server.operator_proposal_queue.latch_result_persistence_failure()
            possible_after_result = bool(
                type(persisted_result) is not dict
                or persisted_result.get("no_console_send_proven") is not True
            )
            activated = bool(
                type(persisted_result) is dict
                and persisted_result.get("emergency_stop_activated") is True
                and getattr(
                    server,
                    "_manual_write_emergency_stopped",
                    False,
                )
                is True
            )
            teardown_complete = (
                persisted_result.get("emergency_stop_teardown_complete")
                if type(persisted_result) is dict
                and isinstance(
                    persisted_result.get("emergency_stop_teardown_complete"),
                    bool,
                )
                else None
            )
            if possible_after_result and not activated:
                activated, teardown_complete = await activate_apply_emergency_stop(
                    "operator_proposal_result_persistence_failed_after_send:"
                    f"{identity.proposal_id}"
                )
            failure_result = {
                **(
                    persisted_result
                    if type(persisted_result) is dict
                    else terminal_candidate
                ),
                "status": "error",
                "reason": "operator_proposal_result_persistence_failed",
                "no_console_send_proven": (
                    False if possible_after_result else True
                ),
                "write_state_unknown": possible_after_result,
                "physical_state_restored": None,
                "manual_review_required": possible_after_result,
                "emergency_stop_activated": (
                    activated if possible_after_result else False
                ),
                "emergency_stop_required": bool(
                    possible_after_result
                    and (not activated or teardown_complete is not True)
                ),
                "emergency_stop_teardown_complete": (
                    teardown_complete if possible_after_result else None
                ),
                "result_persistence_failed": True,
            }
            proposal_status_after_result_failure = getattr(
                proposal,
                "status",
                "unknown",
            )
            if not isinstance(proposal_status_after_result_failure, str):
                proposal_status_after_result_failure = "unknown"
            receipt = finalized_failure_receipt(
                identity,
                failure_result,
                proposal_status=proposal_status_after_result_failure,
                idempotency_key=idempotency_key,
                result_evidence_pinned=possible_after_result,
            )
            failed_outer = {
                **result,
                "status": "apply_error",
                "success": False,
                "reason": "operator_proposal_result_persistence_failed",
                "apply_result": failure_result,
                "result_digest": None,
                "result_detail_available": False,
            }
            return FinalizedApplyResult(
                failed_outer,
                trusted_receipt=receipt,
            )

        final_status = result.get("status")
        if final_status not in {
            "applied",
            "proof_passed",
            "apply_blocked",
            "apply_error",
        }:
            final_status = "apply_error"
        terminal_proposal_status = getattr(proposal, "status", "unknown")
        if not isinstance(terminal_proposal_status, str):
            terminal_proposal_status = "unknown"
        terminal_result_evidence_pinned = bool(
            getattr(proposal, "result_evidence_pinned", False) is True
        )
        operator_idempotency_released = None
        if frozen_result.get("no_console_send_proven") is True:
            release_idempotency = getattr(
                server,
                "release_operator_apply_idempotency_key_if_no_send",
                None,
            )
            if callable(release_idempotency):
                operator_idempotency_released = bool(
                    release_idempotency(
                        idempotency_key,
                        proposal_id=identity.proposal_id,
                    )
                )
                result["operator_idempotency_released"] = (
                    operator_idempotency_released
                )
        receipt = _freeze_trusted_apply_receipt(
            identity,
            terminal_result=frozen_result,
            proposal_status=terminal_proposal_status,
            final_status=final_status,
            final_success=result.get("success") is True,
            result_digest=result_digest,
            result_detail_available=True,
            result_evidence_pinned=terminal_result_evidence_pinned,
            reason_code=classify_trusted_apply_reason_code(
                frozen_result.get("reason"),
                source=effective_reason_source,
                final_status=final_status,
            ),
            idempotency_key=idempotency_key,
            operator_idempotency_released=operator_idempotency_released,
        )
        return FinalizedApplyResult(result, trusted_receipt=receipt)

    async def linked_change_blocked_result(
        proposal,
        reason,
        *,
        evidence_lifecycle,
        idempotency_key,
        **extra,
    ):
        return await finalize_apply_result(
            proposal,
            {
                "status": "blocked",
                "reason": reason,
                "no_console_send_proven": True,
                **extra,
            },
            evidence_lifecycle=evidence_lifecycle,
            idempotency_key=idempotency_key,
            reason_source="linked_apply",
        )

    def preclaim_apply_blocked_result(proposal, reason, **extra):
        """Return a compact retry-safe refusal without mutating queue state."""

        apply_result = {
            "status": "blocked",
            "reason": reason,
            "no_console_send_proven": True,
        }
        apply_result.update(extra)
        receipt = _freeze_trusted_prewrite_no_send_receipt(
            proposal,
            reason_code=reason,
        )
        return FinalizedApplyResult(
            {
                "status": "blocked",
                "success": False,
                "reason": reason,
                "proposal": (
                    compact_proposal_reference(proposal)
                    if proposal is not None
                    else None
                ),
                "proposal_status": getattr(proposal, "status", None),
                "proposal_status_mutation": False,
                "result_evidence_reserved": False,
                "idempotency_reserved": False,
                "no_console_send_proven": True,
                "apply_result": apply_result,
            },
            trusted_receipt=receipt,
        )

    async def send_operator_apply_response(websocket, response_type, result):
        trusted_receipt = getattr(result, "trusted_receipt", None)
        await server.send_to_client(
            websocket,
            {"type": response_type, **result},
            trusted_receipt=trusted_receipt,
        )
        return result

    async def finalize_apply_result_cancellation_safe(*args, **kwargs):
        """Finish the post-claim evidence lifecycle despite outer cancellation."""

        cleanup = asyncio.create_task(finalize_apply_result(*args, **kwargs))
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                if cleanup.done():
                    break
                continue
        return cleanup.result()

    def claim_failure_apply_result(proposal, claim_result):
        finalized = claim_result.get("finalized_result")
        if isinstance(finalized, FinalizedApplyResult):
            return finalized
        return preclaim_apply_blocked_result(
            proposal,
            claim_result.get("reason") or "proposal_apply_claim_failed",
            result_evidence_reserved=False,
            idempotency_reserved=False,
        )

    async def claim_and_reserve_operator_apply(
        proposal,
        *,
        request_facts,
        evidence_lifecycle=None,
    ):
        if evidence_lifecycle is None:
            result_reservation = (
                server.operator_proposal_queue.reserve_result_evidence(proposal)
            )
            if not result_reservation.get("reserved"):
                return {
                    "ready": False,
                    "reason": result_reservation.get("reason"),
                    "result_evidence": result_reservation,
                    "no_console_send_proven": True,
                    "proposal_status_mutation": False,
                    "idempotency_reserved": False,
                }
            evidence_lifecycle = ApplyEvidenceLifecycle(
                result_reservation.get("reservation_binding")
            )
        elif not (
            isinstance(evidence_lifecycle, ApplyEvidenceLifecycle)
            and evidence_lifecycle.reservation_binding is not None
        ):
            return {
                "ready": False,
                "reason": "operator_proposal_result_evidence_reservation_invalid",
                "no_console_send_proven": True,
                "proposal_status_mutation": False,
                "idempotency_reserved": False,
            }
        try:
            claim = server.operator_proposal_queue.claim_verified_for_apply(
                proposal,
                expected_review_contract_version=(
                    request_facts.expected_review_contract_version
                ),
                expected_review_content_digest=(
                    request_facts.expected_review_content_digest
                ),
            )
        except Exception as exc:
            if getattr(proposal, "status", None) == "applying":
                if evidence_lifecycle.trusted_identity is None:
                    evidence_lifecycle.bind_verified_execution(
                        SimpleNamespace(
                            proposal_id=getattr(proposal, "id", None),
                            review_contract_version=getattr(
                                proposal,
                                "review_contract_version",
                                None,
                            ),
                            review_content_digest=getattr(
                                proposal,
                                "review_content_digest",
                                None,
                            ),
                        )
                    )
                finalized = await finalize_apply_result_cancellation_safe(
                    proposal,
                    {
                        "status": "blocked",
                        "reason": "operator_proposal_claim_exception",
                        "exception_type": type(exc).__name__,
                        "no_console_send_proven": True,
                    },
                    evidence_lifecycle=evidence_lifecycle,
                    idempotency_key=request_facts.idempotency_key,
                    reason_source="finalizer",
                )
                return {
                    "ready": False,
                    "reason": "operator_proposal_claim_exception",
                    "finalized_result": finalized,
                    "no_console_send_proven": True,
                }
            released = release_apply_evidence_safely(evidence_lifecycle)
            return {
                "ready": False,
                "reason": (
                    "operator_proposal_claim_exception"
                    if released
                    else "operator_proposal_result_persistence_failed"
                ),
                "result_evidence_released": released,
                "no_console_send_proven": True,
            }
        if not claim.get("claimed"):
            released = release_apply_evidence_safely(evidence_lifecycle)
            if not released:
                return {
                    "ready": False,
                    "reason": "operator_proposal_result_persistence_failed",
                    "claim": claim,
                    "result_evidence_released": False,
                    "no_console_send_proven": True,
                }
            return {
                "ready": False,
                "reason": claim.get("reason") or "proposal_not_pending",
                "claim": claim,
                "result_evidence_released": True,
            }
        try:
            evidence_lifecycle.bind_verified_execution(claim.get("execution"))
            reservation = server.reserve_operator_apply_idempotency_key(
                request_facts.idempotency_key,
                proposal_id=getattr(proposal, "id", None),
                approval_id=request_facts.approval_id,
            )
        except Exception as exc:
            finalized = await finalize_apply_result_cancellation_safe(
                proposal,
                {
                    "status": "blocked",
                    "reason": "operator_proposal_claim_finalize_exception",
                    "exception_type": type(exc).__name__,
                    "no_console_send_proven": True,
                },
                evidence_lifecycle=evidence_lifecycle,
                idempotency_key=request_facts.idempotency_key,
                reason_source="finalizer",
            )
            return {
                "ready": False,
                "reason": "operator_proposal_claim_finalize_exception",
                "finalized_result": finalized,
                "no_console_send_proven": True,
            }
        if not reservation.get("reserved"):
            finalized = await finalize_apply_result_cancellation_safe(
                proposal,
                {
                    "status": "blocked",
                    "reason": reservation.get("reason"),
                    "idempotency": reservation,
                    "no_console_send_proven": True,
                },
                evidence_lifecycle=evidence_lifecycle,
                idempotency_key=request_facts.idempotency_key,
                reason_source="claim",
            )
            return {
                "ready": False,
                "reason": reservation.get("reason"),
                "claim": {
                    "claimed": True,
                    "proposal_status": claim.get("proposal_status"),
                },
                "idempotency": reservation,
                "finalized_result": finalized,
            }
        return {
            "ready": True,
            "claim": claim,
            "execution": claim.get("execution"),
            "idempotency": reservation,
            "evidence_lifecycle": evidence_lifecycle,
        }

    def verified_execution_view(execution):
        content = execution.content_copy()
        view = SimpleNamespace(
            id=content.get("proposal_id"),
            source=content.get("source"),
            reason=content.get("reason"),
            title=content.get("title"),
            raw=content.get("raw") if isinstance(content.get("raw"), dict) else {},
            requested_change=(
                content.get("requested_change")
                if isinstance(content.get("requested_change"), dict)
                else None
            ),
            requested_changes=(
                content.get("requested_changes")
                if isinstance(content.get("requested_changes"), list)
                else []
            ),
        )
        return content, view

    async def finalize_unknown_apply_failure(
        proposal,
        *,
        proposal_id,
        reason,
        evidence_lifecycle,
        idempotency_key,
        reason_source,
        error=None,
    ):
        return await finalize_apply_result_cancellation_safe(
            proposal,
            {
                "status": "error",
                "success": False,
                "reason": reason,
                "error": str(error) if error is not None else None,
                "manual_review_required": True,
                "no_console_send_proven": False,
            },
            evidence_lifecycle=evidence_lifecycle,
            idempotency_key=idempotency_key,
            reason_source=reason_source,
        )

    def normalize_linked_change(change):
        value_type = server._normalize_manual_write_kind(change.get("value_type"))
        channel = coerce_channel(change.get("channel"))
        if channel is None:
            return None, "invalid_channel"
        value = coerce_float(change.get("value"))
        if value is None:
            return None, "invalid_value"
        address = server._manual_write_address(channel, value_type)
        if not address:
            return None, "unsupported_value_type"
        return {
            "value_type": value_type,
            "channel": channel,
            "value": value,
            "effective_value": server._clamp_manual_write_value(value, value_type),
            "address": address,
        }, None

    async def rollback_linked_results(
        websocket,
        proposal_id,
        applied_results,
        approval_id,
        reason,
        evidence_lifecycle,
    ):
        rollback_results = []
        for index, applied in reversed(list(enumerate(applied_results))):
            value_type = str(applied.get("value_type") or "")
            channel = coerce_channel(applied.get("channel"))
            expected_current_value = applied.get("effective_value")
            rollback_value = applied.get("rollback_value")
            address = str(
                applied.get("rollback_address")
                or (
                    server._manual_write_address(channel, value_type)
                    if channel is not None
                    else ""
                )
            )
            if (
                channel is None
                or rollback_value is None
                or expected_current_value is None
                or not address
            ):
                rollback_results.append(
                    {
                        "status": "blocked",
                        "success": False,
                        "reason": "rollback_preimage_unavailable",
                        "rollback": True,
                        "rollback_verified": False,
                        "channel": applied.get("channel"),
                        "value_type": value_type,
                    }
                )
                continue
            current_read = await server._manual_write_read_value(
                channel=channel,
                value_type=value_type,
                address=address,
                prefer_physical=True,
            )
            current_status = server._manual_write_readback_status(
                old_value=expected_current_value,
                readback_value=current_read.get("value"),
                effective_value=expected_current_value,
                value_type=value_type,
                readback_source=str(current_read.get("source") or "unavailable"),
                readback_cache_freshness=current_read.get("cache_freshness"),
            )
            if current_status != "verified_from_physical_readback":
                rollback_results.append(
                    {
                        "status": "blocked",
                        "success": False,
                        "reason": "rollback_cas_target_changed_or_unverified",
                        "rollback": True,
                        "rollback_verified": False,
                        "channel": channel,
                        "value_type": value_type,
                        "expected_current_value": expected_current_value,
                        "current_readback_value": current_read.get("value"),
                        "current_readback_status": current_status,
                    }
                )
                continue
            rollback_result = await server._apply_manual_console_write(
                websocket,
                channel=channel,
                value=rollback_value,
                value_type=value_type,
                source=f"operator_proposal_linked_rollback:{proposal_id}",
                reason=reason,
                approved=True,
                approval_id=f"rollback:{approval_id}:{index + 1}",
                expected_current_value=expected_current_value,
                bypass_cooldown=True,
                emergency_recovery=True,
                result_evidence_reservation_binding=(
                    evidence_lifecycle.reservation_binding
                ),
            )
            rollback_result.update({
                "rollback": True,
                "rollback_verified": (
                    rollback_result.get("status") == "applied"
                    and manual_write_result_verified(rollback_result)
                ),
                "rollback_of": applied.get("approval_id"),
                "rollback_source": applied.get("source"),
            })
            rollback_results.append(rollback_result)
        return rollback_results

    async def apply_linked_operator_proposal(
        websocket,
        approval_facts,
        proposal,
        verified_execution,
        execution_proposal,
        operator_mode,
        requested_changes,
        idempotency_key,
        evidence_lifecycle,
    ):
        allowed_kinds = set(operator_mode["capabilities"].get("allowed_live_write_kinds") or [])
        operator_authored = is_operator_authored(
            execution_proposal,
            approval_facts.approval_token,
        )
        if not operator_mode["capabilities"]["can_apply_to_console"]:
            result = await linked_change_blocked_result(
                proposal,
                "operator_mode_blocks_live_write",
                evidence_lifecycle=evidence_lifecycle,
                idempotency_key=idempotency_key,
                operator_mode=operator_mode,
                requested_changes=requested_changes,
            )
            record_apply_trace(
                proposal=execution_proposal,
                operator_mode=operator_mode,
                requested_change=requested_changes[0] if requested_changes else {},
                idempotency_key=idempotency_key,
                status="blocked",
                blocked_reason="operator_mode_blocks_live_write",
                metadata={"linked": True, "requested_changes": requested_changes},
            )
            return await send_operator_apply_response(
                websocket, "operator_proposal_apply_blocked", result
            )
            return

        if not approval_facts.approved:
            record_apply_trace(
                proposal=execution_proposal,
                operator_mode=operator_mode,
                requested_change=requested_changes[0] if requested_changes else {},
                idempotency_key=idempotency_key,
                status="blocked",
                blocked_reason="approval_required",
                metadata={"linked": True},
            )
            result = await linked_change_blocked_result(
                proposal,
                "approval_required",
                evidence_lifecycle=evidence_lifecycle,
                idempotency_key=idempotency_key,
            )
            return await send_operator_apply_response(
                websocket, "operator_proposal_apply_blocked", result
            )
            return

        if not server.mixer_client or not getattr(server.mixer_client, "is_connected", True):
            result = await linked_change_blocked_result(
                proposal,
                "mixer_not_connected",
                evidence_lifecycle=evidence_lifecycle,
                idempotency_key=idempotency_key,
            )
            record_apply_trace(
                proposal=execution_proposal,
                operator_mode=operator_mode,
                requested_change=requested_changes[0] if requested_changes else {},
                idempotency_key=idempotency_key,
                status="blocked",
                blocked_reason="mixer_not_connected",
                metadata={"linked": True},
            )
            return await send_operator_apply_response(
                websocket, "operator_proposal_apply_blocked", result
            )
            return

        apply_manual_console_write = getattr(server, "_apply_manual_console_write", None)
        if apply_manual_console_write is None:
            result = await linked_change_blocked_result(
                proposal,
                "supervised_apply_path_unavailable",
                evidence_lifecycle=evidence_lifecycle,
                idempotency_key=idempotency_key,
            )
            record_apply_trace(
                proposal=execution_proposal,
                operator_mode=operator_mode,
                requested_change=requested_changes[0] if requested_changes else {},
                idempotency_key=idempotency_key,
                status="blocked",
                blocked_reason="supervised_apply_path_unavailable",
                metadata={"linked": True},
            )
            return await send_operator_apply_response(
                websocket, "operator_proposal_apply_blocked", result
            )
            return

        approval_id = approval_facts.approval_id
        if not approval_id:
            result = await linked_change_blocked_result(
                proposal,
                "approval_id_required",
                evidence_lifecycle=evidence_lifecycle,
                idempotency_key=idempotency_key,
            )
            return await send_operator_apply_response(
                websocket, "operator_proposal_apply_blocked", result
            )
            return
        normalized_changes = []
        for raw_change in requested_changes:
            normalized, error = await validate_change_before_apply(
                raw_change,
                proposal=execution_proposal,
                operator_authored=operator_authored,
                operator_mode=operator_mode,
                linked=True,
            )
            if error:
                result = await linked_change_blocked_result(
                    proposal,
                    error["reason"],
                    evidence_lifecycle=evidence_lifecycle,
                    idempotency_key=idempotency_key,
                    requested_change=raw_change,
                    requested_changes=requested_changes,
                    **{key: value for key, value in error.items() if key not in {"reason", "requested_change"}},
                )
                record_apply_trace(
                    proposal=execution_proposal,
                    operator_mode=operator_mode,
                    requested_change=raw_change,
                    normalized_change=normalized,
                    apply_result={"status": "blocked", **error},
                    idempotency_key=idempotency_key,
                    status="blocked",
                    blocked_reason=error["reason"],
                    metadata={"linked": True, "requested_changes": requested_changes},
                )
                return await send_operator_apply_response(
                    websocket, "operator_proposal_apply_blocked", result
                )
                return
            if normalized["value_type"] not in allowed_kinds:
                result = await linked_change_blocked_result(
                    proposal,
                    "unsupported_value_type",
                    evidence_lifecycle=evidence_lifecycle,
                    idempotency_key=idempotency_key,
                    value_type=normalized["value_type"],
                    allowed_live_write_kinds=sorted(allowed_kinds),
                )
                record_apply_trace(
                    proposal=execution_proposal,
                    operator_mode=operator_mode,
                    normalized_change=normalized,
                    idempotency_key=idempotency_key,
                    status="blocked",
                    blocked_reason="unsupported_value_type",
                    metadata={"linked": True},
                )
                return await send_operator_apply_response(
                    websocket, "operator_proposal_apply_blocked", result
                )
                return
            normalized_changes.append(normalized)

        applied_results = []
        for index, change in enumerate(normalized_changes):
            apply_result = await apply_manual_console_write(
                websocket,
                channel=change["channel"],
                value=change["effective_value"],
                value_type=change["value_type"],
                source=(
                    "operator_proposal_linked:"
                    f"{verified_execution.proposal_id}:{index + 1}"
                ),
                reason=execution_proposal.reason or execution_proposal.title,
                approved=True,
                approval_id=f"{approval_id}:{index + 1}",
                idempotency_key=idempotency_key,
                proposal_id=verified_execution.proposal_id,
                expected_current_value=change["old_value"],
                approval_expires_at_epoch=verified_execution.expires_at_epoch,
                result_evidence_reservation_binding=(
                    evidence_lifecycle.reservation_binding
                ),
            )
            applied_results.append(apply_result)
            record_apply_trace(
                proposal=execution_proposal,
                operator_mode=operator_mode,
                normalized_change=change,
                apply_result=apply_result,
                idempotency_key=idempotency_key,
                status=apply_result.get("status", "unknown"),
                blocked_reason=apply_result.get("reason") if apply_result.get("status") != "applied" else None,
                metadata={"linked": True, "linked_index": index + 1},
            )
            verified = manual_write_result_verified(apply_result)
            rollback_ready = bool(apply_result.get("rollback_available"))
            if apply_result.get("status") != "applied" or not verified or not rollback_ready:
                possible_mutations = [
                    result
                    for result in applied_results
                    if result.get("status") in {"applied", "sent_unverified"}
                    or result.get("manual_review_required")
                    or result.get("emergency_stop_activated")
                ]
                rollback_candidates = [
                    result
                    for result in possible_mutations
                    if result.get("rollback_available")
                ]
                rollback_results = await rollback_linked_results(
                    websocket,
                    verified_execution.proposal_id,
                    rollback_candidates,
                    approval_id,
                    "rollback:linked_operator_proposal_failed",
                    evidence_lifecycle,
                )
                recovered_count = sum(
                    1 for item in rollback_results if item.get("rollback_verified")
                )
                forward_wal_committed = bool(
                    possible_mutations
                    and all(
                        item.get("wal_terminal") == "committed"
                        for item in possible_mutations
                    )
                )
                rollback_wal_committed = bool(
                    rollback_results
                    and all(
                        item.get("wal_terminal") == "committed"
                        for item in rollback_results
                    )
                )
                recovery_complete = (
                    bool(possible_mutations)
                    and len(possible_mutations) == len(rollback_candidates)
                    and len(rollback_candidates) == len(rollback_results)
                    and recovered_count == len(possible_mutations)
                    and forward_wal_committed
                    and rollback_wal_committed
                )
                first_step_no_send = bool(
                    not possible_mutations
                    and apply_result.get("no_console_send_proven") is True
                )
                if first_step_no_send:
                    result = await linked_change_blocked_result(
                        proposal,
                        apply_result.get("reason")
                        or "linked_apply_failed_before_send",
                        evidence_lifecycle=evidence_lifecycle,
                        idempotency_key=idempotency_key,
                        failed_result=apply_result,
                        apply_results=applied_results,
                        rollback_results=[],
                        requested_changes=requested_changes,
                    )
                elif recovery_complete:
                    result = await linked_change_blocked_result(
                        proposal,
                        "linked_apply_failed_rolled_back",
                        evidence_lifecycle=evidence_lifecycle,
                        idempotency_key=idempotency_key,
                        no_console_send_proven=False,
                        physical_state_restored=True,
                        terminal_after_console_send=True,
                        wal_terminal="rolled_back",
                        fresh_analysis_required=True,
                        failed_result=apply_result,
                        apply_results=applied_results,
                        rollback_results=rollback_results,
                        requested_changes=requested_changes,
                    )
                else:
                    emergency_stop = getattr(
                        server,
                        "activate_system_emergency_stop",
                        None,
                    )
                    if callable(emergency_stop):
                        await emergency_stop(
                            "linked_apply_recovery_incomplete:"
                            f"{verified_execution.proposal_id}"
                        )
                    result = await finalize_apply_result(
                        proposal,
                        {
                            "status": "error",
                            "success": False,
                            "reason": "linked_apply_failed_manual_review_required",
                            "no_console_send_proven": False,
                            "physical_state_restored": False,
                            "manual_review_required": True,
                            "emergency_stop_activated": callable(emergency_stop),
                            "failed_result": apply_result,
                            "apply_results": applied_results,
                            "rollback_results": rollback_results,
                            "requested_changes": requested_changes,
                        },
                        evidence_lifecycle=evidence_lifecycle,
                        idempotency_key=idempotency_key,
                        reason_source="linked_apply",
                    )
                for rollback_result in rollback_results:
                    record_apply_trace(
                        proposal=execution_proposal,
                        operator_mode=operator_mode,
                        apply_result=rollback_result,
                        idempotency_key=idempotency_key,
                        status=rollback_result.get("status", "unknown"),
                        blocked_reason=rollback_result.get("reason") if rollback_result.get("status") != "applied" else None,
                        metadata={"linked": True, "rollback": True},
                    )
                return await send_operator_apply_response(
                    websocket, "operator_proposal_apply_blocked", result
                )
                return

        for change, apply_result in zip(normalized_changes, applied_results):
            remember_applied_delta(change, apply_result)

        linked_physical_verified = bool(
            applied_results
            and all(
                item.get("physical_verified") is True
                for item in applied_results
            )
        )
        linked_wal_committed = bool(
            applied_results
            and all(
                item.get("wal_terminal") == "committed"
                for item in applied_results
            )
        )
        linked_audio_recheck_not_required = bool(
            applied_results
            and all(
                item.get("audio_recheck_required") is False
                and item.get("audio_recheck_disposition")
                == "not_required_by_manual_write_contract"
                and "audio_recheck_passed" not in item
                for item in applied_results
            )
        )
        linked_terminal_verified = bool(
            linked_physical_verified
            and linked_wal_committed
            and linked_audio_recheck_not_required
        )
        if not linked_terminal_verified:
            emergency_stop = getattr(
                server,
                "activate_system_emergency_stop",
                None,
            )
            if callable(emergency_stop):
                await emergency_stop(
                    "linked_apply_terminal_proof_incomplete:"
                    f"{verified_execution.proposal_id}"
                )
            result = await finalize_apply_result_cancellation_safe(
                proposal,
                {
                    "status": "error",
                    "success": False,
                    "linked": True,
                    "reason": "linked_apply_terminal_proof_incomplete",
                    "no_console_send_proven": False,
                    "physical_verified": linked_physical_verified,
                    "wal_terminal": (
                        "committed" if linked_wal_committed else "not_committed"
                    ),
                    "write_state_unknown": True,
                    "physical_state_restored": False,
                    "manual_review_required": True,
                    "fresh_analysis_required": True,
                    "emergency_stop_activated": callable(emergency_stop),
                    "apply_results": applied_results,
                    "requested_changes": requested_changes,
                },
                evidence_lifecycle=evidence_lifecycle,
                idempotency_key=idempotency_key,
                reason_source="linked_apply",
            )
            return await send_operator_apply_response(
                websocket, "operator_proposal_apply_blocked", result
            )
            return
        result = await finalize_apply_result(
            proposal,
            {
                "status": "applied",
                "success": True,
                "linked": True,
                "reason": execution_proposal.reason or execution_proposal.title,
                "approval_id": approval_id,
                "apply_results": applied_results,
                "readback_status": "verified_from_physical_readback"
                if any(result.get("readback_status") == "verified_from_physical_readback" for result in applied_results)
                else "verified_from_cache",
                "proof_status": "verified",
                "no_console_send_proven": False,
                "physical_verified": linked_physical_verified,
                "wal_terminal": (
                    "committed" if linked_wal_committed else "not_committed"
                ),
                **(
                    {
                        "audio_recheck_required": False,
                        "audio_recheck_disposition": (
                            "not_required_by_linked_manual_write_contract"
                        ),
                        "write_state_unknown": False,
                        "rollback_state_unknown": False,
                        "manual_review_required": False,
                    }
                    if linked_terminal_verified
                    else {}
                ),
                "rollback_available": all(result.get("rollback_available") for result in applied_results),
                "requested_changes": requested_changes,
            },
            evidence_lifecycle=evidence_lifecycle,
            idempotency_key=idempotency_key,
            reason_source="linked_apply",
        )
        return await send_operator_apply_response(
            websocket, "operator_proposal_apply_result", result
        )

    def idempotency_preflight(data):
        idempotency_key = str(data.get("idempotency_key") or "").strip()
        store = getattr(server, "_operator_apply_idempotency_keys", {})
        duplicate = bool(
            idempotency_key
            and isinstance(store, dict)
            and idempotency_key in store
        )
        return {
            "required": True,
            "provided": bool(idempotency_key),
            "idempotency_key": idempotency_key or None,
            "reserved": False,
            "would_reserve_on_apply": bool(idempotency_key and not duplicate),
            "duplicate": duplicate,
            "first_seen": store.get(idempotency_key)
            if duplicate and isinstance(store, dict)
            else None,
        }

    def operator_apply_preflight_payload(
        *,
        proposal=None,
        operator_mode=None,
        status="blocked",
        reason=None,
        idempotency=None,
        requested_change=None,
        requested_changes=None,
        normalized_change=None,
        normalized_changes=None,
        extra=None,
    ):
        payload = {
            "type": "operator_proposal_apply_preflight",
            "status": status,
            "success": status == "ready",
            "ready_to_apply": status == "ready",
            "reason": reason,
            "proposal": proposal.to_dict() if proposal else None,
            "operator_mode": operator_mode,
            "idempotency": idempotency or {},
            "requested_change": requested_change,
            "requested_changes": requested_changes,
            "normalized_change": normalized_change,
            "normalized_changes": normalized_changes or [],
            "safety": {
                "preflight_only": True,
                "live_write": False,
                "apply_attempted": False,
                "queue_mutation": False,
                "proposal_status_mutation": False,
                "idempotency_reserved": False,
                "manual_console_write_called": False,
                "allowed_live_write_kinds": (
                    sorted(
                        set(
                            (operator_mode or {})
                            .get("capabilities", {})
                            .get("allowed_live_write_kinds")
                            or []
                        )
                    )
                    if isinstance(operator_mode, dict)
                    else []
                ),
                "live_apply_entrypoint": "apply_operator_proposal",
            },
        }
        if extra:
            payload.update(extra)
        return payload

    async def handle_preflight_operator_proposal_apply(websocket, data):
        proposal_id, index = proposal_id_or_index(data)
        proposal = server.operator_proposal_queue.find(proposal_id=proposal_id, index=index)
        operator_mode = server.get_operator_mode_status()
        idempotency = idempotency_preflight(data)
        if proposal is None:
            await server.send_to_client(
                websocket,
                operator_apply_preflight_payload(
                    operator_mode=operator_mode,
                    reason="proposal_not_found",
                    idempotency=idempotency,
                    extra={"proposal_id": proposal_id, "index": index},
                ),
            )
            return

        if proposal.status != "pending":
            result = preclaim_apply_blocked_result(
                proposal,
                "proposal_not_pending",
            )
            return await send_operator_apply_response(
                websocket, "operator_proposal_apply_blocked", result
            )
            return

        await server.send_to_client(
            websocket,
            await build_operator_proposal_apply_preflight_result(proposal, data),
        )

    def batch_idempotency_key(data, proposal_id):
        key_map = data.get("idempotency_keys")
        if isinstance(key_map, dict) and proposal_id in key_map:
            return str(key_map.get(proposal_id) or "").strip()
        prefix = str(data.get("idempotency_key_prefix") or "").strip()
        if prefix and proposal_id:
            return f"{prefix}:{proposal_id}"
        return str(data.get("idempotency_key") or "").strip()

    async def build_operator_proposal_apply_preflight_result(
        proposal,
        data,
        *,
        batch=False,
    ):
        proposal_id = getattr(proposal, "id", None)
        preflight_data = dict(data)
        preflight_data["proposal_id"] = proposal_id
        preflight_data["idempotency_key"] = (
            batch_idempotency_key(data, proposal_id)
            if batch
            else str(data.get("idempotency_key") or "").strip()
        )
        operator_mode = server.get_operator_mode_status()
        idempotency = idempotency_preflight(preflight_data)
        operator_authored = is_operator_authored(
            proposal,
            str(preflight_data.get("approval_token") or ""),
        )
        preflight_extra = {"batch": True} if batch else {}

        if proposal.is_expired():
            return operator_apply_preflight_payload(
                proposal=proposal,
                operator_mode=operator_mode,
                reason="proposal_expired",
                idempotency=idempotency,
                extra={
                    **preflight_extra,
                    "expires_at": proposal.expires_at,
                    "ttl_seconds": proposal.ttl_seconds,
                },
            )

        source_blocker = soundcheck_proposal_apply_blocker(proposal)
        if source_blocker:
            return operator_apply_preflight_payload(
                proposal=proposal,
                operator_mode=operator_mode,
                reason=source_blocker,
                idempotency=idempotency,
                requested_change=proposal.requested_change,
                requested_changes=proposal.requested_changes or None,
                extra={
                    **preflight_extra,
                    "apply_blockers": [source_blocker],
                    "no_console_send_proven": True,
                },
            )

        if not idempotency["provided"]:
            return operator_apply_preflight_payload(
                proposal=proposal,
                operator_mode=operator_mode,
                reason="missing_idempotency_key",
                idempotency=idempotency,
                extra=preflight_extra,
            )
        if idempotency["duplicate"]:
            return operator_apply_preflight_payload(
                proposal=proposal,
                operator_mode=operator_mode,
                reason="duplicate_idempotency_key",
                idempotency=idempotency,
                extra=preflight_extra,
            )

        requested_changes = getattr(proposal, "requested_changes", []) or []
        if requested_changes:
            allowed_kinds = set(
                operator_mode["capabilities"].get("allowed_live_write_kinds") or []
            )
            if not operator_mode["capabilities"]["can_apply_to_console"]:
                return operator_apply_preflight_payload(
                    proposal=proposal,
                    operator_mode=operator_mode,
                    reason="operator_mode_blocks_live_write",
                    idempotency=idempotency,
                    requested_changes=requested_changes,
                    extra=preflight_extra,
                )
            if data.get("approved") is not True:
                return operator_apply_preflight_payload(
                    proposal=proposal,
                    operator_mode=operator_mode,
                    reason="approval_required",
                    idempotency=idempotency,
                    requested_changes=requested_changes,
                    extra=preflight_extra,
                )
            if not server.mixer_client or not getattr(server.mixer_client, "is_connected", True):
                return operator_apply_preflight_payload(
                    proposal=proposal,
                    operator_mode=operator_mode,
                    reason="mixer_not_connected",
                    idempotency=idempotency,
                    requested_changes=requested_changes,
                    extra=preflight_extra,
                )
            if getattr(server, "_apply_manual_console_write", None) is None:
                return operator_apply_preflight_payload(
                    proposal=proposal,
                    operator_mode=operator_mode,
                    reason="supervised_apply_path_unavailable",
                    idempotency=idempotency,
                    requested_changes=requested_changes,
                    extra=preflight_extra,
                )

            normalized_changes = []
            for change_index, raw_change in enumerate(requested_changes):
                normalized, error = await validate_change_before_apply(
                    raw_change,
                    proposal=proposal,
                    operator_authored=operator_authored,
                    operator_mode=operator_mode,
                    linked=True,
                    update_delta_history=False,
                )
                if error:
                    return operator_apply_preflight_payload(
                        proposal=proposal,
                        operator_mode=operator_mode,
                        reason=error["reason"],
                        idempotency=idempotency,
                        requested_change=raw_change,
                        requested_changes=requested_changes,
                        normalized_change=normalized,
                        normalized_changes=normalized_changes,
                        extra={
                            **preflight_extra,
                            "blocked_change_index": change_index,
                            **{
                                key: value
                                for key, value in error.items()
                                if key != "reason"
                            },
                        },
                    )
                if normalized["value_type"] not in allowed_kinds:
                    return operator_apply_preflight_payload(
                        proposal=proposal,
                        operator_mode=operator_mode,
                        reason="unsupported_value_type",
                        idempotency=idempotency,
                        requested_change=raw_change,
                        requested_changes=requested_changes,
                        normalized_change=normalized,
                        normalized_changes=normalized_changes,
                        extra={
                            **preflight_extra,
                            "blocked_change_index": change_index,
                            "value_type": normalized["value_type"],
                            "allowed_live_write_kinds": sorted(allowed_kinds),
                        },
                    )
                normalized_changes.append(normalized)

            return operator_apply_preflight_payload(
                proposal=proposal,
                operator_mode=operator_mode,
                status="ready",
                idempotency=idempotency,
                requested_changes=requested_changes,
                normalized_changes=normalized_changes,
                extra={**preflight_extra, "linked": True},
            )

        requested_change = proposal.requested_change or {}
        value_type = str(requested_change.get("value_type") or "").strip().lower()
        allowed_kinds = set(operator_mode["capabilities"].get("allowed_live_write_kinds") or [])
        if (
            not value_type
            or requested_change.get("channel") is None
            or requested_change.get("value") is None
        ):
            return operator_apply_preflight_payload(
                proposal=proposal,
                operator_mode=operator_mode,
                reason="proposal_has_no_applyable_requested_change",
                idempotency=idempotency,
                requested_change=requested_change or None,
                extra=preflight_extra,
            )

        if not operator_mode["capabilities"]["can_apply_to_console"]:
            return operator_apply_preflight_payload(
                proposal=proposal,
                operator_mode=operator_mode,
                reason="operator_mode_blocks_live_write",
                idempotency=idempotency,
                requested_change=requested_change,
                extra=preflight_extra,
            )

        if value_type not in allowed_kinds:
            return operator_apply_preflight_payload(
                proposal=proposal,
                operator_mode=operator_mode,
                reason="unsupported_value_type",
                idempotency=idempotency,
                requested_change=requested_change,
                extra={
                    **preflight_extra,
                    "value_type": value_type,
                    "allowed_live_write_kinds": sorted(allowed_kinds),
                },
            )

        if data.get("approved") is not True:
            return operator_apply_preflight_payload(
                proposal=proposal,
                operator_mode=operator_mode,
                reason="approval_required",
                idempotency=idempotency,
                requested_change=requested_change,
                extra=preflight_extra,
            )

        if not server.mixer_client or not getattr(server.mixer_client, "is_connected", True):
            return operator_apply_preflight_payload(
                proposal=proposal,
                operator_mode=operator_mode,
                reason="mixer_not_connected",
                idempotency=idempotency,
                requested_change=requested_change,
                extra=preflight_extra,
            )

        if getattr(server, "_apply_manual_console_write", None) is None:
            return operator_apply_preflight_payload(
                proposal=proposal,
                operator_mode=operator_mode,
                reason="supervised_apply_path_unavailable",
                idempotency=idempotency,
                requested_change=requested_change,
                extra=preflight_extra,
            )

        normalized_change, safety_error = await validate_change_before_apply(
            requested_change,
            proposal=proposal,
            operator_authored=operator_authored,
            operator_mode=operator_mode,
            linked=False,
            update_delta_history=False,
        )
        if safety_error:
            return operator_apply_preflight_payload(
                proposal=proposal,
                operator_mode=operator_mode,
                reason=safety_error["reason"],
                idempotency=idempotency,
                requested_change=requested_change,
                normalized_change=normalized_change,
                extra={
                    **preflight_extra,
                    **{
                        key: value
                        for key, value in safety_error.items()
                        if key != "reason"
                    },
                },
            )

        return operator_apply_preflight_payload(
            proposal=proposal,
            operator_mode=operator_mode,
            status="ready",
            idempotency=idempotency,
            requested_change=requested_change,
            normalized_change=normalized_change,
            extra=preflight_extra,
        )

    def batch_result_id(result):
        proposal = result.get("proposal")
        if isinstance(proposal, dict):
            return str(proposal.get("id") or "")
        return str(result.get("proposal_id") or "")

    def preflight_result_normalized_changes(result):
        changes = result.get("normalized_changes")
        if isinstance(changes, list) and changes:
            return [change for change in changes if isinstance(change, dict)]
        change = result.get("normalized_change")
        if isinstance(change, dict) and change:
            return [change]
        return []

    def preflight_result_title(result):
        proposal = result.get("proposal")
        if isinstance(proposal, dict):
            return proposal.get("title")
        return None

    def plan_eligible_preflight_safety(result):
        safety = result.get("safety")
        if not isinstance(safety, dict):
            return False, "missing_preflight_safety"
        required_false = (
            "live_write",
            "apply_attempted",
            "queue_mutation",
            "proposal_status_mutation",
            "idempotency_reserved",
            "manual_console_write_called",
        )
        for key in required_false:
            if safety.get(key) is not False:
                return False, f"preflight_safety_{key}_not_false"
        if safety.get("live_apply_entrypoint") != "apply_operator_proposal":
            return False, "preflight_live_apply_entrypoint_mismatch"
        if safety.get("preflight_only") is not True:
            return False, "preflight_only_flag_missing"
        return True, None

    def plan_eligible_preflight_idempotency(result):
        idempotency = result.get("idempotency")
        if not isinstance(idempotency, dict):
            return False, "missing_preflight_idempotency", None
        key = str(idempotency.get("idempotency_key") or "").strip()
        if not key:
            return False, "missing_preflight_idempotency_key", None
        if idempotency.get("provided") is not True:
            return False, "preflight_idempotency_not_provided", key
        if idempotency.get("reserved") is not False:
            return False, "preflight_idempotency_already_reserved", key
        if idempotency.get("duplicate") is True:
            return False, "preflight_idempotency_duplicate", key
        return True, None, key

    def current_queue_proposal_for_plan(result, proposal_id):
        proposal = server.operator_proposal_queue.find(proposal_id=proposal_id)
        if proposal is None:
            return None, "proposal_not_found"
        current = proposal.to_dict()
        if current.get("status") != "pending":
            return current, "proposal_not_pending"
        if current.get("expired") is True:
            return current, "proposal_expired"
        if current.get("can_apply") is not True:
            return current, "proposal_no_longer_applyable"

        preflight_proposal = result.get("proposal")
        if isinstance(preflight_proposal, dict):
            preflight_updated_at = preflight_proposal.get("updated_at")
            current_updated_at = current.get("updated_at")
            if (
                preflight_updated_at
                and current_updated_at
                and preflight_updated_at != current_updated_at
            ):
                return current, "proposal_changed_since_preflight"
            for key in (
                "requested_change",
                "requested_changes",
                "review_contract_version",
                "review_content_digest",
            ):
                if preflight_proposal.get(key) != current.get(key):
                    return current, f"proposal_{key}_changed_since_preflight"
        return current, None

    def current_queue_proposal_for_sequence_step(step, proposal_id):
        proposal = server.operator_proposal_queue.find(proposal_id=proposal_id)
        if proposal is None:
            return None, "proposal_not_found"
        current = proposal.to_dict()
        if current.get("status") != "pending":
            return current, "proposal_not_pending"
        if current.get("expired") is True:
            return current, "proposal_expired"
        if current.get("can_apply") is not True:
            return current, "proposal_no_longer_applyable"

        step_proposal = step.get("current_proposal")
        if isinstance(step_proposal, dict):
            for key in (
                "updated_at",
                "requested_change",
                "requested_changes",
                "review_contract_version",
                "review_content_digest",
            ):
                if step_proposal.get(key) != current.get(key):
                    return current, f"proposal_{key}_changed_since_plan"
        apply_request = step.get("apply_request")
        if isinstance(apply_request, dict):
            if apply_request.get("expected_review_contract_version") is None:
                return current, "proposal_review_contract_version_required"
            if (
                apply_request.get("expected_review_contract_version")
                != current.get("review_contract_version")
            ):
                return current, "proposal_review_contract_version_mismatch"
            if not apply_request.get("expected_review_content_digest"):
                return current, "proposal_review_digest_required"
            if (
                apply_request.get("expected_review_content_digest")
                != current.get("review_content_digest")
            ):
                return current, "proposal_review_digest_mismatch"
        return current, None

    def plan_change_from_preflight(result, change, change_index):
        safety = result.get("safety") if isinstance(result.get("safety"), dict) else {}
        current_allowed = live_write_kind_set_for_safety_surface()
        allowed = set(safety.get("allowed_live_write_kinds") or [])
        if not allowed:
            allowed = set(current_allowed)
        else:
            allowed &= current_allowed
        value_type = str(change.get("value_type") or "").strip().lower()
        if value_type not in current_allowed or value_type not in allowed:
            return None, "preflight_change_value_type_not_plan_eligible"
        channel = change.get("channel")
        if channel is None:
            return None, "preflight_change_missing_channel"
        effective_value = coerce_float(change.get("effective_value"))
        if effective_value is None:
            return None, "preflight_change_missing_effective_value"
        if value_type == "fader" and effective_value > 0.0:
            return None, "preflight_change_fader_ceiling_exceeded"
        pre_read_value = change.get("pre_read_value", change.get("old_value"))
        return {
            "change_index": change_index,
            "channel": channel,
            "value_type": value_type,
            "address": change.get("address"),
            "wing_surface": change.get("wing_surface"),
            "wing_address": change.get("wing_address"),
            "proof_direction": change.get("proof_direction"),
            "old_value": change.get("old_value"),
            "pre_read_value": pre_read_value,
            "pre_read_status": change.get("pre_read_status"),
            "pre_read_source": change.get("pre_read_source"),
            "pre_read_cache_freshness": change.get("pre_read_cache_freshness"),
            "effective_value": effective_value,
            "delta_db": change.get("delta_db"),
            "rollback_value": pre_read_value,
            "rollback_expected": pre_read_value is not None,
        }, None

    def apply_sequence_plan_payload(
        *,
        status,
        preflight_results,
        planned_steps=None,
        blocked_results=None,
        reason=None,
    ):
        planned_steps = planned_steps or []
        blocked_results = blocked_results or []
        ready_count = sum(
            1
            for result in preflight_results
            if isinstance(result, dict) and result.get("ready_to_apply") is True
        )
        blocked_count = len(preflight_results) - ready_count
        operator_mode = server.get_operator_mode_status()
        return {
            "type": "operator_proposal_apply_sequence_plan",
            "status": status,
            "success": status == "ok",
            "reason": reason,
            "generated_at": time.time(),
            "operator_mode": operator_mode,
            "summary": {
                "input_preflight_result_count": len(preflight_results),
                "ready_result_count": ready_count,
                "blocked_result_count": blocked_count,
                "planned_apply_count": len(planned_steps),
                "plan_blocked_count": len(blocked_results),
                "planned_apply_proposal_ids": [
                    step["proposal_id"] for step in planned_steps
                ],
                "blocked_proposal_ids": [
                    item["proposal_id"]
                    for item in blocked_results
                    if item.get("proposal_id")
                ],
                "operator_mode_allows_apply_now": bool(
                    operator_mode.get("capabilities", {}).get("can_apply_to_console")
                ),
                "execution_allowed_by_plan": False,
                "live_write": False,
                "apply_attempted": False,
                "queue_mutation": False,
                "idempotency_reserved": False,
                "manual_console_write_called": False,
                "mutates_console_state": False,
                "queries_console": False,
                "live_widening_allowed": False,
                "queue_freshness_checked": True,
            },
            "safety": {
                "sequence_plan_only": True,
                "preflight_results_required": True,
                "current_queue_proposal_required": True,
                "preflight_result_is_not_apply_authority": True,
                "requires_operator_apply": True,
                "execution_allowed_by_plan": False,
                "live_write": False,
                "apply_attempted": False,
                "queue_mutation": False,
                "proposal_status_mutation": False,
                "idempotency_reserved": False,
                "manual_console_write_called": False,
                "mutates_console_state": False,
                "queries_console": False,
                "live_widening_allowed": False,
                "live_apply_entrypoint": "apply_operator_proposal",
                "allowed_live_write_kinds": allowed_live_write_kinds_for_safety_surface(
                    operator_mode
                ),
            },
            "planned_apply_sequence": planned_steps,
            "blocked_results": blocked_results,
        }

    async def handle_plan_operator_proposal_apply_sequence(websocket, data):
        state_payload, state_error = resolve_sequence_stage_payload(
            websocket,
            data,
            expected_stage="batch_preflight",
        )
        if state_error:
            await server.send_to_client(
                websocket,
                apply_sequence_plan_payload(
                    status="blocked",
                    reason=state_error,
                    preflight_results=[],
                ),
            )
            return
        source_payload = state_payload if state_payload is not None else data
        raw_results = source_payload.get("preflight_results")
        if not isinstance(raw_results, list):
            await server.send_to_client(
                websocket,
                apply_sequence_plan_payload(
                    status="blocked",
                    reason="missing_preflight_results",
                    preflight_results=[],
                ),
            )
            return

        preflight_results = [
            result for result in raw_results if isinstance(result, dict)
        ]
        invalid_count = len(raw_results) - len(preflight_results)
        planned_steps = []
        blocked_results = []
        if invalid_count:
            blocked_results.append({
                "proposal_id": None,
                "preflight_result_index": None,
                "reason": "invalid_preflight_result",
                "count": invalid_count,
            })

        for result_index, result in enumerate(preflight_results):
            proposal_id = batch_result_id(result)
            if result.get("ready_to_apply") is not True or result.get("status") != "ready":
                blocked_results.append({
                    "proposal_id": proposal_id,
                    "preflight_result_index": result_index,
                    "reason": result.get("reason") or "preflight_result_not_ready",
                    "preflight_status": result.get("status"),
                })
                continue
            if not proposal_id:
                blocked_results.append({
                    "proposal_id": None,
                    "preflight_result_index": result_index,
                    "reason": "missing_preflight_proposal_id",
                })
                continue

            current_proposal, current_proposal_reason = current_queue_proposal_for_plan(
                result,
                proposal_id,
            )
            if current_proposal_reason:
                blocked_results.append({
                    "proposal_id": proposal_id,
                    "preflight_result_index": result_index,
                    "reason": current_proposal_reason,
                    "current_proposal": current_proposal,
                })
                continue

            safety_ok, safety_reason = plan_eligible_preflight_safety(result)
            if not safety_ok:
                blocked_results.append({
                    "proposal_id": proposal_id,
                    "preflight_result_index": result_index,
                    "reason": safety_reason,
                })
                continue

            idempotency_ok, idempotency_reason, idempotency_key = (
                plan_eligible_preflight_idempotency(result)
            )
            if not idempotency_ok:
                blocked_results.append({
                    "proposal_id": proposal_id,
                    "preflight_result_index": result_index,
                    "reason": idempotency_reason,
                })
                continue

            normalized_changes = preflight_result_normalized_changes(result)
            if not normalized_changes:
                blocked_results.append({
                    "proposal_id": proposal_id,
                    "preflight_result_index": result_index,
                    "reason": "preflight_has_no_normalized_change",
                })
                continue

            planned_changes = []
            blocked_change_reason = None
            for change_index, change in enumerate(normalized_changes):
                planned_change, blocked_change_reason = plan_change_from_preflight(
                    result,
                    change,
                    change_index,
                )
                if blocked_change_reason:
                    break
                planned_changes.append(planned_change)
            if blocked_change_reason:
                blocked_results.append({
                    "proposal_id": proposal_id,
                    "preflight_result_index": result_index,
                    "reason": blocked_change_reason,
                })
                continue

            planned_steps.append({
                "planned_apply_index": len(planned_steps),
                "proposal_id": proposal_id,
                "proposal_title": preflight_result_title(result),
                "idempotency_key": idempotency_key,
                "approval_required": True,
                "requires_operator_apply": True,
                "live_apply_entrypoint": "apply_operator_proposal",
                "execution_allowed_by_plan": False,
                "current_proposal": current_proposal,
                "change_count": len(planned_changes),
                "normalized_changes": planned_changes,
                "apply_request": {
                    "type": "apply_operator_proposal",
                    "proposal_id": proposal_id,
                    "approved": True,
                    "idempotency_key": idempotency_key,
                    "expected_review_contract_version": current_proposal.get(
                        "review_contract_version"
                    ),
                    "expected_review_content_digest": current_proposal.get(
                        "review_content_digest"
                    ),
                },
            })

        payload = apply_sequence_plan_payload(
            status="ok",
            preflight_results=preflight_results,
            planned_steps=planned_steps,
            blocked_results=blocked_results,
        )
        wire_payload, state_error = bind_sequence_stage_payload(
            websocket,
            payload,
            stage="sequence_plan",
        )
        if state_error:
            wire_payload = apply_sequence_plan_payload(
                status="blocked",
                reason=state_error,
                preflight_results=preflight_results,
                planned_steps=[],
                blocked_results=blocked_results,
            )
        await server.send_to_client(websocket, wire_payload)

    def planned_sequence_from_data(data):
        raw_plan = data.get("sequence_plan")
        if isinstance(raw_plan, dict):
            raw_steps = raw_plan.get("planned_apply_sequence")
        else:
            raw_steps = data.get("planned_apply_sequence")
        if isinstance(raw_steps, list):
            return raw_steps
        return None

    def dry_run_results_from_data(data):
        raw_payload = data.get("dry_run_payload")
        if isinstance(raw_payload, dict):
            raw_results = raw_payload.get("dry_run_results")
        else:
            raw_results = data.get("dry_run_results")
        if isinstance(raw_results, list):
            return raw_results
        return None

    def dry_run_step_change(change):
        if not isinstance(change, dict):
            return None, "invalid_planned_change"
        value_type = str(change.get("value_type") or "").strip().lower()
        if value_type not in live_write_kind_set_for_safety_surface():
            return None, "planned_change_value_type_not_dry_run_eligible"
        channel = change.get("channel")
        if channel is None:
            return None, "planned_change_missing_channel"
        effective_value = coerce_float(change.get("effective_value"))
        if effective_value is None:
            return None, "planned_change_missing_effective_value"
        if value_type == "fader" and effective_value > 0.0:
            return None, "planned_change_fader_ceiling_exceeded"
        return {
            "change_index": change.get("change_index"),
            "channel": channel,
            "value_type": value_type,
            "address": change.get("address"),
            "wing_surface": change.get("wing_surface"),
            "wing_address": change.get("wing_address"),
            "proof_direction": change.get("proof_direction"),
            "effective_value": effective_value,
            "delta_db": change.get("delta_db"),
            "pre_read_value": change.get("pre_read_value"),
            "pre_read_status": change.get("pre_read_status"),
            "rollback_value": change.get("rollback_value"),
            "rollback_expected": bool(change.get("rollback_expected")),
        }, None

    def dry_run_sequence_payload(*, status, steps, results=None, reason=None):
        results = results or []
        would_apply = [item for item in results if item.get("status") == "would_apply"]
        blocked = [item for item in results if item.get("status") == "blocked"]
        return {
            "type": "operator_proposal_apply_sequence_dry_run",
            "status": status,
            "success": status == "ok",
            "reason": reason,
            "generated_at": time.time(),
            "summary": {
                "input_step_count": len(steps),
                "result_count": len(results),
                "would_apply_count": len(would_apply),
                "blocked_count": len(blocked),
                "would_apply_proposal_ids": [
                    item["proposal_id"] for item in would_apply if item.get("proposal_id")
                ],
                "blocked_proposal_ids": [
                    item["proposal_id"] for item in blocked if item.get("proposal_id")
                ],
                "execution_allowed_by_dry_run": False,
                "live_write": False,
                "apply_attempted": False,
                "queue_mutation": False,
                "proposal_status_mutation": False,
                "idempotency_reserved": False,
                "manual_console_write_called": False,
                "mutates_console_state": False,
                "queries_console": False,
                "live_widening_allowed": False,
                "queue_freshness_checked": True,
            },
            "safety": {
                "dry_run_only": True,
                "sequence_plan_required": True,
                "current_queue_proposal_required": True,
                "does_not_call_apply_operator_proposal": True,
                "execution_allowed_by_dry_run": False,
                "requires_operator_apply": True,
                "live_write": False,
                "apply_attempted": False,
                "queue_mutation": False,
                "proposal_status_mutation": False,
                "idempotency_reserved": False,
                "manual_console_write_called": False,
                "mutates_console_state": False,
                "queries_console": False,
                "live_widening_allowed": False,
                "live_apply_entrypoint": "apply_operator_proposal",
                "allowed_live_write_kinds": allowed_live_write_kinds_for_safety_surface(),
            },
            "dry_run_results": results,
        }

    def execution_gate_payload(
        *,
        status,
        dry_run_results,
        candidates=None,
        blocked=None,
        reason=None,
        operator_ack=False,
    ):
        candidates = candidates or []
        blocked = blocked or []
        operator_mode = server.get_operator_mode_status()
        return {
            "type": "operator_proposal_apply_sequence_execution_gate",
            "status": status,
            "success": status == "ready",
            "reason": reason,
            "generated_at": time.time(),
            "operator_mode": operator_mode,
            "summary": {
                "input_dry_run_result_count": len(dry_run_results),
                "ready_for_executor_count": len(candidates),
                "blocked_count": len(blocked),
                "ready_for_executor_proposal_ids": [
                    item["proposal_id"] for item in candidates if item.get("proposal_id")
                ],
                "blocked_proposal_ids": [
                    item["proposal_id"] for item in blocked if item.get("proposal_id")
                ],
                "operator_mode_allows_apply_now": bool(
                    operator_mode.get("capabilities", {}).get("can_apply_to_console")
                ),
                "operator_ack_received": operator_ack,
                "execution_allowed_by_gate": False,
                "requires_separate_executor": True,
                "live_write": False,
                "apply_attempted": False,
                "queue_mutation": False,
                "proposal_status_mutation": False,
                "idempotency_reserved": False,
                "manual_console_write_called": False,
                "mutates_console_state": False,
                "queries_console": False,
                "live_widening_allowed": False,
                "queue_freshness_checked": True,
            },
            "safety": {
                "execution_gate_only": True,
                "dry_run_results_required": True,
                "operator_ack_required": True,
                "current_queue_proposal_required": True,
                "does_not_call_apply_operator_proposal": True,
                "execution_allowed_by_gate": False,
                "requires_separate_executor": True,
                "live_write": False,
                "apply_attempted": False,
                "queue_mutation": False,
                "proposal_status_mutation": False,
                "idempotency_reserved": False,
                "manual_console_write_called": False,
                "mutates_console_state": False,
                "queries_console": False,
                "live_widening_allowed": False,
                "live_apply_entrypoint": "apply_operator_proposal",
                "allowed_live_write_kinds": allowed_live_write_kinds_for_safety_surface(
                    operator_mode
                ),
            },
            "ready_for_executor": candidates,
            "blocked_results": blocked,
        }

    def execution_gate_row_from_dry_run(result, result_index, seen_idempotency_keys):
        proposal_id = str(result.get("proposal_id") or "").strip()
        planned_apply_index = result.get("planned_apply_index", result_index)
        if result.get("status") != "would_apply":
            return None, {
                "planned_apply_index": planned_apply_index,
                "proposal_id": proposal_id or None,
                "dry_run_result_index": result_index,
                "status": "blocked",
                "reason": result.get("reason") or "dry_run_result_not_would_apply",
            }
        if not proposal_id:
            return None, {
                "planned_apply_index": planned_apply_index,
                "proposal_id": None,
                "dry_run_result_index": result_index,
                "status": "blocked",
                "reason": "missing_dry_run_proposal_id",
            }

        current_proposal, current_reason = current_queue_proposal_for_sequence_step(
            result,
            proposal_id,
        )
        if current_reason:
            return None, {
                "planned_apply_index": planned_apply_index,
                "proposal_id": proposal_id,
                "dry_run_result_index": result_index,
                "status": "blocked",
                "reason": current_reason,
                "current_proposal": current_proposal,
            }

        apply_request = result.get("apply_request")
        if not isinstance(apply_request, dict):
            return None, {
                "planned_apply_index": planned_apply_index,
                "proposal_id": proposal_id,
                "dry_run_result_index": result_index,
                "status": "blocked",
                "reason": "missing_apply_request",
            }
        if apply_request.get("type") != "apply_operator_proposal":
            return None, {
                "planned_apply_index": planned_apply_index,
                "proposal_id": proposal_id,
                "dry_run_result_index": result_index,
                "status": "blocked",
                "reason": "apply_request_type_mismatch",
            }
        if str(apply_request.get("proposal_id") or "").strip() != proposal_id:
            return None, {
                "planned_apply_index": planned_apply_index,
                "proposal_id": proposal_id,
                "dry_run_result_index": result_index,
                "status": "blocked",
                "reason": "apply_request_proposal_id_mismatch",
            }
        if apply_request.get("approved") is not True:
            return None, {
                "planned_apply_index": planned_apply_index,
                "proposal_id": proposal_id,
                "dry_run_result_index": result_index,
                "status": "blocked",
                "reason": "apply_request_approval_required",
            }

        idempotency_key = str(apply_request.get("idempotency_key") or "").strip()
        if not idempotency_key:
            return None, {
                "planned_apply_index": planned_apply_index,
                "proposal_id": proposal_id,
                "dry_run_result_index": result_index,
                "status": "blocked",
                "reason": "missing_idempotency_key",
            }
        store = getattr(server, "_operator_apply_idempotency_keys", {})
        if isinstance(store, dict) and idempotency_key in store:
            return None, {
                "planned_apply_index": planned_apply_index,
                "proposal_id": proposal_id,
                "dry_run_result_index": result_index,
                "status": "blocked",
                "reason": "duplicate_idempotency_key",
            }
        if idempotency_key in seen_idempotency_keys:
            return None, {
                "planned_apply_index": planned_apply_index,
                "proposal_id": proposal_id,
                "dry_run_result_index": result_index,
                "status": "blocked",
                "reason": "duplicate_idempotency_key_in_sequence",
            }
        seen_idempotency_keys.add(idempotency_key)

        raw_changes = result.get("normalized_changes")
        if not isinstance(raw_changes, list) or not raw_changes:
            return None, {
                "planned_apply_index": planned_apply_index,
                "proposal_id": proposal_id,
                "dry_run_result_index": result_index,
                "status": "blocked",
                "reason": "missing_planned_changes",
            }
        changes = []
        for raw_change in raw_changes:
            change, blocker = dry_run_step_change(raw_change)
            if blocker:
                return None, {
                    "planned_apply_index": planned_apply_index,
                    "proposal_id": proposal_id,
                    "dry_run_result_index": result_index,
                    "status": "blocked",
                    "reason": blocker,
                }
            changes.append(change)

        return {
            "planned_apply_index": planned_apply_index,
            "proposal_id": proposal_id,
            "dry_run_result_index": result_index,
            "status": "ready_for_executor",
            "execution_allowed_by_gate": False,
            "requires_separate_executor": True,
            "apply_request": {
                "type": "apply_operator_proposal",
                "proposal_id": proposal_id,
                "approved": True,
                "approval_id": str(apply_request.get("approval_id")).strip(),
                "idempotency_key": idempotency_key,
                "expected_review_contract_version": current_proposal.get(
                    "review_contract_version"
                ),
                "expected_review_content_digest": current_proposal.get(
                    "review_content_digest"
                ),
            },
            "current_proposal": current_proposal,
            "normalized_changes": changes,
        }, None

    async def handle_gate_operator_proposal_apply_sequence_execution(websocket, data):
        state_payload, state_error = resolve_sequence_stage_payload(
            websocket,
            data,
            expected_stage="dry_run",
        )
        if state_error:
            await server.send_to_client(
                websocket,
                execution_gate_payload(
                    status="blocked",
                    reason=state_error,
                    dry_run_results=[],
                    operator_ack=data.get("operator_ack") is True,
                ),
            )
            return
        source_payload = state_payload if state_payload is not None else data
        raw_results = dry_run_results_from_data(source_payload)
        operator_ack = data.get("operator_ack") is True
        approval_id = str(data.get("approval_id") or "").strip()
        if raw_results is None:
            await server.send_to_client(
                websocket,
                execution_gate_payload(
                    status="blocked",
                    reason="missing_dry_run_results",
                    dry_run_results=[],
                    operator_ack=operator_ack,
                ),
            )
            return

        dry_run_results = [
            result for result in raw_results if isinstance(result, dict)
        ]
        blocked = []
        invalid_count = len(raw_results) - len(dry_run_results)
        if invalid_count:
            blocked.append({
                "planned_apply_index": None,
                "proposal_id": None,
                "dry_run_result_index": None,
                "status": "blocked",
                "reason": "invalid_dry_run_result",
                "count": invalid_count,
            })

        if not operator_ack:
            await server.send_to_client(
                websocket,
                execution_gate_payload(
                    status="blocked",
                    reason="operator_ack_required",
                    dry_run_results=dry_run_results,
                    operator_ack=operator_ack,
                    blocked=[
                        *blocked,
                        {
                            "planned_apply_index": None,
                            "proposal_id": None,
                            "dry_run_result_index": None,
                            "status": "blocked",
                            "reason": "operator_ack_required",
                        },
                    ],
                ),
            )
            return
        if not approval_id:
            await server.send_to_client(
                websocket,
                execution_gate_payload(
                    status="blocked",
                    reason="approval_id_required",
                    dry_run_results=dry_run_results,
                    operator_ack=operator_ack,
                    blocked=[
                        *blocked,
                        {
                            "planned_apply_index": None,
                            "proposal_id": None,
                            "dry_run_result_index": None,
                            "status": "blocked",
                            "reason": "approval_id_required",
                        },
                    ],
                ),
            )
            return

        operator_mode = server.get_operator_mode_status()
        if not operator_mode.get("capabilities", {}).get("can_apply_to_console"):
            await server.send_to_client(
                websocket,
                execution_gate_payload(
                    status="blocked",
                    reason="operator_mode_blocks_live_write",
                    dry_run_results=dry_run_results,
                    operator_ack=operator_ack,
                    blocked=[
                        *blocked,
                        {
                            "planned_apply_index": None,
                            "proposal_id": None,
                            "dry_run_result_index": None,
                            "status": "blocked",
                            "reason": "operator_mode_blocks_live_write",
                        },
                    ],
                ),
            )
            return

        candidates = []
        seen_idempotency_keys = set()
        for result_index, result in enumerate(dry_run_results):
            candidate, blocker = execution_gate_row_from_dry_run(
                result,
                result_index,
                seen_idempotency_keys,
            )
            if blocker:
                blocked.append(blocker)
            else:
                candidate["apply_request"] = {
                    **dict(candidate["apply_request"]),
                    "approval_id": approval_id,
                }
                candidates.append(candidate)

        payload = execution_gate_payload(
            status="ready" if candidates and not blocked else "blocked",
            dry_run_results=dry_run_results,
            candidates=candidates,
            blocked=blocked,
            operator_ack=operator_ack,
            reason=None if candidates and not blocked else "execution_gate_blocked",
        )
        wire_payload, state_error = bind_sequence_stage_payload(
            websocket,
            payload,
            stage="execution_gate",
        )
        if state_error:
            wire_payload = execution_gate_payload(
                status="blocked",
                dry_run_results=dry_run_results,
                candidates=[],
                blocked=blocked,
                operator_ack=operator_ack,
                reason=state_error,
            )
        await server.send_to_client(websocket, wire_payload)

    async def handle_dry_run_operator_proposal_apply_sequence(websocket, data):
        state_payload, state_error = resolve_sequence_stage_payload(
            websocket,
            data,
            expected_stage="sequence_plan",
        )
        if state_error:
            await server.send_to_client(
                websocket,
                dry_run_sequence_payload(
                    status="blocked",
                    reason=state_error,
                    steps=[],
                ),
            )
            return
        source_payload = state_payload if state_payload is not None else data
        raw_steps = planned_sequence_from_data(source_payload)
        if raw_steps is None:
            await server.send_to_client(
                websocket,
                dry_run_sequence_payload(
                    status="blocked",
                    reason="missing_planned_apply_sequence",
                    steps=[],
                ),
            )
            return

        steps = [step for step in raw_steps if isinstance(step, dict)]
        invalid_count = len(raw_steps) - len(steps)
        results = []
        if invalid_count:
            results.append({
                "planned_apply_index": None,
                "proposal_id": None,
                "status": "blocked",
                "reason": "invalid_planned_step",
                "count": invalid_count,
            })

        idempotency_store = getattr(server, "_operator_apply_idempotency_keys", {})
        if not isinstance(idempotency_store, dict):
            idempotency_store = {}
        seen_idempotency_keys = set()

        for step_index, step in enumerate(steps):
            proposal_id = str(step.get("proposal_id") or "").strip()
            planned_apply_index = step.get("planned_apply_index", step_index)
            if not proposal_id:
                results.append({
                    "planned_apply_index": planned_apply_index,
                    "proposal_id": None,
                    "status": "blocked",
                    "reason": "missing_planned_proposal_id",
                })
                continue
            if step.get("requires_operator_apply") is not True:
                results.append({
                    "planned_apply_index": planned_apply_index,
                    "proposal_id": proposal_id,
                    "status": "blocked",
                    "reason": "planned_step_missing_operator_apply_requirement",
                })
                continue
            if step.get("execution_allowed_by_plan") is not False:
                results.append({
                    "planned_apply_index": planned_apply_index,
                    "proposal_id": proposal_id,
                    "status": "blocked",
                    "reason": "planned_step_execution_authority_forged",
                })
                continue
            if step.get("live_apply_entrypoint") != "apply_operator_proposal":
                results.append({
                    "planned_apply_index": planned_apply_index,
                    "proposal_id": proposal_id,
                    "status": "blocked",
                    "reason": "planned_step_live_apply_entrypoint_mismatch",
                })
                continue

            current_proposal, current_reason = current_queue_proposal_for_sequence_step(
                step,
                proposal_id,
            )
            if current_reason:
                results.append({
                    "planned_apply_index": planned_apply_index,
                    "proposal_id": proposal_id,
                    "status": "blocked",
                    "reason": current_reason,
                    "current_proposal": current_proposal,
                })
                continue

            apply_request = step.get("apply_request")
            if not isinstance(apply_request, dict):
                results.append({
                    "planned_apply_index": planned_apply_index,
                    "proposal_id": proposal_id,
                    "status": "blocked",
                    "reason": "missing_apply_request",
                })
                continue
            if apply_request.get("type") != "apply_operator_proposal":
                results.append({
                    "planned_apply_index": planned_apply_index,
                    "proposal_id": proposal_id,
                    "status": "blocked",
                    "reason": "apply_request_type_mismatch",
                })
                continue
            if str(apply_request.get("proposal_id") or "").strip() != proposal_id:
                results.append({
                    "planned_apply_index": planned_apply_index,
                    "proposal_id": proposal_id,
                    "status": "blocked",
                    "reason": "apply_request_proposal_id_mismatch",
                })
                continue
            if apply_request.get("approved") is not True:
                results.append({
                    "planned_apply_index": planned_apply_index,
                    "proposal_id": proposal_id,
                    "status": "blocked",
                    "reason": "apply_request_approval_required",
                })
                continue
            step_idempotency_key = str(step.get("idempotency_key") or "").strip()
            request_idempotency_key = str(
                apply_request.get("idempotency_key") or ""
            ).strip()
            if (
                step_idempotency_key
                and request_idempotency_key
                and step_idempotency_key != request_idempotency_key
            ):
                results.append({
                    "planned_apply_index": planned_apply_index,
                    "proposal_id": proposal_id,
                    "status": "blocked",
                    "reason": "idempotency_key_mismatch",
                })
                continue
            idempotency_key = str(
                step_idempotency_key
                or request_idempotency_key
                or ""
            ).strip()
            if not idempotency_key:
                results.append({
                    "planned_apply_index": planned_apply_index,
                    "proposal_id": proposal_id,
                    "status": "blocked",
                    "reason": "missing_idempotency_key",
                })
                continue
            if idempotency_key in idempotency_store:
                results.append({
                    "planned_apply_index": planned_apply_index,
                    "proposal_id": proposal_id,
                    "status": "blocked",
                    "reason": "duplicate_idempotency_key",
                })
                continue
            if idempotency_key in seen_idempotency_keys:
                results.append({
                    "planned_apply_index": planned_apply_index,
                    "proposal_id": proposal_id,
                    "status": "blocked",
                    "reason": "duplicate_idempotency_key_in_sequence",
                })
                continue
            seen_idempotency_keys.add(idempotency_key)

            raw_changes = step.get("normalized_changes")
            if not isinstance(raw_changes, list) or not raw_changes:
                results.append({
                    "planned_apply_index": planned_apply_index,
                    "proposal_id": proposal_id,
                    "status": "blocked",
                    "reason": "missing_planned_changes",
                })
                continue
            changes = []
            change_blocker = None
            for raw_change in raw_changes:
                change, change_blocker = dry_run_step_change(raw_change)
                if change_blocker:
                    break
                changes.append(change)
            if change_blocker:
                results.append({
                    "planned_apply_index": planned_apply_index,
                    "proposal_id": proposal_id,
                    "status": "blocked",
                    "reason": change_blocker,
                })
                continue

            results.append({
                "planned_apply_index": planned_apply_index,
                "proposal_id": proposal_id,
                "status": "would_apply",
                "reason": None,
                "apply_request": {
                    "type": "apply_operator_proposal",
                    "proposal_id": proposal_id,
                    "approved": bool(apply_request.get("approved", True)),
                    "idempotency_key": idempotency_key,
                    "expected_review_contract_version": current_proposal.get(
                        "review_contract_version"
                    ),
                    "expected_review_content_digest": current_proposal.get(
                        "review_content_digest"
                    ),
                },
                "current_proposal": current_proposal,
                "normalized_changes": changes,
            })

        payload = dry_run_sequence_payload(
            status="ok",
            steps=steps,
            results=results,
        )
        wire_payload, state_error = bind_sequence_stage_payload(
            websocket,
            payload,
            stage="dry_run",
        )
        if state_error:
            wire_payload = dry_run_sequence_payload(
                status="blocked",
                reason=state_error,
                steps=steps,
                results=[],
            )
        await server.send_to_client(websocket, wire_payload)

    async def handle_preflight_operator_proposal_apply_batch(websocket, data):
        raw_ids = data.get("proposal_ids")
        explicit_ids = [
            str(item)
            for item in raw_ids
            if str(item).strip()
        ] if isinstance(raw_ids, list) else []
        try:
            limit = int(data.get("limit", 20))
        except (TypeError, ValueError):
            limit = 20
        limit = max(1, min(limit, 50))
        include_analysis_only = bool(data.get("include_analysis_only", False))

        pending_actions = server.operator_proposal_queue.pending_actions()
        skipped_analysis_only_ids = []
        selected_ids = []
        if explicit_ids:
            selected_ids = explicit_ids[:limit]
        else:
            for action in pending_actions:
                proposal_id = str(action.get("id") or "")
                if not proposal_id:
                    continue
                if action.get("can_apply") is True or include_analysis_only:
                    selected_ids.append(proposal_id)
                else:
                    skipped_analysis_only_ids.append(proposal_id)
                if len(selected_ids) >= limit:
                    break

        results = []
        not_found_ids = []
        for proposal_id in selected_ids:
            proposal = server.operator_proposal_queue.find(proposal_id=proposal_id)
            if proposal is None:
                not_found_ids.append(proposal_id)
                results.append(
                    operator_apply_preflight_payload(
                        operator_mode=server.get_operator_mode_status(),
                        reason="proposal_not_found",
                        idempotency=idempotency_preflight({
                            **dict(data),
                            "proposal_id": proposal_id,
                            "idempotency_key": batch_idempotency_key(data, proposal_id),
                        }),
                        extra={"batch": True, "proposal_id": proposal_id},
                    )
                )
                continue
            results.append(
                await build_operator_proposal_apply_preflight_result(
                    proposal,
                    data,
                    batch=True,
                )
            )

        ready_results = [result for result in results if result.get("ready_to_apply")]
        blocked_results = [result for result in results if not result.get("ready_to_apply")]
        ready_ids = [batch_result_id(result) for result in ready_results if batch_result_id(result)]
        blocked_ids = [batch_result_id(result) for result in blocked_results if batch_result_id(result)]
        payload = {
                "type": "operator_proposal_apply_preflight_batch",
                "status": "ok",
                "success": True,
                "generated_at": time.time(),
                "summary": {
                    "selected_count": len(selected_ids),
                    "preflight_result_count": len(results),
                    "ready_count": len(ready_results),
                    "blocked_count": len(blocked_results),
                    "skipped_analysis_only_count": len(skipped_analysis_only_ids),
                    "not_found_count": len(not_found_ids),
                    "ready_proposal_ids": ready_ids,
                    "blocked_proposal_ids": blocked_ids,
                    "skipped_analysis_only_ids": skipped_analysis_only_ids,
                    "not_found_ids": not_found_ids,
                    "approved": data.get("approved") is True,
                    "idempotency_key_prefix": (
                        str(data.get("idempotency_key_prefix") or "").strip() or None
                    ),
                    "live_write": False,
                    "apply_attempted": False,
                    "queue_mutation": False,
                    "idempotency_reserved": False,
                },
                "safety": {
                    "batch_preflight_only": True,
                    "live_write": False,
                    "apply_attempted": False,
                    "queue_mutation": False,
                    "proposal_status_mutation": False,
                    "idempotency_reserved": False,
                    "manual_console_write_called": False,
                    "live_apply_entrypoint": "apply_operator_proposal",
                },
                "preflight_results": results,
            }
        wire_payload, state_error = bind_sequence_stage_payload(
            websocket,
            payload,
            stage="batch_preflight",
        )
        if state_error:
            payload = {
                **payload,
                "status": "blocked",
                "success": False,
                "reason": state_error,
                "preflight_results": [],
            }
            wire_payload = payload
        await server.send_to_client(websocket, wire_payload)

    async def handle_m110_capability_apply(websocket, data):
        """Apply one exact restore-only proof without widening normal mode."""

        token = data.pop("supervised_capability_token", None)
        request = dict(data)
        valid, validation_reason = validate_m110_capability_apply_request(
            {**request, "supervised_capability_token": token}
        )
        if not valid:
            result = preclaim_apply_blocked_result(
                None,
                validation_reason,
                proposal_id=(
                    request.get("proposal_id")
                    if isinstance(request.get("proposal_id"), str)
                    else None
                ),
            )
            return await send_operator_apply_response(
                websocket,
                "operator_proposal_apply_blocked",
                result,
            )

        request_facts = FrozenApplyRequestFacts(
            approved=True,
            approval_id=request["approval_id"],
            approval_token="",
            idempotency_key=request["idempotency_key"],
            expected_review_contract_version=request[
                "expected_review_contract_version"
            ],
            expected_review_content_digest=request[
                "expected_review_content_digest"
            ],
        )
        proposal_id = request["proposal_id"]
        proposal = server.operator_proposal_queue.find(
            proposal_id=proposal_id
        )
        if proposal is None:
            result = preclaim_apply_blocked_result(
                None,
                "proposal_not_found",
                proposal_id=proposal_id,
            )
            return await send_operator_apply_response(
                websocket,
                "operator_proposal_apply_blocked",
                result,
            )

        review_validation = (
            server.operator_proposal_queue.validate_review_request(
                proposal,
                expected_review_contract_version=(
                    request_facts.expected_review_contract_version
                ),
                expected_review_content_digest=(
                    request_facts.expected_review_content_digest
                ),
            )
        )
        if not review_validation.get("valid"):
            result = preclaim_apply_blocked_result(
                proposal,
                review_validation.get("reason"),
            )
            return await send_operator_apply_response(
                websocket,
                "operator_proposal_apply_blocked",
                result,
            )
        if not server.operator_proposal_queue.result_evidence_contract_ready:
            reason = (
                "operator_proposal_result_recovery_required"
                if server.operator_proposal_queue.result_recovery_hazards
                else "operator_proposal_result_evidence_contract_incomplete"
            )
            result = preclaim_apply_blocked_result(proposal, reason)
            return await send_operator_apply_response(
                websocket,
                "operator_proposal_apply_blocked",
                result,
            )

        preclaim_blockers = []
        if getattr(server, "_is_shutting_down", False):
            preclaim_blockers.append("server_shutdown_active")
        mixer = getattr(server, "mixer_client", None)
        if mixer is None or not getattr(mixer, "is_connected", False):
            preclaim_blockers.append("mixer_disconnected")
        if getattr(server, "_manual_write_emergency_stopped", False):
            preclaim_blockers.append("emergency_stop_active")
        if getattr(server, "_system_emergency_stop_in_progress", False):
            preclaim_blockers.append("emergency_stop_in_progress")
        if getattr(server, "_system_emergency_stop_errors", None):
            preclaim_blockers.append("emergency_stop_teardown_incomplete")
        if getattr(server, "_manual_write_wal_integrity_ok", True) is not True:
            preclaim_blockers.append("manual_write_wal_integrity_failed")
        if getattr(server, "_manual_write_recovery_transactions", None):
            preclaim_blockers.append("manual_write_wal_recovery_required")
        if getattr(server, "_manual_write_prepared_transactions", None):
            preclaim_blockers.append("manual_write_wal_transaction_unresolved")
        if getattr(server, "_console_writer_conflict_recorded", False):
            preclaim_blockers.append("console_writer_conflict_recorded")
        if preclaim_blockers:
            result = preclaim_apply_blocked_result(
                proposal,
                "m110_capability_global_hazard_active",
                blockers=sorted(set(preclaim_blockers)),
            )
            return await send_operator_apply_response(
                websocket,
                "operator_proposal_apply_blocked",
                result,
            )

        if proposal.status != "pending":
            result = preclaim_apply_blocked_result(
                proposal,
                "proposal_not_pending",
            )
            return await send_operator_apply_response(
                websocket,
                "operator_proposal_apply_blocked",
                result,
            )
        if proposal.is_expired():
            revoke = getattr(server, "revoke_m110_capability_for_proposal", None)
            if callable(revoke):
                revoke(
                    proposal.id,
                    reason="proposal_expired_before_capability_apply",
                )
            expired = server.operator_proposal_queue.expire_pending(proposal)
            expired["apply_result"] = {
                "status": "blocked",
                "reason": "proposal_expired",
                "expires_at": proposal.expires_at,
                "ttl_seconds": proposal.ttl_seconds,
                "no_console_send_proven": True,
            }
            result = FinalizedApplyResult(
                expired,
                trusted_receipt=_freeze_trusted_prewrite_no_send_receipt(
                    proposal,
                    reason_code="proposal_expired",
                ),
            )
            return await send_operator_apply_response(
                websocket,
                "operator_proposal_apply_blocked",
                result,
            )

        transaction_plan = (
            proposal.raw.get("supervised_transaction_plan")
            if isinstance(getattr(proposal, "raw", None), dict)
            else None
        )
        if not isinstance(transaction_plan, dict):
            result = preclaim_apply_blocked_result(
                proposal,
                "m110_restore_proof_supervised_plan_required",
            )
            return await send_operator_apply_response(
                websocket,
                "operator_proposal_apply_blocked",
                result,
            )

        preview = getattr(server, "preview_m110_capability_apply", None)
        if not callable(preview):
            result = preclaim_apply_blocked_result(
                proposal,
                "m110_capability_apply_path_unavailable",
            )
            return await send_operator_apply_response(
                websocket,
                "operator_proposal_apply_blocked",
                result,
            )
        preview_result = preview(
            websocket=websocket,
            request=request,
            token=token,
            proposal=proposal,
        )
        if preview_result.get("valid") is not True:
            result = preclaim_apply_blocked_result(
                proposal,
                preview_result.get("reason")
                or "m110_capability_apply_preview_failed",
            )
            return await send_operator_apply_response(
                websocket,
                "operator_proposal_apply_blocked",
                result,
            )

        apply_capability = getattr(
            server,
            "apply_m110_restore_only_proof_transaction",
            None,
        )
        record_receipt = getattr(
            server,
            "record_m110_restored_proof_receipt",
            None,
        )
        if not callable(apply_capability) or not callable(record_receipt):
            result = preclaim_apply_blocked_result(
                proposal,
                "m110_capability_apply_path_unavailable",
            )
            return await send_operator_apply_response(
                websocket,
                "operator_proposal_apply_blocked",
                result,
            )

        idempotency_store = getattr(
            server,
            "_operator_apply_idempotency_keys",
            {},
        )
        if request_facts.idempotency_key in idempotency_store:
            reservation = server.reserve_operator_apply_idempotency_key(
                request_facts.idempotency_key,
                proposal_id=proposal.id,
                approval_id=request_facts.approval_id,
            )
            result = preclaim_apply_blocked_result(
                proposal,
                reservation.get("reason")
                or "operator_apply_idempotency_conflict",
                idempotency=reservation,
            )
            return await send_operator_apply_response(
                websocket,
                "operator_proposal_apply_blocked",
                result,
            )

        result_reservation = (
            server.operator_proposal_queue.reserve_result_evidence(proposal)
        )
        if not result_reservation.get("reserved"):
            result = preclaim_apply_blocked_result(
                proposal,
                result_reservation.get("reason")
                or "operator_proposal_result_evidence_reservation_failed",
            )
            return await send_operator_apply_response(
                websocket,
                "operator_proposal_apply_blocked",
                result,
            )
        evidence_lifecycle = ApplyEvidenceLifecycle(
            result_reservation.get("reservation_binding")
        )
        consume_capability = getattr(
            server,
            "consume_m110_capability_for_apply",
            None,
        )
        if not callable(consume_capability):
            release_apply_evidence_safely(evidence_lifecycle)
            result = preclaim_apply_blocked_result(
                proposal,
                "m110_capability_apply_path_unavailable",
            )
            return await send_operator_apply_response(
                websocket,
                "operator_proposal_apply_blocked",
                result,
            )
        consumed = consume_capability(
            websocket=websocket,
            request=request,
            token=token,
            proposal=proposal,
        )
        grant = consumed.get("grant") if isinstance(consumed, dict) else None
        if not (
            isinstance(consumed, dict)
            and consumed.get("consumed") is True
            and grant is not None
        ):
            release_apply_evidence_safely(evidence_lifecycle)
            result = preclaim_apply_blocked_result(
                proposal,
                (
                    consumed.get("reason")
                    if isinstance(consumed, dict)
                    else None
                )
                or "m110_capability_consume_failed",
                blockers=(
                    consumed.get("blockers")
                    if isinstance(consumed, dict)
                    else None
                ),
            )
            return await send_operator_apply_response(
                websocket,
                "operator_proposal_apply_blocked",
                result,
            )
        token = ""

        try:
            claim_result = await claim_and_reserve_operator_apply(
                proposal,
                request_facts=request_facts,
                evidence_lifecycle=evidence_lifecycle,
            )
        except (asyncio.CancelledError, Exception) as exc:
            cleanup_result = None
            if (
                evidence_lifecycle.state == "active"
                and evidence_lifecycle.trusted_identity is not None
            ):
                try:
                    cleanup_result = (
                        await finalize_apply_result_cancellation_safe(
                            proposal,
                            {
                                "status": "blocked",
                                "reason": (
                                    "m110_capability_claim_cancelled"
                                    if isinstance(exc, asyncio.CancelledError)
                                    else "m110_capability_claim_exception"
                                ),
                                "no_console_send_proven": True,
                            },
                            evidence_lifecycle=evidence_lifecycle,
                            idempotency_key=request_facts.idempotency_key,
                            reason_source="finalizer",
                        )
                    )
                except Exception:
                    server.operator_proposal_queue.latch_result_persistence_failure()
            elif evidence_lifecycle.state == "active":
                release_apply_evidence_safely(evidence_lifecycle)
            elif evidence_lifecycle.state != "consumed":
                server.operator_proposal_queue.latch_result_persistence_failure()
            mark_terminal = getattr(grant, "mark_terminal", None)
            if callable(mark_terminal):
                mark_terminal("claim_aborted_after_capability_consume")
            revoke = getattr(server, "revoke_m110_capability_for_proposal", None)
            if callable(revoke):
                revoke(
                    proposal.id,
                    reason="claim_aborted_after_capability_consume",
                )
            if isinstance(exc, asyncio.CancelledError):
                raise
            if isinstance(cleanup_result, FinalizedApplyResult):
                return await send_operator_apply_response(
                    websocket,
                    "operator_proposal_apply_blocked",
                    cleanup_result,
                )
            result = preclaim_apply_blocked_result(
                proposal,
                "m110_capability_claim_exception",
            )
            result["proposal_status_mutation"] = proposal.status != "pending"
            result["result_evidence_reserved"] = (
                evidence_lifecycle.state != "consumed"
            )
            return await send_operator_apply_response(
                websocket,
                "operator_proposal_apply_blocked",
                result,
            )
        if not claim_result.get("ready"):
            mark_terminal = getattr(grant, "mark_terminal", None)
            if callable(mark_terminal):
                mark_terminal("claim_failed_after_capability_consume")
            if (
                claim_result.get("reason") == "proposal_expired"
                and proposal.status == "pending"
            ):
                server.operator_proposal_queue.expire_pending(proposal)
            revoke = getattr(server, "revoke_m110_capability_for_proposal", None)
            if callable(revoke):
                revoke(
                    proposal.id,
                    reason="claim_failed_after_capability_consume",
                )
            result = claim_failure_apply_result(proposal, claim_result)
            return await send_operator_apply_response(
                websocket,
                "operator_proposal_apply_blocked",
                result,
            )
        execution = claim_result["execution"]
        evidence_lifecycle = claim_result["evidence_lifecycle"]
        _execution_content, execution_proposal = verified_execution_view(
            execution
        )
        transaction_plan = execution_proposal.raw.get(
            "supervised_transaction_plan"
        )
        if not isinstance(transaction_plan, dict):
            mark_terminal = getattr(grant, "mark_terminal", None)
            if callable(mark_terminal):
                mark_terminal("review_changed_after_capability_consume")
            result = await finalize_apply_result_cancellation_safe(
                proposal,
                {
                    "status": "blocked",
                    "reason": "proposal_review_content_integrity_error",
                    "no_console_send_proven": True,
                },
                evidence_lifecycle=evidence_lifecycle,
                idempotency_key=request_facts.idempotency_key,
                reason_source="validation",
            )
            return await send_operator_apply_response(
                websocket,
                "operator_proposal_apply_blocked",
                result,
            )

        try:
            apply_result = await apply_capability(
                websocket=websocket,
                request=request,
                capability_grant=grant,
                proposal=proposal,
                plan_payload=transaction_plan,
                result_evidence_reservation_binding=(
                    evidence_lifecycle.reservation_binding
                ),
            )
        except asyncio.CancelledError:
            # The private server lane drains every post-consume transaction.
            # Therefore a propagated cancellation is after consume but before
            # the private server lane began an adapter snapshot or send.
            await finalize_apply_result_cancellation_safe(
                proposal,
                {
                    "status": "blocked",
                    "success": False,
                    "reason": (
                        "m110_capability_apply_cancelled_after_consume_"
                        "before_execution"
                    ),
                    "no_console_send_proven": True,
                    "physical_verified": False,
                    "terminal_after_console_send": False,
                    "write_state_unknown": False,
                    "rollback_state_unknown": False,
                    "manual_review_required": False,
                    "wal_terminal": "not_prepared",
                },
                evidence_lifecycle=evidence_lifecycle,
                idempotency_key=request_facts.idempotency_key,
                reason_source="finalizer",
            )
            revoke = getattr(server, "revoke_m110_capability_for_proposal", None)
            if callable(revoke):
                revoke(
                    proposal.id,
                    reason="capability_apply_cancelled_after_consume",
                )
            mark_terminal = getattr(grant, "mark_terminal", None)
            if callable(mark_terminal):
                mark_terminal("cancelled_after_capability_consume")
            raise
        except Exception as exc:
            mark_terminal = getattr(grant, "mark_terminal", None)
            if callable(mark_terminal):
                mark_terminal("exception_after_capability_consume")
            result = await finalize_unknown_apply_failure(
                proposal,
                proposal_id=execution.proposal_id,
                reason="m110_capability_apply_exception",
                evidence_lifecycle=evidence_lifecycle,
                idempotency_key=request_facts.idempotency_key,
                reason_source="finalizer",
                error=exc,
            )
            revoke = getattr(server, "revoke_m110_capability_for_proposal", None)
            if callable(revoke) and proposal.status != "pending":
                revoke(
                    proposal.id,
                    reason="capability_apply_exception_terminal",
                )
            return await send_operator_apply_response(
                websocket,
                "operator_proposal_apply_result",
                result,
            )

        async def finalize_capability_terminal():
            try:
                finalized = await finalize_apply_result_cancellation_safe(
                    proposal,
                    apply_result,
                    evidence_lifecycle=evidence_lifecycle,
                    idempotency_key=request_facts.idempotency_key,
                    reason_source="supervised_transaction",
                    expected_operation_digest=request[
                        "expected_operation_digest"
                    ],
                )
                persisted_result = finalized.get("apply_result")
                if (
                    finalized.get("status") == "proof_passed"
                    and terminal_result_proves_verified_restored_proof(
                        persisted_result
                    )
                ):
                    try:
                        receipt_result = record_receipt(
                            websocket=websocket,
                            proposal=proposal,
                            terminal_result=persisted_result,
                            result_digest=finalized.get("result_digest"),
                        )
                    except Exception:
                        receipt_result = {
                            "recorded": False,
                            "reason": "m110_restored_proof_receipt_exception",
                        }
                    if receipt_result.get("recorded") is not True:
                        revoke_route = getattr(
                            server,
                            "_revoke_m110_route_authority",
                            None,
                        )
                        if callable(revoke_route):
                            revoke_route("restored_proof_receipt_not_recorded")
                try:
                    record_apply_trace(
                        proposal=execution_proposal,
                        operator_mode=server.get_operator_mode_status(),
                        requested_change={
                            "value_type": "supervised_transaction",
                            "transaction_id": transaction_plan.get(
                                "transaction_id"
                            ),
                            "operation_count": len(
                                transaction_plan.get("operations") or []
                            ),
                        },
                        idempotency_key=request_facts.idempotency_key,
                        status=finalized.get("status", "unknown"),
                        blocked_reason=(
                            (
                                persisted_result.get("reason")
                                if isinstance(persisted_result, dict)
                                else finalized.get("reason")
                            )
                            if finalized.get("status") != "proof_passed"
                            else None
                        ),
                        metadata={
                            "supervised_transaction": True,
                            "m110_restore_only_proof": True,
                            "capability_lane_used": True,
                            "normal_operator_mode_unchanged": True,
                            "normal_live_allowlist_changed": False,
                        },
                    )
                except Exception:
                    pass
                return finalized
            finally:
                revoke = getattr(
                    server,
                    "revoke_m110_capability_for_proposal",
                    None,
                )
                if callable(revoke):
                    revoke(
                        proposal.id,
                        reason="capability_apply_terminal",
                    )

        finalization_task = asyncio.create_task(
            finalize_capability_terminal()
        )
        while not finalization_task.done():
            try:
                await asyncio.shield(finalization_task)
            except asyncio.CancelledError:
                continue
        try:
            result = finalization_task.result()
        except Exception as exc:
            server.operator_proposal_queue.latch_result_persistence_failure()
            activated, teardown_complete = await activate_apply_emergency_stop(
                "m110_capability_finalization_failed_after_send:"
                f"{proposal.id}"
            )
            revoke_route = getattr(server, "_revoke_m110_route_authority", None)
            if callable(revoke_route):
                revoke_route("m110_capability_finalization_failed")
            identity = evidence_lifecycle.trusted_identity
            failure_result = {
                "status": "error",
                "reason": "operator_proposal_result_persistence_failed",
                "error_type": type(exc).__name__,
                "no_console_send_proven": False,
                "write_state_unknown": True,
                "physical_state_restored": None,
                "rollback_state_unknown": True,
                "manual_review_required": True,
                "emergency_stop_activated": activated,
                "emergency_stop_required": bool(
                    not activated or teardown_complete is not True
                ),
                "emergency_stop_teardown_complete": teardown_complete,
                "result_persistence_failed": True,
            }
            receipt = finalized_failure_receipt(
                identity,
                failure_result,
                proposal_status=str(getattr(proposal, "status", "unknown")),
                idempotency_key=request_facts.idempotency_key,
                result_evidence_pinned=True,
            )
            result = FinalizedApplyResult(
                {
                    "status": "apply_error",
                    "success": False,
                    "reason": "operator_proposal_result_persistence_failed",
                    "proposal": compact_proposal_reference(proposal),
                    "apply_result": failure_result,
                    "result_digest": None,
                    "result_detail_available": False,
                },
                trusted_receipt=receipt,
            )
        return await send_operator_apply_response(
            websocket,
            "operator_proposal_apply_result",
            result,
        )

    async def handle_apply_operator_proposal(websocket, data):
        launch_mode = (
            getattr(server, "_runtime_identity", {})
            .get("launch_binding", {})
            .get("mode")
        )
        if (
            launch_mode == "m110_attended_runner"
            or "supervised_capability_token" in data
        ):
            return await handle_m110_capability_apply(websocket, data)
        request_facts = FrozenApplyRequestFacts(
            approved=data.get("approved") is True,
            approval_id=str(data.get("approval_id") or "").strip(),
            approval_token=str(data.get("approval_token") or ""),
            idempotency_key=str(data.get("idempotency_key") or "").strip(),
            expected_review_contract_version=data.get(
                "expected_review_contract_version"
            ),
            expected_review_content_digest=str(
                data.get("expected_review_content_digest") or ""
            ),
        )
        proposal_id = str(data.get("proposal_id") or "").strip()
        if not proposal_id:
            result = preclaim_apply_blocked_result(
                None,
                "proposal_id_required_for_apply",
                proposal_id=None,
            )
            return await send_operator_apply_response(
                websocket, "operator_proposal_apply_blocked", result
            )
            return

        index = None
        proposal = server.operator_proposal_queue.find(proposal_id=proposal_id)
        operator_mode = server.get_operator_mode_status()
        if proposal is None:
            recorder = getattr(server, "record_automation_decision_trace", None)
            if callable(recorder):
                recorder(
                    idempotency_key=request_facts.idempotency_key,
                    proposal_id=proposal_id,
                    source="operator_apply",
                    mode=operator_mode.get("mode"),
                    target=None,
                    action_type="proposal_apply",
                    status="blocked",
                    safety_limits={"requires_idempotency_key": True},
                    proof_status="blocked",
                    blocked_reason="proposal_not_found",
                    metadata={"index": index},
                )
            result = preclaim_apply_blocked_result(
                None,
                "proposal_not_found",
                proposal_id=proposal_id,
                index=index,
            )
            return await send_operator_apply_response(
                websocket, "operator_proposal_apply_blocked", result
            )
            return

        review_validation = server.operator_proposal_queue.validate_review_request(
            proposal,
            expected_review_contract_version=(
                request_facts.expected_review_contract_version
            ),
            expected_review_content_digest=(
                request_facts.expected_review_content_digest
            ),
        )
        if not review_validation.get("valid"):
            result = preclaim_apply_blocked_result(
                proposal,
                review_validation.get("reason"),
            )
            return await send_operator_apply_response(
                websocket, "operator_proposal_apply_blocked", result
            )
            return
        if not server.operator_proposal_queue.result_evidence_contract_ready:
            recovery_hazards = (
                server.operator_proposal_queue.result_recovery_hazards
            )
            global_hazards = (
                collect_global_hazards(
                    server,
                    collect_compact_rows(server),
                )
                if recovery_hazards
                else None
            )
            result = preclaim_apply_blocked_result(
                proposal,
                (
                    "operator_proposal_result_recovery_required"
                    if recovery_hazards
                    else "operator_proposal_result_evidence_contract_incomplete"
                ),
            )
            return await send_operator_apply_response(
                websocket, "operator_proposal_apply_blocked", result
            )
            return

        global_hazards = collect_global_hazards(
            server,
            collect_compact_rows(server),
        )
        if global_hazards.get("fatal") is True:
            result = preclaim_apply_blocked_result(
                proposal,
                "operator_proposal_global_hazard_active",
            )
            return await send_operator_apply_response(
                websocket, "operator_proposal_apply_blocked", result
            )
            return

        if proposal.status != "pending":
            result = preclaim_apply_blocked_result(
                proposal,
                "proposal_not_pending",
            )
            return await send_operator_apply_response(
                websocket, "operator_proposal_apply_blocked", result
            )
            return

        if proposal.is_expired():
            result = server.operator_proposal_queue.expire_pending(proposal)
            result["apply_result"] = {
                "status": "blocked",
                "reason": "proposal_expired",
                "expires_at": proposal.expires_at,
                "ttl_seconds": proposal.ttl_seconds,
            }
            result = FinalizedApplyResult(
                result,
                trusted_receipt=_freeze_trusted_prewrite_no_send_receipt(
                    proposal,
                    reason_code="proposal_expired",
                ),
            )
            record_apply_trace(
                proposal=proposal,
                operator_mode=operator_mode,
                requested_change=(proposal.requested_change or {}),
                idempotency_key=request_facts.idempotency_key,
                status="blocked",
                blocked_reason="proposal_expired",
                metadata={"expires_at": proposal.expires_at, "ttl_seconds": proposal.ttl_seconds},
            )
            return await send_operator_apply_response(
                websocket, "operator_proposal_apply_blocked", result
            )
            return

        source_blocker = soundcheck_proposal_apply_blocker(proposal)
        if source_blocker:
            record_apply_trace(
                proposal=proposal,
                operator_mode=operator_mode,
                requested_change=(proposal.requested_change or {}),
                idempotency_key=request_facts.idempotency_key,
                status="blocked",
                blocked_reason=source_blocker,
                metadata={
                    "no_console_send_proven": True,
                    "proposal_status_mutation": False,
                    "idempotency_reserved": False,
                },
            )
            result = preclaim_apply_blocked_result(
                proposal,
                source_blocker,
                apply_blockers=[source_blocker],
            )
            return await send_operator_apply_response(
                websocket, "operator_proposal_apply_blocked", result
            )
            return

        idempotency_key = request_facts.idempotency_key
        idempotency_store = getattr(server, "_operator_apply_idempotency_keys", {})
        if not idempotency_key or (
            isinstance(idempotency_store, dict)
            and idempotency_key in idempotency_store
        ):
            reservation = server.reserve_operator_apply_idempotency_key(
                idempotency_key,
                proposal_id=getattr(proposal, "id", None),
                approval_id=request_facts.approval_id,
            )
            reason = reservation.get("reason")
            record_apply_trace(
                proposal=proposal,
                operator_mode=operator_mode,
                requested_change=(proposal.requested_change or {}),
                idempotency_key=idempotency_key or None,
                status="blocked",
                blocked_reason=reason,
                metadata={"idempotency": reservation},
            )
            result = preclaim_apply_blocked_result(
                proposal,
                reason,
                idempotency=reservation,
            )
            return await send_operator_apply_response(
                websocket, "operator_proposal_apply_blocked", result
            )
            return

        transaction_plan = (
            proposal.raw.get("supervised_transaction_plan")
            if isinstance(getattr(proposal, "raw", None), dict)
            else None
        )
        if isinstance(transaction_plan, dict):
            proposal_view = proposal.to_dict()
            apply_blockers = list(proposal_view.get("apply_blockers") or [])
            if apply_blockers:
                blocked_reason = str(apply_blockers[0]).split(":", 1)[0]
                result = preclaim_apply_blocked_result(
                    proposal,
                    blocked_reason,
                    apply_blockers=apply_blockers,
                )
                return await send_operator_apply_response(
                    websocket, "operator_proposal_apply_blocked", result
                )
                return
            approval_id = request_facts.approval_id
            if not approval_id:
                result = preclaim_apply_blocked_result(
                    proposal,
                    "approval_id_required",
                )
                return await send_operator_apply_response(
                    websocket, "operator_proposal_apply_blocked", result
                )
                return
            if not request_facts.approved:
                result = preclaim_apply_blocked_result(
                    proposal,
                    "approval_required",
                )
                return await send_operator_apply_response(
                    websocket, "operator_proposal_apply_blocked", result
                )
                return
            claim_result = await claim_and_reserve_operator_apply(
                proposal,
                request_facts=request_facts,
            )
            if not claim_result.get("ready"):
                result = claim_failure_apply_result(proposal, claim_result)
                return await send_operator_apply_response(
                    websocket, "operator_proposal_apply_blocked", result
                )
                return
            execution = claim_result["execution"]
            evidence_lifecycle = claim_result["evidence_lifecycle"]
            _execution_content, execution_proposal = verified_execution_view(
                execution
            )
            transaction_plan = execution_proposal.raw.get(
                "supervised_transaction_plan"
            )
            if not isinstance(transaction_plan, dict):
                result = await finalize_apply_result(
                    proposal,
                    {
                        "status": "blocked",
                        "reason": "proposal_review_content_integrity_error",
                        "no_console_send_proven": True,
                    },
                    evidence_lifecycle=evidence_lifecycle,
                    idempotency_key=idempotency_key,
                    reason_source="validation",
                )
                return await send_operator_apply_response(
                    websocket, "operator_proposal_apply_blocked", result
                )
                return
            apply_transaction = getattr(
                server,
                "apply_supervised_soundcheck_transaction",
                None,
            )
            if not callable(apply_transaction):
                apply_result = {
                    "status": "blocked",
                    "success": False,
                    "reason": "supervised_transaction_path_unavailable",
                    "no_console_send_proven": True,
                }
            else:
                try:
                    apply_result = await apply_transaction(
                        transaction_plan,
                        approved=True,
                        approval_id=approval_id,
                        idempotency_key=idempotency_key,
                        proof_run=False,
                        approval_expires_at_epoch=execution.expires_at_epoch,
                        result_evidence_reservation_binding=(
                            evidence_lifecycle.reservation_binding
                        ),
                    )
                except asyncio.CancelledError:
                    await finalize_unknown_apply_failure(
                        proposal,
                        proposal_id=execution.proposal_id,
                        reason="supervised_transaction_cancelled",
                        evidence_lifecycle=evidence_lifecycle,
                        idempotency_key=idempotency_key,
                        reason_source="supervised_transaction",
                    )
                    raise
                except Exception as exc:
                    result = await finalize_unknown_apply_failure(
                        proposal,
                        proposal_id=execution.proposal_id,
                        reason="supervised_transaction_exception",
                        evidence_lifecycle=evidence_lifecycle,
                        idempotency_key=idempotency_key,
                        reason_source="supervised_transaction",
                        error=exc,
                    )
                    return await send_operator_apply_response(
                        websocket, "operator_proposal_apply_result", result
                    )
                    return
            # Retry/release authority requires an explicit server proof.  The
            # absence of events or WAL fields is not evidence that a legacy or
            # partial result never reached the console.
            result = await finalize_apply_result(
                proposal,
                apply_result,
                evidence_lifecycle=evidence_lifecycle,
                idempotency_key=idempotency_key,
                reason_source="supervised_transaction",
            )
            record_child_result = getattr(
                server,
                "record_soundcheck_child_result",
                None,
            )
            if callable(record_child_result) and not isinstance(
                apply_result.get("soundcheck_readiness"),
                dict,
            ):
                readiness = record_child_result(
                    execution_proposal,
                    apply_result,
                )
                if isinstance(readiness, dict):
                    result["soundcheck_readiness"] = readiness
            elif isinstance(apply_result.get("soundcheck_readiness"), dict):
                result["soundcheck_readiness"] = dict(
                    apply_result["soundcheck_readiness"]
                )
            record_apply_trace(
                proposal=execution_proposal,
                operator_mode=operator_mode,
                requested_change={
                    "value_type": "supervised_transaction",
                    "transaction_id": transaction_plan.get("transaction_id"),
                    "operation_count": len(transaction_plan.get("operations") or []),
                },
                idempotency_key=idempotency_key,
                status=apply_result.get("status", "unknown"),
                blocked_reason=(
                    apply_result.get("reason")
                    if apply_result.get("status") != "applied"
                    else None
                ),
                metadata={"supervised_transaction": True},
            )
            return await send_operator_apply_response(
                websocket, "operator_proposal_apply_result", result
            )
            return

        requested_changes = getattr(proposal, "requested_changes", []) or []
        if requested_changes:
            approval_id = request_facts.approval_id
            if not approval_id:
                result = preclaim_apply_blocked_result(
                    proposal,
                    "approval_id_required",
                )
                return await send_operator_apply_response(
                    websocket, "operator_proposal_apply_blocked", result
                )
                return
            if not request_facts.approved:
                result = preclaim_apply_blocked_result(
                    proposal,
                    "approval_required",
                )
                return await send_operator_apply_response(
                    websocket, "operator_proposal_apply_blocked", result
                )
                return
            claim_result = await claim_and_reserve_operator_apply(
                proposal,
                request_facts=request_facts,
            )
            if not claim_result.get("ready"):
                result = claim_failure_apply_result(proposal, claim_result)
                return await send_operator_apply_response(
                    websocket, "operator_proposal_apply_blocked", result
                )
                return
            execution = claim_result["execution"]
            evidence_lifecycle = claim_result["evidence_lifecycle"]
            _execution_content, execution_proposal = verified_execution_view(
                execution
            )
            requested_changes = execution_proposal.requested_changes
            write_scope = getattr(server, "_console_write_scope", None)
            try:
                if callable(write_scope):
                    async with write_scope(
                        deadline_epoch=execution.expires_at_epoch,
                    ) as writer_acquired:
                        if writer_acquired is not True:
                            result = await linked_change_blocked_result(
                                proposal,
                                "operator_approval_expired",
                                evidence_lifecycle=evidence_lifecycle,
                                idempotency_key=idempotency_key,
                            )
                            return await send_operator_apply_response(
                                websocket,
                                "operator_proposal_apply_blocked",
                                result,
                            )
                            return
                        return await apply_linked_operator_proposal(
                            websocket,
                            request_facts,
                            proposal,
                            execution,
                            execution_proposal,
                            operator_mode,
                            requested_changes,
                            idempotency_key,
                            evidence_lifecycle,
                        )
                else:
                    return await apply_linked_operator_proposal(
                        websocket,
                        request_facts,
                        proposal,
                        execution,
                        execution_proposal,
                        operator_mode,
                        requested_changes,
                        idempotency_key,
                        evidence_lifecycle,
                    )
            except asyncio.CancelledError:
                if evidence_lifecycle.reservation_binding is not None:
                    await finalize_unknown_apply_failure(
                        proposal,
                        proposal_id=execution.proposal_id,
                        reason="linked_operator_proposal_cancelled",
                        evidence_lifecycle=evidence_lifecycle,
                        idempotency_key=idempotency_key,
                        reason_source="linked_apply",
                    )
                raise
            except Exception as exc:
                if evidence_lifecycle.reservation_binding is not None:
                    result = await finalize_unknown_apply_failure(
                        proposal,
                        proposal_id=execution.proposal_id,
                        reason="linked_operator_proposal_exception",
                        evidence_lifecycle=evidence_lifecycle,
                        idempotency_key=idempotency_key,
                        reason_source="linked_apply",
                        error=exc,
                    )
                    return await send_operator_apply_response(
                        websocket, "operator_proposal_apply_result", result
                    )
            return

        requested_change = proposal.requested_change or {}
        value_type = str(requested_change.get("value_type") or "").strip().lower()
        allowed_kinds = set(operator_mode["capabilities"].get("allowed_live_write_kinds") or [])
        if (
            not value_type
            or requested_change.get("channel") is None
            or requested_change.get("value") is None
        ):
            result = preclaim_apply_blocked_result(
                proposal,
                "proposal_has_no_applyable_requested_change",
                requested_change=requested_change or None,
            )
            record_apply_trace(
                proposal=proposal,
                operator_mode=operator_mode,
                requested_change=requested_change,
                idempotency_key=idempotency_key,
                status="blocked",
                blocked_reason="proposal_has_no_applyable_requested_change",
            )
            return await send_operator_apply_response(
                websocket, "operator_proposal_apply_blocked", result
            )
            return

        if not operator_mode["capabilities"]["can_apply_to_console"]:
            result = preclaim_apply_blocked_result(
                proposal,
                "operator_mode_blocks_live_write",
                operator_mode=operator_mode,
            )
            record_apply_trace(
                proposal=proposal,
                operator_mode=operator_mode,
                requested_change=requested_change,
                idempotency_key=idempotency_key,
                status="blocked",
                blocked_reason="operator_mode_blocks_live_write",
            )
            return await send_operator_apply_response(
                websocket, "operator_proposal_apply_blocked", result
            )
            return

        if value_type not in allowed_kinds:
            result = preclaim_apply_blocked_result(
                proposal,
                "unsupported_value_type",
                value_type=value_type,
                allowed_live_write_kinds=sorted(allowed_kinds),
            )
            record_apply_trace(
                proposal=proposal,
                operator_mode=operator_mode,
                requested_change=requested_change,
                idempotency_key=idempotency_key,
                status="blocked",
                blocked_reason="unsupported_value_type",
            )
            return await send_operator_apply_response(
                websocket, "operator_proposal_apply_blocked", result
            )
            return

        if not request_facts.approved:
            record_apply_trace(
                proposal=proposal,
                operator_mode=operator_mode,
                requested_change=requested_change,
                idempotency_key=idempotency_key,
                status="blocked",
                blocked_reason="approval_required",
            )
            result = preclaim_apply_blocked_result(
                proposal,
                "approval_required",
            )
            return await send_operator_apply_response(
                websocket, "operator_proposal_apply_blocked", result
            )
            return

        approval_id = request_facts.approval_id
        if not approval_id:
            result = preclaim_apply_blocked_result(
                proposal,
                "approval_id_required",
            )
            return await send_operator_apply_response(
                websocket, "operator_proposal_apply_blocked", result
            )
            return

        if not server.mixer_client or not getattr(server.mixer_client, "is_connected", True):
            result = preclaim_apply_blocked_result(
                proposal,
                "mixer_not_connected",
            )
            record_apply_trace(
                proposal=proposal,
                operator_mode=operator_mode,
                requested_change=requested_change,
                idempotency_key=idempotency_key,
                status="blocked",
                blocked_reason="mixer_not_connected",
            )
            return await send_operator_apply_response(
                websocket, "operator_proposal_apply_blocked", result
            )
            return

        apply_manual_console_write = getattr(server, "_apply_manual_console_write", None)
        if apply_manual_console_write is None:
            result = preclaim_apply_blocked_result(
                proposal,
                "supervised_apply_path_unavailable",
            )
            record_apply_trace(
                proposal=proposal,
                operator_mode=operator_mode,
                requested_change=requested_change,
                idempotency_key=idempotency_key,
                status="blocked",
                blocked_reason="supervised_apply_path_unavailable",
            )
            return await send_operator_apply_response(
                websocket, "operator_proposal_apply_blocked", result
            )
            return

        claim_result = await claim_and_reserve_operator_apply(
            proposal,
            request_facts=request_facts,
        )
        if not claim_result.get("ready"):
            result = claim_failure_apply_result(proposal, claim_result)
            return await send_operator_apply_response(
                websocket, "operator_proposal_apply_blocked", result
            )
            return

        execution = claim_result["execution"]
        evidence_lifecycle = claim_result["evidence_lifecycle"]
        _execution_content, execution_proposal = verified_execution_view(execution)
        requested_change = execution_proposal.requested_change or {}

        try:
            normalized_change, safety_error = await validate_change_before_apply(
                requested_change,
                proposal=execution_proposal,
                operator_authored=is_operator_authored(
                    execution_proposal,
                    request_facts.approval_token,
                ),
                operator_mode=operator_mode,
                linked=False,
            )
        except asyncio.CancelledError:
            await finalize_apply_result_cancellation_safe(
                proposal,
                {
                    "status": "blocked",
                    "success": False,
                    "reason": "operator_proposal_pre_read_cancelled",
                    "no_console_send_proven": True,
                    "physical_verified": False,
                    "terminal_after_console_send": False,
                    "write_state_unknown": False,
                    "rollback_state_unknown": False,
                    "manual_review_required": False,
                    "wal_terminal": "not_prepared",
                },
                evidence_lifecycle=evidence_lifecycle,
                idempotency_key=idempotency_key,
                reason_source="finalizer",
            )
            raise
        except Exception as exc:
            result = await finalize_apply_result(
                proposal,
                {
                    "status": "blocked",
                    "success": False,
                    "reason": "operator_proposal_pre_read_exception",
                    "no_console_send_proven": True,
                    "physical_verified": False,
                    "terminal_after_console_send": False,
                    "write_state_unknown": False,
                    "rollback_state_unknown": False,
                    "manual_review_required": False,
                    "wal_terminal": "not_prepared",
                    "error_type": type(exc).__name__,
                },
                evidence_lifecycle=evidence_lifecycle,
                idempotency_key=idempotency_key,
                reason_source="finalizer",
            )
            return await send_operator_apply_response(
                websocket,
                "operator_proposal_apply_blocked",
                result,
            )
        if safety_error:
            result = await finalize_apply_result(
                proposal,
                {
                    "status": "blocked",
                    "reason": safety_error["reason"],
                    "requested_change": requested_change,
                    "no_console_send_proven": True,
                    **{key: value for key, value in safety_error.items() if key not in {"reason", "requested_change"}},
                },
                evidence_lifecycle=evidence_lifecycle,
                idempotency_key=idempotency_key,
                reason_source="validation",
            )
            record_apply_trace(
                proposal=proposal,
                operator_mode=operator_mode,
                requested_change=requested_change,
                normalized_change=normalized_change,
                apply_result={"status": "blocked", **safety_error},
                idempotency_key=idempotency_key,
                status="blocked",
                blocked_reason=safety_error["reason"],
            )
            return await send_operator_apply_response(
                websocket, "operator_proposal_apply_blocked", result
            )
            return

        try:
            apply_result = await apply_manual_console_write(
                websocket,
                channel=normalized_change["channel"],
                value=normalized_change["effective_value"],
                value_type=normalized_change["value_type"],
                source=f"operator_proposal:{execution.proposal_id}",
                reason=execution_proposal.reason or execution_proposal.title,
                approved=True,
                approval_id=approval_id,
                idempotency_key=idempotency_key,
                proposal_id=execution.proposal_id,
                expected_current_value=normalized_change["old_value"],
                approval_expires_at_epoch=execution.expires_at_epoch,
                result_evidence_reservation_binding=(
                    evidence_lifecycle.reservation_binding
                ),
            )
        except asyncio.CancelledError:
            await finalize_unknown_apply_failure(
                proposal,
                proposal_id=execution.proposal_id,
                reason="operator_proposal_apply_cancelled",
                evidence_lifecycle=evidence_lifecycle,
                idempotency_key=idempotency_key,
                reason_source="finalizer",
            )
            raise
        except Exception as exc:
            result = await finalize_unknown_apply_failure(
                proposal,
                proposal_id=execution.proposal_id,
                reason="operator_proposal_apply_exception",
                evidence_lifecycle=evidence_lifecycle,
                idempotency_key=idempotency_key,
                reason_source="finalizer",
                error=exc,
            )
            return await send_operator_apply_response(
                websocket, "operator_proposal_apply_result", result
            )
            return
        result = await finalize_apply_result(
            proposal,
            apply_result,
            evidence_lifecycle=evidence_lifecycle,
            idempotency_key=idempotency_key,
            reason_source="manual_write",
        )
        record_apply_trace(
            proposal=execution_proposal,
            operator_mode=operator_mode,
            requested_change=requested_change,
            normalized_change=normalized_change,
            apply_result=apply_result,
            idempotency_key=idempotency_key,
            status=apply_result.get("status", "unknown"),
            blocked_reason=apply_result.get("reason") if apply_result.get("status") != "applied" else None,
        )
        remember_applied_delta(normalized_change, apply_result)
        return await send_operator_apply_response(
            websocket, "operator_proposal_apply_result", result
        )

    def executor_ready_rows_from_data(data):
        raw_payload = data.get("execution_gate_payload")
        if isinstance(raw_payload, dict):
            raw_rows = raw_payload.get("ready_for_executor")
        else:
            raw_rows = data.get("ready_for_executor")
        if isinstance(raw_rows, list):
            return raw_rows
        return None

    def sequence_executor_row_from_gate_candidate(
        candidate,
        candidate_index,
        seen_idempotency_keys,
    ):
        proposal_id = str(candidate.get("proposal_id") or "").strip()
        planned_apply_index = candidate.get("planned_apply_index", candidate_index)
        if candidate.get("status") != "ready_for_executor":
            return None, {
                "planned_apply_index": planned_apply_index,
                "proposal_id": proposal_id or None,
                "candidate_index": candidate_index,
                "status": "blocked",
                "reason": "gate_row_not_ready_for_executor",
            }
        if candidate.get("requires_separate_executor") is not True:
            return None, {
                "planned_apply_index": planned_apply_index,
                "proposal_id": proposal_id or None,
                "candidate_index": candidate_index,
                "status": "blocked",
                "reason": "gate_row_missing_separate_executor_requirement",
            }
        if candidate.get("execution_allowed_by_gate") is not False:
            return None, {
                "planned_apply_index": planned_apply_index,
                "proposal_id": proposal_id or None,
                "candidate_index": candidate_index,
                "status": "blocked",
                "reason": "gate_row_execution_authority_forged",
            }
        if not proposal_id:
            return None, {
                "planned_apply_index": planned_apply_index,
                "proposal_id": None,
                "candidate_index": candidate_index,
                "status": "blocked",
                "reason": "missing_gate_proposal_id",
            }

        current_proposal, current_reason = current_queue_proposal_for_sequence_step(
            candidate,
            proposal_id,
        )
        if current_reason:
            return None, {
                "planned_apply_index": planned_apply_index,
                "proposal_id": proposal_id,
                "candidate_index": candidate_index,
                "status": "blocked",
                "reason": current_reason,
                "current_proposal": current_proposal,
            }

        apply_request = candidate.get("apply_request")
        if not isinstance(apply_request, dict):
            return None, {
                "planned_apply_index": planned_apply_index,
                "proposal_id": proposal_id,
                "candidate_index": candidate_index,
                "status": "blocked",
                "reason": "missing_apply_request",
            }
        if apply_request.get("type") != "apply_operator_proposal":
            return None, {
                "planned_apply_index": planned_apply_index,
                "proposal_id": proposal_id,
                "candidate_index": candidate_index,
                "status": "blocked",
                "reason": "apply_request_type_mismatch",
            }
        if str(apply_request.get("proposal_id") or "").strip() != proposal_id:
            return None, {
                "planned_apply_index": planned_apply_index,
                "proposal_id": proposal_id,
                "candidate_index": candidate_index,
                "status": "blocked",
                "reason": "apply_request_proposal_id_mismatch",
            }
        if apply_request.get("approved") is not True:
            return None, {
                "planned_apply_index": planned_apply_index,
                "proposal_id": proposal_id,
                "candidate_index": candidate_index,
                "status": "blocked",
                "reason": "apply_request_approval_required",
            }
        if not str(apply_request.get("approval_id") or "").strip():
            return None, {
                "planned_apply_index": planned_apply_index,
                "proposal_id": proposal_id,
                "candidate_index": candidate_index,
                "status": "blocked",
                "reason": "apply_request_approval_id_required",
            }

        idempotency_key = str(apply_request.get("idempotency_key") or "").strip()
        if not idempotency_key:
            return None, {
                "planned_apply_index": planned_apply_index,
                "proposal_id": proposal_id,
                "candidate_index": candidate_index,
                "status": "blocked",
                "reason": "missing_idempotency_key",
            }
        store = getattr(server, "_operator_apply_idempotency_keys", {})
        if isinstance(store, dict) and idempotency_key in store:
            return None, {
                "planned_apply_index": planned_apply_index,
                "proposal_id": proposal_id,
                "candidate_index": candidate_index,
                "status": "blocked",
                "reason": "duplicate_idempotency_key",
            }
        if idempotency_key in seen_idempotency_keys:
            return None, {
                "planned_apply_index": planned_apply_index,
                "proposal_id": proposal_id,
                "candidate_index": candidate_index,
                "status": "blocked",
                "reason": "duplicate_idempotency_key_in_sequence",
            }
        seen_idempotency_keys.add(idempotency_key)

        raw_changes = candidate.get("normalized_changes")
        if not isinstance(raw_changes, list) or not raw_changes:
            return None, {
                "planned_apply_index": planned_apply_index,
                "proposal_id": proposal_id,
                "candidate_index": candidate_index,
                "status": "blocked",
                "reason": "missing_planned_changes",
            }
        changes = []
        for raw_change in raw_changes:
            change, blocker = dry_run_step_change(raw_change)
            if blocker:
                return None, {
                    "planned_apply_index": planned_apply_index,
                    "proposal_id": proposal_id,
                    "candidate_index": candidate_index,
                    "status": "blocked",
                    "reason": blocker,
                }
            changes.append(change)

        return {
            "planned_apply_index": planned_apply_index,
            "proposal_id": proposal_id,
            "candidate_index": candidate_index,
            "status": "executor_ready",
            "apply_request": {
                "type": "apply_operator_proposal",
                "proposal_id": proposal_id,
                "approved": True,
                "approval_id": str(apply_request.get("approval_id")).strip(),
                "idempotency_key": idempotency_key,
                "expected_review_contract_version": current_proposal.get(
                    "review_contract_version"
                ),
                "expected_review_content_digest": current_proposal.get(
                    "review_content_digest"
                ),
            },
            "current_proposal": current_proposal,
            "normalized_changes": changes,
        }, None

    def sequence_executor_payload(
        *,
        status,
        candidates,
        results=None,
        blocked=None,
        reason=None,
        operator_ack=False,
        execute=False,
        approved=False,
    ):
        results = results or []
        blocked = blocked or []
        applied = [item for item in results if item.get("status") == "applied"]
        apply_blocked = [item for item in results if item.get("status") == "blocked"]
        apply_unknown = [item for item in results if item.get("status") == "unknown"]
        attempted = [item for item in results if item.get("apply_attempted")]
        applied_count = len(applied)
        attempted_count = len(attempted)
        possible_console_send_count = sum(
            1
            for item in attempted
            if (
                not isinstance(item.get("apply_receipt"), dict)
                or item["apply_receipt"].get("no_console_send_proven") is not True
            )
        )
        no_console_send_proven_count = sum(
            1
            for item in attempted
            if isinstance(item.get("apply_receipt"), dict)
            and item["apply_receipt"].get("no_console_send_proven") is True
        )
        live_write = possible_console_send_count > 0
        delegated_apply_attempted = attempted_count > 0
        operator_mode = server.get_operator_mode_status()
        return {
            "type": "operator_proposal_apply_sequence_execution_result",
            "status": status,
            "success": status == "applied",
            "reason": reason,
            "generated_at": time.time(),
            "operator_mode": operator_mode,
            "summary": {
                "input_candidate_count": len(candidates),
                "attempted_count": attempted_count,
                "applied_count": applied_count,
                "blocked_count": len(blocked) + len(apply_blocked),
                "unknown_count": len(apply_unknown),
                "possible_console_send_count": possible_console_send_count,
                "no_console_send_proven_count": no_console_send_proven_count,
                "applied_proposal_ids": [
                    item["proposal_id"] for item in applied if item.get("proposal_id")
                ],
                "blocked_proposal_ids": [
                    item["proposal_id"]
                    for item in [*blocked, *apply_blocked]
                    if item.get("proposal_id")
                ],
                "operator_mode_allows_apply_now": bool(
                    operator_mode.get("capabilities", {}).get("can_apply_to_console")
                ),
                "operator_ack_received": operator_ack,
                "execute_requested": execute,
                "approved": approved,
                "requires_gate_ready": True,
                "uses_existing_apply_operator_proposal": True,
                "sequence_executor_called_apply_operator_proposal": (
                    delegated_apply_attempted
                ),
                "live_apply_entrypoint": "apply_operator_proposal",
                "live_write": live_write,
                "direct_live_write": False,
                "apply_attempted": delegated_apply_attempted,
                "queue_mutation": delegated_apply_attempted,
                "proposal_status_mutation": delegated_apply_attempted,
                "idempotency_reserved": delegated_apply_attempted,
                "idempotency_reserved_by_executor_directly": False,
                "idempotency_reservation_delegated_to_apply_operator_proposal": (
                    delegated_apply_attempted
                ),
                "manual_console_write_called_by_apply_operator_proposal": live_write,
                "manual_console_write_called_by_sequence_executor": False,
                "direct_manual_console_write_called": False,
                "mutates_console_state": live_write,
                "queries_console": False,
                "live_widening_allowed": False,
                "stopped_on_block": bool(apply_blocked or apply_unknown),
                "manual_review_required": bool(apply_unknown),
                "queue_freshness_checked": True,
            },
            "safety": {
                "sequence_executor": True,
                "execution_gate_payload_required": True,
                "ready_for_executor_required": True,
                "requires_execute_true": True,
                "requires_operator_ack": True,
                "requires_approved_true": True,
                "requires_gate_ready": True,
                "uses_existing_apply_operator_proposal": True,
                "live_apply_entrypoint": "apply_operator_proposal",
                "allowed_live_write_kinds": allowed_live_write_kinds_for_safety_surface(
                    operator_mode
                ),
                "live_write": live_write,
                "direct_live_write": False,
                "apply_attempted": delegated_apply_attempted,
                "queue_mutation_by_executor_directly": False,
                "proposal_status_mutation_by_executor_directly": False,
                "idempotency_reserved_by_executor_directly": False,
                "idempotency_reservation_delegated_to_apply_operator_proposal": (
                    delegated_apply_attempted
                ),
                "does_not_call_manual_console_write_directly": True,
                "manual_console_write_called_by_sequence_executor": False,
                "direct_manual_console_write_called": False,
                "manual_console_write_called_by_apply_operator_proposal": live_write,
                "mutates_console_state": live_write,
                "manual_review_required": bool(apply_unknown),
                "queries_console": False,
                "live_widening_allowed": False,
            },
            "execution_results": results,
            "blocked_results": blocked,
        }

    def sequence_apply_receipt_matches_ready_row(receipt, ready_row):
        if not isinstance(receipt, dict):
            return False
        current = ready_row.get("current_proposal")
        if not isinstance(current, dict):
            return False
        return bool(
            receipt.get("proposal_id") == ready_row.get("proposal_id")
            and receipt.get("review_contract_version")
            == current.get("review_contract_version")
            and receipt.get("review_content_digest")
            == current.get("review_content_digest")
        )

    def sequence_apply_receipt_proves_applied(receipt):
        if not isinstance(receipt, dict):
            return False
        audio_recheck_complete = bool(
            receipt.get("audio_recheck_required") is False
            or (
                receipt.get("audio_recheck_required") is True
                and receipt.get("audio_recheck_passed") is True
            )
        )
        return bool(
            receipt.get("status") == "applied"
            and receipt.get("proposal_status") == "applied"
            and receipt.get("apply_status") == "applied"
            and receipt.get("success") is True
            and receipt.get("no_console_send_proven") is False
            and receipt.get("physical_verified") is True
            and receipt.get("write_state_unknown") is False
            and receipt.get("rollback_state_unknown") is False
            and receipt.get("manual_review_required") is False
            and receipt.get("emergency_stop_activated") is False
            and receipt.get("emergency_stop_required") is False
            and receipt.get("wal_terminal") == "committed"
            and receipt.get("result_detail_available") is True
            and isinstance(receipt.get("result_digest"), str)
            and receipt.get("result_digest").startswith("sha256:")
            and audio_recheck_complete
        )

    def sequence_apply_receipt_proves_safe_block(receipt):
        if not isinstance(receipt, dict):
            return False
        no_send_block = bool(
            receipt.get("no_console_send_proven") is True
            and receipt.get("write_state_unknown") is False
            and receipt.get("rollback_state_unknown") is False
            and receipt.get("manual_review_required") is False
        )
        verified_rollback = bool(
            receipt.get("physical_state_restored") is True
            and receipt.get("rollback_attempted") is True
            and receipt.get("rollback_succeeded") is True
            and receipt.get("write_state_unknown") is False
            and receipt.get("rollback_state_unknown") is False
            and receipt.get("manual_review_required") is False
            and receipt.get("emergency_stop_required") is False
        )
        return bool(no_send_block or verified_rollback)

    async def send_sequence_executor_response(
        websocket,
        payload,
        *,
        child_receipts=(),
        expected_candidate_count=0,
    ):
        receipt = None
        try:
            receipt = _freeze_trusted_sequence_execution_receipt(
                payload,
                child_receipts=tuple(child_receipts),
                expected_candidate_count=expected_candidate_count,
            )
        except Exception:
            if child_receipts:
                try:
                    await activate_apply_emergency_stop(
                        "operator_proposal_sequence_response_projection_failed"
                    )
                except Exception:
                    pass
        await server.send_to_client(
            websocket,
            payload,
            trusted_receipt=receipt,
        )

    async def handle_execute_operator_proposal_apply_sequence(websocket, data):
        state_payload, state_error = resolve_sequence_stage_payload(
            websocket,
            data,
            expected_stage="execution_gate",
        )
        operator_ack = data.get("operator_ack") is True
        execute = data.get("execute") is True
        approved = data.get("approved") is True
        approval_id = str(data.get("approval_id") or "").strip()
        if state_error:
            await send_sequence_executor_response(
                websocket,
                sequence_executor_payload(
                    status="blocked",
                    reason=state_error,
                    candidates=[],
                    operator_ack=operator_ack,
                    execute=execute,
                    approved=approved,
                ),
            )
            return
        source_payload = state_payload if state_payload is not None else data
        raw_candidates = executor_ready_rows_from_data(source_payload)
        if raw_candidates is None:
            await send_sequence_executor_response(
                websocket,
                sequence_executor_payload(
                    status="blocked",
                    reason="missing_ready_for_executor",
                    candidates=[],
                    operator_ack=operator_ack,
                    execute=execute,
                    approved=approved,
                ),
            )
            return

        candidates = [
            candidate for candidate in raw_candidates if isinstance(candidate, dict)
        ]
        blocked = []
        invalid_count = len(raw_candidates) - len(candidates)
        if invalid_count:
            blocked.append({
                "planned_apply_index": None,
                "proposal_id": None,
                "candidate_index": None,
                "status": "blocked",
                "reason": "invalid_ready_for_executor_row",
                "count": invalid_count,
            })

        if not execute:
            blocked.append({
                "planned_apply_index": None,
                "proposal_id": None,
                "candidate_index": None,
                "status": "blocked",
                "reason": "execute_true_required",
            })
        if not operator_ack:
            blocked.append({
                "planned_apply_index": None,
                "proposal_id": None,
                "candidate_index": None,
                "status": "blocked",
                "reason": "operator_ack_required",
            })
        if not approved:
            blocked.append({
                "planned_apply_index": None,
                "proposal_id": None,
                "candidate_index": None,
                "status": "blocked",
                "reason": "approved_true_required",
            })
        operator_mode = server.get_operator_mode_status()
        if not operator_mode.get("capabilities", {}).get("can_apply_to_console"):
            blocked.append({
                "planned_apply_index": None,
                "proposal_id": None,
                "candidate_index": None,
                "status": "blocked",
                "reason": "operator_mode_blocks_live_write",
            })

        ready_rows = []
        seen_idempotency_keys = set()
        for candidate_index, candidate in enumerate(candidates):
            ready_row, blocker = sequence_executor_row_from_gate_candidate(
                candidate,
                candidate_index,
                seen_idempotency_keys,
            )
            if blocker:
                blocked.append(blocker)
            else:
                ready_rows.append(ready_row)

        if not blocked and ready_rows and approval_id:
            for ready_row in ready_rows:
                bound_approval_id = str(
                    ready_row["apply_request"].get("approval_id") or ""
                ).strip()
                if bound_approval_id != approval_id:
                    blocked.append({
                        "planned_apply_index": ready_row["planned_apply_index"],
                        "proposal_id": ready_row["proposal_id"],
                        "candidate_index": ready_row["candidate_index"],
                        "status": "blocked",
                        "reason": "approval_id_binding_mismatch",
                    })

        if blocked or not ready_rows:
            await send_sequence_executor_response(
                websocket,
                sequence_executor_payload(
                    status="blocked",
                    reason="sequence_executor_blocked",
                    candidates=candidates,
                    blocked=blocked or [{
                        "planned_apply_index": None,
                        "proposal_id": None,
                        "candidate_index": None,
                        "status": "blocked",
                        "reason": "no_ready_for_executor_rows",
                    }],
                    operator_ack=operator_ack,
                    execute=execute,
                    approved=approved,
                ),
                expected_candidate_count=len(candidates),
            )
            return

        results = []
        trusted_child_receipts = []
        for ready_row in ready_rows:
            apply_request = dict(ready_row["apply_request"])
            apply_request["approval_id"] = approval_id or apply_request["approval_id"]
            child_result = await handle_apply_operator_proposal(
                websocket,
                apply_request,
            )
            trusted_receipt = (
                child_result.trusted_receipt
                if isinstance(child_result, FinalizedApplyResult)
                else None
            )
            trusted_child_receipts.append(trusted_receipt)
            receipt = compact_trusted_apply_receipt_reference(trusted_receipt)
            receipt_matches = sequence_apply_receipt_matches_ready_row(
                receipt,
                ready_row,
            )
            row_applied = bool(
                receipt_matches
                and sequence_apply_receipt_proves_applied(receipt)
            )
            row_safe_blocked = bool(
                receipt_matches
                and not row_applied
                and sequence_apply_receipt_proves_safe_block(receipt)
            )
            row_status = (
                "applied"
                if row_applied
                else "blocked"
                if row_safe_blocked
                else "unknown"
            )
            result_row = {
                "planned_apply_index": ready_row["planned_apply_index"],
                "proposal_id": ready_row["proposal_id"],
                "candidate_index": ready_row["candidate_index"],
                "status": row_status,
                "apply_attempted": True,
                "apply_receipt": receipt if receipt_matches else None,
            }
            if not receipt_matches:
                result_row["reason"] = "apply_receipt_missing_or_identity_mismatch"
            elif not row_applied:
                result_row["reason"] = (
                    "apply_safely_blocked"
                    if row_safe_blocked
                    else "apply_terminal_state_unknown"
                )
            if row_status == "unknown":
                activated, teardown_complete = await activate_apply_emergency_stop(
                    "operator_proposal_sequence_receipt_untrusted:"
                    f"{ready_row['proposal_id']}"
                )
                result_row.update(
                    {
                        "manual_review_required": True,
                        "no_console_send_proven": None,
                        "executor_emergency_stop_activated": activated,
                        "executor_emergency_stop_teardown_complete": (
                            teardown_complete
                        ),
                        "executor_emergency_stop_required": bool(
                            not activated or teardown_complete is not True
                        ),
                    }
                )
            results.append(result_row)
            if not row_applied:
                break

        applied_count = sum(1 for item in results if item.get("status") == "applied")
        blocked_count = sum(1 for item in results if item.get("status") == "blocked")
        unknown_count = sum(1 for item in results if item.get("status") == "unknown")
        if applied_count == len(ready_rows) and blocked_count == 0 and unknown_count == 0:
            status = "applied"
            reason = None
        elif applied_count > 0:
            status = "partial"
            reason = (
                "sequence_executor_stopped_on_unknown"
                if unknown_count
                else "sequence_executor_stopped_on_block"
            )
        elif unknown_count:
            status = "unknown"
            reason = "sequence_executor_stopped_on_unknown"
        else:
            status = "blocked"
            reason = "sequence_executor_stopped_on_block"

        await send_sequence_executor_response(
            websocket,
            sequence_executor_payload(
                status=status,
                reason=reason,
                candidates=candidates,
                results=results,
                operator_ack=operator_ack,
                execute=execute,
                approved=approved,
            ),
            child_receipts=trusted_child_receipts,
            expected_candidate_count=len(ready_rows),
        )

    def sequence_execution_payload_from_data(data):
        raw_payload = data.get("execution_result_payload")
        if isinstance(raw_payload, dict):
            return raw_payload
        if isinstance(data.get("execution_results"), list):
            return data
        return None

    def sequence_execution_audit_blocker(reason, *, proposal_id=None, row_index=None):
        return {
            "proposal_id": proposal_id,
            "execution_result_index": row_index,
            "status": "blocked",
            "reason": reason,
        }

    def audit_sequence_execution_envelope(payload):
        blockers = []
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
        if payload.get("type") != "operator_proposal_apply_sequence_execution_result":
            blockers.append(
                sequence_execution_audit_blocker("execution_payload_type_mismatch")
            )
        if payload.get("status") != "applied":
            blockers.append(
                sequence_execution_audit_blocker("execution_payload_not_applied")
            )
        if summary.get("uses_existing_apply_operator_proposal") is not True:
            blockers.append(
                sequence_execution_audit_blocker(
                    "summary_missing_apply_operator_proposal_delegation"
                )
            )
        if safety.get("uses_existing_apply_operator_proposal") is not True:
            blockers.append(
                sequence_execution_audit_blocker(
                    "safety_missing_apply_operator_proposal_delegation"
                )
            )
        if summary.get("live_apply_entrypoint") != "apply_operator_proposal":
            blockers.append(
                sequence_execution_audit_blocker("summary_live_apply_entrypoint_mismatch")
            )
        if safety.get("live_apply_entrypoint") != "apply_operator_proposal":
            blockers.append(
                sequence_execution_audit_blocker("safety_live_apply_entrypoint_mismatch")
            )
        for field in (
            "direct_live_write",
            "direct_manual_console_write_called",
            "manual_console_write_called_by_sequence_executor",
            "idempotency_reserved_by_executor_directly",
            "live_widening_allowed",
            "queries_console",
        ):
            if summary.get(field) not in (False, None):
                blockers.append(sequence_execution_audit_blocker(f"summary_{field}_true"))
            if safety.get(field) not in (False, None):
                blockers.append(sequence_execution_audit_blocker(f"safety_{field}_true"))
        if safety.get("does_not_call_manual_console_write_directly") is not True:
            blockers.append(
                sequence_execution_audit_blocker(
                    "safety_missing_direct_manual_write_denial"
                )
            )
        allowed = set(safety.get("allowed_live_write_kinds") or [])
        active_allowed = live_write_kind_set_for_safety_surface()
        if not allowed or allowed - active_allowed:
            blockers.append(
                sequence_execution_audit_blocker(
                    "safety_allowed_live_write_kinds_widened"
                )
            )
        return blockers

    def audit_sequence_execution_row(row, row_index):
        proposal_id = str(row.get("proposal_id") or "").strip() or None
        blockers = []
        if row.get("status") != "applied":
            blockers.append(
                sequence_execution_audit_blocker(
                    "execution_row_not_applied",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
        if row.get("apply_attempted") is not True:
            blockers.append(
                sequence_execution_audit_blocker(
                    "execution_row_apply_not_attempted",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )

        receipt = row.get("apply_receipt")
        if not isinstance(receipt, dict):
            blockers.append(
                sequence_execution_audit_blocker(
                    "execution_row_missing_apply_receipt",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
            receipt = {}
        if receipt.get("proposal_id") != proposal_id:
            blockers.append(
                sequence_execution_audit_blocker(
                    "execution_row_apply_receipt_identity_mismatch",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
        if receipt.get("status") != "applied":
            blockers.append(
                sequence_execution_audit_blocker(
                    "execution_row_apply_receipt_not_applied",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
        if receipt.get("proposal_status") != "applied":
            blockers.append(
                sequence_execution_audit_blocker(
                    "execution_row_proposal_status_not_applied",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
        if receipt.get("apply_status") != "applied" or receipt.get("success") is not True:
            blockers.append(
                sequence_execution_audit_blocker(
                    "execution_row_apply_status_not_applied",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
        if receipt.get("no_console_send_proven") is not False:
            blockers.append(
                sequence_execution_audit_blocker(
                    "execution_row_console_send_truth_missing",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
        value_type = str(receipt.get("value_type") or "").strip().lower()
        if value_type not in live_write_kind_set_for_safety_surface():
            blockers.append(
                sequence_execution_audit_blocker(
                    "execution_row_value_type_not_live_allowed",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
        if receipt.get("physical_verified") is not True:
            blockers.append(
                sequence_execution_audit_blocker(
                    "execution_row_physical_state_not_verified",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
        if receipt.get("wal_terminal") != "committed":
            blockers.append(
                sequence_execution_audit_blocker(
                    "execution_row_wal_not_committed",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
        if (
            receipt.get("write_state_unknown") is not False
            or receipt.get("rollback_state_unknown") is not False
            or receipt.get("manual_review_required") is not False
            or receipt.get("emergency_stop_activated") is not False
            or receipt.get("emergency_stop_required") is not False
        ):
            blockers.append(
                sequence_execution_audit_blocker(
                    "execution_row_terminal_safety_unknown",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
        result_digest = receipt.get("result_digest")
        if (
            receipt.get("result_detail_available") is not True
            or not isinstance(result_digest, str)
            or not result_digest.startswith("sha256:")
        ):
            blockers.append(
                sequence_execution_audit_blocker(
                    "execution_row_result_detail_unavailable",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
        if not (
            receipt.get("audio_recheck_required") is False
            or (
                receipt.get("audio_recheck_required") is True
                and receipt.get("audio_recheck_passed") is True
            )
        ):
            blockers.append(
                sequence_execution_audit_blocker(
                    "execution_row_audio_recheck_incomplete",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
        idempotency_key_digest = receipt.get("idempotency_key_digest")
        if (
            not isinstance(idempotency_key_digest, str)
            or not idempotency_key_digest.startswith("sha256:")
        ):
            blockers.append(
                sequence_execution_audit_blocker(
                    "execution_row_idempotency_digest_missing",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
        if blockers:
            return None, blockers

        return {
            "proposal_id": proposal_id,
            "execution_result_index": row_index,
            "status": "passed",
            "value_type": value_type,
            "channel": receipt.get("channel"),
            "physical_verified": True,
            "wal_terminal": "committed",
            "result_digest": result_digest,
            "idempotency_key_digest": idempotency_key_digest,
            "live_apply_entrypoint": "apply_operator_proposal",
        }, []

    def sequence_execution_audit_payload(*, status, source_payload, rows, blockers):
        rows = rows or []
        blockers = blockers or []
        passed_ids = [row["proposal_id"] for row in rows if row.get("proposal_id")]
        blocked_ids = [
            item["proposal_id"] for item in blockers if item.get("proposal_id")
        ]
        return {
            "type": "operator_proposal_apply_sequence_execution_audit",
            "status": status,
            "success": status == "passed",
            "reason": None if status == "passed" else "sequence_execution_audit_blocked",
            "generated_at": time.time(),
            "source_execution_status": source_payload.get("status"),
            "summary": {
                "input_execution_result_count": len(
                    source_payload.get("execution_results") or []
                ),
                "audited_count": len(rows),
                "passed_count": len(rows),
                "blocked_count": len(blockers),
                "passed_proposal_ids": passed_ids,
                "blocked_proposal_ids": blocked_ids,
                "audit_only": True,
                "execution_allowed_by_audit": False,
                "live_write": False,
                "apply_attempted": False,
                "queue_mutation": False,
                "proposal_status_mutation": False,
                "idempotency_reserved": False,
                "manual_console_write_called": False,
                "mutates_console_state": False,
                "queries_console": False,
                "live_widening_allowed": False,
                "live_authority_granted": False,
            },
            "safety": {
                "audit_only": True,
                "execution_result_payload_required": True,
                "does_not_call_apply_operator_proposal": True,
                "does_not_call_manual_console_write": True,
                "live_apply_entrypoint_required": "apply_operator_proposal",
                "allowed_live_write_kinds": allowed_live_write_kinds_for_safety_surface(),
                "execution_allowed_by_audit": False,
                "live_write": False,
                "apply_attempted": False,
                "queue_mutation": False,
                "proposal_status_mutation": False,
                "idempotency_reserved": False,
                "manual_console_write_called": False,
                "mutates_console_state": False,
                "queries_console": False,
                "live_widening_allowed": False,
                "live_authority_granted": False,
            },
            "audit_rows": rows,
            "blocked_results": blockers,
        }

    async def handle_audit_operator_proposal_apply_sequence_execution(websocket, data):
        source_payload = sequence_execution_payload_from_data(data)
        if not isinstance(source_payload, dict):
            await server.send_to_client(
                websocket,
                sequence_execution_audit_payload(
                    status="blocked",
                    source_payload={},
                    rows=[],
                    blockers=[
                        sequence_execution_audit_blocker(
                            "missing_execution_result_payload"
                        )
                    ],
                ),
            )
            return

        raw_rows = source_payload.get("execution_results")
        if not isinstance(raw_rows, list):
            await server.send_to_client(
                websocket,
                sequence_execution_audit_payload(
                    status="blocked",
                    source_payload=source_payload,
                    rows=[],
                    blockers=[
                        sequence_execution_audit_blocker(
                            "missing_execution_result_rows"
                        )
                    ],
                ),
            )
            return

        blockers = audit_sequence_execution_envelope(source_payload)
        audit_rows = []
        for row_index, row in enumerate(raw_rows):
            if not isinstance(row, dict):
                blockers.append(
                    sequence_execution_audit_blocker(
                        "invalid_execution_result_row",
                        row_index=row_index,
                    )
                )
                continue
            audit_row, row_blockers = audit_sequence_execution_row(row, row_index)
            blockers.extend(row_blockers)
            if audit_row:
                audit_rows.append(audit_row)

        await server.send_to_client(
            websocket,
            sequence_execution_audit_payload(
                status="passed" if audit_rows and not blockers else "blocked",
                source_payload=source_payload,
                rows=audit_rows,
                blockers=blockers,
            ),
        )

    def sequence_execution_audit_payload_from_data(data):
        raw_payload = data.get("execution_audit_payload") or data.get("audit_payload")
        if isinstance(raw_payload, dict):
            return raw_payload
        if isinstance(data.get("audit_rows"), list):
            return data
        return None

    def audit_policy_gate_blocker(reason, *, proposal_id=None, row_index=None):
        return {
            "proposal_id": proposal_id,
            "audit_row_index": row_index,
            "status": "blocked",
            "reason": reason,
        }

    def audit_policy_gate_envelope_blockers(payload):
        blockers = []
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
        if payload.get("type") != "operator_proposal_apply_sequence_execution_audit":
            blockers.append(audit_policy_gate_blocker("audit_payload_type_mismatch"))
        if payload.get("status") != "passed":
            blockers.append(audit_policy_gate_blocker("audit_payload_not_passed"))
        if summary.get("audit_only") is not True:
            blockers.append(audit_policy_gate_blocker("summary_audit_only_not_true"))
        if safety.get("audit_only") is not True:
            blockers.append(audit_policy_gate_blocker("safety_audit_only_not_true"))
        if safety.get("does_not_call_apply_operator_proposal") is not True:
            blockers.append(
                audit_policy_gate_blocker(
                    "safety_missing_apply_operator_proposal_denial"
                )
            )
        if safety.get("does_not_call_manual_console_write") is not True:
            blockers.append(
                audit_policy_gate_blocker("safety_missing_manual_console_write_denial")
            )
        for field in (
            "execution_allowed_by_audit",
            "live_write",
            "apply_attempted",
            "queue_mutation",
            "proposal_status_mutation",
            "idempotency_reserved",
            "manual_console_write_called",
            "mutates_console_state",
            "queries_console",
            "live_widening_allowed",
            "live_authority_granted",
        ):
            if summary.get(field) not in (False, None):
                blockers.append(audit_policy_gate_blocker(f"summary_{field}_true"))
            if safety.get(field) not in (False, None):
                blockers.append(audit_policy_gate_blocker(f"safety_{field}_true"))
        allowed = set(safety.get("allowed_live_write_kinds") or [])
        active_allowed = live_write_kind_set_for_safety_surface()
        if not allowed or allowed - active_allowed:
            blockers.append(
                audit_policy_gate_blocker("safety_allowed_live_write_kinds_widened")
            )
        blocked_results = payload.get("blocked_results")
        if isinstance(blocked_results, list) and blocked_results:
            blockers.append(audit_policy_gate_blocker("audit_payload_has_blocked_results"))
        return blockers

    def audit_policy_gate_row(row, row_index):
        proposal_id = str(row.get("proposal_id") or "").strip() or None
        blockers = []
        if row.get("status") != "passed":
            blockers.append(
                audit_policy_gate_blocker(
                    "audit_row_not_passed",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
        value_type = str(row.get("value_type") or "").strip().lower()
        if value_type not in live_write_kind_set_for_safety_surface():
            blockers.append(
                audit_policy_gate_blocker(
                    "audit_row_value_type_not_policy_allowed",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
        idempotency_key = str(row.get("idempotency_key") or "").strip()
        if not idempotency_key:
            blockers.append(
                audit_policy_gate_blocker(
                    "audit_row_missing_idempotency_key",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
        readback_status = row.get("readback_status")
        if readback_status not in VERIFIED_MANUAL_READBACK_STATUSES:
            blockers.append(
                audit_policy_gate_blocker(
                    "audit_row_readback_not_verified",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
        proof_status = str(row.get("proof_status") or "").strip()
        if proof_status != "verified":
            blockers.append(
                audit_policy_gate_blocker(
                    "audit_row_proof_not_verified",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
        if row.get("rollback_available") is not True:
            blockers.append(
                audit_policy_gate_blocker(
                    "audit_row_rollback_unavailable",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
        effective_value = coerce_float(row.get("effective_value"))
        if value_type == "fader" and effective_value is not None and effective_value > 0.0:
            blockers.append(
                audit_policy_gate_blocker(
                    "audit_row_fader_ceiling_exceeded",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
        if blockers:
            return None, blockers

        return {
            "proposal_id": proposal_id,
            "audit_row_index": row_index,
            "status": "policy_evidence_ready",
            "eligible_for_supervised_sequence_policy": True,
            "execution_allowed_by_policy_gate": False,
            "policy_mutation_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "value_type": value_type,
            "channel": row.get("channel"),
            "effective_value": effective_value,
            "idempotency_key": idempotency_key,
            "readback_status": readback_status,
            "proof_status": proof_status,
            "rollback_available": True,
            "live_apply_entrypoint": "apply_operator_proposal",
        }, []

    def audit_policy_gate_payload(*, status, source_payload, candidates, blockers):
        candidates = candidates or []
        blockers = blockers or []
        return {
            "type": "operator_proposal_apply_sequence_audit_policy_gate",
            "status": status,
            "success": status == "ready",
            "reason": None if status == "ready" else "sequence_audit_policy_gate_blocked",
            "generated_at": time.time(),
            "source_audit_status": source_payload.get("status"),
            "summary": {
                "input_audit_row_count": len(source_payload.get("audit_rows") or []),
                "policy_evidence_ready_count": len(candidates),
                "blocked_count": len(blockers),
                "policy_evidence_ready_proposal_ids": [
                    item["proposal_id"]
                    for item in candidates
                    if item.get("proposal_id")
                ],
                "blocked_proposal_ids": [
                    item["proposal_id"] for item in blockers if item.get("proposal_id")
                ],
                "policy_gate_only": True,
                "audit_payload_required": True,
                "execution_allowed_by_policy_gate": False,
                "execution_allowed_by_audit": False,
                "live_write": False,
                "apply_attempted": False,
                "queue_mutation": False,
                "proposal_status_mutation": False,
                "idempotency_reserved": False,
                "manual_console_write_called": False,
                "mutates_console_state": False,
                "queries_console": False,
                "policy_mutation": False,
                "runtime_memory_mutation": False,
                "model_mutation": False,
                "live_widening_allowed": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
            },
            "safety": {
                "policy_gate_only": True,
                "audit_payload_required": True,
                "does_not_call_apply_operator_proposal": True,
                "does_not_call_manual_console_write": True,
                "allowed_live_write_kinds": allowed_live_write_kinds_for_safety_surface(),
                "execution_allowed_by_policy_gate": False,
                "execution_allowed_by_audit": False,
                "live_write": False,
                "apply_attempted": False,
                "queue_mutation": False,
                "proposal_status_mutation": False,
                "idempotency_reserved": False,
                "manual_console_write_called": False,
                "mutates_console_state": False,
                "queries_console": False,
                "policy_mutation": False,
                "runtime_memory_mutation": False,
                "model_mutation": False,
                "live_widening_allowed": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
            },
            "policy_evidence": candidates,
            "blocked_results": blockers,
        }

    async def handle_gate_operator_proposal_apply_sequence_audit_policy(websocket, data):
        source_payload = sequence_execution_audit_payload_from_data(data)
        if not isinstance(source_payload, dict):
            await server.send_to_client(
                websocket,
                audit_policy_gate_payload(
                    status="blocked",
                    source_payload={},
                    candidates=[],
                    blockers=[
                        audit_policy_gate_blocker("missing_execution_audit_payload")
                    ],
                ),
            )
            return

        raw_rows = source_payload.get("audit_rows")
        if not isinstance(raw_rows, list):
            await server.send_to_client(
                websocket,
                audit_policy_gate_payload(
                    status="blocked",
                    source_payload=source_payload,
                    candidates=[],
                    blockers=[audit_policy_gate_blocker("missing_audit_rows")],
                ),
            )
            return

        blockers = audit_policy_gate_envelope_blockers(source_payload)
        candidates = []
        for row_index, row in enumerate(raw_rows):
            if not isinstance(row, dict):
                blockers.append(
                    audit_policy_gate_blocker(
                        "invalid_audit_row",
                        row_index=row_index,
                    )
                )
                continue
            candidate, row_blockers = audit_policy_gate_row(row, row_index)
            blockers.extend(row_blockers)
            if candidate:
                candidates.append(candidate)

        await server.send_to_client(
            websocket,
            audit_policy_gate_payload(
                status="ready" if candidates and not blockers else "blocked",
                source_payload=source_payload,
                candidates=candidates,
                blockers=blockers,
            ),
        )

    def sequence_policy_gate_payload_from_data(data):
        raw_payload = data.get("policy_gate_payload")
        if isinstance(raw_payload, dict):
            return raw_payload
        if isinstance(data.get("policy_evidence"), list):
            return data
        return None

    def policy_memory_draft_blocker(reason, *, proposal_id=None, row_index=None):
        return {
            "proposal_id": proposal_id,
            "policy_evidence_index": row_index,
            "status": "blocked",
            "reason": reason,
        }

    def policy_memory_draft_envelope_blockers(payload):
        blockers = []
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
        if payload.get("type") != "operator_proposal_apply_sequence_audit_policy_gate":
            blockers.append(policy_memory_draft_blocker("policy_gate_payload_type_mismatch"))
        if payload.get("status") != "ready":
            blockers.append(policy_memory_draft_blocker("policy_gate_payload_not_ready"))
        if summary.get("policy_gate_only") is not True:
            blockers.append(policy_memory_draft_blocker("summary_policy_gate_only_not_true"))
        if safety.get("policy_gate_only") is not True:
            blockers.append(policy_memory_draft_blocker("safety_policy_gate_only_not_true"))
        if safety.get("does_not_call_apply_operator_proposal") is not True:
            blockers.append(
                policy_memory_draft_blocker(
                    "safety_missing_apply_operator_proposal_denial"
                )
            )
        if safety.get("does_not_call_manual_console_write") is not True:
            blockers.append(
                policy_memory_draft_blocker("safety_missing_manual_console_write_denial")
            )
        for field in (
            "execution_allowed_by_policy_gate",
            "execution_allowed_by_audit",
            "live_write",
            "apply_attempted",
            "queue_mutation",
            "proposal_status_mutation",
            "idempotency_reserved",
            "manual_console_write_called",
            "mutates_console_state",
            "queries_console",
            "policy_mutation",
            "runtime_memory_mutation",
            "model_mutation",
            "live_widening_allowed",
            "live_authority_granted",
            "full_autonomous_ready",
        ):
            if summary.get(field) not in (False, None):
                blockers.append(policy_memory_draft_blocker(f"summary_{field}_true"))
            if safety.get(field) not in (False, None):
                blockers.append(policy_memory_draft_blocker(f"safety_{field}_true"))
        allowed = set(safety.get("allowed_live_write_kinds") or [])
        if not allowed or allowed - live_write_kind_set_for_safety_surface():
            blockers.append(
                policy_memory_draft_blocker("safety_allowed_live_write_kinds_widened")
            )
        blocked_results = payload.get("blocked_results")
        if isinstance(blocked_results, list) and blocked_results:
            blockers.append(policy_memory_draft_blocker("policy_gate_has_blocked_results"))
        return blockers

    def policy_memory_draft_record(row, row_index):
        proposal_id = str(row.get("proposal_id") or "").strip() or None
        blockers = []
        if row.get("status") != "policy_evidence_ready":
            blockers.append(
                policy_memory_draft_blocker(
                    "policy_evidence_not_ready",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
        if row.get("eligible_for_supervised_sequence_policy") is not True:
            blockers.append(
                policy_memory_draft_blocker(
                    "policy_evidence_not_supervised_sequence_eligible",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
        for field in (
            "execution_allowed_by_policy_gate",
            "policy_mutation_allowed",
            "live_authority_granted",
            "full_autonomous_ready",
        ):
            if row.get(field) not in (False, None):
                blockers.append(
                    policy_memory_draft_blocker(
                        f"policy_evidence_{field}_true",
                        proposal_id=proposal_id,
                        row_index=row_index,
                    )
                )
        value_type = str(row.get("value_type") or "").strip().lower()
        if value_type not in live_write_kind_set_for_safety_surface():
            blockers.append(
                policy_memory_draft_blocker(
                    "policy_evidence_value_type_not_memory_allowed",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
        idempotency_key = str(row.get("idempotency_key") or "").strip()
        if not idempotency_key:
            blockers.append(
                policy_memory_draft_blocker(
                    "policy_evidence_missing_idempotency_key",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
        if row.get("readback_status") not in VERIFIED_MANUAL_READBACK_STATUSES:
            blockers.append(
                policy_memory_draft_blocker(
                    "policy_evidence_readback_not_verified",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
        if str(row.get("proof_status") or "").strip() != "verified":
            blockers.append(
                policy_memory_draft_blocker(
                    "policy_evidence_proof_not_verified",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
        if row.get("rollback_available") is not True:
            blockers.append(
                policy_memory_draft_blocker(
                    "policy_evidence_rollback_unavailable",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
        if row.get("live_apply_entrypoint") != "apply_operator_proposal":
            blockers.append(
                policy_memory_draft_blocker(
                    "policy_evidence_live_apply_entrypoint_mismatch",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
        effective_value = coerce_float(row.get("effective_value"))
        if value_type == "fader" and effective_value is not None and effective_value > 0.0:
            blockers.append(
                policy_memory_draft_blocker(
                    "policy_evidence_fader_ceiling_exceeded",
                    proposal_id=proposal_id,
                    row_index=row_index,
                )
            )
        if blockers:
            return None, blockers

        return {
            "draft_id": f"operator-sequence-policy-memory:{proposal_id}:{idempotency_key}",
            "proposal_id": proposal_id,
            "policy_evidence_index": row_index,
            "status": "memory_draft_ready",
            "learning_scope": "supervised_fader_gain_sequence_execution",
            "target_memory_path": ".ai/memory/project.md",
            "target_section": "Project Memory Lessons",
            "append_only": True,
            "memory_write_allowed_by_draft": False,
            "append_only_memory_write_allowed_by_draft": False,
            "requires_separate_memory_writer": True,
            "operator_ack_required_for_write": True,
            "memory_mutation": False,
            "runtime_memory_mutation": False,
            "policy_mutation": False,
            "model_mutation": False,
            "queue_mutation": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "evidence": {
                "value_type": value_type,
                "channel": row.get("channel"),
                "effective_value": effective_value,
                "idempotency_key": idempotency_key,
                "readback_status": row.get("readback_status"),
                "proof_status": row.get("proof_status"),
                "rollback_available": True,
                "live_apply_entrypoint": "apply_operator_proposal",
            },
            "draft_text": (
                "Supervised fader/gain sequence execution was verified from "
                f"`{proposal_id}` with `{value_type}` evidence, idempotency "
                f"`{idempotency_key}`, readback `{row.get('readback_status')}`, "
                "and rollback available. Future policy or autonomous use still "
                "requires separate audited gates and safety limits."
            ),
        }, []

    def policy_memory_draft_payload(*, status, source_payload, drafts, blockers):
        drafts = drafts or []
        blockers = blockers or []
        return {
            "type": "operator_proposal_apply_sequence_policy_memory_draft",
            "status": status,
            "success": status == "ready",
            "reason": None if status == "ready" else "sequence_policy_memory_draft_blocked",
            "generated_at": time.time(),
            "source_policy_gate_status": source_payload.get("status"),
            "summary": {
                "input_policy_evidence_count": len(source_payload.get("policy_evidence") or []),
                "memory_draft_count": len(drafts),
                "blocked_count": len(blockers),
                "memory_draft_proposal_ids": [
                    item["proposal_id"] for item in drafts if item.get("proposal_id")
                ],
                "blocked_proposal_ids": [
                    item["proposal_id"] for item in blockers if item.get("proposal_id")
                ],
                "memory_draft_only": True,
                "policy_gate_payload_required": True,
                "append_only_memory_write_allowed_by_draft": False,
                "requires_separate_memory_writer": True,
                "execution_allowed_by_draft": False,
                "live_write": False,
                "apply_attempted": False,
                "queue_mutation": False,
                "proposal_status_mutation": False,
                "idempotency_reserved": False,
                "manual_console_write_called": False,
                "mutates_console_state": False,
                "queries_console": False,
                "policy_mutation": False,
                "memory_mutation": False,
                "runtime_memory_mutation": False,
                "model_mutation": False,
                "live_widening_allowed": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
            },
            "safety": {
                "memory_draft_only": True,
                "policy_gate_payload_required": True,
                "does_not_call_apply_operator_proposal": True,
                "does_not_call_manual_console_write": True,
                "allowed_live_write_kinds": allowed_live_write_kinds_for_safety_surface(),
                "append_only_memory_write_allowed_by_draft": False,
                "requires_separate_memory_writer": True,
                "execution_allowed_by_draft": False,
                "live_write": False,
                "apply_attempted": False,
                "queue_mutation": False,
                "proposal_status_mutation": False,
                "idempotency_reserved": False,
                "manual_console_write_called": False,
                "mutates_console_state": False,
                "queries_console": False,
                "policy_mutation": False,
                "memory_mutation": False,
                "runtime_memory_mutation": False,
                "model_mutation": False,
                "live_widening_allowed": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
            },
            "memory_draft_records": drafts,
            "blocked_results": blockers,
        }

    async def handle_draft_operator_proposal_apply_sequence_policy_memory(websocket, data):
        source_payload = sequence_policy_gate_payload_from_data(data)
        if not isinstance(source_payload, dict):
            await server.send_to_client(
                websocket,
                policy_memory_draft_payload(
                    status="blocked",
                    source_payload={},
                    drafts=[],
                    blockers=[
                        policy_memory_draft_blocker("missing_policy_gate_payload")
                    ],
                ),
            )
            return

        raw_rows = source_payload.get("policy_evidence")
        if not isinstance(raw_rows, list):
            await server.send_to_client(
                websocket,
                policy_memory_draft_payload(
                    status="blocked",
                    source_payload=source_payload,
                    drafts=[],
                    blockers=[policy_memory_draft_blocker("missing_policy_evidence")],
                ),
            )
            return

        blockers = policy_memory_draft_envelope_blockers(source_payload)
        drafts = []
        for row_index, row in enumerate(raw_rows):
            if not isinstance(row, dict):
                blockers.append(
                    policy_memory_draft_blocker(
                        "invalid_policy_evidence",
                        row_index=row_index,
                    )
                )
                continue
            draft, row_blockers = policy_memory_draft_record(row, row_index)
            blockers.extend(row_blockers)
            if draft:
                drafts.append(draft)

        await server.send_to_client(
            websocket,
            policy_memory_draft_payload(
                status="ready" if drafts and not blockers else "blocked",
                source_payload=source_payload,
                drafts=drafts,
                blockers=blockers,
            ),
        )

    def sequence_policy_memory_draft_payload_from_data(data):
        raw_payload = data.get("memory_draft_payload") or data.get("draft_payload")
        if isinstance(raw_payload, dict):
            return raw_payload
        if isinstance(data.get("memory_draft_records"), list):
            return data
        return None

    def policy_memory_writer_blocker(reason, *, draft_id=None, row_index=None):
        return {
            "draft_id": draft_id,
            "memory_draft_index": row_index,
            "status": "blocked",
            "reason": reason,
        }

    def _sequence_policy_memory_target_allowed(path_value):
        if path_value != ".ai/memory/project.md":
            return False
        path = Path(str(path_value))
        return not path.is_absolute() and ".." not in path.parts

    def policy_memory_writer_envelope_blockers(payload):
        blockers = []
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
        if payload.get("type") != "operator_proposal_apply_sequence_policy_memory_draft":
            blockers.append(policy_memory_writer_blocker("memory_draft_payload_type_mismatch"))
        if payload.get("status") != "ready":
            blockers.append(policy_memory_writer_blocker("memory_draft_payload_not_ready"))
        if summary.get("memory_draft_only") is not True:
            blockers.append(policy_memory_writer_blocker("summary_memory_draft_only_not_true"))
        if safety.get("memory_draft_only") is not True:
            blockers.append(policy_memory_writer_blocker("safety_memory_draft_only_not_true"))
        if safety.get("does_not_call_apply_operator_proposal") is not True:
            blockers.append(
                policy_memory_writer_blocker(
                    "safety_missing_apply_operator_proposal_denial"
                )
            )
        if safety.get("does_not_call_manual_console_write") is not True:
            blockers.append(
                policy_memory_writer_blocker("safety_missing_manual_console_write_denial")
            )
        for field in (
            "append_only_memory_write_allowed_by_draft",
            "execution_allowed_by_draft",
            "live_write",
            "apply_attempted",
            "queue_mutation",
            "proposal_status_mutation",
            "idempotency_reserved",
            "manual_console_write_called",
            "mutates_console_state",
            "queries_console",
            "policy_mutation",
            "memory_mutation",
            "runtime_memory_mutation",
            "model_mutation",
            "live_widening_allowed",
            "live_authority_granted",
            "full_autonomous_ready",
        ):
            if summary.get(field) not in (False, None):
                blockers.append(policy_memory_writer_blocker(f"summary_{field}_true"))
            if safety.get(field) not in (False, None):
                blockers.append(policy_memory_writer_blocker(f"safety_{field}_true"))
        allowed = set(safety.get("allowed_live_write_kinds") or [])
        if not allowed or allowed - live_write_kind_set_for_safety_surface():
            blockers.append(
                policy_memory_writer_blocker("safety_allowed_live_write_kinds_widened")
            )
        blocked_results = payload.get("blocked_results")
        if isinstance(blocked_results, list) and blocked_results:
            blockers.append(policy_memory_writer_blocker("memory_draft_has_blocked_results"))
        return blockers

    def policy_memory_writer_record(record, row_index):
        draft_id = str(record.get("draft_id") or "").strip() or None
        blockers = []
        if record.get("status") != "memory_draft_ready":
            blockers.append(
                policy_memory_writer_blocker(
                    "memory_draft_record_not_ready",
                    draft_id=draft_id,
                    row_index=row_index,
                )
            )
        if record.get("append_only") is not True:
            blockers.append(
                policy_memory_writer_blocker(
                    "memory_draft_record_append_only_not_true",
                    draft_id=draft_id,
                    row_index=row_index,
                )
            )
        if not _sequence_policy_memory_target_allowed(record.get("target_memory_path")):
            blockers.append(
                policy_memory_writer_blocker(
                    "memory_draft_record_target_path_not_allowed",
                    draft_id=draft_id,
                    row_index=row_index,
                )
            )
        if not isinstance(record.get("draft_text"), str) or not record.get("draft_text").strip():
            blockers.append(
                policy_memory_writer_blocker(
                    "memory_draft_record_draft_text_missing",
                    draft_id=draft_id,
                    row_index=row_index,
                )
            )
        for field in (
            "memory_write_allowed_by_draft",
            "append_only_memory_write_allowed_by_draft",
            "memory_mutation",
            "runtime_memory_mutation",
            "policy_mutation",
            "model_mutation",
            "queue_mutation",
            "live_authority_granted",
            "full_autonomous_ready",
        ):
            if record.get(field) not in (False, None):
                blockers.append(
                    policy_memory_writer_blocker(
                        f"memory_draft_record_{field}_true",
                        draft_id=draft_id,
                        row_index=row_index,
                    )
                )
        evidence = record.get("evidence") if isinstance(record.get("evidence"), dict) else {}
        value_type = str(evidence.get("value_type") or "").strip().lower()
        if value_type not in live_write_kind_set_for_safety_surface():
            blockers.append(
                policy_memory_writer_blocker(
                    "memory_draft_record_value_type_not_allowed",
                    draft_id=draft_id,
                    row_index=row_index,
                )
            )
        if evidence.get("live_apply_entrypoint") != "apply_operator_proposal":
            blockers.append(
                policy_memory_writer_blocker(
                    "memory_draft_record_live_apply_entrypoint_mismatch",
                    draft_id=draft_id,
                    row_index=row_index,
                )
            )
        if not str(evidence.get("idempotency_key") or "").strip():
            blockers.append(
                policy_memory_writer_blocker(
                    "memory_draft_record_idempotency_missing",
                    draft_id=draft_id,
                    row_index=row_index,
                )
            )
        if evidence.get("readback_status") not in VERIFIED_MANUAL_READBACK_STATUSES:
            blockers.append(
                policy_memory_writer_blocker(
                    "memory_draft_record_readback_not_verified",
                    draft_id=draft_id,
                    row_index=row_index,
                )
            )
        if str(evidence.get("proof_status") or "").strip() != "verified":
            blockers.append(
                policy_memory_writer_blocker(
                    "memory_draft_record_proof_not_verified",
                    draft_id=draft_id,
                    row_index=row_index,
                )
            )
        if evidence.get("rollback_available") is not True:
            blockers.append(
                policy_memory_writer_blocker(
                    "memory_draft_record_rollback_unavailable",
                    draft_id=draft_id,
                    row_index=row_index,
                )
            )
        return blockers

    def render_sequence_policy_memory_block(record):
        evidence = record.get("evidence") if isinstance(record.get("evidence"), dict) else {}
        return "\n".join(
            [
                "---",
                "",
                "### Supervised sequence policy memory",
                "",
                f"- Draft ID: `{record.get('draft_id')}`",
                f"- Proposal ID: `{record.get('proposal_id')}`",
                f"- Learning scope: `{record.get('learning_scope')}`",
                f"- Value type: `{evidence.get('value_type')}`",
                f"- Channel: `{evidence.get('channel')}`",
                f"- Effective value: `{evidence.get('effective_value')}`",
                f"- Idempotency key: `{evidence.get('idempotency_key')}`",
                f"- Readback status: `{evidence.get('readback_status')}`",
                f"- Proof status: `{evidence.get('proof_status')}`",
                "- Runtime authority: no live, queue, model, or policy mutation",
                "",
                str(record.get("draft_text") or "").strip(),
                "",
            ]
        )

    def append_sequence_policy_memory_records(root, records):
        results = []
        root_path = Path(root or ".").resolve()
        for record in records:
            target = root_path / str(record["target_memory_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            current = (
                target.read_text(encoding="utf-8")
                if target.exists()
                else "# Project Memory\n"
            )
            block = render_sequence_policy_memory_block(record)
            target.write_text(current.rstrip() + "\n\n" + block, encoding="utf-8")
            results.append({
                "draft_id": record.get("draft_id"),
                "target_memory_path": record.get("target_memory_path"),
                "applied": True,
                "bytes_appended": len(block.encode("utf-8")),
            })
        return results

    def policy_memory_writer_payload(
        *,
        status,
        source_payload,
        records,
        blockers,
        write_results=None,
        apply_requested=False,
        operator_ack=False,
    ):
        records = records or []
        blockers = blockers or []
        write_results = write_results or []
        applied = status == "append_only_memory_write_applied"
        return {
            "type": "operator_proposal_apply_sequence_policy_memory_writer",
            "status": status,
            "success": applied,
            "reason": None if applied else "sequence_policy_memory_write_not_applied",
            "generated_at": time.time(),
            "source_memory_draft_status": source_payload.get("status"),
            "summary": {
                "memory_draft_record_count": len(records),
                "invalid_record_count": len(blockers),
                "write_result_count": len(write_results),
                "append_only_memory_write_applied": applied,
                "operator_ack_required": True,
                "operator_ack": operator_ack,
                "apply_requested": apply_requested,
                "target_memory_path": ".ai/memory/project.md",
                "live_write": False,
                "apply_attempted": False,
                "queue_mutation": False,
                "proposal_status_mutation": False,
                "idempotency_reserved": False,
                "manual_console_write_called": False,
                "mutates_console_state": False,
                "queries_console": False,
                "policy_mutation": False,
                "memory_file_mutation": applied,
                "runtime_memory_mutation": False,
                "model_mutation": False,
                "live_widening_allowed": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
                "blocked_by": [item["reason"] for item in blockers],
            },
            "safety": {
                "operator_required": True,
                "operator_ack_required": True,
                "apply_requested": apply_requested,
                "operator_ack": operator_ack,
                "append_only_memory_file_write_applied": applied,
                "target_memory_path": ".ai/memory/project.md",
                "does_not_call_apply_operator_proposal": True,
                "does_not_call_manual_console_write": True,
                "allowed_live_write_kinds": allowed_live_write_kinds_for_safety_surface(),
                "live_write": False,
                "apply_attempted": False,
                "queue_mutation": False,
                "proposal_status_mutation": False,
                "idempotency_reserved": False,
                "manual_console_write_called": False,
                "mutates_console_state": False,
                "queries_console": False,
                "policy_mutation": False,
                "memory_file_mutation": applied,
                "runtime_memory_mutation": False,
                "model_mutation": False,
                "live_widening_allowed": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
            },
            "write_results": write_results,
            "blocked_results": blockers,
        }

    async def handle_write_operator_proposal_apply_sequence_policy_memory(websocket, data):
        source_payload = sequence_policy_memory_draft_payload_from_data(data)
        apply_requested = data.get("apply") is True
        operator_ack = data.get("operator_ack") is True
        if not isinstance(source_payload, dict):
            await server.send_to_client(
                websocket,
                policy_memory_writer_payload(
                    status="memory_write_not_ready",
                    source_payload={},
                    records=[],
                    blockers=[policy_memory_writer_blocker("missing_memory_draft_payload")],
                    apply_requested=apply_requested,
                    operator_ack=operator_ack,
                ),
            )
            return

        raw_records = source_payload.get("memory_draft_records")
        records = [item for item in raw_records if isinstance(item, dict)] if isinstance(raw_records, list) else []
        blockers = policy_memory_writer_envelope_blockers(source_payload)
        if not isinstance(raw_records, list):
            blockers.append(policy_memory_writer_blocker("missing_memory_draft_records"))
        invalid_count = (len(raw_records) - len(records)) if isinstance(raw_records, list) else 0
        if invalid_count:
            blockers.append(
                policy_memory_writer_blocker(
                    "invalid_memory_draft_record",
                    row_index=None,
                )
            )
        for row_index, record in enumerate(records):
            blockers.extend(policy_memory_writer_record(record, row_index))

        write_results = []
        if blockers:
            status = "memory_write_replacement_required"
        elif not records:
            status = "memory_write_not_ready"
            blockers.append(policy_memory_writer_blocker("memory_draft_records_required"))
        elif apply_requested and not operator_ack:
            status = "memory_write_operator_ack_required"
            blockers.append(
                policy_memory_writer_blocker(
                    "operator_ack_required_before_append_only_memory_write"
                )
            )
        elif apply_requested and operator_ack:
            status = "append_only_memory_write_applied"
            root = data.get("root_path") or data.get("root") or "."
            write_results = append_sequence_policy_memory_records(root, records)
        else:
            status = "append_only_memory_write_dry_run_ready"
            blockers.append(
                policy_memory_writer_blocker(
                    "operator_ack_and_apply_required_before_append_only_memory_write"
                )
            )

        await server.send_to_client(
            websocket,
            policy_memory_writer_payload(
                status=status,
                source_payload=source_payload,
                records=records,
                blockers=blockers,
                write_results=write_results,
                apply_requested=apply_requested,
                operator_ack=operator_ack,
            ),
        )

    async def handle_index_operator_proposal_apply_sequence_policy_memory(websocket, data):
        result = build_operator_proposal_apply_sequence_policy_memory_index(
            memory_text=data.get("memory_text"),
            root=data.get("root_path") or data.get("root"),
            target_memory_path=data.get("target_memory_path") or ".ai/memory/project.md",
        )
        result["handler"] = "index_operator_proposal_apply_sequence_policy_memory"
        await server.send_to_client(websocket, result)

    async def handle_review_operator_proposal_apply_sequence_policy_memory_trust(
        websocket,
        data,
    ):
        result = build_operator_proposal_apply_sequence_policy_memory_trust_review(
            memory_index=data.get("memory_index"),
            current_sequence_review_freshness=(
                data.get("current_sequence_review_freshness")
                or data.get("sequence_review_freshness")
                or data.get("freshness")
            ),
        )
        result["handler"] = "review_operator_proposal_apply_sequence_policy_memory_trust"
        await server.send_to_client(websocket, result)

    async def handle_record_operator_proposal_apply_sequence_policy_memory_trust_acceptance(
        websocket,
        data,
    ):
        result = build_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger(
            memory_trust_review=(
                data.get("memory_trust_review")
                or data.get("memory_trust_review_result")
                or data.get("review")
            ),
            acceptance_records=data.get("acceptance_records"),
        )
        result["handler"] = (
            "record_operator_proposal_apply_sequence_policy_memory_trust_acceptance"
        )
        await server.send_to_client(websocket, result)

    async def handle_plan_operator_proposal_apply_sequence_policy_memory_trust_promotion(
        websocket,
        data,
    ):
        result = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_plan(
            acceptance_ledger=(
                data.get("acceptance_ledger")
                or data.get("memory_trust_acceptance_ledger")
                or data.get("memory_trust_acceptance_ledger_result")
                or data.get("ledger")
            ),
        )
        result["handler"] = (
            "plan_operator_proposal_apply_sequence_policy_memory_trust_promotion"
        )
        await server.send_to_client(websocket, result)

    async def handle_contract_operator_proposal_apply_sequence_policy_memory_trust_promotion(
        websocket,
        data,
    ):
        result = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract(
            promotion_plan=(
                data.get("promotion_plan")
                or data.get("promotion_plan_result")
                or data.get("plan")
            ),
            current_sequence_policy_context=(
                data.get("current_sequence_policy_context")
                or data.get("sequence_policy_context")
                or data.get("current_sequence_review_freshness")
                or data.get("freshness")
            ),
            rollback_evidence=(
                data.get("rollback_evidence")
                or data.get("sequence_policy_rollback_evidence")
            ),
        )
        result["handler"] = (
            "contract_operator_proposal_apply_sequence_policy_memory_trust_promotion"
        )
        await server.send_to_client(websocket, result)

    async def handle_record_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance(
        websocket,
        data,
    ):
        result = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger(
            promotion_contract=(
                data.get("promotion_contract")
                or data.get("promotion_contract_result")
                or data.get("contract")
            ),
            acceptance_records=data.get("acceptance_records"),
        )
        result["handler"] = (
            "record_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance"
        )
        await server.send_to_client(websocket, result)

    async def handle_plan_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution(
        websocket,
        data,
    ):
        result = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan(
            promotion_acceptance_ledger=(
                data.get("promotion_acceptance_ledger")
                or data.get("promotion_acceptance_ledger_result")
                or data.get("acceptance_ledger")
                or data.get("ledger")
            ),
        )
        result["handler"] = (
            "plan_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution"
        )
        await server.send_to_client(websocket, result)

    async def handle_dry_run_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor(
        websocket,
        data,
    ):
        result = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run(
            promotion_execution_plan=(
                data.get("promotion_execution_plan")
                or data.get("promotion_execution_plan_result")
                or data.get("execution_plan")
                or data.get("plan")
            ),
            apply=data.get("apply") is True,
            operator_ack=data.get("operator_ack") is True,
        )
        result["handler"] = (
            "dry_run_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor"
        )
        await server.send_to_client(websocket, result)

    async def handle_execute_operator_proposal_apply_sequence_policy_memory_trust_promotion(
        websocket,
        data,
    ):
        if "target_store" in data:
            target_store = data.get("target_store")
        elif "sequence_policy_trust_store" in data:
            target_store = data.get("sequence_policy_trust_store")
        else:
            target_store = None
        result = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor(
            executor_dry_run=(
                data.get("executor_dry_run")
                or data.get("promotion_executor_dry_run_result")
                or data.get("promotion_executor_dry_run")
                or data.get("dry_run")
            ),
            target_store=target_store,
            apply=data.get("apply") is True,
            operator_ack=data.get("operator_ack") is True,
        )
        result["handler"] = (
            "execute_operator_proposal_apply_sequence_policy_memory_trust_promotion"
        )
        await server.send_to_client(websocket, result)

    async def handle_audit_operator_proposal_apply_sequence_policy_memory_trust_promotion(
        websocket,
        data,
    ):
        result = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit(
            promotion_executor=(
                data.get("promotion_executor")
                or data.get("promotion_executor_result")
                or data.get("executor")
            ),
        )
        result["handler"] = (
            "audit_operator_proposal_apply_sequence_policy_memory_trust_promotion"
        )
        await server.send_to_client(websocket, result)

    async def handle_verify_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback(
        websocket,
        data,
    ):
        result = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification(
            apply_audit=(
                data.get("apply_audit")
                or data.get("promotion_apply_audit")
                or data.get("promotion_apply_audit_result")
                or data.get("audit")
            ),
            verification_records=(
                data.get("verification_records")
                or data.get("rollback_verification_records")
                or data.get("records")
            ),
        )
        result["handler"] = (
            "verify_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback"
        )
        await server.send_to_client(websocket, result)

    async def handle_gate_operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy(
        websocket,
        data,
    ):
        result = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate(
            rollback_verification=(
                data.get("rollback_verification")
                or data.get("promotion_rollback_verification")
                or data.get("promotion_rollback_verification_result")
                or data.get("verification")
            ),
        )
        result["handler"] = (
            "gate_operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy"
        )
        await server.send_to_client(websocket, result)

    async def handle_consume_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy(
        websocket,
        data,
    ):
        result = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer(
            proposals=(
                data.get("proposals")
                or data.get("preview_proposals")
                or data.get("proposal_preview")
                or data.get("operator_proposals")
            ),
            sequence_policy_gate=(
                data.get("sequence_policy_gate")
                or data.get("promotion_sequence_policy_gate")
                or data.get("promotion_sequence_policy_gate_result")
                or data.get("gate")
            ),
        )
        result["handler"] = (
            "consume_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy"
        )
        await server.send_to_client(websocket, result)

    async def handle_preview_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering(
        websocket,
        data,
    ):
        result = build_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview(
            proposals=(
                data.get("proposals")
                or data.get("preview_proposals")
                or data.get("proposal_preview")
                or data.get("operator_proposals")
            ),
            policy_consumer=(
                data.get("policy_consumer")
                or data.get("promotion_policy_consumer")
                or data.get("promotion_policy_consumer_result")
                or data.get("consumer")
            ),
        )
        result["handler"] = (
            "preview_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering"
        )
        await server.send_to_client(websocket, result)

    async def handle_plan_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering(
        websocket,
        data,
    ):
        result = (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan(
                ordering_preview=(
                    data.get("ordering_preview")
                    or data.get("promotion_ordering_preview")
                    or data.get("promotion_ordering_preview_result")
                    or data.get("preview")
                ),
            )
        )
        result["handler"] = (
            "plan_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering"
        )
        await server.send_to_client(websocket, result)

    handlers = {
        "get_dashboard_snapshot": handle_get_dashboard_snapshot,
        "get_channel_inventory": handle_get_channel_inventory,
        "get_channel_processing_snapshot": handle_get_channel_processing_snapshot,
        "get_decision_queue": handle_get_decision_queue,
        "get_operator_proposal_detail": handle_get_operator_proposal_detail,
        "get_operator_proposal_result_detail": (
            handle_get_operator_proposal_result_detail
        ),
        "get_decision_traces": handle_get_decision_traces,
        "get_safety_monitor": handle_get_safety_monitor,
        "get_replay_runs": handle_get_replay_runs,
        "start_replay_run": handle_start_replay_run,
        "get_channel_intelligence": handle_get_channel_intelligence,
        "get_connection_topology": handle_get_connection_topology,
        "get_operator_agent_roster": handle_get_operator_agent_roster,
        "get_groups_buses_snapshot": handle_get_groups_buses_snapshot,
        "get_bus_main_dca_snapshot": handle_get_bus_main_dca_snapshot,
        "get_matrix_snapshot": handle_get_matrix_snapshot,
        "get_routing_sends_snapshot": handle_get_routing_sends_snapshot,
        "get_fx_inserts_snapshot": handle_get_fx_inserts_snapshot,
        "get_dlive_state_snapshot": handle_get_dlive_state_snapshot,
        "get_snapshots_scenes_snapshot": handle_get_snapshots_scenes_snapshot,
        "get_user_presets_snapshot": handle_get_user_presets_snapshot,
        "get_memory_knowledge_snapshot": handle_get_memory_knowledge_snapshot,
        "get_training_sessions_snapshot": handle_get_training_sessions_snapshot,
        "get_telemetry_alerts_snapshot": handle_get_telemetry_alerts_snapshot,
        "get_rules_library_snapshot": handle_get_rules_library_snapshot,
        "get_agent_communications_snapshot": handle_get_agent_communications_snapshot,
        "get_processing_pipeline_snapshot": handle_get_processing_pipeline_snapshot,
        "get_main_pipeline_preview": handle_get_main_pipeline_preview,
        "get_operator_mode": handle_get_operator_mode,
        "set_operator_mode": handle_set_operator_mode,
        "run_operator_analysis": handle_run_operator_analysis,
        "import_safe_gain_suggestions": handle_import_safe_gain_suggestions,
        "import_soundcheck_recommendations": handle_import_soundcheck_recommendations,
        "import_source_identity_queue_ordering_plan": (
            handle_import_source_identity_queue_ordering_plan
        ),
        "import_operator_proposal_apply_sequence_policy_queue_ordering_plan": (
            handle_import_operator_proposal_apply_sequence_policy_queue_ordering_plan
        ),
        "plan_operator_proposal_apply_sequence_policy_queue_import_preflight": (
            handle_plan_operator_proposal_apply_sequence_policy_queue_import_preflight
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_preflight_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_preflight_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_sequence_plan_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_sequence_plan_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_dry_run_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_dry_run_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_execution_gate_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_execution_gate_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_execution_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_execution_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_execution_audit_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_execution_audit_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_audit_policy_gate_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_audit_policy_gate_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_memory_draft_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_memory_draft_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_memory_writer_dry_run_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_memory_writer_dry_run_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_memory_writer_apply_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_memory_writer_apply_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_memory_index_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_memory_index_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_memory_trust_review_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_memory_trust_review_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_memory_trust_acceptance_ledger_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_memory_trust_acceptance_ledger_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_plan_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_plan_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_contract_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_contract_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_acceptance_ledger_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_acceptance_ledger_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_execution_plan_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_execution_plan_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_executor_dry_run_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_executor_dry_run_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_executor_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_executor_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_apply_audit_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_apply_audit_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_rollback_verification_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_rollback_verification_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_sequence_policy_gate_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_sequence_policy_gate_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_policy_consumer_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_policy_consumer_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_ordering_preview_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_ordering_preview_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_queue_ordering_plan_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_queue_ordering_plan_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_queue_importer_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_queue_importer_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_preflight_plan_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_preflight_plan_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_batch_preflight_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_batch_preflight_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_sequence_plan_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_sequence_plan_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_dry_run_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_dry_run_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_execution_gate_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_execution_gate_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_execution_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_execution_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_execution_audit_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_execution_audit_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_audit_policy_gate_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_audit_policy_gate_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_draft_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_draft_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_writer_dry_run_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_writer_dry_run_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_writer_apply_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_writer_apply_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_index_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_index_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_trust_review_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_trust_review_result
        ),
        "verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_trust_acceptance_ledger_result": (
            handle_verify_operator_proposal_apply_sequence_policy_queue_import_promotion_memory_trust_acceptance_ledger_result
        ),
        "create_operator_proposal": handle_create_operator_proposal,
        "accept_operator_proposal": handle_accept_operator_proposal,
        "dismiss_operator_proposal": handle_dismiss_operator_proposal,
        "preflight_operator_proposal_apply": handle_preflight_operator_proposal_apply,
        "preflight_operator_proposal_apply_batch": (
            handle_preflight_operator_proposal_apply_batch
        ),
        "plan_operator_proposal_apply_sequence": (
            handle_plan_operator_proposal_apply_sequence
        ),
        "dry_run_operator_proposal_apply_sequence": (
            handle_dry_run_operator_proposal_apply_sequence
        ),
        "gate_operator_proposal_apply_sequence_execution": (
            handle_gate_operator_proposal_apply_sequence_execution
        ),
        "execute_operator_proposal_apply_sequence": (
            handle_execute_operator_proposal_apply_sequence
        ),
        "audit_operator_proposal_apply_sequence_execution": (
            handle_audit_operator_proposal_apply_sequence_execution
        ),
        "gate_operator_proposal_apply_sequence_audit_policy": (
            handle_gate_operator_proposal_apply_sequence_audit_policy
        ),
        "draft_operator_proposal_apply_sequence_policy_memory": (
            handle_draft_operator_proposal_apply_sequence_policy_memory
        ),
        "write_operator_proposal_apply_sequence_policy_memory": (
            handle_write_operator_proposal_apply_sequence_policy_memory
        ),
        "index_operator_proposal_apply_sequence_policy_memory": (
            handle_index_operator_proposal_apply_sequence_policy_memory
        ),
        "review_operator_proposal_apply_sequence_policy_memory_trust": (
            handle_review_operator_proposal_apply_sequence_policy_memory_trust
        ),
        "record_operator_proposal_apply_sequence_policy_memory_trust_acceptance": (
            handle_record_operator_proposal_apply_sequence_policy_memory_trust_acceptance
        ),
        "plan_operator_proposal_apply_sequence_policy_memory_trust_promotion": (
            handle_plan_operator_proposal_apply_sequence_policy_memory_trust_promotion
        ),
        "contract_operator_proposal_apply_sequence_policy_memory_trust_promotion": (
            handle_contract_operator_proposal_apply_sequence_policy_memory_trust_promotion
        ),
        "record_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance": (
            handle_record_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance
        ),
        "plan_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution": (
            handle_plan_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution
        ),
        "dry_run_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor": (
            handle_dry_run_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor
        ),
        "execute_operator_proposal_apply_sequence_policy_memory_trust_promotion": (
            handle_execute_operator_proposal_apply_sequence_policy_memory_trust_promotion
        ),
        "audit_operator_proposal_apply_sequence_policy_memory_trust_promotion": (
            handle_audit_operator_proposal_apply_sequence_policy_memory_trust_promotion
        ),
        "verify_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback": (
            handle_verify_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback
        ),
        "gate_operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy": (
            handle_gate_operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy
        ),
        "consume_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy": (
            handle_consume_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy
        ),
        "preview_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering": (
            handle_preview_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering
        ),
        "plan_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering": (
            handle_plan_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering
        ),
        "apply_operator_proposal": handle_apply_operator_proposal,
    }

    def post_executor_migration_required_handler(handler_name):
        async def handle_migration_required(websocket, data):
            del data
            await server.send_to_client(
                websocket,
                {
                    "type": "operator_proposal_pipeline_migration_required",
                    "contract_version": 2,
                    "status": "migration_required",
                    "success": False,
                    "reason": "server_owned_sequence_state_required",
                    "blocked_handler": handler_name,
                    "no_additional_console_send_proven": True,
                    "manual_review_required": True,
                    "soundcheck_completion_blocked": True,
                    "memory_mutation": False,
                    "policy_mutation": False,
                    "queue_mutation": False,
                    "model_mutation": False,
                    "live_authority_granted": False,
                },
            )

        return handle_migration_required

    for handler_name in POST_EXECUTOR_BLOCKED_HANDLER_NAMES:
        handlers[handler_name] = post_executor_migration_required_handler(
            handler_name
        )

    for handler_name in LEGACY_CLIENT_ECHO_CONSUMER_HANDLERS:
        handlers[handler_name] = post_executor_migration_required_handler(
            handler_name
        )

    for handler_name in MUTATION_CAPABLE_LEGACY_HANDLER_NAMES:
        original_handler = handlers[handler_name]
        migration_handler = post_executor_migration_required_handler(handler_name)

        async def guard_legacy_mutation(
            websocket,
            data,
            *,
            _original_handler=original_handler,
            _migration_handler=migration_handler,
        ):
            if data.get("apply") is True and data.get("operator_ack") is True:
                return await _migration_handler(websocket, data)
            return await _original_handler(websocket, data)

        handlers[handler_name] = guard_legacy_mutation
    return handlers
