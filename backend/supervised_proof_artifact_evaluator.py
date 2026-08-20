"""Fail-closed evaluator for supervised fader/gain proof artifacts.

The evaluator is intentionally transport-free. It reviews JSON-like proof
summaries from dry-run or physical validation runners and decides whether the
artifact is only dry-run-ready, a supervised promotion candidate, or blocked.
It never opens mixer connections and never creates live write requests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Tuple

try:
    from operator_mode_policy import WING_DEPLOYMENT_LIVE_WRITE_KINDS
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_mode_policy import WING_DEPLOYMENT_LIVE_WRITE_KINDS


SUPERVISED_PROOF_METHOD_ID = "supervised_proof_artifact_evaluator"
SUPPORTED_VALUE_TYPES = set(WING_DEPLOYMENT_LIVE_WRITE_KINDS)
EVALUATION_STATUSES = {"dry_run_ready", "supervised_candidate", "blocked"}
LEGACY_AMBIGUOUS_PROOF_MODE = "legacy_ambiguous"
VERIFIED_READBACK_STATUSES = {
    "verified_from_cache",
    "verified_from_physical_readback",
    "verified_from_simulated_readback",
    "simulated_readback_verified",
}
PHYSICAL_READBACK_STATUSES = {"verified_from_physical_readback"}
VERIFIED_PROOF_STATUSES = {"verified", "passed"}
PHYSICAL_READBACK_SOURCES = {
    "physical_readback",
    "wing_physical_readback",
    "physical_wing",
    "wing",
}
DRY_RUN_SOURCES = {
    "fake_mixer",
    "dry_run",
    "mixer_state_cache",
    "cache",
    "offline_simulation",
}
NON_PROMOTABLE_SUMMARY_STATUSES = {"manual_review_required", "blocked", "failed", "error"}


@dataclass(frozen=True)
class SupervisedProofArtifactInput:
    proof_id: str
    value_type: str
    channel: int
    old_value: float | None = None
    proposed_value: float | None = None
    delta_db: float | None = None
    proof_mode: str = "dry_run"
    pre_read_status: str = ""
    pre_read_source: str = ""
    readback_status: str = ""
    readback_source: str = ""
    proof_status: str = ""
    rollback_available: bool = False
    rollback_value: float | None = None
    rollback_readback_status: str = ""
    rollback_readback_source: str = ""
    rollback_proof_status: str = ""
    emergency_stop_proof_status: str = ""
    blocked_followup_status: str = ""
    operator_approval_recorded: bool = False
    idempotency_key_present: bool = False
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized_delta(self) -> float | None:
        if self.delta_db is not None:
            return _coerce_float(self.delta_db)
        if self.old_value is None or self.proposed_value is None:
            return None
        return round(float(self.proposed_value) - float(self.old_value), 6)


@dataclass(frozen=True)
class SupervisedProofArtifactEvaluation:
    method_id: str
    proof_id: str
    value_type: str
    channel: int
    proof_mode: str
    status: str
    promotion_allowed: bool
    autonomous_ready: bool
    blockers: Tuple[str, ...]
    evidence: Dict[str, Any]

    def __post_init__(self) -> None:
        if self.status not in EVALUATION_STATUSES:
            raise ValueError(f"Unknown supervised proof status: {self.status}")
        if self.autonomous_ready:
            raise ValueError("Proof artifacts cannot mark Full Autonomous ready")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["blockers"] = list(self.blockers)
        return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _flag_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).strip().lower() in {"1", "true", "yes", "y"}


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _nested_dict(payload: Mapping[str, Any], key: str) -> Dict[str, Any]:
    value = payload.get(key)
    return dict(value) if isinstance(value, Mapping) else {}


def _inner_apply_result(payload: Mapping[str, Any]) -> Dict[str, Any]:
    result = payload.get("apply_result")
    return dict(result) if isinstance(result, Mapping) else dict(payload)


def _status_verified(status: str) -> bool:
    return status in VERIFIED_READBACK_STATUSES


def _proof_verified(status: str) -> bool:
    return status in VERIFIED_PROOF_STATUSES


def _positive_delta_requires_reason(artifact: SupervisedProofArtifactInput) -> bool:
    delta = artifact.normalized_delta()
    return delta is not None and delta > 0.0 and not artifact.reason.strip()


def _source_is_physical(source: str) -> bool:
    return source in PHYSICAL_READBACK_SOURCES


def _infer_source(source: str, status: str, *, physical_mode: bool, fallback: str) -> str:
    if source:
        return source
    if status == "verified_from_physical_readback" and physical_mode:
        return "physical_readback"
    if status == "verified_from_cache":
        return "mixer_state_cache"
    return fallback


def _append_common_blockers(artifact: SupervisedProofArtifactInput, blockers: list[str]) -> None:
    metadata = artifact.metadata or {}
    summary_status = _text(metadata.get("summary_status"))
    summary_reason = _text(metadata.get("summary_reason"))
    fail_closed = _flag_true(metadata.get("fail_closed"))
    live_apply_request_sent = _flag_true(metadata.get("live_apply_request_sent"))

    if summary_status in NON_PROMOTABLE_SUMMARY_STATUSES:
        blockers.append("summary_status_not_promotable")
    if summary_reason == "connection_or_timeout_after_apply_request" or (
        summary_status == "manual_review_required" and live_apply_request_sent
    ):
        blockers.append("transport_failure_after_apply_requires_manual_review")
    if summary_reason == "connection_or_timeout_fail_closed" or (fail_closed and not live_apply_request_sent):
        blockers.append("transport_failure_before_apply_fail_closed")
    if fail_closed and live_apply_request_sent:
        blockers.append("invalid_fail_closed_after_live_apply")

    if artifact.proof_mode == LEGACY_AMBIGUOUS_PROOF_MODE:
        blockers.append("legacy_physical_schema_missing")
    if artifact.value_type not in SUPPORTED_VALUE_TYPES:
        blockers.append("unsupported_value_type")
    if artifact.channel < 1 or artifact.channel > 48:
        blockers.append("invalid_channel")
    if not _status_verified(artifact.pre_read_status):
        blockers.append("pre_read_unavailable")
    if not _status_verified(artifact.readback_status):
        blockers.append("apply_readback_missing")
    if not _proof_verified(artifact.proof_status):
        blockers.append("proof_status_not_verified")
    if not artifact.rollback_available:
        blockers.append("rollback_unavailable")
    if artifact.rollback_value is None:
        blockers.append("rollback_value_missing")
    if not _status_verified(artifact.rollback_readback_status):
        blockers.append("rollback_readback_missing")
    if not _proof_verified(artifact.rollback_proof_status):
        blockers.append("rollback_proof_missing")
    if not _proof_verified(artifact.emergency_stop_proof_status):
        blockers.append("emergency_stop_proof_missing")
    if not _proof_verified(artifact.blocked_followup_status):
        blockers.append("blocked_followup_missing")
    if not artifact.operator_approval_recorded:
        blockers.append("operator_approval_missing")
    if not artifact.idempotency_key_present:
        blockers.append("idempotency_key_missing")
    if artifact.value_type == "gain" and _positive_delta_requires_reason(artifact):
        blockers.append("gain_up_requires_operator_authored_reason")


def _append_physical_blockers(
    artifact: SupervisedProofArtifactInput,
    blockers: list[str],
    *,
    require_physical: bool,
) -> None:
    physical_mode = artifact.proof_mode == "physical"
    if require_physical and not physical_mode:
        blockers.append("physical_proof_required_for_promotion")
        return
    if not physical_mode:
        return
    if artifact.pre_read_status not in PHYSICAL_READBACK_STATUSES:
        blockers.append("physical_pre_read_status_required")
    if not _source_is_physical(artifact.pre_read_source):
        blockers.append("physical_pre_read_required")
    if artifact.readback_status not in PHYSICAL_READBACK_STATUSES:
        blockers.append("physical_apply_readback_status_required")
    if not _source_is_physical(artifact.readback_source):
        blockers.append("physical_apply_readback_required")
    if artifact.rollback_readback_status not in PHYSICAL_READBACK_STATUSES:
        blockers.append("physical_rollback_readback_status_required")
    if not _source_is_physical(artifact.rollback_readback_source):
        blockers.append("physical_rollback_readback_required")


def _verified_fake_provenance_rejection(
    artifact: SupervisedProofArtifactInput,
) -> bool:
    """Recognize a dry-run that proves fake WING proof is rejected pre-write."""

    metadata = artifact.metadata or {}
    return bool(
        artifact.proof_mode == "dry_run"
        and metadata.get("physical_wing") is False
        and metadata.get("physical_proof_claimed") is False
        and metadata.get("fake_physical_proof_rejection_status") == "verified"
        and metadata.get("summary_status") == "passed"
        and artifact.pre_read_status == "readback_unavailable"
        and artifact.pre_read_source == "fake_mixer"
        and artifact.proof_status == "blocked"
        and artifact.rollback_available is False
        and artifact.rollback_readback_status == "not_attempted"
        and artifact.rollback_readback_source == "not_attempted"
        and artifact.rollback_proof_status == "not_attempted"
        and _proof_verified(artifact.emergency_stop_proof_status)
        and _proof_verified(artifact.blocked_followup_status)
        and artifact.operator_approval_recorded
        and artifact.idempotency_key_present
    )


def evaluate_supervised_proof_artifact(
    artifact: SupervisedProofArtifactInput,
    *,
    require_physical: bool = False,
) -> SupervisedProofArtifactEvaluation:
    """Evaluate one proof artifact without granting autonomous authority."""

    blockers: list[str] = []
    fake_provenance_rejection = _verified_fake_provenance_rejection(artifact)
    if not fake_provenance_rejection:
        _append_common_blockers(artifact, blockers)
    common_blockers = tuple(blockers)
    _append_physical_blockers(artifact, blockers, require_physical=require_physical)
    blockers_tuple = tuple(sorted(set(blockers)))

    if common_blockers:
        status = "blocked"
    elif artifact.proof_mode == "physical":
        status = "supervised_candidate" if not blockers_tuple else "blocked"
    else:
        status = "dry_run_ready"

    promotion_allowed = status == "supervised_candidate" and not blockers_tuple
    evidence = {
        "pre_read_status": artifact.pre_read_status,
        "pre_read_source": artifact.pre_read_source,
        "readback_status": artifact.readback_status,
        "readback_source": artifact.readback_source,
        "proof_status": artifact.proof_status,
        "rollback_available": artifact.rollback_available,
        "rollback_readback_status": artifact.rollback_readback_status,
        "rollback_readback_source": artifact.rollback_readback_source,
        "rollback_proof_status": artifact.rollback_proof_status,
        "emergency_stop_proof_status": artifact.emergency_stop_proof_status,
        "blocked_followup_status": artifact.blocked_followup_status,
        "operator_approval_recorded": artifact.operator_approval_recorded,
        "idempotency_key_present": artifact.idempotency_key_present,
        "fake_physical_proof_rejection_status": (
            "verified" if fake_provenance_rejection else ""
        ),
        "physical_proof_claimed": bool(
            (artifact.metadata or {}).get("physical_proof_claimed")
        ),
        "delta_db": artifact.normalized_delta(),
        "require_physical": require_physical,
    }
    if artifact.metadata:
        evidence["metadata"] = dict(artifact.metadata)

    return SupervisedProofArtifactEvaluation(
        method_id=SUPERVISED_PROOF_METHOD_ID,
        proof_id=artifact.proof_id,
        value_type=artifact.value_type,
        channel=artifact.channel,
        proof_mode=artifact.proof_mode,
        status=status,
        promotion_allowed=promotion_allowed,
        autonomous_ready=False,
        blockers=blockers_tuple,
        evidence=evidence,
    )


def _proof_mode_from_payload(payload: Mapping[str, Any]) -> str:
    explicit = _text(payload.get("proof_mode"))
    if explicit in {"physical", "dry_run"}:
        return explicit
    if payload.get("physical_wing") is True:
        return "physical"
    if payload.get("physical_wing") is False:
        return "dry_run"
    run_hint = " ".join(
        _text(value)
        for value in (
            payload.get("proof_runner"),
            payload.get("run_id"),
            payload.get("proof_id"),
            payload.get("summary_path"),
        )
    )
    if _flag_true(payload.get("execute")) and "wing_operator_proposal" in run_hint:
        return LEGACY_AMBIGUOUS_PROOF_MODE
    return "dry_run"


def _value_type_from_payload(payload: Mapping[str, Any], apply_result: Mapping[str, Any]) -> str:
    value_type = _text(payload.get("value_type") or apply_result.get("value_type"))
    if value_type:
        return value_type
    address = _text(payload.get("address") or apply_result.get("address"))
    if "trim" in address:
        return "gain"
    if "fdr" in address:
        return "fader"
    run_id = _text(payload.get("run_id") or payload.get("proof_id"))
    return "gain" if "gain" in run_id else value_type


def _verified_if(condition: bool) -> str:
    return "verified" if condition else ""


def _blocked_followup_status(payload: Mapping[str, Any]) -> str:
    cooldown = _inner_apply_result(_nested_dict(payload, "cooldown_apply_result"))
    blocked = _inner_apply_result(_nested_dict(payload, "blocked_apply_result"))
    if cooldown.get("reason") == "cooldown":
        return "verified"
    if blocked.get("reason") == "operator_mode_blocks_live_write":
        return "verified"
    return _text(payload.get("blocked_followup_status"))


def _emergency_status(payload: Mapping[str, Any]) -> str:
    emergency_apply = _inner_apply_result(_nested_dict(payload, "post_emergency_apply_result"))
    if emergency_apply.get("reason") == "emergency_stop_active":
        return "verified"
    return _text(payload.get("emergency_stop_proof_status"))


def supervised_proof_artifact_from_dict(payload: Mapping[str, Any]) -> SupervisedProofArtifactInput:
    """Normalize a runner summary or already-normalized dict into an input row."""

    apply_payload = _nested_dict(payload, "apply_result")
    apply_result = _inner_apply_result(apply_payload)
    rollback_result = _nested_dict(payload, "rollback_result")
    proof_mode = _proof_mode_from_payload(payload)
    physical_mode = proof_mode == "physical"

    value_type = _value_type_from_payload(payload, apply_result)
    pre_read_status = _text(
        payload.get("pre_read_status")
        or apply_result.get("pre_read_status")
        or payload.get("old_value_status")
    )
    pre_read_source = _infer_source(
        _text(payload.get("pre_read_source") or apply_result.get("pre_read_source") or payload.get("old_value_source")),
        pre_read_status,
        physical_mode=physical_mode,
        fallback="fake_mixer" if payload.get("physical_wing") is False else "",
    )
    readback_status = _text(payload.get("readback_status") or apply_result.get("readback_status"))
    readback_source = _infer_source(
        _text(payload.get("readback_source") or apply_result.get("readback_source")),
        readback_status,
        physical_mode=physical_mode,
        fallback="fake_mixer" if payload.get("physical_wing") is False else "",
    )
    proof_status = _text(payload.get("proof_status") or apply_result.get("proof_status"))
    if not proof_status and apply_result.get("status") == "applied" and _status_verified(readback_status):
        proof_status = "verified"

    rollback_readback_status = _text(
        payload.get("rollback_readback_status")
        or rollback_result.get("rollback_readback_status")
        or rollback_result.get("readback_status")
    )
    rollback_readback_source = _infer_source(
        _text(payload.get("rollback_readback_source") or rollback_result.get("rollback_readback_source")),
        rollback_readback_status,
        physical_mode=physical_mode,
        fallback="fake_mixer" if payload.get("physical_wing") is False else "",
    )
    rollback_proof_status = _text(
        payload.get("rollback_proof_status")
        or rollback_result.get("rollback_proof_status")
        or rollback_result.get("proof_status")
    )
    if not rollback_proof_status and rollback_result.get("status") == "applied" and _status_verified(rollback_readback_status):
        rollback_proof_status = "verified"

    old_value = _coerce_float(_first_present(payload.get("old_value"), apply_result.get("old_value")))
    proposed_value = _coerce_float(
        _first_present(payload.get("target_value"), apply_result.get("effective_value"), apply_result.get("value"))
    )
    delta_db = _coerce_float(payload.get("delta_db"))
    if delta_db is None and old_value is not None and proposed_value is not None:
        delta_db = round(proposed_value - old_value, 6)

    proof_id = _text(payload.get("proof_id") or payload.get("run_id") or apply_result.get("proof_id") or "unnamed_proof")
    status_passed = payload.get("status") in {"passed", "applied"} or apply_result.get("status") == "applied"
    supervised_policy = bool(payload.get("write_policy") or apply_payload or apply_result)

    return SupervisedProofArtifactInput(
        proof_id=proof_id,
        value_type=value_type,
        channel=_coerce_int(payload.get("channel") or apply_result.get("channel")),
        old_value=old_value,
        proposed_value=proposed_value,
        delta_db=delta_db,
        proof_mode=proof_mode,
        pre_read_status=pre_read_status,
        pre_read_source=pre_read_source,
        readback_status=readback_status,
        readback_source=readback_source,
        proof_status=proof_status,
        rollback_available=bool(payload.get("rollback_available", apply_result.get("rollback_available", False))),
        rollback_value=_coerce_float(_first_present(payload.get("rollback_value"), apply_result.get("rollback_value"), old_value)),
        rollback_readback_status=rollback_readback_status,
        rollback_readback_source=rollback_readback_source,
        rollback_proof_status=rollback_proof_status,
        emergency_stop_proof_status=_emergency_status(payload),
        blocked_followup_status=_blocked_followup_status(payload),
        operator_approval_recorded=bool(payload.get("operator_approval_recorded", status_passed and supervised_policy)),
        idempotency_key_present=bool(payload.get("idempotency_key_present", status_passed and supervised_policy)),
        reason=_text(payload.get("operator_reason") or payload.get("approval_reason")),
        metadata={
            "summary_status": payload.get("status"),
            "summary_reason": payload.get("reason"),
            "fail_closed": payload.get("fail_closed"),
            "live_apply_request_sent": payload.get("live_apply_request_sent"),
            "rollback_request_sent": payload.get("rollback_request_sent"),
            "write_policy": payload.get("write_policy"),
            "execute": payload.get("execute"),
            "proof_artifact_schema": payload.get("proof_artifact_schema"),
            "proof_runner": payload.get("proof_runner"),
            "physical_wing": payload.get("physical_wing"),
            "physical_proof_claimed": payload.get("physical_proof_claimed"),
            "fake_physical_proof_rejection_status": payload.get(
                "fake_physical_proof_rejection_status"
            ),
        },
    )


def evaluate_supervised_proof_artifact_dict(
    payload: Mapping[str, Any],
    *,
    require_physical: bool = False,
) -> SupervisedProofArtifactEvaluation:
    return evaluate_supervised_proof_artifact(
        supervised_proof_artifact_from_dict(payload),
        require_physical=require_physical,
    )


def build_supervised_proof_artifact_snapshot(
    artifacts: Iterable[SupervisedProofArtifactInput | Mapping[str, Any]],
    *,
    require_physical: bool = False,
) -> Dict[str, Any]:
    rows = []
    for item in artifacts:
        if isinstance(item, SupervisedProofArtifactInput):
            evaluation = evaluate_supervised_proof_artifact(item, require_physical=require_physical)
        else:
            evaluation = evaluate_supervised_proof_artifact_dict(item, require_physical=require_physical)
        rows.append(evaluation)

    status_counts: Dict[str, int] = {status: 0 for status in sorted(EVALUATION_STATUSES)}
    for row in rows:
        status_counts[row.status] += 1

    return {
        "type": "supervised_proof_artifact_evaluation",
        "generated_at": _utc_now(),
        "status": "ok",
        "safety": {
            "read_only": True,
            "live_write": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "supervised_only": True,
        },
        "summary": {
            "artifact_count": len(rows),
            "status_counts": status_counts,
            "promotion_allowed_proof_ids": [row.proof_id for row in rows if row.promotion_allowed],
            "autonomous_ready_proof_ids": [],
            "require_physical": require_physical,
            "supported_value_types": sorted(SUPPORTED_VALUE_TYPES),
        },
        "rows": [row.to_dict() for row in rows],
    }
