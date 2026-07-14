"""Canonical supervised transaction coordinator for console corrections.

The coordinator owns approval, lifecycle checks, idempotency, emergency-state
checks, sequencing and reverse rollback. Family adapters own WING semantics.
No adapter is registered automatically and this module imports no mixer client.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
import json
import math
import time
from typing import Any, Awaitable, Callable, Dict, Mapping, Protocol, Tuple

from supervised_console_parameter_registry import (
    AUDIO_RECHECK_NOT_APPLICABLE,
    AUDIO_RECHECK_REQUIRED,
    SupervisedConsoleFamilyContract,
)
from supervised_family_proof_plan import (
    canonical_operation_bundle_bytes,
    canonicalize_attended_operation,
    validate_canonical_operation_bundle,
)
from supervised_proof_capability import (
    MaintenanceDspResetCommitOnceGrant,
    ProcessorEnableCommitOnceGrant,
    RestoreOnlyProofGrant,
    SoundcheckChildCommitOnceGrant,
)


@dataclass(frozen=True)
class ConsoleTransactionOperation:
    operation_id: str
    family_id: str
    write_kind: str
    target_value: Any
    channel: int | None = None
    parameter: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConsoleTransactionPlan:
    transaction_id: str
    operations: Tuple[ConsoleTransactionOperation, ...]
    reason: str


@dataclass(frozen=True)
class ConsoleTransactionContext:
    approved: bool
    approval_id: str
    idempotency_key: str
    emergency_generation: int
    proof_run: bool = False
    restore_after_verify: bool = False
    approval_deadline_epoch: float | None = None
    capability_grant: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class AdapterStepResult:
    success: bool
    physical_verified: bool
    reason: str
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AudioRecheckResult:
    status: str
    reason: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status in {"passed", "not_applicable"}


class SupervisedConsoleFamilyAdapter(Protocol):
    family_id: str

    def prepare(
        self,
        operation: ConsoleTransactionOperation,
    ) -> Mapping[str, Any]: ...

    def validate_snapshot(
        self,
        operation: ConsoleTransactionOperation,
        snapshot: Mapping[str, Any],
    ) -> str | None: ...

    async def snapshot(
        self,
        operation: ConsoleTransactionOperation,
    ) -> Mapping[str, Any]: ...

    async def apply(
        self,
        operation: ConsoleTransactionOperation,
        snapshot: Mapping[str, Any],
    ) -> AdapterStepResult: ...

    async def verify(
        self,
        operation: ConsoleTransactionOperation,
        snapshot: Mapping[str, Any],
        apply_result: AdapterStepResult,
    ) -> AdapterStepResult: ...

    async def rollback(
        self,
        operation: ConsoleTransactionOperation,
        snapshot: Mapping[str, Any],
        apply_result: AdapterStepResult,
    ) -> AdapterStepResult: ...

    async def assess_audio_failure_retention(
        self,
        operation: ConsoleTransactionOperation,
        snapshot: Mapping[str, Any],
        apply_result: AdapterStepResult,
    ) -> AdapterStepResult: ...

    async def audio_recheck(
        self,
        operation: ConsoleTransactionOperation,
        snapshot: Mapping[str, Any],
        apply_result: AdapterStepResult,
    ) -> AudioRecheckResult: ...


@dataclass
class ConsoleTransactionResult:
    transaction_id: str
    status: str
    reason: str
    applied_operation_ids: list[str] = field(default_factory=list)
    rollback_operation_ids: list[str] = field(default_factory=list)
    already_satisfied_operation_ids: list[str] = field(default_factory=list)
    contained_operation_ids: list[str] = field(default_factory=list)
    retention_unverified_operation_ids: list[str] = field(
        default_factory=list
    )
    safe_target_retention_attempted: bool = False
    contained_target_physical_verified: bool | None = None
    audio_failure_recovery_state_known: bool | None = None
    events: list[Dict[str, Any]] = field(default_factory=list)
    emergency_stop_activated: bool = False
    emergency_latch_succeeded: bool = False
    emergency_teardown_succeeded: bool = False
    emergency_stop_complete: bool = False
    live_authority_granted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


MaybeAwaitable = Awaitable[Any] | Any


async def _maybe_await(value: MaybeAwaitable) -> Any:
    if asyncio.iscoroutine(value) or hasattr(value, "__await__"):
        return await value
    return value


class SupervisedConsoleTransactionCoordinator:
    """Execute reviewed plans through explicit fail-closed family adapters."""

    def __init__(
        self,
        *,
        family_contracts: Mapping[str, SupervisedConsoleFamilyContract],
        adapters: Mapping[str, SupervisedConsoleFamilyAdapter],
        emergency_generation_getter: Callable[[], int],
        emergency_latch: Callable[[str], Any],
        emergency_stop: Callable[[str], MaybeAwaitable],
        prepared_callback: Callable[
            [
                ConsoleTransactionPlan,
                ConsoleTransactionContext,
                Tuple[Mapping[str, Any], ...],
            ],
            MaybeAwaitable,
        ]
        | None = None,
        before_operation_apply_callback: Callable[
            [ConsoleTransactionOperation], MaybeAwaitable
        ]
        | None = None,
        before_apply_callback: Callable[[], MaybeAwaitable] | None = None,
    ) -> None:
        self.family_contracts = dict(family_contracts)
        self.adapters = dict(adapters)
        self.emergency_generation_getter = emergency_generation_getter
        self.emergency_latch = emergency_latch
        self.emergency_stop = emergency_stop
        self.prepared_callback = prepared_callback
        self.before_operation_apply_callback = (
            before_operation_apply_callback
        )
        self.before_apply_callback = before_apply_callback
        self._idempotency_keys: Dict[str, str] = {}
        self._lock = asyncio.Lock()

    def _blocked(
        self,
        plan: ConsoleTransactionPlan,
        reason: str,
        *,
        events: list[Dict[str, Any]] | None = None,
    ) -> ConsoleTransactionResult:
        return ConsoleTransactionResult(
            transaction_id=plan.transaction_id,
            status="blocked",
            reason=reason,
            events=list(events or []),
        )

    def _validate_plan(
        self,
        plan: ConsoleTransactionPlan,
        context: ConsoleTransactionContext,
    ) -> str | None:
        grant = context.capability_grant
        private_commit_grant = bool(
            type(grant)
            in {
                MaintenanceDspResetCommitOnceGrant,
                ProcessorEnableCommitOnceGrant,
                SoundcheckChildCommitOnceGrant,
            }
            and grant.is_authentic()
        )
        if grant is not None:
            if not (
                type(grant)
                in {
                    RestoreOnlyProofGrant,
                    MaintenanceDspResetCommitOnceGrant,
                    ProcessorEnableCommitOnceGrant,
                    SoundcheckChildCommitOnceGrant,
                }
                and grant.is_authentic()
            ):
                return "m110_typed_capability_grant_required"
            if not grant.begin_execution():
                return "m110_capability_grant_already_used"
        if context.approved is not True:
            return "approval_required"
        if not context.approval_id:
            return "approval_id_required"
        if not context.idempotency_key:
            return "idempotency_key_required"
        if not plan.transaction_id:
            return "transaction_id_required"
        if not plan.operations:
            return "transaction_has_no_operations"
        if len(plan.operations) > 64:
            return "transaction_operation_limit_exceeded"
        operation_ids = [operation.operation_id for operation in plan.operations]
        if any(not operation_id for operation_id in operation_ids):
            return "operation_id_required"
        if len(operation_ids) != len(set(operation_ids)):
            return "duplicate_operation_id"

        proof_lane_requested = bool(
            context.proof_run is True
            or context.restore_after_verify is True
            or type(grant) is RestoreOnlyProofGrant
        )
        if proof_lane_requested and not (
            type(grant) is RestoreOnlyProofGrant and grant.is_authentic()
        ):
            return "restore_only_proof_grant_required"
        capability_bundle: Mapping[str, Any] | None = None
        if grant is not None:
            if private_commit_grant and (
                context.proof_run is True or context.restore_after_verify is True
            ):
                return "m110_commit_once_flags_must_be_false"
            if type(grant) is not RestoreOnlyProofGrant or not grant.is_authentic():
                if not private_commit_grant:
                    return "m110_typed_capability_grant_required"
            elif (
                context.proof_run is not True
                or context.restore_after_verify is not True
            ):
                return "restore_only_proof_flags_forced"
            if (
                context.approval_id != grant.binding.approval_id
                or context.idempotency_key != grant.binding.idempotency_key
            ):
                return "m110_capability_approval_binding_mismatch"
            try:
                decoded = json.loads(
                    grant.binding.canonical_bundle_bytes.decode("utf-8")
                )
                capability_bundle = validate_canonical_operation_bundle(decoded)
                if (
                    canonical_operation_bundle_bytes(capability_bundle)
                    != grant.binding.canonical_bundle_bytes
                ):
                    return "m110_capability_bundle_bytes_not_canonical"
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                return "m110_capability_bundle_invalid"
            if len(capability_bundle["operations"]) != len(plan.operations):
                return "m110_capability_operation_count_mismatch"

        separate_families = set()
        plan_families = {operation.family_id for operation in plan.operations}
        prepared_addresses: set[str] = set()
        proof_operation_rows: list[Mapping[str, Any]] = []
        for operation in plan.operations:
            contract = self.family_contracts.get(operation.family_id)
            if contract is None:
                return "unknown_family"
            if operation.write_kind not in contract.write_kinds:
                return "write_kind_family_mismatch"
            if not private_commit_grant and not contract.permits_execution(
                write_kind=operation.write_kind,
                proof_run=context.proof_run,
            ):
                return "write_kind_not_live_or_proof_allowed"
            adapter = self.adapters.get(operation.family_id)
            if adapter is None:
                return "family_adapter_unavailable"
            if adapter.family_id != operation.family_id:
                return "family_adapter_identity_mismatch"
            prepare = getattr(adapter, "prepare", None)
            if callable(prepare):
                try:
                    prepared = prepare(operation)
                except Exception as exc:
                    return (
                        f"operation_preflight_failed:{operation.operation_id}:"
                        f"{type(exc).__name__}:{exc}"
                    )
                addresses = tuple(
                    str(address)
                    for address in (prepared or {}).get("addresses", ())
                    if str(address)
                )
                if not addresses:
                    return f"operation_preflight_has_no_addresses:{operation.operation_id}"
                if len(addresses) != len(set(addresses)):
                    return f"operation_preflight_duplicate_address:{operation.operation_id}"
                overlap = prepared_addresses.intersection(addresses)
                if overlap:
                    return "transaction_address_overlap:" + ",".join(sorted(overlap))
                prepared_addresses.update(addresses)
                if capability_bundle is not None:
                    try:
                        proof_operation_rows.append(
                            canonicalize_attended_operation(
                                {
                                    "operation_id": operation.operation_id,
                                    "family_id": operation.family_id,
                                    "write_kind": operation.write_kind,
                                    "channel": operation.channel,
                                    "parameter": operation.parameter,
                                },
                                prepared,
                            )
                        )
                    except ValueError:
                        return "m110_capability_operation_canonicalization_failed"
            if contract.separate_approval:
                separate_families.add(contract.family_id)

        if separate_families and (
            len(separate_families) != 1 or len(plan_families) != 1
        ):
            return "high_risk_family_requires_separate_transaction"
        if capability_bundle is not None and proof_operation_rows != list(
            capability_bundle["operations"]
        ):
            return "m110_capability_exact_bundle_mismatch"
        return None

    def _generation_matches(self, context: ConsoleTransactionContext) -> bool:
        return int(self.emergency_generation_getter()) == int(
            context.emergency_generation
        )

    @staticmethod
    def _approval_is_current(context: ConsoleTransactionContext) -> bool:
        deadline = context.approval_deadline_epoch
        return bool(
            deadline is None
            or (
                isinstance(deadline, (int, float))
                and not isinstance(deadline, bool)
                and math.isfinite(float(deadline))
                and time.time() < float(deadline)
            )
        )

    async def _activate_emergency_stop(
        self,
        result: ConsoleTransactionResult,
        reason: str,
    ) -> None:
        latch_ok = self._latch_emergency_stop(result, reason)
        teardown_ok = await self._complete_emergency_stop(result, reason)
        result.emergency_stop_complete = bool(latch_ok and teardown_ok)

    def _latch_emergency_stop(
        self,
        result: ConsoleTransactionResult,
        reason: str,
    ) -> bool:
        """Synchronously revoke forward-write authority before recovery I/O."""

        try:
            latch_result = self.emergency_latch(reason)
            if asyncio.iscoroutine(latch_result) or hasattr(
                latch_result,
                "__await__",
            ):
                raise TypeError("emergency_latch_must_be_synchronous")
        except (asyncio.CancelledError, Exception) as exc:
            result.events.append(
                {
                    "phase": "emergency_latch",
                    "success": False,
                    "reason": "emergency_latch_failed",
                    "exception_type": type(exc).__name__,
                }
            )
            return False
        mapping_result = (
            dict(latch_result)
            if isinstance(latch_result, Mapping)
            else None
        )
        latch_ok = bool(
            latch_result is None
            or (
                mapping_result is not None
                and mapping_result.get("success") is True
                and mapping_result.get("status") == "latched"
                and not list(mapping_result.get("phase_zero_errors") or [])
            )
        )
        result.emergency_stop_activated = bool(
            latch_ok
            or (
                mapping_result is not None
                and mapping_result.get("status") == "latched"
            )
        )
        result.emergency_latch_succeeded = latch_ok
        result.events.append(
            {
                "phase": "emergency_latch",
                "success": latch_ok,
                "reason": (
                    reason if latch_ok else "emergency_latch_reported_incomplete"
                ),
                "reported_status": (
                    mapping_result.get("status")
                    if mapping_result is not None
                    else None
                ),
                "phase_zero_errors": (
                    list(mapping_result.get("phase_zero_errors") or [])
                    if mapping_result is not None
                    else []
                ),
            }
        )
        return latch_ok

    async def _complete_emergency_stop(
        self,
        result: ConsoleTransactionResult,
        reason: str,
    ) -> bool:
        async def invoke_teardown() -> Any:
            return await _maybe_await(self.emergency_stop(reason))

        teardown = asyncio.create_task(invoke_teardown())
        cancellation_deferred = False
        while not teardown.done():
            try:
                await asyncio.shield(teardown)
            except asyncio.CancelledError:
                cancellation_deferred = True
                continue
            except BaseException:
                break
        try:
            teardown_result = teardown.result()
        except BaseException as exc:
            result.events.append(
                {
                    "phase": "emergency_teardown",
                    "success": False,
                    "reason": "emergency_teardown_failed",
                    "exception_type": type(exc).__name__,
                }
            )
            result.emergency_teardown_succeeded = False
            result.emergency_stop_complete = False
            return False
        mapping_result = (
            dict(teardown_result)
            if isinstance(teardown_result, Mapping)
            else None
        )
        teardown_ok = bool(
            teardown_result is None
            or (
                mapping_result is not None
                and mapping_result.get("success") is True
                and mapping_result.get("status") == "stopped"
                and mapping_result.get("teardown_complete") is True
                and not list(mapping_result.get("teardown_errors") or [])
            )
        )
        result.events.append(
            {
                "phase": "emergency_teardown",
                "success": teardown_ok,
                "reason": (
                    "emergency_teardown_complete"
                    if teardown_ok
                    else "emergency_teardown_reported_incomplete"
                ),
                "reported_status": (
                    mapping_result.get("status")
                    if mapping_result is not None
                    else None
                ),
                "teardown_errors": (
                    list(mapping_result.get("teardown_errors") or [])
                    if mapping_result is not None
                    else []
                ),
                "caller_cancellation_deferred": cancellation_deferred,
            }
        )
        # The synchronous latch is the authoritative forward-write freeze;
        # teardown completion is tracked separately by the server.
        result.emergency_stop_activated = bool(
            result.emergency_stop_activated or teardown_ok
        )
        result.emergency_teardown_succeeded = teardown_ok
        result.emergency_stop_complete = bool(
            result.emergency_latch_succeeded and teardown_ok
        )
        return teardown_ok

    def _write_authority_is_current(
        self,
        plan: ConsoleTransactionPlan,
    ) -> bool:
        for operation in plan.operations:
            checker = getattr(
                self.adapters[operation.family_id],
                "write_authority_is_current",
                None,
            )
            if callable(checker) and checker() is not True:
                return False
        return True

    @staticmethod
    def _snapshot_values_equal(
        expected: Any,
        actual: Any,
        tolerance: float,
    ) -> bool:
        try:
            left = float(expected)
            right = float(actual)
            return bool(
                math.isfinite(left)
                and math.isfinite(right)
                and abs(left - right) <= max(0.0, float(tolerance))
            )
        except (TypeError, ValueError):
            return str(expected).strip().upper() == str(actual).strip().upper()

    async def _atomic_member_snapshot_still_current(
        self,
        operation: ConsoleTransactionOperation,
        original_snapshot: Mapping[str, Any],
    ) -> tuple[bool, Mapping[str, Any]]:
        adapter = self.adapters[operation.family_id]
        current = await adapter.snapshot(operation)
        if not isinstance(current, Mapping) or current.get("physical_verified") is not True:
            return False, dict(current or {})
        expected_values = dict(original_snapshot.get("values") or {})
        current_values = dict(current.get("values") or {})
        prepared = adapter.prepare(operation)
        specs = [
            *list(prepared.get("writes") or []),
            *list(prepared.get("audible_state_reads") or []),
        ]
        if not specs:
            return expected_values == current_values, dict(current)
        for spec in specs:
            address = str(spec.get("address") or "")
            if (
                not address
                or address not in expected_values
                or address not in current_values
                or not self._snapshot_values_equal(
                    expected_values[address],
                    current_values[address],
                    float(spec.get("tolerance") or 0.0),
                )
            ):
                return False, dict(current)
        return True, dict(current)

    async def _reverify_atomic_members(
        self,
        result: ConsoleTransactionResult,
        members: list[
            tuple[
                ConsoleTransactionOperation,
                Mapping[str, Any],
                AdapterStepResult,
            ]
        ],
    ) -> bool:
        for operation, snapshot, apply_result in members:
            adapter = self.adapters[operation.family_id]
            try:
                verification = await adapter.verify(
                    operation,
                    snapshot,
                    apply_result,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                verification = AdapterStepResult(
                    False,
                    False,
                    "atomic_group_reverify_exception",
                    {"exception_type": type(exc).__name__},
                )
            result.events.append(
                {
                    "operation_id": operation.operation_id,
                    "phase": "atomic_group_reverify",
                    "result": asdict(verification),
                }
            )
            if not verification.success or not verification.physical_verified:
                return False
        return True

    @staticmethod
    def _apply_result_may_have_written(result: AdapterStepResult) -> bool:
        evidence = dict(result.evidence or {})
        if evidence.get("write_state_unknown") is True:
            return True
        writes = evidence.get("writes")
        if isinstance(writes, list):
            return any(
                isinstance(write, Mapping)
                and write.get("sent") in {True, None}
                for write in writes
            )
        return result.success is True

    async def _recover_after_audio_safety_failure(
        self,
        result: ConsoleTransactionResult,
        applied: list[
            tuple[
                ConsoleTransactionOperation,
                Mapping[str, Any],
                AdapterStepResult,
            ]
        ],
        context: ConsoleTransactionContext,
    ) -> None:
        """Freeze writes and retain only physically proven attenuation targets.

        A normal rollback can be an acoustic gain increase when the applied
        operation was a fader/trim/send/EQ cut.  Once an audio-safety boundary
        has failed, such an increase is forbidden.  Family adapters classify
        the direction without writing and physically re-read eligible targets;
        ineligible operations retain the existing CAS rollback behavior.
        """

        emergency_reason = (
            "supervised_transaction_audio_safety_failed:"
            f"{result.transaction_id}"
        )
        latch_ok = self._latch_emergency_stop(result, emergency_reason)
        rollback_candidates: list[
            tuple[
                ConsoleTransactionOperation,
                Mapping[str, Any],
                AdapterStepResult,
            ]
        ] = []
        retention_seen = False
        retention_verified = True
        for operation, snapshot, apply_result in applied:
            adapter = self.adapters[operation.family_id]
            assessor = getattr(
                adapter,
                "assess_audio_failure_retention",
                None,
            )
            if not callable(assessor):
                rollback_candidates.append(
                    (operation, snapshot, apply_result)
                )
                continue
            try:
                assessment = await assessor(
                    operation,
                    snapshot,
                    apply_result,
                )
            except asyncio.CancelledError:
                assessment = AdapterStepResult(
                    False,
                    False,
                    "audio_failure_retention_cancelled",
                    {
                        "retention_eligible": None,
                        "target_physical_verified": False,
                        "cancellation_requested": True,
                        "write_attempted": False,
                    },
                )
            except Exception as exc:
                assessment = AdapterStepResult(
                    False,
                    False,
                    "audio_failure_retention_exception",
                    {
                        "retention_eligible": None,
                        "target_physical_verified": False,
                        "exception_type": type(exc).__name__,
                        "write_attempted": False,
                    },
                )
            evidence = dict(assessment.evidence or {})
            result.events.append(
                {
                    "operation_id": operation.operation_id,
                    "phase": "audio_failure_retention",
                    "result": asdict(assessment),
                }
            )
            eligible = evidence.get("retention_eligible")
            if eligible is False:
                rollback_candidates.append(
                    (operation, snapshot, apply_result)
                )
                continue
            # An unknown classification must never fall through to a blind
            # rollback that could raise level.  Freeze and require review.
            retention_seen = True
            result.safe_target_retention_attempted = True
            if (
                eligible is True
                and assessment.success
                and assessment.physical_verified
                and evidence.get("target_physical_verified") is True
            ):
                result.contained_operation_ids.append(
                    operation.operation_id
                )
            else:
                retention_verified = False
                result.retention_unverified_operation_ids.append(
                    operation.operation_id
                )

        rollback_state_known = True
        if rollback_candidates:
            await self._rollback_applied(
                result,
                rollback_candidates,
                context,
            )
            rollback_state_known = result.status == "rolled_back"

        if retention_seen:
            contained_verified = bool(
                retention_verified
                and result.contained_operation_ids
                and not result.retention_unverified_operation_ids
            )
            result.contained_target_physical_verified = contained_verified
            result.audio_failure_recovery_state_known = bool(
                contained_verified and rollback_state_known
            )
            result.status = "manual_review_required"
        else:
            result.audio_failure_recovery_state_known = rollback_state_known
        teardown_ok = await self._complete_emergency_stop(
            result,
            emergency_reason,
        )
        if not latch_ok or not teardown_ok:
            result.status = "manual_review_required"

    async def execute(
        self,
        plan: ConsoleTransactionPlan,
        context: ConsoleTransactionContext,
    ) -> ConsoleTransactionResult:
        async with self._lock:
            validation_error = self._validate_plan(plan, context)
            if validation_error:
                return self._blocked(plan, validation_error)
            if context.idempotency_key in self._idempotency_keys:
                return self._blocked(plan, "duplicate_idempotency_key")
            if not self._generation_matches(context):
                return self._blocked(plan, "emergency_generation_changed")
            if not self._approval_is_current(context):
                return self._blocked(plan, "operator_approval_expired")
            self._idempotency_keys[context.idempotency_key] = plan.transaction_id

            result = ConsoleTransactionResult(
                transaction_id=plan.transaction_id,
                status="running",
                reason="transaction_started",
            )
            grant = context.capability_grant
            authenticated_private_lane = bool(
                type(grant)
                in {
                    RestoreOnlyProofGrant,
                    MaintenanceDspResetCommitOnceGrant,
                    ProcessorEnableCommitOnceGrant,
                    SoundcheckChildCommitOnceGrant,
                }
                and grant.is_authentic()
            )
            applied: list[
                tuple[
                    ConsoleTransactionOperation,
                    Mapping[str, Any],
                    AdapterStepResult,
                ]
            ] = []

            async def abort_before_next_apply(
                reason: str,
                *,
                cancellation_safe: bool = False,
                audio_safety_failure: bool = False,
            ) -> None:
                """Contain a durable-boundary abort after earlier operations."""

                result.reason = reason
                if not applied:
                    result.status = "blocked"
                    if audio_safety_failure:
                        await self._recover_after_audio_safety_failure(
                            result,
                            [],
                            context,
                        )
                    return
                recovery = (
                    self._recover_after_audio_safety_failure(
                        result,
                        applied,
                        context,
                    )
                    if audio_safety_failure
                    else self._rollback_applied(result, applied, context)
                )
                if not cancellation_safe and not audio_safety_failure:
                    await recovery
                    return
                rollback_task = asyncio.create_task(
                    recovery
                )
                while not rollback_task.done():
                    try:
                        await asyncio.shield(rollback_task)
                    except asyncio.CancelledError:
                        result.events.append(
                            {
                                "phase": "rollback_cancellation_deferred",
                                "reason": reason,
                            }
                        )
                        continue
                await rollback_task
            already_satisfied_operation_ids: set[str] = set()
            atomic_group = bool(
                len(plan.operations) >= 2
                and all(
                    operation.metadata.get("atomic_group_commit_required") is True
                    for operation in plan.operations
                )
            )
            processed_atomic_members: list[
                tuple[
                    ConsoleTransactionOperation,
                    Mapping[str, Any],
                    AdapterStepResult,
                ]
            ] = []
            proof_audio_passed_operation_ids: set[str] = set()

            snapshots: list[Mapping[str, Any]] = []
            for operation in plan.operations:
                if not self._approval_is_current(context):
                    result.status = "blocked"
                    result.reason = "operator_approval_expired_during_snapshot"
                    return result
                adapter = self.adapters[operation.family_id]
                try:
                    snapshot = await adapter.snapshot(operation)
                except asyncio.CancelledError:
                    result.status = "blocked"
                    result.reason = "physical_snapshot_cancelled"
                    result.events.append(
                        {
                            "operation_id": operation.operation_id,
                            "phase": "snapshot",
                            "reason": "cancelled",
                        }
                    )
                    return result
                except Exception as exc:
                    result.status = "blocked"
                    result.reason = "physical_snapshot_failed"
                    result.events.append(
                        {
                            "operation_id": operation.operation_id,
                            "phase": "snapshot",
                            "reason": type(exc).__name__,
                        }
                    )
                    return result
                if not snapshot or snapshot.get("physical_verified") is not True:
                    result.status = "blocked"
                    result.reason = "physical_snapshot_unverified"
                    result.events.append(
                        {
                            "operation_id": operation.operation_id,
                            "phase": "snapshot",
                            "snapshot": dict(snapshot or {}),
                        }
                    )
                    return result
                result.events.append(
                    {
                        "operation_id": operation.operation_id,
                        "phase": "snapshot",
                        "snapshot": dict(snapshot),
                    }
                )
                validate_snapshot = getattr(adapter, "validate_snapshot", None)
                if callable(validate_snapshot):
                    transition_error = validate_snapshot(operation, snapshot)
                    if (
                        transition_error
                        == "physical_target_already_satisfied"
                        and (
                            operation.metadata.get(
                                "atomic_group_commit_required"
                            )
                            is True
                            or authenticated_private_lane
                        )
                    ):
                        already_satisfied_operation_ids.add(
                            operation.operation_id
                        )
                        result.events.append(
                            {
                                "operation_id": operation.operation_id,
                                "phase": "snapshot_transition_validation",
                                "reason": transition_error,
                                "result": "already_satisfied_private_member",
                            }
                        )
                        transition_error = None
                    if transition_error:
                        result.status = "blocked"
                        result.reason = transition_error
                        result.events.append(
                            {
                                "operation_id": operation.operation_id,
                                "phase": "snapshot_transition_validation",
                                "reason": transition_error,
                            }
                        )
                        return result
                if (
                    atomic_group or authenticated_private_lane
                ) and operation.operation_id not in already_satisfied_operation_ids:
                    prepared = adapter.prepare(operation)
                    prepared_writes = list(prepared.get("writes") or [])
                    prepared_reads = list(
                        prepared.get("audible_state_reads") or []
                    )
                    snapshot_values = dict(snapshot.get("values") or {})
                    target_already_satisfied = bool(
                        prepared_writes
                        and all(
                            self._snapshot_values_equal(
                                spec.get("target_value"),
                                snapshot_values.get(str(spec.get("address") or "")),
                                float(spec.get("tolerance") or 0.0),
                            )
                            for spec in prepared_writes
                        )
                        and all(
                            self._snapshot_values_equal(
                                spec.get("expected_value"),
                                snapshot_values.get(str(spec.get("address") or "")),
                                float(spec.get("tolerance") or 0.0),
                            )
                            for spec in prepared_reads
                        )
                    )
                    if target_already_satisfied:
                        already_satisfied_operation_ids.add(
                            operation.operation_id
                        )
                        result.events.append(
                            {
                                "operation_id": operation.operation_id,
                                "phase": "snapshot_transition_validation",
                                "reason": "physical_target_already_satisfied",
                                "result": "already_satisfied_private_member",
                            }
                        )
                snapshots.append(snapshot)

            if not self._approval_is_current(context):
                result.status = "blocked"
                result.reason = "operator_approval_expired_after_snapshot"
                return result

            if self.prepared_callback is not None:
                try:
                    prepared_result = await _maybe_await(
                        self.prepared_callback(
                            plan,
                            context,
                            tuple(snapshots),
                        )
                    )
                except asyncio.CancelledError:
                    result.status = "blocked"
                    result.reason = "durable_prepare_cancelled"
                    return result
                except Exception as exc:
                    result.status = "blocked"
                    result.reason = "durable_prepare_failed"
                    result.events.append(
                        {
                            "phase": "durable_prepare",
                            "reason": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    return result
                if prepared_result is False or prepared_result is None:
                    result.status = "blocked"
                    result.reason = "durable_prepare_failed"
                    return result
                result.events.append(
                    {
                        "phase": "durable_prepare",
                        "result": "persisted",
                    }
                )

            if not self._approval_is_current(context):
                result.status = "blocked"
                result.reason = "operator_approval_expired_before_apply"
                return result

            for operation, snapshot in zip(plan.operations, snapshots):
                adapter = self.adapters[operation.family_id]
                if atomic_group and processed_atomic_members:
                    try:
                        atomic_members_current = await self._reverify_atomic_members(
                            result,
                            processed_atomic_members,
                        )
                    except asyncio.CancelledError:
                        await abort_before_next_apply(
                            "atomic_group_reverify_cancelled",
                            cancellation_safe=True,
                        )
                        return result
                    if not atomic_members_current:
                        result.reason = "atomic_group_member_drift_detected"
                        if applied:
                            await self._rollback_applied(result, applied, context)
                        else:
                            result.status = "blocked"
                        return result
                if atomic_group:
                    try:
                        snapshot_current, current_snapshot = (
                            await self._atomic_member_snapshot_still_current(
                                operation,
                                snapshot,
                            )
                        )
                    except asyncio.CancelledError:
                        await abort_before_next_apply(
                            "atomic_group_preapply_snapshot_cancelled",
                            cancellation_safe=True,
                        )
                        return result
                    except Exception as exc:
                        snapshot_current, current_snapshot = False, {
                            "exception_type": type(exc).__name__,
                        }
                    result.events.append(
                        {
                            "operation_id": operation.operation_id,
                            "phase": "atomic_group_preapply_snapshot_cas",
                            "snapshot_current": snapshot_current,
                            "snapshot": dict(current_snapshot),
                        }
                    )
                    if not snapshot_current:
                        result.reason = "atomic_group_member_changed_before_apply"
                        if applied:
                            await self._rollback_applied(result, applied, context)
                        else:
                            result.status = "blocked"
                        return result
                if (
                    operation.operation_id
                    in already_satisfied_operation_ids
                ):
                    result.already_satisfied_operation_ids.append(
                        operation.operation_id
                    )
                    already_satisfied_result = AdapterStepResult(
                        True,
                        True,
                        "physical_target_already_satisfied",
                        {"write_attempted": False},
                    )
                    result.events.append(
                        {
                            "operation_id": operation.operation_id,
                            "phase": "apply",
                            "result": {
                                "skipped": True,
                                "reason": (
                                    "physical_target_already_satisfied"
                                ),
                                "physical_verified": True,
                            },
                        }
                    )
                    try:
                        audio_result = await adapter.audio_recheck(
                            operation,
                            snapshot,
                            already_satisfied_result,
                        )
                    except asyncio.CancelledError:
                        audio_result = AudioRecheckResult(
                            "failed",
                            "audio_recheck_cancelled",
                            {"cancellation_requested": True},
                        )
                    except Exception as exc:
                        audio_result = AudioRecheckResult(
                            "failed",
                            "audio_recheck_exception",
                            {"exception_type": type(exc).__name__},
                        )
                    result.events.append(
                        {
                            "operation_id": operation.operation_id,
                            "phase": "audio_recheck",
                            "result": asdict(audio_result),
                        }
                    )
                    if audio_result.status == "passed":
                        proof_audio_passed_operation_ids.add(
                            operation.operation_id
                        )
                    contract = self.family_contracts[operation.family_id]
                    audio_ok = audio_result.passed
                    if contract.audio_recheck_policy == AUDIO_RECHECK_REQUIRED:
                        audio_ok = audio_result.status == "passed"
                    elif (
                        contract.audio_recheck_policy
                        == AUDIO_RECHECK_NOT_APPLICABLE
                    ):
                        audio_ok = audio_result.status in {
                            "passed",
                            "not_applicable",
                        }
                    if context.proof_run is True:
                        audio_ok = audio_result.status == "passed"
                    if not audio_ok:
                        await abort_before_next_apply(
                            audio_result.reason or "audio_recheck_failed",
                            audio_safety_failure=True,
                        )
                        return result
                    if not self._generation_matches(context):
                        await abort_before_next_apply(
                            "emergency_generation_changed_after_audio_recheck",
                            audio_safety_failure=True,
                        )
                        return result
                    if not self._write_authority_is_current(plan):
                        await abort_before_next_apply(
                            "write_authority_revoked_after_audio_recheck",
                            audio_safety_failure=True,
                        )
                        return result
                    processed_atomic_members.append(
                        (
                            operation,
                            snapshot,
                            already_satisfied_result,
                        )
                    )
                    continue
                if not self._generation_matches(context):
                    result.status = "blocked"
                    result.reason = "emergency_generation_changed_before_apply"
                    if applied:
                        await self._rollback_applied(result, applied, context)
                    return result
                if not self._write_authority_is_current(plan):
                    result.reason = "write_authority_revoked_before_apply"
                    if applied:
                        await self._rollback_applied(result, applied, context)
                    else:
                        result.status = "blocked"
                    return result
                if not self._approval_is_current(context):
                    result.reason = "operator_approval_expired_before_apply"
                    if applied:
                        await self._rollback_applied(result, applied, context)
                    else:
                        result.status = "blocked"
                    return result

                if self.before_operation_apply_callback is not None:
                    try:
                        operation_boundary_result = await _maybe_await(
                            self.before_operation_apply_callback(operation)
                        )
                    except asyncio.CancelledError:
                        await abort_before_next_apply(
                            "operation_send_safety_boundary_cancelled",
                            cancellation_safe=True,
                            audio_safety_failure=True,
                        )
                        return result
                    except Exception as exc:
                        result.events.append(
                            {
                                "operation_id": operation.operation_id,
                                "phase": "operation_send_safety_boundary",
                                "reason": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                        await abort_before_next_apply(
                            str(exc).strip()
                            or "operation_send_safety_boundary_failed",
                            audio_safety_failure=True,
                        )
                        return result
                    if operation_boundary_result is not True:
                        await abort_before_next_apply(
                            "operation_send_safety_boundary_failed",
                            audio_safety_failure=True,
                        )
                        return result
                    if not self._generation_matches(context):
                        await abort_before_next_apply(
                            "emergency_generation_changed_after_operation_safety",
                            audio_safety_failure=True,
                        )
                        return result
                    if not self._write_authority_is_current(plan):
                        await abort_before_next_apply(
                            "write_authority_revoked_after_operation_safety",
                            audio_safety_failure=True,
                        )
                        return result

                if self.before_apply_callback is not None:
                    try:
                        boundary_result = await _maybe_await(
                            self.before_apply_callback()
                        )
                    except asyncio.CancelledError:
                        await abort_before_next_apply(
                            "durable_send_boundary_cancelled",
                            cancellation_safe=True,
                        )
                        return result
                    except Exception as exc:
                        result.events.append(
                            {
                                "operation_id": operation.operation_id,
                                "phase": "durable_send_boundary",
                                "reason": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                        await abort_before_next_apply(
                            "durable_send_boundary_failed"
                        )
                        return result
                    if boundary_result is not True:
                        await abort_before_next_apply(
                            "durable_send_boundary_failed"
                        )
                        return result
                    result.events.append(
                        {
                            "operation_id": operation.operation_id,
                            "phase": "durable_send_boundary",
                            "result": "send_may_have_occurred_persisted",
                        }
                    )
                    if not self._approval_is_current(context):
                        await abort_before_next_apply(
                            "operator_approval_expired_after_durable_send_boundary"
                        )
                        return result
                    if not self._generation_matches(context):
                        await abort_before_next_apply(
                            "emergency_generation_changed_after_durable_send_boundary"
                        )
                        return result
                    if not self._write_authority_is_current(plan):
                        await abort_before_next_apply(
                            "write_authority_revoked_after_durable_send_boundary"
                        )
                        return result

                try:
                    apply_result = await adapter.apply(operation, snapshot)
                except asyncio.CancelledError:
                    apply_result = AdapterStepResult(
                        False,
                        False,
                        "apply_cancelled",
                        {
                            "write_state_unknown": True,
                            "cancellation_requested": True,
                        },
                    )
                except Exception as exc:
                    apply_result = AdapterStepResult(
                        False,
                        False,
                        "apply_exception",
                        {
                            "exception_type": type(exc).__name__,
                            "write_state_unknown": True,
                        },
                    )
                result.events.append(
                    {
                        "operation_id": operation.operation_id,
                        "phase": "apply",
                        "result": asdict(apply_result),
                    }
                )
                # The current operation enters the recovery set as soon as its
                # adapter is invoked. A multi-address adapter can mutate one
                # parameter before a later send fails or raises.
                rollback_anchor = AdapterStepResult(
                    apply_result.success,
                    apply_result.physical_verified,
                    apply_result.reason,
                    dict(apply_result.evidence),
                )
                current_may_have_written = self._apply_result_may_have_written(
                    apply_result
                )
                if current_may_have_written:
                    applied.append((operation, snapshot, rollback_anchor))
                if not apply_result.success:
                    result.reason = apply_result.reason or "apply_failed"
                    if applied:
                        await self._rollback_applied(result, applied, context)
                    else:
                        result.status = "blocked"
                    return result

                if not self._generation_matches(context):
                    result.reason = "emergency_generation_changed_after_apply"
                    await self._rollback_applied(result, applied, context)
                    return result
                if not self._write_authority_is_current(plan):
                    result.reason = "write_authority_revoked_after_apply"
                    await self._rollback_applied(result, applied, context)
                    return result

                try:
                    verify_result = await adapter.verify(
                        operation,
                        snapshot,
                        apply_result,
                    )
                except asyncio.CancelledError:
                    verify_result = AdapterStepResult(
                        False,
                        False,
                        "verify_cancelled",
                        {"cancellation_requested": True},
                    )
                except Exception as exc:
                    verify_result = AdapterStepResult(
                        False,
                        False,
                        "verify_exception",
                        {"exception_type": type(exc).__name__},
                    )
                result.events.append(
                    {
                        "operation_id": operation.operation_id,
                        "phase": "verify",
                        "result": asdict(verify_result),
                    }
                )
                rollback_anchor = AdapterStepResult(
                    apply_result.success,
                    apply_result.physical_verified,
                    apply_result.reason,
                    {
                        **dict(apply_result.evidence),
                        "verified_values": dict(verify_result.evidence.get("values") or {}),
                    },
                )
                if current_may_have_written:
                    applied[-1] = (operation, snapshot, rollback_anchor)
                result.applied_operation_ids.append(operation.operation_id)
                if not verify_result.success or not verify_result.physical_verified:
                    result.reason = verify_result.reason or "physical_readback_unverified"
                    await self._rollback_applied(result, applied, context)
                    return result

                contract = self.family_contracts[operation.family_id]
                if not self._generation_matches(context):
                    result.reason = "emergency_generation_changed_after_verify"
                    await self._rollback_applied(result, applied, context)
                    return result

                try:
                    audio_result = await adapter.audio_recheck(
                        operation,
                        snapshot,
                        apply_result,
                    )
                except asyncio.CancelledError:
                    audio_result = AudioRecheckResult(
                        "failed",
                        "audio_recheck_cancelled",
                        {"cancellation_requested": True},
                    )
                except Exception as exc:
                    audio_result = AudioRecheckResult(
                        "failed",
                        "audio_recheck_exception",
                        {"exception_type": type(exc).__name__},
                    )
                result.events.append(
                    {
                        "operation_id": operation.operation_id,
                        "phase": "audio_recheck",
                        "result": asdict(audio_result),
                    }
                )
                if audio_result.status == "passed":
                    proof_audio_passed_operation_ids.add(
                        operation.operation_id
                    )
                audio_ok = audio_result.passed
                if contract.audio_recheck_policy == AUDIO_RECHECK_REQUIRED:
                    audio_ok = audio_result.status == "passed"
                elif contract.audio_recheck_policy == AUDIO_RECHECK_NOT_APPLICABLE:
                    audio_ok = audio_result.status in {"passed", "not_applicable"}
                if context.proof_run is True:
                    audio_ok = audio_result.status == "passed"
                if not audio_ok:
                    result.reason = (
                        audio_result.reason or "audio_recheck_failed"
                    )
                    recovery_task = asyncio.create_task(
                        self._recover_after_audio_safety_failure(
                            result,
                            applied,
                            context,
                        )
                    )
                    while not recovery_task.done():
                        try:
                            await asyncio.shield(recovery_task)
                        except asyncio.CancelledError:
                            result.events.append(
                                {
                                    "phase": "audio_failure_recovery_cancellation_deferred",
                                    "reason": result.reason,
                                }
                            )
                            continue
                    await recovery_task
                    return result

                if not self._generation_matches(context):
                    result.reason = "emergency_generation_changed_after_audio_recheck"
                    await self._recover_after_audio_safety_failure(
                        result,
                        applied,
                        context,
                    )
                    return result
                if not self._write_authority_is_current(plan):
                    result.reason = "write_authority_revoked_after_audio_recheck"
                    await self._recover_after_audio_safety_failure(
                        result,
                        applied,
                        context,
                    )
                    return result
                if atomic_group:
                    processed_atomic_members.append(
                        (operation, snapshot, rollback_anchor)
                    )

            if context.restore_after_verify:
                result.reason = "proof_verified_restoring_original_state"
                expected_operation_ids = {
                    operation.operation_id for operation in plan.operations
                }
                already_satisfied_ids = set(
                    result.already_satisfied_operation_ids
                )
                required_mutation_ids = (
                    expected_operation_ids - already_satisfied_ids
                )
                exact_apply_and_audio_coverage = bool(
                    required_mutation_ids
                    and already_satisfied_ids.issubset(expected_operation_ids)
                    and required_mutation_ids.union(already_satisfied_ids)
                    == expected_operation_ids
                    and set(result.applied_operation_ids)
                    == required_mutation_ids
                    and len(result.applied_operation_ids)
                    == len(required_mutation_ids)
                    and proof_audio_passed_operation_ids
                    == expected_operation_ids
                )
                exact_already_satisfied_coverage = bool(
                    not required_mutation_ids
                    and already_satisfied_ids == expected_operation_ids
                    and not result.applied_operation_ids
                    and not result.rollback_operation_ids
                    and proof_audio_passed_operation_ids
                    == expected_operation_ids
                )
                if exact_already_satisfied_coverage:
                    try:
                        all_members_current = await self._reverify_atomic_members(
                            result,
                            processed_atomic_members,
                        )
                    except asyncio.CancelledError:
                        result.status = "blocked"
                        result.reason = (
                            "proof_already_satisfied_reverify_cancelled"
                        )
                        return result
                    if not all_members_current:
                        result.status = "blocked"
                        result.reason = (
                            "proof_already_satisfied_member_drift_detected"
                        )
                        return result
                    result.status = "proof_passed"
                    result.reason = (
                        "proof_already_satisfied_two_pass_readback_verified"
                    )
                    return result
                await self._rollback_applied(result, applied, context)
                exact_rollback_coverage = bool(
                    set(result.rollback_operation_ids)
                    == required_mutation_ids
                    and len(result.rollback_operation_ids)
                    == len(required_mutation_ids)
                )
                if (
                    result.status == "rolled_back"
                    and exact_apply_and_audio_coverage
                    and exact_rollback_coverage
                ):
                    result.status = "proof_passed"
                    result.reason = (
                        "proof_apply_readback_audio_and_rollback_verified"
                    )
                elif result.status == "rolled_back":
                    result.reason = "proof_operation_coverage_incomplete"
                return result

            if not self._generation_matches(context):
                result.reason = "emergency_generation_changed_before_commit"
                await self._rollback_applied(result, applied, context)
                return result

            if atomic_group or (
                authenticated_private_lane
                and already_satisfied_operation_ids
            ):
                try:
                    atomic_members_current = await self._reverify_atomic_members(
                        result,
                        processed_atomic_members,
                    )
                except asyncio.CancelledError:
                    await abort_before_next_apply(
                        "atomic_group_final_reverify_cancelled",
                        cancellation_safe=True,
                    )
                    return result
                if not atomic_members_current:
                    result.reason = "atomic_group_member_drift_detected_before_commit"
                    await self._rollback_applied(result, applied, context)
                    return result

            if not self._approval_is_current(context):
                result.reason = "operator_approval_expired_before_commit"
                await self._rollback_applied(result, applied, context)
                return result
            if not self._write_authority_is_current(plan):
                result.reason = "write_authority_revoked_before_commit"
                await self._rollback_applied(result, applied, context)
                return result

            result.status = "applied"
            result.reason = "transaction_applied_and_verified"
            return result

    async def _rollback_applied(
        self,
        result: ConsoleTransactionResult,
        applied: list[
            tuple[
                ConsoleTransactionOperation,
                Mapping[str, Any],
                AdapterStepResult,
            ]
        ],
        context: ConsoleTransactionContext,
    ) -> None:
        rollback_failed = False
        for operation, snapshot, apply_result in reversed(applied):
            if not self._generation_matches(context):
                result.events.append(
                    {
                        "operation_id": operation.operation_id,
                        "phase": "rollback_generation_check",
                        "reason": "emergency_generation_changed_continuing_cas_recovery",
                        "compare_and_swap_required": True,
                    }
                )
                # A generation change blocks new writes, but a verified
                # compare-and-swap rollback is recovery, not new authority.
                # The adapter will refuse to overwrite any externally changed
                # value while still restoring parameters this transaction left
                # at its own target.
            adapter = self.adapters[operation.family_id]
            try:
                rollback_result = await adapter.rollback(
                    operation,
                    snapshot,
                    apply_result,
                )
            except asyncio.CancelledError:
                rollback_result = AdapterStepResult(
                    False,
                    False,
                    "rollback_cancelled",
                    {
                        "cancellation_requested": True,
                        "write_state_unknown": True,
                    },
                )
            except Exception as exc:
                rollback_result = AdapterStepResult(
                    False,
                    False,
                    "rollback_exception",
                    {"exception_type": type(exc).__name__},
                )
            result.events.append(
                {
                    "operation_id": operation.operation_id,
                    "phase": "rollback",
                    "result": asdict(rollback_result),
                }
            )
            if rollback_result.success and rollback_result.physical_verified:
                result.rollback_operation_ids.append(operation.operation_id)
            else:
                rollback_failed = True
                # Keep restoring earlier operations. One externally changed or
                # unverified parameter must not strand the rest of the plan.
                continue

        if rollback_failed:
            result.status = "manual_review_required"
            await self._activate_emergency_stop(
                result,
                f"supervised_transaction_rollback_unverified:{result.transaction_id}",
            )
        else:
            result.status = "rolled_back" if applied else "blocked"
