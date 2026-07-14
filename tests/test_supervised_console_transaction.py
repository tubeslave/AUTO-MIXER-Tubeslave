from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace

import pytest

from supervised_console_parameter_registry import (
    AUDIO_RECHECK_REQUIRED,
    LIFECYCLE_LIVE_ALLOWED,
    LIFECYCLE_PROOF_CANDIDATE,
    SupervisedConsoleFamilyContract,
)
from supervised_console_transaction import (
    AdapterStepResult,
    AudioRecheckResult,
    ConsoleTransactionContext,
    ConsoleTransactionOperation,
    ConsoleTransactionPlan,
    SupervisedConsoleTransactionCoordinator,
)
from supervised_family_proof_plan import (
    build_canonical_operation_bundle_from_prepared,
    canonical_bundle_scope,
    canonical_operation_bundle_bytes,
    canonical_operation_bundle_digest,
)
from supervised_proof_capability import (
    M110CapabilityBinding,
    SupervisedCapabilityStore,
)


class FakeAdapter:
    def __init__(self, family_id: str, *, fail_verify: bool = False, fail_rollback: bool = False):
        self.family_id = family_id
        self.fail_verify = fail_verify
        self.fail_rollback = fail_rollback
        self.calls = []

    def prepare(self, operation):
        if operation.target_value == "invalid":
            raise ValueError("synthetic_invalid_target")
        address = f"/{self.family_id}/{operation.operation_id}"
        return {
            "addresses": (address,),
            "write_count": 1,
            "writes": (
                {
                    "address": address,
                    "target_value": operation.target_value,
                    "tolerance": 0.01,
                },
            ),
            "audible_state_reads": (),
        }

    def validate_snapshot(self, operation, snapshot):
        return None

    async def snapshot(self, operation):
        self.calls.append(("snapshot", operation.operation_id))
        return {"physical_verified": True, "old_value": 0.0}

    async def apply(self, operation, snapshot):
        self.calls.append(("apply", operation.operation_id))
        return AdapterStepResult(True, True, "sent", {"target": operation.target_value})

    async def verify(self, operation, snapshot, apply_result):
        self.calls.append(("verify", operation.operation_id))
        if self.fail_verify:
            return AdapterStepResult(False, False, "readback_mismatch")
        return AdapterStepResult(True, True, "verified")

    async def rollback(self, operation, snapshot, apply_result):
        self.calls.append(("rollback", operation.operation_id))
        if self.fail_rollback:
            return AdapterStepResult(False, False, "rollback_mismatch")
        return AdapterStepResult(True, True, "rollback_verified")

    async def audio_recheck(self, operation, snapshot, apply_result):
        self.calls.append(("audio", operation.operation_id))
        return AudioRecheckResult("passed")


def contract(
    family_id: str,
    write_kind: str,
    *,
    lifecycle: str = LIFECYCLE_LIVE_ALLOWED,
    separate_approval: bool = False,
):
    return SupervisedConsoleFamilyContract(
        family_id=family_id,
        label=family_id,
        write_kinds=(write_kind,),
        risk_class="high" if separate_approval else "low",
        lifecycle=lifecycle,
        separate_approval=separate_approval,
        audio_recheck_policy=AUDIO_RECHECK_REQUIRED,
        hardware_proof="accepted" if lifecycle == LIFECYCLE_LIVE_ALLOWED else "missing",
        live_allowed_write_kinds=(write_kind,) if lifecycle == LIFECYCLE_LIVE_ALLOWED else (),
    )


def operation(operation_id: str, family_id: str = "eq_hpf", write_kind: str = "eq"):
    return ConsoleTransactionOperation(
        operation_id=operation_id,
        family_id=family_id,
        write_kind=write_kind,
        target_value=-2.0,
        channel=3,
    )


def context(
    *,
    proof_run: bool = False,
    restore_after_verify: bool = False,
    key: str = "txn-key",
    capability_grant=None,
):
    return ConsoleTransactionContext(
        approved=True,
        approval_id="operator-approval",
        idempotency_key=key,
        emergency_generation=7,
        proof_run=proof_run,
        restore_after_verify=restore_after_verify,
        capability_grant=capability_grant,
    )


def restore_only_grant(
    plan,
    adapter,
    *,
    key="proof-restore",
    capability_kind="restore_only_proof",
):
    safety_binding = {
        "adapter_revision": "unit-adapter-v1",
        "protocol_revision": "unit-protocol-v1",
        "runtime_instance_nonce": "1" * 32,
        "runtime_manifest_sha256": "sha256:" + "2" * 64,
        "source_head": "3" * 40,
        "source_tree": "4" * 40,
        "console_serial": "unit-console",
        "firmware_version": "3.0.5",
        "route_proof_id": "unit-route",
        "route_artifact_sha256": "sha256:" + "5" * 64,
        "capture_hardware_uid": "unit-capture",
        "connection_generation": 1,
        "transport_generation": 2,
        "emergency_generation": 7,
    }
    operation_rows = [
        {
            "operation_id": item.operation_id,
            "family_id": item.family_id,
            "write_kind": item.write_kind,
            "channel": item.channel,
            "parameter": item.parameter,
        }
        for item in plan.operations
    ]
    bundle = build_canonical_operation_bundle_from_prepared(
        bundle_kind=(
            "single_soundcheck_operation"
            if len(plan.operations) == 1
            else "soundcheck_atomic_group"
        ),
        operations=operation_rows,
        prepared_operations=[adapter.prepare(item) for item in plan.operations],
        safety_binding=safety_binding,
    )
    bundle_bytes = canonical_operation_bundle_bytes(bundle)
    scope, targets = canonical_bundle_scope(bundle)
    owner = object()
    route = object()
    transport = object()
    capture = object()
    control = object()
    proof_receipt = object()
    binding = M110CapabilityBinding(
        capability_kind=capability_kind,
        runtime_instance_nonce="1" * 32,
        runtime_manifest_sha256="sha256:" + "2" * 64,
        launch_manifest_sha256="sha256:" + "6" * 64,
        source_head="3" * 40,
        source_tree="4" * 40,
        connection_generation=1,
        transport_generation=2,
        console_serial="unit-console",
        firmware_version="3.0.5",
        route_proof_id="unit-route",
        route_artifact_sha256="sha256:" + "5" * 64,
        capture_hardware_uid="unit-capture",
        emergency_generation=7,
        owner_connection_id=id(owner),
        proposal_id="unit-proposal",
        review_contract_version=2,
        review_content_digest="sha256:" + "7" * 64,
        operation_digest=canonical_operation_bundle_digest(bundle),
        canonical_bundle_bytes=bundle_bytes,
        bundle_kind=bundle["bundle_kind"],
        approval_id="operator-approval",
        idempotency_key=key,
        scope_addresses=scope,
        target_fingerprints=targets,
        proof_receipt_id=(
            "m110proof_v1_" + "9" * 64
            if capability_kind != "restore_only_proof"
            else ""
        ),
        _owner_ref=owner,
        _route_ref=route,
        _transport_ref=transport,
        _capture_ref=capture,
        _control_ref=(
            control if capability_kind != "restore_only_proof" else None
        ),
        _proof_receipt_ref=(
            proof_receipt
            if capability_kind != "restore_only_proof"
            else None
        ),
    )
    store = SupervisedCapabilityStore(token_factory=lambda: "8" * 64)
    issued, _ = store.issue(binding)
    grant, _ = store.consume(issued.token, binding)
    return grant


def coordinator(
    contracts,
    adapters,
    generation=lambda: 7,
    emergency=None,
    emergency_latch=None,
    prepared_callback=None,
    before_operation_apply_callback=None,
    before_apply_callback=None,
):
    emergency_calls = []

    async def stop(reason):
        emergency_calls.append(reason)

    def latch(_reason):
        return {"status": "latched", "success": True}

    return (
        SupervisedConsoleTransactionCoordinator(
            family_contracts=contracts,
            adapters=adapters,
            emergency_generation_getter=generation,
            emergency_latch=emergency_latch or latch,
            emergency_stop=emergency or stop,
            prepared_callback=prepared_callback,
            before_operation_apply_callback=(
                before_operation_apply_callback
            ),
            before_apply_callback=before_apply_callback,
        ),
        emergency_calls,
    )


class AudioFailureRetentionAdapter(FakeAdapter):
    def __init__(self, family_id: str, *, retention_verified: bool = True):
        super().__init__(family_id)
        self.retention_verified = retention_verified
        self.state = 0.0

    async def snapshot(self, operation):
        self.calls.append(("snapshot", operation.operation_id))
        return {
            "physical_verified": True,
            "values": {
                f"/{self.family_id}/{operation.operation_id}": self.state
            },
        }

    async def apply(self, operation, snapshot):
        self.calls.append(("apply", operation.operation_id))
        self.state = float(operation.target_value)
        return AdapterStepResult(
            True,
            True,
            "sent",
            {
                "writes": [
                    {
                        "address": f"/{self.family_id}/{operation.operation_id}",
                        "value": operation.target_value,
                        "sent": True,
                    }
                ]
            },
        )

    async def audio_recheck(self, operation, snapshot, apply_result):
        self.calls.append(("audio", operation.operation_id))
        return AudioRecheckResult("failed", "feedback_risk_detected")

    async def assess_audio_failure_retention(
        self,
        operation,
        snapshot,
        apply_result,
    ):
        self.calls.append(("retain", operation.operation_id))
        return AdapterStepResult(
            self.retention_verified,
            self.retention_verified,
            (
                "audio_failure_safe_target_retained"
                if self.retention_verified
                else "audio_failure_retention_target_unverified"
            ),
            {
                "retention_eligible": True,
                "target_physical_verified": self.retention_verified,
                "write_attempted": False,
            },
        )

    async def rollback(self, operation, snapshot, apply_result):
        self.calls.append(("rollback", operation.operation_id))
        self.state = float(
            snapshot["values"][
                f"/{self.family_id}/{operation.operation_id}"
            ]
        )
        return AdapterStepResult(True, True, "rollback_verified")


@pytest.mark.asyncio
async def test_audio_failure_retains_verified_cut_without_rollback_up():
    adapter = AudioFailureRetentionAdapter("fader")
    service, emergency = coordinator(
        {"fader": contract("fader", "fader")},
        {"fader": adapter},
    )
    cut = ConsoleTransactionOperation(
        operation_id="fader-cut",
        family_id="fader",
        write_kind="fader",
        target_value=-2.0,
        channel=1,
    )

    result = await service.execute(
        ConsoleTransactionPlan("txn-retain-cut", (cut,), "safer cut"),
        context(key="retain-cut"),
    )

    assert result.status == "manual_review_required"
    assert result.reason == "feedback_risk_detected"
    assert result.contained_operation_ids == ["fader-cut"]
    assert result.contained_target_physical_verified is True
    assert result.audio_failure_recovery_state_known is True
    assert adapter.state == -2.0
    assert ("rollback", "fader-cut") not in adapter.calls
    assert emergency == [
        "supervised_transaction_audio_safety_failed:txn-retain-cut"
    ]


@pytest.mark.asyncio
async def test_audio_failure_rolls_back_before_long_emergency_teardown():
    order = []
    teardown_started = asyncio.Event()
    release_teardown = asyncio.Event()
    adapter = FakeAdapter("eq_hpf")

    async def failed_audio(operation, snapshot, apply_result):
        adapter.calls.append(("audio", operation.operation_id))
        return AudioRecheckResult("failed", "feedback_risk_detected")

    adapter.audio_recheck = failed_audio
    original_rollback = adapter.rollback

    async def ordered_rollback(operation, snapshot, apply_result):
        order.append("rollback")
        return await original_rollback(operation, snapshot, apply_result)

    adapter.rollback = ordered_rollback

    def latch(_reason):
        order.append("latch")
        return {"status": "latched", "success": True}

    async def delayed_teardown(_reason):
        order.append("teardown_started")
        teardown_started.set()
        await release_teardown.wait()

    service, _ = coordinator(
        {"eq_hpf": contract("eq_hpf", "eq")},
        {"eq_hpf": adapter},
        emergency=delayed_teardown,
        emergency_latch=latch,
    )
    task = asyncio.create_task(
        service.execute(
            ConsoleTransactionPlan(
                "txn-immediate-recovery",
                (operation("eq-cut"),),
                "recover before teardown",
            ),
            context(key="immediate-recovery"),
        )
    )

    await asyncio.wait_for(teardown_started.wait(), timeout=1.0)

    assert order == ["latch", "rollback", "teardown_started"]
    assert task.done() is False
    release_teardown.set()
    result = await asyncio.wait_for(task, timeout=1.0)
    assert result.status == "rolled_back"
    assert result.rollback_operation_ids == ["eq-cut"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("latch_result", "teardown_result", "expected_latch", "expected_teardown"),
    [
        (
            False,
            {
                "status": "stopped",
                "success": True,
                "teardown_complete": True,
            },
            False,
            True,
        ),
        (
            {"status": "latched", "success": True},
            False,
            True,
            False,
        ),
        (
            {"status": "latched", "success": False},
            {
                "status": "stopped",
                "success": True,
                "teardown_complete": True,
            },
            False,
            True,
        ),
        (
            {"status": "latched", "success": True},
            {
                "status": "stop_incomplete",
                "success": False,
                "teardown_complete": False,
                "teardown_errors": [{"step": "unit"}],
            },
            True,
            False,
        ),
    ],
)
async def test_audio_failure_reports_incomplete_emergency_phases(
    latch_result,
    teardown_result,
    expected_latch,
    expected_teardown,
):
    adapter = FakeAdapter("eq_hpf")

    async def failed_audio(operation, snapshot, apply_result):
        return AudioRecheckResult("failed", "feedback_risk_detected")

    adapter.audio_recheck = failed_audio
    service, _ = coordinator(
        {"eq_hpf": contract("eq_hpf", "eq")},
        {"eq_hpf": adapter},
        emergency_latch=lambda _reason: latch_result,
        emergency=lambda _reason: teardown_result,
    )

    result = await service.execute(
        ConsoleTransactionPlan(
            "txn-incomplete-emergency-phase",
            (operation("eq-cut"),),
            "report incomplete emergency phase",
        ),
        context(key="incomplete-emergency-phase"),
    )

    assert result.status == "manual_review_required"
    assert result.emergency_latch_succeeded is expected_latch
    assert result.emergency_teardown_succeeded is expected_teardown
    assert result.emergency_stop_complete is False
    emergency_events = [
        event
        for event in result.events
        if event.get("phase") in {"emergency_latch", "emergency_teardown"}
    ]
    assert [event["success"] for event in emergency_events] == [
        expected_latch,
        expected_teardown,
    ]


@pytest.mark.asyncio
async def test_emergency_latch_cancelled_error_is_terminal_not_propagated():
    adapter = FakeAdapter("eq_hpf")

    async def failed_audio(operation, snapshot, apply_result):
        return AudioRecheckResult("failed", "feedback_risk_detected")

    def cancelled_latch(_reason):
        raise asyncio.CancelledError()

    adapter.audio_recheck = failed_audio
    service, _ = coordinator(
        {"eq_hpf": contract("eq_hpf", "eq")},
        {"eq_hpf": adapter},
        emergency_latch=cancelled_latch,
    )

    result = await service.execute(
        ConsoleTransactionPlan(
            "txn-cancelled-latch",
            (operation("eq-cut"),),
            "contain cancelled latch",
        ),
        context(key="cancelled-latch"),
    )

    assert result.status == "manual_review_required"
    assert result.emergency_latch_succeeded is False
    assert result.emergency_teardown_succeeded is True
    assert result.emergency_stop_complete is False
    latch_event = next(
        event
        for event in result.events
        if event.get("phase") == "emergency_latch"
    )
    assert latch_event["exception_type"] == "CancelledError"


@pytest.mark.asyncio
async def test_synchronous_emergency_teardown_exception_is_contained():
    adapter = FakeAdapter("eq_hpf")

    async def failed_audio(operation, snapshot, apply_result):
        return AudioRecheckResult("failed", "feedback_risk_detected")

    def broken_teardown(_reason):
        raise RuntimeError("synthetic teardown failure")

    adapter.audio_recheck = failed_audio
    service, _ = coordinator(
        {"eq_hpf": contract("eq_hpf", "eq")},
        {"eq_hpf": adapter},
        emergency=broken_teardown,
    )

    result = await service.execute(
        ConsoleTransactionPlan(
            "txn-sync-teardown-error",
            (operation("eq-cut"),),
            "contain synchronous teardown error",
        ),
        context(key="sync-teardown-error"),
    )

    assert result.status == "manual_review_required"
    assert result.emergency_latch_succeeded is True
    assert result.emergency_teardown_succeeded is False
    assert result.emergency_stop_complete is False
    teardown_event = next(
        event
        for event in result.events
        if event.get("phase") == "emergency_teardown"
    )
    assert teardown_event["exception_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_emergency_teardown_cancelled_error_is_terminal_not_propagated():
    adapter = FakeAdapter("eq_hpf")

    async def failed_audio(operation, snapshot, apply_result):
        return AudioRecheckResult("failed", "feedback_risk_detected")

    async def cancelled_teardown(_reason):
        raise asyncio.CancelledError()

    adapter.audio_recheck = failed_audio
    service, _ = coordinator(
        {"eq_hpf": contract("eq_hpf", "eq")},
        {"eq_hpf": adapter},
        emergency=cancelled_teardown,
    )

    result = await service.execute(
        ConsoleTransactionPlan(
            "txn-cancelled-teardown",
            (operation("eq-cut"),),
            "contain cancelled teardown",
        ),
        context(key="cancelled-teardown"),
    )

    assert result.status == "manual_review_required"
    assert result.emergency_latch_succeeded is True
    assert result.emergency_teardown_succeeded is False
    assert result.emergency_stop_complete is False
    teardown_event = next(
        event
        for event in result.events
        if event.get("phase") == "emergency_teardown"
    )
    assert teardown_event["exception_type"] == "CancelledError"


@pytest.mark.asyncio
async def test_unverified_retained_cut_never_blindly_rolls_back_up():
    adapter = AudioFailureRetentionAdapter(
        "fader",
        retention_verified=False,
    )
    service, _ = coordinator(
        {"fader": contract("fader", "fader")},
        {"fader": adapter},
    )
    cut = ConsoleTransactionOperation(
        operation_id="fader-unverified",
        family_id="fader",
        write_kind="fader",
        target_value=-2.0,
        channel=1,
    )

    result = await service.execute(
        ConsoleTransactionPlan(
            "txn-retain-unverified",
            (cut,),
            "safer cut",
        ),
        context(key="retain-unverified"),
    )

    assert result.status == "manual_review_required"
    assert result.retention_unverified_operation_ids == [
        "fader-unverified"
    ]
    assert result.audio_failure_recovery_state_known is False
    assert ("rollback", "fader-unverified") not in adapter.calls


@pytest.mark.asyncio
async def test_pre_send_audio_failure_retains_prior_cut_and_skips_next_send():
    adapter = AudioFailureRetentionAdapter("fader")
    audio_calls = 0

    async def audio_recheck(operation, snapshot, apply_result):
        nonlocal audio_calls
        audio_calls += 1
        adapter.calls.append(("audio", operation.operation_id))
        return AudioRecheckResult("passed")

    adapter.audio_recheck = audio_recheck

    async def before_operation(operation):
        if operation.operation_id == "fader-cut-2":
            raise RuntimeError("feedback_risk_detected")
        return True

    service, _ = coordinator(
        {"fader": contract("fader", "fader")},
        {"fader": adapter},
        before_operation_apply_callback=before_operation,
    )
    cuts = (
        ConsoleTransactionOperation(
            operation_id="fader-cut-1",
            family_id="fader",
            write_kind="fader",
            target_value=-2.0,
            channel=1,
        ),
        ConsoleTransactionOperation(
            operation_id="fader-cut-2",
            family_id="fader",
            write_kind="fader",
            target_value=-4.0,
            channel=2,
        ),
    )

    result = await service.execute(
        ConsoleTransactionPlan("txn-pre-send-retain", cuts, "two cuts"),
        context(key="pre-send-retain"),
    )

    assert result.status == "manual_review_required"
    assert result.reason == "feedback_risk_detected"
    assert result.applied_operation_ids == ["fader-cut-1"]
    assert result.contained_operation_ids == ["fader-cut-1"]
    assert ("apply", "fader-cut-2") not in adapter.calls
    assert ("rollback", "fader-cut-1") not in adapter.calls
    assert audio_calls == 1


@pytest.mark.asyncio
async def test_deadline_is_rechecked_after_durable_send_boundary(
    monkeypatch,
):
    adapter = FakeAdapter("eq_hpf")
    row = contract("eq_hpf", "eq")
    clock = [100.0]

    def persist_send_boundary():
        clock[0] = 102.0
        return True

    service, _ = coordinator(
        {"eq_hpf": row},
        {"eq_hpf": adapter},
        before_apply_callback=persist_send_boundary,
    )
    plan = ConsoleTransactionPlan(
        "txn-boundary-expiry",
        (operation("eq-1"),),
        "deadline",
    )
    transaction_context = replace(context(), approval_deadline_epoch=101.0)
    monkeypatch.setattr(
        "supervised_console_transaction.time.time",
        lambda: clock[0],
    )

    result = await service.execute(plan, transaction_context)

    assert result.status == "blocked"
    assert result.reason == (
        "operator_approval_expired_after_durable_send_boundary"
    )
    assert adapter.calls == [("snapshot", "eq-1")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("boundary_mode", "expected_reason"),
    [
        ("cancelled", "durable_send_boundary_cancelled"),
        ("exception", "durable_send_boundary_failed"),
        ("false", "durable_send_boundary_failed"),
        (
            "approval_expired",
            "operator_approval_expired_after_durable_send_boundary",
        ),
    ],
)
async def test_second_durable_boundary_abort_rolls_back_first_operation(
    monkeypatch,
    boundary_mode,
    expected_reason,
):
    adapter = FakeAdapter("eq_hpf")
    row = contract("eq_hpf", "eq")
    boundary_calls = 0
    clock = [100.0]

    def persist_send_boundary():
        nonlocal boundary_calls
        boundary_calls += 1
        if boundary_calls == 1:
            return True
        if boundary_mode == "cancelled":
            raise asyncio.CancelledError()
        if boundary_mode == "exception":
            raise RuntimeError("synthetic boundary failure")
        if boundary_mode == "approval_expired":
            clock[0] = 102.0
            return True
        return False

    service, emergency_calls = coordinator(
        {"eq_hpf": row},
        {"eq_hpf": adapter},
        before_apply_callback=persist_send_boundary,
    )
    plan = ConsoleTransactionPlan(
        f"txn-second-boundary-{boundary_mode}",
        (operation("eq-1"), operation("eq-2")),
        "two operations",
    )
    transaction_context = replace(context(), approval_deadline_epoch=101.0)
    monkeypatch.setattr(
        "supervised_console_transaction.time.time",
        lambda: clock[0],
    )

    result = await service.execute(plan, transaction_context)

    assert result.status == "rolled_back"
    assert result.reason == expected_reason
    assert result.applied_operation_ids == ["eq-1"]
    assert result.rollback_operation_ids == ["eq-1"]
    assert ("apply", "eq-2") not in adapter.calls
    assert ("rollback", "eq-1") in adapter.calls
    assert emergency_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [1, "true", "false"])
async def test_transaction_rejects_non_literal_approval_before_snapshot(approved):
    adapter = FakeAdapter("eq_hpf")
    row = contract("eq_hpf", "eq")
    service, _ = coordinator({"eq_hpf": row}, {"eq_hpf": adapter})
    plan = ConsoleTransactionPlan("txn-non-literal", (operation("eq-1"),), "blocked")
    transaction_context = replace(context(), approved=approved)

    result = await service.execute(plan, transaction_context)

    assert result.status == "blocked"
    assert result.reason == "approval_required"
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_durable_prepare_runs_after_all_snapshots_and_before_first_apply():
    adapter = FakeAdapter("eq_hpf")
    row = contract("eq_hpf", "eq")
    callback_calls = []

    def prepared_callback(plan, callback_context, snapshots):
        callback_calls.append(
            {
                "transaction_id": plan.transaction_id,
                "approval_id": callback_context.approval_id,
                "snapshot_count": len(snapshots),
                "adapter_calls": list(adapter.calls),
            }
        )
        return True

    service, _ = coordinator(
        {"eq_hpf": row},
        {"eq_hpf": adapter},
        prepared_callback=prepared_callback,
    )
    plan = ConsoleTransactionPlan(
        "txn-durable-prepare",
        (operation("eq-1"), operation("eq-2")),
        "two bands",
    )

    result = await service.execute(plan, context())

    assert result.status == "applied"
    assert callback_calls == [
        {
            "transaction_id": "txn-durable-prepare",
            "approval_id": "operator-approval",
            "snapshot_count": 2,
            "adapter_calls": [("snapshot", "eq-1"), ("snapshot", "eq-2")],
        }
    ]
    assert adapter.calls[2] == ("apply", "eq-1")


@pytest.mark.asyncio
async def test_durable_prepare_failure_blocks_before_any_apply():
    adapter = FakeAdapter("eq_hpf")
    row = contract("eq_hpf", "eq")
    service, _ = coordinator(
        {"eq_hpf": row},
        {"eq_hpf": adapter},
        prepared_callback=lambda *_args: None,
    )
    plan = ConsoleTransactionPlan(
        "txn-durable-prepare-failed",
        (operation("eq-1"),),
        "blocked",
    )

    result = await service.execute(plan, context())

    assert result.status == "blocked"
    assert result.reason == "durable_prepare_failed"
    assert adapter.calls == [("snapshot", "eq-1")]


@pytest.mark.asyncio
async def test_transaction_applies_only_after_physical_snapshot_verify_and_audio_recheck():
    adapter = FakeAdapter("eq_hpf")
    row = contract("eq_hpf", "eq")
    service, emergency = coordinator({"eq_hpf": row}, {"eq_hpf": adapter})
    plan = ConsoleTransactionPlan("txn-1", (operation("eq-1"),), "cut resonance")

    result = await service.execute(plan, context())

    assert result.status == "applied"
    assert result.applied_operation_ids == ["eq-1"]
    assert adapter.calls == [
        ("snapshot", "eq-1"),
        ("apply", "eq-1"),
        ("verify", "eq-1"),
        ("audio", "eq-1"),
    ]
    assert emergency == []
    assert result.live_authority_granted is False


class AtomicStateAdapter:
    def __init__(self, state, *, on_audio=None, authority=None, cas_abort=False):
        self.family_id = "fader"
        self.state = state
        self.on_audio = on_audio
        self.authority = authority
        self.cas_abort = cas_abort
        self.calls = []

    def prepare(self, operation):
        address = f"/ch/{operation.channel}/fdr"
        return {
            "addresses": [address],
            "write_count": 1,
            "writes": [
                {
                    "address": address,
                    "target_value": operation.target_value,
                    "tolerance": 0.05,
                }
            ],
            "audible_state_reads": [],
        }

    def validate_snapshot(self, operation, snapshot):
        current = snapshot["values"][f"/ch/{operation.channel}/fdr"]
        if abs(float(current) - float(operation.target_value)) <= 0.05:
            return "physical_target_already_satisfied"
        return None

    def write_authority_is_current(self):
        return True if self.authority is None else self.authority["current"]

    async def snapshot(self, operation):
        self.calls.append(("snapshot", operation.operation_id))
        return {
            "physical_verified": True,
            "values": {
                f"/ch/{operation.channel}/fdr": self.state[operation.operation_id]
            },
        }

    async def apply(self, operation, snapshot):
        self.calls.append(("apply", operation.operation_id))
        if self.cas_abort:
            self.state[operation.operation_id] = operation.target_value
            return AdapterStepResult(
                False,
                False,
                "apply_cas_target_changed",
                {"writes": [], "write_state_unknown": False},
            )
        self.state[operation.operation_id] = operation.target_value
        return AdapterStepResult(
            True,
            False,
            "sent",
            {
                "writes": [
                    {
                        "address": f"/ch/{operation.channel}/fdr",
                        "value": operation.target_value,
                        "sent": True,
                    }
                ]
            },
        )

    async def verify(self, operation, snapshot, apply_result):
        self.calls.append(("verify", operation.operation_id))
        matches = abs(
            float(self.state[operation.operation_id])
            - float(operation.target_value)
        ) <= 0.05
        return AdapterStepResult(
            matches,
            matches,
            "verified" if matches else "drifted",
        )

    async def rollback(self, operation, snapshot, apply_result):
        self.calls.append(("rollback", operation.operation_id))
        if abs(
            float(self.state[operation.operation_id])
            - float(operation.target_value)
        ) > 0.05:
            return AdapterStepResult(False, False, "rollback_cas_mismatch")
        self.state[operation.operation_id] = snapshot["values"][
            f"/ch/{operation.channel}/fdr"
        ]
        return AdapterStepResult(True, True, "rollback_verified")

    async def audio_recheck(self, operation, snapshot, apply_result):
        self.calls.append(("audio", operation.operation_id))
        if self.on_audio is not None:
            self.on_audio(operation)
        return AudioRecheckResult("passed")


def atomic_fader_operation(operation_id, channel, target):
    return ConsoleTransactionOperation(
        operation_id=operation_id,
        family_id="fader",
        write_kind="fader",
        target_value=target,
        channel=channel,
        parameter="channel_fader_db",
        metadata={"atomic_group_commit_required": True},
    )


@pytest.mark.asyncio
async def test_atomic_group_skips_satisfied_member_and_applies_only_changed_member():
    state = {"fader-1": -2.0, "fader-2": 0.0}
    adapter = AtomicStateAdapter(state)
    row = contract("fader", "fader")
    service, _ = coordinator({"fader": row}, {"fader": adapter})
    plan = ConsoleTransactionPlan(
        "atomic-mixed",
        (
            atomic_fader_operation("fader-1", 1, -2.0),
            atomic_fader_operation("fader-2", 2, -2.0),
        ),
        "atomic balance",
    )

    result = await service.execute(plan, context())

    assert result.status == "applied"
    assert result.applied_operation_ids == ["fader-2"]
    assert ("apply", "fader-1") not in adapter.calls
    assert ("apply", "fader-2") in adapter.calls


@pytest.mark.asyncio
async def test_restore_only_proof_all_targets_satisfied_uses_two_pass_no_write():
    state = {"fader-1": -2.0, "fader-2": -2.0}
    adapter = AtomicStateAdapter(state)
    row = replace(
        contract("fader", "fader"),
        lifecycle=LIFECYCLE_PROOF_CANDIDATE,
        hardware_proof="missing",
        live_allowed_write_kinds=(),
    )
    service, emergency = coordinator({"fader": row}, {"fader": adapter})
    plan = ConsoleTransactionPlan(
        "atomic-proof-all-satisfied",
        (
            atomic_fader_operation("fader-1", 1, -2.0),
            atomic_fader_operation("fader-2", 2, -2.0),
        ),
        "verify already-restored targets",
    )
    grant = restore_only_grant(
        plan,
        adapter,
        key="proof-all-satisfied",
    )

    result = await service.execute(
        plan,
        context(
            proof_run=True,
            restore_after_verify=True,
            key="proof-all-satisfied",
            capability_grant=grant,
        ),
    )

    assert result.status == "proof_passed"
    assert result.reason == (
        "proof_already_satisfied_two_pass_readback_verified"
    )
    assert result.applied_operation_ids == []
    assert result.rollback_operation_ids == []
    assert result.already_satisfied_operation_ids == ["fader-1", "fader-2"]
    assert [call for call in adapter.calls if call[0] == "apply"] == []
    assert [call for call in adapter.calls if call[0] == "rollback"] == []
    assert [call for call in adapter.calls if call[0] == "audio"] == [
        ("audio", "fader-1"),
        ("audio", "fader-2"),
    ]
    assert [call for call in adapter.calls if call[0] == "snapshot"] == [
        ("snapshot", "fader-1"),
        ("snapshot", "fader-2"),
        ("snapshot", "fader-1"),
        ("snapshot", "fader-2"),
    ]
    assert emergency == []


@pytest.mark.asyncio
async def test_restore_only_proof_all_targets_satisfied_requires_audio_recheck():
    state = {"fader-1": -2.0, "fader-2": -2.0}

    def fail_second_audio(operation):
        if operation.operation_id == "fader-2":
            raise RuntimeError("live_audio_recheck_failed")

    adapter = AtomicStateAdapter(state, on_audio=fail_second_audio)
    row = replace(
        contract("fader", "fader"),
        lifecycle=LIFECYCLE_PROOF_CANDIDATE,
        hardware_proof="missing",
        live_allowed_write_kinds=(),
    )
    service, emergency = coordinator({"fader": row}, {"fader": adapter})
    plan = ConsoleTransactionPlan(
        "atomic-proof-all-satisfied-audio-fails",
        (
            atomic_fader_operation("fader-1", 1, -2.0),
            atomic_fader_operation("fader-2", 2, -2.0),
        ),
        "reject already-restored targets without audio evidence",
    )
    grant = restore_only_grant(
        plan,
        adapter,
        key="proof-all-satisfied-audio-fails",
    )

    result = await service.execute(
        plan,
        context(
            proof_run=True,
            restore_after_verify=True,
            key="proof-all-satisfied-audio-fails",
            capability_grant=grant,
        ),
    )

    assert result.status == "blocked"
    assert result.reason == "audio_recheck_exception"
    assert result.applied_operation_ids == []
    assert result.rollback_operation_ids == []
    assert [call for call in adapter.calls if call[0] == "apply"] == []
    assert [call for call in adapter.calls if call[0] == "rollback"] == []
    assert [call for call in adapter.calls if call[0] == "audio"] == [
        ("audio", "fader-1"),
        ("audio", "fader-2"),
    ]
    assert len(emergency) == 1
    assert emergency[0].startswith(
        "supervised_transaction_audio_safety_failed:"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("capability_kind", "proof_run", "expected_status"),
    [
        ("restore_only_proof", True, "proof_passed"),
        ("soundcheck_child_commit_once", False, "applied"),
    ],
)
async def test_private_single_operation_already_satisfied_is_two_pass_no_write(
    capability_kind,
    proof_run,
    expected_status,
):
    state = {"fader-1": -2.0}
    adapter = AtomicStateAdapter(state)
    row = replace(
        contract("fader", "fader"),
        lifecycle=LIFECYCLE_PROOF_CANDIDATE,
        hardware_proof="missing",
        live_allowed_write_kinds=(),
    )
    service, emergency = coordinator({"fader": row}, {"fader": adapter})
    single_operation = replace(
        atomic_fader_operation("fader-1", 1, -2.0),
        metadata={},
    )
    plan = ConsoleTransactionPlan(
        f"private-single-already-satisfied-{capability_kind}",
        (single_operation,),
        "idempotent private single-operation lane",
    )
    key = f"private-single-{capability_kind}"
    grant = restore_only_grant(
        plan,
        adapter,
        key=key,
        capability_kind=capability_kind,
    )

    result = await service.execute(
        plan,
        context(
            proof_run=proof_run,
            restore_after_verify=proof_run,
            key=key,
            capability_grant=grant,
        ),
    )

    assert result.status == expected_status
    assert result.applied_operation_ids == []
    assert result.rollback_operation_ids == []
    assert result.already_satisfied_operation_ids == ["fader-1"]
    assert [call for call in adapter.calls if call[0] == "apply"] == []
    assert [call for call in adapter.calls if call[0] == "rollback"] == []
    assert [call for call in adapter.calls if call[0] == "audio"] == [
        ("audio", "fader-1"),
    ]
    assert [call for call in adapter.calls if call[0] == "snapshot"] == [
        ("snapshot", "fader-1"),
    ]
    assert [call for call in adapter.calls if call[0] == "verify"] == [
        ("verify", "fader-1"),
    ]
    assert emergency == []


@pytest.mark.asyncio
async def test_atomic_group_detects_prior_member_drift_before_next_send():
    state = {"fader-1": 0.0, "fader-2": 0.0}

    def drift_first(operation):
        if operation.operation_id == "fader-1":
            state["fader-1"] = -1.0

    adapter = AtomicStateAdapter(state, on_audio=drift_first)
    row = contract("fader", "fader")
    service, emergency = coordinator({"fader": row}, {"fader": adapter})
    plan = ConsoleTransactionPlan(
        "atomic-drift",
        (
            atomic_fader_operation("fader-1", 1, -2.0),
            atomic_fader_operation("fader-2", 2, -2.0),
        ),
        "atomic balance",
    )

    result = await service.execute(plan, context())

    assert result.status == "manual_review_required"
    assert result.reason == "atomic_group_member_drift_detected"
    assert ("apply", "fader-2") not in adapter.calls
    assert emergency


@pytest.mark.asyncio
async def test_approval_expiry_during_last_audio_recheck_rolls_back(monkeypatch):
    clock = {"now": 1.0}
    state = {"fader-1": 0.0}

    def expire(_operation):
        clock["now"] = 10.0

    adapter = AtomicStateAdapter(state, on_audio=expire)
    row = contract("fader", "fader")
    service, _ = coordinator({"fader": row}, {"fader": adapter})
    monkeypatch.setattr(
        "supervised_console_transaction.time.time",
        lambda: clock["now"],
    )
    expiring_context = replace(context(), approval_deadline_epoch=5.0)
    plan = ConsoleTransactionPlan(
        "approval-expires",
        (atomic_fader_operation("fader-1", 1, -2.0),),
        "expiry proof",
    )

    result = await service.execute(plan, expiring_context)

    assert result.status == "rolled_back"
    assert result.reason == "operator_approval_expired_before_commit"
    assert state["fader-1"] == 0.0


@pytest.mark.asyncio
async def test_authority_revocation_after_first_audio_prevents_second_send():
    authority = {"current": True}
    state = {"fader-1": 0.0, "fader-2": 0.0}

    def revoke(operation):
        if operation.operation_id == "fader-1":
            authority["current"] = False

    adapter = AtomicStateAdapter(
        state,
        on_audio=revoke,
        authority=authority,
    )
    row = contract("fader", "fader")
    service, _ = coordinator({"fader": row}, {"fader": adapter})
    plan = ConsoleTransactionPlan(
        "authority-revoked",
        (
            atomic_fader_operation("fader-1", 1, -2.0),
            atomic_fader_operation("fader-2", 2, -2.0),
        ),
        "authority proof",
    )

    result = await service.execute(plan, context())

    assert result.status == "rolled_back"
    assert result.reason == "write_authority_revoked_after_audio_recheck"
    assert ("apply", "fader-2") not in adapter.calls
    assert state["fader-1"] == 0.0


@pytest.mark.asyncio
async def test_cas_abort_with_no_send_never_rolls_back_human_target():
    state = {"fader-1": 0.0}
    adapter = AtomicStateAdapter(state, cas_abort=True)
    row = contract("fader", "fader")
    service, _ = coordinator({"fader": row}, {"fader": adapter})
    plan = ConsoleTransactionPlan(
        "cas-no-send",
        (atomic_fader_operation("fader-1", 1, -2.0),),
        "CAS proof",
    )

    result = await service.execute(plan, context())

    assert result.status == "blocked"
    assert result.reason == "apply_cas_target_changed"
    assert ("rollback", "fader-1") not in adapter.calls
    assert state["fader-1"] == -2.0


@pytest.mark.asyncio
async def test_proposal_only_family_is_blocked_even_in_proof_run():
    adapter = FakeAdapter("eq_hpf")
    row = replace(
        contract("eq_hpf", "eq"),
        lifecycle="proposal_only",
        hardware_proof="missing",
        live_allowed_write_kinds=(),
    )
    service, _ = coordinator({"eq_hpf": row}, {"eq_hpf": adapter})
    plan = ConsoleTransactionPlan("txn-2", (operation("eq-1"),), "blocked")
    grant = restore_only_grant(plan, adapter, key="proposal-only-proof")

    result = await service.execute(
        plan,
        context(
            proof_run=True,
            restore_after_verify=True,
            key="proposal-only-proof",
            capability_grant=grant,
        ),
    )

    assert result.status == "blocked"
    assert result.reason == "write_kind_not_live_or_proof_allowed"
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_proof_candidate_requires_explicit_proof_run():
    adapter = FakeAdapter("eq_hpf")
    row = replace(
        contract("eq_hpf", "eq"),
        lifecycle=LIFECYCLE_PROOF_CANDIDATE,
        hardware_proof="missing",
        live_allowed_write_kinds=(),
    )
    service, _ = coordinator({"eq_hpf": row}, {"eq_hpf": adapter})
    plan = ConsoleTransactionPlan("txn-3", (operation("eq-1"),), "proof")

    blocked = await service.execute(plan, context(key="normal"))
    public_flag_blocked = await service.execute(
        plan,
        context(proof_run=True, restore_after_verify=True, key="public-proof"),
    )
    grant = restore_only_grant(plan, adapter, key="proof")
    passed = await service.execute(
        plan,
        context(
            proof_run=True,
            restore_after_verify=True,
            key="proof",
            capability_grant=grant,
        ),
    )

    assert blocked.reason == "write_kind_not_live_or_proof_allowed"
    assert public_flag_blocked.reason == "restore_only_proof_grant_required"
    assert passed.status == "proof_passed"


@pytest.mark.asyncio
async def test_partial_failure_rolls_back_in_reverse_order():
    adapter = FakeAdapter("eq_hpf")
    row = contract("eq_hpf", "eq")
    service, _ = coordinator({"eq_hpf": row}, {"eq_hpf": adapter})
    operations = (operation("eq-1"), operation("eq-2"))
    plan = ConsoleTransactionPlan("txn-4", operations, "two bands")

    original_verify = adapter.verify

    async def verify(op, snapshot, apply_result):
        if op.operation_id == "eq-2":
            return AdapterStepResult(False, False, "readback_mismatch")
        return await original_verify(op, snapshot, apply_result)

    adapter.verify = verify
    result = await service.execute(plan, context())

    assert result.status == "rolled_back"
    assert result.rollback_operation_ids == ["eq-2", "eq-1"]
    rollback_calls = [call for call in adapter.calls if call[0] == "rollback"]
    assert rollback_calls == [("rollback", "eq-2"), ("rollback", "eq-1")]


@pytest.mark.asyncio
async def test_failed_rollback_activates_emergency_stop():
    adapter = FakeAdapter("eq_hpf", fail_verify=True, fail_rollback=True)
    row = contract("eq_hpf", "eq")
    service, emergency = coordinator({"eq_hpf": row}, {"eq_hpf": adapter})
    plan = ConsoleTransactionPlan("txn-5", (operation("eq-1"),), "failure")

    result = await service.execute(plan, context())

    assert result.status == "manual_review_required"
    assert result.emergency_stop_activated is True
    assert emergency == ["supervised_transaction_rollback_unverified:txn-5"]


@pytest.mark.asyncio
async def test_rollback_continues_restoring_earlier_operations_after_one_failure():
    adapter = FakeAdapter("eq_hpf")
    row = contract("eq_hpf", "eq")
    service, emergency = coordinator({"eq_hpf": row}, {"eq_hpf": adapter})
    original_verify = adapter.verify

    async def verify(op, snapshot, apply_result):
        if op.operation_id == "eq-2":
            return AdapterStepResult(False, False, "readback_mismatch")
        return await original_verify(op, snapshot, apply_result)

    async def rollback(op, snapshot, apply_result):
        adapter.calls.append(("rollback", op.operation_id))
        if op.operation_id == "eq-2":
            return AdapterStepResult(False, False, "rollback_mismatch")
        return AdapterStepResult(True, True, "rollback_verified")

    adapter.verify = verify
    adapter.rollback = rollback
    plan = ConsoleTransactionPlan("txn-continue", (operation("eq-1"), operation("eq-2")), "rollback all")

    result = await service.execute(plan, context(key="continue"))

    assert result.status == "manual_review_required"
    assert result.rollback_operation_ids == ["eq-1"]
    assert [call for call in adapter.calls if call[0] == "rollback"] == [
        ("rollback", "eq-2"),
        ("rollback", "eq-1"),
    ]
    assert emergency == ["supervised_transaction_rollback_unverified:txn-continue"]


@pytest.mark.asyncio
async def test_hardware_proof_restores_original_state_after_success():
    adapter = FakeAdapter("eq_hpf")
    row = replace(
        contract("eq_hpf", "eq"),
        lifecycle=LIFECYCLE_PROOF_CANDIDATE,
        hardware_proof="missing",
        live_allowed_write_kinds=(),
    )
    service, emergency = coordinator({"eq_hpf": row}, {"eq_hpf": adapter})
    plan = ConsoleTransactionPlan("proof-1", (operation("eq-1"),), "proof")
    grant = restore_only_grant(plan, adapter, key="proof-restore")

    result = await service.execute(
        plan,
        context(
            proof_run=True,
            restore_after_verify=True,
            key="proof-restore",
            capability_grant=grant,
        ),
    )

    assert result.status == "proof_passed"
    assert result.reason == "proof_apply_readback_audio_and_rollback_verified"
    assert result.rollback_operation_ids == ["eq-1"]
    assert adapter.calls[-1] == ("rollback", "eq-1")
    assert emergency == []


@pytest.mark.asyncio
async def test_high_risk_family_cannot_be_mixed_with_normal_dsp_transaction():
    eq_adapter = FakeAdapter("eq_hpf")
    routing_adapter = FakeAdapter("routing")
    contracts = {
        "eq_hpf": contract("eq_hpf", "eq"),
        "routing": contract("routing", "routing", separate_approval=True),
    }
    service, _ = coordinator(
        contracts,
        {"eq_hpf": eq_adapter, "routing": routing_adapter},
    )
    plan = ConsoleTransactionPlan(
        "txn-6",
        (
            operation("eq-1"),
            operation("route-1", "routing", "routing"),
        ),
        "unsafe batch",
    )

    result = await service.execute(plan, context())

    assert result.status == "blocked"
    assert result.reason == "high_risk_family_requires_separate_transaction"


@pytest.mark.asyncio
async def test_duplicate_idempotency_and_emergency_generation_fail_closed():
    adapter = FakeAdapter("eq_hpf")
    row = contract("eq_hpf", "eq")
    generation = {"value": 7}
    service, _ = coordinator(
        {"eq_hpf": row},
        {"eq_hpf": adapter},
        generation=lambda: generation["value"],
    )
    plan = ConsoleTransactionPlan("txn-7", (operation("eq-1"),), "idempotent")

    first = await service.execute(plan, context(key="same"))
    duplicate = await service.execute(plan, context(key="same"))
    generation["value"] = 8
    stale_generation = await service.execute(plan, context(key="new"))

    assert first.status == "applied"
    assert duplicate.reason == "duplicate_idempotency_key"
    assert stale_generation.reason == "emergency_generation_changed"


@pytest.mark.asyncio
async def test_current_operation_is_recovered_when_apply_reports_failure():
    adapter = FakeAdapter("eq_hpf")
    row = contract("eq_hpf", "eq")
    service, emergency = coordinator({"eq_hpf": row}, {"eq_hpf": adapter})

    async def fail_apply(op, snapshot):
        adapter.calls.append(("apply", op.operation_id))
        return AdapterStepResult(
            False,
            False,
            "second_parameter_send_failed",
            {"writes": [{"address": "/first", "sent": True}]},
        )

    adapter.apply = fail_apply
    result = await service.execute(
        ConsoleTransactionPlan("txn-apply-false", (operation("eq-1"),), "recover"),
        context(key="apply-false"),
    )

    assert result.status == "rolled_back"
    assert result.reason == "second_parameter_send_failed"
    assert ("rollback", "eq-1") in adapter.calls
    assert emergency == []


@pytest.mark.asyncio
async def test_apply_exception_enters_recovery_set():
    adapter = FakeAdapter("eq_hpf")
    row = contract("eq_hpf", "eq")
    service, emergency = coordinator({"eq_hpf": row}, {"eq_hpf": adapter})

    async def raise_apply(op, snapshot):
        adapter.calls.append(("apply", op.operation_id))
        raise RuntimeError("synthetic partial send")

    adapter.apply = raise_apply
    result = await service.execute(
        ConsoleTransactionPlan("txn-apply-exception", (operation("eq-1"),), "recover"),
        context(key="apply-exception"),
    )

    assert result.status == "rolled_back"
    assert result.reason == "apply_exception"
    assert ("rollback", "eq-1") in adapter.calls
    assert emergency == []


@pytest.mark.asyncio
async def test_verify_and_audio_exceptions_fail_closed_and_rollback():
    row = contract("eq_hpf", "eq")

    verify_adapter = FakeAdapter("eq_hpf")

    async def raise_verify(op, snapshot, apply_result):
        verify_adapter.calls.append(("verify", op.operation_id))
        raise RuntimeError("synthetic readback failure")

    verify_adapter.verify = raise_verify
    verify_service, _ = coordinator(
        {"eq_hpf": row},
        {"eq_hpf": verify_adapter},
    )
    verify_result = await verify_service.execute(
        ConsoleTransactionPlan("txn-verify-exception", (operation("eq-1"),), "recover"),
        context(key="verify-exception"),
    )

    audio_adapter = FakeAdapter("eq_hpf")

    async def raise_audio(op, snapshot, apply_result):
        audio_adapter.calls.append(("audio", op.operation_id))
        raise RuntimeError("synthetic meter failure")

    audio_adapter.audio_recheck = raise_audio
    audio_service, _ = coordinator(
        {"eq_hpf": row},
        {"eq_hpf": audio_adapter},
    )
    audio_result = await audio_service.execute(
        ConsoleTransactionPlan("txn-audio-exception", (operation("eq-1"),), "recover"),
        context(key="audio-exception"),
    )

    assert verify_result.status == "rolled_back"
    assert verify_result.reason == "verify_exception"
    assert audio_result.status == "rolled_back"
    assert audio_result.reason == "audio_recheck_exception"
    assert ("rollback", "eq-1") in verify_adapter.calls
    assert ("rollback", "eq-1") in audio_adapter.calls


@pytest.mark.asyncio
async def test_emergency_generation_change_after_apply_still_runs_cas_recovery():
    generation = {"value": 7}
    adapter = FakeAdapter("eq_hpf")
    row = contract("eq_hpf", "eq")
    service, emergency = coordinator(
        {"eq_hpf": row},
        {"eq_hpf": adapter},
        generation=lambda: generation["value"],
    )
    original_apply = adapter.apply

    async def apply_and_stop(op, snapshot):
        result = await original_apply(op, snapshot)
        generation["value"] = 8
        return result

    adapter.apply = apply_and_stop
    result = await service.execute(
        ConsoleTransactionPlan("txn-generation-race", (operation("eq-1"),), "recover"),
        context(key="generation-race"),
    )

    assert result.status == "rolled_back"
    assert result.reason == "emergency_generation_changed_after_apply"
    assert result.rollback_operation_ids == ["eq-1"]
    assert any(
        event.get("reason")
        == "emergency_generation_changed_continuing_cas_recovery"
        for event in result.events
    )
    assert emergency == []


@pytest.mark.asyncio
async def test_all_operation_targets_are_preflighted_before_first_snapshot_or_apply():
    adapter = FakeAdapter("eq_hpf")
    row = contract("eq_hpf", "eq")
    service, _ = coordinator({"eq_hpf": row}, {"eq_hpf": adapter})
    invalid = ConsoleTransactionOperation(
        operation_id="eq-invalid",
        family_id="eq_hpf",
        write_kind="eq",
        target_value="invalid",
        channel=3,
    )
    plan = ConsoleTransactionPlan(
        "txn-preflight-all",
        (operation("eq-valid"), invalid),
        "must not partially apply",
    )

    result = await service.execute(plan, context(key="preflight-all"))

    assert result.status == "blocked"
    assert result.reason.startswith("operation_preflight_failed:eq-invalid")
    assert adapter.calls == []
