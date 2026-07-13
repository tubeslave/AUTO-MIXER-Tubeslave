from __future__ import annotations

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


class FakeAdapter:
    def __init__(self, family_id: str, *, fail_verify: bool = False, fail_rollback: bool = False):
        self.family_id = family_id
        self.fail_verify = fail_verify
        self.fail_rollback = fail_rollback
        self.calls = []

    def prepare(self, operation):
        if operation.target_value == "invalid":
            raise ValueError("synthetic_invalid_target")
        return {"addresses": [f"/{self.family_id}/{operation.operation_id}"]}

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
):
    return ConsoleTransactionContext(
        approved=True,
        approval_id="operator-approval",
        idempotency_key=key,
        emergency_generation=7,
        proof_run=proof_run,
        restore_after_verify=restore_after_verify,
    )


def coordinator(
    contracts,
    adapters,
    generation=lambda: 7,
    emergency=None,
    prepared_callback=None,
):
    emergency_calls = []

    async def stop(reason):
        emergency_calls.append(reason)

    return (
        SupervisedConsoleTransactionCoordinator(
            family_contracts=contracts,
            adapters=adapters,
            emergency_generation_getter=generation,
            emergency_stop=emergency or stop,
            prepared_callback=prepared_callback,
        ),
        emergency_calls,
    )


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

    result = await service.execute(plan, context(proof_run=True))

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
    passed = await service.execute(plan, context(proof_run=True, key="proof"))

    assert blocked.reason == "write_kind_not_live_or_proof_allowed"
    assert passed.status == "applied"


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

    result = await service.execute(
        plan,
        context(
            proof_run=True,
            restore_after_verify=True,
            key="proof-restore",
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
