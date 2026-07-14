from __future__ import annotations

import asyncio
import copy
from dataclasses import replace
import time
from unittest.mock import AsyncMock, patch

import pytest

from operator_proposal_queue import (
    OperatorProposalQueue,
    SOUNDCHECK_CORRECTION_REQUIRES_SUPERVISED_TRANSACTION,
)
from operator_proposal_result_store import ResultEvidenceStore
from operator_recommendation_bridge import build_soundcheck_operator_proposals
from operator_product_state import build_decision_queue
from server import AutoMixerServer
from supervised_console_parameter_registry import (
    FAMILY_CONTRACTS,
    LIFECYCLE_LIVE_ALLOWED,
)
from supervised_console_transaction import (
    AdapterStepResult,
    AudioRecheckResult,
    ConsoleTransactionResult,
)


def transaction_plan():
    return {
        "transaction_id": "soundcheck:normal:test",
        "reason": "test EQ cut",
        "operations": [
            {
                "operation_id": "eq-test",
                "family_id": "eq_hpf",
                "write_kind": "eq",
                "target_value": {
                    "band": 2,
                    "freq_hz": 300.0,
                    "gain_db": -2.0,
                    "q": 1.1,
                },
                "channel": 3,
                "parameter": "eq_band_2",
                "metadata": {"absolute_target": True},
            }
        ],
        "separate_approval_required": False,
    }


def supervised_mode(*allowed_live_write_kinds):
    return {
        "mode": "supervised",
        "capabilities": {
            "can_create_proposals": True,
            "can_apply_to_console": True,
            "allowed_live_write_kinds": list(allowed_live_write_kinds),
        },
    }


def evidence_queue(tmp_path, name):
    return OperatorProposalQueue(
        result_store=ResultEvidenceStore(
            tmp_path / name,
            min_free_bytes=0,
        )
    )


def reserve_evidence(queue, proposal):
    reservation = queue.reserve_result_evidence(proposal)
    assert reservation["reserved"] is True
    return reservation["reservation_binding"]


def verified_apply_result():
    return {
        "status": "applied",
        "success": True,
        "no_console_send_proven": False,
        "physical_verified": True,
        "wal_terminal": "committed",
        "audio_recheck_required": False,
        "audio_recheck_disposition": "not_required_by_manual_write_contract",
        "write_state_unknown": False,
        "rollback_state_unknown": False,
        "manual_review_required": False,
    }


def verified_already_satisfied_apply_result():
    return {
        "status": "applied",
        "success": True,
        "applied_operation_ids": [],
        "rollback_operation_ids": [],
        "already_satisfied_operation_ids": ["already-satisfied-operation"],
        "proof_run": False,
        "restore_after_verify": False,
        "accepted": True,
        "owner_review_required": False,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "no_console_send_proven": True,
        "physical_verified": True,
        "proof_target_verified": False,
        "physical_state_restored": False,
        "terminal_after_console_send": False,
        "rollback_attempted": False,
        "rollback_succeeded": False,
        "rollback_status": "not_attempted",
        "wal_terminal": "committed",
        "audio_recheck_passed": True,
        "write_state_unknown": False,
        "rollback_state_unknown": False,
        "manual_review_required": False,
        "emergency_stop_activated": False,
        "emergency_stop_required": False,
        "fresh_analysis_required": False,
    }


def test_soundcheck_bridge_imports_transaction_plan_into_operator_queue():
    plan = transaction_plan()
    bundle = {
        "proposals": [],
        "supervised_transaction_plans": {"plans": [plan]},
    }

    proposals = build_soundcheck_operator_proposals(bundle)
    queue = OperatorProposalQueue()
    result = queue.create(proposals[0], supervised_mode())

    assert len(proposals) == 1
    assert proposals[0]["supervised_transaction_plan"] == plan
    assert proposals[0]["ttl_seconds"] == 300
    assert result["status"] == "created"
    queued = result["proposal"]
    assert queued["kind"] == "supervised_soundcheck_transaction"
    assert queued["can_apply"] is False
    assert queued["apply_blockers"] == ["write_kind_not_live_allowed:eq_hpf:eq"]


def test_unmatched_soundcheck_fader_stays_operator_review_only():
    proposals = build_soundcheck_operator_proposals({
        "schema_version": "soundcheck_recommendation_bundle/v1",
        "proposals": [
            {
                "proposal_id": "legacy-fader",
                "action_type": "set_fader_db",
                "target": {"mixer_channel": 2},
                "current_state": {"fader_db": -10.0},
                "requested_state": {"fader_db": -8.5},
            }
        ],
    })

    assert len(proposals) == 1
    assert proposals[0]["requested_change"]["value_type"] == "fader"
    assert proposals[0]["operator_review_only"] is True
    assert proposals[0]["operator_review_only_reason"] == (
        SOUNDCHECK_CORRECTION_REQUIRES_SUPERVISED_TRANSACTION
    )

    queue = OperatorProposalQueue()
    created = queue.create(proposals[0], supervised_mode("fader"))

    # This legacy review packet repeats several full evidence surfaces and is
    # larger than the v2 per-proposal immutable-review admission limit.  It
    # remains review-only and is now rejected before consuming its ID rather
    # than entering an unbounded WebSocket queue.
    assert created["status"] == "blocked"
    assert created["reason"] == "proposal_review_content_too_large"
    assert created["review_content_bytes"] > created["max_review_content_bytes"]
    assert queue.proposal_id_was_seen("soundcheck:legacy-fader") is False


def test_operator_queue_nested_plan_is_immutable_on_ingress_and_egress():
    plan = transaction_plan()
    payload = build_soundcheck_operator_proposals(
        {
            "proposals": [],
            "supervised_transaction_plans": {"plans": [plan]},
        }
    )[0]
    queue = OperatorProposalQueue()
    created = queue.create(payload, supervised_mode())
    queued = queue.find(proposal_id=created["proposal"]["id"])
    original_target = copy.deepcopy(
        queued.raw["supervised_transaction_plan"]["operations"][0][
            "target_value"
        ]
    )

    payload["supervised_transaction_plan"]["operations"][0][
        "target_value"
    ] = {"gain_db": 15.0}
    created["proposal"]["raw"]["supervised_transaction_plan"]["operations"][0][
        "target_value"
    ] = {"gain_db": 12.0}

    assert queued.raw["supervised_transaction_plan"]["operations"][0][
        "target_value"
    ] == original_target
    fresh_view = queued.to_dict()
    fresh_view["raw"]["supervised_transaction_plan"]["operations"][0][
        "target_value"
    ] = {"gain_db": 9.0}
    assert queued.raw["supervised_transaction_plan"]["operations"][0][
        "target_value"
    ] == original_target


@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
@pytest.mark.asyncio
async def test_proven_no_send_transaction_block_releases_retry_reservations(
    _mock_config,
    _mock_bleed,
    tmp_path,
):
    class FakeWing:
        is_connected = True

        def query_parameter(self, address, timeout):
            raise AssertionError("fake coordinator must not query")

        def send(self, address, value):
            raise AssertionError("proven no-send block must not write")

    class NoSendCoordinator:
        def __init__(self, **_kwargs):
            pass

        async def execute(self, plan, _context):
            return ConsoleTransactionResult(
                transaction_id=plan.transaction_id,
                status="blocked",
                reason="apply_cas_target_changed",
                events=[
                    {
                        "operation_id": plan.operations[0].operation_id,
                        "phase": "apply",
                        "result": {
                            "success": False,
                            "physical_verified": False,
                            "reason": "apply_cas_target_changed",
                            "evidence": {
                                "writes": [],
                                "write_state_unknown": False,
                            },
                        },
                    }
                ],
            )

    server = AutoMixerServer()
    server.mixer_client = FakeWing()
    server.connection_mode = "wing"
    server._wing_observation_only = False
    server.get_operator_mode_status = lambda: supervised_mode("fader")
    server._record_manual_write_event = lambda *_args, **_kwargs: None
    fader_contract = next(
        contract for contract in FAMILY_CONTRACTS if contract.family_id == "fader"
    )
    promoted_contract = replace(
        fader_contract,
        lifecycle=LIFECYCLE_LIVE_ALLOWED,
        hardware_proof="accepted",
        live_allowed_write_kinds=("fader",),
        blockers=(),
    )
    plan = transaction_plan()
    plan["transaction_id"] = "retryable-no-send"
    plan["operations"][0].update(
        operation_id="retry-fader",
        family_id="fader",
        write_kind="fader",
        target_value=-12.0,
        channel=3,
        parameter="channel_fader_db",
    )

    with patch("server.FAMILY_CONTRACTS", (promoted_contract,)), patch(
        "server.SupervisedConsoleTransactionCoordinator",
        NoSendCoordinator,
    ):
        first = await server._apply_supervised_soundcheck_transaction_unlocked(
            plan,
            approved=True,
            approval_id="approval",
            idempotency_key="retry-key",
        )
        second = await server._apply_supervised_soundcheck_transaction_unlocked(
            plan,
            approved=True,
            approval_id="approval",
            idempotency_key="retry-key",
        )

    assert first["reason"] == second["reason"] == "apply_cas_target_changed"
    assert first["no_console_send_proven"] is True
    assert second["no_console_send_proven"] is True
    assert first["wal_terminal"] == second["wal_terminal"] == "not_prepared"
    assert first["physical_verified"] is False
    assert second["physical_verified"] is False
    assert server._supervised_transaction_idempotency_keys == {}
    assert server._supervised_transaction_ids == {}

    queue = evidence_queue(tmp_path, "actual-not-prepared-result")
    created = queue.create(
        {
            "id": "actual-not-prepared-result",
            "title": "No-send transaction result",
            "requested_change": {
                "value_type": "fader",
                "channel": 3,
                "value": -12.0,
            },
        },
        supervised_mode("fader"),
    )
    proposal = queue.find(proposal_id=created["proposal"]["id"])
    reservation = reserve_evidence(queue, proposal)
    assert queue.claim_for_apply(proposal)["claimed"] is True

    completed = queue.mark_apply_result(
        proposal,
        first,
        evidence_reservation_binding=reservation,
    )

    assert completed["status"] == "apply_blocked"
    assert proposal.result_evidence_pinned is False
    restarted = OperatorProposalQueue(
        result_store=ResultEvidenceStore(
            queue.result_store.root,
            min_free_bytes=0,
            process_instance_id="c" * 32,
        ),
        result_evidence_lifecycle_ready=True,
    )
    assert restarted.result_recovery_hazards == ()
    assert restarted.result_evidence_contract_ready is True


@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
@pytest.mark.asyncio
async def test_verified_transaction_rollback_is_terminal_fresh_analysis_block(
    _mock_config,
    _mock_bleed,
    tmp_path,
):
    class FakeWing:
        is_connected = True

        def query_parameter(self, address, timeout):
            raise AssertionError("fake coordinator owns this test")

        def send(self, address, value):
            raise AssertionError("fake coordinator owns this test")

    class RolledBackCoordinator:
        def __init__(self, *, prepared_callback=None, **_kwargs):
            self.prepared_callback = prepared_callback

        async def execute(self, plan, context):
            assert self.prepared_callback is not None
            prepared = self.prepared_callback(
                plan,
                context,
                (
                    {
                        "physical_verified": True,
                        "values": {"/ch/3/fdr": -10.0},
                    },
                ),
            )
            if asyncio.iscoroutine(prepared):
                await prepared
            return ConsoleTransactionResult(
                transaction_id=plan.transaction_id,
                status="rolled_back",
                reason="audio_recheck_failed",
                applied_operation_ids=[plan.operations[0].operation_id],
                rollback_operation_ids=[plan.operations[0].operation_id],
                events=[
                    {
                        "operation_id": plan.operations[0].operation_id,
                        "phase": "apply",
                        "result": {
                            "success": True,
                            "physical_verified": True,
                            "reason": "write_readback_verified",
                            "evidence": {"writes": ["/ch/3/fdr"]},
                        },
                    },
                    {
                        "operation_id": plan.operations[0].operation_id,
                        "phase": "rollback",
                        "result": {
                            "success": True,
                            "physical_verified": True,
                            "reason": "rollback_verified",
                            "evidence": {"values": {"/ch/3/fdr": -10.0}},
                        },
                    },
                ],
            )

    server = AutoMixerServer()
    server.mixer_client = FakeWing()
    server.connection_mode = "wing"
    server._wing_observation_only = False
    server.get_operator_mode_status = lambda: supervised_mode("fader")
    server._record_manual_write_event = lambda *_args, **_kwargs: None
    server._append_manual_write_wal = lambda *_args, **_kwargs: "/tmp/test-wal"
    fader_contract = next(
        contract for contract in FAMILY_CONTRACTS if contract.family_id == "fader"
    )
    promoted_contract = replace(
        fader_contract,
        lifecycle=LIFECYCLE_LIVE_ALLOWED,
        hardware_proof="accepted",
        live_allowed_write_kinds=("fader",),
        blockers=(),
    )
    plan = transaction_plan()
    plan["transaction_id"] = "verified-rollback"
    plan["operations"][0].update(
        operation_id="verified-rollback-fader",
        family_id="fader",
        write_kind="fader",
        target_value=-12.0,
        channel=3,
        parameter="channel_fader_db",
    )

    with patch("server.FAMILY_CONTRACTS", (promoted_contract,)), patch(
        "server.SupervisedConsoleTransactionCoordinator",
        RolledBackCoordinator,
    ):
        apply_result = await server._apply_supervised_soundcheck_transaction_unlocked(
            plan,
            approved=True,
            approval_id="rollback-approval",
            idempotency_key="rollback-idempotency",
        )

    assert apply_result["status"] == "rolled_back"
    assert apply_result["no_console_send_proven"] is False
    assert apply_result["physical_state_restored"] is True
    assert apply_result["terminal_after_console_send"] is True
    assert apply_result["fresh_analysis_required"] is True

    queue = evidence_queue(tmp_path, "verified-rollback")
    created = queue.create(
        {
            "id": "verified-rollback-proposal",
            "title": "Verified rollback proposal",
            "requested_change": {
                "value_type": "fader",
                "channel": 3,
                "value": -12.0,
            },
        },
        supervised_mode("fader"),
    )
    proposal = queue.find(proposal_id=created["proposal"]["id"])
    evidence_token = reserve_evidence(queue, proposal)
    assert queue.claim_for_apply(proposal)["claimed"] is True
    terminal = queue.mark_apply_result(
        proposal,
        apply_result,
        evidence_reservation_binding=evidence_token,
    )

    assert terminal["status"] == "apply_blocked"
    assert terminal["proposal"]["status"] == "apply_blocked"
    assert terminal["proposal"]["can_apply"] is False
    assert queue.summary()["pending_count"] == 0


@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
@pytest.mark.asyncio
async def test_supervised_idempotency_history_never_evicts_old_live_claims(
    _mock_config,
    _mock_bleed,
):
    class FakeWing:
        is_connected = True

        def query_parameter(self, address, timeout):
            raise AssertionError("fake coordinator owns this test")

        def send(self, address, value):
            raise AssertionError("fake coordinator owns this test")

    class AppliedCoordinator:
        def __init__(self, *, prepared_callback=None, **_kwargs):
            self.prepared_callback = prepared_callback

        async def execute(self, plan, context):
            prepared = self.prepared_callback(
                plan,
                context,
                (
                    {
                        "physical_verified": True,
                        "values": {"/ch/3/fdr": -10.0},
                    },
                ),
            )
            if asyncio.iscoroutine(prepared):
                await prepared
            operation_id = plan.operations[0].operation_id
            return ConsoleTransactionResult(
                transaction_id=plan.transaction_id,
                status="applied",
                reason="unit_applied",
                applied_operation_ids=[operation_id],
                events=[
                    {
                        "operation_id": operation_id,
                        "phase": "apply",
                        "result": {
                            "success": True,
                            "physical_verified": True,
                            "evidence": {"writes": ["/ch/3/fdr"]},
                        },
                    }
                ],
            )

    server = AutoMixerServer()
    server.mixer_client = FakeWing()
    server.connection_mode = "wing"
    server._wing_observation_only = False
    server.get_operator_mode_status = lambda: supervised_mode("fader")
    server._record_manual_write_event = lambda *_args, **_kwargs: None
    server._append_manual_write_wal = lambda *_args, **_kwargs: "/tmp/test-wal"
    for index in range(1000):
        key = f"prior-idempotency-{index}"
        transaction_id = f"prior-transaction-{index}"
        server._supervised_transaction_idempotency_keys[key] = transaction_id
        server._supervised_transaction_ids[transaction_id] = key

    fader_contract = next(
        contract
        for contract in FAMILY_CONTRACTS
        if contract.family_id == "fader"
    )
    promoted_contract = replace(
        fader_contract,
        lifecycle=LIFECYCLE_LIVE_ALLOWED,
        hardware_proof="accepted",
        live_allowed_write_kinds=("fader",),
        blockers=(),
    )
    plan = transaction_plan()
    plan["transaction_id"] = "new-transaction-after-1000"
    plan["operations"][0].update(
        operation_id="new-fader-after-1000",
        family_id="fader",
        write_kind="fader",
        target_value=-12.0,
        parameter="channel_fader_db",
    )

    with patch("server.FAMILY_CONTRACTS", (promoted_contract,)), patch(
        "server.SupervisedConsoleTransactionCoordinator",
        AppliedCoordinator,
    ):
        applied = await server._apply_supervised_soundcheck_transaction_unlocked(
            plan,
            approved=True,
            approval_id="idempotency-retention-approval",
            idempotency_key="new-idempotency-after-1000",
        )
        replay = await server._apply_supervised_soundcheck_transaction_unlocked(
            plan,
            approved=True,
            approval_id="idempotency-retention-approval",
            idempotency_key="prior-idempotency-0",
        )

    assert applied["status"] == "applied"
    assert len(server._supervised_transaction_idempotency_keys) == 1001
    assert server._supervised_transaction_idempotency_keys[
        "prior-idempotency-0"
    ] == "prior-transaction-0"
    assert replay["status"] == "blocked"
    assert replay["reason"] == "duplicate_idempotency_key"
    assert replay["no_console_send_proven"] is True


@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
@pytest.mark.asyncio
async def test_audio_failure_retained_cut_is_durable_manual_review_not_rollback_up(
    _mock_config,
    _mock_bleed,
):
    class FakeWing:
        is_connected = True

        def __init__(self):
            self.values = {"/ch/3/fdr": -8.0}
            self.sent = []

        def query_parameter(self, address, timeout):
            return {
                "status": "received",
                "address": address,
                "value": self.values.get(address),
                "confirmed_round_trips": 2,
            }

        def send(self, address, value):
            self.sent.append((address, value))
            self.values[address] = value
            return True

    server = AutoMixerServer()
    server.mixer_client = FakeWing()
    server.connection_mode = "wing"
    server._wing_observation_only = False
    server.get_operator_mode_status = lambda: supervised_mode("fader")
    server._record_manual_write_event = lambda *_args, **_kwargs: None
    wal_events = []

    def append_wal(phase, transaction_id, **fields):
        wal_events.append(
            {
                "phase": phase,
                "transaction_id": transaction_id,
                **fields,
            }
        )
        return "/tmp/test-retained-cut-wal"

    server._append_manual_write_wal = append_wal
    server._supervised_transaction_audio_recheck = AsyncMock(
        return_value=AudioRecheckResult(
            "failed",
            "feedback_risk_detected",
        )
    )

    emergency_initiators = []

    async def emergency_stop(_reason, *, initiator_task=None):
        emergency_initiators.append(initiator_task)
        server._manual_write_emergency_stopped = True
        return {
            "status": "stopped",
            "success": True,
            "teardown_complete": True,
        }

    server.activate_system_emergency_stop = emergency_stop
    fader_contract = next(
        contract for contract in FAMILY_CONTRACTS if contract.family_id == "fader"
    )
    promoted_contract = replace(
        fader_contract,
        lifecycle=LIFECYCLE_LIVE_ALLOWED,
        hardware_proof="accepted",
        live_allowed_write_kinds=("fader",),
        blockers=(),
    )
    plan = transaction_plan()
    plan["transaction_id"] = "retained-cut-manual-review"
    plan["operations"][0].update(
        operation_id="retained-fader-cut",
        family_id="fader",
        write_kind="fader",
        target_value=-12.0,
        channel=3,
        parameter="channel_fader_db",
    )

    with patch("server.FAMILY_CONTRACTS", (promoted_contract,)):
        result = await server._apply_supervised_soundcheck_transaction_unlocked(
            plan,
            approved=True,
            approval_id="retained-cut-approval",
            idempotency_key="retained-cut-idempotency",
        )

    assert server.mixer_client.sent == [("/ch/3/fdr", -12.0)]
    assert server.mixer_client.values["/ch/3/fdr"] == -12.0
    assert result["status"] == "manual_review_required"
    assert result["reason"] == "feedback_risk_detected"
    assert result["wal_terminal"] == "manual_review_required"
    assert result["contained_operation_ids"] == ["retained-fader-cut"]
    assert result["contained_target_physical_verified"] is True
    assert result["audio_failure_recovery_state_known"] is True
    assert result["physical_verified"] is True
    assert result["physical_state_restored"] is False
    assert result["write_state_unknown"] is False
    assert result["rollback_state_unknown"] is False
    assert result["manual_review_required"] is True
    assert result["emergency_stop_activated"] is True
    assert result["emergency_latch_succeeded"] is True
    assert result["emergency_teardown_succeeded"] is True
    assert result["emergency_stop_complete"] is True
    assert result["emergency_stop_teardown_complete"] is True
    assert result["emergency_stop_required"] is False
    assert emergency_initiators == [asyncio.current_task()]
    assert [event["phase"] for event in wal_events] == [
        "prepared",
        "manual_review_required",
    ]
    terminal = wal_events[-1]
    assert terminal["contained_operation_ids"] == ["retained-fader-cut"]
    assert terminal["contained_target_physical_verified"] is True
    assert terminal["physical_state_restored"] is False
    assert terminal["write_state_unknown"] is False
    assert terminal["emergency_stop_complete"] is True
    assert terminal["emergency_stop_required"] is False


@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
@pytest.mark.asyncio
async def test_non_retention_incomplete_emergency_is_preserved_in_wal_and_result(
    _mock_config,
    _mock_bleed,
):
    class FakeWing:
        is_connected = True

        async def query_parameter(self, address, timeout):
            return {
                "status": "received",
                "address": address,
                "value": 0.0,
                "confirmed_round_trips": 2,
            }

    class RollbackFailureAdapter:
        family_id = "eq_hpf"

        def prepare(self, operation):
            return {
                "addresses": ("/ch/3/eq/2/g",),
                "write_count": 1,
                "writes": (
                    {
                        "address": "/ch/3/eq/2/g",
                        "target_value": -2.0,
                        "tolerance": 0.01,
                    },
                ),
                "audible_state_reads": (),
            }

        def validate_snapshot(self, operation, snapshot):
            return None

        async def snapshot(self, operation):
            return {
                "physical_verified": True,
                "values": {"/ch/3/eq/2/g": 0.0},
            }

        async def apply(self, operation, snapshot):
            return AdapterStepResult(
                True,
                True,
                "sent",
                {"writes": [{"sent": True}]},
            )

        async def verify(self, operation, snapshot, apply_result):
            return AdapterStepResult(False, False, "readback_mismatch")

        async def rollback(self, operation, snapshot, apply_result):
            return AdapterStepResult(False, False, "rollback_mismatch")

        async def audio_recheck(self, operation, snapshot, apply_result):
            return AudioRecheckResult("passed")

    server = AutoMixerServer()
    server.mixer_client = FakeWing()
    server.connection_mode = "wing"
    server._wing_observation_only = False
    server.get_operator_mode_status = lambda: supervised_mode("eq")
    server._record_manual_write_event = lambda *_args, **_kwargs: None
    wal_events = []

    def append_wal(phase, transaction_id, **fields):
        wal_events.append(
            {
                "phase": phase,
                "transaction_id": transaction_id,
                **fields,
            }
        )
        return "/tmp/test-incomplete-emergency-wal"

    server._append_manual_write_wal = append_wal

    async def incomplete_stop(_reason, *, initiator_task=None):
        return {
            "status": "stop_incomplete",
            "success": False,
            "teardown_complete": False,
            "teardown_errors": [
                {"step": "unit", "error": "RuntimeError"}
            ],
        }

    server.activate_system_emergency_stop = incomplete_stop
    eq_contract = next(
        contract
        for contract in FAMILY_CONTRACTS
        if contract.family_id == "eq_hpf"
    )
    promoted_contract = replace(
        eq_contract,
        lifecycle=LIFECYCLE_LIVE_ALLOWED,
        hardware_proof="accepted",
        live_allowed_write_kinds=("eq",),
        blockers=(),
    )

    with patch("server.FAMILY_CONTRACTS", (promoted_contract,)), patch(
        "server.build_wing_supervised_family_adapters",
        return_value={"eq_hpf": RollbackFailureAdapter()},
    ):
        result = await server._apply_supervised_soundcheck_transaction_unlocked(
            transaction_plan(),
            approved=True,
            approval_id="rollback-failure-approval",
            idempotency_key="rollback-failure-idempotency",
        )

    assert result["status"] == "manual_review_required"
    assert result["safe_target_retention_attempted"] is False
    assert result["emergency_latch_succeeded"] is True
    assert result["emergency_teardown_succeeded"] is False
    assert result["emergency_stop_complete"] is False
    assert result["emergency_stop_teardown_complete"] is False
    assert result["emergency_stop_required"] is True
    terminal = wal_events[-1]
    assert terminal["phase"] == "manual_review_required"
    assert terminal["emergency_latch_succeeded"] is True
    assert terminal["emergency_teardown_succeeded"] is False
    assert terminal["emergency_stop_complete"] is False
    assert terminal["emergency_stop_required"] is True
    assert [
        event["phase"] for event in terminal["emergency_events"]
    ] == ["emergency_latch", "emergency_teardown"]


@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "failure_phase",
        "expected_status",
        "expected_reason",
        "expected_wal_terminal",
        "expected_wal_calls",
    ),
    [
        (
            "prepare",
            "blocked",
            "durable_write_journal_unavailable",
            "not_prepared",
            ["prepared"],
        ),
        (
            "terminal",
            "manual_review_required",
            "durable_terminal_record_failed",
            "terminal_record_failed",
            ["prepared", "aborted_before_send"],
        ),
    ],
)
async def test_supervised_wal_failure_preserves_incomplete_emergency_truth(
    _mock_config,
    _mock_bleed,
    failure_phase,
    expected_status,
    expected_reason,
    expected_wal_terminal,
    expected_wal_calls,
):
    class FakeWing:
        is_connected = True

        def query_parameter(self, address, timeout):
            raise AssertionError("fake coordinator owns this test")

        def send(self, address, value):
            raise AssertionError("WAL failure test must not write")

    class NoSendCoordinator:
        def __init__(self, *, prepared_callback=None, **_kwargs):
            self.prepared_callback = prepared_callback

        async def execute(self, plan, context):
            try:
                prepared = self.prepared_callback(
                    plan,
                    context,
                    (
                        {
                            "physical_verified": True,
                            "values": {"/ch/3/fdr": -8.0},
                        },
                    ),
                )
                if asyncio.iscoroutine(prepared):
                    await prepared
            except RuntimeError:
                return ConsoleTransactionResult(
                    transaction_id=plan.transaction_id,
                    status="blocked",
                    reason="durable_prepare_failed",
                )
            return ConsoleTransactionResult(
                transaction_id=plan.transaction_id,
                status="blocked",
                reason="unit_no_send",
            )

    server = AutoMixerServer()
    server.mixer_client = FakeWing()
    server.connection_mode = "wing"
    server._wing_observation_only = False
    server.get_operator_mode_status = lambda: supervised_mode("fader")
    server._record_manual_write_event = lambda *_args, **_kwargs: None

    wal_calls = []

    def append_wal(phase, _transaction_id, **_fields):
        wal_calls.append(phase)
        if failure_phase == "prepare":
            return None
        return "/tmp/test-wal" if phase == "prepared" else None

    server._append_manual_write_wal = append_wal
    emergency_reasons = []

    async def incomplete_stop(reason, **_kwargs):
        emergency_reasons.append(reason)
        server._manual_write_emergency_stopped = True
        server._system_emergency_teardown_complete = False
        server._system_emergency_stop_errors = [
            {"step": "unit", "error": "RuntimeError"}
        ]
        return {
            "status": "stop_incomplete",
            "success": False,
            "teardown_complete": False,
            "teardown_errors": list(server._system_emergency_stop_errors),
        }

    server.activate_system_emergency_stop = incomplete_stop
    fader_contract = next(
        contract
        for contract in FAMILY_CONTRACTS
        if contract.family_id == "fader"
    )
    promoted_contract = replace(
        fader_contract,
        lifecycle=LIFECYCLE_LIVE_ALLOWED,
        hardware_proof="accepted",
        live_allowed_write_kinds=("fader",),
        blockers=(),
    )
    plan = transaction_plan()
    plan["transaction_id"] = f"wal-{failure_phase}-incomplete-stop"
    plan["operations"][0].update(
        operation_id=f"wal-{failure_phase}-fader",
        family_id="fader",
        write_kind="fader",
        target_value=-12.0,
        parameter="channel_fader_db",
    )

    with patch("server.FAMILY_CONTRACTS", (promoted_contract,)), patch(
        "server.SupervisedConsoleTransactionCoordinator",
        NoSendCoordinator,
    ):
        result = await server._apply_supervised_soundcheck_transaction_unlocked(
            plan,
            approved=True,
            approval_id="wal-failure-approval",
            idempotency_key=f"wal-{failure_phase}-key",
        )

    assert result["status"] == expected_status
    assert result["reason"] == expected_reason
    assert result["wal_terminal"] == expected_wal_terminal
    assert result["emergency_stop_activated"] is True
    assert result["emergency_stop_complete"] is False
    assert result["emergency_stop_teardown_complete"] is False
    assert result["emergency_stop_required"] is True
    assert wal_calls == expected_wal_calls
    assert emergency_reasons == [
        f"supervised_transaction_wal_{failure_phase}_failed:"
        f"{plan['transaction_id']}"
    ]


@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
@pytest.mark.asyncio
@pytest.mark.parametrize("deadline_mutation", ["missing", "extended"])
async def test_source_analysis_deadline_revokes_server_authority_mid_group(
    _mock_config,
    _mock_bleed,
    deadline_mutation,
):
    class FakeWing:
        is_connected = True

        def query_parameter(self, address, timeout):
            raise AssertionError("deadline probe coordinator must not query")

        def send(self, address, value):
            raise AssertionError("expired source analysis must not write")

    clock = {"now": 1000.0}
    authority_checks = []

    class DeadlineProbeCoordinator:
        def __init__(self, *, adapters, **_kwargs):
            self.adapters = adapters

        async def execute(self, plan, _context):
            authority_checks.append(
                self.adapters["fader"].write_authority_is_current()
            )
            preflight = server._soundcheck_preflight_assessments[
                plan_payload_run_id
            ]
            deadlines = preflight["source_balance_deadline_by_transaction"]
            if deadline_mutation == "missing":
                deadlines.pop(plan.transaction_id)
            else:
                deadlines[plan.transaction_id] = 2000.0
            authority_checks.append(
                self.adapters["fader"].write_authority_is_current()
            )
            return ConsoleTransactionResult(
                transaction_id=plan.transaction_id,
                status="blocked",
                reason="write_authority_revoked",
            )

    server = AutoMixerServer()
    server.mixer_client = FakeWing()
    server.connection_mode = "wing"
    server._wing_observation_only = False
    server.get_operator_mode_status = lambda: supervised_mode("fader")
    server._record_manual_write_event = lambda *_args, **_kwargs: None
    server._soundcheck_console_fence_still_current = (
        lambda *_args, **_kwargs: True
    )
    fader_contract = next(
        contract for contract in FAMILY_CONTRACTS if contract.family_id == "fader"
    )
    promoted_contract = replace(
        fader_contract,
        lifecycle=LIFECYCLE_LIVE_ALLOWED,
        hardware_proof="accepted",
        live_allowed_write_kinds=("fader",),
        blockers=(),
    )
    plan = transaction_plan()
    plan.update(
        transaction_id="source-deadline-group",
        run_id="source-deadline-run",
        plan_revision=2,
        plan_fingerprint="source-deadline-fingerprint",
        operations=[
            {
                "operation_id": "deadline-fader-1",
                "family_id": "fader",
                "write_kind": "fader",
                "target_value": -12.0,
                "channel": 1,
                "parameter": "channel_fader_db",
                "metadata": {"absolute_target": True},
            },
            {
                "operation_id": "deadline-fader-2",
                "family_id": "fader",
                "write_kind": "fader",
                "target_value": -14.0,
                "channel": 2,
                "parameter": "channel_fader_db",
                "metadata": {"absolute_target": True},
            },
        ],
    )
    plan_payload_run_id = plan["run_id"]
    server._soundcheck_manifests = {
        plan_payload_run_id: {
            "run_id": plan_payload_run_id,
            "plans": [plan],
            "source_coverage": {"selected_channel_ids": [1, 2]},
        }
    }
    server._active_soundcheck_run_id = plan["run_id"]
    server._soundcheck_global_blockers = {plan["run_id"]: []}
    server._soundcheck_preflight_assessments = {
        plan["run_id"]: {
            "source_balance_analysis_deadline_epoch": 1001.0,
            "source_balance_deadline_by_transaction": {
                plan["transaction_id"]: 1001.0,
            },
            "child_local_blockers_by_transaction": {},
            "console_change_fence": {"verified": True},
        }
    }

    with patch("server.FAMILY_CONTRACTS", (promoted_contract,)), patch(
        "server.SupervisedConsoleTransactionCoordinator",
        DeadlineProbeCoordinator,
    ), patch("server.time.time", side_effect=lambda: clock["now"]):
        result = await server._apply_supervised_soundcheck_transaction_unlocked(
            plan,
            approved=True,
            approval_id="deadline-approval",
            idempotency_key="deadline-idempotency",
        )

    assert authority_checks == [True, False]
    assert result["status"] == "blocked"
    assert result["reason"] == "write_authority_revoked"
    assert result["no_console_send_proven"] is True


@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
@pytest.mark.asyncio
async def test_source_analysis_deadline_revokes_non_fader_authority_mid_transaction(
    _mock_config,
    _mock_bleed,
):
    class FakeWing:
        is_connected = True

        def query_parameter(self, address, timeout):
            raise AssertionError("deadline probe coordinator must not query")

        def send(self, address, value):
            raise AssertionError("expired source analysis must not write")

    clock = {"now": 1000.0}
    authority_checks = []

    class DeadlineProbeCoordinator:
        def __init__(self, *, adapters, **_kwargs):
            self.adapters = adapters

        async def execute(self, plan, _context):
            authority_checks.append(
                self.adapters["eq_hpf"].write_authority_is_current()
            )
            clock["now"] = 1002.0
            authority_checks.append(
                self.adapters["eq_hpf"].write_authority_is_current()
            )
            return ConsoleTransactionResult(
                transaction_id=plan.transaction_id,
                status="blocked",
                reason="write_authority_revoked_before_apply",
            )

    server = AutoMixerServer()
    server.mixer_client = FakeWing()
    server.connection_mode = "wing"
    server._wing_observation_only = False
    server.get_operator_mode_status = lambda: supervised_mode("eq")
    server._record_manual_write_event = lambda *_args, **_kwargs: None
    server._soundcheck_console_fence_still_current = (
        lambda *_args, **_kwargs: True
    )
    eq_contract = next(
        contract for contract in FAMILY_CONTRACTS if contract.family_id == "eq_hpf"
    )
    promoted_contract = replace(
        eq_contract,
        lifecycle=LIFECYCLE_LIVE_ALLOWED,
        hardware_proof="accepted",
        live_allowed_write_kinds=("eq",),
        blockers=(),
    )
    plan = transaction_plan()
    plan.update(
        transaction_id="source-deadline-tonal",
        run_id="source-deadline-tonal-run",
        plan_revision=1,
        plan_fingerprint="source-deadline-tonal-fingerprint",
    )
    server._soundcheck_manifests = {
        plan["run_id"]: {
            "run_id": plan["run_id"],
            "plans": [plan],
            "source_coverage": {"selected_channel_ids": [3]},
        }
    }
    server._active_soundcheck_run_id = plan["run_id"]
    server._soundcheck_global_blockers = {plan["run_id"]: []}
    server._soundcheck_preflight_assessments = {
        plan["run_id"]: {
            "source_balance_analysis_deadline_epoch": 1001.0,
            "source_balance_deadline_by_transaction": {
                plan["transaction_id"]: 1001.0,
            },
            "child_local_blockers_by_transaction": {},
            "console_change_fence": {"verified": True},
        }
    }

    with patch("server.FAMILY_CONTRACTS", (promoted_contract,)), patch(
        "server.SupervisedConsoleTransactionCoordinator",
        DeadlineProbeCoordinator,
    ), patch("server.time.time", side_effect=lambda: clock["now"]):
        result = await server._apply_supervised_soundcheck_transaction_unlocked(
            plan,
            approved=True,
            approval_id="deadline-tonal-approval",
            idempotency_key="deadline-tonal-idempotency",
        )

    assert authority_checks == [True, False]
    assert result["status"] == "blocked"
    assert result["reason"] == "write_authority_revoked_before_apply"
    assert result["no_console_send_proven"] is True


def test_decision_queue_limit_bounds_evidence_rich_pending_rows():
    queue = OperatorProposalQueue()
    mode = supervised_mode()
    for channel in range(1, 6):
        queue.create(
            {
                "id": f"bounded-{channel}",
                "title": f"Channel {channel}",
                "kind": "fader",
                "requested_change": {
                    "value_type": "fader",
                    "channel": channel,
                    "value": -12.0,
                },
                "large_evidence": "x" * 10_000,
            },
            mode,
        )

    server = type(
        "Server",
        (),
        {
            "operator_proposal_queue": queue,
            "mixing_agent": None,
            "get_operator_mode_status": lambda self: mode,
        },
    )()

    snapshot = build_decision_queue(server, limit=2)

    assert [row["id"] for row in snapshot["pending_actions"]] == [
        "bounded-4",
        "bounded-5",
    ]
    assert snapshot["summary"]["pending_count"] == 2


def test_applied_or_claimed_proposal_cannot_be_claimed_again(tmp_path):
    queue = evidence_queue(tmp_path, "claim-once")
    created = queue.create(
        {
            "id": "claim-once",
            "title": "One reviewed fader move",
            "requested_change": {
                "value_type": "fader",
                "channel": 3,
                "value": -12.0,
            },
        },
        supervised_mode("fader"),
    )
    proposal = queue.find(proposal_id=created["proposal"]["id"])

    evidence_token = reserve_evidence(queue, proposal)
    assert queue.claim_for_apply(proposal)["claimed"] is True
    assert queue.claim_for_apply(proposal) == {
        "claimed": False,
        "reason": "proposal_not_pending",
        "proposal_status": "applying",
    }
    queue.mark_apply_result(
        proposal,
        verified_apply_result(),
        evidence_reservation_binding=evidence_token,
    )
    historical = queue.find(proposal_id=proposal.id)
    assert historical.status == "applied"
    assert queue.claim_for_apply(historical)["claimed"] is False


def test_verified_already_satisfied_apply_is_green_and_not_a_hazard(tmp_path):
    queue = evidence_queue(tmp_path, "already-satisfied-apply")
    created = queue.create(
        {
            "id": "already-satisfied-apply",
            "title": "Already satisfied reviewed operation",
            "requested_change": {
                "value_type": "fader",
                "channel": 3,
                "value": -12.0,
            },
        },
        supervised_mode("fader"),
    )
    proposal = queue.find(proposal_id=created["proposal"]["id"])
    evidence_token = reserve_evidence(queue, proposal)
    assert queue.claim_for_apply(proposal)["claimed"] is True

    completed = queue.mark_apply_result(
        proposal,
        verified_already_satisfied_apply_result(),
        evidence_reservation_binding=evidence_token,
    )

    assert completed["status"] == "applied"
    assert completed["proposal"]["status"] == "applied"
    assert completed["proposal"]["result"]["no_console_send_proven"] is True
    assert queue.iter_terminal_hazards() == ()


@pytest.mark.parametrize("no_send_value", [False, None, "missing"])
def test_possible_console_send_is_terminal_even_when_result_says_blocked(
    no_send_value,
    tmp_path,
):
    queue = evidence_queue(tmp_path, f"possible-send-{no_send_value}")
    created = queue.create(
        {
            "id": "possible-send-terminal",
            "title": "Reviewed fader transaction",
            "requested_change": {
                "value_type": "fader",
                "channel": 3,
                "value": -12.0,
            },
        },
        supervised_mode("fader"),
    )
    proposal = queue.find(proposal_id=created["proposal"]["id"])
    evidence_token = reserve_evidence(queue, proposal)
    assert queue.claim_for_apply(proposal)["claimed"] is True

    apply_result = {
        "status": "blocked",
        "reason": "write_authority_revoked",
        "wal_terminal": "manual_review_required",
        "manual_review_required": True,
        "events": [{"phase": "apply", "result": {"writes": ["/ch/3/fdr"]}}],
    }
    if no_send_value != "missing":
        apply_result["no_console_send_proven"] = no_send_value
    result = queue.mark_apply_result(
        proposal,
        apply_result,
        evidence_reservation_binding=evidence_token,
    )

    assert result["status"] == "apply_error"
    assert result["proposal"]["status"] == "apply_error"
    assert result["proposal"]["can_apply"] is False
    assert queue.summary()["pending_count"] == 0
    assert queue.claim_for_apply(queue.find(proposal_id=proposal.id))[
        "claimed"
    ] is False
    apply_result["events"][0]["result"]["writes"][0] = "/tampered"
    assert queue.find(proposal_id=proposal.id).result["events"][0]["result"][
        "writes"
    ] == ["/ch/3/fdr"]


def test_explicit_possible_send_is_terminal_without_optional_event_markers(tmp_path):
    queue = evidence_queue(tmp_path, "possible-send-no-markers")
    created = queue.create(
        {
            "id": "possible-send-no-markers",
            "title": "Reviewed fader transaction",
            "requested_change": {
                "value_type": "fader",
                "channel": 3,
                "value": -12.0,
            },
        },
        supervised_mode("fader"),
    )
    proposal = queue.find(proposal_id=created["proposal"]["id"])
    evidence_token = reserve_evidence(queue, proposal)
    assert queue.claim_for_apply(proposal)["claimed"] is True

    result = queue.mark_apply_result(
        proposal,
        {
            "status": "blocked",
            "reason": "write_authority_revoked",
            "no_console_send_proven": False,
        },
        evidence_reservation_binding=evidence_token,
    )

    assert result["status"] == "apply_error"
    assert result["proposal"]["status"] == "apply_error"
    assert result["proposal"]["result"]["manual_review_required"] is True
    assert result["proposal"]["result"]["fresh_analysis_required"] is True
    assert queue.summary()["pending_count"] == 0


@pytest.mark.parametrize(
    "reason",
    [
        "operator_approval_expired",
        "operator_approval_expired_during_snapshot",
        "operator_approval_expired_after_snapshot",
        "operator_approval_expired_before_apply",
        "operator_approval_expired_before_commit",
        "proposal_expired",
    ],
)
def test_claimed_expiry_is_terminal_and_requires_fresh_analysis(reason, tmp_path):
    queue = evidence_queue(tmp_path, f"expiry-{reason}")
    created = queue.create(
        {
            "id": f"terminal-expiry-{reason}",
            "title": "Expiring reviewed correction",
            "requested_change": {
                "value_type": "fader",
                "channel": 3,
                "value": -12.0,
            },
        },
        supervised_mode("fader"),
    )
    proposal = queue.find(proposal_id=created["proposal"]["id"])
    evidence_token = reserve_evidence(queue, proposal)
    assert queue.claim_for_apply(proposal)["claimed"] is True

    result = queue.mark_apply_result(
        proposal,
        {
            "status": "blocked",
            "reason": reason,
            "no_console_send_proven": True,
        },
        evidence_reservation_binding=evidence_token,
    )

    assert result["status"] == "apply_blocked"
    assert result["proposal"]["status"] == "apply_blocked"
    assert result["proposal"]["result"]["fresh_analysis_required"] is True
    assert queue.summary()["pending_count"] == 0
    assert queue.find(proposal_id=proposal.id).status == "apply_blocked"


def test_historical_proposal_id_cannot_be_reused(tmp_path):
    queue = evidence_queue(tmp_path, "immutable-id")
    created = queue.create(
        {
            "id": "immutable-id",
            "title": "First correction",
            "requested_change": {
                "value_type": "fader",
                "channel": 3,
                "value": -12.0,
            },
        },
        supervised_mode("fader"),
    )
    proposal = queue.find(proposal_id=created["proposal"]["id"])
    evidence_token = reserve_evidence(queue, proposal)
    assert queue.claim_for_apply(proposal)["claimed"] is True
    assert queue.mark_apply_result(
        proposal,
        verified_apply_result(),
        evidence_reservation_binding=evidence_token,
    )["status"] == "applied"

    duplicate = queue.create(
        {
            "id": "immutable-id",
            "title": "Different correction under stale id",
            "requested_change": {
                "value_type": "fader",
                "channel": 4,
                "value": -18.0,
            },
        },
        supervised_mode("fader"),
    )

    assert duplicate["status"] == "blocked"
    assert duplicate["reason"] == "proposal_id_already_used"
    assert queue.summary()["pending_count"] == 0


@pytest.mark.parametrize("action", ["accept", "dismiss"])
def test_applying_proposal_cannot_be_accepted_or_dismissed(action, tmp_path):
    queue = evidence_queue(tmp_path, f"lifecycle-{action}")
    created = queue.create(
        {
            "id": f"lifecycle-{action}",
            "title": "Claimed correction",
            "requested_change": {
                "value_type": "fader",
                "channel": 3,
                "value": -12.0,
            },
        },
        supervised_mode("fader"),
    )
    proposal = queue.find(proposal_id=created["proposal"]["id"])
    evidence_token = reserve_evidence(queue, proposal)
    assert queue.claim_for_apply(proposal)["claimed"] is True
    assert proposal.to_dict()["can_apply"] is False

    result = getattr(queue, action)(proposal_id=proposal.id)

    assert result["status"] == "blocked"
    assert result["reason"] == "proposal_not_pending"
    assert queue.summary()["pending_count"] == 1
    assert queue.mark_apply_result(
        proposal,
        verified_apply_result(),
        evidence_reservation_binding=evidence_token,
    )["status"] == "applied"
    assert queue.summary()["history_count"] == 1


@pytest.mark.parametrize("channel", [True, 2.9, 0, 49])
def test_queue_rejects_non_physical_channel_targets(channel):
    queue = OperatorProposalQueue()

    created = queue.create(
        {
            "id": f"invalid-channel-{channel}",
            "title": "Invalid channel",
            "value_type": "fader",
            "channel": channel,
            "value": -12.0,
        },
        supervised_mode("fader"),
    )

    assert created["proposal"]["can_apply"] is False
    assert created["proposal"]["requested_change"] is None


def test_claim_rechecks_expiry_atomically():
    queue = OperatorProposalQueue()
    created = queue.create(
        {
            "id": "expired-at-claim",
            "title": "Expired correction",
            "value_type": "fader",
            "channel": 3,
            "value": -12.0,
        },
        supervised_mode("fader"),
    )
    proposal = queue.find(proposal_id=created["proposal"]["id"])
    proposal.expires_at = "2000-01-01T00:00:00Z"

    claim = queue.claim_for_apply(proposal)

    assert claim["claimed"] is False
    assert claim["reason"] == "proposal_expired"


def test_expired_pending_reimport_is_not_reported_as_live_existing_proposal():
    queue = OperatorProposalQueue()
    payload = {
        "id": "expired-reimport",
        "title": "Expired soundcheck child",
        "value_type": "fader",
        "channel": 3,
        "value": -12.0,
    }
    created = queue.create(payload, supervised_mode("fader"))
    proposal = queue.find(proposal_id=created["proposal"]["id"])
    proposal.expires_at = "2000-01-01T00:00:00Z"

    duplicate = queue.create(payload, supervised_mode("fader"))

    assert duplicate["status"] == "blocked"
    assert duplicate["reason"] == (
        "proposal_expired_fresh_analysis_required"
    )
    assert queue.summary()["pending_count"] == 0
    historical = queue.find(proposal_id="expired-reimport")
    assert historical.status == "apply_blocked"
    assert historical.result["fresh_analysis_required"] is True


@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
@pytest.mark.asyncio
async def test_server_transaction_path_blocks_unproven_write_kind_before_wing_write(
    _mock_config,
    _mock_bleed,
):
    class FakeWing:
        is_connected = True
        sent = []

        def query_parameter(self, address, timeout):
            raise AssertionError("unproven family must block before physical query")

        def send(self, address, value):
            self.sent.append((address, value))
            return True

    server = AutoMixerServer()
    server.mixer_client = FakeWing()
    server.get_operator_mode_status = lambda: supervised_mode("fader")
    plan = transaction_plan()
    plan["operations"][0].update(
        family_id="fader",
        write_kind="fader",
        target_value=-12.0,
        channel=3,
    )

    result = await server.apply_supervised_soundcheck_transaction(
        plan,
        approved=True,
        approval_id="operator-test",
        idempotency_key="idempotency-test",
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "write_kind_not_live_or_proof_allowed"
    assert server.mixer_client.sent == []


@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
@pytest.mark.asyncio
async def test_server_transaction_path_blocks_empty_operator_allowlist_before_query(
    _mock_config,
    _mock_bleed,
):
    class FakeWing:
        is_connected = True
        sent = []

        def query_parameter(self, address, timeout):
            raise AssertionError("empty allowlist must block before physical query")

    server = AutoMixerServer()
    server.mixer_client = FakeWing()
    server.get_operator_mode_status = lambda: supervised_mode()

    result = await server.apply_supervised_soundcheck_transaction(
        transaction_plan(),
        approved=True,
        approval_id="operator-test",
        idempotency_key="empty-allowlist",
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "live_write_kind_not_allowed_in_mode"
    assert result["allowed_live_write_kinds"] == []


@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
@pytest.mark.asyncio
async def test_server_transaction_path_blocks_wing_observation_only_before_query(
    _mock_config,
    _mock_bleed,
):
    class FakeWing:
        is_connected = True

        def query_parameter(self, address, timeout):
            raise AssertionError("observation-only must block before physical query")

    server = AutoMixerServer()
    server.connection_mode = "wing"
    server._wing_observation_only = True
    server.mixer_client = FakeWing()
    server.get_operator_mode_status = lambda: supervised_mode("fader")
    plan = transaction_plan()
    plan["operations"][0].update(
        family_id="fader",
        write_kind="fader",
        target_value=-12.0,
    )

    result = await server.apply_supervised_soundcheck_transaction(
        plan,
        approved=True,
        approval_id="operator-test",
        idempotency_key="observation-only",
    )

    assert result == {
        "status": "blocked",
        "success": False,
        "reason": "wing_observation_only_active",
        "observation_only": True,
        "no_console_send_proven": True,
    }


@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
@pytest.mark.asyncio
async def test_remote_operator_path_cannot_enable_hardware_proof_mode(
    _mock_config,
    _mock_bleed,
):
    server = AutoMixerServer()

    result = await server.apply_supervised_soundcheck_transaction(
        transaction_plan(),
        approved=True,
        approval_id="operator-test",
        idempotency_key="idempotency-test",
        proof_run=True,
    )

    assert result == {
        "status": "blocked",
        "success": False,
        "reason": "restore_only_proof_grant_required",
        "no_console_send_proven": True,
    }


@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
@pytest.mark.asyncio
async def test_transaction_parser_rejects_fractional_channel_before_query(
    _mock_config,
    _mock_bleed,
):
    class FakeWing:
        is_connected = True

        def query_parameter(self, address, timeout):
            raise AssertionError("invalid channel must block before query")

    server = AutoMixerServer()
    server.mixer_client = FakeWing()
    server.get_operator_mode_status = lambda: supervised_mode("fader")
    plan = transaction_plan()
    plan["operations"][0].update(
        family_id="fader",
        write_kind="fader",
        target_value=-12.0,
        channel=3.9,
    )

    result = await server.apply_supervised_soundcheck_transaction(
        plan,
        approved=True,
        approval_id="operator-test",
        idempotency_key="fractional-channel",
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "transaction_channel_must_be_integer_1_to_48"


@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
@pytest.mark.asyncio
async def test_manual_and_transaction_paths_share_one_server_write_lock(
    _mock_config,
    _mock_bleed,
):
    server = AutoMixerServer()
    active = 0
    maximum_active = 0

    async def critical_section(result):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return result

    async def fake_transaction(_plan_payload, **_kwargs):
        return await critical_section({"status": "blocked", "reason": "unit"})

    async def fake_manual(_websocket, **_kwargs):
        return await critical_section({"status": "blocked", "reason": "unit"})

    server._apply_supervised_soundcheck_transaction_unlocked = fake_transaction
    server._apply_manual_console_write_unlocked = fake_manual

    await asyncio.gather(
        server.apply_supervised_soundcheck_transaction(
            transaction_plan(),
            approved=True,
            approval_id="operator-test",
            idempotency_key="lock-transaction",
        ),
        server._apply_manual_console_write(
            None,
            channel=3,
            value=-12.0,
            value_type="fader",
            approved=True,
            approval_id="operator-test",
        ),
    )

    assert maximum_active == 1


@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
@pytest.mark.asyncio
async def test_claimed_soundcheck_approval_expires_while_waiting_for_writer(
    _mock_config,
    _mock_bleed,
    tmp_path,
):
    class FakeWing:
        is_connected = True

        def query_parameter(self, address, timeout):
            raise AssertionError("expired approval must block before physical query")

    server = AutoMixerServer()
    server._console_runtime_dir = tmp_path
    server._console_writer_lock_path = tmp_path / "console_writer.lock"
    server.mixer_client = FakeWing()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def hold_writer():
        async with server._console_write_scope():
            entered.set()
            await release.wait()

    holder = asyncio.create_task(hold_writer())
    await entered.wait()
    plan = transaction_plan()
    plan.update(
        run_id="approval-expiry-run",
        plan_revision=1,
        plan_fingerprint="approval-expiry-fingerprint",
    )
    try:
        result = await server.apply_supervised_soundcheck_transaction(
            plan,
            approved=True,
            approval_id="operator-expiring",
            idempotency_key="approval-expiring-idempotency",
            approval_expires_at_epoch=time.time() + 0.05,
        )
    finally:
        release.set()
        await holder

    assert result["status"] == "blocked"
    assert result["reason"] == "operator_approval_expired"


@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
@pytest.mark.asyncio
async def test_claimed_soundcheck_approval_expires_during_fresh_preflight(
    _mock_config,
    _mock_bleed,
    tmp_path,
):
    server = AutoMixerServer()
    server._console_runtime_dir = tmp_path
    server._console_writer_lock_path = tmp_path / "console_writer.lock"
    apply_called = False

    async def delayed_refresh(_plan):
        await asyncio.sleep(0.05)
        return {"refreshed": True}

    def allowed(_proposal):
        return {"allowed": True}

    async def fake_apply(*_args, **_kwargs):
        nonlocal apply_called
        apply_called = True
        return {"status": "applied"}

    server.refresh_soundcheck_child_preflight = delayed_refresh
    server.validate_soundcheck_child_for_apply = allowed
    server._apply_supervised_soundcheck_transaction_unlocked = fake_apply
    plan = transaction_plan()
    plan.update(
        run_id="approval-expiry-preflight",
        plan_revision=1,
        plan_fingerprint="approval-expiry-preflight-fingerprint",
    )

    result = await server.apply_supervised_soundcheck_transaction(
        plan,
        approved=True,
        approval_id="operator-expiring-preflight",
        idempotency_key="approval-expiring-preflight-idempotency",
        approval_expires_at_epoch=time.time() + 0.02,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "operator_approval_expired"
    assert apply_called is False
