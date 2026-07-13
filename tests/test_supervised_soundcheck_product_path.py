from __future__ import annotations

import asyncio
import copy
from dataclasses import replace
import time
from unittest.mock import patch

import pytest

from operator_proposal_queue import OperatorProposalQueue
from operator_recommendation_bridge import build_soundcheck_operator_proposals
from operator_product_state import build_decision_queue
from server import AutoMixerServer
from supervised_console_parameter_registry import (
    FAMILY_CONTRACTS,
    LIFECYCLE_LIVE_ALLOWED,
)
from supervised_console_transaction import ConsoleTransactionResult


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
    assert server._supervised_transaction_idempotency_keys == {}
    assert server._supervised_transaction_ids == {}


@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
@pytest.mark.asyncio
async def test_verified_transaction_rollback_is_terminal_fresh_analysis_block(
    _mock_config,
    _mock_bleed,
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

    queue = OperatorProposalQueue()
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
    assert queue.claim_for_apply(proposal)["claimed"] is True
    terminal = queue.mark_apply_result(proposal, apply_result)

    assert terminal["status"] == "apply_blocked"
    assert terminal["proposal"]["status"] == "apply_blocked"
    assert terminal["proposal"]["can_apply"] is False
    assert queue.summary()["pending_count"] == 0


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


def test_applied_or_claimed_proposal_cannot_be_claimed_again():
    queue = OperatorProposalQueue()
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

    assert queue.claim_for_apply(proposal)["claimed"] is True
    assert queue.claim_for_apply(proposal) == {
        "claimed": False,
        "reason": "proposal_not_pending",
        "proposal_status": "applying",
    }
    queue.mark_apply_result(proposal, {"status": "applied"})
    historical = queue.find(proposal_id=proposal.id)
    assert historical.status == "applied"
    assert queue.claim_for_apply(historical)["claimed"] is False


@pytest.mark.parametrize("no_send_value", [False, None, "missing"])
def test_possible_console_send_is_terminal_even_when_result_says_blocked(
    no_send_value,
):
    queue = OperatorProposalQueue()
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
    result = queue.mark_apply_result(proposal, apply_result)

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


def test_explicit_possible_send_is_terminal_without_optional_event_markers():
    queue = OperatorProposalQueue()
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
    assert queue.claim_for_apply(proposal)["claimed"] is True

    result = queue.mark_apply_result(
        proposal,
        {
            "status": "blocked",
            "reason": "write_authority_revoked",
            "no_console_send_proven": False,
        },
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
def test_claimed_expiry_is_terminal_and_requires_fresh_analysis(reason):
    queue = OperatorProposalQueue()
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
    assert queue.claim_for_apply(proposal)["claimed"] is True

    result = queue.mark_apply_result(
        proposal,
        {
            "status": "blocked",
            "reason": reason,
            "no_console_send_proven": True,
        },
    )

    assert result["status"] == "apply_blocked"
    assert result["proposal"]["status"] == "apply_blocked"
    assert result["proposal"]["result"]["fresh_analysis_required"] is True
    assert queue.summary()["pending_count"] == 0
    assert queue.find(proposal_id=proposal.id).status == "apply_blocked"


def test_historical_proposal_id_cannot_be_reused():
    queue = OperatorProposalQueue()
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
    assert queue.claim_for_apply(proposal)["claimed"] is True
    assert queue.mark_apply_result(proposal, {"status": "applied"})["status"] == "applied"

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
def test_applying_proposal_cannot_be_accepted_or_dismissed(action):
    queue = OperatorProposalQueue()
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
    assert queue.claim_for_apply(proposal)["claimed"] is True
    assert proposal.to_dict()["can_apply"] is False

    result = getattr(queue, action)(proposal_id=proposal.id)

    assert result["status"] == "blocked"
    assert result["reason"] == "proposal_not_pending"
    assert queue.summary()["pending_count"] == 1
    assert queue.mark_apply_result(proposal, {"status": "applied"})["status"] == "applied"
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
        "reason": "hardware_proof_requires_local_proof_runner",
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
