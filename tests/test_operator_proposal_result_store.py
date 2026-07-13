import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import stat
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from operator_proposal_result_store import (
    ResultEvidenceError,
    ResultEvidenceReservationBinding,
    ResultEvidenceStore,
)
from operator_proposal_queue import OperatorProposalQueue


def _disk_with_free(free):
    return lambda _path: SimpleNamespace(total=free * 2, used=0, free=free)


def _result(**overrides):
    return {
        "status": "applied",
        "no_console_send_proven": False,
        "wal_terminal": "committed",
        "physical_verified": True,
        "audio_recheck_passed": True,
        "physical_state_restored": True,
        "write_state_unknown": False,
        "rollback_state_unknown": False,
        "manual_review_required": False,
        **overrides,
    }


def _reserve(store, marker="test"):
    proposal_id = f"test:{marker}"
    review_content_digest = "sha256:" + ("0" * 64)
    return ResultEvidenceReservationBinding(
        token=store.reserve(
            proposal_id=proposal_id,
            review_content_digest=review_content_digest,
        ),
        proposal_id=proposal_id,
        review_content_digest=review_content_digest,
    )


def _binding(token, proposal_id, review_content_digest):
    return ResultEvidenceReservationBinding(
        token=token,
        proposal_id=proposal_id,
        review_content_digest=review_content_digest,
    )


def _mode():
    return {
        "mode": "supervised",
        "capabilities": {
            "can_create_proposals": True,
            "can_apply_to_console": True,
            "allowed_live_write_kinds": ["fader"],
        },
    }


def _claimed(queue, proposal_id="proposal-1"):
    created = queue.create(
        {
            "id": proposal_id,
            "title": "Lead vocal",
            "requested_change": {
                "value_type": "fader",
                "channel": 1,
                "value": -9.0,
            },
        },
        _mode(),
    )
    assert created["status"] == "created"
    proposal = queue.find(proposal_id=proposal_id)
    claim = queue.claim_verified_for_apply(
        proposal,
        expected_review_contract_version=proposal.review_contract_version,
        expected_review_content_digest=proposal.review_content_digest,
    )
    assert claim["claimed"] is True
    return proposal


def test_store_reserves_then_persists_digest_only_atomic_artifact(tmp_path):
    store = ResultEvidenceStore(
        tmp_path / "results",
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )

    token = _reserve(store)
    record = store.commit(
        token,
        _result(proposal_id="../../operator-controlled-path", marker="ёж"),
    )

    assert record.digest.startswith("sha256:")
    assert record.artifact_name == f"{record.digest[7:]}.json"
    assert "operator-controlled" not in record.artifact_name
    path = store.root / record.artifact_name
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(store.load(record.digest)) ["marker"] == "ёж"
    assert store.active_reservation_count == 0


def test_store_rejects_symlink_target_without_replacing_it(tmp_path):
    store = ResultEvidenceStore(
        tmp_path / "results",
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    result = _result(marker="symlink")
    _content, digest, _pinned = store.canonical_result(result)
    token = _reserve(store)
    outside = tmp_path / "outside"
    outside.write_text("untouched", encoding="utf-8")
    (store.root / f"{digest[7:]}.json").symlink_to(outside)

    with pytest.raises(ResultEvidenceError) as exc_info:
        store.commit(token, result)

    assert exc_info.value.code == "operator_proposal_result_evidence_unsafe_target"
    assert outside.read_text(encoding="utf-8") == "untouched"
    assert store.active_reservation_count == 1
    assert json.loads(
        (store.root / f".reservation.{token.token}").read_text(encoding="utf-8")
    )["phase"] == "persistence_failed_after_send"


def test_reservation_fails_before_send_when_free_space_cannot_cover_slot(tmp_path):
    store = ResultEvidenceStore(
        tmp_path / "results",
        max_artifact_bytes=128,
        min_free_bytes=512,
        disk_usage=_disk_with_free(639),
    )

    with pytest.raises(ResultEvidenceError) as exc_info:
        _reserve(store)

    assert (
        exc_info.value.code
        == "operator_proposal_result_evidence_capacity_unavailable"
    )
    assert store.active_reservation_count == 0


def test_capacity_evicts_known_safe_evidence_but_never_pinned_unknown_write(tmp_path):
    store = ResultEvidenceStore(
        tmp_path / "results",
        max_artifact_bytes=512,
        max_artifacts=1,
        max_total_bytes=4_096,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    safe = store.commit(_reserve(store), _result(marker="safe"))
    assert (store.root / safe.artifact_name).exists()

    replacement_token = _reserve(store)
    assert not (store.root / safe.artifact_name).exists()
    store.release(replacement_token)

    pinned = store.commit(
        _reserve(store),
        _result(
            marker="unknown",
            no_console_send_proven=False,
            physical_state_restored=False,
            write_state_unknown=True,
            manual_review_required=True,
        ),
    )
    with pytest.raises(ResultEvidenceError) as exc_info:
        _reserve(store)

    assert (
        exc_info.value.code
        == "operator_proposal_result_recovery_required"
    )
    assert (store.root / pinned.artifact_name).exists()


def test_possible_send_commit_failure_retains_reservation_and_leaves_no_partial_file(
    tmp_path,
    monkeypatch,
):
    store = ResultEvidenceStore(
        tmp_path / "results",
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    token = _reserve(store)
    original_replace = os.replace

    def fail_artifact_replace(source, target, **kwargs):
        if str(target).startswith(".reservation."):
            return original_replace(source, target, **kwargs)
        raise OSError("disk failure")

    monkeypatch.setattr(os, "replace", fail_artifact_replace)
    with pytest.raises(ResultEvidenceError) as exc_info:
        store.commit(token, _result(marker="failure"))

    assert (
        exc_info.value.code
        == "operator_proposal_result_evidence_persistence_failed"
    )
    assert store.active_reservation_count == 1
    marker = store.root / f".reservation.{token.token}"
    assert marker.exists()
    assert json.loads(marker.read_text(encoding="utf-8"))["phase"] == (
        "persistence_failed_after_send"
    )
    assert store.current_apply_hazards()[0]["phase"] == (
        "persistence_failed_after_send"
    )


def test_reservation_binds_proposal_digest_and_crosses_send_phase(tmp_path):
    store = ResultEvidenceStore(
        tmp_path / "results",
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    digest = "sha256:" + ("a" * 64)
    token = store.reserve(
        proposal_id="bound-proposal",
        review_content_digest=digest,
    )
    binding = _binding(token, "bound-proposal", digest)
    marker = store.root / f".reservation.{token}"
    reserved = json.loads(marker.read_text(encoding="utf-8"))

    assert reserved["proposal_id"] == "bound-proposal"
    assert reserved["review_content_digest"] == digest
    assert reserved["phase"] == "reserved_no_send"

    store.mark_send_may_have_occurred(binding)
    send_phase = json.loads(marker.read_text(encoding="utf-8"))
    assert send_phase["phase"] == "send_may_have_occurred"


@pytest.mark.parametrize("operation", ["transition", "commit"])
@pytest.mark.parametrize("mismatch", ["proposal", "digest"])
def test_cross_reservation_binding_mismatch_never_mutates_or_consumes_markers(
    tmp_path,
    operation,
    mismatch,
):
    store = ResultEvidenceStore(
        tmp_path / f"results-{operation}-{mismatch}",
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    digest_a = "sha256:" + ("a" * 64)
    digest_b = "sha256:" + ("b" * 64)
    token_a = store.reserve(
        proposal_id="proposal-a",
        review_content_digest=digest_a,
    )
    token_b = store.reserve(
        proposal_id="proposal-b",
        review_content_digest=digest_b,
    )
    binding_a = _binding(token_a, "proposal-a", digest_a)
    binding_b = _binding(token_b, "proposal-b", digest_b)
    forged = _binding(
        token_b,
        "proposal-a" if mismatch == "proposal" else "proposal-b",
        digest_b if mismatch == "proposal" else digest_a,
    )

    with pytest.raises(ResultEvidenceError) as exc_info:
        if operation == "transition":
            store.mark_send_may_have_occurred(forged)
        else:
            store.commit(forged, _result(marker="must-not-persist"))

    assert exc_info.value.code == (
        "operator_proposal_result_evidence_reservation_binding_mismatch"
    )
    marker_a = store.root / f".reservation.{token_a}"
    marker_b = store.root / f".reservation.{token_b}"
    assert marker_a.exists() and marker_b.exists()
    assert json.loads(marker_a.read_text(encoding="utf-8"))["phase"] == (
        "reserved_no_send"
    )
    assert json.loads(marker_b.read_text(encoding="utf-8"))["phase"] == (
        "reserved_no_send"
    )
    assert list(store.root.glob("*.json")) == []
    assert store.active_reservation_count == 2

    if operation == "transition":
        store.mark_send_may_have_occurred(binding_b)
        assert json.loads(marker_b.read_text(encoding="utf-8"))["phase"] == (
            "send_may_have_occurred"
        )
    else:
        record = store.commit(binding_b, _result(marker="correct-binding"))
        assert (store.root / record.artifact_name).exists()
    store.release(binding_a)


def test_post_send_commit_failure_survives_restart_as_global_hazard(tmp_path):
    root = tmp_path / "results"
    first = ResultEvidenceStore(
        root,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
        process_instance_id="a" * 32,
    )
    digest = "sha256:" + ("b" * 64)
    token = first.reserve(
        proposal_id="restart-hazard",
        review_content_digest=digest,
    )
    binding = _binding(token, "restart-hazard", digest)
    first.mark_send_may_have_occurred(binding)
    cyclic = {"status": "blocked", "no_console_send_proven": False}
    cyclic["self"] = cyclic

    with pytest.raises(ResultEvidenceError):
        first.commit(binding, cyclic)

    marker = root / f".reservation.{token}"
    assert marker.exists()
    assert json.loads(marker.read_text(encoding="utf-8"))["phase"] == (
        "persistence_failed_after_send"
    )

    restarted = ResultEvidenceStore(
        root,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
        process_instance_id="c" * 32,
    )
    queue = OperatorProposalQueue(result_store=restarted)

    assert queue.result_persistence_failed is False
    assert queue.result_evidence_contract_ready is False
    assert queue.result_recovery_hazards[0]["proposal_id"] == "restart-hazard"
    assert marker.exists()


def test_committed_unknown_result_recovers_exact_hazards_before_reserve_or_claim(
    tmp_path,
):
    root = tmp_path / "results"
    first_store = ResultEvidenceStore(
        root,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
        process_instance_id="1" * 32,
    )
    first_queue = OperatorProposalQueue(
        result_store=first_store,
        result_evidence_lifecycle_ready=True,
    )
    proposal = _claimed(first_queue, proposal_id="committed-unknown")
    reservation = first_queue.reserve_result_evidence(proposal)
    assert first_queue.mark_result_evidence_send_may_have_occurred(
        reservation["reservation_binding"]
    )["transitioned"] is True

    completed = first_queue.mark_apply_result(
        proposal,
        {
            "status": "blocked",
            "reason": "readback_lost_after_send",
            "no_console_send_proven": False,
            "write_state_unknown": True,
            "rollback_state_unknown": True,
            "physical_state_restored": False,
            "manual_review_required": True,
        },
        evidence_reservation_binding=reservation["reservation_binding"],
    )

    assert completed["status"] == "apply_error"
    result_digest = completed["result_digest"]
    artifact = root / f"{result_digest[7:]}.json"
    assert artifact.exists()
    assert not list(root.glob(".reservation.*"))

    restarted_store = ResultEvidenceStore(
        root,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
        process_instance_id="2" * 32,
    )
    restarted_queue = OperatorProposalQueue(
        result_store=restarted_store,
        result_evidence_lifecycle_ready=True,
    )

    assert restarted_queue.result_persistence_failed is False
    assert restarted_queue.result_evidence_contract_ready is False
    assert [
        item["code"] for item in restarted_queue.result_recovery_hazards
    ] == [
        "terminal_write_state_unknown",
        "terminal_rollback_state_unknown",
        "terminal_manual_review_after_send",
    ]
    assert {
        item["result_digest"]
        for item in restarted_queue.result_recovery_hazards
    } == {result_digest}
    assert artifact.exists()

    pending = _create_recovery_test_proposal(
        restarted_queue,
        proposal_id="blocked-by-recovered-result",
    )
    original_reserve = restarted_store.reserve
    restarted_store.reserve = MagicMock(wraps=original_reserve)
    original_validate = restarted_queue.validate_review_request
    restarted_queue.validate_review_request = MagicMock(
        wraps=original_validate
    )
    reservations_before = restarted_store.active_reservation_count

    reserve = restarted_queue.reserve_result_evidence(pending)
    claim = restarted_queue.claim_verified_for_apply(
        pending,
        expected_review_contract_version=pending.review_contract_version,
        expected_review_content_digest=pending.review_content_digest,
    )

    assert reserve["reserved"] is False
    assert reserve["reason"] == "operator_proposal_result_recovery_required"
    assert claim["claimed"] is False
    assert claim["reason"] == "operator_proposal_result_recovery_required"
    assert pending.status == "pending"
    assert restarted_store.active_reservation_count == reservations_before == 0
    restarted_store.reserve.assert_not_called()
    restarted_queue.validate_review_request.assert_not_called()
    assert artifact.exists()


def test_live_peer_unknown_artifact_refresh_blocks_reserve_and_claim(tmp_path):
    root = tmp_path / "results"
    store_a = ResultEvidenceStore(
        root,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
        process_instance_id="a" * 32,
    )
    store_b = ResultEvidenceStore(
        root,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
        process_instance_id="b" * 32,
    )
    queue_a = OperatorProposalQueue(
        result_store=store_a,
        result_evidence_lifecycle_ready=True,
    )
    queue_b = OperatorProposalQueue(
        result_store=store_b,
        result_evidence_lifecycle_ready=True,
    )
    assert queue_a.result_recovery_hazards == ()
    assert queue_b.result_recovery_hazards == ()

    proposal_a = _claimed(queue_a, proposal_id="peer-a-unknown")
    reservation_a = queue_a.reserve_result_evidence(proposal_a)
    assert queue_a.mark_result_evidence_send_may_have_occurred(
        reservation_a["reservation_binding"]
    )["transitioned"] is True
    queue_a.mark_apply_result(
        proposal_a,
        {
            "status": "error",
            "no_console_send_proven": False,
            "write_state_unknown": True,
            "physical_state_restored": False,
            "manual_review_required": True,
        },
        evidence_reservation_binding=reservation_a["reservation_binding"],
    )
    proposal_b = _create_recovery_test_proposal(
        queue_b,
        proposal_id="peer-b-blocked",
    )
    store_b.reserve = MagicMock(wraps=store_b.reserve)
    queue_b.validate_review_request = MagicMock(
        wraps=queue_b.validate_review_request
    )

    reserve_b = queue_b.reserve_result_evidence(proposal_b)
    claim_b = queue_b.claim_verified_for_apply(
        proposal_b,
        expected_review_contract_version=proposal_b.review_contract_version,
        expected_review_content_digest=proposal_b.review_content_digest,
    )

    assert reserve_b["reserved"] is False
    assert reserve_b["reason"] == "operator_proposal_result_recovery_required"
    assert claim_b["claimed"] is False
    assert claim_b["reason"] == "operator_proposal_result_recovery_required"
    assert proposal_b.status == "pending"
    assert store_b.active_reservation_count == 0
    store_b.reserve.assert_not_called()
    queue_b.validate_review_request.assert_not_called()
    assert [item["code"] for item in queue_b.result_recovery_hazards] == [
        "terminal_write_state_unknown",
        "terminal_rollback_state_unknown",
        "terminal_manual_review_after_send",
    ]


def test_live_peer_artifact_arriving_after_claim_blocks_atomic_send_boundary(
    tmp_path,
):
    root = tmp_path / "results"
    store_a = ResultEvidenceStore(
        root,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
        process_instance_id="c" * 32,
    )
    store_b = ResultEvidenceStore(
        root,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
        process_instance_id="d" * 32,
    )
    queue_a = OperatorProposalQueue(
        result_store=store_a,
        result_evidence_lifecycle_ready=True,
    )
    queue_b = OperatorProposalQueue(
        result_store=store_b,
        result_evidence_lifecycle_ready=True,
    )
    proposal_b = _create_recovery_test_proposal(
        queue_b,
        proposal_id="peer-b-preclaimed",
    )
    reservation_b = queue_b.reserve_result_evidence(proposal_b)
    claim_b = queue_b.claim_verified_for_apply(
        proposal_b,
        expected_review_contract_version=proposal_b.review_contract_version,
        expected_review_content_digest=proposal_b.review_content_digest,
    )
    assert reservation_b["reserved"] is True
    assert claim_b["claimed"] is True

    proposal_a = _claimed(queue_a, proposal_id="peer-a-late-unknown")
    reservation_a = queue_a.reserve_result_evidence(proposal_a)
    assert queue_a.mark_result_evidence_send_may_have_occurred(
        reservation_a["reservation_binding"]
    )["transitioned"] is True
    queue_a.mark_apply_result(
        proposal_a,
        {
            "status": "error",
            "no_console_send_proven": False,
            "write_state_unknown": True,
            "physical_state_restored": False,
            "manual_review_required": True,
        },
        evidence_reservation_binding=reservation_a["reservation_binding"],
    )

    boundary_b = queue_b.mark_result_evidence_send_may_have_occurred(
        reservation_b["reservation_binding"]
    )
    marker_b = root / (
        f".reservation.{reservation_b['reservation_binding'].token}"
    )

    assert boundary_b == {
        "transitioned": False,
        "reason": "operator_proposal_result_recovery_required",
    }
    assert json.loads(marker_b.read_text(encoding="utf-8"))["phase"] == (
        "reserved_no_send"
    )
    assert proposal_b.status == "applying"
    assert queue_b.result_persistence_failed is False
    assert queue_b.result_evidence_contract_ready is False
    queue_b.release_result_evidence(reservation_b["reservation_binding"])


def test_other_live_possible_send_marker_blocks_boundary_but_own_is_idempotent(
    tmp_path,
):
    root = tmp_path / "results"
    store_a = ResultEvidenceStore(
        root,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
        process_instance_id="e" * 32,
    )
    store_b = ResultEvidenceStore(
        root,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
        process_instance_id="f" * 32,
    )
    binding_a = _reserve(store_a, marker="possible-a")
    binding_b = _reserve(store_b, marker="possible-b")

    store_a.mark_send_may_have_occurred(binding_a)
    store_a.mark_send_may_have_occurred(binding_a)
    with pytest.raises(ResultEvidenceError) as exc_info:
        store_b.mark_send_may_have_occurred(binding_b)

    assert exc_info.value.code == "operator_proposal_result_recovery_required"
    assert exc_info.value.hazards[0]["phase"] == "send_may_have_occurred"
    marker_b = root / f".reservation.{binding_b.token}"
    assert json.loads(marker_b.read_text(encoding="utf-8"))["phase"] == (
        "reserved_no_send"
    )
    store_a.commit(
        binding_a,
        {
            "status": "error",
            "no_console_send_proven": False,
            "write_state_unknown": True,
            "physical_state_restored": False,
            "manual_review_required": True,
        },
    )
    store_b.release(binding_b)


@pytest.mark.parametrize(
    "result",
    [
        {
            "status": "sent_unverified",
            "no_console_send_proven": False,
            "physical_verified": False,
            "wal_terminal": "manual_review_required",
        },
        {
            "status": "applied",
            "no_console_send_proven": False,
            "wal_terminal": "committed",
            "physical_verified": True,
            "audio_recheck_passed": False,
            "audio_recheck_required": False,
            "audio_recheck_disposition": (
                "not_required_by_manual_write_contract"
            ),
            "write_state_unknown": False,
            "rollback_state_unknown": False,
            "manual_review_required": False,
        },
        {
            "status": "applied",
            "no_console_send_proven": True,
            "write_state_unknown": "unknown",
            "rollback_state_unknown": False,
            "manual_review_required": False,
        },
    ],
    ids=["legacy-sent-unverified", "audio-recheck-failed", "malformed-flag"],
)
def test_pinned_result_without_exact_fatal_disposition_recovers_generic_hazard(
    tmp_path,
    result,
):
    root = tmp_path / "results"
    first = ResultEvidenceStore(
        root,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
        process_instance_id="3" * 32,
    )
    record = first.commit(_reserve(first), result)
    artifact = root / record.artifact_name
    assert record.pinned is True

    restarted = ResultEvidenceStore(
        root,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
        process_instance_id="4" * 32,
    )
    queue = OperatorProposalQueue(
        result_store=restarted,
        result_evidence_lifecycle_ready=True,
    )

    assert queue.result_persistence_failed is False
    assert queue.result_evidence_contract_ready is False
    assert [item["code"] for item in queue.result_recovery_hazards] == [
        "operator_proposal_result_recovery_required"
    ]
    assert queue.result_recovery_hazards[0]["result_digest"] == record.digest
    assert artifact.exists()


def test_malformed_digest_named_result_recovers_generic_hazard_without_deletion(
    tmp_path,
):
    root = tmp_path / "results"
    first = ResultEvidenceStore(
        root,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    binding = _reserve(first)
    first.release(binding)
    content = b"{not-valid-json"
    digest = hashlib.sha256(content).hexdigest()
    artifact = root / f"{digest}.json"
    artifact.write_bytes(content)
    artifact.chmod(0o600)

    restarted = ResultEvidenceStore(
        root,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
        process_instance_id="5" * 32,
    )
    queue = OperatorProposalQueue(
        result_store=restarted,
        result_evidence_lifecycle_ready=True,
    )

    assert queue.result_persistence_failed is False
    assert queue.result_evidence_contract_ready is False
    assert queue.result_recovery_hazards == (
        {
            "code": "operator_proposal_result_recovery_required",
            "proposal_id": None,
            "review_content_digest": None,
            "result_digest": f"sha256:{digest}",
            "phase": "retained_result_conservative_recovery",
        },
    )
    assert artifact.read_bytes() == content


@pytest.mark.parametrize(
    "disposition",
    [
        "not_required_by_manual_write_contract",
        "not_required_by_linked_manual_write_contract",
    ],
)
def test_verified_direct_and_linked_results_restart_safe_and_evictable(
    tmp_path,
    disposition,
):
    root = tmp_path / disposition
    first = ResultEvidenceStore(
        root,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    record = first.commit(
        _reserve(first),
        {
            "status": "applied",
            "no_console_send_proven": False,
            "wal_terminal": "committed",
            "physical_verified": True,
            "audio_recheck_required": False,
            "audio_recheck_disposition": disposition,
            "write_state_unknown": False,
            "rollback_state_unknown": False,
            "manual_review_required": False,
        },
    )
    restarted = ResultEvidenceStore(
        root,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
        process_instance_id="6" * 32,
    )
    queue = OperatorProposalQueue(
        result_store=restarted,
        result_evidence_lifecycle_ready=True,
    )

    assert record.pinned is False
    assert queue.result_persistence_failed is False
    assert queue.result_recovery_hazards == ()
    assert queue.result_evidence_contract_ready is True


@pytest.mark.parametrize("producer", ["linked", "transaction"])
def test_verified_rollback_normalizes_exact_safe_truth_before_digest_and_restart(
    tmp_path,
    producer,
):
    root = tmp_path / producer
    store = ResultEvidenceStore(
        root,
        max_artifacts=1,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    queue = OperatorProposalQueue(
        result_store=store,
        result_evidence_lifecycle_ready=True,
    )
    proposal = _claimed(queue, proposal_id=f"{producer}-verified-rollback")
    reservation = queue.reserve_result_evidence(proposal)
    assert queue.mark_result_evidence_send_may_have_occurred(
        reservation["reservation_binding"]
    )["transitioned"] is True

    completed = queue.mark_apply_result(
        proposal,
        {
            "status": "blocked" if producer == "linked" else "rolled_back",
            "reason": f"{producer}_apply_failed_rolled_back",
            "no_console_send_proven": False,
            "physical_state_restored": True,
            "terminal_after_console_send": True,
            "wal_terminal": "rolled_back",
            "fresh_analysis_required": True,
        },
        evidence_reservation_binding=reservation["reservation_binding"],
    )
    detail = json.loads(
        queue.load_result_evidence(
            proposal,
            expected_result_digest=completed["result_digest"],
        )
    )

    assert completed["status"] == "apply_blocked"
    assert proposal.result_evidence_pinned is False
    for field in (
        "write_state_unknown",
        "rollback_state_unknown",
        "manual_review_required",
    ):
        assert detail[field] is False
        assert proposal.result[field] is False

    restarted = ResultEvidenceStore(
        root,
        max_artifacts=1,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
        process_instance_id="9" * 32,
    )
    restarted_queue = OperatorProposalQueue(
        result_store=restarted,
        result_evidence_lifecycle_ready=True,
    )
    assert restarted_queue.result_recovery_hazards == ()
    assert restarted_queue.result_evidence_contract_ready is True
    replacement = _reserve(restarted, marker=f"{producer}-replacement")
    assert not (root / f"{completed['result_digest'][7:]}.json").exists()
    restarted.release(replacement)


def test_contradictory_verified_rollback_unknown_true_stays_pinned_on_restart(
    tmp_path,
):
    root = tmp_path / "contradictory-rollback"
    store = ResultEvidenceStore(
        root,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    queue = OperatorProposalQueue(
        result_store=store,
        result_evidence_lifecycle_ready=True,
    )
    proposal = _claimed(queue, proposal_id="contradictory-rollback")
    reservation = queue.reserve_result_evidence(proposal)
    queue.mark_result_evidence_send_may_have_occurred(
        reservation["reservation_binding"]
    )

    completed = queue.mark_apply_result(
        proposal,
        {
            "status": "rolled_back",
            "no_console_send_proven": False,
            "physical_state_restored": True,
            "terminal_after_console_send": True,
            "wal_terminal": "rolled_back",
            "write_state_unknown": True,
        },
        evidence_reservation_binding=reservation["reservation_binding"],
    )
    restarted = ResultEvidenceStore(
        root,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
        process_instance_id="8" * 32,
    )
    restarted_queue = OperatorProposalQueue(
        result_store=restarted,
        result_evidence_lifecycle_ready=True,
    )

    assert proposal.result_evidence_pinned is True
    assert restarted_queue.result_evidence_contract_ready is False
    assert [
        item["code"] for item in restarted_queue.result_recovery_hazards
    ] == ["terminal_write_state_unknown"]
    detail = json.loads(store.load(completed["result_digest"]))
    assert detail["write_state_unknown"] is True
    assert detail["rollback_state_unknown"] is False
    assert detail["manual_review_required"] is False


@pytest.mark.parametrize(
    "missing_field",
    ["no_console_send_proven", "terminal_after_console_send", "wal_terminal"],
)
def test_restored_true_without_complete_post_send_envelope_stays_pinned(
    tmp_path,
    missing_field,
):
    result = {
        "status": "rolled_back",
        "no_console_send_proven": False,
        "physical_state_restored": True,
        "terminal_after_console_send": True,
        "wal_terminal": "rolled_back",
        "write_state_unknown": False,
        "rollback_state_unknown": False,
        "manual_review_required": False,
    }
    result.pop(missing_field)
    root = tmp_path / missing_field
    store = ResultEvidenceStore(
        root,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    record = store.commit(_reserve(store), result)
    restarted = ResultEvidenceStore(
        root,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
        process_instance_id="7" * 32,
    )
    queue = OperatorProposalQueue(
        result_store=restarted,
        result_evidence_lifecycle_ready=True,
    )

    assert record.pinned is True
    assert queue.result_evidence_contract_ready is False
    assert [item["code"] for item in queue.result_recovery_hazards] == [
        "operator_proposal_result_recovery_required"
    ]


@pytest.mark.parametrize("wal_terminal", [None, "committed", "not_committed"])
def test_restored_true_with_wrong_wal_terminal_stays_pinned(
    tmp_path,
    wal_terminal,
):
    result = {
        "status": "rolled_back",
        "no_console_send_proven": False,
        "physical_state_restored": True,
        "terminal_after_console_send": True,
        "wal_terminal": wal_terminal,
        "write_state_unknown": False,
        "rollback_state_unknown": False,
        "manual_review_required": False,
    }
    store = ResultEvidenceStore(
        tmp_path / str(wal_terminal),
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    record = store.commit(_reserve(store), result)

    assert record.pinned is True


def test_queue_missing_rollback_wal_is_apply_error_and_manual_review(tmp_path):
    root = tmp_path / "missing-rollback-wal"
    store = ResultEvidenceStore(
        root,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    queue = OperatorProposalQueue(
        result_store=store,
        result_evidence_lifecycle_ready=True,
    )
    proposal = _claimed(queue, proposal_id="missing-rollback-wal")
    reservation = queue.reserve_result_evidence(proposal)
    queue.mark_result_evidence_send_may_have_occurred(
        reservation["reservation_binding"]
    )

    completed = queue.mark_apply_result(
        proposal,
        {
            "status": "blocked",
            "no_console_send_proven": False,
            "physical_state_restored": True,
            "terminal_after_console_send": True,
        },
        evidence_reservation_binding=reservation["reservation_binding"],
    )
    detail = json.loads(store.load(completed["result_digest"]))

    assert completed["status"] == "apply_error"
    assert proposal.status == "apply_error"
    assert detail["manual_review_required"] is True
    assert proposal.result_evidence_pinned is True


def _create_recovery_test_proposal(queue, *, proposal_id):
    created = queue.create(
        {
            "id": proposal_id,
            "title": "Recovery gate probe",
            "requested_change": {
                "value_type": "fader",
                "channel": 1,
                "value": -12.0,
            },
        },
        _mode(),
    )
    assert created["status"] == "created"
    return queue.find(proposal_id=proposal_id)


def test_restart_reaps_foreign_reserved_no_send_marker(tmp_path):
    root = tmp_path / "results"
    first = ResultEvidenceStore(
        root,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
        process_instance_id="d" * 32,
    )
    token = first.reserve(
        proposal_id="safe-orphan",
        review_content_digest="sha256:" + ("e" * 64),
    )

    restarted = ResultEvidenceStore(
        root,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
        process_instance_id="f" * 32,
    )
    queue = OperatorProposalQueue(result_store=restarted)

    assert queue.result_persistence_failed is False
    assert queue.result_recovery_hazards == ()
    assert not (root / f".reservation.{token}").exists()


def test_reservation_directory_fsync_failure_cleans_unowned_marker(
    tmp_path,
    monkeypatch,
):
    store = ResultEvidenceStore(
        tmp_path / "results",
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    assert store.active_reservation_count == 0
    original = store._fsync_directory_fd
    calls = {"count": 0}

    def fail_marker_directory_fsync(descriptor, path):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("directory fsync failed")
        return original(descriptor, path)

    monkeypatch.setattr(store, "_fsync_directory_fd", fail_marker_directory_fsync)
    with pytest.raises(ResultEvidenceError) as exc_info:
        store.reserve(
            proposal_id="fsync-failure",
            review_content_digest="sha256:" + ("1" * 64),
        )

    assert exc_info.value.code == (
        "operator_proposal_result_evidence_persistence_failed"
    )
    assert not list(store.root.glob(".reservation.*"))


def test_load_rejects_digest_named_corrupt_content(tmp_path):
    store = ResultEvidenceStore(
        tmp_path / "results",
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    record = store.commit(_reserve(store), _result(marker="original"))
    (store.root / record.artifact_name).write_bytes(b"{}")

    with pytest.raises(ResultEvidenceError) as exc_info:
        store.load(record.digest)

    assert exc_info.value.code == "operator_proposal_result_evidence_store_corrupt"


def test_queue_commits_result_before_terminal_transition_and_serves_digest(tmp_path):
    store = ResultEvidenceStore(
        tmp_path / "results",
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    queue = OperatorProposalQueue(result_store=store)
    proposal = _claimed(queue)
    reservation = queue.reserve_result_evidence(proposal)

    completed = queue.mark_apply_result(
        proposal,
        _result(marker="durable-result"),
        evidence_reservation_binding=reservation["reservation_binding"],
    )

    assert completed["status"] == "applied"
    assert completed["result_detail_available"] is True
    content = queue.load_result_evidence(
        proposal,
        expected_result_digest=completed["result_digest"],
    )
    assert json.loads(content)["marker"] == "durable-result"


def test_sent_unverified_artifact_matches_terminal_manual_review_truth(tmp_path):
    store = ResultEvidenceStore(
        tmp_path / "results",
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    queue = OperatorProposalQueue(result_store=store)
    proposal = _claimed(queue, proposal_id="sent-unverified-detail")
    reservation = queue.reserve_result_evidence(proposal)
    assert queue.mark_result_evidence_send_may_have_occurred(
        reservation["reservation_binding"]
    )["transitioned"] is True

    completed = queue.mark_apply_result(
        proposal,
        {
            "status": "sent_unverified",
            "reason": "physical_readback_not_verified",
            "no_console_send_proven": False,
            "physical_verified": False,
            "wal_terminal": "manual_review_required",
        },
        evidence_reservation_binding=reservation["reservation_binding"],
    )

    assert completed["status"] == "apply_error"
    detail = json.loads(
        queue.load_result_evidence(
            proposal,
            expected_result_digest=proposal.result_digest,
        )
    )
    for field, expected in {
        "no_console_send_proven": False,
        "write_state_unknown": True,
        "physical_state_restored": False,
        "manual_review_required": True,
        "fresh_analysis_required": True,
    }.items():
        assert detail[field] is expected
        assert proposal.result[field] is expected


def test_configured_store_does_not_open_apply_lifecycle_gate(tmp_path):
    store = ResultEvidenceStore(
        tmp_path / "results",
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    queue = OperatorProposalQueue(result_store=store)

    assert queue.result_store is store
    assert queue.result_evidence_contract_ready is False


def test_explicit_verified_lifecycle_flag_opens_gate_for_isolated_tests(tmp_path):
    store = ResultEvidenceStore(
        tmp_path / "results",
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    queue = OperatorProposalQueue(
        result_store=store,
        result_evidence_lifecycle_ready=True,
    )

    assert queue.result_evidence_contract_ready is True


def test_latched_persistence_failure_closes_reported_lifecycle_readiness(tmp_path):
    queue = OperatorProposalQueue(
        result_store=ResultEvidenceStore(
            tmp_path / "results",
            min_free_bytes=0,
            disk_usage=_disk_with_free(1_000_000_000),
        ),
        result_evidence_lifecycle_ready=True,
    )
    assert queue.result_evidence_contract_ready is True

    queue.latch_result_persistence_failure()

    assert queue.result_persistence_failed is True
    assert queue.result_evidence_contract_ready is False


def test_queue_latches_persistence_failure_and_retains_unknown_write_truth(tmp_path):
    store = ResultEvidenceStore(
        tmp_path / "results",
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    queue = OperatorProposalQueue(result_store=store)
    proposal = _claimed(queue)

    completed = queue.mark_apply_result(
        proposal,
        _result(marker="missing-reservation"),
    )

    assert completed["status"] == "apply_error"
    assert queue.result_persistence_failed is True
    assert proposal.result["reason"] == "operator_proposal_result_persistence_failed"
    assert proposal.result["write_state_unknown"] is True
    assert proposal.result["manual_review_required"] is True
    assert proposal.result_detail_available is False


def test_terminal_byte_pressure_keeps_digest_tombstone_after_persistence(tmp_path):
    store = ResultEvidenceStore(
        tmp_path / "results",
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    queue = OperatorProposalQueue(
        max_history_bytes=1,
        result_store=store,
    )
    proposal = _claimed(queue, proposal_id="terminal-tombstone")
    reservation = queue.reserve_result_evidence(proposal)

    queue.mark_apply_result(
        proposal,
        _result(marker="retained-evidence"),
        evidence_reservation_binding=reservation["reservation_binding"],
    )

    retained = queue.find(proposal_id="terminal-tombstone")
    assert retained is proposal
    assert retained.terminal_tombstone is True
    assert retained.review_content_bytes == b""
    assert retained.result_digest
    assert json.loads(
        queue.load_result_evidence(
            retained,
            expected_result_digest=retained.result_digest,
        )
    )["marker"] == "retained-evidence"


def test_post_send_persistence_failure_retains_full_row_without_false_no_send(
    tmp_path,
):
    store = ResultEvidenceStore(
        tmp_path / "results",
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    queue = OperatorProposalQueue(
        max_history_bytes=1,
        result_store=store,
    )
    proposal = _claimed(queue, proposal_id="post-send-tombstone")
    reservation = queue.reserve_result_evidence(proposal)
    boundary = queue.mark_result_evidence_send_may_have_occurred(
        reservation["reservation_binding"]
    )
    assert boundary["transitioned"] is True
    result = {
        "status": "blocked",
        "reason": "post_send_result_not_serializable",
        "no_console_send_proven": False,
        "write_state_unknown": True,
        "physical_state_restored": False,
        "manual_review_required": True,
    }
    result["cycle"] = result

    completed = queue.mark_apply_result(
        proposal,
        result,
        evidence_reservation_binding=reservation["reservation_binding"],
    )

    assert completed["status"] == "apply_error"
    assert proposal.terminal_tombstone is False
    assert proposal.review_content_bytes
    assert proposal.result_digest is None
    assert proposal.result["no_console_send_proven"] is False
    assert proposal.result["write_state_unknown"] is True
    assert proposal.result["physical_state_restored"] is False
    assert proposal.result["manual_review_required"] is True
    assert proposal.result["result_persistence_failed"] is True
    marker = store.root / (
        f".reservation.{reservation['reservation_binding'].token}"
    )
    assert json.loads(marker.read_text(encoding="utf-8"))["phase"] == (
        "persistence_failed_after_send"
    )


def test_reservations_are_shared_across_store_instances(tmp_path):
    root = tmp_path / "results"
    first = ResultEvidenceStore(
        root,
        max_artifacts=1,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    second = ResultEvidenceStore(
        root,
        max_artifacts=1,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )

    token = _reserve(first)
    with pytest.raises(ResultEvidenceError) as exc_info:
        _reserve(second)

    assert (
        exc_info.value.code
        == "operator_proposal_result_evidence_capacity_unavailable"
    )
    assert first.active_reservation_count == 1
    assert second.active_reservation_count == 1
    second.release(token)
    assert first.active_reservation_count == 0


def test_reservation_capacity_is_atomic_across_threads(tmp_path):
    root = tmp_path / "results"
    stores = [
        ResultEvidenceStore(
            root,
            max_artifacts=1,
            min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
        )
        for _ in range(2)
    ]
    barrier = threading.Barrier(2)
    tokens = []
    errors = []

    def reserve(store):
        barrier.wait()
        try:
            tokens.append(_reserve(store))
        except ResultEvidenceError as exc:
            errors.append(exc.code)

    threads = [threading.Thread(target=reserve, args=(store,)) for store in stores]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert len(tokens) == 1
    assert errors == ["operator_proposal_result_evidence_capacity_unavailable"]
    stores[0].release(tokens[0])


def test_reservation_marker_blocks_a_separate_process(tmp_path):
    root = tmp_path / "results"
    store = ResultEvidenceStore(
        root,
        max_artifacts=1,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    token = _reserve(store)
    code = """
from operator_proposal_result_store import ResultEvidenceError, ResultEvidenceStore
import pathlib, sys
store = ResultEvidenceStore(pathlib.Path(sys.argv[1]), max_artifacts=1, min_free_bytes=0)
try:
    store.reserve(
        proposal_id='subprocess-test',
        review_content_digest='sha256:' + ('0' * 64),
    )
except ResultEvidenceError as exc:
    print(exc.code)
else:
    print('unexpected_success')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(root)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "backend"},
    )

    assert completed.stdout.strip() == (
        "operator_proposal_result_evidence_capacity_unavailable"
    )
    store.release(token)


def test_store_rejects_symlink_ancestor(tmp_path):
    real_parent = tmp_path / "real"
    real_parent.mkdir(mode=0o700)
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    store = ResultEvidenceStore(
        linked_parent / "results",
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )

    with pytest.raises(ResultEvidenceError) as exc_info:
        _reserve(store)

    assert exc_info.value.code == "operator_proposal_result_evidence_store_unsafe_root"
    assert list(real_parent.iterdir()) == []


def test_root_replacement_race_never_writes_through_new_symlink(
    tmp_path,
    monkeypatch,
):
    store = ResultEvidenceStore(
        tmp_path / "results",
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    token = _reserve(store)
    moved = tmp_path / "moved-results"
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    original_replace = os.replace
    replaced = False

    def replace_after_root_swap(source, target, **kwargs):
        nonlocal replaced
        if not replaced:
            replaced = True
            os.rename(store.root, moved)
            store.root.symlink_to(outside, target_is_directory=True)
        return original_replace(source, target, **kwargs)

    monkeypatch.setattr(os, "replace", replace_after_root_swap)
    with pytest.raises(ResultEvidenceError) as exc_info:
        store.commit(token, _result(marker="root-race"))

    assert exc_info.value.code == "operator_proposal_result_evidence_store_unsafe_root"
    assert list(outside.iterdir()) == []
    assert any(path.suffix == ".json" for path in moved.iterdir())


def test_nested_creation_fsyncs_new_directory_chain_bottom_up(tmp_path, monkeypatch):
    root = tmp_path / "one" / "two" / "results"
    store = ResultEvidenceStore(
        root,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    fsynced = []

    monkeypatch.setattr(
        store,
        "_fsync_directory_fd",
        lambda _descriptor, path: fsynced.append(path),
    )
    token = _reserve(store)

    assert fsynced[:4] == [root, root.parent, root.parent.parent, tmp_path]
    store.release(token)


def test_nested_creation_fsync_failure_is_typed_and_fail_closed(tmp_path, monkeypatch):
    root = tmp_path / "one" / "two" / "results"
    store = ResultEvidenceStore(
        root,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )

    def fail_on_parent(_descriptor, path):
        if path == root.parent:
            raise OSError("fsync failed")

    monkeypatch.setattr(store, "_fsync_directory_fd", fail_on_parent)
    with pytest.raises(ResultEvidenceError) as exc_info:
        _reserve(store)

    assert (
        exc_info.value.code
        == "operator_proposal_result_evidence_directory_fsync_failed"
    )


def test_existing_identical_target_is_fsynced_again(tmp_path, monkeypatch):
    store = ResultEvidenceStore(
        tmp_path / "results",
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    result = _result(marker="same-content")
    first = store.commit(_reserve(store), result)
    reads = []
    directories = []
    original_read = store._read_regular_file

    def record_read(root_fd, name, **kwargs):
        reads.append((name, kwargs.get("fsync_before_close")))
        return original_read(root_fd, name, **kwargs)

    monkeypatch.setattr(store, "_read_regular_file", record_read)
    monkeypatch.setattr(
        store,
        "_fsync_directory_fd",
        lambda _descriptor, path: directories.append(path),
    )
    second = store.commit(_reserve(store), result)

    assert second.digest == first.digest
    assert (first.artifact_name, True) in reads
    assert store.root in directories


def test_production_verified_applied_result_is_evictable_without_restored_field(tmp_path):
    store = ResultEvidenceStore(
        tmp_path / "results",
        max_artifacts=1,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    result = {
        "status": "applied",
        "no_console_send_proven": False,
        "wal_terminal": "committed",
        "physical_verified": True,
        "audio_recheck_passed": True,
        "write_state_unknown": False,
        "rollback_state_unknown": False,
        "manual_review_required": False,
    }
    record = store.commit(_reserve(store), result)

    assert record.pinned is False
    token = _reserve(store)
    assert not (store.root / record.artifact_name).exists()
    store.release(token)


def test_manual_contract_disposition_is_evictable_without_fake_audio_recheck(tmp_path):
    store = ResultEvidenceStore(
        tmp_path / "results",
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    result = {
        "status": "applied",
        "no_console_send_proven": False,
        "wal_terminal": "committed",
        "physical_verified": True,
        "audio_recheck_required": False,
        "audio_recheck_disposition": "not_required_by_manual_write_contract",
        "write_state_unknown": False,
        "rollback_state_unknown": False,
        "manual_review_required": False,
    }

    record = store.commit(_reserve(store), result)

    assert record.pinned is False
    assert "audio_recheck_passed" not in json.loads(store.load(record.digest))


def test_explicit_audio_recheck_failure_remains_pinned(tmp_path):
    store = ResultEvidenceStore(
        tmp_path / "results",
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    result = {
        "status": "applied",
        "no_console_send_proven": False,
        "wal_terminal": "committed",
        "physical_verified": True,
        "audio_recheck_passed": False,
        "audio_recheck_required": False,
        "audio_recheck_disposition": "not_required_by_manual_write_contract",
        "write_state_unknown": False,
        "rollback_state_unknown": False,
        "manual_review_required": False,
    }

    record = store.commit(_reserve(store), result)

    assert record.pinned is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("write_state_unknown", 1),
        ("rollback_state_unknown", "false"),
        ("manual_review_required", 0),
        ("physical_state_restored", "true"),
        ("physical_verified", 1),
        ("audio_recheck_passed", "true"),
        ("audio_recheck_required", 0),
    ],
)
def test_malformed_safety_flags_are_pinned_and_remain_pinned_after_reload(
    tmp_path,
    field,
    value,
):
    root = tmp_path / f"results-{field}"
    store = ResultEvidenceStore(
        root,
        max_artifacts=1,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    result = {
        "status": "applied",
        "no_console_send_proven": True,
        "wal_terminal": "committed",
        "physical_verified": True,
        "audio_recheck_passed": True,
        field: value,
    }
    record = store.commit(_reserve(store), result)
    reloaded = ResultEvidenceStore(
        root,
        max_artifacts=1,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )

    assert record.pinned is True
    with pytest.raises(ResultEvidenceError) as exc_info:
        _reserve(reloaded)
    assert (
        exc_info.value.code
        == "operator_proposal_result_recovery_required"
    )


def test_unsafe_cycle_depth_and_size_failures_retain_the_durable_reservation(tmp_path):
    cyclic = {"status": "blocked", "no_console_send_proven": False}
    cyclic["self"] = cyclic
    deep = {"leaf": True}
    for _ in range(70):
        deep = {"nested": deep}
    cases = [
        (cyclic, "operator_proposal_result_evidence_cycle_detected", 1_024),
        (deep, "operator_proposal_result_evidence_depth_limit_exceeded", 1_024),
        (
            {"blob": "x" * 1_024},
            "operator_proposal_result_evidence_too_large",
            128,
        ),
    ]
    for index, (result, reason, max_bytes) in enumerate(cases):
        store = ResultEvidenceStore(
            tmp_path / f"results-{index}",
            max_artifact_bytes=max_bytes,
            max_total_bytes=8_192,
            min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
        )
        token = _reserve(store)
        with pytest.raises(ResultEvidenceError) as exc_info:
            store.commit(token, result)
        assert exc_info.value.code == reason
        assert store.active_reservation_count == 1
        marker = store.root / f".reservation.{token.token}"
        assert json.loads(marker.read_text(encoding="utf-8"))["phase"] == (
            "persistence_failed_after_send"
        )


@pytest.mark.parametrize(
    "contradiction",
    [
        {"physical_state_restored": False},
        {"terminal_after_console_send": True},
        {"result_persistence_failed": True},
        {"status": "applied"},
        {"status": "sent_unverified"},
        {"status": "rolled_back"},
        {"status": "manual_review_required"},
        {"physical_verified": True},
        {"physical_state_restored": None},
        {"physical_state_restored": "false"},
        {"physical_state_restored": 0},
        {"physical_state_restored": 1},
        {"audio_recheck_passed": None},
        {"audio_recheck_passed": "true"},
        {"audio_recheck_required": 0},
        {"audio_recheck_required": 1},
        {"wal_terminal": "committed"},
        {"wal_terminal": "rolled_back"},
        {"wal_terminal": "manual_review_required"},
    ],
)
def test_contradictory_no_send_serialization_failure_retains_hazard_marker(
    tmp_path,
    contradiction,
):
    store = ResultEvidenceStore(
        tmp_path / "results",
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    binding = _reserve(store)
    result = {
        "status": "blocked",
        "no_console_send_proven": True,
        "write_state_unknown": False,
        "rollback_state_unknown": False,
        "manual_review_required": False,
        **contradiction,
    }
    result["self"] = result

    with pytest.raises(ResultEvidenceError) as exc_info:
        store.commit(binding, result)

    assert exc_info.value.code == "operator_proposal_result_evidence_cycle_detected"
    marker = store.root / f".reservation.{binding.token}"
    assert json.loads(marker.read_text(encoding="utf-8"))["phase"] == (
        "persistence_failed_after_send"
    )
    restarted = ResultEvidenceStore(
        store.root,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
        process_instance_id="d" * 32,
    )
    assert restarted.recovery_hazards()[0]["phase"] == (
        "persistence_failed_after_send"
    )


@pytest.mark.parametrize(
    "contradiction",
    [
        {"status": "applied"},
        {"status": "sent_unverified"},
        {"status": "rolled_back"},
        {"status": "manual_review_required"},
        {"physical_verified": True},
        {"terminal_after_console_send": True},
        {"wal_terminal": "committed"},
        {"wal_terminal": "rolled_back"},
        {"wal_terminal": "manual_review_required"},
    ],
)
def test_contradictory_no_send_result_is_pinned_for_recovery(
    tmp_path,
    contradiction,
):
    store = ResultEvidenceStore(
        tmp_path / "results",
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    result = {
        "status": "blocked",
        "no_console_send_proven": True,
        "write_state_unknown": False,
        "rollback_state_unknown": False,
        "manual_review_required": False,
        **contradiction,
    }

    record = store.commit(_reserve(store), result)

    assert record.pinned is True
    restarted = ResultEvidenceStore(
        store.root,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
        process_instance_id="e" * 32,
    )
    with pytest.raises(ResultEvidenceError) as exc_info:
        _reserve(restarted)
    assert exc_info.value.code == "operator_proposal_result_recovery_required"


def test_exact_safe_no_send_serialization_failure_consumes_reserved_marker(tmp_path):
    store = ResultEvidenceStore(
        tmp_path / "results",
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    binding = _reserve(store)
    result = {
        "status": "blocked",
        "no_console_send_proven": True,
        "write_state_unknown": False,
        "rollback_state_unknown": False,
        "manual_review_required": False,
    }
    result["self"] = result

    with pytest.raises(ResultEvidenceError) as exc_info:
        store.commit(binding, result)

    assert exc_info.value.code == "operator_proposal_result_evidence_cycle_detected"
    assert store.active_reservation_count == 0


def test_exact_safe_no_send_with_not_prepared_wal_is_evictable(tmp_path):
    store = ResultEvidenceStore(
        tmp_path / "results",
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    record = store.commit(
        _reserve(store),
        {
            "status": "blocked",
            "no_console_send_proven": True,
            "write_state_unknown": False,
            "rollback_state_unknown": False,
            "manual_review_required": False,
            "physical_verified": False,
            "wal_terminal": "not_prepared",
        },
    )

    assert record.pinned is False


def test_cyclic_post_send_result_latches_queue_hazard_and_unknown_write(tmp_path):
    store = ResultEvidenceStore(
        tmp_path / "results",
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    queue = OperatorProposalQueue(result_store=store)
    proposal = _claimed(queue)
    reservation = queue.reserve_result_evidence(proposal)
    result = {"status": "blocked", "no_console_send_proven": False}
    result["self"] = result

    completed = queue.mark_apply_result(
        proposal,
        result,
        evidence_reservation_binding=reservation["reservation_binding"],
    )

    assert completed["status"] == "apply_error"
    assert queue.result_persistence_failed is True
    assert proposal.result["write_state_unknown"] is True
    assert proposal.result["manual_review_required"] is True
    assert store.active_reservation_count == 1
    marker = store.root / (
        f".reservation.{reservation['reservation_binding'].token}"
    )
    assert json.loads(marker.read_text(encoding="utf-8"))["phase"] == (
        "persistence_failed_after_send"
    )

    restarted_store = ResultEvidenceStore(
        store.root,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
        process_instance_id="f" * 32,
    )
    restarted_queue = OperatorProposalQueue(result_store=restarted_store)
    assert restarted_queue.result_evidence_contract_ready is False
    assert restarted_queue.result_recovery_hazards == (
        {
            "code": "operator_proposal_result_persistence_failed",
            "proposal_id": proposal.id,
            "review_content_digest": proposal.review_content_digest,
            "phase": "persistence_failed_after_send",
        },
    )


def test_stale_owned_temp_is_cleaned_before_capacity_accounting(tmp_path):
    store = ResultEvidenceStore(
        tmp_path / "results",
        max_total_bytes=1_024,
        max_artifact_bytes=128,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    token = _reserve(store)
    store.release(token)
    stale = store.root / f".{('a' * 64)}.{('b' * 16)}.tmp"
    stale.write_bytes(b"x" * 2_048)
    stale.chmod(0o600)

    replacement = _reserve(store)

    assert not stale.exists()
    store.release(replacement)


def test_unexpected_regular_entry_fails_closed(tmp_path):
    store = ResultEvidenceStore(
        tmp_path / "results",
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    token = _reserve(store)
    store.release(token)
    unexpected = store.root / "unowned.bin"
    unexpected.write_bytes(b"x")
    unexpected.chmod(0o600)

    with pytest.raises(ResultEvidenceError) as exc_info:
        _reserve(store)

    assert (
        exc_info.value.code
        == "operator_proposal_result_evidence_unexpected_store_entry"
    )


def test_unsafe_temp_symlink_is_never_cleaned_or_followed(tmp_path):
    store = ResultEvidenceStore(
        tmp_path / "results",
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    token = _reserve(store)
    store.release(token)
    outside = tmp_path / "outside"
    outside.write_text("untouched", encoding="utf-8")
    temporary = store.root / f".{('a' * 64)}.{('b' * 16)}.tmp"
    temporary.symlink_to(outside)

    with pytest.raises(ResultEvidenceError) as exc_info:
        _reserve(store)

    assert exc_info.value.code == "operator_proposal_result_evidence_unsafe_temp"
    assert outside.read_text(encoding="utf-8") == "untouched"


def test_active_temp_is_protected_by_root_lock_until_commit_finishes(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "results"
    first = ResultEvidenceStore(
        root,
        max_artifacts=2,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    second = ResultEvidenceStore(
        root,
        max_artifacts=2,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    token = _reserve(first)
    replace_started = threading.Event()
    allow_replace = threading.Event()
    second_finished = threading.Event()
    original_replace = os.replace
    second_tokens = []

    def pause_replace(source, target, **kwargs):
        replace_started.set()
        assert allow_replace.wait(timeout=5)
        return original_replace(source, target, **kwargs)

    monkeypatch.setattr(os, "replace", pause_replace)
    commit_thread = threading.Thread(
        target=lambda: first.commit(token, _result(marker="active-temp"))
    )
    commit_thread.start()
    assert replace_started.wait(timeout=5)
    temporary = next(path for path in root.iterdir() if path.suffix == ".tmp")

    def reserve_second():
        second_tokens.append(_reserve(second))
        second_finished.set()

    reserve_thread = threading.Thread(target=reserve_second)
    reserve_thread.start()
    time.sleep(0.05)
    assert temporary.exists()
    assert second_finished.is_set() is False
    allow_replace.set()
    commit_thread.join(timeout=5)
    reserve_thread.join(timeout=5)

    assert not temporary.exists()
    assert second_finished.is_set() is True
    second.release(second_tokens[0])


def test_existing_insecure_root_and_artifact_modes_are_rejected(tmp_path):
    insecure_root = tmp_path / "insecure-results"
    insecure_root.mkdir(mode=0o700)
    insecure_root.chmod(0o777)
    store = ResultEvidenceStore(
        insecure_root,
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    with pytest.raises(ResultEvidenceError) as root_error:
        _reserve(store)
    assert root_error.value.code == "operator_proposal_result_evidence_store_insecure_root"

    private = ResultEvidenceStore(
        tmp_path / "private-results",
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    record = private.commit(_reserve(private), _result(marker="mode"))
    artifact = private.root / record.artifact_name
    artifact.chmod(0o666)
    with pytest.raises(ResultEvidenceError) as artifact_error:
        private.load(record.digest)
    assert artifact_error.value.code == "operator_proposal_result_evidence_unsafe_target"


def test_private_modes_ignore_permissive_umask(tmp_path):
    previous = os.umask(0)
    try:
        store = ResultEvidenceStore(
            tmp_path / "results",
            min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
        )
        record = store.commit(_reserve(store), _result(marker="umask"))
    finally:
        os.umask(previous)

    assert stat.S_IMODE(store.root.stat().st_mode) == 0o700
    assert stat.S_IMODE((store.root / record.artifact_name).stat().st_mode) == 0o600
    assert stat.S_IMODE((store.root / ".store.lock").stat().st_mode) == 0o600


def test_digest_load_rejects_regular_to_symlink_swap_at_open(
    tmp_path,
    monkeypatch,
):
    store = ResultEvidenceStore(
        tmp_path / "results",
        min_free_bytes=0,
        disk_usage=_disk_with_free(1_000_000_000),
    )
    record = store.commit(_reserve(store), _result(marker="swap"))
    artifact = store.root / record.artifact_name
    outside = tmp_path / "outside"
    outside.write_bytes(artifact.read_bytes())
    original_open = os.open
    swapped = False

    def swap_before_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == record.artifact_name and kwargs.get("dir_fd") is not None and not swapped:
            swapped = True
            artifact.unlink()
            artifact.symlink_to(outside)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swap_before_open)
    with pytest.raises(ResultEvidenceError) as exc_info:
        store.load(record.digest)

    assert exc_info.value.code == "operator_proposal_result_evidence_unsafe_target"
    assert outside.exists()
