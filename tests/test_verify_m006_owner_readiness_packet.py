"""Offline tests for the M006 owner readiness packet verifier."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "verify_m006_owner_readiness_packet.py"
PACKET_SUMMARY = (
    PROJECT_ROOT
    / "artifacts"
    / "wing_owner_readiness_review"
    / "20260704T191101Z_m006_owner_readiness_review_packet"
    / "summary.json"
)
PARTIAL_M005_SUMMARY = (
    PROJECT_ROOT
    / "artifacts"
    / "wing_timeout_disconnect_proof"
    / "20260704T074537Z_wing_m005_disconnect_timeout_stale_proof"
    / "summary.json"
)
COMPLETE_M005_SUMMARY_PATH = (
    "artifacts/wing_timeout_disconnect_proof/"
    "unit_m005_disconnect_timeout_stale_proof/summary.json"
)
COMPLETE_M005_EVENTS_PATH = (
    "artifacts/wing_timeout_disconnect_proof/"
    "unit_m005_disconnect_timeout_stale_proof/events.jsonl"
)
M005_TARGET = {
    "channel": 3,
    "value_type": "fader",
    "address": "/ch/3/fdr",
}

SPEC = importlib.util.spec_from_file_location("verify_m006_owner_readiness_packet", TOOL_PATH)
assert SPEC is not None
verifier = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verifier)


@pytest.fixture(autouse=True)
def clean_current_release_expectations(monkeypatch: pytest.MonkeyPatch):
    actual = verifier.current_release_provenance_expectations()
    expected = copy.deepcopy(actual)
    expected["source"].update({
        "worktree_dirty": False,
        "status_sha256": verifier._sha256_bytes(b""),
        "tracked_diff_sha256": verifier._sha256_bytes(b""),
    })
    expected["runtime_source"]["source"].update({
        "worktree_dirty": False,
        "status_sha256": verifier._sha256_bytes(b""),
        "tracked_diff_sha256": verifier._sha256_bytes(b""),
    })
    monkeypatch.setattr(
        verifier,
        "current_release_provenance_expectations",
        lambda _m005_summary_path=None: copy.deepcopy(expected),
    )


def load_packet() -> dict:
    return json.loads(PACKET_SUMMARY.read_text(encoding="utf-8"))


def test_generated_m006_verifier_and_decision_outputs_are_gitignored() -> None:
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "artifacts/wing_owner_readiness_review_verifier/" in gitignore
    assert "artifacts/wing_owner_readiness_decision_verifier/" in gitignore
    assert "artifacts/wing_owner_readiness_review/*/owner_decision.json" in gitignore


def load_packet_proofs(packet: dict | None = None) -> dict:
    return verifier.load_packet_proof_summaries(packet or load_packet())


def with_m005_target(payload: dict) -> dict:
    return {**payload, **M005_TARGET}


def complete_m005_summary() -> dict:
    expected_provenance = verifier.current_release_provenance_expectations()
    challenge_identity = {
        "path": (
            "artifacts/wing_console_identity_owner_challenge/"
            "unit-owner-challenge/summary.json"
        ),
        "exists": True,
        "size": 512,
        "sha256": "b" * 64,
    }
    captured_at = "2026-07-12T00:00:00Z"
    runtime_launch_nonce = "d" * 32
    runtime_challenge_nonce = "e" * 32
    runtime_pid = 4242
    summary = {
        "proof_artifact_schema": "wing_timeout_disconnect_proof.v1",
        "status": "passed",
        "disconnect_mode": "physical-cable-loss",
        "url": verifier.EXPECTED_M005_CANONICAL_URL,
        "wing_ip": "192.0.2.10",
        **M005_TARGET,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "readiness_matrix_update_allowed": False,
        "summary_path": COMPLETE_M005_SUMMARY_PATH,
        "events_path": COMPLETE_M005_EVENTS_PATH,
        "direct_live_write_request_sent": False,
        "rollback_request_sent": False,
        "forbidden_request_types_sent": [],
        "sent_request_types": [
            "get_runtime_identity",
            "connect_wing",
            "get_operator_mode",
            "get_state",
            "query_mixer_parameter",
            "set_operator_mode",
            "create_operator_proposal",
            "query_mixer_parameter",
            "apply_operator_proposal",
            "connect_wing",
            "query_mixer_parameter",
        ],
        "required_m005_evidence": list(verifier.EXPECTED_M005_EVIDENCE),
        "completed_m005_evidence": list(verifier.EXPECTED_M005_EVIDENCE),
        "missing_m005_evidence": [],
        "remaining_m005_evidence": [],
        "m005_complete": True,
        "initial_connection_verified": True,
        "preflight_decision_gate": {
            "status": "accepted_decision_ready",
            "accepted": True,
            "expected_accepted_readonly_source_id": load_packet()[
                "accepted_readonly_source_id"
            ],
            "live_authority_granted": False,
        },
        "backend_launch_mode": "proof_runner_bound",
        "launched_backend_pid": runtime_pid,
        "runtime_identity_challenge_nonce": runtime_challenge_nonce,
        "runtime_identity": None,
        "m005_backend_runtime_identity_captured": True,
        "provenance": {
            "schema": verifier.EXPECTED_M005_PROVENANCE_SCHEMA,
            "captured_at_utc": captured_at,
            "runtime_launch_nonce": runtime_launch_nonce,
            "source": expected_provenance["source"],
            "runtime_source": expected_provenance["runtime_source"],
            "runner": expected_provenance["runner"],
            "backend": {
                "runtime_kind": "source",
                **expected_provenance["source_backend"],
            },
            "package": None,
            "console": {
                "model": verifier.EXPECTED_M005_CONSOLE_MODEL,
                "serial": "TEST-WING-RACK-SERIAL",
                "firmware": verifier.EXPECTED_M005_CONSOLE_FIRMWARE,
                "identity_source": "operator_attested_owner_challenge",
                "console_identity_query_available": False,
            },
            "console_identity_challenge": challenge_identity,
            "events": {
                "path": COMPLETE_M005_EVENTS_PATH,
                "exists": True,
                "size": 1,
                "sha256": "a" * 64,
            },
        },
        "phases": {
            "initial_connection": {
                "status": "passed",
                "connection_status": {
                    "type": "connection_status",
                    "connected": True,
                    "mode": "wing",
                    "ip": "192.0.2.10",
                    "observation_only": False,
                    "record_telemetry": False,
                },
            },
            "baseline": {
                "status": "passed",
                "readback": with_m005_target(
                    {
                        "type": "mixer_parameter_value",
                        "status": "verified_from_physical_readback",
                        "source": "physical_readback",
                        "value": 0.0,
                    }
                ),
            },
            "disconnect": {
                "status": "operator_confirmed_physical_cable_loss",
                "confirmation": "WING_CABLE_PULLED_BY_OPERATOR",
                "operator_gate": True,
            },
            "stale_readback": {
                "status": "passed",
                "readback": with_m005_target(
                    {
                        "type": "mixer_parameter_value",
                        "status": "readback_unavailable",
                        "source": "unavailable",
                        "value": None,
                    }
                ),
            },
            "blocked_write_while_disconnected": {
                "status": "passed",
                "block_reason": "mixer_not_connected",
                "apply_result": {
                    "type": "operator_proposal_apply_blocked",
                    "status": "apply_blocked",
                    "apply_result": {"status": "blocked", "reason": "mixer_not_connected"},
                },
            },
            "reconnect_recovery": {
                "status": "passed",
                "connection_status": {"connected": True},
                "readback": with_m005_target(
                    {
                        "type": "mixer_parameter_value",
                        "status": "verified_from_physical_readback",
                        "source": "physical_readback",
                        "value": 0.0,
                    }
                ),
                "unchanged_from_baseline": True,
            },
        },
    }
    prestart = copy.deepcopy(summary["provenance"])
    prestart["events"] = None
    summary["provenance_prestart_sha256"] = verifier._canonical_json_sha256(prestart)
    runtime_identity = {
        "type": "runtime_identity",
        "schema": verifier.EXPECTED_M005_RUNTIME_IDENTITY_SCHEMA,
        "runtime_kind": "source",
        "instance_nonce": "c" * 32,
        "started_at_utc": captured_at,
        "process": {
            "pid": runtime_pid,
            "instance_nonce": "c" * 32,
            "started_at_utc": captured_at,
        },
        "source": copy.deepcopy(expected_provenance["runtime_source"]["source"]),
        "code": copy.deepcopy(expected_provenance["runtime_source"]["code"]),
        "backend": expected_provenance["source_backend"],
        "launch_binding": {
            "mode": "proof_runner",
            "launch_nonce": runtime_launch_nonce,
            "prestart_provenance_schema": verifier.EXPECTED_M005_PROVENANCE_SCHEMA,
            "prestart_provenance_size": len(verifier.canonical_json_bytes(prestart)),
            "prestart_provenance_sha256": summary["provenance_prestart_sha256"],
            "bound_before_process_start": True,
            "proof_eligible": True,
        },
        "websocket": {"host": "127.0.0.1", "port": 8765, "loopback_only": True},
        "read_only_identity": True,
        "live_authority_granted": False,
        "request_challenge_nonce": runtime_challenge_nonce,
    }
    runtime_identity["manifest_sha256"] = verifier.runtime_identity_manifest_sha256(
        runtime_identity
    )
    summary["runtime_identity"] = runtime_identity
    return summary


def complete_console_identity_challenge(m005: dict) -> dict:
    return {
        "identity": copy.deepcopy(m005["provenance"]["console_identity_challenge"]),
        "content": {
            "schema": verifier.EXPECTED_M005_CONSOLE_IDENTITY_SCHEMA,
            "operator_confirmed": True,
            "console_model": verifier.EXPECTED_M005_CONSOLE_MODEL,
            "serial": m005["provenance"]["console"]["serial"],
            "firmware": verifier.EXPECTED_M005_CONSOLE_FIRMWARE,
            "accepted_readonly_source_id": m005["preflight_decision_gate"][
                "expected_accepted_readonly_source_id"
            ],
        },
    }


def complete_event_audit(m005: dict) -> dict:
    prestart = copy.deepcopy(m005["provenance"])
    prestart["events"] = None
    return {
        "request_types": list(m005["sent_request_types"]),
        "initial_connection": m005["phases"]["initial_connection"][
            "connection_status"
        ],
        "stale_readback": m005["phases"]["stale_readback"]["readback"],
        "reconnect_readback": m005["phases"]["reconnect_recovery"]["readback"],
        "first_event": {
            "seq": 1,
            "event_type": "provenance_capture",
            "payload": {
                "schema": verifier.EXPECTED_M005_PROVENANCE_SCHEMA,
                "provenance": prestart,
                "provenance_sha256": m005["provenance_prestart_sha256"],
                "url": verifier.EXPECTED_M005_CANONICAL_URL,
                "captured_before_backend_contact": True,
            },
        },
        "runtime_identity": copy.deepcopy(m005["runtime_identity"]),
        "sha256": m005["provenance"]["events"]["sha256"],
    }


def complete_m005_verify_kwargs(m005: dict) -> dict:
    audit = complete_event_audit(m005)
    return {
        "m005_event_request_types": list(audit["request_types"]),
        "m005_event_initial_connection": audit["initial_connection"],
        "m005_event_stale_readback": audit["stale_readback"],
        "m005_event_reconnect_readback": audit["reconnect_readback"],
        "m005_event_stale_readback_required": True,
        "m005_events_sha256": audit["sha256"],
        "m005_event_audit": audit,
        "m005_console_identity_challenge": complete_console_identity_challenge(m005),
    }


def complete_m005_event_request_types(m005: dict) -> list[str]:
    return list(m005["sent_request_types"])


def timeout_stale_readback() -> dict:
    return with_m005_target(
        {
            "type": "mixer_parameter_read_blocked",
            "status": "readback_timeout",
            "source": "timeout",
            "reason": "stale_readback_timeout_after_disconnect",
            "value": None,
            "transport_failure": {"type": "TimeoutError"},
        }
    )


def write_m005_event_log(
    events_path: Path,
    request_types: list[str],
    stale_readback: dict | None = None,
    stale_event_type: str | None = None,
    reconnect_readback: dict | None = None,
    reconnect_event_type: str | None = None,
) -> None:
    rows = []
    seq = 1
    query_count = 0
    connect_count = 0
    stale_inserted = False
    reconnect_inserted = False
    if stale_readback is None:
        effective_stale_event_type = None
    else:
        effective_stale_event_type = stale_event_type or (
            "synthetic_recv_timeout"
            if stale_readback.get("status") == "readback_timeout"
            else "recv"
        )
    effective_reconnect_event_type = (
        None if reconnect_readback is None else reconnect_event_type or "recv"
    )
    for request_type in request_types:
        rows.append({"seq": seq, "event_type": "send", "payload": {"type": request_type}})
        seq += 1
        if request_type == "connect_wing":
            connect_count += 1
            if connect_count == 1:
                rows.append(
                    {
                        "seq": seq,
                        "event_type": "recv",
                        "payload": {
                            "type": "connection_status",
                            "connected": True,
                            "mode": "wing",
                            "ip": "192.0.2.10",
                            "observation_only": False,
                            "record_telemetry": False,
                        },
                    }
                )
                seq += 1
        if request_type == "query_mixer_parameter":
            query_count += 1
        if (
            stale_readback is not None
            and effective_stale_event_type is not None
            and not stale_inserted
            and request_type == "query_mixer_parameter"
            and query_count == 2
        ):
            rows.append(
                {
                    "seq": seq,
                    "event_type": effective_stale_event_type,
                    "payload": stale_readback,
                }
            )
            seq += 1
            stale_inserted = True
        if (
            reconnect_readback is not None
            and effective_reconnect_event_type is not None
            and not reconnect_inserted
            and request_type == "query_mixer_parameter"
            and query_count == 3
        ):
            rows.append(
                {
                    "seq": seq,
                    "event_type": effective_reconnect_event_type,
                    "payload": reconnect_readback,
                }
            )
            seq += 1
            reconnect_inserted = True

    events_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_m005_event_audit_from_log(
    tmp_path: Path,
    m005: dict,
    *,
    stale_readback: dict | None = None,
    reconnect_readback: dict | None = None,
    reconnect_event_type: str | None = None,
) -> dict:
    summary_path = tmp_path / "summary.json"
    events_path = tmp_path / "events.jsonl"
    audit_m005 = copy.deepcopy(m005)
    audit_m005["events_path"] = str(events_path)
    summary_path.write_text(json.dumps(audit_m005), encoding="utf-8")
    write_m005_event_log(
        events_path,
        complete_m005_event_request_types(m005),
        stale_readback=stale_readback,
        reconnect_readback=reconnect_readback,
        reconnect_event_type=reconnect_event_type,
    )

    audit, _display_events_path = verifier.load_m005_event_audit(
        m005_summary_path=summary_path,
        m005_summary=audit_m005,
    )

    assert isinstance(audit, dict)
    return audit


def test_packet_without_m005_summary_fails_closed() -> None:
    result = verifier.verify_packet(load_packet())

    assert result["status"] == "blocked"
    assert result["reason"] == "m005_summary_required_for_owner_review"
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_closure_allowed"] is False
    assert result["readiness_matrix_update_allowed"] is False
    assert result["contacts_wing"] is False
    assert result["contacts_backend"] is False
    assert result["live_write"] is False
    assert "m005_summary_supplied" in result["failed_checks"]


def test_partial_m005_summary_keeps_owner_review_blocked() -> None:
    partial = copy.deepcopy(complete_m005_summary())
    partial["m005_complete"] = False
    partial["completed_m005_evidence"] = ["physical_baseline_readback"]
    partial["missing_m005_evidence"] = [
        "operator_confirmed_physical_cable_loss",
        "stale_readback_unavailable",
        "blocked_write_while_disconnected",
        "reconnect_recovery_verified",
    ]
    partial["remaining_m005_evidence"] = [
        "operator-confirmed physical cable-loss phase still requires supervised WING cable pull"
    ]

    result = verifier.verify_packet(load_packet(), m005_summary=partial)

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert result["owner_review_input_ready"] is False
    assert "m005_summary_complete_true" in result["failed_checks"]
    assert "m005_summary_completed_evidence_matches" in result["failed_checks"]
    assert "m005_summary_missing_evidence_empty" in result["failed_checks"]
    assert "m005_summary_remaining_evidence_empty" in result["failed_checks"]
    assert result["live_authority_granted"] is False
    assert result["readiness_matrix_update_allowed"] is False


def test_current_partial_m005_artifact_keeps_owner_review_blocked() -> None:
    packet = load_packet()
    partial_m005 = json.loads(PARTIAL_M005_SUMMARY.read_text(encoding="utf-8"))

    result = verifier.verify_packet(
        packet,
        m005_summary=partial_m005,
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=str(PARTIAL_M005_SUMMARY.relative_to(PROJECT_ROOT)),
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert result["owner_review_input_ready"] is False
    assert result["m005_summary"] == (
        "artifacts/wing_timeout_disconnect_proof/"
        "20260704T074537Z_wing_m005_disconnect_timeout_stale_proof/summary.json"
    )
    assert "m005_summary_schema_matches" in result["failed_checks"]
    assert "m005_summary_status_passed" in result["failed_checks"]
    assert "m005_summary_complete_true" in result["failed_checks"]
    assert "m005_summary_missing_evidence_empty" in result["failed_checks"]
    assert "m005_disconnect_phase_operator_confirmed_physical_loss" in result["failed_checks"]
    assert result["live_authority_granted"] is False
    assert result["readiness_closure_allowed"] is False
    assert result["readiness_matrix_update_allowed"] is False


def test_complete_m005_summary_allows_owner_review_input_only() -> None:
    m005 = complete_m005_summary()
    m005["m005_complete"] = True
    packet = load_packet()

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
        **complete_m005_verify_kwargs(m005),
    )

    assert result["status"] == "ready_for_owner_review"
    assert result["reason"] == "m006_owner_review_inputs_verified"
    assert result["owner_review_input_ready"] is True
    assert result["failed_checks"] == []
    assert all(result["checks"].values())
    assert result["proof_artifact_summaries"] == packet["input_proof_artifacts"]
    assert result["live_authority_granted"] is False
    assert result["readiness_closure_allowed"] is False
    assert result["readiness_matrix_update_allowed"] is False
    assert result["contacts_wing"] is False
    assert result["contacts_backend"] is False
    assert result["live_write"] is False
    assert result["checks"]["m005_initial_connection_phase_passed"] is True
    assert result["checks"]["m005_events_initial_connection_write_capable"] is True


def test_packet_readonly_source_must_match_m005_preflight_gate() -> None:
    m005 = complete_m005_summary()
    packet = load_packet()
    packet["accepted_readonly_source_id"] = (
        "artifacts/wing_readonly_preflight/other-run/summary.json"
    )

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
        **complete_m005_verify_kwargs(m005),
    )

    assert result["status"] == "blocked"
    assert result["owner_review_input_ready"] is False
    assert (
        "packet_accepted_readonly_source_matches_m005_gate"
        in result["failed_checks"]
    )


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (
            lambda m005: m005.update({"url": "ws://127.0.0.1:9999"}),
            "m005_summary_url_is_canonical_loopback",
        ),
        (
            lambda m005: m005["phases"]["disconnect"].update({"operator_gate": False}),
            "m005_disconnect_phase_operator_gate_required",
        ),
        (
            lambda m005: m005["provenance"]["source"].update({"worktree_dirty": True}),
            "m005_provenance_source_clean_before_run",
        ),
        (
            lambda m005: m005["provenance"]["backend"].update({
                "runtime_kind": "packaged",
                "path": "/etc/hosts",
            }),
            "m005_provenance_backend_runtime_kind_supported",
        ),
    ],
)
def test_current_release_m005_binding_mutations_fail_closed(mutation, failed_check) -> None:
    m005 = complete_m005_summary()
    kwargs = complete_m005_verify_kwargs(m005)
    mutation(m005)

    result = verifier.verify_packet(
        load_packet(),
        m005_summary=m005,
        proof_summaries_by_artifact=load_packet_proofs(),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
        **kwargs,
    )

    assert result["status"] == "blocked"
    assert result["owner_review_input_ready"] is False
    assert failed_check in result["failed_checks"]


def test_m005_event_hash_is_required_for_direct_verifier_calls() -> None:
    m005 = complete_m005_summary()
    kwargs = complete_m005_verify_kwargs(m005)
    kwargs["m005_events_sha256"] = None

    result = verifier.verify_packet(
        load_packet(),
        m005_summary=m005,
        proof_summaries_by_artifact=load_packet_proofs(),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
        **kwargs,
    )

    assert result["status"] == "blocked"
    assert "m005_provenance_events_hash_matches_loaded" in result["failed_checks"]


def test_m005_prestart_and_runtime_identity_events_are_required() -> None:
    m005 = complete_m005_summary()
    kwargs = complete_m005_verify_kwargs(m005)
    kwargs["m005_event_audit"] = {
        **kwargs["m005_event_audit"],
        "first_event": None,
        "runtime_identity": None,
    }

    result = verifier.verify_packet(
        load_packet(),
        m005_summary=m005,
        proof_summaries_by_artifact=load_packet_proofs(),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
        **kwargs,
    )

    assert result["status"] == "blocked"
    assert "m005_provenance_first_event_is_prestart_capture" in result["failed_checks"]
    assert (
        "m005_provenance_backend_runtime_identity_event_matches_summary"
        in result["failed_checks"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda identity: identity.update(
            {"schema": "automixer.backend_runtime_identity.v1"}
        ),
        lambda identity: (
            identity.update({"started_at_utc": "1999-01-01T00:00:00Z"}),
            identity["process"].update({"started_at_utc": "1999-01-01T00:00:00Z"}),
        ),
        lambda identity: identity["code"]["files"][0].update(
            {"sha256": "f" * 64}
        ),
        lambda identity: identity["launch_binding"].update(
            {"bound_before_process_start": False, "proof_eligible": False}
        ),
    ],
)
def test_m005_runtime_identity_v2_rejects_stale_unbound_or_changed_code(mutation) -> None:
    m005 = complete_m005_summary()
    mutation(m005["runtime_identity"])
    m005["runtime_identity"]["manifest_sha256"] = (
        verifier.runtime_identity_manifest_sha256(m005["runtime_identity"])
    )
    kwargs = complete_m005_verify_kwargs(m005)

    result = verifier.verify_packet(
        load_packet(),
        m005_summary=m005,
        proof_summaries_by_artifact=load_packet_proofs(),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
        **kwargs,
    )

    assert result["status"] == "blocked"
    assert result["owner_review_input_ready"] is False
    assert "m005_provenance_backend_runtime_identity_matches" in result["failed_checks"]


def test_complete_behavioral_m005_without_current_provenance_fails_closed() -> None:
    m005 = complete_m005_summary()
    verification_kwargs = complete_m005_verify_kwargs(m005)
    m005.pop("provenance")
    packet = load_packet()

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
        **verification_kwargs,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m005_current_release_provenance_missing_or_mismatch"
    assert result["m005_behavioral_proof_complete"] is True
    assert result["m005_owner_review_shape_ready"] is True
    assert result["m005_provenance_ready"] is False
    assert result["m005_current_release_accepted"] is False
    assert result["owner_review_input_ready"] is False
    assert "m005_provenance_schema_matches" in result["failed_checks"]


def test_complete_m005_timeout_readback_allows_owner_review_input_only() -> None:
    m005 = complete_m005_summary()
    m005["m005_complete"] = True
    m005["phases"]["stale_readback"]["readback"] = timeout_stale_readback()
    packet = load_packet()

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
        **complete_m005_verify_kwargs(m005),
    )

    assert result["status"] == "ready_for_owner_review"
    assert result["reason"] == "m006_owner_review_inputs_verified"
    assert result["owner_review_input_ready"] is True
    assert "m005_stale_readback_unavailable" not in result["failed_checks"]
    assert result["live_authority_granted"] is False
    assert result["readiness_matrix_update_allowed"] is False


def test_complete_m005_event_stale_readback_allows_owner_review_input_only() -> None:
    m005 = complete_m005_summary()
    m005["m005_complete"] = True
    stale_readback = timeout_stale_readback()
    m005["phases"]["stale_readback"]["readback"] = stale_readback
    packet = load_packet()

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
        **complete_m005_verify_kwargs(m005),
    )

    assert result["status"] == "ready_for_owner_review"
    assert result["reason"] == "m006_owner_review_inputs_verified"
    assert result["owner_review_input_ready"] is True
    assert result["failed_checks"] == []
    assert result["checks"]["m005_events_stale_readback_payload_present"] is True
    assert result["checks"]["m005_events_stale_readback_matches_summary"] is True
    assert result["checks"]["m005_events_stale_readback_unavailable"] is True
    assert result["checks"]["m005_events_reconnect_readback_payload_present"] is True
    assert result["checks"]["m005_events_reconnect_readback_matches_summary"] is True
    assert result["checks"]["m005_events_reconnect_readback_physical"] is True
    assert result["live_authority_granted"] is False
    assert result["readiness_matrix_update_allowed"] is False


def test_m005_event_stale_readback_required_blocks_missing_payload() -> None:
    m005 = complete_m005_summary()
    m005["m005_complete"] = True
    m005["phases"]["stale_readback"]["readback"] = timeout_stale_readback()
    packet = load_packet()

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        m005_event_request_types=complete_m005_event_request_types(m005),
        m005_event_reconnect_readback=m005["phases"]["reconnect_recovery"]["readback"],
        m005_event_stale_readback_required=True,
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "m005_events_stale_readback_payload_present" in result["failed_checks"]
    assert "m005_events_stale_readback_matches_summary" in result["failed_checks"]
    assert "m005_events_stale_readback_unavailable" in result["failed_checks"]
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_matrix_update_allowed"] is False


def test_m005_event_stale_readback_mismatch_blocks_owner_review() -> None:
    m005 = complete_m005_summary()
    m005["m005_complete"] = True
    summary_readback = timeout_stale_readback()
    event_readback = with_m005_target(
        {
            "type": "mixer_parameter_value",
            "status": "readback_unavailable",
            "source": "unavailable",
            "value": None,
        }
    )
    m005["phases"]["stale_readback"]["readback"] = summary_readback
    packet = load_packet()

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        m005_event_request_types=complete_m005_event_request_types(m005),
        m005_event_stale_readback=event_readback,
        m005_event_reconnect_readback=m005["phases"]["reconnect_recovery"]["readback"],
        m005_event_stale_readback_required=True,
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "m005_events_stale_readback_matches_summary" in result["failed_checks"]
    assert "m005_events_stale_readback_payload_present" not in result["failed_checks"]
    assert "m005_events_stale_readback_unavailable" not in result["failed_checks"]
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_closure_allowed"] is False


def test_m005_blocked_unsupported_address_cannot_satisfy_stale_readback() -> None:
    m005 = complete_m005_summary()
    m005["m005_complete"] = True
    m005["phases"]["stale_readback"]["readback"] = with_m005_target(
        {
            "type": "mixer_parameter_read_blocked",
            "status": "blocked",
            "reason": "unsupported_address",
            "value": None,
        }
    )
    packet = load_packet()

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        m005_event_request_types=complete_m005_event_request_types(m005),
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "m005_stale_readback_unavailable" in result["failed_checks"]
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_matrix_update_allowed"] is False


def test_m005_timeout_readback_must_have_canonical_timeout_shape() -> None:
    m005 = complete_m005_summary()
    m005["m005_complete"] = True
    malformed_timeout = timeout_stale_readback()
    malformed_timeout["reason"] = "generic_timeout"
    m005["phases"]["stale_readback"]["readback"] = malformed_timeout
    packet = load_packet()

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        m005_event_request_types=complete_m005_event_request_types(m005),
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "m005_stale_readback_unavailable" in result["failed_checks"]
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_closure_allowed"] is False


def test_packet_contract_must_require_blocked_apply_result() -> None:
    packet = load_packet()
    packet["m005_completion_contract"].pop("required_blocked_write_apply_result_type")
    m005 = complete_m005_summary()
    m005["m005_complete"] = True

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        m005_event_request_types=complete_m005_event_request_types(m005),
        proof_summaries_by_artifact=load_packet_proofs(load_packet()),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "m005_contract_requires_blocked_apply_result" in result["failed_checks"]
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_matrix_update_allowed"] is False


def test_packet_contract_must_forbid_m005_rollback_remediation() -> None:
    packet = load_packet()
    packet["m005_completion_contract"].pop("required_rollback_request_sent")
    packet["m005_completion_contract"].pop("forbidden_remediation_request_types")
    m005 = complete_m005_summary()
    m005["m005_complete"] = True

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        m005_event_request_types=complete_m005_event_request_types(m005),
        proof_summaries_by_artifact=load_packet_proofs(load_packet()),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "m005_contract_requires_no_rollback_request" in result["failed_checks"]
    assert "m005_contract_forbids_remediation_request_types" in result["failed_checks"]
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_matrix_update_allowed"] is False


def test_packet_contract_must_require_m005_readback_target_binding() -> None:
    packet = load_packet()
    packet["m005_completion_contract"].pop("required_readback_target_binding")
    m005 = complete_m005_summary()
    m005["m005_complete"] = True

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        m005_event_request_types=complete_m005_event_request_types(m005),
        proof_summaries_by_artifact=load_packet_proofs(load_packet()),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "m005_contract_requires_readback_target_binding" in result["failed_checks"]
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_matrix_update_allowed"] is False


def test_m005_missing_request_audit_blocks_owner_review() -> None:
    packet = load_packet()
    m005 = complete_m005_summary()
    m005["m005_complete"] = True
    m005.pop("sent_request_types")

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        proof_summaries_by_artifact=load_packet_proofs(packet),
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "m005_summary_sent_request_types_present" in result["failed_checks"]
    assert "m005_summary_apply_probe_sent" in result["failed_checks"]
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_matrix_update_allowed"] is False


def test_m005_forbidden_request_type_blocks_owner_review_even_if_flags_claim_clean() -> None:
    packet = load_packet()
    m005 = complete_m005_summary()
    m005["m005_complete"] = True
    m005["sent_request_types"].append("set_" + "fader")
    m005["forbidden_request_types_sent"] = []
    m005["direct_live_write_request_sent"] = False

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        m005_event_request_types=complete_m005_event_request_types(m005),
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "m005_summary_no_forbidden_direct_request_types" in result["failed_checks"]
    assert "m005_summary_forbidden_requests_match_sent_audit" in result["failed_checks"]
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_closure_allowed"] is False


def test_m005_rollback_request_type_blocks_owner_review_even_if_flags_claim_clean() -> None:
    packet = load_packet()
    m005 = complete_m005_summary()
    m005["m005_complete"] = True
    m005["sent_request_types"].append("rollback_last_manual_write")
    m005["rollback_request_sent"] = False
    m005["forbidden_request_types_sent"] = []
    m005["direct_live_write_request_sent"] = False

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        m005_event_request_types=complete_m005_event_request_types(m005),
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "m005_summary_no_forbidden_remediation_request_types" in result["failed_checks"]
    assert "m005_summary_rollback_request_matches_sent_audit" in result["failed_checks"]
    assert "m005_events_no_forbidden_remediation_request_types" in result["failed_checks"]
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_closure_allowed"] is False


def test_m005_rollback_flag_blocks_owner_review_even_if_request_audit_omits_it() -> None:
    packet = load_packet()
    m005 = complete_m005_summary()
    m005["m005_complete"] = True
    m005["rollback_request_sent"] = True

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        m005_event_request_types=complete_m005_event_request_types(m005),
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "m005_summary_rollback_request_not_sent" in result["failed_checks"]
    assert "m005_summary_rollback_request_matches_sent_audit" in result["failed_checks"]
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_closure_allowed"] is False


def test_m005_reconnect_without_physical_readback_blocks_owner_review() -> None:
    packet = load_packet()
    m005 = complete_m005_summary()
    m005["m005_complete"] = True
    m005["phases"]["reconnect_recovery"]["readback"] = with_m005_target(
        {
            "type": "mixer_parameter_value",
            "status": "readback_unavailable",
            "source": "unavailable",
            "value": None,
        }
    )

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        m005_event_request_types=complete_m005_event_request_types(m005),
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "m005_reconnect_recovery_physical_readback" in result["failed_checks"]
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_matrix_update_allowed"] is False


def test_m005_stale_readback_wrong_target_blocks_owner_review() -> None:
    packet = load_packet()
    m005 = complete_m005_summary()
    m005["m005_complete"] = True
    wrong_target = dict(m005["phases"]["stale_readback"]["readback"])
    wrong_target["channel"] = 4
    wrong_target["address"] = "/ch/4/fdr"
    m005["phases"]["stale_readback"]["readback"] = wrong_target

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        m005_event_request_types=complete_m005_event_request_types(m005),
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "m005_stale_readback_target_matches" in result["failed_checks"]
    assert "m005_stale_readback_unavailable" not in result["failed_checks"]
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_matrix_update_allowed"] is False


def test_m005_reconnect_readback_wrong_target_blocks_owner_review() -> None:
    packet = load_packet()
    m005 = complete_m005_summary()
    m005["m005_complete"] = True
    wrong_target = dict(m005["phases"]["reconnect_recovery"]["readback"])
    wrong_target["channel"] = 4
    wrong_target["address"] = "/ch/4/fdr"
    m005["phases"]["reconnect_recovery"]["readback"] = wrong_target

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        m005_event_request_types=complete_m005_event_request_types(m005),
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "m005_reconnect_readback_target_matches" in result["failed_checks"]
    assert "m005_reconnect_recovery_physical_readback" not in result["failed_checks"]
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_matrix_update_allowed"] is False


def test_m005_event_reconnect_readback_required_blocks_missing_payload(tmp_path: Path) -> None:
    packet = load_packet()
    m005 = complete_m005_summary()
    m005["m005_complete"] = True
    stale_readback = timeout_stale_readback()
    m005["phases"]["stale_readback"]["readback"] = stale_readback
    audit = load_m005_event_audit_from_log(
        tmp_path,
        m005,
        stale_readback=stale_readback,
        reconnect_readback=None,
    )

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        m005_event_request_types=audit["request_types"],
        m005_event_stale_readback=audit["stale_readback"],
        m005_event_reconnect_readback=audit["reconnect_readback"],
        m005_event_stale_readback_required=True,
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "m005_events_reconnect_readback_payload_present" in result["failed_checks"]
    assert "m005_events_reconnect_readback_matches_summary" in result["failed_checks"]
    assert "m005_events_reconnect_readback_physical" in result["failed_checks"]
    assert "m005_events_stale_readback_matches_summary" not in result["failed_checks"]
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_matrix_update_allowed"] is False


def test_m005_event_reconnect_readback_mismatch_blocks_owner_review(tmp_path: Path) -> None:
    packet = load_packet()
    m005 = complete_m005_summary()
    m005["m005_complete"] = True
    stale_readback = timeout_stale_readback()
    summary_reconnect = m005["phases"]["reconnect_recovery"]["readback"]
    event_reconnect = dict(summary_reconnect)
    event_reconnect["value"] = -3.0
    m005["phases"]["stale_readback"]["readback"] = stale_readback
    audit = load_m005_event_audit_from_log(
        tmp_path,
        m005,
        stale_readback=stale_readback,
        reconnect_readback=event_reconnect,
    )

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        m005_event_request_types=audit["request_types"],
        m005_event_stale_readback=audit["stale_readback"],
        m005_event_reconnect_readback=audit["reconnect_readback"],
        m005_event_stale_readback_required=True,
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "m005_events_reconnect_readback_payload_present" not in result["failed_checks"]
    assert "m005_events_reconnect_readback_matches_summary" in result["failed_checks"]
    assert "m005_events_reconnect_readback_physical" not in result["failed_checks"]
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_matrix_update_allowed"] is False


def test_m005_event_reconnect_readback_wrong_target_blocks_owner_review(tmp_path: Path) -> None:
    packet = load_packet()
    m005 = complete_m005_summary()
    m005["m005_complete"] = True
    stale_readback = timeout_stale_readback()
    event_reconnect = dict(m005["phases"]["reconnect_recovery"]["readback"])
    event_reconnect["channel"] = 4
    event_reconnect["address"] = "/ch/4/fdr"
    m005["phases"]["stale_readback"]["readback"] = stale_readback
    audit = load_m005_event_audit_from_log(
        tmp_path,
        m005,
        stale_readback=stale_readback,
        reconnect_readback=event_reconnect,
    )

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        m005_event_request_types=audit["request_types"],
        m005_event_stale_readback=audit["stale_readback"],
        m005_event_reconnect_readback=audit["reconnect_readback"],
        m005_event_stale_readback_required=True,
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "m005_events_reconnect_readback_matches_summary" in result["failed_checks"]
    assert "m005_events_reconnect_readback_target_matches" in result["failed_checks"]
    assert "m005_events_reconnect_readback_physical" not in result["failed_checks"]
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_matrix_update_allowed"] is False


def test_m005_event_reconnect_readback_nonphysical_blocks_owner_review(tmp_path: Path) -> None:
    packet = load_packet()
    m005 = complete_m005_summary()
    m005["m005_complete"] = True
    stale_readback = timeout_stale_readback()
    event_reconnect = with_m005_target(
        {
            "type": "mixer_parameter_value",
            "status": "readback_unavailable",
            "source": "unavailable",
            "value": None,
        }
    )
    m005["phases"]["stale_readback"]["readback"] = stale_readback
    audit = load_m005_event_audit_from_log(
        tmp_path,
        m005,
        stale_readback=stale_readback,
        reconnect_readback=event_reconnect,
    )

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        m005_event_request_types=audit["request_types"],
        m005_event_stale_readback=audit["stale_readback"],
        m005_event_reconnect_readback=audit["reconnect_readback"],
        m005_event_stale_readback_required=True,
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "m005_events_reconnect_readback_payload_present" not in result["failed_checks"]
    assert "m005_events_reconnect_readback_matches_summary" in result["failed_checks"]
    assert "m005_events_reconnect_readback_physical" in result["failed_checks"]
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_matrix_update_allowed"] is False


def test_m005_blocked_write_apply_result_must_be_blocked() -> None:
    packet = load_packet()
    m005 = complete_m005_summary()
    m005["m005_complete"] = True
    m005["phases"]["blocked_write_while_disconnected"]["apply_result"] = {
        "type": "operator_proposal_apply_result",
        "status": "applied",
        "apply_result": {"status": "applied", "reason": "mixer_not_connected"},
    }

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        m005_event_request_types=complete_m005_event_request_types(m005),
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "m005_blocked_write_apply_result_blocked" in result["failed_checks"]
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_matrix_update_allowed"] is False


def test_m005_event_log_request_mismatch_blocks_owner_review() -> None:
    packet = load_packet()
    m005 = complete_m005_summary()
    m005["m005_complete"] = True
    event_request_types = complete_m005_event_request_types(m005)[:-1]

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        m005_event_request_types=event_request_types,
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "m005_events_request_audit_matches_summary" in result["failed_checks"]
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_closure_allowed"] is False


def test_m005_apply_before_stale_query_blocks_owner_review_even_if_flags_claim_complete() -> None:
    packet = load_packet()
    m005 = complete_m005_summary()
    m005["m005_complete"] = True
    bad_sequence = [
        "get_operator_mode",
        "get_state",
        "query_mixer_parameter",
        "set_operator_mode",
        "create_operator_proposal",
        "apply_operator_proposal",
        "query_mixer_parameter",
        "connect_wing",
        "query_mixer_parameter",
    ]
    m005["sent_request_types"] = bad_sequence

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        m005_event_request_types=bad_sequence,
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "m005_summary_request_sequence_proves_disconnect_flow" in result["failed_checks"]
    assert "m005_events_request_sequence_proves_disconnect_flow" in result["failed_checks"]
    assert "m005_events_request_audit_matches_summary" not in result["failed_checks"]
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_closure_allowed"] is False


@pytest.mark.parametrize("mutation", ["missing", "after_baseline", "extra"])
def test_m005_requires_exact_initial_connect_and_reconnect_sequence(mutation: str) -> None:
    packet = load_packet()
    m005 = complete_m005_summary()
    sequence = list(m005["sent_request_types"])
    if mutation == "missing":
        sequence.pop(sequence.index("connect_wing"))
    elif mutation == "after_baseline":
        initial = sequence.pop(sequence.index("connect_wing"))
        baseline = sequence.index("query_mixer_parameter")
        sequence.insert(baseline + 1, initial)
    else:
        sequence.insert(sequence.index("set_operator_mode"), "connect_wing")
    m005["sent_request_types"] = sequence
    kwargs = complete_m005_verify_kwargs(m005)
    kwargs["m005_event_request_types"] = list(sequence)

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
        **kwargs,
    )

    assert result["status"] == "blocked"
    assert "m005_summary_request_sequence_proves_disconnect_flow" in result[
        "failed_checks"
    ]
    assert "m005_events_request_sequence_proves_disconnect_flow" in result[
        "failed_checks"
    ]
    assert result["live_authority_granted"] is False


def test_m005_initial_connection_must_be_write_capable_and_event_bound() -> None:
    packet = load_packet()
    m005 = complete_m005_summary()
    m005["phases"]["initial_connection"]["connection_status"][
        "observation_only"
    ] = True
    kwargs = complete_m005_verify_kwargs(m005)

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
        **kwargs,
    )

    assert result["status"] == "blocked"
    assert "m005_initial_connection_phase_passed" in result["failed_checks"]
    assert "m005_events_initial_connection_write_capable" in result["failed_checks"]
    assert result["live_authority_granted"] is False


def test_m005_event_log_forbidden_request_type_blocks_owner_review_even_if_summary_claims_clean() -> None:
    packet = load_packet()
    m005 = complete_m005_summary()
    m005["m005_complete"] = True
    event_request_types = complete_m005_event_request_types(m005)
    event_request_types.append("set_" + "fader")
    m005["forbidden_request_types_sent"] = []
    m005["direct_live_write_request_sent"] = False

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        m005_event_request_types=event_request_types,
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "m005_events_request_audit_matches_summary" in result["failed_checks"]
    assert "m005_events_no_forbidden_direct_request_types" in result["failed_checks"]
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_closure_allowed"] is False


def test_m005_event_audit_extracts_stale_timeout_before_apply(tmp_path: Path) -> None:
    m005 = complete_m005_summary()
    stale_readback = timeout_stale_readback()
    reconnect_readback = m005["phases"]["reconnect_recovery"]["readback"]
    m005["phases"]["stale_readback"]["readback"] = stale_readback
    summary_path = tmp_path / "summary.json"
    events_path = tmp_path / "events.jsonl"
    m005["events_path"] = str(events_path)
    summary_path.write_text(json.dumps(m005), encoding="utf-8")
    write_m005_event_log(
        events_path,
        complete_m005_event_request_types(m005),
        stale_readback=stale_readback,
        reconnect_readback=reconnect_readback,
    )

    audit, display_events_path = verifier.load_m005_event_audit(
        m005_summary_path=summary_path,
        m005_summary=m005,
    )

    assert display_events_path == str(events_path)
    assert audit["request_types"] == complete_m005_event_request_types(m005)
    assert audit["stale_readback"] == stale_readback
    assert audit["reconnect_readback"] == reconnect_readback
    assert audit["sha256"] == verifier._sha256_bytes(events_path.read_bytes())
    assert audit["first_event"]["seq"] == 1
    assert audit["runtime_identity"] is None


def test_verify_packet_paths_rejects_m005_artifact_outside_canonical_root(tmp_path: Path) -> None:
    m005 = complete_m005_summary()
    m005["m005_complete"] = True
    summary_path = tmp_path / "summary.json"
    events_path = tmp_path / "events.jsonl"
    m005["events_path"] = str(events_path)
    summary_path.write_text(json.dumps(m005), encoding="utf-8")
    write_m005_event_log(
        events_path,
        complete_m005_event_request_types(m005),
        stale_readback=m005["phases"]["stale_readback"]["readback"],
        reconnect_readback=m005["phases"]["reconnect_recovery"]["readback"],
    )

    result = verifier.verify_packet_paths(PACKET_SUMMARY, summary_path)

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert result["owner_review_input_ready"] is False
    assert result["m005_events"] == str(events_path)
    assert "m005_summary_path_matches_required_glob" in result["failed_checks"]
    assert "m005_summary_declared_path_matches_input" in result["failed_checks"]
    assert "m005_events_path_matches_summary_sibling" in result["failed_checks"]
    assert result["live_authority_granted"] is False
    assert result["readiness_closure_allowed"] is False


def test_m005_summary_declared_path_mismatch_blocks_owner_review() -> None:
    packet = load_packet()
    m005 = complete_m005_summary()
    m005["m005_complete"] = True
    m005["summary_path"] = (
        "artifacts/wing_timeout_disconnect_proof/"
        "different_m005_disconnect_timeout_stale_proof/summary.json"
    )

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        m005_event_request_types=complete_m005_event_request_types(m005),
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "m005_summary_declared_path_matches_input" in result["failed_checks"]
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False


def test_m005_summary_declared_events_path_mismatch_blocks_owner_review() -> None:
    packet = load_packet()
    m005 = complete_m005_summary()
    m005["m005_complete"] = True
    m005["events_path"] = (
        "artifacts/wing_timeout_disconnect_proof/"
        "unit_m005_disconnect_timeout_stale_proof/other-events.jsonl"
    )

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        m005_event_request_types=complete_m005_event_request_types(m005),
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "m005_summary_declared_events_path_matches_loaded" in result["failed_checks"]
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False


def test_m005_noncanonical_summary_path_blocks_owner_review() -> None:
    packet = load_packet()
    m005 = complete_m005_summary()
    m005["m005_complete"] = True
    noncanonical_summary = "artifacts/no_wing_release_check/unit_m005/summary.json"
    noncanonical_events = "artifacts/no_wing_release_check/unit_m005/events.jsonl"
    m005["summary_path"] = noncanonical_summary
    m005["events_path"] = noncanonical_events

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        m005_event_request_types=complete_m005_event_request_types(m005),
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=noncanonical_summary,
        m005_events_path=noncanonical_events,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "m005_summary_path_matches_required_glob" in result["failed_checks"]
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False


def test_packet_attempting_to_grant_authority_fails_closed() -> None:
    packet = load_packet()
    packet["live_authority_granted"] = True
    m005 = complete_m005_summary()
    m005["m005_complete"] = True

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        m005_event_request_types=complete_m005_event_request_types(m005),
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "packet_live_authority_granted_denied" in result["failed_checks"]
    assert result["live_authority_granted"] is False
    assert result["readiness_closure_allowed"] is False


def test_missing_referenced_proof_artifact_blocks_owner_review() -> None:
    packet = load_packet()
    packet["proofs"][0]["artifact"] = (
        "artifacts/wing_operator_proposal_validation/"
        "missing_wing_operator_proposal_ch3_fader/summary.json"
    )
    packet["input_proof_artifacts"][0] = packet["proofs"][0]["artifact"]
    m005 = complete_m005_summary()
    m005["m005_complete"] = True

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        m005_event_request_types=complete_m005_event_request_types(m005),
        proof_summaries_by_artifact=load_packet_proofs(packet),
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "fader_supervised_artifact_summary_loaded" in result["failed_checks"]
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_matrix_update_allowed"] is False


def test_proof_artifact_summary_mismatch_blocks_owner_review() -> None:
    packet = load_packet()
    proofs = dict(load_packet_proofs(packet))
    first_artifact = packet["proofs"][0]["artifact"]
    proofs[first_artifact] = copy.deepcopy(proofs[first_artifact])
    proofs[first_artifact]["reason"] = "stale_or_mutated_summary"
    m005 = complete_m005_summary()
    m005["m005_complete"] = True

    result = verifier.verify_packet(
        packet,
        m005_summary=m005,
        m005_event_request_types=complete_m005_event_request_types(m005),
        proof_summaries_by_artifact=proofs,
        m005_summary_path=COMPLETE_M005_SUMMARY_PATH,
        m005_events_path=COMPLETE_M005_EVENTS_PATH,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "m006_owner_review_inputs_failed"
    assert "fader_supervised_artifact_reason_complete" in result["failed_checks"]
    assert result["owner_review_input_ready"] is False
    assert result["live_authority_granted"] is False
    assert result["readiness_closure_allowed"] is False


def test_write_artifact_preserves_blocked_authority_denial(tmp_path: Path) -> None:
    result = verifier.verify_packet(load_packet())

    output = verifier.write_verifier_artifact(
        result,
        artifact_root=tmp_path / "m006_verifier",
        run_id="unit_m006_verifier",
    )

    summary_path = Path(output["summary_path"])
    assert summary_path.exists()
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved["artifact_written"] is True
    assert saved["run_id"] == "unit_m006_verifier"
    assert saved["status"] == "blocked"
    assert saved["reason"] == "m005_summary_required_for_owner_review"
    assert saved["owner_review_input_ready"] is False
    assert saved["live_authority_granted"] is False
    assert saved["readiness_closure_allowed"] is False
    assert saved["readiness_matrix_update_allowed"] is False
    assert saved["contacts_wing"] is False
    assert saved["contacts_backend"] is False
    assert saved["live_write"] is False
