from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "migrate_m005_current_release_provenance.py"
SPEC = importlib.util.spec_from_file_location(
    "migrate_m005_current_release_provenance",
    TOOL_PATH,
)
assert SPEC is not None
migration = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(migration)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _legacy_fixture(tmp_path: Path) -> tuple[Path, Path, str, str]:
    run_id = migration.EXPECTED_HISTORICAL_RUN_ID
    baseline = {
        "type": "mixer_parameter_value",
        "request_id": f"{run_id}:baseline",
        "channel": 3,
        "value_type": "fader",
        "address": "/ch/3/fdr",
        "value": -12.1,
        "status": "verified_from_physical_readback",
        "source": "physical_readback",
    }
    stale = {
        "type": "mixer_parameter_value",
        "request_id": f"{run_id}:stale-probe",
        "channel": 3,
        "value_type": "fader",
        "address": "/ch/3/fdr",
        "value": None,
        "status": "readback_unavailable",
        "source": "physical_readback",
    }
    reconnect = {
        "type": "mixer_parameter_value",
        "request_id": f"{run_id}:reconnect",
        "channel": 3,
        "value_type": "fader",
        "address": "/ch/3/fdr",
        "value": -12.1,
        "status": "verified_from_physical_readback",
        "source": "physical_readback",
    }
    summary = {
        "run_id": run_id,
        "proof_artifact_schema": "wing_timeout_disconnect_proof.v1",
        "status": "passed",
        "m005_complete": True,
        "required_m005_evidence": list(migration.EXPECTED_EVIDENCE),
        "completed_m005_evidence": list(migration.EXPECTED_EVIDENCE),
        "missing_m005_evidence": [],
        "channel": 3,
        "value_type": "fader",
        "address": "/ch/3/fdr",
        "old_value": -12.1,
        "disconnect_mode": "physical-cable-loss",
        "physical_cable_confirmation": "WING_CABLE_PULLED_BY_OPERATOR",
        "sent_request_types": list(migration.EXPECTED_LEGACY_REQUEST_SEQUENCE),
        "forbidden_request_types_sent": [],
        "rollback_request_sent": False,
        "direct_live_write_request_sent": False,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "readiness_matrix_update_allowed": False,
        "preflight_decision_gate": {
            "accepted": True,
            "status": "accepted_decision_ready",
            "expected_accepted_readonly_source_id": (
                migration.EXPECTED_ACCEPTED_READONLY_SOURCE_ID
            ),
        },
        "phases": {
            "baseline": {"status": "passed", "readback": baseline},
            "disconnect": {
                "status": "operator_confirmed_physical_cable_loss",
                "confirmation": "WING_CABLE_PULLED_BY_OPERATOR",
                "operator_gate": True,
            },
            "stale_readback": {"status": "passed", "readback": stale},
            "blocked_write_while_disconnected": {
                "status": "passed",
                "block_reason": "pre_read_cache_not_fresh",
                "apply_result": {
                    "type": "operator_proposal_apply_blocked",
                    "status": "apply_blocked",
                },
            },
            "reconnect_recovery": {
                "status": "passed",
                "unchanged_from_baseline": True,
                "readback": reconnect,
            },
        },
    }
    rows = [
        {"seq": seq, "event_type": "recv", "payload": {"type": "ignored"}}
        for seq in range(1, 63)
    ]
    send_sequences = (1, 4, 6, 8, 12, 14, 16, 18, 21)
    for seq, request_type in zip(send_sequences, migration.EXPECTED_LEGACY_REQUEST_SEQUENCE):
        rows[seq - 1] = {
            "seq": seq,
            "event_type": "send",
            "payload": {"type": request_type},
        }
    rows[6] = {"seq": 7, "event_type": "recv", "payload": baseline}
    rows[14] = {"seq": 15, "event_type": "recv", "payload": stale}
    rows[16] = {
        "seq": 17,
        "event_type": "recv",
        "payload": {
            "type": "operator_proposal_apply_blocked",
            "status": "apply_blocked",
            "apply_result": {"reason": "pre_read_cache_not_fresh"},
        },
    }
    rows[61] = {"seq": 62, "event_type": "recv", "payload": reconnect}
    root = tmp_path / "legacy"
    summary_path = root / "summary.json"
    events_path = root / "events.jsonl"
    _write_json(summary_path, summary)
    events_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary_sha = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    events_sha = hashlib.sha256(events_path.read_bytes()).hexdigest()
    return summary_path, events_path, summary_sha, events_sha


def _console_attestation(
    *,
    challenge: dict[str, object],
    challenge_sha256: str,
) -> dict[str, object]:
    return {
        "schema": migration.CONSOLE_ATTESTATION_SCHEMA,
        "status": "accepted",
        "identity_source": "operator_attested_same_console",
        "hardware_identity_query_performed": False,
        "console_model": "WING Rack",
        "firmware": "3.0.5",
        "serial": "UNIT-WING-RACK-001",
        "migration_run_id": challenge["migration_run_id"],
        "challenge_nonce": challenge["challenge_nonce"],
        "challenge_sha256": challenge_sha256,
        "source_head": challenge["source_head"],
        "source_tree": challenge["source_tree"],
        "source_binding_sha256": challenge["source_binding_sha256"],
        "same_console_as_historical_m005": True,
        "historical_runtime_identity_claimed": False,
        "historical_prestart_provenance_claimed": False,
        "historical_summary_sha256": migration.EXPECTED_HISTORICAL_SUMMARY_SHA256,
        "historical_events_sha256": migration.EXPECTED_HISTORICAL_EVENTS_SHA256,
        "reviewed_by": "unit-owner",
        "reviewed_at_utc": challenge["captured_at_utc"],
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "readiness_matrix_update_allowed": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "live_write": False,
    }


def test_legacy_extractor_accepts_exact_bytes_and_rejects_bit_flip(tmp_path: Path) -> None:
    summary, events, summary_sha, events_sha = _legacy_fixture(tmp_path)
    result = migration.verify_historical_evidence(
        summary,
        events,
        expected_summary_sha256=summary_sha,
        expected_events_sha256=events_sha,
    )
    assert result["ready"] is True
    assert result["failed_checks"] == []

    events.write_bytes(events.read_bytes() + b"\n")
    mutated = migration.verify_historical_evidence(
        summary,
        events,
        expected_summary_sha256=summary_sha,
        expected_events_sha256=events_sha,
    )
    assert mutated["ready"] is False
    assert "historical_events_hash_matches" in mutated["failed_checks"]


def test_real_historical_artifact_and_owner_decision_match_immutable_hashes() -> None:
    historical = migration.verify_historical_evidence(
        migration.EXPECTED_HISTORICAL_SUMMARY_LOGICAL_PATH,
        migration.EXPECTED_HISTORICAL_EVENTS_LOGICAL_PATH,
    )
    decision = migration.verify_owner_decision(
        migration.EXPECTED_OWNER_DECISION_LOGICAL_PATH
    )

    assert historical["ready"] is True
    assert historical["summary_sha256"] == migration.EXPECTED_HISTORICAL_SUMMARY_SHA256
    assert historical["events_sha256"] == migration.EXPECTED_HISTORICAL_EVENTS_SHA256
    assert historical["accepted_readonly_source_id"] == migration.EXPECTED_ACCEPTED_READONLY_SOURCE_ID
    assert decision["ready"] is True
    assert decision["identity"]["sha256"] == migration.EXPECTED_OWNER_DECISION_SHA256


def test_console_attestation_requires_serial_same_console_and_denials(tmp_path: Path) -> None:
    source = migration.capture_current_source()
    run_id = "unit_m005_attestation"
    challenge = migration.build_console_attestation_challenge_payload(
        source=source,
        migration_run_id=run_id,
        nonce="c" * 64,
        captured_at=datetime.now(timezone.utc),
    )
    challenge_path = tmp_path / "challenge.json"
    _write_json(challenge_path, challenge)
    challenge_sha256 = hashlib.sha256(challenge_path.read_bytes()).hexdigest()
    path = tmp_path / "attestation.json"
    _write_json(
        path,
        _console_attestation(
            challenge=challenge,
            challenge_sha256=challenge_sha256,
        ),
    )
    assert migration.verify_console_attestation(
        path,
        challenge_path=challenge_path,
        expected_source=source,
        expected_run_id=run_id,
    )["ready"] is True

    for field, bad_value in (
        ("serial", ""),
        ("same_console_as_historical_m005", False),
        ("firmware", "3.0.4"),
        ("migration_run_id", "other-run"),
        ("challenge_nonce", "d" * 64),
        ("source_head", "0" * 40),
        ("live_authority_granted", True),
        ("reviewed_at_utc", "2000-01-01T00:00:00Z"),
        ("reviewed_at_utc", "2099-01-01T00:00:00Z"),
    ):
        value = _console_attestation(
            challenge=challenge,
            challenge_sha256=challenge_sha256,
        )
        value[field] = bad_value
        _write_json(path, value)
        result = migration.verify_console_attestation(
            path,
            challenge_path=challenge_path,
            expected_source=source,
            expected_run_id=run_id,
        )
        assert result["ready"] is False

    replayed_source = copy.deepcopy(source)
    replayed_source["head"] = "f" * 40
    challenge_result = migration.verify_console_attestation_challenge(
        challenge_path,
        expected_source=replayed_source,
        expected_run_id=run_id,
    )
    assert challenge_result["ready"] is False
    assert (
        "console_attestation_challenge_source_matches"
        in challenge_result["failed_checks"]
    )

    stale_challenge = migration.build_console_attestation_challenge_payload(
        source=source,
        migration_run_id=run_id,
        nonce="e" * 64,
        captured_at=datetime.now(timezone.utc) - timedelta(days=2),
    )
    _write_json(challenge_path, stale_challenge)
    _write_json(
        path,
        _console_attestation(
            challenge=stale_challenge,
            challenge_sha256=hashlib.sha256(
                challenge_path.read_bytes()
            ).hexdigest(),
        ),
    )
    stale = migration.verify_console_attestation(
        path,
        challenge_path=challenge_path,
        expected_source=source,
        expected_run_id=run_id,
    )
    assert stale["ready"] is False
    assert "console_attestation_challenge_ready" in stale["failed_checks"]


@pytest.mark.parametrize("run_id", ["../escape", "..", "/tmp/escape", "bad/name"])
def test_build_rejects_unsafe_run_id_before_writing(
    tmp_path: Path,
    run_id: str,
) -> None:
    with pytest.raises(ValueError, match="run_id"):
        migration.build_migration_artifact(
            historical_summary=migration.EXPECTED_HISTORICAL_SUMMARY_LOGICAL_PATH,
            owner_decision=migration.EXPECTED_OWNER_DECISION_LOGICAL_PATH,
            console_attestation=None,
            console_attestation_challenge=None,
            artifact_root=tmp_path / "runs",
            run_id=run_id,
            execute_replay=False,
            source_override=migration.capture_current_source(),
        )
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    "boundary",
    (
        "root_symlink",
        "parent_symlink",
        "run_symlink",
        "dangling_run_symlink",
    ),
)
def test_build_rejects_symlinked_artifact_boundaries_before_writing(
    tmp_path: Path,
    boundary: str,
) -> None:
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    run_id = "unit_m005_symlink_boundary"

    if boundary == "root_symlink":
        artifact_root = tmp_path / "linked-root"
        artifact_root.symlink_to(real_root, target_is_directory=True)
    elif boundary == "parent_symlink":
        real_parent = tmp_path / "real-parent"
        real_parent.mkdir()
        linked_parent = tmp_path / "linked-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        artifact_root = linked_parent / "missing-root"
    elif boundary == "run_symlink":
        artifact_root = real_root
        outside_run = tmp_path / "outside-run"
        outside_run.mkdir()
        (artifact_root / run_id).symlink_to(
            outside_run,
            target_is_directory=True,
        )
    else:
        artifact_root = real_root
        (artifact_root / run_id).symlink_to(
            artifact_root / "redirected-run",
            target_is_directory=True,
        )

    with pytest.raises((ValueError, FileExistsError)):
        migration.build_migration_artifact(
            historical_summary=migration.EXPECTED_HISTORICAL_SUMMARY_LOGICAL_PATH,
            owner_decision=migration.EXPECTED_OWNER_DECISION_LOGICAL_PATH,
            console_attestation=None,
            console_attestation_challenge=None,
            artifact_root=artifact_root,
            run_id=run_id,
            execute_replay=False,
            source_override=migration.capture_current_source(),
        )

    assert not (real_root / run_id / "summary.json").exists()
    if boundary == "parent_symlink":
        assert not (real_parent / "missing-root").exists()
    if boundary == "run_symlink":
        assert list((tmp_path / "outside-run").iterdir()) == []
    if boundary == "dangling_run_symlink":
        assert not (artifact_root / "redirected-run").exists()


def test_build_and_verify_paths_round_trip_and_reject_mutations(
    tmp_path: Path,
) -> None:
    source = migration.capture_current_source()
    if not all(migration._source_checks(source).values()):
        pytest.skip("full migration round-trip requires a clean source checkout")
    canonical_inputs = (
        migration._resolve(migration.EXPECTED_HISTORICAL_SUMMARY_LOGICAL_PATH),
        migration._resolve(migration.EXPECTED_HISTORICAL_EVENTS_LOGICAL_PATH),
        migration._resolve(migration.EXPECTED_OWNER_DECISION_LOGICAL_PATH),
    )
    if not all(path.is_file() for path in canonical_inputs):
        pytest.skip("canonical local M005/M006 evidence is unavailable")

    run_id = "unit_m005_migration_round_trip"
    challenge_path = tmp_path / "challenge.json"
    migration.write_console_attestation_challenge(
        challenge_path,
        migration_run_id=run_id,
        source_override=source,
    )
    challenge = json.loads(challenge_path.read_text(encoding="utf-8"))
    attestation_path = tmp_path / "attestation.json"
    _write_json(
        attestation_path,
        _console_attestation(
            challenge=challenge,
            challenge_sha256=hashlib.sha256(
                challenge_path.read_bytes()
            ).hexdigest(),
        ),
    )
    artifact_root = tmp_path / "runs"
    built = migration.build_migration_artifact(
        historical_summary=migration.EXPECTED_HISTORICAL_SUMMARY_LOGICAL_PATH,
        owner_decision=migration.EXPECTED_OWNER_DECISION_LOGICAL_PATH,
        console_attestation=attestation_path,
        console_attestation_challenge=challenge_path,
        artifact_root=artifact_root,
        run_id=run_id,
        execute_replay=True,
        source_override=source,
    )
    assert built["migration_ready"] is True
    summary_path = artifact_root / run_id / "summary.json"
    verified = migration.verify_migration_paths(
        summary_path,
        current_source_override=source,
        artifact_root=artifact_root,
    )
    assert verified["migration_ready"] is True
    assert verified["failed_checks"] == []

    report_path = artifact_root / run_id / "compatibility-replay.json"
    original_report = report_path.read_bytes()
    report = json.loads(original_report)
    report["writer_call_counts"]["set_channel_fader"] = 1
    _write_json(report_path, report)
    assert migration.verify_migration_paths(
        summary_path,
        current_source_override=source,
        artifact_root=artifact_root,
    )["migration_ready"] is False
    report_path.write_bytes(original_report)

    decision_snapshot = artifact_root / run_id / "owner" / "m006_owner_decision.json"
    original_decision = decision_snapshot.read_bytes()
    decision_snapshot.unlink()
    decision_snapshot.symlink_to(canonical_inputs[2])
    symlink_result = migration.verify_migration_paths(
        summary_path,
        current_source_override=source,
        artifact_root=artifact_root,
    )
    assert symlink_result["migration_ready"] is False
    assert (
        "migration_owner_decision_snapshot_confined"
        in symlink_result["failed_checks"]
    )
    decision_snapshot.unlink()
    decision_snapshot.write_bytes(original_decision)

    invocation_path = artifact_root / run_id / "compatibility-replay.invocation.json"
    original_invocation = invocation_path.read_bytes()
    invocation = json.loads(original_invocation)
    invocation["challenge"] = "0" * 64
    _write_json(invocation_path, invocation)
    invocation_result = migration.verify_migration_paths(
        summary_path,
        current_source_override=source,
        artifact_root=artifact_root,
    )
    assert invocation_result["migration_ready"] is False
    invocation_path.write_bytes(original_invocation)


def test_verify_summary_cli_forwards_explicit_artifact_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary_path = tmp_path / "private" / "run" / "summary.json"
    artifact_root = tmp_path / "private"
    captured: dict[str, object] = {}

    def fake_verify(
        summary: str | Path,
        *,
        artifact_root: str | Path = migration.DEFAULT_ARTIFACT_ROOT,
    ) -> dict[str, object]:
        captured["summary"] = summary
        captured["artifact_root"] = artifact_root
        return {"status": "blocked"}

    monkeypatch.setattr(migration, "verify_migration_paths", fake_verify)
    result = migration.main(
        [
            "--verify-summary",
            str(summary_path),
            "--artifact-root",
            str(artifact_root),
        ]
    )

    assert result == 2
    assert captured == {
        "summary": str(summary_path),
        "artifact_root": str(artifact_root),
    }
    assert json.loads(capsys.readouterr().out) == {"status": "blocked"}


def test_verify_summary_cli_accepts_explicit_confinement_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact_root = tmp_path / "private"
    summary_path = artifact_root / "unit_private" / "summary.json"
    _write_json(summary_path, {})

    result = migration.main(
        [
            "--verify-summary",
            str(summary_path),
            "--artifact-root",
            str(artifact_root),
        ]
    )
    verified = json.loads(capsys.readouterr().out)

    assert result == 2
    assert verified["checks"]["migration_artifact_root_real_directory"] is True
    assert (
        verified["checks"]["migration_run_directory_real_direct_child"] is True
    )
    assert verified["checks"]["migration_summary_regular_confined_file"] is True
    assert "migration_summary_regular_confined_file" not in verified["failed_checks"]
    for flag in (
        "contacts_backend",
        "contacts_wing",
        "live_authority_granted",
        "readiness_closure_allowed",
        "readiness_matrix_update_allowed",
        "live_write",
    ):
        assert verified[flag] is False


def test_verify_summary_cli_rejects_summary_outside_explicit_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary_path = tmp_path / "private" / "unit_private" / "summary.json"
    _write_json(summary_path, {})

    result = migration.main(
        [
            "--verify-summary",
            str(summary_path),
            "--artifact-root",
            str(tmp_path / "different-root"),
        ]
    )
    verified = json.loads(capsys.readouterr().out)

    assert result == 2
    assert verified["checks"]["migration_summary_regular_confined_file"] is False
    assert "migration_summary_regular_confined_file" in verified["failed_checks"]


def test_verify_summary_cli_rejects_broad_ancestor_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact_root = tmp_path / "private"
    summary_path = artifact_root / "unit_private" / "summary.json"
    _write_json(summary_path, {})

    result = migration.main(
        [
            "--verify-summary",
            str(summary_path),
            "--artifact-root",
            str(tmp_path),
        ]
    )
    verified = json.loads(capsys.readouterr().out)

    assert result == 2
    assert (
        verified["checks"]["migration_run_directory_real_direct_child"] is False
    )
    assert "migration_summary_regular_confined_file" in verified["failed_checks"]


def test_verify_summary_cli_rejects_symlink_artifact_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real_root = tmp_path / "real-root"
    summary_path = real_root / "unit_private" / "summary.json"
    _write_json(summary_path, {})
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(real_root, target_is_directory=True)

    result = migration.main(
        [
            "--verify-summary",
            str(summary_path),
            "--artifact-root",
            str(linked_root),
        ]
    )
    verified = json.loads(capsys.readouterr().out)

    assert result == 2
    assert verified["checks"]["migration_artifact_root_real_directory"] is False
    assert "migration_summary_regular_confined_file" in verified["failed_checks"]


def test_verify_summary_cli_rejects_symlink_run_directory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact_root = tmp_path / "private"
    real_run = artifact_root / "real-run"
    _write_json(real_run / "summary.json", {})
    linked_run = artifact_root / "linked-run"
    linked_run.symlink_to(real_run, target_is_directory=True)

    result = migration.main(
        [
            "--verify-summary",
            str(linked_run / "summary.json"),
            "--artifact-root",
            str(artifact_root),
        ]
    )
    verified = json.loads(capsys.readouterr().out)

    assert result == 2
    assert (
        verified["checks"]["migration_run_directory_real_direct_child"] is False
    )
    assert "migration_summary_regular_confined_file" in verified["failed_checks"]


def test_verify_summary_cli_rejects_symlink_summary_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact_root = tmp_path / "private"
    run_dir = artifact_root / "unit_private"
    run_dir.mkdir(parents=True)
    real_summary = tmp_path / "real-summary.json"
    _write_json(real_summary, {})
    summary_path = run_dir / "summary.json"
    summary_path.symlink_to(real_summary)

    result = migration.main(
        [
            "--verify-summary",
            str(summary_path),
            "--artifact-root",
            str(artifact_root),
        ]
    )
    verified = json.loads(capsys.readouterr().out)

    assert result == 2
    assert verified["checks"]["migration_summary_regular_confined_file"] is False
    assert verified["migration_ready"] is False
    assert "migration_summary_regular_confined_file" in verified["failed_checks"]


def test_verify_summary_cli_default_root_rejects_private_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary_path = tmp_path / "private" / "unit_private" / "summary.json"
    _write_json(summary_path, {})

    result = migration.main(["--verify-summary", str(summary_path)])
    verified = json.loads(capsys.readouterr().out)

    assert result == 2
    assert verified["checks"]["migration_summary_regular_confined_file"] is False
    assert "migration_summary_regular_confined_file" in verified["failed_checks"]


def test_replay_verifier_rejects_writer_call_mutation(tmp_path: Path) -> None:
    # Reuse the real child runner once; this is no-network and writes only tmp.
    source = migration.capture_current_source()
    if not all(migration._source_checks(source).values()):
        pytest.skip("compatibility replay verification requires a clean source checkout")
    replay = migration.run_compatibility_replay(run_dir=tmp_path, source=source)
    assert replay["ready"] is True
    report = copy.deepcopy(replay["report"])
    report["writer_call_counts"]["set_channel_fader"] = 1
    result = migration.verify_replay_report(
        report,
        source=source,
        prestart=replay["prestart"],
        challenge=report["challenge"],
        exit_code=0,
    )
    assert result["ready"] is False
    assert "replay_writer_counters_exact_zero" in result["failed_checks"]


def test_source_checks_require_both_git_and_runtime_source_clean() -> None:
    source = migration.capture_current_source()
    dirty = copy.deepcopy(source)
    dirty["worktree_dirty"] = True
    dirty["status_sha256"] = "1" * 64
    dirty["runtime_source"]["source"]["worktree_dirty"] = True
    dirty["runtime_source"]["source"]["status_sha256"] = "2" * 64
    dirty_checks = migration._source_checks(dirty)
    assert dirty_checks["current_source_clean"] is False
    assert dirty_checks["current_runtime_source_clean"] is False

    clean = copy.deepcopy(source)
    empty = hashlib.sha256(b"").hexdigest()
    clean.update(
        {
            "worktree_dirty": False,
            "status_sha256": empty,
            "tracked_diff_sha256": empty,
        }
    )
    clean["runtime_source"]["source"].update(
        {
            "worktree_dirty": False,
            "status_sha256": empty,
            "tracked_diff_sha256": empty,
        }
    )
    empty_audit = {
        "schema": "automixer.m005_migration_worktree_status_audit.v1",
        "status_sha256": empty,
        "entries": [],
        "allowed_generated_untracked_prefixes": list(
            migration.ALLOWED_GENERATED_UNTRACKED_PREFIXES
        ),
        "allowed_untracked_paths": [],
        "disallowed_entries": [],
    }
    clean["worktree_status_audit"] = copy.deepcopy(empty_audit)
    clean["runtime_worktree_status_audit"] = copy.deepcopy(empty_audit)
    assert all(migration._source_checks(clean).values())

    allowed = copy.deepcopy(clean)
    allowed_path = "reports/master_mix_eval_unit.json"
    allowed_raw = f"?? {allowed_path}\0".encode("utf-8")
    allowed_audit = {
        **empty_audit,
        "status_sha256": hashlib.sha256(allowed_raw).hexdigest(),
        "entries": [{"status": "??", "path": allowed_path}],
        "allowed_untracked_paths": [allowed_path],
    }
    allowed.update(
        {
            "worktree_dirty": True,
            "status_sha256": allowed_audit["status_sha256"],
            "worktree_status_audit": copy.deepcopy(allowed_audit),
            "runtime_worktree_status_audit": copy.deepcopy(allowed_audit),
        }
    )
    allowed["runtime_source"]["source"].update(
        {
            "worktree_dirty": True,
            "status_sha256": allowed_audit["status_sha256"],
        }
    )
    assert all(migration._source_checks(allowed).values())
