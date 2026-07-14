from __future__ import annotations

import asyncio
from copy import deepcopy
import fcntl
import hashlib
import json
import os
import py_compile
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from master_capture_route_proof import (
    MASTER_CAPTURE_ROUTE_ADDRESSES,
    MASTER_CAPTURE_ROUTE_EXPECTED,
    validate_master_capture_route_request,
    write_master_capture_route_artifact,
)
from m110_cp05_hardware_gate import compute_m110_cp05_source_binding
from tools import wing_attended_soundcheck_runner as runner
from runtime_identity import RUNTIME_IDENTITY_SCHEMA, runtime_identity_manifest_sha256
from supervised_proof_capability import (
    M110_RUNTIME_CRITICAL_PATHS,
    validate_m110_arm_request,
    validate_m110_capability_issue_request,
)


def test_m110_runtime_critical_path_binds_manual_write_wal():
    assert "backend/m110_manual_write_wal.py" in M110_RUNTIME_CRITICAL_PATHS
    assert tuple(M110_RUNTIME_CRITICAL_PATHS) == (
        runner.EARLY_M110_RUNTIME_CRITICAL_PATHS
    )


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["/usr/bin/git", *args],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _committed_runner_repo(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    root.mkdir()
    for relative_path in M110_RUNTIME_CRITICAL_PATHS:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative_path}\n", encoding="utf-8")
    output = tmp_path / "child-result.json"
    fake = root / "fake_backend.py"
    fake.write_text(
        """
import json
import os
import sys
import time

def read_all(fd):
    chunks = []
    while True:
        chunk = os.read(fd, 65536)
        if not chunk:
            return b''.join(chunks)
        chunks.append(chunk)

authority_fd = int(os.environ.pop('AUTOMIXER_M110_AUTHORITY_FD'))
token_fd = int(os.environ.pop('AUTOMIXER_WS_AUTH_TOKEN_FD'))
authority = read_all(authority_fd)
token = read_all(token_fd)
os.close(authority_fd)
os.close(token_fd)
with open(sys.argv[1], 'w', encoding='utf-8') as handle:
    json.dump({
        'authority_fd': authority_fd,
        'token_fd': token_fd,
        'authority_size': len(authority),
        'token_size': len(token),
    }, handle)
while True:
    time.sleep(1)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "runner@example.invalid")
    _git(root, "config", "user.name", "Runner Test")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "fixture")
    return root, output


def _write_open_cp05_gate(root: Path) -> None:
    pending_evidence = {
        "schema": "automixer.m110_cp05_software_validation.v1",
        "status": "pending",
        "source_binding_sha256": None,
        "checks": {
            "focused_tests": False,
            "frontend_build_and_tests": False,
            "full_backend_suite": False,
            "independent_review": False,
            "no_wing_release_check": False,
            "released_development_lease": False,
            "static_scan": False,
        },
    }
    evidence_path = root / "Docs" / "reports" / "M110_CP05_SOFTWARE_VALIDATION.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_bytes(
        runner._canonical_json_bytes(pending_evidence) + b"\n"
    )
    pending_gate = {
        "schema": "automixer.m110_cp05_hardware_gate.v1",
        "checkpoint": "M110-CP01",
        "hardware_enabled": False,
        "software_evidence_sha256": None,
    }
    gate_path = root / "ops" / "m110" / "CP05_HARDWARE_GATE.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_bytes(runner._canonical_json_bytes(pending_gate) + b"\n")
    (root / "tracked-source.txt").write_text("validated source\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "runner@example.invalid")
    _git(root, "config", "user.name", "Runner Test")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "pending gate fixture")

    binding = compute_m110_cp05_source_binding(
        repo_root=root,
        critical_paths=runner.M110_RUNTIME_CRITICAL_PATHS,
    )
    evidence = {
        **pending_evidence,
        "status": "passed",
        "source_binding_sha256": binding["sha256"],
        "checks": {name: True for name in pending_evidence["checks"]},
    }
    evidence_raw = runner._canonical_json_bytes(evidence) + b"\n"
    evidence_path.write_bytes(evidence_raw)
    gate = {
        **pending_gate,
        "checkpoint": "M110-CP05",
        "hardware_enabled": True,
        "software_evidence_sha256": "sha256:"
        + hashlib.sha256(evidence_raw).hexdigest(),
    }
    gate_path.write_bytes(runner._canonical_json_bytes(gate) + b"\n")
    _git(root, "add", evidence_path.relative_to(root).as_posix())
    _git(root, "add", gate_path.relative_to(root).as_posix())
    _git(root, "commit", "-q", "-m", "open gate fixture")


class _FakeProcess:
    def __init__(self):
        self.pid = 4242
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.returncode


class _StubbornFakeProcess(_FakeProcess):
    def __init__(self):
        super().__init__()
        self.terminate_called = threading.Event()
        self.kill_count = 0

    def terminate(self):
        self.terminate_called.set()

    def kill(self):
        self.kill_count += 1
        self.returncode = -9


class _ReplacingWriterLockFakeProcess(_StubbornFakeProcess):
    def __init__(self, lock_path):
        super().__init__()
        self.lock_path = lock_path
        self.replaced = threading.Event()

    def wait(self, timeout=None):
        if self.returncode is not None:
            return self.returncode
        if not self.replaced.is_set():
            self.lock_path.unlink()
            replacement_fd = os.open(
                self.lock_path,
                os.O_CREAT | os.O_RDWR,
                0o600,
            )
            os.close(replacement_fd)
            self.replaced.set()
        raise subprocess.TimeoutExpired("fake", timeout)


class _FakeLog:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakeWebSocket:
    def __init__(self, responder=None):
        self.responder = responder
        self.sent = []
        self.responses = asyncio.Queue()
        self.closed = False

    async def send(self, raw):
        payload = json.loads(raw)
        self.sent.append(payload)
        if self.responder is not None:
            responses = self.responder(payload) or []
            for response in responses:
                await self.responses.put(json.dumps(response))

    async def recv(self):
        return await self.responses.get()

    async def close(self, **kwargs):
        self.closed = True


class _BlockingCloseWebSocket(_FakeWebSocket):
    def __init__(self):
        super().__init__()
        self.close_started = asyncio.Event()
        self.allow_close = asyncio.Event()

    async def close(self, **kwargs):
        self.close_started.set()
        await self.allow_close.wait()
        self.closed = True


def _identity_result():
    return {
        "verified": True,
        "runtime_instance_nonce": "e" * 32,
        "runtime_manifest_sha256": "f" * 64,
    }


def _capture_status(**overrides):
    status = {
        "type": "audio_capture_status",
        "running": True,
        "source_type": "sounddevice",
        "real_input": False,
        "callback_age_sec": None,
        "captured_sample_count": 0,
        "device_name": "WING",
        "device_index": 7,
        "num_channels": 48,
        "sample_rate": 48_000,
    }
    status.update(overrides)
    return status


def _fake_backend(root: Path):
    return runner.ProofOwnedBackendProcess(
        process=_FakeProcess(),
        repo_root=root,
        manifest={"launch_nonce": "a" * 32},
        secret=bytearray(b"s" * 32),
        token=bytearray(b"t" * 43),
        ws_url="ws://127.0.0.1:18765",
        log_handle=_FakeLog(),
    )


def _operator_review_callbacks():
    events = []

    def emit(payload):
        events.append(dict(payload))

    async def approve(_prompt):
        for event in reversed(events):
            digest = event.get("operation_digest")
            if event.get("event") == "m110_operator_bundle_review_required":
                return digest
        raise AssertionError("operator review event missing before input")

    return approve, emit, events


def _replan_terminal(**overrides):
    terminal = {
        "type": "m110_soundcheck_replan_result",
        "status": "advanced",
        "success": True,
        "reason": "m110_bounded_split_revision_advanced",
        "run_id": "m110-run-a",
        "from_plan_revision": 1,
        "to_plan_revision": 2,
        "server_owned_selection": True,
        "client_targets_accepted": False,
        "split_operation_count": 1,
        "safe_alternative_operation_count": 0,
        "replan_basis_refreshed": True,
        "new_operator_approval_required_for_each_child": True,
        "console_mutation_write_attempt_count": 0,
        "live_mixer_writes": False,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
    }
    terminal.update(overrides)
    return terminal


def _lane_created_terminal(*, lane="soundcheck", role="proof", **overrides):
    operation_id = "source-operation-1"
    terminal = {
        "type": (
            "m110_processor_enable_proposal_result"
            if lane == "processor"
            else "m110_soundcheck_child_proposal_result"
        ),
        "status": "created",
        "success": True,
        "proposal_id": (
            f"m110-processor-{role}-" + "a" * 32
            if lane == "processor"
            else f"m110-soundcheck-child-{role}-" + "a" * 32
        ),
        "review_contract_version": 2,
        "review_content_digest": "sha256:" + "b" * 64,
        "operation_digest": "sha256:" + "c" * 64,
        "review_preimage_sha256": "",
        "required_capability_kind": (
            "restore_only_proof"
            if role == "proof"
            else "processor_enable_commit_once"
            if lane == "processor"
            else "soundcheck_child_commit_once"
        ),
        "bundle_kind": (
            "processor_enable_bundle"
            if lane == "processor"
            else runner.M110_SOUNDCHECK_SINGLE_BUNDLE_KIND
        ),
        "operation_count": 1,
        "scope_addresses": ["/ch/1/eq/on"],
        "review_operations": [
            {
                "operation_id": (
                    "m110:processor:eq:ch1:0123456789abcdef"
                    if lane == "processor"
                    else operation_id
                ),
                "family_id": "eq_hpf",
                "write_kind": "eq",
                "target_value": {"enabled": True},
            }
        ],
        "source_operation_id": (
            None if lane == "processor" else operation_id
        ),
        "source_operation_ids": [operation_id],
        "candidate_id": (
            "m110-processor-eq-ch1-0123456789abcdef"
            if lane == "processor"
            else None
        ),
        "matching_restored_proof_receipt": role == "commit",
        "token_issued": False,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
    }
    terminal.update(overrides)
    return terminal


@pytest.mark.parametrize("lane", ["processor", "soundcheck"])
def test_lane_created_terminal_binds_role_and_source_provenance(lane):
    terminal = _lane_created_terminal(lane=lane, role="proof")

    created = runner.ProofOwnedBackendSession._validate_lane_proposal_terminal(
        terminal,
        lane=lane,
        role="proof",
    )

    assert created["source_operation_ids"] == ["source-operation-1"]
    assert created["review_operations"] == terminal["review_operations"]


@pytest.mark.parametrize(
    "override",
    [
        {"source_operation_ids": []},
        {"source_operation_ids": ["different-operation"]},
        {"proposal_id": "m110-soundcheck-child-commit-" + "a" * 32},
    ],
)
def test_soundcheck_lane_created_terminal_rejects_provenance_or_role_drift(
    override,
):
    terminal = _lane_created_terminal(
        lane="soundcheck",
        role="proof",
        **override,
    )

    with pytest.raises(RuntimeError, match="m110_lane_proposal_invalid"):
        runner.ProofOwnedBackendSession._validate_lane_proposal_terminal(
            terminal,
            lane="soundcheck",
            role="proof",
        )


def _lane_apply_terminal(created, capability_kind):
    proof_run = capability_kind == "restore_only_proof"
    status = "proof_passed" if proof_run else "applied"
    result_digest = "sha256:" + ("d" if proof_run else "e") * 64
    operation_ids = [
        item["operation_id"] for item in created["review_operations"]
    ]
    return {
        "type": "operator_proposal_apply_result",
        "contract_version": 2,
        "status": status,
        "success": True,
        "proposal_id": created["proposal_id"],
        "proposal_status": status,
        "apply_status": status,
        "accepted": not proof_run,
        "owner_review_required": proof_run,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "proof_run": proof_run,
        "restore_after_verify": proof_run,
        "capability_kind": capability_kind,
        "operation_digest": created["operation_digest"],
        "bundle_kind": created["bundle_kind"],
        "review_preimage_sha256": created.get(
            "review_preimage_sha256"
        ) or None,
        "proof_target_verified": proof_run,
        "applied_operation_ids": operation_ids,
        "rollback_operation_ids": operation_ids if proof_run else [],
        "already_satisfied_operation_ids": [],
        "no_console_send_proven": False,
        "physical_verified": True,
        "audio_recheck_passed": True,
        "terminal_after_console_send": True,
        "wal_terminal": "proof_restored" if proof_run else "committed",
        "write_state_unknown": False,
        "rollback_state_unknown": False,
        "manual_review_required": False,
        "emergency_stop_activated": False,
        "emergency_stop_required": False,
        "physical_state_restored": proof_run,
        "rollback_attempted": proof_run,
        "rollback_succeeded": proof_run,
        "rollback_status": "verified" if proof_run else "not_attempted",
        "parent_commit_recorded": None if proof_run else True,
        "parent_commit_reason": (
            "" if proof_run else "m110_plan_commit_recorded"
        ),
        "proof_receipt_id": (
            None if proof_run else "m110proof_v1_" + "f" * 64
        ),
        "result_digest": result_digest,
        "result_detail_available": True,
        "result_evidence_pinned": False,
        "proposal": {
            "id": created["proposal_id"],
            "status": status,
            "review_content_digest": created["review_content_digest"],
            "result_digest": result_digest,
        },
    }


def _already_satisfied_lane_apply_terminal(created, capability_kind):
    terminal = _lane_apply_terminal(created, capability_kind)
    operation_ids = [
        item["operation_id"] for item in created["review_operations"]
    ]
    terminal.update(
        {
            "applied_operation_ids": [],
            "rollback_operation_ids": [],
            "already_satisfied_operation_ids": operation_ids,
            "no_console_send_proven": True,
            "terminal_after_console_send": False,
            "rollback_attempted": False,
            "rollback_succeeded": False,
            "rollback_status": (
                "not_required_already_satisfied"
                if capability_kind == "restore_only_proof"
                else "not_attempted"
            ),
        }
    )
    return terminal


@pytest.mark.parametrize(
    ("role", "capability_kind"),
    [
        ("proof", "restore_only_proof"),
        ("commit", "soundcheck_child_commit_once"),
    ],
)
def test_lane_apply_terminal_accepts_exact_already_satisfied_two_pass_truth(
    role,
    capability_kind,
):
    created = (
        runner.ProofOwnedBackendSession._validate_lane_proposal_terminal(
            _lane_created_terminal(lane="soundcheck", role=role),
            lane="soundcheck",
            role=role,
        )
    )

    runner.ProofOwnedBackendSession._validate_lane_apply_terminal(
        _already_satisfied_lane_apply_terminal(created, capability_kind),
        created=created,
        capability_kind=capability_kind,
    )


def test_lane_apply_terminal_rejects_partial_already_satisfied_coverage():
    created = (
        runner.ProofOwnedBackendSession._validate_lane_proposal_terminal(
            _lane_created_terminal(lane="soundcheck", role="proof"),
            lane="soundcheck",
            role="proof",
        )
    )
    terminal = _already_satisfied_lane_apply_terminal(
        created,
        "restore_only_proof",
    )
    terminal["already_satisfied_operation_ids"] = ["different-operation"]

    with pytest.raises(RuntimeError, match="m110_lane_apply_terminal_invalid"):
        runner.ProofOwnedBackendSession._validate_lane_apply_terminal(
            terminal,
            created=created,
            capability_kind="restore_only_proof",
        )


def test_lane_apply_terminal_rejects_unrecorded_parent_commit():
    created = (
        runner.ProofOwnedBackendSession._validate_lane_proposal_terminal(
            _lane_created_terminal(lane="soundcheck", role="commit"),
            lane="soundcheck",
            role="commit",
        )
    )
    terminal = _lane_apply_terminal(
        created,
        "soundcheck_child_commit_once",
    )
    terminal["parent_commit_recorded"] = False
    terminal["parent_commit_reason"] = "m110_plan_commit_journal_failed"

    with pytest.raises(RuntimeError, match="m110_lane_apply_terminal_invalid"):
        runner.ProofOwnedBackendSession._validate_lane_apply_terminal(
            terminal,
            created=created,
            capability_kind="soundcheck_child_commit_once",
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("applied_operation_ids", []),
        ("applied_operation_ids", ["source-operation-1"] * 2),
        ("already_satisfied_operation_ids", ["source-operation-1"]),
        ("rollback_operation_ids", []),
        ("applied_operation_ids", ["different-operation"]),
    ],
)
def test_lane_apply_terminal_rejects_inexact_operation_coverage(
    field,
    replacement,
):
    created = (
        runner.ProofOwnedBackendSession._validate_lane_proposal_terminal(
            _lane_created_terminal(lane="soundcheck", role="proof"),
            lane="soundcheck",
            role="proof",
        )
    )
    terminal = _lane_apply_terminal(created, "restore_only_proof")
    terminal[field] = replacement

    with pytest.raises(RuntimeError, match="m110_lane_apply_terminal_invalid"):
        runner.ProofOwnedBackendSession._validate_lane_apply_terminal(
            terminal,
            created=created,
            capability_kind="restore_only_proof",
        )


@pytest.mark.parametrize(
    "capability_kind",
    ["restore_only_proof", "soundcheck_child_commit_once"],
)
def test_lane_apply_terminal_accepts_exact_mixed_operation_coverage(
    capability_kind,
):
    role = (
        "proof" if capability_kind == "restore_only_proof" else "commit"
    )
    created = (
        runner.ProofOwnedBackendSession._validate_lane_proposal_terminal(
            _lane_created_terminal(lane="soundcheck", role=role),
            lane="soundcheck",
            role=role,
        )
    )
    first = created["review_operations"][0]
    second_id = "source-operation-2"
    created = {
        **created,
        "review_operations": [
            first,
            {**first, "operation_id": second_id},
        ],
    }
    terminal = _lane_apply_terminal(created, capability_kind)
    first_id = first["operation_id"]
    terminal.update(
        {
            "applied_operation_ids": [first_id],
            "rollback_operation_ids": (
                [first_id]
                if capability_kind == "restore_only_proof"
                else []
            ),
            "already_satisfied_operation_ids": [second_id],
        }
    )

    runner.ProofOwnedBackendSession._validate_lane_apply_terminal(
        terminal,
        created=created,
        capability_kind=capability_kind,
    )


def _valid_final_audit_terminal(**overrides):
    terminal = {
        "type": "m110_soundcheck_final_audit_result",
        "contract_version": 1,
        "status": "verified",
        "success": True,
        "verified": True,
        "reason": "m110_show_mix_full_reconciliation_verified",
        "run_id": "m110-run-final",
        "mix_readiness": "show_mix_ready",
        "verified_state_sha256": "sha256:" + "a" * 64,
        "committed_surface_ledger_sha256": "sha256:" + "b" * 64,
        "audited_address_count": 248,
        "audit_pass_count": 2,
        "operator_listening_required": True,
        "console_mutation_write_attempt_count": 0,
        "live_mixer_writes": False,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
    }
    terminal.update(overrides)
    return terminal


def test_final_audit_terminal_requires_read_only_two_pass_show_mix_truth():
    validator = (
        runner.ProofOwnedBackendSession
        ._validate_m110_soundcheck_final_audit_terminal
    )
    validated = validator(
        _valid_final_audit_terminal()
    )
    assert validated["verified_state_sha256"] == "sha256:" + "a" * 64

    for override in (
        {"audit_pass_count": 1},
        {"operator_listening_required": False},
        {"readiness_closure_allowed": True},
        {"live_mixer_writes": True},
        {"verified_state_sha256": None},
    ):
        with pytest.raises(
            (RuntimeError, ValueError),
            match="final_audit|verified_state_sha256",
        ):
            validator(_valid_final_audit_terminal(**override))


@pytest.mark.asyncio
async def test_sha_bound_listening_confirmation_rejects_late_ack(monkeypatch):
    clock = iter([100.0, 106.0])
    monkeypatch.setattr(runner.time, "time", lambda: next(clock))

    with pytest.raises(
        RuntimeError,
        match="m110_operator_checkpoint_deadline_expired",
    ):
        await runner._require_exact_operator_literal(
            lambda _prompt: asyncio.sleep(0, result="CONFIRM"),
            prompt="listen",
            expected="CONFIRM",
            deadline_epoch=105.0,
        )


@pytest.mark.asyncio
async def test_session_final_audit_sends_only_fixed_read_only_request(tmp_path):
    def responder(payload):
        if payload["type"] == "run_m110_soundcheck_final_audit":
            assert payload == {
                "type": "run_m110_soundcheck_final_audit",
                "contract_version": 1,
                "audit_selector": "active_soundcheck_full_reconciliation",
            }
            return [_valid_final_audit_terminal()]
        return []

    session = runner.ProofOwnedBackendSession(
        backend=_fake_backend(tmp_path),
        websocket=_FakeWebSocket(responder),
        identity_result=_identity_result(),
        route_established=True,
        route_result={"proof_id": "m110-route"},
        control_armed=True,
        control_approval_id="operator-approved",
    )

    terminal = await session.run_m110_soundcheck_final_audit(
        timeout_seconds=1.0
    )

    assert terminal["verified"] is True
    assert terminal["readiness_closure_allowed"] is False
    await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "responses",
    [
        [
            {
                "type": "auto_soundcheck_status",
                "status": "readiness_invalidated",
            },
            _valid_final_audit_terminal(),
        ],
        [
            _valid_final_audit_terminal(),
            {
                "type": "auto_soundcheck_status",
                "status": "readiness_invalidated",
            },
        ],
    ],
    ids=("before_terminal", "after_terminal_during_grace"),
)
async def test_session_final_audit_latches_interleaved_readiness_invalidation(
    tmp_path,
    responses,
):
    websocket = _FakeWebSocket(
        lambda payload: (
            responses
            if payload["type"] == "run_m110_soundcheck_final_audit"
            else []
        )
    )
    session = runner.ProofOwnedBackendSession(
        backend=_fake_backend(tmp_path),
        websocket=websocket,
        identity_result=_identity_result(),
        route_established=True,
        route_result={"proof_id": "m110-route"},
        control_armed=True,
        control_approval_id="operator-approved",
    )

    with pytest.raises(
        RuntimeError,
        match="m110_soundcheck_readiness_invalidated",
    ):
        await session.run_m110_soundcheck_final_audit(timeout_seconds=1.0)

    assert session._readiness_invalidation_latched is True
    with pytest.raises(
        RuntimeError,
        match="m110_soundcheck_readiness_invalidated",
    ):
        await session._receive_terminal(
            "m110_soundcheck_final_audit_result",
            timeout_seconds=1.0,
        )
    await session.close()


class _FullSoundcheckSession:
    def __init__(self, *, second_audit=None, teardown=None):
        self.identity_result = {"runtime": "m110-test"}
        self.trace = []
        first_audit = _valid_final_audit_terminal()
        self.audits = [
            first_audit,
            dict(second_audit or first_audit),
        ]
        self.teardown = teardown or {"verified": True, "errors": []}
        self.close_count = 0

    async def establish_master_route(self, **_kwargs):
        self.trace.append("route")
        return {"verified": True}

    async def arm_m110_attended_control(self, **_kwargs):
        self.trace.append("arm")
        return {"armed": True}

    async def verify_m110_main_output_checkpoint(self, *, checkpoint):
        self.trace.append(f"main:{checkpoint}")
        return {"verified": True, "checkpoint": checkpoint}

    async def execute_fixed_m110_maintenance_reset(self, **_kwargs):
        self.trace.append("reset")
        return {"verified": True}

    async def execute_m110_m006_exact_preimage_restore(self, **_kwargs):
        self.trace.append("m006")
        return {"verified": True, "m006_restore_complete": True}

    async def run_m110_soundcheck_analysis(self):
        self.trace.append("analysis")
        return {"status": "analyzed"}

    async def execute_m110_interleaved_soundcheck(self, **_kwargs):
        self.trace.append("lanes")
        return {
            "verified": True,
            "processor": {"verified": True},
            "soundcheck": {
                "reason": "show_mix_ready",
                "soundcheck_readiness": {
                    "mix_readiness": "show_mix_ready",
                },
            },
        }

    async def run_m110_soundcheck_final_audit(self):
        self.trace.append("audit")
        return self.audits.pop(0)

    async def close(self):
        self.trace.append("close")
        self.close_count += 1
        return dict(self.teardown)


async def _run_full_soundcheck_with_fake(monkeypatch, tmp_path, session):
    monkeypatch.setattr(
        runner,
        "assert_m110_cp05_hardware_gate",
        lambda _repo_root: {"verified": True},
    )

    async def open_session(**_kwargs):
        return session

    monkeypatch.setattr(
        runner,
        "open_proof_owned_backend_session",
        open_session,
    )
    monkeypatch.setattr(runner.time, "time", lambda: 1_000.0)
    responses = iter(
        [
            runner.M110_RESET_CONTAINMENT_CONFIRMATION,
            runner.M110_SOUNDCHECK_UNMUTE_CONFIRMATION,
            "CONFIRM_M110_HW05_LISTENING_" + "sha256:" + "a" * 64,
        ]
    )

    async def operator_input(_prompt):
        session.trace.append("operator")
        return next(responses)

    async def terminal_emitter(payload):
        session.trace.append("emit:" + str(payload.get("checkpoint")))

    return await runner.run_proof_owned_full_soundcheck(
        repo_root=tmp_path,
        ws_port=18765,
        artifact_root=tmp_path / "artifacts",
        wing_ip="192.0.2.10",
        audio_device_index=7,
        approval_id="m110-test-approval",
        operator_input=operator_input,
        terminal_emitter=terminal_emitter,
    )


@pytest.mark.asyncio
async def test_full_soundcheck_orchestrates_sha_bound_hw05_and_teardown(
    monkeypatch,
    tmp_path,
):
    session = _FullSoundcheckSession()

    result = await _run_full_soundcheck_with_fake(
        monkeypatch,
        tmp_path,
        session,
    )

    assert result["verified"] is True
    assert result["status"] == "show_mix_ready"
    assert result["readiness_closure_allowed"] is True
    assert result["final_audit_before_listening"][
        "verified_state_sha256"
    ] == result["final_audit_after_listening"]["verified_state_sha256"]
    assert session.trace == [
        "route",
        "arm",
        "emit:reset_muted",
        "operator",
        "main:reset_muted",
        "reset",
        "m006",
        "emit:soundcheck_unmuted",
        "operator",
        "main:soundcheck_unmuted",
        "analysis",
        "lanes",
        "main:soundcheck_unmuted",
        "audit",
        "emit:hw05_verified_listening",
        "operator",
        "audit",
        "close",
    ]
    assert session.close_count == 1


@pytest.mark.asyncio
async def test_full_soundcheck_rejects_state_change_after_listening_and_closes(
    monkeypatch,
    tmp_path,
):
    session = _FullSoundcheckSession(
        second_audit=_valid_final_audit_terminal(
            verified_state_sha256="sha256:" + "c" * 64,
        )
    )

    with pytest.raises(
        RuntimeError,
        match="m110_hw05_verified_state_changed_after_listening",
    ):
        await _run_full_soundcheck_with_fake(
            monkeypatch,
            tmp_path,
            session,
        )

    assert session.close_count == 1
    assert session.trace[-1] == "close"


@pytest.mark.asyncio
async def test_full_soundcheck_rejects_unverified_teardown(
    monkeypatch,
    tmp_path,
):
    session = _FullSoundcheckSession(
        teardown={
            "verified": False,
            "errors": ["writer_lock_not_released"],
        }
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "m110_full_soundcheck_teardown_failed:"
            "writer_lock_not_released"
        ),
    ):
        await _run_full_soundcheck_with_fake(
            monkeypatch,
            tmp_path,
            session,
        )

    assert session.close_count == 1


@pytest.mark.parametrize(
    ("role", "capability_kind", "override"),
    [
        ("proof", "restore_only_proof", {"proof_target_verified": False}),
        ("proof", "restore_only_proof", {"no_console_send_proven": True}),
        ("proof", "restore_only_proof", {"terminal_after_console_send": False}),
        ("proof", "restore_only_proof", {"rollback_status": "unknown"}),
        (
            "commit",
            "soundcheck_child_commit_once",
            {"review_preimage_sha256": "tampered"},
        ),
        (
            "commit",
            "soundcheck_child_commit_once",
            {"terminal_after_console_send": False},
        ),
    ],
)
def test_lane_apply_terminal_rejects_missing_physical_truth(
    role,
    capability_kind,
    override,
):
    created = (
        runner.ProofOwnedBackendSession._validate_lane_proposal_terminal(
            _lane_created_terminal(lane="soundcheck", role=role),
            lane="soundcheck",
            role=role,
        )
    )
    terminal = _lane_apply_terminal(created, capability_kind)
    terminal.update(override)

    with pytest.raises(RuntimeError, match="m110_lane_apply_terminal_invalid"):
        runner.ProofOwnedBackendSession._validate_lane_apply_terminal(
            terminal,
            created=created,
            capability_kind=capability_kind,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["processor", "soundcheck"])
async def test_real_lane_step_executes_proof_and_commit_capabilities(
    tmp_path,
    lane,
):
    created_by_role = {
        role: _lane_created_terminal(lane=lane, role=role)
        for role in ("proof", "commit")
    }
    created_by_role["commit"]["operation_digest"] = created_by_role[
        "proof"
    ]["operation_digest"]
    created_by_role["commit"]["scope_addresses"] = list(
        created_by_role["proof"]["scope_addresses"]
    )
    created_by_role["commit"]["review_operations"] = deepcopy(
        created_by_role["proof"]["review_operations"]
    )
    created_by_role["commit"]["source_operation_ids"] = list(
        created_by_role["proof"]["source_operation_ids"]
    )
    created_by_role["commit"]["source_operation_id"] = created_by_role[
        "proof"
    ]["source_operation_id"]
    created_by_role["commit"]["candidate_id"] = created_by_role[
        "proof"
    ]["candidate_id"]
    proposal_lookup = {
        created["proposal_id"]: (role, created)
        for role, created in created_by_role.items()
    }
    issued_tokens = []

    def responder(payload):
        message_type = payload["type"]
        if message_type.startswith("create_m110_"):
            role = "commit" if "_commit_" in message_type else "proof"
            created = deepcopy(created_by_role[role])
            created["type"] = (
                "m110_processor_enable_proposal_result"
                if lane == "processor"
                else "m110_soundcheck_child_proposal_result"
            )
            return [created]
        if message_type == "prepare_m110_restore_only_proof":
            role, created = proposal_lookup[payload["proposal_id"]]
            capability_kind = (
                "restore_only_proof"
                if role == "proof"
                else "processor_enable_commit_once"
                if lane == "processor"
                else "soundcheck_child_commit_once"
            )
            return [
                {
                    "type": "m110_restore_proof_prepare_result",
                    "status": "prepared",
                    "success": True,
                    "reason": (
                        "m110_processor_bundle_current"
                        if lane == "processor"
                        else "m110_soundcheck_child_bundle_current"
                    ),
                    "proposal_id": created["proposal_id"],
                    "review_contract_version": 2,
                    "review_content_digest": created[
                        "review_content_digest"
                    ],
                    "operation_digest": created["operation_digest"],
                    "bundle_kind": created["bundle_kind"],
                    "required_capability_kind": capability_kind,
                    "operation_count": created["operation_count"],
                    "scope_addresses": created["scope_addresses"],
                    "token_issued": False,
                    "live_authority_granted": False,
                }
            ]
        if message_type == "issue_supervised_capability":
            kind = payload["capability_kind"]
            token = "m110cap_v1_" + (
                "1" if kind == "restore_only_proof" else "2"
            ) * 64
            issued_tokens.append(token)
            return [
                {
                    "type": "supervised_capability_result",
                    "status": "issued",
                    "success": True,
                    "reason": "m110_capability_issued",
                    "capability_kind": kind,
                    "proposal_id": payload["proposal_id"],
                    "review_content_digest": payload[
                        "expected_review_content_digest"
                    ],
                    "operation_digest": payload[
                        "expected_operation_digest"
                    ],
                    "approval_id": payload["approval_id"],
                    "idempotency_key": (
                        "m110-proof:" + "3" * 32
                        if kind == "restore_only_proof"
                        else "m110-processor-commit:" + "4" * 32
                        if lane == "processor"
                        else "m110-soundcheck-commit:" + "4" * 32
                    ),
                    "latest_start_epoch": time.time() + 10.0,
                    "supervised_capability_token": token,
                    "single_use": True,
                    "stores_plaintext_token": False,
                    "live_authority_granted": False,
                }
            ]
        if message_type == "apply_operator_proposal":
            role, created = proposal_lookup[payload["proposal_id"]]
            kind = (
                "restore_only_proof"
                if role == "proof"
                else "processor_enable_commit_once"
                if lane == "processor"
                else "soundcheck_child_commit_once"
            )
            return [_lane_apply_terminal(created, kind)]
        return []

    websocket = _FakeWebSocket(responder)
    session = runner.ProofOwnedBackendSession(
        backend=_fake_backend(tmp_path),
        websocket=websocket,
        identity_result=_identity_result(),
    )
    operator_input, terminal_emitter, review_events = (
        _operator_review_callbacks()
    )

    result = await session.execute_m110_lane_step(
        lane=lane,
        approval_id="operator-approved",
        operator_input=operator_input,
        terminal_emitter=terminal_emitter,
    )

    assert result["state"] == "committed"
    assert result["proof"]["capability_kind"] == "restore_only_proof"
    assert result["commit"]["capability_kind"] == (
        "processor_enable_commit_once"
        if lane == "processor"
        else "soundcheck_child_commit_once"
    )
    assert sum(
        event.get("event") == "m110_operator_bundle_review_required"
        for event in review_events
    ) == 2
    issue_requests = [
        item
        for item in websocket.sent
        if item["type"] == "issue_supervised_capability"
    ]
    assert [item["capability_kind"] for item in issue_requests] == [
        "restore_only_proof",
        (
            "processor_enable_commit_once"
            if lane == "processor"
            else "soundcheck_child_commit_once"
        ),
    ]
    durable = json.dumps(result, sort_keys=True) + repr(session)
    assert all(token not in durable for token in issued_tokens)


@pytest.mark.asyncio
async def test_m110_replan_request_is_strict_and_terminal_is_run_bound(tmp_path):
    websocket = _FakeWebSocket(
        lambda payload: (
            [_replan_terminal()]
            if payload.get("type") == "advance_m110_soundcheck_replan"
            else []
        )
    )
    session = runner.ProofOwnedBackendSession(
        backend=_fake_backend(tmp_path),
        websocket=websocket,
        identity_result=_identity_result(),
    )

    result = await session.advance_m110_soundcheck_replan()

    assert result == {
        "state": "advanced",
        "success": True,
        "run_id": "m110-run-a",
        "from_plan_revision": 1,
        "to_plan_revision": 2,
        "split_operation_count": 1,
        "safe_alternative_operation_count": 0,
    }
    assert websocket.sent == [
        {
            "type": "advance_m110_soundcheck_replan",
            "contract_version": 1,
            "replan_selector": "next_bounded_split_revision",
        }
    ]


@pytest.mark.asyncio
async def test_m110_replan_accepts_safe_alternative_only_revision(tmp_path):
    websocket = _FakeWebSocket(
        lambda payload: (
            [
                _replan_terminal(
                    split_operation_count=0,
                    safe_alternative_operation_count=2,
                )
            ]
            if payload.get("type") == "advance_m110_soundcheck_replan"
            else []
        )
    )
    session = runner.ProofOwnedBackendSession(
        backend=_fake_backend(tmp_path),
        websocket=websocket,
        identity_result=_identity_result(),
    )

    result = await session.advance_m110_soundcheck_replan()

    assert result["state"] == "advanced"
    assert result["split_operation_count"] == 0
    assert result["safe_alternative_operation_count"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "override",
    [
        {"run_id": ""},
        {"from_plan_revision": 0, "to_plan_revision": 1},
        {"safe_alternative_operation_count": False},
        {
            "split_operation_count": 0,
            "safe_alternative_operation_count": 0,
        },
        {
            "split_operation_count": 0,
            "safe_alternative_operation_count": 1,
        },
    ],
)
async def test_m110_replan_terminal_rejects_ambiguous_identity_and_counts(
    tmp_path,
    override,
):
    websocket = _FakeWebSocket(lambda _payload: [_replan_terminal(**override)])
    session = runner.ProofOwnedBackendSession(
        backend=_fake_backend(tmp_path),
        websocket=websocket,
        identity_result=_identity_result(),
    )

    with pytest.raises(RuntimeError, match="replan_terminal_invalid"):
        await session.advance_m110_soundcheck_replan()


@pytest.mark.asyncio
async def test_interleaved_replan_revision_identity_is_scoped_per_run(tmp_path):
    session = runner.ProofOwnedBackendSession(
        backend=_fake_backend(tmp_path),
        websocket=_FakeWebSocket(),
        identity_result=_identity_result(),
    )
    lane_steps = {
        "processor": [
            {"state": "complete", "success": True},
            {"state": "complete", "success": True},
            {"state": "complete", "success": True},
            {"state": "complete", "success": True},
        ],
        "soundcheck": [
            {
                "state": "incomplete",
                "success": False,
                "run_id": "m110-run-a",
                "replan_available": True,
                "replan_from_plan_revision": 1,
                "replan_to_plan_revision": 2,
                "replan_split_operation_count": 1,
                "replan_safe_alternative_operation_count": 0,
                "residuals": [{"reason": "operator_review_remaining"}],
            },
            {
                "state": "deferred",
                "success": False,
                "retry_class": "fresh_soundcheck_analysis_required",
            },
            {
                "state": "incomplete",
                "success": False,
                "run_id": "m110-run-b",
                "replan_available": True,
                "replan_from_plan_revision": 1,
                "replan_to_plan_revision": 2,
                "replan_split_operation_count": 0,
                "replan_safe_alternative_operation_count": 2,
                "residuals": [{"reason": "operator_review_remaining"}],
            },
            {
                "state": "complete",
                "success": True,
                "run_id": "m110-run-b",
                "residuals": [],
                "soundcheck_readiness": {"mix_readiness": "show_mix_ready"},
            },
        ],
    }
    replans = [
        {
            "state": "advanced",
            "success": True,
            "run_id": "m110-run-a",
            "from_plan_revision": 1,
            "to_plan_revision": 2,
            "split_operation_count": 1,
            "safe_alternative_operation_count": 0,
        },
        {
            "state": "advanced",
            "success": True,
            "run_id": "m110-run-b",
            "from_plan_revision": 1,
            "to_plan_revision": 2,
            "split_operation_count": 0,
            "safe_alternative_operation_count": 2,
        },
    ]

    async def lane_step(*, lane, **_kwargs):
        return lane_steps[lane].pop(0)

    async def advance(**_kwargs):
        return replans.pop(0)

    async def analyze(**_kwargs):
        return {"verified": True, "run_id": "m110-run-b", "plan_revision": 1}

    session.execute_m110_lane_step = lane_step
    session.advance_m110_soundcheck_replan = advance
    session.run_m110_soundcheck_analysis = analyze

    result = await session.execute_m110_interleaved_soundcheck(
        approval_id="operator-approved",
        operator_input=_operator_review_callbacks()[0],
        terminal_emitter=_operator_review_callbacks()[1],
    )

    assert result["verified"] is True
    assert [item["run_id"] for item in result["replans"]] == [
        "m110-run-a",
        "m110-run-b",
    ]


@pytest.mark.asyncio
async def test_interleaved_replan_rejects_run_binding_mismatch(tmp_path):
    session = runner.ProofOwnedBackendSession(
        backend=_fake_backend(tmp_path),
        websocket=_FakeWebSocket(),
        identity_result=_identity_result(),
    )

    async def lane_step(*, lane, **_kwargs):
        if lane == "processor":
            return {"state": "complete", "success": True}
        return {
            "state": "incomplete",
            "success": False,
            "run_id": "m110-current-run",
            "replan_available": True,
            "replan_from_plan_revision": 1,
            "replan_to_plan_revision": 2,
            "replan_split_operation_count": 1,
            "replan_safe_alternative_operation_count": 0,
            "residuals": [{"reason": "split_target_remaining"}],
        }

    async def advance(**_kwargs):
        return {
            "state": "advanced",
            "success": True,
            "run_id": "m110-other-run",
            "from_plan_revision": 1,
            "to_plan_revision": 2,
            "split_operation_count": 1,
            "safe_alternative_operation_count": 0,
        }

    session.execute_m110_lane_step = lane_step
    session.advance_m110_soundcheck_replan = advance

    with pytest.raises(RuntimeError, match="replan_run_binding_mismatch"):
        await session.execute_m110_interleaved_soundcheck(
            approval_id="operator-approved",
            operator_input=_operator_review_callbacks()[0],
            terminal_emitter=_operator_review_callbacks()[1],
        )


@pytest.mark.asyncio
async def test_interleaved_replan_accepts_fresh_server_recomputed_counts(
    tmp_path,
):
    session = runner.ProofOwnedBackendSession(
        backend=_fake_backend(tmp_path),
        websocket=_FakeWebSocket(),
        identity_result=_identity_result(),
    )
    soundcheck_steps = [
        {
            "state": "incomplete",
            "success": False,
            "run_id": "m110-current-run",
            "replan_available": True,
            "replan_from_plan_revision": 1,
            "replan_to_plan_revision": 2,
            "replan_split_operation_count": 8,
            "replan_safe_alternative_operation_count": 14,
            "residuals": [{"reason": "split_target_remaining"}],
        },
        {
            "state": "complete",
            "success": True,
            "run_id": "m110-current-run",
            "residuals": [],
            "soundcheck_readiness": {"mix_readiness": "show_mix_ready"},
        },
    ]

    async def lane_step(*, lane, **_kwargs):
        if lane == "processor":
            return {"state": "complete", "success": True}
        return soundcheck_steps.pop(0)

    async def advance(**_kwargs):
        return {
            "state": "advanced",
            "success": True,
            "run_id": "m110-current-run",
            "from_plan_revision": 1,
            "to_plan_revision": 2,
            "split_operation_count": 8,
            "safe_alternative_operation_count": 0,
        }

    session.execute_m110_lane_step = lane_step
    session.advance_m110_soundcheck_replan = advance

    result = await session.execute_m110_interleaved_soundcheck(
        approval_id="operator-approved",
        operator_input=_operator_review_callbacks()[0],
        terminal_emitter=_operator_review_callbacks()[1],
    )

    assert result["verified"] is True
    assert result["replans"][0]["safe_alternative_operation_count"] == 0


@pytest.mark.asyncio
async def test_interleaved_scheduler_does_not_infer_replan_from_residual_reason(
    tmp_path,
):
    session = runner.ProofOwnedBackendSession(
        backend=_fake_backend(tmp_path),
        websocket=_FakeWebSocket(),
        identity_result=_identity_result(),
    )

    async def lane_step(*, lane, **_kwargs):
        if lane == "processor":
            return {"state": "complete", "success": True}
        return {
            "state": "incomplete",
            "success": False,
            "run_id": "m110-current-run",
            "replan_available": False,
            "replan_from_plan_revision": None,
            "replan_to_plan_revision": None,
            "replan_split_operation_count": 0,
            "replan_safe_alternative_operation_count": 0,
            "residuals": [{"reason": "split_target_remaining"}],
            "soundcheck_readiness": {
                "mix_readiness": "soundcheck_incomplete"
            },
        }

    session.execute_m110_lane_step = lane_step
    session.advance_m110_soundcheck_replan = AsyncMock()

    result = await session.execute_m110_interleaved_soundcheck(
        approval_id="operator-approved",
        operator_input=_operator_review_callbacks()[0],
        terminal_emitter=_operator_review_callbacks()[1],
    )

    assert result["status"] == "incomplete"
    session.advance_m110_soundcheck_replan.assert_not_awaited()


def _route_values(*, firmware="3.0.5"):
    return {
        "/$syscfg/$serial": "0100Q1V0604AAE",
        "/$syscfg/$firmware": firmware,
        **MASTER_CAPTURE_ROUTE_EXPECTED,
    }


def _route_pass(values):
    return {
        "verified": True,
        "reason": "fixed_master_route_pass_verified",
        "addresses": list(MASTER_CAPTURE_ROUTE_ADDRESSES),
        "values": dict(values),
        "readbacks": {
            address: {
                "status": "received",
                "address": address,
                "value": value,
                "confirmed_round_trips": 2,
            }
            for address, value in values.items()
        },
        "errors": [],
        "confirmed_round_trips_required": 2,
    }


def _hardware_identity():
    binding = {
        "schema": "automixer.wing_capture_hardware_identity.v1",
        "device_name": "WING",
        "device_index": 7,
        "console_serial": "0100Q1V0604AAE",
        "coreaudio": {
            "manufacturer": "BEHRINGER",
            "transport": "usb",
            "input_channels": 48,
            "output_channels": 48,
            "sample_rate_hz": 48_000,
        },
        "usb": {
            "vendor_id": 5015,
            "product_id": 1291,
            "serial": "0100Q1V0604AAE214",
            "location_id": 1_048_576,
        },
        "serial_binding": "chassis_serial_plus_three_digit_usb_interface",
    }
    return {
        **binding,
        "verified": True,
        "reason": "physical_wing_usb_capture_bound_to_console",
        "hardware_uid": "usb:1397:050b:0100Q1V0604AAE214:00100000",
        "binding_sha256": hashlib.sha256(
            runner._canonical_json_bytes(binding)
        ).hexdigest(),
    }


def _valid_route_evidence(request, identity, *, firmware="3.0.5"):
    values = _route_values(firmware=firmware)
    return {
        "request": {
            "request_id": request["request_id"],
            "approval_id": request["approval_id"],
            "request_nonce": request["request_nonce"],
        },
        "runtime": {
            "identity_schema": "automixer.runtime_identity.v2",
            "instance_nonce": identity["runtime_instance_nonce"],
            "manifest_sha256": identity["runtime_manifest_sha256"],
            "head": "a" * 40,
            "tree": "b" * 40,
            "launch_manifest_sha256": "c" * 64,
            "tracked_tree_clean": True,
        },
        "console": {
            "client_type": "EnhancedOSCClient",
            "serial": values["/$syscfg/$serial"],
            "firmware": values["/$syscfg/$firmware"],
            "connection_generation": 1,
            "transport_generation": 1,
            "transport_instance_id": 1234,
        },
        "route": {
            "addresses": list(MASTER_CAPTURE_ROUTE_ADDRESSES),
            "expected": dict(MASTER_CAPTURE_ROUTE_EXPECTED),
            "pass_1": _route_pass(values),
            "pass_2": _route_pass(values),
            "logical_query_count": 12,
            "confirmed_round_trips_required": 2,
            "complete_passes_required": 2,
            "consistent": True,
        },
        "capture": {
            "verified": True,
            "reason": "fresh_real_master_capture_verified",
            "device_name": "WING",
            "device_index": 7,
            "source_type": "sounddevice",
            "capture_instance_id": 1235,
            "capture_stream_instance_id": 1236,
            "channel_count": 48,
            "sample_rate_hz": 48_000,
            "master_channels": [47, 48],
            "captured_sample_count_before": 1_000,
            "required_fresh_sample_count": 12_000,
            "captured_sample_count_snapshot": 13_000,
            "captured_sample_count_after": 13_500,
            "sample_counts": {"47": 12_000, "48": 12_000},
            "sample_peaks_dbfs": {"47": -12.0, "48": -10.0},
            "true_peak_allowance_db": 3.0,
            "conservative_true_peak_dbtp": -7.0,
            "headroom_ceiling_dbtp": -1.0,
            "hardware_identity": _hardware_identity(),
        },
        "safety": {
            "authenticated_control_session": True,
            "operator_mode": "analyze",
            "observation_only": True,
            "console_writer_scope_acquired": True,
            "wal_integrity_ok": True,
            "recovery_transaction_count": 0,
            "emergency_generation": 0,
            "emergency_generation_stable": True,
            "console_mutation_write_attempt_count": 0,
            "normal_live_allowlist_changed": False,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
        },
    }


def _route_result(
    root: Path,
    request,
    identity,
    *,
    mutation=None,
    ttl_seconds=1_800,
):
    proof_id = secrets.token_hex(16)
    issued = time.time() - 1.0
    evidence = _valid_route_evidence(request, identity)
    if mutation is not None:
        mutation(evidence)
    artifact = write_master_capture_route_artifact(
        root=root,
        proof_id=proof_id,
        issued_at_epoch=issued,
        expires_at_epoch=issued + ttl_seconds,
        evidence=evidence,
    )
    capture = evidence["capture"]
    hardware = capture["hardware_identity"]
    return {
        "type": "wing_master_route_proof_result",
        "status": "issued",
        "success": True,
        "accepted": True,
        "reason": "process_private_master_capture_route_authority_issued",
        "proof_id": proof_id,
        "issued_at_epoch": issued,
        "expires_at_epoch": issued + ttl_seconds,
        "artifact": artifact,
        "console_serial": evidence["console"]["serial"],
        "firmware_version": evidence["console"]["firmware"],
        "capture_device": capture["device_name"],
        "capture_device_index": capture["device_index"],
        "capture_hardware_uid": hardware["hardware_uid"],
        "capture_usb_serial": hardware["usb"]["serial"],
        "master_channels": [47, 48],
        "conservative_true_peak_dbtp": capture[
            "conservative_true_peak_dbtp"
        ],
        "console_mutation_write_attempt_count": 0,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
    }


def _valid_restore_only_terminal(
    *,
    proposal_id="proposal-1",
    review_digest="sha256:" + "b" * 64,
    operation_digest="sha256:" + "c" * 64,
):
    result_digest = "sha256:" + "f" * 64
    return {
        "type": "operator_proposal_apply_result",
        "contract_version": 2,
        "status": "proof_passed",
        "success": True,
        "reason": None,
        "proposal_id": proposal_id,
        "proposal_status": "proof_passed",
        "proposal": {
            "id": proposal_id,
            "review_contract_version": 2,
            "review_content_digest": review_digest,
            "status": "proof_passed",
            "result_digest": result_digest,
            "result_detail_available": True,
            "result_evidence_pinned": False,
        },
        "apply_status": "proof_passed",
        "accepted": False,
        "owner_review_required": True,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "proof_run": True,
        "restore_after_verify": True,
        "capability_kind": "restore_only_proof",
        "operation_digest": operation_digest,
        "proof_target_verified": True,
        "no_console_send_proven": False,
        "physical_verified": True,
        "physical_state_restored": True,
        "terminal_after_console_send": True,
        "write_state_unknown": False,
        "rollback_attempted": True,
        "rollback_succeeded": True,
        "rollback_status": "verified",
        "rollback_state_unknown": False,
        "manual_review_required": False,
        "emergency_stop_activated": False,
        "emergency_stop_required": False,
        "wal_terminal": "proof_restored",
        "audio_recheck_passed": True,
        "result_digest": result_digest,
        "result_detail_available": True,
        "result_evidence_pinned": False,
    }


def _fixed_reset_created(
    role,
    *,
    operation_digest="sha256:" + "c" * 64,
    review_preimage_sha256="sha256:" + "9" * 64,
):
    capability_kind = (
        "restore_only_proof"
        if role == "proof"
        else "maintenance_dsp_reset_commit_once"
    )
    return {
        "role": role,
        "proposal_id": f"m110-maintenance-reset-{role}-" + "a" * 32,
        "review_contract_version": 2,
        "review_content_digest": (
            "sha256:" + ("b" if role == "proof" else "d") * 64
        ),
        "operation_digest": operation_digest,
        "review_preimage_sha256": review_preimage_sha256,
        "required_capability_kind": capability_kind,
        "bundle_kind": "maintenance_dsp_reset_bundle",
        "operation_count": len(runner.M110_RESET_ADDRESSES),
        "scope_addresses": list(runner.M110_RESET_ADDRESSES),
        "review_operations": [
            {
                "operation_id": str(row["operation_id"]),
                "family_id": str(row["family_id"]),
                "write_kind": str(row["write_kind"]),
                "channel": int(row["channel"]),
                "parameter": str(row["parameter"]),
                "address": str(row["address"]),
                "target_value": row["target_value"],
            }
            for row in runner.M110_RESET_OPERATION_DEFINITIONS
        ],
    }


def _valid_fixed_reset_terminal(created, capability_kind):
    proof_run = capability_kind == "restore_only_proof"
    status = "proof_passed" if proof_run else "applied"
    result_digest = "sha256:" + ("e" if proof_run else "f") * 64
    after_manifest_sha256 = "sha256:" + "2" * 64
    operation_ids = [
        item["operation_id"] for item in created["review_operations"]
    ]
    return {
        "type": "operator_proposal_apply_result",
        "contract_version": 2,
        "status": status,
        "success": True,
        "reason": "",
        "proposal_id": created["proposal_id"],
        "proposal_status": status,
        "proposal": {
            "id": created["proposal_id"],
            "review_contract_version": 2,
            "review_content_digest": created["review_content_digest"],
            "status": status,
            "result_digest": result_digest,
            "result_detail_available": True,
            "result_evidence_pinned": False,
        },
        "apply_status": status,
        "accepted": False,
        "owner_review_required": True,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "proof_run": proof_run,
        "restore_after_verify": proof_run,
        "capability_kind": capability_kind,
        "operation_digest": created["operation_digest"],
        "bundle_kind": "maintenance_dsp_reset_bundle",
        "review_preimage_sha256": created["review_preimage_sha256"],
        "before_manifest_sha256": "sha256:" + "1" * 64,
        "after_manifest_sha256": after_manifest_sha256,
        "verification_cut_reset_surface_sha256": "sha256:" + "3" * 64,
        "verification_cut_preservation_manifest_sha256": (
            after_manifest_sha256
        ),
        "proof_target_verified": proof_run,
        "applied_operation_ids": operation_ids,
        "rollback_operation_ids": operation_ids if proof_run else [],
        "already_satisfied_operation_ids": [],
        "no_console_send_proven": False,
        "physical_verified": True,
        "physical_state_restored": proof_run,
        "terminal_after_console_send": True,
        "write_state_unknown": False,
        "rollback_attempted": proof_run,
        "rollback_succeeded": proof_run,
        "rollback_state_unknown": False,
        "manual_review_required": False,
        "emergency_stop_activated": False,
        "emergency_stop_required": False,
        "wal_terminal": "proof_restored" if proof_run else "committed",
        "audio_recheck_passed": True,
        "rollback_status": "verified" if proof_run else "not_attempted",
        "no_outside_mutation_attempt_proven": True,
        "current_state_guaranteed_at_response": False,
        "fresh_readback_required_before_next_stage": True,
        "attended_no_touch_required_during_execution": True,
        "later_console_event_invalidates_readiness_freshness": True,
        "result_digest": result_digest,
        "result_detail_available": True,
        "result_evidence_pinned": False,
        "parent_commit_recorded": None if proof_run else True,
        "parent_commit_reason": (
            None if proof_run else "m110_maintenance_reset_commit_recorded"
        ),
    }


def test_authority_packet_is_length_prefixed_manifest_plus_secret():
    secret = secrets.token_bytes(32)
    manifest = {
        "schema": "automixer.m110_attended_prestart_manifest.v1",
        "mode": "m110_attended_runner",
        "head": "a" * 40,
        "tree": "b" * 40,
        "launch_nonce": "c" * 32,
        "files": [],
    }

    packet = runner.build_authority_pipe_packet(manifest, secret=secret)
    size = int.from_bytes(packet[:4], "big")

    assert packet[4 : 4 + size] == runner._canonical_json_bytes(manifest)
    assert packet[4 + size :] == secret


def test_runner_builds_exact_hmac_route_request():
    secret = secrets.token_bytes(32)
    request = runner.build_route_capture_request(
        request_id="m110-route-runner",
        approval_id="operator-approved",
        request_nonce="d" * 32,
        secret=secret,
    )

    assert validate_master_capture_route_request(request) == ()
    assert request["authority_proof"].startswith("hmac-sha256:")


def test_runner_builds_exact_hmac_control_and_capability_requests():
    secret = secrets.token_bytes(32)
    arm = runner.build_m110_attended_control_request(
        approval_id="operator-approved",
        request_nonce="a" * 32,
        secret=secret,
    )
    issue = runner.build_m110_restore_only_capability_request(
        proposal_id="proposal-1",
        review_contract_version=2,
        review_content_digest="sha256:" + "b" * 64,
        operation_digest="sha256:" + "c" * 64,
        approval_id="operator-approved",
        request_nonce="d" * 32,
        secret=secret,
    )
    commit_issue = runner.build_m110_capability_request(
        capability_kind="maintenance_dsp_reset_commit_once",
        proposal_id="proposal-2",
        review_contract_version=2,
        review_content_digest="sha256:" + "e" * 64,
        operation_digest="sha256:" + "f" * 64,
        approval_id="operator-approved",
        request_nonce="1" * 32,
        secret=secret,
    )
    processor_issue = runner.build_m110_capability_request(
        capability_kind="processor_enable_commit_once",
        proposal_id="proposal-3",
        review_contract_version=2,
        review_content_digest="sha256:" + "2" * 64,
        operation_digest="sha256:" + "3" * 64,
        approval_id="operator-approved",
        request_nonce="4" * 32,
        secret=secret,
    )
    soundcheck_issue = runner.build_m110_capability_request(
        capability_kind="soundcheck_child_commit_once",
        proposal_id="proposal-4",
        review_contract_version=2,
        review_content_digest="sha256:" + "5" * 64,
        operation_digest="sha256:" + "6" * 64,
        approval_id="operator-approved",
        request_nonce="7" * 32,
        secret=secret,
    )

    assert validate_m110_arm_request(arm)[0] is True
    assert validate_m110_capability_issue_request(issue)[0] is True
    assert validate_m110_capability_issue_request(commit_issue)[0] is True
    assert validate_m110_capability_issue_request(processor_issue)[0] is True
    assert validate_m110_capability_issue_request(soundcheck_issue)[0] is True
    assert commit_issue["confirmation"] == (
        "ISSUE_MAINTENANCE_DSP_RESET_COMMIT_ONCE"
    )
    assert processor_issue["confirmation"] == (
        "ISSUE_PROCESSOR_ENABLE_COMMIT_ONCE"
    )
    assert soundcheck_issue["confirmation"] == (
        "ISSUE_SOUNDCHECK_CHILD_COMMIT_ONCE"
    )
    with pytest.raises(ValueError, match="review_version"):
        runner.build_m110_restore_only_capability_request(
            proposal_id="proposal-1",
            review_contract_version=True,
            review_content_digest="sha256:" + "b" * 64,
            operation_digest="sha256:" + "c" * 64,
            approval_id="operator-approved",
            request_nonce="d" * 32,
            secret=secret,
        )


def test_restore_only_terminal_validator_rejects_unsafe_or_unbound_truth():
    proposal_id = "proposal-1"
    review_digest = "sha256:" + "b" * 64
    operation_digest = "sha256:" + "c" * 64
    valid = _valid_restore_only_terminal(
        proposal_id=proposal_id,
        review_digest=review_digest,
        operation_digest=operation_digest,
    )
    runner.ProofOwnedBackendSession._validate_restore_only_proof_terminal(
        valid,
        proposal_id=proposal_id,
        review_digest=review_digest,
        operation_digest=operation_digest,
    )

    malformed = []
    for field_name, unsafe_value in (
        ("no_console_send_proven", True),
        ("emergency_stop_activated", True),
        ("emergency_stop_required", True),
        ("result_digest", None),
    ):
        candidate = deepcopy(valid)
        if unsafe_value is None:
            candidate.pop(field_name)
        else:
            candidate[field_name] = unsafe_value
        malformed.append(candidate)
    nested_digest_mismatch = deepcopy(valid)
    nested_digest_mismatch["proposal"]["result_digest"] = (
        "sha256:" + "0" * 64
    )
    malformed.append(nested_digest_mismatch)
    nested_identity_mismatch = deepcopy(valid)
    nested_identity_mismatch["proposal"]["id"] = "proposal-other"
    malformed.append(nested_identity_mismatch)

    for candidate in malformed:
        with pytest.raises(RuntimeError, match="proof_terminal|result_digest"):
            runner.ProofOwnedBackendSession._validate_restore_only_proof_terminal(
                candidate,
                proposal_id=proposal_id,
                review_digest=review_digest,
                operation_digest=operation_digest,
            )


@pytest.mark.parametrize(
    "capability_kind,unsafe_field,unsafe_value",
    (
        ("restore_only_proof", "bundle_kind", "single_soundcheck_operation"),
        ("restore_only_proof", "current_state_guaranteed_at_response", True),
        (
            "maintenance_dsp_reset_commit_once",
            "physical_state_restored",
            True,
        ),
        (
            "maintenance_dsp_reset_commit_once",
            "rollback_attempted",
            True,
        ),
        (
            "maintenance_dsp_reset_commit_once",
            "wal_terminal",
            "proof_restored",
        ),
        (
            "maintenance_dsp_reset_commit_once",
            "parent_commit_recorded",
            False,
        ),
    ),
)
def test_fixed_reset_terminal_validator_rejects_unbound_or_unsafe_truth(
    capability_kind,
    unsafe_field,
    unsafe_value,
):
    role = (
        "proof" if capability_kind == "restore_only_proof" else "commit"
    )
    created = _fixed_reset_created(role)
    valid = _valid_fixed_reset_terminal(created, capability_kind)
    runner.ProofOwnedBackendSession._validate_fixed_reset_apply_terminal(
        valid,
        created=created,
        capability_kind=capability_kind,
    )

    invalid = deepcopy(valid)
    invalid[unsafe_field] = unsafe_value
    with pytest.raises(RuntimeError, match="fixed_reset_apply_terminal"):
        runner.ProofOwnedBackendSession._validate_fixed_reset_apply_terminal(
            invalid,
            created=created,
            capability_kind=capability_kind,
        )


@pytest.mark.parametrize(
    "capability_kind",
    ["restore_only_proof", "maintenance_dsp_reset_commit_once"],
)
def test_fixed_reset_terminal_accepts_exact_already_satisfied_truth(
    capability_kind,
):
    role = (
        "proof"
        if capability_kind == "restore_only_proof"
        else "commit"
    )
    created = _fixed_reset_created(role)
    terminal = _valid_fixed_reset_terminal(created, capability_kind)
    operation_ids = [
        item["operation_id"] for item in created["review_operations"]
    ]
    terminal.update(
        {
            "applied_operation_ids": [],
            "rollback_operation_ids": [],
            "already_satisfied_operation_ids": operation_ids,
            "no_console_send_proven": True,
            "terminal_after_console_send": False,
            "rollback_attempted": False,
            "rollback_succeeded": False,
            "rollback_status": (
                "not_required_already_satisfied"
                if capability_kind == "restore_only_proof"
                else "not_attempted"
            ),
        }
    )

    runner.ProofOwnedBackendSession._validate_fixed_reset_apply_terminal(
        terminal,
        created=created,
        capability_kind=capability_kind,
    )


def test_plan_only_runner_contacts_neither_backend_nor_wing(tmp_path):
    exit_code = runner.main(["--artifact-root", str(tmp_path)])
    summary = json.loads((tmp_path / "plan-summary.json").read_text())

    assert exit_code == 0
    assert summary["status"] == "planned"
    assert summary["contacts_backend"] is False
    assert summary["contacts_wing"] is False
    assert summary["opens_udp_or_osc"] is False
    assert summary["live_write"] is False


def test_default_artifact_root_is_outside_source_checkout():
    resolved = runner.DEFAULT_ARTIFACT_ROOT.expanduser().resolve()
    checkout = runner.REPO_ROOT.resolve()

    assert resolved != checkout
    assert checkout not in resolved.parents


def test_runner_manifest_rejects_ignored_nested_import_shadow(tmp_path):
    root, _ = _committed_runner_repo(tmp_path)
    (root / ".gitignore").write_text("*.so\n", encoding="utf-8")
    _git(root, "add", ".gitignore")
    _git(root, "commit", "-q", "-m", "ignore extension modules")
    shadow = root / "backend" / "osc" / "enhanced_osc_client.so"
    shadow.parent.mkdir(parents=True, exist_ok=True)
    shadow.write_bytes(b"ignored extension shadow")

    try:
        runner.build_runner_prestart_manifest(root, launch_nonce="a" * 32)
    except RuntimeError as exc:
        assert "m110_ignored_import_shadow" in str(exc)
    else:
        raise AssertionError("ignored import shadow must fail closed")


def test_isolated_bootstrap_rejects_ignored_sitecustomize_before_server(
    tmp_path,
):
    root, _ = _committed_runner_repo(tmp_path)
    bootstrap = root / "tools" / "wing_attended_soundcheck_runner.py"
    bootstrap.write_text(Path(runner.__file__).read_text(encoding="utf-8"))
    _git(root, "add", bootstrap.relative_to(root).as_posix())
    _git(root, "commit", "-q", "-m", "install real bootstrap")
    marker = tmp_path / "sitecustomize-executed"
    shadow = root / "backend" / "sitecustomize.py"
    shadow.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n",
        encoding="utf-8",
    )
    (root / ".git" / "info" / "exclude").write_text(
        "backend/sitecustomize.py\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(bootstrap),
            "--_m110-backend-bootstrap",
        ],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=10,
    )

    assert completed.returncode != 0
    assert b"m110_bootstrap_ignored_import_shadow" in completed.stderr
    assert not marker.exists()


def test_early_source_binding_rejects_clean_commit_switch(tmp_path):
    root, _ = _committed_runner_repo(tmp_path)
    _, early_binding = runner._early_validate_source_before_repo_imports(root)
    changed = root / M110_RUNTIME_CRITICAL_PATHS[0]
    changed.write_text("# committed source B\n", encoding="utf-8")
    _git(root, "add", changed.relative_to(root).as_posix())
    _git(root, "commit", "-q", "-m", "switch clean source")
    manifest = runner.build_runner_prestart_manifest(
        root,
        launch_nonce="c" * 32,
    )

    try:
        runner.assert_manifest_matches_early_source_binding(
            manifest,
            early_binding,
        )
    except RuntimeError as exc:
        assert "source_changed_after_early_bootstrap" in str(exc)
    else:
        raise AssertionError("clean commit switch must invalidate early binding")


def test_early_source_binding_matches_later_manifest_without_source_change(
    tmp_path,
):
    root, _ = _committed_runner_repo(tmp_path)
    _, early_binding = runner._early_validate_source_before_repo_imports(root)
    manifest = runner.build_runner_prestart_manifest(
        root,
        launch_nonce="d" * 32,
    )

    runner.assert_manifest_matches_early_source_binding(
        manifest,
        early_binding,
    )
    assert [item["path"] for item in early_binding["files"]] == sorted(
        M110_RUNTIME_CRITICAL_PATHS
    )


def test_isolated_bootstrap_can_load_required_external_dependencies(tmp_path):
    root, _ = _committed_runner_repo(tmp_path)
    bootstrap = root / "tools" / "wing_attended_soundcheck_runner.py"
    bootstrap.write_text(Path(runner.__file__).read_text(encoding="utf-8"))
    _git(root, "add", bootstrap.relative_to(root).as_posix())
    _git(root, "commit", "-q", "-m", "install real bootstrap")

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(bootstrap),
            "--_m110-dependency-smoke",
        ],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=20,
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")


def test_isolated_bootstrap_ignores_checkout_local_valid_bytecode(tmp_path):
    root, _ = _committed_runner_repo(tmp_path)
    bootstrap = root / "tools" / "wing_attended_soundcheck_runner.py"
    bootstrap.write_text(Path(runner.__file__).read_text(encoding="utf-8"))
    marker = tmp_path / "probe-result"
    probe = root / "backend" / "probe_module.py"
    bad_source = (
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('BAD!')\n"
    )
    safe_source = bad_source.replace("BAD!", "SAFE")
    assert len(bad_source.encode()) == len(safe_source.encode())
    probe.write_text(bad_source, encoding="utf-8")
    fixed_mtime = 1_700_000_000
    os.utime(probe, (fixed_mtime, fixed_mtime))
    py_compile.compile(str(probe), doraise=True)
    probe.write_text(safe_source, encoding="utf-8")
    os.utime(probe, (fixed_mtime, fixed_mtime))
    (root / "backend" / "server.py").write_text(
        "import probe_module\n",
        encoding="utf-8",
    )
    (root / ".gitignore").write_text(
        "__pycache__/\n*.pyc\n",
        encoding="utf-8",
    )
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "install bytecode bootstrap fixture")

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(bootstrap),
            "--_m110-backend-bootstrap",
        ],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=20,
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert marker.read_text(encoding="utf-8") == "SAFE"


def test_execute_requires_exact_hardware_coordinates_before_contact(tmp_path):
    exit_code = runner.main(
        [
            "--execute",
            "--approval-id",
            "operator-approved",
            "--confirmation",
            runner.ATTENDED_EXECUTION_CONFIRMATION,
            "--artifact-root",
            str(tmp_path),
        ]
    )
    summary = json.loads((tmp_path / "execution-summary.json").read_text())

    assert exit_code == 2
    assert summary["status"] == "blocked"
    assert "wing_ip_required" in summary["authorization_failures"]
    assert "audio_device_index_required" in summary[
        "authorization_failures"
    ]
    assert all(
        "remaining_capability_checkpoints" not in failure
        for failure in summary["authorization_failures"]
    )
    assert summary["contacts_backend"] is False
    assert summary["contacts_wing"] is False


def test_runner_source_has_no_direct_mixer_transport_import():
    source = Path(runner.__file__).read_text(encoding="utf-8")

    assert "from wing_client import" not in source
    assert "import wing_client" not in source
    assert "socket.socket" not in source


def test_spawn_uses_distinct_one_shot_fds_and_guarantees_child_teardown(
    tmp_path,
):
    root, output = _committed_runner_repo(tmp_path)
    backend = runner.spawn_proof_owned_backend(
        repo_root=root,
        ws_port=19876,
        artifact_root=tmp_path / "run-artifacts",
        backend_command=[sys.executable, str(root / "fake_backend.py"), str(output)],
    )
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and not output.exists():
        time.sleep(0.02)
    assert output.exists()
    child_result = json.loads(output.read_text(encoding="utf-8"))
    assert child_result["authority_fd"] != child_result["token_fd"]
    assert child_result["authority_size"] > 36
    assert child_result["token_size"] == 43
    assert backend.process.poll() is None
    secret_hex = bytes(backend.secret).hex()
    token_text = bytes(backend.token).decode("ascii")
    assert secret_hex not in repr(backend)
    assert token_text not in repr(backend)
    assert backend.writer_lock_fd is not None
    pinned_lock_fd = backend.writer_lock_fd
    pinned_metadata = os.fstat(pinned_lock_fd)
    path_metadata = backend.writer_lock_path.lstat()
    assert (pinned_metadata.st_dev, pinned_metadata.st_ino) == (
        path_metadata.st_dev,
        path_metadata.st_ino,
    )
    rebuilt = runner.build_runner_prestart_manifest(
        root,
        launch_nonce="b" * 32,
    )
    assert rebuilt["launch_nonce"] == "b" * 32

    backend.close()

    assert backend.process.poll() is not None
    assert set(backend.secret) == {0}
    assert set(backend.token) == {0}
    assert backend.writer_lock_fd is None
    with pytest.raises(OSError):
        os.fstat(pinned_lock_fd)


def test_forced_child_kill_waits_until_parent_owns_writer_lock(tmp_path):
    lock_path = tmp_path / "console_writer.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    process = _StubbornFakeProcess()
    result_holder = []

    thread = threading.Thread(
        target=lambda: result_holder.append(
            runner._terminate_exact_child_safely(
                process,
                writer_lock_path=lock_path,
                graceful_timeout_seconds=0.01,
                poll_interval_seconds=0.01,
            )
        ),
        daemon=True,
    )
    thread.start()
    assert process.terminate_called.wait(timeout=1.0)
    time.sleep(0.05)
    assert process.kill_count == 0
    assert thread.is_alive()

    fcntl.flock(lock_fd, fcntl.LOCK_UN)
    os.close(lock_fd)
    thread.join(timeout=1.0)

    assert thread.is_alive() is False
    assert process.kill_count == 1
    assert result_holder[0]["forced"] is True
    assert result_holder[0]["clean_shutdown"] is False
    assert result_holder[0]["writer_lock_busy_observed"] is True
    assert result_holder[0][
        "writer_lock_acquired_before_forced_kill"
    ] is True


def test_writer_lock_inode_replacement_can_never_authorize_sigkill(tmp_path):
    lock_path = tmp_path / "console_writer.lock"
    old_lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(old_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    process = _ReplacingWriterLockFakeProcess(lock_path)
    result_holder = []

    thread = threading.Thread(
        target=lambda: result_holder.append(
            runner._terminate_exact_child_safely(
                process,
                writer_lock_path=lock_path,
                graceful_timeout_seconds=0.01,
                poll_interval_seconds=0.01,
            )
        ),
        daemon=True,
    )
    thread.start()
    assert process.terminate_called.wait(timeout=1.0)
    assert process.replaced.wait(timeout=1.0)
    fcntl.flock(old_lock_fd, fcntl.LOCK_UN)
    os.close(old_lock_fd)
    time.sleep(0.05)

    assert process.kill_count == 0
    assert thread.is_alive()
    process.returncode = 0
    thread.join(timeout=1.0)

    assert thread.is_alive() is False
    assert process.kill_count == 0
    result = result_holder[0]
    assert result["forced"] is False
    assert result["writer_lock_identity_changed"] is True
    assert result["sigkill_withheld"] is True
    assert "writer_lock_identity_changed" in result["errors"]


def test_backend_process_close_records_clean_nonforced_shutdown(tmp_path):
    backend = _fake_backend(tmp_path)

    result = backend.close()

    assert result["terminated"] is True
    assert result["clean_shutdown"] is True
    assert result["forced"] is False
    assert result["returncode"] == 0
    assert backend.close() == result
    assert set(backend.secret) == {0}
    assert set(backend.token) == {0}


@pytest.mark.asyncio
async def test_forced_backend_shutdown_cannot_verify_session_teardown(tmp_path):
    backend = _fake_backend(tmp_path)
    backend.process.returncode = -9
    backend.shutdown_result = {
        "terminated": True,
        "clean_shutdown": False,
        "forced": True,
        "returncode": -9,
        "errors": [],
    }
    session = runner.ProofOwnedBackendSession(
        backend=backend,
        websocket=_FakeWebSocket(),
        identity_result=_identity_result(),
    )

    teardown = await session.close()

    assert teardown["verified"] is False
    assert teardown["backend_process_terminated"] is True
    assert teardown["backend_shutdown"]["forced"] is True
    assert "proof_owned_backend_shutdown_unverified" in teardown["errors"]


def test_spawn_rejects_existing_nonprivate_artifact_root_without_chmod(tmp_path):
    root, output = _committed_runner_repo(tmp_path)
    sentinel = tmp_path / "shared-artifacts"
    sentinel.mkdir(mode=0o755)
    sentinel.chmod(0o755)

    try:
        runner.spawn_proof_owned_backend(
            repo_root=root,
            ws_port=19877,
            artifact_root=sentinel,
            backend_command=[
                sys.executable,
                str(root / "fake_backend.py"),
                str(output),
            ],
        )
    except RuntimeError as exc:
        assert "mode_must_be_0700" in str(exc)
    else:
        raise AssertionError("non-private artifact root must be rejected")

    assert sentinel.stat().st_mode & 0o777 == 0o755


def test_challenged_runtime_identity_binds_child_manifest_and_secret():
    secret = secrets.token_bytes(32)
    manifest = {
        "schema": "automixer.m110_attended_prestart_manifest.v1",
        "mode": "m110_attended_runner",
        "head": "a" * 40,
        "tree": "b" * 40,
        "launch_nonce": "c" * 32,
        "files": [],
    }
    challenge = "d" * 32
    identity = {
        "schema": RUNTIME_IDENTITY_SCHEMA,
        "runtime_kind": "source",
        "instance_nonce": "e" * 32,
        "process": {"pid": 1234},
        "source": {"head": manifest["head"], "tree": manifest["tree"]},
        "websocket": {"port": 19876, "loopback_only": True},
        "launch_binding": {
            "mode": "m110_attended_runner",
            "launch_nonce": manifest["launch_nonce"],
            "head": manifest["head"],
            "tree": manifest["tree"],
            "prestart_manifest_sha256": __import__("hashlib").sha256(
                runner._canonical_json_bytes(manifest)
            ).hexdigest(),
            "authority_secret_sha256": __import__("hashlib").sha256(secret).hexdigest(),
            "tracked_tree_clean": True,
            "bound_before_process_start": True,
        },
        "live_authority_granted": False,
        "read_only_identity": True,
        "request_challenge_nonce": challenge,
    }
    identity["manifest_sha256"] = runtime_identity_manifest_sha256(identity)

    result = runner.validate_challenged_runtime_identity(
        identity=identity,
        challenge_nonce=challenge,
        manifest=manifest,
        secret=secret,
        child_pid=1234,
        ws_port=19876,
    )

    assert result["verified"] is True
    identity["process"]["pid"] = 9999
    assert runner.validate_challenged_runtime_identity(
        identity=identity,
        challenge_nonce=challenge,
        manifest=manifest,
        secret=secret,
        child_pid=1234,
        ws_port=19876,
    )["verified"] is False


def test_persistent_route_session_uses_one_websocket_until_verified_teardown(
    tmp_path,
):
    async def scenario():
        repo = tmp_path / "repo"
        repo.mkdir()
        _write_open_cp05_gate(repo)
        artifact_root = tmp_path / "route-artifacts"
        identity = _identity_result()

        def responder(payload):
            if payload["type"] == "connect_wing":
                return [
                    {"type": "mixer_update", "address": "/ch/1/fdr"},
                    {
                        "type": "connection_status",
                        "connected": True,
                        "mode": "wing",
                        "ip": "192.168.1.102",
                        "observation_only": True,
                        "record_telemetry": False,
                    },
                ]
            if payload["type"] == "start_audio_capture_preview":
                return [
                    {
                        "type": "audio_capture_status",
                        "running": True,
                        "preview": True,
                        "owned_by_requester": True,
                        "source_type": "sounddevice",
                        "real_input": True,
                        "callback_age_sec": 0.0,
                        "captured_sample_count": 12_000,
                        "device_name": "WING",
                        "device_index": 7,
                        "num_channels": 48,
                        "sample_rate": 48_000,
                    }
                ]
            if payload["type"] == "capture_wing_master_route_proof":
                return [_route_result(artifact_root, payload, identity)]
            if payload["type"] == "stop_audio_capture_preview":
                return [
                    {
                        "type": "audio_capture_status",
                        "running": False,
                        "preview": True,
                        "owned_by_requester": True,
                    }
                ]
            if payload["type"] == "disconnect_wing":
                return [{"type": "connection_status", "connected": False}]
            return []

        backend = _fake_backend(repo)
        websocket = _FakeWebSocket(responder)
        session = runner.ProofOwnedBackendSession(
            backend=backend,
            websocket=websocket,
            identity_result=identity,
        )
        route = await session.establish_master_route(
            wing_ip="192.168.1.102",
            audio_device_index=7,
            approval_id="operator-approved",
            request_id="m110-route-test",
            request_nonce="d" * 32,
            route_artifact_root=artifact_root,
        )

        assert route["verified"] is True
        assert route["audio_capture_status"]["readiness_poll_count"] == 0
        assert route["same_websocket_retained"] is True
        assert route["backend_process_retained"] is True
        assert session.closed is False
        assert backend.process.poll() is None
        assert [item["type"] for item in websocket.sent] == [
            "connect_wing",
            "start_audio_capture_preview",
            "capture_wing_master_route_proof",
        ]

        teardown = await session.close()
        assert teardown["verified"] is True
        assert teardown["route_authority_retained"] is False
        assert backend.process.poll() is not None
        assert websocket.closed is True
        assert [item["type"] for item in websocket.sent][-2:] == [
            "stop_audio_capture_preview",
            "disconnect_wing",
        ]

    asyncio.run(scenario())


def test_route_waits_for_same_capture_first_callback_before_proof(
    tmp_path,
    monkeypatch,
):
    async def scenario():
        repo = tmp_path / "repo"
        repo.mkdir()
        _write_open_cp05_gate(repo)
        artifact_root = tmp_path / "route-artifacts"
        identity = _identity_result()
        monkeypatch.setattr(
            runner,
            "M110_CAPTURE_READY_POLL_INTERVAL_SECONDS",
            0.001,
        )

        def responder(payload):
            if payload["type"] == "connect_wing":
                return [
                    {
                        "type": "connection_status",
                        "connected": True,
                        "mode": "wing",
                        "ip": "192.168.1.102",
                        "observation_only": True,
                        "record_telemetry": False,
                    }
                ]
            if payload["type"] == "start_audio_capture_preview":
                return [
                    _capture_status(
                        preview=True,
                        owned_by_requester=True,
                    )
                ]
            if payload["type"] == "get_audio_capture_status":
                return [
                    _capture_status(
                        real_input=True,
                        callback_age_sec=0.01,
                        captured_sample_count=1_024,
                    )
                ]
            if payload["type"] == "capture_wing_master_route_proof":
                return [_route_result(artifact_root, payload, identity)]
            if payload["type"] == "stop_audio_capture_preview":
                return [
                    {
                        "type": "audio_capture_status",
                        "running": False,
                        "preview": True,
                        "owned_by_requester": True,
                    }
                ]
            if payload["type"] == "disconnect_wing":
                return [{"type": "connection_status", "connected": False}]
            return []

        backend = _fake_backend(repo)
        websocket = _FakeWebSocket(responder)
        session = runner.ProofOwnedBackendSession(
            backend=backend,
            websocket=websocket,
            identity_result=identity,
        )
        route = await session.establish_master_route(
            wing_ip="192.168.1.102",
            audio_device_index=7,
            approval_id="operator-approved",
            request_id="m110-delayed-callback",
            request_nonce="d" * 32,
            route_artifact_root=artifact_root,
        )

        assert route["verified"] is True
        assert route["audio_capture_status"]["readiness_poll_count"] == 1
        assert route["audio_capture_status"]["captured_sample_count"] == 1_024
        assert [item["type"] for item in websocket.sent[:4]] == [
            "connect_wing",
            "start_audio_capture_preview",
            "get_audio_capture_status",
            "capture_wing_master_route_proof",
        ]
        teardown = await session.close()
        assert teardown["verified"] is True
        assert teardown["capture_stopped"] is True
        assert teardown["mixer_disconnected"] is True
        assert teardown["backend_process_terminated"] is True
        assert teardown["route_authority_retained"] is False
        assert teardown["errors"] == []

    asyncio.run(scenario())


def test_capture_callback_timeout_stops_preview_before_disconnect(
    tmp_path,
    monkeypatch,
):
    async def scenario():
        repo = tmp_path / "repo"
        repo.mkdir()
        _write_open_cp05_gate(repo)
        monkeypatch.setattr(
            runner,
            "M110_CAPTURE_READY_TIMEOUT_SECONDS",
            0.04,
        )
        monkeypatch.setattr(
            runner,
            "M110_CAPTURE_READY_POLL_INTERVAL_SECONDS",
            0.001,
        )

        def responder(payload):
            if payload["type"] == "connect_wing":
                return [
                    {
                        "type": "connection_status",
                        "connected": True,
                        "mode": "wing",
                        "ip": "192.168.1.102",
                        "observation_only": True,
                        "record_telemetry": False,
                    }
                ]
            if payload["type"] == "start_audio_capture_preview":
                return [
                    _capture_status(
                        preview=True,
                        owned_by_requester=True,
                    )
                ]
            if payload["type"] == "get_audio_capture_status":
                return [_capture_status()]
            if payload["type"] == "stop_audio_capture_preview":
                return [
                    {
                        "type": "audio_capture_status",
                        "running": False,
                        "preview": True,
                        "owned_by_requester": True,
                    }
                ]
            if payload["type"] == "disconnect_wing":
                return [{"type": "connection_status", "connected": False}]
            return []

        backend = _fake_backend(repo)
        websocket = _FakeWebSocket(responder)
        session = runner.ProofOwnedBackendSession(
            backend=backend,
            websocket=websocket,
            identity_result=_identity_result(),
        )
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="not_ready_within_1s"):
            await session.establish_master_route(
                wing_ip="192.168.1.102",
                audio_device_index=7,
                approval_id="operator-approved",
            )
        assert time.monotonic() - started < 0.25
        message_types = [item["type"] for item in websocket.sent]
        assert "capture_wing_master_route_proof" not in message_types
        assert message_types[-2:] == [
            "stop_audio_capture_preview",
            "disconnect_wing",
        ]
        assert session.closed is True

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("initial_override", "failure"),
    [
        ({"error": "capture_failed"}, "capture_failed"),
        ({"device_index": True}, "device_index"),
    ],
)
def test_owned_capture_with_invalid_initial_status_is_explicitly_stopped(
    tmp_path,
    initial_override,
    failure,
):
    async def scenario():
        repo = tmp_path / "repo"
        repo.mkdir()
        _write_open_cp05_gate(repo)

        def responder(payload):
            if payload["type"] == "connect_wing":
                return [
                    {
                        "type": "connection_status",
                        "connected": True,
                        "mode": "wing",
                        "ip": "192.168.1.102",
                        "observation_only": True,
                        "record_telemetry": False,
                    }
                ]
            if payload["type"] == "start_audio_capture_preview":
                return [
                    _capture_status(
                        preview=True,
                        owned_by_requester=True,
                        **initial_override,
                    )
                ]
            if payload["type"] == "stop_audio_capture_preview":
                return [
                    {
                        "type": "audio_capture_status",
                        "running": False,
                        "preview": True,
                        "owned_by_requester": True,
                    }
                ]
            if payload["type"] == "disconnect_wing":
                return [{"type": "connection_status", "connected": False}]
            return []

        backend = _fake_backend(repo)
        websocket = _FakeWebSocket(responder)
        session = runner.ProofOwnedBackendSession(
            backend=backend,
            websocket=websocket,
            identity_result=_identity_result(),
        )
        with pytest.raises(RuntimeError, match=failure):
            await session.establish_master_route(
                wing_ip="192.168.1.102",
                audio_device_index=1,
                approval_id="operator-approved",
            )
        assert [item["type"] for item in websocket.sent] == [
            "connect_wing",
            "start_audio_capture_preview",
            "stop_audio_capture_preview",
            "disconnect_wing",
        ]
        assert session.closed is True

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("poll_override", "failure"),
    [
        ({"running": False}, "running"),
        ({"error": "capture_failed"}, "error_absent"),
        ({"source_type": "silence"}, "source_type"),
        ({"device_name": "OTHER"}, "device_name"),
        ({"device_name": "WING Rack"}, "device_name"),
        ({"device_index": True}, "device_index"),
        ({"num_channels": 47}, "num_channels"),
        ({"sample_rate": 48_000.0}, "sample_rate"),
        ({"real_input": "true"}, "real_input_boolean"),
        ({"captured_sample_count": True}, "sample_count"),
        ({"owned_by_requester": False}, "optional_owner"),
        ({"preview": False}, "optional_preview"),
    ],
)
def test_capture_status_drift_fails_before_route_and_stops_owned_preview(
    tmp_path,
    monkeypatch,
    poll_override,
    failure,
):
    async def scenario():
        repo = tmp_path / "repo"
        repo.mkdir()
        _write_open_cp05_gate(repo)
        monkeypatch.setattr(
            runner,
            "M110_CAPTURE_READY_POLL_INTERVAL_SECONDS",
            0.001,
        )

        def responder(payload):
            if payload["type"] == "connect_wing":
                return [
                    {
                        "type": "connection_status",
                        "connected": True,
                        "mode": "wing",
                        "ip": "192.168.1.102",
                        "observation_only": True,
                        "record_telemetry": False,
                    }
                ]
            if payload["type"] == "start_audio_capture_preview":
                return [
                    _capture_status(
                        preview=True,
                        owned_by_requester=True,
                    )
                ]
            if payload["type"] == "get_audio_capture_status":
                return [_capture_status(**poll_override)]
            if payload["type"] == "stop_audio_capture_preview":
                return [
                    {
                        "type": "audio_capture_status",
                        "running": False,
                        "preview": True,
                        "owned_by_requester": True,
                    }
                ]
            if payload["type"] == "disconnect_wing":
                return [{"type": "connection_status", "connected": False}]
            return []

        backend = _fake_backend(repo)
        websocket = _FakeWebSocket(responder)
        session = runner.ProofOwnedBackendSession(
            backend=backend,
            websocket=websocket,
            identity_result=_identity_result(),
        )
        with pytest.raises(RuntimeError, match=failure):
            await session.establish_master_route(
                wing_ip="192.168.1.102",
                audio_device_index=7,
                approval_id="operator-approved",
            )
        message_types = [item["type"] for item in websocket.sent]
        assert "capture_wing_master_route_proof" not in message_types
        assert message_types[-2:] == [
            "stop_audio_capture_preview",
            "disconnect_wing",
        ]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("ready_override", "failure"),
    [
        (
            {
                "real_input": True,
                "callback_age_sec": 1.01,
                "captured_sample_count": 1_024,
            },
            "invalid_callback_freshness",
        ),
        (
            {
                "real_input": True,
                "callback_age_sec": 0.01,
                "captured_sample_count": 0,
            },
            "missing_sample_progression",
        ),
    ],
)
def test_capture_ready_claim_requires_fresh_callback_and_sample_progression(
    tmp_path,
    monkeypatch,
    ready_override,
    failure,
):
    async def scenario():
        repo = tmp_path / "repo"
        repo.mkdir()
        _write_open_cp05_gate(repo)
        monkeypatch.setattr(
            runner,
            "M110_CAPTURE_READY_POLL_INTERVAL_SECONDS",
            0.001,
        )

        def responder(payload):
            if payload["type"] == "connect_wing":
                return [
                    {
                        "type": "connection_status",
                        "connected": True,
                        "mode": "wing",
                        "ip": "192.168.1.102",
                        "observation_only": True,
                        "record_telemetry": False,
                    }
                ]
            if payload["type"] == "start_audio_capture_preview":
                return [
                    _capture_status(
                        preview=True,
                        owned_by_requester=True,
                    )
                ]
            if payload["type"] == "get_audio_capture_status":
                return [_capture_status(**ready_override)]
            if payload["type"] == "stop_audio_capture_preview":
                return [
                    {
                        "type": "audio_capture_status",
                        "running": False,
                        "preview": True,
                        "owned_by_requester": True,
                    }
                ]
            if payload["type"] == "disconnect_wing":
                return [{"type": "connection_status", "connected": False}]
            return []

        backend = _fake_backend(repo)
        websocket = _FakeWebSocket(responder)
        session = runner.ProofOwnedBackendSession(
            backend=backend,
            websocket=websocket,
            identity_result=_identity_result(),
        )
        with pytest.raises(RuntimeError, match=failure):
            await session.establish_master_route(
                wing_ip="192.168.1.102",
                audio_device_index=7,
                approval_id="operator-approved",
            )
        assert "capture_wing_master_route_proof" not in [
            item["type"] for item in websocket.sent
        ]

    asyncio.run(scenario())


def test_cancellation_during_capture_poll_drains_owned_preview_teardown(
    tmp_path,
    monkeypatch,
):
    async def scenario():
        repo = tmp_path / "repo"
        repo.mkdir()
        _write_open_cp05_gate(repo)
        monkeypatch.setattr(
            runner,
            "M110_CAPTURE_READY_POLL_INTERVAL_SECONDS",
            0.001,
        )
        get_sent = asyncio.Event()

        def responder(payload):
            if payload["type"] == "connect_wing":
                return [
                    {
                        "type": "connection_status",
                        "connected": True,
                        "mode": "wing",
                        "ip": "192.168.1.102",
                        "observation_only": True,
                        "record_telemetry": False,
                    }
                ]
            if payload["type"] == "start_audio_capture_preview":
                return [
                    _capture_status(
                        preview=True,
                        owned_by_requester=True,
                    )
                ]
            if payload["type"] == "get_audio_capture_status":
                get_sent.set()
                return []
            if payload["type"] == "stop_audio_capture_preview":
                return [
                    _capture_status(),
                    {
                        "type": "audio_capture_status",
                        "running": False,
                        "preview": True,
                        "owned_by_requester": True,
                    }
                ]
            if payload["type"] == "disconnect_wing":
                return [{"type": "connection_status", "connected": False}]
            return []

        backend = _fake_backend(repo)
        websocket = _FakeWebSocket(responder)
        session = runner.ProofOwnedBackendSession(
            backend=backend,
            websocket=websocket,
            identity_result=_identity_result(),
        )
        task = asyncio.create_task(
            session.establish_master_route(
                wing_ip="192.168.1.102",
                audio_device_index=7,
                approval_id="operator-approved",
            )
        )
        await asyncio.wait_for(get_sent.wait(), timeout=0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert [item["type"] for item in websocket.sent][-2:] == [
            "stop_audio_capture_preview",
            "disconnect_wing",
        ]
        assert session.closed is True
        assert session.teardown_result["verified"] is True
        assert session.teardown_result["capture_stopped"] is True
        assert session.teardown_result["mixer_disconnected"] is True
        assert session.teardown_result["errors"] == []

    asyncio.run(scenario())


def test_capture_readiness_poll_rate_respects_backend_sustained_limit():
    assert runner.M110_CAPTURE_READY_POLL_INTERVAL_SECONDS >= 0.1
    assert runner.M110_CAPTURE_READY_TIMEOUT_SECONDS == 1.0


def test_capture_readiness_rejects_ready_terminal_after_absolute_deadline(
    tmp_path,
    monkeypatch,
):
    async def scenario():
        session = runner.ProofOwnedBackendSession(
            backend=_fake_backend(tmp_path),
            websocket=_FakeWebSocket(),
            identity_result=_identity_result(),
        )
        monkeypatch.setattr(
            runner,
            "M110_CAPTURE_READY_TIMEOUT_SECONDS",
            0.04,
        )
        monkeypatch.setattr(
            runner,
            "M110_CAPTURE_READY_POLL_INTERVAL_SECONDS",
            0.001,
        )

        async def late_ready_terminal(_expected_type, *, timeout_seconds):
            assert 0.0 < timeout_seconds < 0.04
            await asyncio.sleep(0.05)
            return _capture_status(
                real_input=True,
                callback_age_sec=0.0,
                captured_sample_count=1_024,
            )

        monkeypatch.setattr(session, "_receive_terminal", late_ready_terminal)
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="not_ready_within_1s"):
            await session._await_m110_capture_ready(
                initial_status=_capture_status(
                    preview=True,
                    owned_by_requester=True,
                ),
                audio_device_index=7,
            )
        assert time.monotonic() - started < 0.1
        assert [item["type"] for item in session.websocket.sent] == [
            "get_audio_capture_status"
        ]

    asyncio.run(scenario())


def test_capture_readiness_actual_poll_cadence_is_bounded(
    tmp_path,
    monkeypatch,
):
    async def scenario():
        get_timestamps = []

        def responder(payload):
            if payload["type"] == "get_audio_capture_status":
                get_timestamps.append(time.monotonic())
                return [_capture_status()]
            return []

        session = runner.ProofOwnedBackendSession(
            backend=_fake_backend(tmp_path),
            websocket=_FakeWebSocket(responder),
            identity_result=_identity_result(),
        )
        monkeypatch.setattr(
            runner,
            "M110_CAPTURE_READY_TIMEOUT_SECONDS",
            0.46,
        )
        with pytest.raises(RuntimeError, match="not_ready_within_1s"):
            await session._await_m110_capture_ready(
                initial_status=_capture_status(
                    preview=True,
                    owned_by_requester=True,
                ),
                audio_device_index=7,
            )
        assert 2 <= len(get_timestamps) <= 4
        assert all(
            later - earlier >= 0.09
            for earlier, later in zip(get_timestamps, get_timestamps[1:])
        )

    asyncio.run(scenario())


def test_same_session_arm_prepare_issue_apply_restores_and_erases_token(
    tmp_path,
):
    async def scenario():
        repo = tmp_path / "repo"
        repo.mkdir()
        _write_open_cp05_gate(repo)
        artifact_root = tmp_path / "route-artifacts"
        identity = _identity_result()
        proposal_id = "m110-proof-proposal-1"
        approval_id = "operator-approved"
        review_digest = "sha256:" + "b" * 64
        operation_digest = "sha256:" + "c" * 64
        capability_token = "m110cap_v1_" + "d" * 64
        idempotency_key = "m110-proof:" + "e" * 32
        route_proof = {}

        def responder(payload):
            if payload["type"] == "connect_wing":
                return [
                    {
                        "type": "connection_status",
                        "connected": True,
                        "mode": "wing",
                        "ip": "192.168.1.102",
                        "observation_only": True,
                        "record_telemetry": False,
                    }
                ]
            if payload["type"] == "start_audio_capture_preview":
                return [
                    {
                        "type": "audio_capture_status",
                        "running": True,
                        "preview": True,
                        "owned_by_requester": True,
                        "source_type": "sounddevice",
                        "real_input": True,
                        "callback_age_sec": 0.0,
                        "captured_sample_count": 12_000,
                        "device_name": "WING",
                        "device_index": 7,
                        "num_channels": 48,
                        "sample_rate": 48_000,
                    }
                ]
            if payload["type"] == "capture_wing_master_route_proof":
                result = _route_result(artifact_root, payload, identity)
                route_proof["id"] = result["proof_id"]
                return [result]
            if payload["type"] == "arm_m110_attended_control":
                assert validate_m110_arm_request(payload)[0] is True
                return [
                    {
                        "type": "m110_attended_control_result",
                        "status": "armed",
                        "success": True,
                        "reason": "m110_supervised_capability_only_armed",
                        "route_proof_id": route_proof["id"],
                        "supervised_capability_only": True,
                        "normal_operator_mode_unchanged": True,
                        "normal_live_allowlist_changed": False,
                        "live_authority_granted": False,
                        "readiness_closure_allowed": False,
                    }
                ]
            if payload["type"] == "prepare_m110_restore_only_proof":
                return [
                    {
                        "type": "m110_restore_proof_prepare_result",
                        "status": "prepared",
                        "success": True,
                        "reason": "m110_restore_proof_bundle_current",
                        "proposal_id": proposal_id,
                        "review_contract_version": 2,
                        "review_content_digest": review_digest,
                        "operation_digest": operation_digest,
                        "operation_count": 1,
                        "scope_addresses": ["/ch/1/eq/on"],
                        "token_issued": False,
                        "live_authority_granted": False,
                    }
                ]
            if payload["type"] == "issue_supervised_capability":
                assert validate_m110_capability_issue_request(payload)[0] is True
                return [
                    {
                        "type": "supervised_capability_result",
                        "status": "issued",
                        "success": True,
                        "reason": "m110_capability_issued",
                        "capability_kind": "restore_only_proof",
                        "proposal_id": proposal_id,
                        "review_contract_version": 2,
                        "review_content_digest": review_digest,
                        "operation_digest": operation_digest,
                        "approval_id": approval_id,
                        "idempotency_key": idempotency_key,
                        "latest_start_epoch": time.time() + 20.0,
                        "supervised_capability_token": capability_token,
                        "single_use": True,
                        "stores_plaintext_token": False,
                        "live_authority_granted": False,
                    }
                ]
            if payload["type"] == "apply_operator_proposal":
                assert payload["supervised_capability_token"] == capability_token
                assert set(payload) == {
                    "type",
                    "proposal_id",
                    "approved",
                    "approval_id",
                    "idempotency_key",
                    "expected_review_contract_version",
                    "expected_review_content_digest",
                    "expected_operation_digest",
                    "supervised_capability_token",
                }
                return [
                    {
                        "type": "operator_proposal_apply_result",
                        "contract_version": 2,
                        "status": "proof_passed",
                        "success": True,
                        "reason": None,
                        "proposal_id": proposal_id,
                        "proposal_status": "proof_passed",
                        "proposal": {
                            "id": proposal_id,
                            "review_contract_version": 2,
                            "review_content_digest": review_digest,
                            "status": "proof_passed",
                            "result_digest": "sha256:" + "f" * 64,
                            "result_detail_available": True,
                            "result_evidence_pinned": False,
                        },
                        "apply_status": "proof_passed",
                        "accepted": False,
                        "owner_review_required": True,
                        "live_authority_granted": False,
                        "readiness_closure_allowed": False,
                        "proof_run": True,
                        "restore_after_verify": True,
                        "capability_kind": "restore_only_proof",
                        "operation_digest": operation_digest,
                        "proof_target_verified": True,
                        "no_console_send_proven": False,
                        "physical_verified": True,
                        "physical_state_restored": True,
                        "terminal_after_console_send": True,
                        "write_state_unknown": False,
                        "rollback_attempted": True,
                        "rollback_succeeded": True,
                        "rollback_status": "verified",
                        "rollback_state_unknown": False,
                        "manual_review_required": False,
                        "emergency_stop_activated": False,
                        "emergency_stop_required": False,
                        "wal_terminal": "proof_restored",
                        "audio_recheck_passed": True,
                        "result_digest": "sha256:" + "f" * 64,
                        "result_detail_available": True,
                        "result_evidence_pinned": False,
                    }
                ]
            if payload["type"] == "stop_audio_capture_preview":
                return [
                    {
                        "type": "audio_capture_status",
                        "running": False,
                        "preview": True,
                        "owned_by_requester": True,
                    }
                ]
            if payload["type"] == "disconnect_wing":
                return [{"type": "connection_status", "connected": False}]
            return []

        backend = _fake_backend(repo)
        websocket = _FakeWebSocket(responder)
        session = runner.ProofOwnedBackendSession(
            backend=backend,
            websocket=websocket,
            identity_result=identity,
        )
        await session.establish_master_route(
            wing_ip="192.168.1.102",
            audio_device_index=7,
            approval_id=approval_id,
            request_id="m110-route-test",
            request_nonce="1" * 32,
            route_artifact_root=artifact_root,
        )
        arm = await session.arm_m110_attended_control(
            approval_id=approval_id,
        )
        proof = await session.prove_pending_restore_only_proposal(
            proposal_id=proposal_id,
            approval_id=approval_id,
        )

        assert arm["verified"] is True
        assert proof["verified"] is True
        assert proof["accepted"] is False
        assert proof["token_retained"] is False
        assert session.closed is False
        assert session.control_armed is True
        durable = json.dumps(proof, sort_keys=True) + repr(session)
        assert capability_token not in durable
        sent_types = [item["type"] for item in websocket.sent]
        assert sent_types == [
            "connect_wing",
            "start_audio_capture_preview",
            "capture_wing_master_route_proof",
            "arm_m110_attended_control",
            "prepare_m110_restore_only_proof",
            "issue_supervised_capability",
            "apply_operator_proposal",
        ]
        assert sum(
            capability_token in json.dumps(item)
            for item in websocket.sent
        ) == 1

        teardown = await session.close()
        assert teardown["verified"] is True
        assert session.control_armed is False
        assert backend.process.poll() is not None

    asyncio.run(scenario())


def test_same_session_fixed_reset_proof_receipt_commit_without_client_plan(
    tmp_path,
):
    async def scenario():
        repo = tmp_path / "repo"
        repo.mkdir()
        _write_open_cp05_gate(repo)
        artifact_root = tmp_path / "route-artifacts"
        identity = _identity_result()
        approval_id = "operator-approved"
        proof_created = _fixed_reset_created("proof")
        commit_created = _fixed_reset_created("commit")
        proof_token = "m110cap_v1_" + "4" * 64
        commit_token = "m110cap_v1_" + "5" * 64
        route_proof = {}

        def created_terminal(created):
            return {
                "type": "m110_maintenance_reset_proposal_result",
                "status": "created",
                "success": True,
                "reason": "m110_maintenance_reset_proposal_created",
                "proposal_id": created["proposal_id"],
                "review_contract_version": 2,
                "review_content_digest": created[
                    "review_content_digest"
                ],
                "required_capability_kind": created[
                    "required_capability_kind"
                ],
                "bundle_kind": created["bundle_kind"],
                "operation_digest": created["operation_digest"],
                "review_preimage_sha256": created[
                    "review_preimage_sha256"
                ],
                "operation_count": len(runner.M110_RESET_ADDRESSES),
                "scope_addresses": list(runner.M110_RESET_ADDRESSES),
                "review_operations": list(created["review_operations"]),
                "matching_restored_proof_receipt": (
                    created["role"] == "commit"
                ),
                "token_issued": False,
                "live_authority_granted": False,
                "readiness_closure_allowed": False,
            }

        def prepared_terminal(created):
            return {
                "type": "m110_restore_proof_prepare_result",
                "status": "prepared",
                "success": True,
                "reason": "m110_fixed_reset_bundle_current",
                "proposal_id": created["proposal_id"],
                "review_contract_version": 2,
                "review_content_digest": created[
                    "review_content_digest"
                ],
                "operation_digest": created["operation_digest"],
                "bundle_kind": created["bundle_kind"],
                "review_preimage_sha256": created[
                    "review_preimage_sha256"
                ],
                "required_capability_kind": created[
                    "required_capability_kind"
                ],
                "operation_count": len(runner.M110_RESET_ADDRESSES),
                "scope_addresses": list(runner.M110_RESET_ADDRESSES),
                "token_issued": False,
                "live_authority_granted": False,
            }

        def responder(payload):
            message_type = payload["type"]
            if message_type == "connect_wing":
                return [
                    {
                        "type": "connection_status",
                        "connected": True,
                        "mode": "wing",
                        "ip": "192.168.1.102",
                        "observation_only": True,
                        "record_telemetry": False,
                    }
                ]
            if message_type == "start_audio_capture_preview":
                return [
                    {
                        "type": "audio_capture_status",
                        "running": True,
                        "preview": True,
                        "owned_by_requester": True,
                        "source_type": "sounddevice",
                        "real_input": True,
                        "callback_age_sec": 0.0,
                        "captured_sample_count": 12_000,
                        "device_name": "WING",
                        "device_index": 7,
                        "num_channels": 48,
                        "sample_rate": 48_000,
                    }
                ]
            if message_type == "capture_wing_master_route_proof":
                result = _route_result(artifact_root, payload, identity)
                route_proof["id"] = result["proof_id"]
                return [result]
            if message_type == "arm_m110_attended_control":
                return [
                    {
                        "type": "m110_attended_control_result",
                        "status": "armed",
                        "success": True,
                        "reason": "m110_supervised_capability_only_armed",
                        "route_proof_id": route_proof["id"],
                        "supervised_capability_only": True,
                        "normal_operator_mode_unchanged": True,
                        "normal_live_allowlist_changed": False,
                        "live_authority_granted": False,
                        "readiness_closure_allowed": False,
                    }
                ]
            if message_type.startswith(
                "create_m110_maintenance_reset_"
            ):
                assert set(payload) == {
                    "type",
                    "contract_version",
                    "bundle_selector",
                }
                created = (
                    proof_created
                    if "_proof_" in message_type
                    else commit_created
                )
                return [created_terminal(created)]
            if message_type == "prepare_m110_restore_only_proof":
                assert set(payload) == {
                    "type",
                    "contract_version",
                    "proposal_id",
                    "bundle_selector",
                }
                created = (
                    proof_created
                    if payload["proposal_id"] == proof_created["proposal_id"]
                    else commit_created
                )
                return [prepared_terminal(created)]
            if message_type == "issue_supervised_capability":
                assert validate_m110_capability_issue_request(payload)[0] is True
                proof_run = payload["capability_kind"] == "restore_only_proof"
                created = proof_created if proof_run else commit_created
                return [
                    {
                        "type": "supervised_capability_result",
                        "status": "issued",
                        "success": True,
                        "reason": "m110_capability_issued",
                        "capability_kind": payload["capability_kind"],
                        "proposal_id": created["proposal_id"],
                        "review_contract_version": 2,
                        "review_content_digest": created[
                            "review_content_digest"
                        ],
                        "operation_digest": created["operation_digest"],
                        "approval_id": approval_id,
                        "idempotency_key": (
                            "m110-proof:" + "6" * 32
                            if proof_run
                            else "m110-reset-commit:" + "7" * 32
                        ),
                        "latest_start_epoch": time.time() + 20.0,
                        "supervised_capability_token": (
                            proof_token if proof_run else commit_token
                        ),
                        "single_use": True,
                        "stores_plaintext_token": False,
                        "live_authority_granted": False,
                    }
                ]
            if message_type == "apply_operator_proposal":
                assert set(payload) == {
                    "type",
                    "proposal_id",
                    "approved",
                    "approval_id",
                    "idempotency_key",
                    "expected_review_contract_version",
                    "expected_review_content_digest",
                    "expected_operation_digest",
                    "supervised_capability_token",
                }
                proof_run = payload["proposal_id"] == proof_created[
                    "proposal_id"
                ]
                created = proof_created if proof_run else commit_created
                assert payload["supervised_capability_token"] == (
                    proof_token if proof_run else commit_token
                )
                return [
                    _valid_fixed_reset_terminal(
                        created,
                        (
                            "restore_only_proof"
                            if proof_run
                            else "maintenance_dsp_reset_commit_once"
                        ),
                    )
                ]
            if message_type == "stop_audio_capture_preview":
                return [
                    {
                        "type": "audio_capture_status",
                        "running": False,
                        "preview": True,
                        "owned_by_requester": True,
                    }
                ]
            if message_type == "disconnect_wing":
                return [{"type": "connection_status", "connected": False}]
            return []

        backend = _fake_backend(repo)
        websocket = _FakeWebSocket(responder)
        session = runner.ProofOwnedBackendSession(
            backend=backend,
            websocket=websocket,
            identity_result=identity,
        )
        await session.establish_master_route(
            wing_ip="192.168.1.102",
            audio_device_index=7,
            approval_id=approval_id,
            request_id="m110-route-test",
            request_nonce="1" * 32,
            route_artifact_root=artifact_root,
        )
        await session.arm_m110_attended_control(approval_id=approval_id)
        operator_input, terminal_emitter, review_events = (
            _operator_review_callbacks()
        )
        reset = await session.execute_fixed_m110_maintenance_reset(
            approval_id=approval_id,
            operator_input=operator_input,
            terminal_emitter=terminal_emitter,
        )

        assert reset["verified"] is True
        assert reset["reset_applied"] is True
        assert reset["proof"]["terminal"]["wal_terminal"] == "proof_restored"
        assert reset["commit"]["terminal"]["wal_terminal"] == "committed"
        assert reset["token_retained"] is False
        durable = json.dumps(reset, sort_keys=True) + repr(session)
        assert proof_token not in durable
        assert commit_token not in durable
        assert sum(
            event.get("event") == "m110_operator_bundle_review_required"
            for event in review_events
        ) == 2
        sent_types = [item["type"] for item in websocket.sent]
        assert sent_types == [
            "connect_wing",
            "start_audio_capture_preview",
            "capture_wing_master_route_proof",
            "arm_m110_attended_control",
            "create_m110_maintenance_reset_proof_proposal",
            "prepare_m110_restore_only_proof",
            "issue_supervised_capability",
            "apply_operator_proposal",
            "create_m110_maintenance_reset_commit_proposal",
            "prepare_m110_restore_only_proof",
            "issue_supervised_capability",
            "apply_operator_proposal",
        ]
        assert sum(
            proof_token in json.dumps(item) for item in websocket.sent
        ) == 1
        assert sum(
            commit_token in json.dumps(item) for item in websocket.sent
        ) == 1
        assert all(
            "transaction_plan" not in item
            and "supervised_transaction_plan" not in item
            for item in websocket.sent
        )

        teardown = await session.close()
        assert teardown["verified"] is True

    asyncio.run(scenario())


def test_fixed_reset_proof_failure_closes_session_before_commit_creation(
    tmp_path,
):
    async def scenario():
        proof_created = _fixed_reset_created("proof")
        token = "m110cap_v1_" + "8" * 64

        def responder(payload):
            message_type = payload["type"]
            if message_type == "create_m110_maintenance_reset_proof_proposal":
                return [
                    {
                        "type": "m110_maintenance_reset_proposal_result",
                        "status": "created",
                        "success": True,
                        "reason": "m110_maintenance_reset_proposal_created",
                        "proposal_id": proof_created["proposal_id"],
                        "review_contract_version": 2,
                        "review_content_digest": proof_created[
                            "review_content_digest"
                        ],
                        "required_capability_kind": "restore_only_proof",
                        "bundle_kind": "maintenance_dsp_reset_bundle",
                        "operation_digest": proof_created[
                            "operation_digest"
                        ],
                        "review_preimage_sha256": proof_created[
                            "review_preimage_sha256"
                        ],
                        "operation_count": len(runner.M110_RESET_ADDRESSES),
                        "scope_addresses": list(runner.M110_RESET_ADDRESSES),
                        "review_operations": list(
                            proof_created["review_operations"]
                        ),
                        "matching_restored_proof_receipt": False,
                        "token_issued": False,
                        "live_authority_granted": False,
                        "readiness_closure_allowed": False,
                    }
                ]
            if message_type == "prepare_m110_restore_only_proof":
                return [
                    {
                        "type": "m110_restore_proof_prepare_result",
                        "status": "prepared",
                        "success": True,
                        "reason": "m110_fixed_reset_bundle_current",
                        "proposal_id": proof_created["proposal_id"],
                        "review_contract_version": 2,
                        "review_content_digest": proof_created[
                            "review_content_digest"
                        ],
                        "operation_digest": proof_created[
                            "operation_digest"
                        ],
                        "bundle_kind": "maintenance_dsp_reset_bundle",
                        "review_preimage_sha256": proof_created[
                            "review_preimage_sha256"
                        ],
                        "required_capability_kind": "restore_only_proof",
                        "operation_count": len(runner.M110_RESET_ADDRESSES),
                        "scope_addresses": list(runner.M110_RESET_ADDRESSES),
                        "token_issued": False,
                        "live_authority_granted": False,
                    }
                ]
            if message_type == "issue_supervised_capability":
                return [
                    {
                        "type": "supervised_capability_result",
                        "status": "issued",
                        "success": True,
                        "reason": "m110_capability_issued",
                        "capability_kind": "restore_only_proof",
                        "proposal_id": proof_created["proposal_id"],
                        "review_contract_version": 2,
                        "review_content_digest": proof_created[
                            "review_content_digest"
                        ],
                        "operation_digest": proof_created[
                            "operation_digest"
                        ],
                        "approval_id": "operator-approved",
                        "idempotency_key": "m110-proof:" + "9" * 32,
                        "latest_start_epoch": time.time() + 20.0,
                        "supervised_capability_token": token,
                        "single_use": True,
                        "stores_plaintext_token": False,
                        "live_authority_granted": False,
                    }
                ]
            if message_type == "apply_operator_proposal":
                terminal = _valid_fixed_reset_terminal(
                    proof_created,
                    "restore_only_proof",
                )
                terminal["physical_state_restored"] = False
                return [terminal]
            return []

        backend = _fake_backend(tmp_path)
        websocket = _FakeWebSocket(responder)
        session = runner.ProofOwnedBackendSession(
            backend=backend,
            websocket=websocket,
            identity_result=_identity_result(),
            route_established=True,
            route_result={"proof_id": "route-proof"},
            control_armed=True,
            control_approval_id="operator-approved",
        )
        operator_input, terminal_emitter, _review_events = (
            _operator_review_callbacks()
        )
        with pytest.raises(RuntimeError, match="fixed_reset_apply_terminal"):
            await session.execute_fixed_m110_maintenance_reset(
                approval_id="operator-approved",
                operator_input=operator_input,
                terminal_emitter=terminal_emitter,
            )

        assert session.closed is True
        assert websocket.closed is True
        sent_types = [item["type"] for item in websocket.sent]
        assert "create_m110_maintenance_reset_commit_proposal" not in sent_types
        assert sent_types == [
            "create_m110_maintenance_reset_proof_proposal",
            "prepare_m110_restore_only_proof",
            "issue_supervised_capability",
            "apply_operator_proposal",
        ]

    asyncio.run(scenario())


def test_route_session_invalid_approval_sends_no_console_or_capture_frame(
    tmp_path,
):
    async def scenario():
        repo = tmp_path / "repo"
        repo.mkdir()
        _write_open_cp05_gate(repo)
        backend = _fake_backend(repo)
        websocket = _FakeWebSocket()
        session = runner.ProofOwnedBackendSession(
            backend=backend,
            websocket=websocket,
            identity_result=_identity_result(),
        )

        with pytest.raises(ValueError, match="authorization_invalid"):
            await session.establish_master_route(
                wing_ip="192.168.1.102",
                audio_device_index=7,
                approval_id="",
                request_id="m110-route-test",
                request_nonce="d" * 32,
                route_artifact_root=tmp_path / "route-artifacts",
            )

        assert websocket.sent == []
        assert session.closed is True
        assert backend.process.poll() is not None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "169.254.1.2",
        "192.168.1.0",
        "192.168.1.255",
        "255.255.255.255",
        "8.8.8.8",
    ],
)
def test_route_session_rejects_non_rfc1918_unicast_before_send(
    tmp_path,
    address,
):
    async def scenario():
        repo = tmp_path / "repo"
        repo.mkdir()
        _write_open_cp05_gate(repo)
        backend = _fake_backend(repo)
        websocket = _FakeWebSocket()
        session = runner.ProofOwnedBackendSession(
            backend=backend,
            websocket=websocket,
            identity_result=_identity_result(),
        )
        with pytest.raises(ValueError, match="m110_wing_ip"):
            await session.establish_master_route(
                wing_ip=address,
                audio_device_index=7,
                approval_id="operator-approved",
            )
        assert websocket.sent == []
        assert backend.process.poll() is not None

    asyncio.run(scenario())


def test_failed_capture_terminal_closes_session_and_disconnects_wing(tmp_path):
    async def scenario():
        repo = tmp_path / "repo"
        repo.mkdir()
        _write_open_cp05_gate(repo)

        def responder(payload):
            if payload["type"] == "connect_wing":
                return [
                    {
                        "type": "connection_status",
                        "connected": True,
                        "mode": "wing",
                        "ip": "192.168.1.102",
                        "observation_only": True,
                        "record_telemetry": False,
                    }
                ]
            if payload["type"] == "start_audio_capture_preview":
                return [
                    {
                        "type": "audio_capture_status",
                        "running": False,
                        "preview": True,
                        "owned_by_requester": False,
                        "error": "capture_failed",
                    }
                ]
            if payload["type"] == "disconnect_wing":
                return [{"type": "connection_status", "connected": False}]
            return []

        backend = _fake_backend(repo)
        websocket = _FakeWebSocket(responder)
        session = runner.ProofOwnedBackendSession(
            backend=backend,
            websocket=websocket,
            identity_result=_identity_result(),
        )
        with pytest.raises(RuntimeError, match="capture_start_failed"):
            await session.establish_master_route(
                wing_ip="192.168.1.102",
                audio_device_index=7,
                approval_id="operator-approved",
            )
        assert session.closed is True
        assert backend.process.poll() is not None
        assert [item["type"] for item in websocket.sent] == [
            "connect_wing",
            "start_audio_capture_preview",
            "disconnect_wing",
        ]

    asyncio.run(scenario())


def test_close_drains_teardown_before_propagating_cancellation(tmp_path):
    async def scenario():
        repo = tmp_path / "repo"
        repo.mkdir()
        backend = _fake_backend(repo)
        websocket = _BlockingCloseWebSocket()
        session = runner.ProofOwnedBackendSession(
            backend=backend,
            websocket=websocket,
            identity_result=_identity_result(),
        )
        close_task = asyncio.create_task(session.close())
        await websocket.close_started.wait()
        close_task.cancel()
        await asyncio.sleep(0)
        websocket.allow_close.set()
        with pytest.raises(asyncio.CancelledError):
            await close_task
        assert session.closed is True
        assert websocket.closed is True
        assert backend.process.poll() is not None
        assert set(backend.secret) == {0}

    asyncio.run(scenario())


def test_route_result_semantic_verifier_accepts_valid_artifact(tmp_path):
    identity = _identity_result()
    request = runner.build_route_capture_request(
        request_id="m110-route-valid",
        approval_id="operator-approved",
        request_nonce="d" * 32,
        secret=b"s" * 32,
    )
    artifact_root = tmp_path / "route-artifacts"
    result = _route_result(artifact_root, request, identity)

    verified = runner.ProofOwnedBackendSession._validate_route_result(
        result=result,
        request=request,
        audio_device_index=7,
        route_artifact_root=artifact_root,
        identity_result=identity,
    )

    assert verified["verified"] is True
    assert verified["request_binding_verified"] is True
    assert verified["hardware_binding_verified"] is True


@pytest.mark.parametrize(
    "mutation_name",
    [
        "route_readback",
        "capture_count",
        "capture_peak",
        "hardware_vendor",
        "hardware_output",
        "hardware_output_missing",
        "hardware_input_float",
        "hardware_output_float",
        "hardware_sample_rate_float",
        "hardware_sha",
        "runtime_binding",
        "firmware_family",
        "ttl",
    ],
)
def test_route_result_semantic_verifier_rejects_hash_valid_false_evidence(
    tmp_path,
    mutation_name,
):
    identity = _identity_result()
    request = runner.build_route_capture_request(
        request_id="m110-route-mutated",
        approval_id="operator-approved",
        request_nonce="d" * 32,
        secret=b"s" * 32,
    )

    def mutate(evidence):
        if mutation_name == "route_readback":
            evidence["route"]["pass_2"]["readbacks"][
                "/io/out/USB/48/in"
            ]["value"] = 0
        elif mutation_name == "capture_count":
            evidence["capture"]["captured_sample_count_snapshot"] = 12_999
        elif mutation_name == "capture_peak":
            evidence["capture"]["conservative_true_peak_dbtp"] = -6.0
        elif mutation_name == "hardware_vendor":
            hardware = evidence["capture"]["hardware_identity"]
            hardware["usb"]["vendor_id"] = 999
            binding = {
                key: hardware[key]
                for key in (
                    "schema",
                    "device_name",
                    "device_index",
                    "console_serial",
                    "coreaudio",
                    "usb",
                    "serial_binding",
                )
            }
            hardware["binding_sha256"] = hashlib.sha256(
                runner._canonical_json_bytes(binding)
            ).hexdigest()
        elif mutation_name in {
            "hardware_output",
            "hardware_output_missing",
            "hardware_input_float",
            "hardware_output_float",
            "hardware_sample_rate_float",
        }:
            hardware = evidence["capture"]["hardware_identity"]
            field, value = {
                "hardware_output": ("output_channels", 0),
                "hardware_output_missing": ("output_channels", None),
                "hardware_input_float": ("input_channels", 48.0),
                "hardware_output_float": ("output_channels", 48.0),
                "hardware_sample_rate_float": ("sample_rate_hz", 48_000.0),
            }[mutation_name]
            if mutation_name == "hardware_output_missing":
                hardware["coreaudio"].pop(field)
            else:
                hardware["coreaudio"][field] = value
            binding = {
                key: hardware[key]
                for key in (
                    "schema",
                    "device_name",
                    "device_index",
                    "console_serial",
                    "coreaudio",
                    "usb",
                    "serial_binding",
                )
            }
            hardware["binding_sha256"] = hashlib.sha256(
                runner._canonical_json_bytes(binding)
            ).hexdigest()
        elif mutation_name == "hardware_sha":
            evidence["capture"]["hardware_identity"]["binding_sha256"] = (
                "0" * 64
            )
        elif mutation_name == "runtime_binding":
            evidence["runtime"]["manifest_sha256"] = "0" * 64
        elif mutation_name == "firmware_family":
            evidence["console"]["firmware"] = "3.0.4"
            for pass_name in ("pass_1", "pass_2"):
                route_pass = evidence["route"][pass_name]
                route_pass["values"]["/$syscfg/$firmware"] = "3.0.4"
                route_pass["readbacks"]["/$syscfg/$firmware"][
                    "value"
                ] = "3.0.4"

    artifact_root = tmp_path / f"route-{mutation_name}"
    result = _route_result(
        artifact_root,
        request,
        identity,
        mutation=mutate,
        ttl_seconds=1_801 if mutation_name == "ttl" else 1_800,
    )

    with pytest.raises(RuntimeError):
        runner.ProofOwnedBackendSession._validate_route_result(
            result=result,
            request=request,
            audio_device_index=7,
            route_artifact_root=artifact_root,
            identity_result=identity,
        )


def test_cp05_gate_is_required_before_route_probe_can_spawn_backend(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    evidence = {
        "schema": "automixer.m110_cp05_software_validation.v1",
        "status": "pending",
        "source_binding_sha256": None,
        "checks": {
            "focused_tests": False,
            "frontend_build_and_tests": False,
            "full_backend_suite": False,
            "independent_review": False,
            "no_wing_release_check": False,
            "released_development_lease": False,
            "static_scan": False,
        },
    }
    evidence_path = repo / "Docs" / "reports" / "M110_CP05_SOFTWARE_VALIDATION.json"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_bytes(runner._canonical_json_bytes(evidence) + b"\n")
    gate_path = repo / "ops" / "m110" / "CP05_HARDWARE_GATE.json"
    gate_path.parent.mkdir(parents=True)
    gate_path.write_bytes(
        runner._canonical_json_bytes(
            {
                "schema": "automixer.m110_cp05_hardware_gate.v1",
                "checkpoint": "M110-CP01",
                "hardware_enabled": False,
                "software_evidence_sha256": None,
            }
        )
        + b"\n"
    )
    called = False

    def forbidden_probe(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("backend must not spawn before CP05")

    monkeypatch.setattr(
        runner,
        "run_proof_owned_master_route_probe_sync",
        forbidden_probe,
    )
    artifact_root = tmp_path / "artifacts"
    exit_code = runner.main(
        [
            "--repo-root",
            str(repo),
            "--verify-route-only",
            "--approval-id",
            "operator-approved",
            "--confirmation",
            runner.MASTER_CAPTURE_ROUTE_REQUEST_CONFIRMATION,
            "--wing-ip",
            "192.168.1.102",
            "--audio-device-index",
            "7",
            "--artifact-root",
            str(artifact_root),
        ]
    )
    summary = json.loads((artifact_root / "plan-summary.json").read_text())

    assert exit_code == 2
    assert called is False
    assert summary["reason"] == "route_execution_blocked"
    assert summary["contacts_backend"] is False
    assert summary["contacts_wing"] is False
    assert any(
        "m110_cp05_hardware_gate_closed" in failure
        for failure in summary["authorization_failures"]
    )


def test_cp05_gate_rejects_later_clean_commit_outside_runtime_tuple(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_open_cp05_gate(repo)
    assert runner.assert_m110_cp05_hardware_gate(repo)["verified"] is True
    tracked = repo / "tracked-source.txt"
    tracked.write_text("later behavior change\n", encoding="utf-8")
    _git(repo, "add", tracked.relative_to(repo).as_posix())
    _git(repo, "commit", "-q", "-m", "unvalidated later change")

    with pytest.raises(RuntimeError, match="source_binding_mismatch"):
        runner.assert_m110_cp05_hardware_gate(repo)


def test_summary_writer_rejects_stale_symlink_without_touching_sentinel(
    tmp_path,
):
    tmp_path.chmod(0o700)
    sentinel = tmp_path.parent / "sentinel.json"
    sentinel.write_text("UNCHANGED", encoding="utf-8")
    summary_path = tmp_path / "plan-summary.json"
    summary_path.symlink_to(sentinel)

    exit_code = runner.main(["--artifact-root", str(tmp_path)])

    assert exit_code == 2
    assert sentinel.read_text(encoding="utf-8") == "UNCHANGED"
    assert summary_path.is_symlink()
