from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import json
import os
import py_compile
import secrets
import subprocess
import sys
import time
from pathlib import Path

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
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.returncode


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

    assert validate_m110_arm_request(arm)[0] is True
    assert validate_m110_capability_issue_request(issue)[0] is True
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


def test_execute_blocks_until_remaining_capability_checkpoints_exist(tmp_path):
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
    summary = json.loads((tmp_path / "plan-summary.json").read_text())

    assert exit_code == 2
    assert summary["status"] == "blocked"
    assert "m110_remaining_capability_checkpoints_not_committed" in summary[
        "authorization_failures"
    ]
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
    rebuilt = runner.build_runner_prestart_manifest(
        root,
        launch_nonce="b" * 32,
    )
    assert rebuilt["launch_nonce"] == "b" * 32

    backend.close()

    assert backend.process.poll() is not None
    assert set(backend.secret) == {0}
    assert set(backend.token) == {0}


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
