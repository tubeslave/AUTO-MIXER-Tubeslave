from __future__ import annotations

import json
import os
import copy
import pickle
import secrets
import subprocess
import threading
from dataclasses import replace
from pathlib import Path

import pytest
import supervised_proof_capability as capability_module

from supervised_proof_capability import (
    M110_AUTHORITY_FD_ENV,
    M110_RUNTIME_CRITICAL_PATHS,
    M110CapabilityBinding,
    M110RestoredProofReceipt,
    RestoreOnlyProofGrant,
    SupervisedCapabilityStore,
    build_m110_prestart_manifest,
    consume_m110_attended_launch_authority,
    new_m110_restored_proof_receipt,
    sign_m110_request,
    validate_m110_arm_request,
    validate_m110_capability_apply_request,
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


def _committed_runtime_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    for relative_path in M110_RUNTIME_CRITICAL_PATHS:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative_path}\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "m110@example.invalid")
    _git(root, "config", "user.name", "M110 Test")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "fixture")
    return root


def _consume_packet(
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
    packet: bytes,
):
    read_fd, write_fd = os.pipe()
    os.write(write_fd, packet)
    os.close(write_fd)
    monkeypatch.setenv(M110_AUTHORITY_FD_ENV, str(read_fd))
    return consume_m110_attended_launch_authority(repo_root)


def test_inherited_fd_manifest_and_hmac_are_process_private_and_single_use(
    tmp_path,
    monkeypatch,
):
    root = _committed_runtime_repo(tmp_path)
    launch_nonce = "a" * 32
    secret = secrets.token_bytes(32)
    manifest = build_m110_prestart_manifest(root, launch_nonce=launch_nonce)
    raw = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    binding, authority = _consume_packet(
        monkeypatch,
        root,
        len(raw).to_bytes(4, "big") + raw + secret,
    )

    assert binding["mode"] == "m110_attended_runner"
    assert binding["tracked_tree_clean"] is True
    assert binding["m110_attended_eligible"] is True
    assert secret.hex() not in json.dumps(binding)
    assert "authority_secret_sha256" in binding
    assert M110_AUTHORITY_FD_ENV not in os.environ
    request = sign_m110_request(
        {
            "type": "capture_wing_master_route_proof",
            "request_nonce": "b" * 32,
        },
        secret=secret,
    )
    assert authority is not None
    assert authority.verify_and_tombstone_request(request) == (
        True,
        "m110_authority_request_verified",
    )
    assert authority.verify_and_tombstone_request(request) == (
        False,
        "m110_request_nonce_replayed",
    )


def test_unrelated_untracked_report_does_not_become_source_authority(
    tmp_path,
):
    root = _committed_runtime_repo(tmp_path)
    report = root / "reports" / "generated.json"
    report.parent.mkdir()
    report.write_text("{}", encoding="utf-8")

    manifest = build_m110_prestart_manifest(root, launch_nonce="c" * 32)

    assert manifest["launch_nonce"] == "c" * 32
    assert report.relative_to(root).as_posix() not in {
        row["path"] for row in manifest["files"]
    }


def test_untracked_python_shadow_file_blocks_manifest(tmp_path):
    root = _committed_runtime_repo(tmp_path)
    (root / "backend" / "numpy.py").write_text("raise SystemExit\n")

    with pytest.raises(RuntimeError, match="unsafe untracked path"):
        build_m110_prestart_manifest(root, launch_nonce="9" * 32)


@pytest.mark.parametrize(
    "relative_path",
    [
        "backend/json.pyc",
        "backend/wing_client.so",
        "backend/handlers/connection_handlers.so",
        "backend/osc/enhanced_osc_client.so",
        "backend/sitecustomize.py",
        "tools/json.pyc",
    ],
)
def test_ignored_import_shadow_artifact_blocks_manifest(tmp_path, relative_path):
    root = _committed_runtime_repo(tmp_path)
    (root / ".gitignore").write_text(
        "*.py\n*.pyw\n*.py[cod]\n*.so\n",
        encoding="utf-8",
    )
    _git(root, "add", ".gitignore")
    _git(root, "commit", "-q", "-m", "ignore runtime artifacts")
    shadow = root / relative_path
    shadow.write_bytes(b"ignored executable shadow")

    with pytest.raises(RuntimeError, match="ignored import-shadow artifact"):
        build_m110_prestart_manifest(root, launch_nonce="7" * 32)


def test_ignored_non_import_build_cache_does_not_block_manifest(tmp_path):
    root = _committed_runtime_repo(tmp_path)
    (root / ".gitignore").write_text("backend/build/\n", encoding="utf-8")
    _git(root, "add", ".gitignore")
    _git(root, "commit", "-q", "-m", "ignore build output")
    cache = root / "backend" / "build" / "server" / "localpycs" / "struct.pyc"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"build output")

    manifest = build_m110_prestart_manifest(root, launch_nonce="4" * 32)

    assert manifest["launch_nonce"] == "4" * 32


def test_untracked_report_symlink_blocks_manifest(tmp_path):
    root = _committed_runtime_repo(tmp_path)
    target = root / "target.json"
    target.write_text("{}")
    report = root / "reports" / "generated.json"
    report.parent.mkdir()
    report.symlink_to(target)

    with pytest.raises(RuntimeError, match="unsafe untracked path|non-symlink"):
        build_m110_prestart_manifest(root, launch_nonce="8" * 32)


def test_dirty_tracked_source_fails_before_authority_issuance(tmp_path):
    root = _committed_runtime_repo(tmp_path)
    path = root / M110_RUNTIME_CRITICAL_PATHS[0]
    path.write_text("dirty\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="tracked source tree is dirty"):
        build_m110_prestart_manifest(root, launch_nonce="d" * 32)


def test_manifest_trailing_bytes_fail_closed(tmp_path, monkeypatch):
    root = _committed_runtime_repo(tmp_path)
    secret = secrets.token_bytes(32)
    manifest = build_m110_prestart_manifest(root, launch_nonce="e" * 32)
    raw = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    with pytest.raises(RuntimeError, match="trailing bytes"):
        _consume_packet(
            monkeypatch,
            root,
            len(raw).to_bytes(4, "big") + raw + secret + b"x",
        )


def test_regular_file_cannot_replace_one_shot_authority_pipe(
    tmp_path,
    monkeypatch,
):
    root = _committed_runtime_repo(tmp_path)
    packet_file = tmp_path / "authority.bin"
    packet_file.write_bytes(b"not-a-pipe")
    fd = os.open(packet_file, os.O_RDONLY)
    monkeypatch.setenv(M110_AUTHORITY_FD_ENV, str(fd))

    with pytest.raises(RuntimeError, match="one-shot pipe"):
        consume_m110_attended_launch_authority(root)


def test_self_declared_manifest_bytes_cannot_hide_source_drift(
    tmp_path,
    monkeypatch,
):
    root = _committed_runtime_repo(tmp_path)
    secret = secrets.token_bytes(32)
    manifest = build_m110_prestart_manifest(root, launch_nonce="f" * 32)
    raw = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    (root / M110_RUNTIME_CRITICAL_PATHS[-1]).write_text(
        "changed after manifest\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="tracked source tree is dirty"):
        _consume_packet(
            monkeypatch,
            root,
            len(raw).to_bytes(4, "big") + raw + secret,
        )


def test_consumed_authority_permanently_closes_on_real_source_drift(
    tmp_path,
    monkeypatch,
):
    root = _committed_runtime_repo(tmp_path)
    secret = secrets.token_bytes(32)
    manifest = build_m110_prestart_manifest(root, launch_nonce="6" * 32)
    raw = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    _, authority = _consume_packet(
        monkeypatch,
        root,
        len(raw).to_bytes(4, "big") + raw + secret,
    )
    assert authority is not None
    assert authority.source_snapshot_status() == (
        True,
        "m110_live_source_snapshot_current",
    )

    (root / M110_RUNTIME_CRITICAL_PATHS[0]).write_text(
        "changed after launch\n",
        encoding="utf-8",
    )

    assert authority.source_snapshot_status() == (
        False,
        "m110_live_source_snapshot_drift",
    )
    request = sign_m110_request(
        {"type": "unit", "request_nonce": "5" * 32},
        secret=secret,
    )
    assert authority.verify_and_tombstone_request(request) == (
        False,
        "m110_launch_authority_closed",
    )


def test_nonce_capacity_fails_closed_without_reenabling_old_nonce(monkeypatch):
    monkeypatch.setattr(capability_module, "M110_MAX_TOMBSTONED_NONCES", 2)
    secret = secrets.token_bytes(32)
    authority = capability_module.M110LaunchAuthority(
        launch_nonce="1" * 32,
        manifest_sha256="2" * 64,
        head="3" * 40,
        tree="4" * 40,
        authority_secret_sha256="5" * 64,
        _secret=bytearray(secret),
        _repo_root=Path.cwd(),
    )
    authority.source_snapshot_status = lambda: (
        True,
        "m110_live_source_snapshot_current",
    )

    def request(nonce):
        return sign_m110_request(
            {"type": "unit", "request_nonce": nonce},
            secret=secret,
        )

    first = request("a" * 32)
    assert authority.verify_and_tombstone_request(first)[0] is True
    assert authority.verify_and_tombstone_request(request("b" * 32))[0] is True
    assert authority.verify_and_tombstone_request(request("c" * 32)) == (
        False,
        "m110_request_nonce_capacity_exhausted",
    )
    assert authority.verify_and_tombstone_request(first) == (
        False,
        "m110_request_nonce_replayed",
    )


def test_closed_launch_authority_rejects_even_zero_key_signature():
    secret = secrets.token_bytes(32)
    authority = capability_module.M110LaunchAuthority(
        launch_nonce="1" * 32,
        manifest_sha256="2" * 64,
        head="3" * 40,
        tree="4" * 40,
        authority_secret_sha256="5" * 64,
        _secret=bytearray(secret),
        _repo_root=Path.cwd(),
    )
    authority.close()
    request = sign_m110_request(
        {"type": "unit", "request_nonce": "d" * 32},
        secret=b"\x00" * 32,
    )

    assert authority.verify_and_tombstone_request(request) == (
        False,
        "m110_launch_authority_closed",
    )


def _capability_binding(
    *,
    owner=None,
    route=None,
    transport=None,
    capture=None,
    owner_connection_id=7,
    proposal_id="proposal-1",
    idempotency_key="m110-idempotency-1",
):
    bundle = b'{"bundle":"unit"}'
    return M110CapabilityBinding(
        capability_kind="restore_only_proof",
        runtime_instance_nonce="1" * 32,
        runtime_manifest_sha256="sha256:" + "2" * 64,
        launch_manifest_sha256="sha256:" + "3" * 64,
        source_head="4" * 40,
        source_tree="5" * 40,
        connection_generation=6,
        transport_generation=7,
        console_serial="0100Q1V0604AAE214",
        firmware_version="3.0.5",
        route_proof_id="m110-route-1",
        route_artifact_sha256="sha256:" + "6" * 64,
        capture_hardware_uid="usb:5015:1291:214",
        emergency_generation=8,
        owner_connection_id=owner_connection_id,
        proposal_id=proposal_id,
        review_contract_version=2,
        review_content_digest="sha256:" + "7" * 64,
        operation_digest="sha256:" + __import__("hashlib").sha256(bundle).hexdigest(),
        canonical_bundle_bytes=bundle,
        approval_id="approval-1",
        idempotency_key=idempotency_key,
        scope_addresses=("/ch/1/eq/on",),
        target_fingerprints=(("/ch/1/eq/on", "1"),),
        _owner_ref=owner if owner is not None else object(),
        _route_ref=route if route is not None else object(),
        _transport_ref=transport if transport is not None else object(),
        _capture_ref=capture if capture is not None else object(),
    )


def test_issue_request_is_exact_and_non_coercing():
    request = {
        "type": "issue_supervised_capability",
        "contract_version": 1,
        "capability_kind": "restore_only_proof",
        "proposal_id": "proposal-1",
        "expected_review_contract_version": 2,
        "expected_review_content_digest": "sha256:" + "a" * 64,
        "expected_operation_digest": "sha256:" + "b" * 64,
        "approval_id": "approval-1",
        "confirmation": "ISSUE_RESTORE_ONLY_PROOF_CAPABILITY",
        "request_nonce": "c" * 32,
        "authority_proof": "hmac-sha256:" + "d" * 64,
    }
    assert validate_m110_capability_issue_request(request) == (
        True,
        "m110_capability_issue_request_valid",
    )
    assert validate_m110_capability_issue_request(
        {**request, "proof_run": True}
    )[0] is False
    assert validate_m110_capability_issue_request(
        {**request, "contract_version": True}
    )[0] is False
    assert validate_m110_capability_issue_request(
        {**request, "confirmation": "ISSUE_PROCESSOR_ENABLE_COMMIT_ONCE"}
    )[0] is False


def test_arm_and_apply_requests_are_exact_and_non_coercing():
    arm = {
        "type": "arm_m110_attended_control",
        "contract_version": 1,
        "approval_id": "approval-1",
        "confirmation": "ARM_M110_ATTENDED_ONE_SHOT_CONTROL",
        "request_nonce": "a" * 32,
        "authority_proof": "hmac-sha256:" + "b" * 64,
    }
    assert validate_m110_arm_request(arm) == (
        True,
        "m110_arm_request_valid",
    )
    assert validate_m110_arm_request({**arm, "contract_version": True})[0] is False
    apply_request = {
        "type": "apply_operator_proposal",
        "proposal_id": "proposal-1",
        "approved": True,
        "approval_id": "approval-1",
        "idempotency_key": "m110-idempotency-1",
        "expected_review_contract_version": 2,
        "expected_review_content_digest": "sha256:" + "c" * 64,
        "expected_operation_digest": "sha256:" + "d" * 64,
        "supervised_capability_token": "m110cap_v1_" + "e" * 64,
    }
    assert validate_m110_capability_apply_request(apply_request) == (
        True,
        "m110_capability_apply_request_valid",
    )
    assert validate_m110_capability_apply_request(
        {**apply_request, "proof_run": True}
    )[0] is False
    assert validate_m110_capability_apply_request(
        {**apply_request, "approved": 1}
    )[0] is False


def test_capability_store_is_digest_only_single_use_and_returns_typed_grant():
    binding = _capability_binding()
    store = SupervisedCapabilityStore(token_factory=lambda: "a" * 64)
    issued, reason = store.issue(binding)

    assert reason == "m110_capability_issued"
    assert issued is not None
    assert issued.token.startswith("m110cap_v1_")
    assert issued.token not in repr(issued)
    assert issued.token not in repr(store.__dict__)
    assert store.status()["stores_plaintext_tokens"] is False
    assert store.preview(issued.token, binding) == (
        True,
        "m110_capability_current",
    )

    grant, reason = store.consume(issued.token, binding)
    assert reason == "m110_capability_consumed"
    assert type(grant) is RestoreOnlyProofGrant
    assert grant.is_authentic() is True
    assert grant.binding.matches_exactly(binding) is True
    assert store.active_count() == 0
    assert store.consume(issued.token, binding) == (
        None,
        "m110_capability_token_replayed_or_revoked",
    )
    with pytest.raises(AttributeError):
        grant.binding = binding


def test_consumed_grant_is_linear_noncopyable_and_nonpickleable():
    binding = _capability_binding()
    store = SupervisedCapabilityStore(token_factory=lambda: "2" * 64)
    issued, _ = store.issue(binding)
    grant, _ = store.consume(issued.token, binding)

    assert grant.execution_status() == "issued"
    assert grant.begin_execution() is True
    assert grant.begin_execution() is False
    assert grant.mark_terminal("verified") is True
    assert grant.execution_status() == "terminal"
    assert grant.mark_terminal("again") is False
    with pytest.raises((AttributeError, TypeError)):
        grant._execution_state._phase = "issued"
    with pytest.raises(TypeError):
        copy.copy(grant)
    with pytest.raises(TypeError):
        copy.deepcopy(grant)
    with pytest.raises(TypeError):
        pickle.dumps(grant)


def test_restored_proof_receipt_is_factory_only_process_private_truth():
    owner = object()
    control = object()
    route = object()
    receipt = new_m110_restored_proof_receipt(
        proposal_id="proposal-1",
        review_contract_version=2,
        review_content_digest="sha256:" + "1" * 64,
        operation_digest="sha256:" + "2" * 64,
        result_digest="sha256:" + "3" * 64,
        transaction_id="transaction-1",
        runtime_instance_nonce="4" * 32,
        source_head="5" * 40,
        source_tree="6" * 40,
        route_proof_id="route-1",
        route_artifact_sha256="sha256:" + "7" * 64,
        owner_connection_id=8,
        created_at_epoch=10.0,
        created_at_monotonic=11.0,
        _owner_ref=owner,
        _control_ref=control,
        _route_ref=route,
    )

    assert type(receipt) is M110RestoredProofReceipt
    assert receipt.is_authentic() is True
    assert receipt._owner_ref is owner
    assert "m110proof_v1_" in receipt.receipt_id
    with pytest.raises(TypeError):
        M110RestoredProofReceipt()
    with pytest.raises(TypeError):
        copy.copy(receipt)
    with pytest.raises(TypeError):
        pickle.dumps(receipt)


def test_capability_binding_mismatch_does_not_consume_token():
    binding = _capability_binding()
    store = SupervisedCapabilityStore(token_factory=lambda: "b" * 64)
    issued, _ = store.issue(binding)
    assert issued is not None

    wrong = replace(binding, proposal_id="proposal-2")
    assert store.consume(issued.token, wrong) == (
        None,
        "m110_capability_binding_mismatch",
    )
    wrong_object = replace(binding, _route_ref=object())
    assert store.preview(issued.token, wrong_object) == (
        False,
        "m110_capability_binding_mismatch",
    )
    assert store.consume(issued.token, binding)[0].is_authentic() is True


def test_capability_latest_start_uses_monotonic_time_but_not_transaction_end():
    clock = {"epoch": 1000.0, "monotonic": 50.0}
    binding = _capability_binding()
    store = SupervisedCapabilityStore(
        epoch_clock=lambda: clock["epoch"],
        monotonic_clock=lambda: clock["monotonic"],
        token_factory=lambda: "c" * 64,
    )
    issued, _ = store.issue(binding)
    assert issued is not None
    clock["epoch"] += 10_000.0
    clock["monotonic"] += 29.999
    grant, reason = store.consume(issued.token, binding)
    assert reason == "m110_capability_consumed"
    assert grant.is_authentic() is True

    second_store = SupervisedCapabilityStore(
        epoch_clock=lambda: clock["epoch"],
        monotonic_clock=lambda: clock["monotonic"],
        token_factory=lambda: "d" * 64,
    )
    second, _ = second_store.issue(binding)
    clock["monotonic"] += 30.0
    assert second_store.consume(second.token, binding) == (
        None,
        "m110_capability_latest_start_expired",
    )


def test_second_owner_issue_revokes_previous_and_global_limit_is_eight():
    tokens = iter(f"{value:064x}" for value in range(1, 20))
    store = SupervisedCapabilityStore(token_factory=lambda: next(tokens))
    owner = object()
    first_binding = _capability_binding(owner=owner)
    first, _ = store.issue(first_binding)
    second_binding = replace(first_binding, proposal_id="proposal-2")
    second, _ = store.issue(second_binding)
    assert first is not None and second is not None
    assert store.consume(first.token, first_binding) == (
        None,
        "m110_capability_token_replayed_or_revoked",
    )
    assert store.active_count() == 1

    for index in range(2, 9):
        binding = _capability_binding(
            owner=object(),
            owner_connection_id=100 + index,
            proposal_id=f"proposal-{index + 1}",
            idempotency_key=f"m110-idempotency-{index + 1}",
        )
        assert store.issue(binding)[0] is not None
    ninth = _capability_binding(
        owner=object(),
        owner_connection_id=999,
        proposal_id="proposal-9",
        idempotency_key="m110-idempotency-9",
    )
    assert store.issue(ninth) == (
        None,
        "m110_capability_global_capacity_exhausted",
    )


def test_concurrent_consume_returns_exactly_one_grant():
    binding = _capability_binding()
    store = SupervisedCapabilityStore(token_factory=lambda: "e" * 64)
    issued, _ = store.issue(binding)
    assert issued is not None
    barrier = threading.Barrier(8)
    results = []

    def consume():
        barrier.wait()
        results.append(store.consume(issued.token, binding))

    threads = [threading.Thread(target=consume) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    grants = [grant for grant, _reason in results if grant is not None]
    assert len(grants) == 1
    assert grants[0].is_authentic() is True
    assert all(
        reason in {
            "m110_capability_consumed",
            "m110_capability_token_replayed_or_revoked",
        }
        for _grant, reason in results
    )


def test_revocation_is_owner_scoped_and_restart_store_cannot_accept_old_token():
    owner_one = object()
    owner_two = object()
    tokens = iter(("f" * 64, "0" * 64))
    store = SupervisedCapabilityStore(token_factory=lambda: next(tokens))
    binding_one = _capability_binding(owner=owner_one)
    binding_two = _capability_binding(
        owner=owner_two,
        owner_connection_id=9,
        proposal_id="proposal-2",
        idempotency_key="m110-idempotency-2",
    )
    issued_one, _ = store.issue(binding_one)
    issued_two, _ = store.issue(binding_two)
    assert store.revoke_owner(owner_one, reason="owner_disconnected") == 1
    assert store.consume(issued_one.token, binding_one)[0] is None
    assert store.consume(issued_two.token, binding_two)[0].is_authentic() is True

    restarted = SupervisedCapabilityStore(token_factory=lambda: "1" * 64)
    assert restarted.consume(issued_one.token, binding_one) == (
        None,
        "m110_capability_token_unknown",
    )


def test_revocation_is_exactly_proposal_scoped():
    tokens = iter(("2" * 64, "3" * 64))
    store = SupervisedCapabilityStore(token_factory=lambda: next(tokens))
    binding_one = _capability_binding(
        owner=object(),
        proposal_id="proposal-terminal",
    )
    binding_two = _capability_binding(
        owner=object(),
        owner_connection_id=10,
        proposal_id="proposal-still-pending",
        idempotency_key="m110-idempotency-still-pending",
    )
    issued_one, _ = store.issue(binding_one)
    issued_two, _ = store.issue(binding_two)

    assert store.revoke_proposal(
        "proposal-terminal",
        reason="proposal_terminal",
    ) == 1
    assert store.consume(issued_one.token, binding_one) == (
        None,
        "m110_capability_token_replayed_or_revoked",
    )
    assert store.consume(issued_two.token, binding_two)[0].is_authentic() is True
