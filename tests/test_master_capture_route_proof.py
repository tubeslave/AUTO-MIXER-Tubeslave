from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import threading
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import master_capture_route_proof as route_module

from master_capture_route_proof import (
    MASTER_CAPTURE_ROUTE_ADDRESSES,
    MASTER_CAPTURE_ROUTE_EXPECTED,
    MASTER_CAPTURE_ROUTE_REQUEST_CONFIRMATION,
    MASTER_CAPTURE_ROUTE_ARTIFACT_MAX_BYTES,
    discard_master_capture_route_artifact,
    is_m110_wing_firmware_version,
    validate_master_capture_route_request,
    verify_master_capture_route_artifact,
    write_master_capture_route_artifact,
)
from supervised_proof_capability import M110LaunchAuthority, sign_m110_request
from osc.enhanced_osc_client import ConnectionState
from server import AutoMixerServer, EnhancedOSCClient, WingClient


def _route_request(secret: bytes, nonce: str = "b" * 32):
    return sign_m110_request(
        {
            "type": "capture_wing_master_route_proof",
            "contract_version": 1,
            "request_id": "m110-route-unit",
            "approved": True,
            "approval_id": "operator-m110-unit",
            "confirmation": MASTER_CAPTURE_ROUTE_REQUEST_CONFIRMATION,
            "request_nonce": nonce,
        },
        secret=secret,
    )


def _launch_authority(server: AutoMixerServer, secret: bytes) -> M110LaunchAuthority:
    source = server._runtime_identity["source"]
    manifest_sha = "a" * 64
    server._runtime_identity["launch_binding"] = {
        "mode": "m110_attended_runner",
        "head": source["head"],
        "tree": source["tree"],
        "tracked_tree_clean": True,
        "prestart_manifest_sha256": manifest_sha,
    }
    authority = M110LaunchAuthority(
        launch_nonce="1" * 32,
        manifest_sha256=manifest_sha,
        head=source["head"],
        tree=source["tree"],
        authority_secret_sha256=hashlib.sha256(secret).hexdigest(),
        _secret=bytearray(secret),
        _repo_root=Path.cwd(),
    )
    authority.source_snapshot_status = MagicMock(
        return_value=(True, "m110_live_source_snapshot_current")
    )
    server._m110_launch_authority = authority
    return authority


def _capture_status(sample_count: int):
    return {
        "running": True,
        "real_input": True,
        "source_type": "sounddevice",
        "callback_age_sec": 0.01,
        "captured_sample_count": sample_count,
        "device_name": "Unit WING USB 48ch",
        "device_index": 7,
        "num_channels": 48,
        "sample_rate": 48000,
    }


def _hardware_identity(
    *,
    binding_sha256: str | None = None,
    usb_serial: str = "WINGUNIT1214",
):
    binding = {
        "schema": "automixer.wing_capture_hardware_identity.v1",
        "device_name": "Unit WING USB 48ch",
        "device_index": 7,
        "console_serial": "WINGUNIT1",
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
            "serial": usb_serial,
            "location_id": 1048576,
        },
        "serial_binding": "chassis_serial_plus_three_digit_usb_interface",
    }
    return {
        **binding,
        "verified": True,
        "reason": "physical_wing_usb_capture_bound_to_console",
        "hardware_uid": f"usb:1397:050b:{usb_serial}:00100000",
        "binding_sha256": binding_sha256
        or hashlib.sha256(
            route_module.canonical_json_bytes(binding)
        ).hexdigest(),
    }


def _configure_route_server(
    server,
    tmp_path,
    websocket,
    secret,
    *,
    firmware="3.0.5-unit",
):
    _launch_authority(server, secret)
    server._verify_m110_cp05_hardware_gate = MagicMock(
        return_value={"verified": True}
    )
    server._ws_auth_required = True
    server._authenticated_websocket_ids.add(id(websocket))
    server.operator_mode = "analyze"
    server._wing_observation_only = True
    server.connection_mode = "wing"
    server._m110_transport_owner_connection_id = id(websocket)
    server._m110_transport_owner_ref = websocket
    client = object.__new__(EnhancedOSCClient)
    client._wing = MagicMock(spec=WingClient)
    client._wing.is_connected = True
    client._state_lock = threading.Lock()
    client._state = ConnectionState.CONNECTED
    client._transport_generation = 1
    client.query_parameter = MagicMock()
    route = {
        "/$syscfg/$serial": "WINGUNIT1",
        "/$syscfg/$firmware": firmware,
        **MASTER_CAPTURE_ROUTE_EXPECTED,
    }
    client.query_parameter.side_effect = lambda address, timeout: {
        "status": "received",
        "address": address,
        "value": route[address],
        "confirmed_round_trips": 2,
    }
    server.mixer_client = client
    capture = MagicMock(running=True, num_channels=48)
    status_call_count = 0

    def capture_status():
        nonlocal status_call_count
        status_call_count += 1
        return _capture_status(100 if status_call_count == 1 else 12_100)

    capture.get_status.side_effect = capture_status
    capture.get_synchronized_buffers.return_value = {
        "buffers": {
            47: np.full(12_000, 0.1, dtype=np.float32),
            48: np.full(12_000, 0.08, dtype=np.float32),
        },
        "captured_sample_count": 12_100,
    }
    server.audio_capture = capture
    server._audio_capture_preview_owner = websocket
    server._master_capture_route_artifact_root = tmp_path / "route-artifacts"
    server._m110_capture_hardware_identity = MagicMock(
        return_value=_hardware_identity()
    )
    control_dir = tmp_path / "autopilot"
    server._m110_autopilot_stop_path = control_dir / "STOP"
    server._m110_autopilot_writer_lease_path = (
        control_dir / "WRITER_LEASE" / "owner.json"
    )
    server._m110_autopilot_writer_lease_dir = (
        server._m110_autopilot_writer_lease_path.parent
    )
    server._m110_legacy_autoloop_lock_path = control_dir / ".codex_autoloop.lock"
    return client


@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
def test_m110_blocks_empty_writer_lease_directory_and_legacy_lock(
    mock_config,
    mock_bleed,
    tmp_path,
):
    server = AutoMixerServer()
    control_dir = tmp_path / "autopilot"
    server._m110_autopilot_stop_path = control_dir / "STOP"
    server._m110_autopilot_writer_lease_path = (
        control_dir / "WRITER_LEASE" / "owner.json"
    )
    server._m110_autopilot_writer_lease_dir = (
        server._m110_autopilot_writer_lease_path.parent
    )
    server._m110_legacy_autoloop_lock_path = control_dir / ".codex_autoloop.lock"

    server._m110_autopilot_writer_lease_dir.mkdir(parents=True)
    assert "autopilot_writer_lease_present" in (
        server._m110_autopilot_control_blockers()
    )
    server._m110_autopilot_writer_lease_dir.rmdir()
    server._m110_legacy_autoloop_lock_path.touch()
    assert "legacy_autoloop_lock_present" in (
        server._m110_autopilot_control_blockers()
    )


def test_route_capture_request_contract_is_exact():
    secret = secrets.token_bytes(32)
    valid = _route_request(secret)
    assert validate_master_capture_route_request(valid) == ()

    for mutation in (
        {**valid, "address": "/ch/1/fdr"},
        {**valid, "approved": 1},
        {**valid, "contract_version": True},
        {**valid, "confirmation": "yes"},
    ):
        assert validate_master_capture_route_request(mutation)


@pytest.mark.parametrize(
    ("firmware", "accepted"),
    [
        ("3.0.5", True),
        ("3.0.5-unit", True),
        ("3.0.5-4-ga4ea833d", True),
        ("3.0.5-4-ga4ea833d:release", True),
        ("3.0.50", False),
        ("3.0.6-4-ga4ea833d:release", False),
        ("3.0.5:release", False),
        ("3.0.5-4-ga4ea833d:debug", False),
        ("3.0.5-4-ga4ea833d:release:extra", False),
        ("3.0.5-4-ga4ea833d/release", False),
    ],
)
def test_m110_wing_firmware_version_contract(firmware, accepted):
    assert is_m110_wing_firmware_version(firmware) is accepted


def test_route_audit_artifact_is_private_content_addressed_evidence(tmp_path):
    result = write_master_capture_route_artifact(
        root=tmp_path / "proofs",
        proof_id="a" * 32,
        issued_at_epoch=1_700_000_000.0,
        expires_at_epoch=1_700_001_800.0,
        evidence={"route": {"addresses": list(MASTER_CAPTURE_ROUTE_ADDRESSES)}},
    )
    path = Path(result["path"])
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert payload["accepted"] is True
    assert payload["proof_id"] == "a" * 32
    verified = verify_master_capture_route_artifact(
        root=tmp_path / "proofs",
        path=path,
        expected_sha256=result["sha256"],
        expected_proof_id="a" * 32,
        expected_evidence_sha256=result["evidence_sha256"],
    )
    assert verified["verified"] is True
    assert verified["live_authority_granted"] is False


@pytest.mark.parametrize("preexisting_kind", ["file", "symlink"])
def test_artifact_writer_never_deletes_preexisting_path(
    tmp_path,
    preexisting_kind,
):
    root = tmp_path / "proofs"
    root.mkdir(mode=0o700)
    proof_id = "e" * 32
    path = root / f"{proof_id}.json"
    target = tmp_path / "sentinel"
    target.write_text("sentinel", encoding="utf-8")
    if preexisting_kind == "file":
        path.write_text("keep", encoding="utf-8")
    else:
        path.symlink_to(target)

    with pytest.raises(FileExistsError):
        write_master_capture_route_artifact(
            root=root,
            proof_id=proof_id,
            issued_at_epoch=1.0,
            expires_at_epoch=2.0,
            evidence={"route": "unit"},
        )

    assert path.exists() or path.is_symlink()
    assert target.read_text(encoding="utf-8") == "sentinel"


def test_artifact_fsync_failure_removes_only_new_file(tmp_path):
    root = tmp_path / "proofs"
    proof_id = "f" * 32
    with patch.object(route_module.os, "fsync", side_effect=OSError("fsync")):
        with pytest.raises(OSError, match="fsync"):
            write_master_capture_route_artifact(
                root=root,
                proof_id=proof_id,
                issued_at_epoch=1.0,
                expires_at_epoch=2.0,
                evidence={"route": "unit"},
            )

    assert not (root / f"{proof_id}.json").exists()


def test_artifact_verifier_rejects_tamper_symlink_and_oversize(tmp_path):
    root = tmp_path / "proofs"
    result = write_master_capture_route_artifact(
        root=root,
        proof_id="1" * 32,
        issued_at_epoch=1.0,
        expires_at_epoch=2.0,
        evidence={"route": "unit"},
    )
    path = Path(result["path"])
    path.write_bytes(path.read_bytes() + b" ")
    assert verify_master_capture_route_artifact(
        root=root,
        path=path,
        expected_sha256=result["sha256"],
        expected_proof_id="1" * 32,
        expected_evidence_sha256=result["evidence_sha256"],
    )["verified"] is False

    path.unlink()
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    path.symlink_to(target)
    assert verify_master_capture_route_artifact(
        root=root,
        path=path,
        expected_sha256=result["sha256"],
        expected_proof_id="1" * 32,
        expected_evidence_sha256=result["evidence_sha256"],
    )["verified"] is False

    path.unlink()
    path.write_bytes(b"x" * (MASTER_CAPTURE_ROUTE_ARTIFACT_MAX_BYTES + 1))
    path.chmod(0o600)
    assert verify_master_capture_route_artifact(
        root=root,
        path=path,
        expected_sha256=result["sha256"],
        expected_proof_id="1" * 32,
        expected_evidence_sha256=result["evidence_sha256"],
    )["verified"] is False


@pytest.mark.asyncio
@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
async def test_fixed_route_capture_issues_only_process_private_authority(
    mock_config,
    mock_bleed,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AUTOMIXER_TEST_CONSOLE_RUNTIME_DIR", str(tmp_path / "runtime"))
    server = AutoMixerServer()
    websocket = object()
    secret = secrets.token_bytes(32)
    firmware = "3.0.5-4-ga4ea833d:release"
    client = _configure_route_server(
        server,
        tmp_path,
        websocket,
        secret,
        firmware=firmware,
    )

    result = await server.capture_wing_master_route_proof(
        websocket=websocket,
        request=_route_request(secret),
    )

    assert result["status"] == "issued"
    assert result["success"] is True
    assert result["firmware_version"] == firmware
    assert result["console_mutation_write_attempt_count"] == 0
    assert result["live_authority_granted"] is False
    assert result["conservative_true_peak_dbtp"] < -1.0
    assert client.query_parameter.call_count == 12
    client._wing.send.assert_not_called()
    assert [call.args[0] for call in client.query_parameter.call_args_list] == list(
        MASTER_CAPTURE_ROUTE_ADDRESSES
    ) * 2
    verified = server._verified_master_capture_route_proof()
    assert verified["verified"] is True
    assert verified["live_authority_source"] == (
        "same_process_non_serializable_object"
    )
    artifact_path = Path(result["artifact"]["path"])
    persisted_sha256 = result["artifact"]["sha256"]
    assert artifact_path.is_file()
    assert persisted_sha256 == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    assert verified["sha256"] == f"sha256:{persisted_sha256}"
    client._transport_generation += 1
    revoked = server._verified_master_capture_route_proof()
    assert revoked["verified"] is False
    assert "transport_generation_matches" in revoked["failed_checks"]


@pytest.mark.asyncio
@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
async def test_bad_hmac_and_replay_block_before_second_console_read(
    mock_config,
    mock_bleed,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AUTOMIXER_TEST_CONSOLE_RUNTIME_DIR", str(tmp_path / "runtime"))
    server = AutoMixerServer()
    websocket = object()
    secret = secrets.token_bytes(32)
    client = _configure_route_server(server, tmp_path, websocket, secret)
    request = _route_request(secret)
    bad = {**request, "authority_proof": "hmac-sha256:" + ("0" * 64)}

    blocked = await server.capture_wing_master_route_proof(
        websocket=websocket,
        request=bad,
    )
    assert blocked["reason"] == "m110_authority_proof_mismatch"
    assert client.query_parameter.call_count == 0

    issued = await server.capture_wing_master_route_proof(
        websocket=websocket,
        request=request,
    )
    assert issued["status"] == "issued"
    replayed = await server.capture_wing_master_route_proof(
        websocket=websocket,
        request=request,
    )
    assert replayed["reason"] == "m110_request_nonce_replayed"
    assert client.query_parameter.call_count == 12
    assert server._verified_master_capture_route_proof()["verified"] is True


@pytest.mark.asyncio
@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
async def test_valid_blocked_new_attempt_revokes_previous_route_authority(
    mock_config,
    mock_bleed,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AUTOMIXER_TEST_CONSOLE_RUNTIME_DIR", str(tmp_path / "runtime"))
    server = AutoMixerServer()
    websocket = object()
    secret = secrets.token_bytes(32)
    _configure_route_server(server, tmp_path, websocket, secret)
    issued = await server.capture_wing_master_route_proof(
        websocket=websocket,
        request=_route_request(secret, nonce="1" * 32),
    )
    assert issued["status"] == "issued"
    assert server._verified_master_capture_route_proof()["verified"] is True

    server._m110_autopilot_stop_path.parent.mkdir(parents=True, exist_ok=True)
    server._m110_autopilot_stop_path.touch()
    blocked = await server.capture_wing_master_route_proof(
        websocket=websocket,
        request=_route_request(secret, nonce="2" * 32),
    )

    assert blocked["reason"] == "route_capture_global_safety_gate_failed"
    assert server._master_capture_route_authority is None
    server._m110_autopilot_stop_path.unlink()
    assert server._verified_master_capture_route_proof()["verified"] is False


@pytest.mark.asyncio
@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
async def test_route_drift_blocks_without_artifact_or_authority(
    mock_config,
    mock_bleed,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AUTOMIXER_TEST_CONSOLE_RUNTIME_DIR", str(tmp_path / "runtime"))
    server = AutoMixerServer()
    websocket = object()
    secret = secrets.token_bytes(32)
    client = _configure_route_server(server, tmp_path, websocket, secret)
    call_count = 0

    def drifting(address, timeout):
        nonlocal call_count
        call_count += 1
        value = {
            "/$syscfg/$serial": "WINGUNIT1",
            "/$syscfg/$firmware": "3.0.5-unit",
            **MASTER_CAPTURE_ROUTE_EXPECTED,
        }[address]
        if call_count > 6 and address == "/io/out/USB/47/grp":
            value = "BUS"
        return {
            "status": "received",
            "address": address,
            "value": value,
            "confirmed_round_trips": 2,
        }

    client.query_parameter.side_effect = drifting
    result = await server.capture_wing_master_route_proof(
        websocket=websocket,
        request=_route_request(secret, nonce="c" * 32),
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "second_fixed_route_pass_failed"
    assert server._master_capture_route_authority is None
    assert not list((tmp_path / "route-artifacts").glob("*.json"))


@pytest.mark.asyncio
@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
async def test_headroom_failure_blocks_after_first_pass_without_authority(
    mock_config,
    mock_bleed,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AUTOMIXER_TEST_CONSOLE_RUNTIME_DIR", str(tmp_path / "runtime"))
    server = AutoMixerServer()
    websocket = object()
    secret = secrets.token_bytes(32)
    client = _configure_route_server(server, tmp_path, websocket, secret)
    server.audio_capture.get_synchronized_buffers.return_value = {
        "buffers": {
            47: np.ones(12_000, dtype=np.float32),
            48: np.full(12_000, 0.5, dtype=np.float32),
        },
        "captured_sample_count": 12_100,
    }

    result = await server.capture_wing_master_route_proof(
        websocket=websocket,
        request=_route_request(secret, nonce="d" * 32),
    )

    assert result["reason"] == "master_true_peak_headroom_insufficient"
    assert client.query_parameter.call_count == 6
    assert server._master_capture_route_authority is None
    assert not list((tmp_path / "route-artifacts").glob("*.json"))


@pytest.mark.asyncio
@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
async def test_route_capture_rejects_firmware_prefix_and_non_wing_device(
    mock_config,
    mock_bleed,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AUTOMIXER_TEST_CONSOLE_RUNTIME_DIR", str(tmp_path / "runtime"))
    server = AutoMixerServer()
    websocket = object()
    secret = secrets.token_bytes(32)
    client = _configure_route_server(server, tmp_path, websocket, secret)
    route = {
        "/$syscfg/$serial": "WINGUNIT1",
        "/$syscfg/$firmware": "3.0.50",
        **MASTER_CAPTURE_ROUTE_EXPECTED,
    }
    client.query_parameter.side_effect = lambda address, timeout: {
        "status": "received",
        "address": address,
        "value": route[address],
        "confirmed_round_trips": 2,
    }
    firmware = await server.capture_wing_master_route_proof(
        websocket=websocket,
        request=_route_request(secret, nonce="6" * 32),
    )
    assert firmware["reason"] == "first_fixed_route_pass_failed"

    client.query_parameter.reset_mock()
    route["/$syscfg/$firmware"] = "3.0.5"
    client.query_parameter.side_effect = lambda address, timeout: {
        "status": "received",
        "address": address,
        "value": route[address],
        "confirmed_round_trips": 2,
    }
    call_count = 0

    def unrelated_status():
        nonlocal call_count
        call_count += 1
        status = _capture_status(100 if call_count == 1 else 12_100)
        status["device_name"] = "Unrelated 48ch USB"
        return status

    server.audio_capture.get_status.side_effect = unrelated_status
    non_wing = await server.capture_wing_master_route_proof(
        websocket=websocket,
        request=_route_request(secret, nonce="7" * 32),
    )
    assert non_wing["reason"] == "capture_identity_or_freshness_invalid"


@pytest.mark.asyncio
@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        ("capture_replace", "capture_identity_current"),
        ("capture_stop", "capture_identity_current"),
        ("owner_auth_loss", "authenticated_owner_current"),
        ("emergency", "emergency_generation_matches"),
        ("stop_file", "autopilot_control_clear"),
        ("ttl", "route_authority_not_expired"),
        ("runtime", "runtime_manifest_matches"),
        ("source", "live_source_snapshot_current"),
    ],
)
async def test_route_authority_revokes_on_runtime_boundary_change(
    mock_config,
    mock_bleed,
    mutation,
    failed_check,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AUTOMIXER_TEST_CONSOLE_RUNTIME_DIR", str(tmp_path / "runtime"))
    server = AutoMixerServer()
    websocket = object()
    secret = secrets.token_bytes(32)
    _configure_route_server(server, tmp_path, websocket, secret)
    issued = await server.capture_wing_master_route_proof(
        websocket=websocket,
        request=_route_request(secret, nonce="8" * 32),
    )
    assert issued["status"] == "issued"

    if mutation == "capture_replace":
        server.audio_capture = MagicMock(running=True)
    elif mutation == "capture_stop":
        server.audio_capture.running = False
    elif mutation == "owner_auth_loss":
        server._authenticated_websocket_ids.remove(id(websocket))
    elif mutation == "emergency":
        server._manual_write_emergency_generation += 1
    elif mutation == "ttl":
        server._master_capture_route_authority = replace(
            server._master_capture_route_authority,
            expires_at_epoch=time.time() - 1.0,
        )
    elif mutation == "runtime":
        server._runtime_identity["manifest_sha256"] = "f" * 64
    elif mutation == "source":
        server._m110_launch_authority.source_snapshot_status.return_value = (
            False,
            "m110_live_source_snapshot_drift",
        )
    else:
        server._m110_autopilot_stop_path.parent.mkdir(parents=True, exist_ok=True)
        server._m110_autopilot_stop_path.touch()

    revoked = server._verified_master_capture_route_proof()
    assert revoked["verified"] is False
    assert failed_check in revoked["failed_checks"]


@pytest.mark.asyncio
@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
async def test_disconnect_mixer_revokes_route_authority(
    mock_config,
    mock_bleed,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AUTOMIXER_TEST_CONSOLE_RUNTIME_DIR", str(tmp_path / "runtime"))
    server = AutoMixerServer()
    websocket = object()
    secret = secrets.token_bytes(32)
    client = _configure_route_server(server, tmp_path, websocket, secret)
    issued = await server.capture_wing_master_route_proof(
        websocket=websocket,
        request=_route_request(secret, nonce="9" * 32),
    )
    assert issued["status"] == "issued"
    client.disconnect = MagicMock()

    await server.disconnect_mixer()

    assert server._master_capture_route_authority is None
    client.disconnect.assert_called_once_with()


@pytest.mark.asyncio
@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
async def test_owner_disconnect_tombstone_blocks_inflight_route_issuance(
    mock_config,
    mock_bleed,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AUTOMIXER_TEST_CONSOLE_RUNTIME_DIR", str(tmp_path / "runtime"))
    server = AutoMixerServer()
    websocket = object()
    secret = secrets.token_bytes(32)
    client = _configure_route_server(server, tmp_path, websocket, secret)
    route = {
        "/$syscfg/$serial": "WINGUNIT1",
        "/$syscfg/$firmware": "3.0.5",
        **MASTER_CAPTURE_ROUTE_EXPECTED,
    }
    query_count = 0

    def disconnecting_query(address, timeout):
        nonlocal query_count
        query_count += 1
        if query_count == 1:
            assert server._begin_websocket_disconnect(websocket) is True
        return {
            "status": "received",
            "address": address,
            "value": route[address],
            "confirmed_round_trips": 2,
        }

    client.query_parameter.side_effect = disconnecting_query
    result = await server.capture_wing_master_route_proof(
        websocket=websocket,
        request=_route_request(secret, nonce="a" * 32),
    )

    assert result["status"] == "blocked"
    assert server._master_capture_route_authority is None
    assert not list((tmp_path / "route-artifacts").glob("*.json"))


@pytest.mark.asyncio
@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
async def test_concurrent_route_attempt_revokes_first_before_second_reads(
    mock_config,
    mock_bleed,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AUTOMIXER_TEST_CONSOLE_RUNTIME_DIR", str(tmp_path / "runtime"))
    server = AutoMixerServer()
    websocket = object()
    secret = secrets.token_bytes(32)
    client = _configure_route_server(server, tmp_path, websocket, secret)
    capture_status_calls = 0

    def advancing_capture_status():
        nonlocal capture_status_calls
        capture_status_calls += 1
        return _capture_status(100 + 12_000 * capture_status_calls)

    def advancing_snapshot(channels, sample_count):
        return {
            "buffers": {
                47: np.full(sample_count, 0.1, dtype=np.float32),
                48: np.full(sample_count, 0.08, dtype=np.float32),
            },
            "captured_sample_count": 100 + 12_000 * capture_status_calls,
        }

    server.audio_capture.get_status.side_effect = advancing_capture_status
    server.audio_capture.get_synchronized_buffers.side_effect = advancing_snapshot
    original_query = client.query_parameter.side_effect
    observed_between_attempts: list[bool] = []
    query_count = 0

    def observing_query(address, timeout):
        nonlocal query_count
        query_count += 1
        if query_count == 13:
            observed_between_attempts.append(
                server._verified_master_capture_route_proof()["verified"]
            )
        return original_query(address, timeout)

    client.query_parameter.side_effect = observing_query
    first, second = await asyncio.gather(
        server.capture_wing_master_route_proof(
            websocket=websocket,
            request=_route_request(secret, nonce="b" * 32),
        ),
        server.capture_wing_master_route_proof(
            websocket=websocket,
            request=_route_request(secret, nonce="c" * 32),
        ),
    )

    assert first["status"] == "issued"
    assert second["status"] == "issued"
    assert observed_between_attempts == [False]
    assert server._verified_master_capture_route_proof()["verified"] is True


@pytest.mark.asyncio
@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
async def test_post_persistence_stop_blocks_issue_and_discards_artifact(
    mock_config,
    mock_bleed,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AUTOMIXER_TEST_CONSOLE_RUNTIME_DIR", str(tmp_path / "runtime"))
    server = AutoMixerServer()
    websocket = object()
    secret = secrets.token_bytes(32)
    _configure_route_server(server, tmp_path, websocket, secret)

    def persist_then_stop(**kwargs):
        result = write_master_capture_route_artifact(**kwargs)
        server._m110_autopilot_stop_path.parent.mkdir(parents=True, exist_ok=True)
        server._m110_autopilot_stop_path.touch()
        return result

    with patch("server.write_master_capture_route_artifact", persist_then_stop):
        result = await server.capture_wing_master_route_proof(
            websocket=websocket,
            request=_route_request(secret, nonce="d" * 32),
        )

    assert result["reason"] == (
        "route_capture_safety_gate_changed_during_artifact_persistence"
    )
    assert server._master_capture_route_authority is None
    assert not list((tmp_path / "route-artifacts").glob("*.json"))


@pytest.mark.asyncio
@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
async def test_post_persistence_capture_stop_blocks_issue_and_discards_artifact(
    mock_config,
    mock_bleed,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AUTOMIXER_TEST_CONSOLE_RUNTIME_DIR", str(tmp_path / "runtime"))
    server = AutoMixerServer()
    websocket = object()
    secret = secrets.token_bytes(32)
    _configure_route_server(server, tmp_path, websocket, secret)

    def persist_then_stop_capture(**kwargs):
        result = write_master_capture_route_artifact(**kwargs)
        server.audio_capture.running = False
        return result

    with patch(
        "server.write_master_capture_route_artifact",
        persist_then_stop_capture,
    ):
        result = await server.capture_wing_master_route_proof(
            websocket=websocket,
            request=_route_request(secret, nonce="e" * 32),
        )

    assert result["reason"] == (
        "route_capture_safety_gate_changed_during_artifact_persistence"
    )
    assert "capture_identity_or_freshness_changed" in result["details"][
        "failures"
    ]
    assert server._master_capture_route_authority is None
    assert not list((tmp_path / "route-artifacts").glob("*.json"))


@pytest.mark.asyncio
@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
async def test_post_persistence_source_drift_blocks_issue_and_discards_artifact(
    mock_config,
    mock_bleed,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AUTOMIXER_TEST_CONSOLE_RUNTIME_DIR", str(tmp_path / "runtime"))
    server = AutoMixerServer()
    websocket = object()
    secret = secrets.token_bytes(32)
    _configure_route_server(server, tmp_path, websocket, secret)

    def persist_then_drift(**kwargs):
        result = write_master_capture_route_artifact(**kwargs)
        server._m110_launch_authority.source_snapshot_status.return_value = (
            False,
            "m110_live_source_snapshot_drift",
        )
        return result

    with patch("server.write_master_capture_route_artifact", persist_then_drift):
        result = await server.capture_wing_master_route_proof(
            websocket=websocket,
            request=_route_request(secret, nonce="f" * 32),
        )

    assert result["reason"] == (
        "route_capture_safety_gate_changed_during_artifact_persistence"
    )
    assert "m110_live_source_snapshot_drift" in result["details"]["failures"]
    assert server._master_capture_route_authority is None
    assert not list((tmp_path / "route-artifacts").glob("*.json"))


@pytest.mark.asyncio
@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
async def test_post_persistence_hardware_identity_change_discards_artifact(
    mock_config,
    mock_bleed,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("AUTOMIXER_TEST_CONSOLE_RUNTIME_DIR", str(tmp_path / "runtime"))
    server = AutoMixerServer()
    websocket = object()
    secret = secrets.token_bytes(32)
    _configure_route_server(server, tmp_path, websocket, secret)
    server._m110_capture_hardware_identity.side_effect = [
        _hardware_identity(),
        _hardware_identity(binding_sha256="b" * 64),
    ]

    result = await server.capture_wing_master_route_proof(
        websocket=websocket,
        request=_route_request(secret, nonce="0" * 32),
    )

    assert result["reason"] == (
        "route_capture_safety_gate_changed_during_artifact_persistence"
    )
    assert "capture_hardware_identity_changed" in result["details"]["failures"]
    assert server._master_capture_route_authority is None
    assert not list((tmp_path / "route-artifacts").glob("*.json"))


@pytest.mark.asyncio
@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
async def test_same_capture_object_stream_swap_discards_route_artifact(
    mock_config,
    mock_bleed,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AUTOMIXER_TEST_CONSOLE_RUNTIME_DIR",
        str(tmp_path / "runtime"),
    )
    server = AutoMixerServer()
    websocket = object()
    secret = secrets.token_bytes(32)
    _configure_route_server(server, tmp_path, websocket, secret)
    measured_stream = server.audio_capture._stream

    def persist_then_swap_stream(**kwargs):
        result = write_master_capture_route_artifact(**kwargs)
        server.audio_capture._stream = object()
        return result

    with patch(
        "server.write_master_capture_route_artifact",
        persist_then_swap_stream,
    ):
        result = await server.capture_wing_master_route_proof(
            websocket=websocket,
            request=_route_request(secret, nonce="9" * 32),
        )

    assert result["reason"] == (
        "route_capture_safety_gate_changed_during_artifact_persistence"
    )
    assert "capture_identity_or_freshness_changed" in result["details"][
        "failures"
    ]
    assert server.audio_capture._stream is not measured_stream
    assert server._master_capture_route_authority is None
    assert not list((tmp_path / "route-artifacts").glob("*.json"))


@pytest.mark.asyncio
@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
async def test_fresh_hardware_drift_revokes_route_before_audio_recheck_use(
    mock_config,
    mock_bleed,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AUTOMIXER_TEST_CONSOLE_RUNTIME_DIR",
        str(tmp_path / "runtime"),
    )
    server = AutoMixerServer()
    websocket = object()
    secret = secrets.token_bytes(32)
    _configure_route_server(server, tmp_path, websocket, secret)
    issued = await server.capture_wing_master_route_proof(
        websocket=websocket,
        request=_route_request(secret, nonce="8" * 32),
    )
    assert issued["status"] == "issued"
    server._m110_capture_hardware_identity.return_value = _hardware_identity(
        binding_sha256="b" * 64,
    )

    refreshed = (
        await server._verified_master_capture_route_proof_with_fresh_hardware()
    )

    assert refreshed["verified"] is False
    assert refreshed["failed_checks"] == [
        "fresh_capture_hardware_identity_current"
    ]
    assert server._master_capture_route_authority is None


@pytest.mark.asyncio
@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
async def test_fresh_hardware_output_mode_failure_revokes_route_authority(
    mock_config,
    mock_bleed,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "AUTOMIXER_TEST_CONSOLE_RUNTIME_DIR",
        str(tmp_path / "runtime"),
    )
    server = AutoMixerServer()
    websocket = object()
    secret = secrets.token_bytes(32)
    _configure_route_server(server, tmp_path, websocket, secret)
    issued = await server.capture_wing_master_route_proof(
        websocket=websocket,
        request=_route_request(secret, nonce="7" * 32),
    )
    assert issued["status"] == "issued"
    server._m110_capture_hardware_identity.return_value = {
        "schema": "automixer.wing_capture_hardware_identity.v1",
        "verified": False,
        "reason": "coreaudio_capture_not_physical_wing",
        "details": {"checks": {"output_channels": False}},
    }

    refreshed = (
        await server._verified_master_capture_route_proof_with_fresh_hardware()
    )

    assert refreshed["verified"] is False
    assert refreshed["failed_checks"] == [
        "fresh_capture_hardware_identity_current"
    ]
    assert refreshed["hardware_identity_reason"] == (
        "coreaudio_capture_not_physical_wing"
    )
    assert server._master_capture_route_authority is None


@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
def test_fresh_server_cannot_reuse_previous_process_route_authority(
    mock_config,
    mock_bleed,
):
    fresh = AutoMixerServer()

    assert fresh._verified_master_capture_route_proof()["verified"] is False
    assert fresh._master_capture_route_authority is None


@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
def test_hand_authored_legacy_json_and_sha_never_grant_route_authority(
    mock_config,
    mock_bleed,
    tmp_path,
):
    server = AutoMixerServer()
    path = tmp_path / "forged.json"
    path.write_text(
        json.dumps(
            {
                "accepted": True,
                "proof_id": "forged",
                "console_serial": "WING-UNIT",
            }
        ),
        encoding="utf-8",
    )
    server._master_capture_route_proof_path = str(path)
    server._master_capture_route_proof_sha256 = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    result = server._verified_master_capture_route_proof()

    assert result["verified"] is False
    assert result["reason"] == "process_private_route_authority_required"
    assert result["legacy_path_sha_ignored"] is True


@patch("server.BleedService")
@patch.object(AutoMixerServer, "_load_config", return_value={})
def test_m110_launch_rejects_manifest_root_other_than_executing_checkout(
    mock_config,
    mock_bleed,
    tmp_path,
):
    authority = MagicMock()
    with patch(
        "server.consume_m110_attended_launch_authority",
        return_value=({"mode": "m110_attended_runner"}, authority),
    ), patch("server._REPO_ROOT", str(tmp_path / "other-checkout")):
        with pytest.raises(RuntimeError, match="checkout executing server.py"):
            AutoMixerServer()

    authority.close.assert_called_once_with()
