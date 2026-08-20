import fcntl
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time

import pytest

from dlive_supervised_fader_proof import (
    DLiveConsoleWriterLock,
    DLiveFaderProofApproval,
    DLiveProofWal,
    build_dlive_fader_proof_confirmation,
    recover_dlive_fader_restore_proof,
    run_dlive_fader_restore_proof,
    validate_dlive_fader_proof_plan,
)
from dlive_client import DLiveClient, fader_value_to_db
from dlive_midi_capture_evidence import review_dlive_fader_midi_capture


class FakeDLiveClient:
    instances = []
    initial_fader_db = -6.0
    initial_muted = True
    expected_name = "Vocal"
    apply_mode = "normal"
    rollback_mode = "normal"
    disconnect_mode = "normal"
    change_on_fader_query = None
    interrupt_on_fader_query_count = None
    configured_ceiling_db = 0.0

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.is_connected = False
        self.fader_db = float(type(self).initial_fader_db)
        self.muted = bool(type(self).initial_muted)
        self.name = type(self).expected_name
        self.sends = []
        self.fader_query_count = 0
        self._fader_ceiling_db = float(type(self).configured_ceiling_db)
        type(self).instances.append(self)

    def connect(self, timeout):
        self.is_connected = True
        return True

    def disconnect(self):
        if type(self).disconnect_mode == "raise":
            raise RuntimeError("simulated_disconnect_failure")
        self.is_connected = False

    def _selector(self, channel, channel_type="input"):
        assert channel_type == "input"
        return self.kwargs.get("midi_base_channel", 0), channel - 1

    def query_parameter(self, address, timeout_sec):
        if address.endswith("/name"):
            value = self.name
        elif address.endswith("/mute"):
            value = self.muted
        else:
            self.fader_query_count += 1
            if self.fader_query_count == type(self).interrupt_on_fader_query_count:
                raise KeyboardInterrupt("simulated_operator_emergency_stop")
            change = type(self).change_on_fader_query
            if change and self.fader_query_count == change[0]:
                self.fader_db = change[1]
            value = self.fader_db
        return {
            "status": "received",
            "address": address,
            "value": value,
            "confirmed_round_trips": 2,
        }

    def _supervised_fader_write(self, channel, value_db, *, authority, phase):
        if not authority.consume_fader_write(
            client=self,
            channel=channel,
            value_db=value_db,
            phase=phase,
        ):
            return False
        self.sends.append((phase, channel, value_db))
        if phase == "apply":
            if type(self).apply_mode == "raise_after_apply":
                self.fader_db = value_db
                raise RuntimeError("simulated_disconnect_after_apply")
            if type(self).apply_mode == "interrupt_after_apply":
                self.fader_db = value_db
                raise KeyboardInterrupt("simulated_operator_emergency_stop")
            if type(self).apply_mode == "mismatch":
                self.fader_db = value_db + 1.5
            elif type(self).apply_mode != "no_change":
                self.fader_db = value_db
        elif type(self).rollback_mode == "normal":
            self.fader_db = value_db
        return type(self).rollback_mode != "send_false" or phase != "rollback"


class LoopbackRealDLiveClient(DLiveClient):
    instances = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fader_db = -6.0
        self.muted = True
        self.name = "Vocal"
        self.sent_values = []
        type(self).instances.append(self)

    def connect(self, timeout):
        self.is_connected = True
        return True

    def disconnect(self):
        self.is_connected = False

    def query_parameter(self, address, timeout_sec):
        value = (
            self.name if address.endswith("/name")
            else self.muted if address.endswith("/mute")
            else self.fader_db
        )
        return {
            "status": "received",
            "address": address,
            "value": value,
            "confirmed_round_trips": 2,
        }

    def _send_raw(self, payload):
        self.fader_db = fader_value_to_db(payload[-1])
        self.sent_values.append(self.fader_db)
        return True

    def _supervised_fader_write(self, channel, value_db, *, authority, phase, channel_type="input"):
        if phase == "rollback":
            self._fader_ceiling_db = -20.0
        return super()._supervised_fader_write(
            channel,
            value_db,
            authority=authority,
            phase=phase,
            channel_type=channel_type,
        )


@pytest.fixture(autouse=True)
def reset_fake():
    FakeDLiveClient.instances = []
    FakeDLiveClient.initial_fader_db = -6.0
    FakeDLiveClient.initial_muted = True
    FakeDLiveClient.expected_name = "Vocal"
    FakeDLiveClient.apply_mode = "normal"
    FakeDLiveClient.rollback_mode = "normal"
    FakeDLiveClient.disconnect_mode = "normal"
    FakeDLiveClient.change_on_fader_query = None
    FakeDLiveClient.interrupt_on_fader_query_count = None
    FakeDLiveClient.configured_ceiling_db = 0.0
    LoopbackRealDLiveClient.instances = []


def approval(**overrides):
    now = time.time()
    values = {
        "approval_id": "approval-1",
        "issued_at_epoch": now,
        "expires_at_epoch": now + 120.0,
        "host": "192.168.3.70",
        "endpoint": "surface",
        "port": 51329,
        "tls": True,
        "midi_base_channel": 0,
        "user_profile": 0,
        "expected_interface": "en7",
        "input_channel": 5,
        "expected_name": "Vocal",
        "operator_attested_model": "S7000",
        "operator_attested_firmware": "2.0",
        "operator_attested_show": "TEST",
        "target_db": -9.0,
        "max_cut_db": 3.0,
        "pa_outputs_contained": True,
        "require_channel_muted": True,
        "receive_midi_confirmed_enabled": True,
        "readonly_artifact_path": "/tmp/dlive-readonly-placeholder.json",
        "readonly_artifact_sha256": "0" * 64,
        "confirmation": "",
        "security_state_attested": True,
        "security_mode_attested": "tls_authenticated",
        "security_changed_for_setup": False,
        "reboot_after_security_change_confirmed": False,
        "wire_evidence_path": "/tmp/dlive-wire-placeholder.json",
        "wire_evidence_sha256": "1" * 64,
    }
    confirmation_overridden = "confirmation" in overrides
    values.update(overrides)
    result = DLiveFaderProofApproval(**values)
    if not confirmation_overridden:
        result = replace(
            result,
            confirmation=build_dlive_fader_proof_confirmation(result),
        )
    return result


def bind_readonly_artifact(tmp_path, proof_approval, **artifact_overrides):
    channel = proof_approval.input_channel
    artifact = {
        "schema": "automixer.dlive_attended_readonly_probe.v1",
        "probe_id": "probe-1",
        "started_at_utc": "2026-08-04T09:00:00+00:00",
        "finished_at_utc": "2026-08-04T09:00:01+00:00",
        "status": "readback_observed",
        "reason": "three_fields_received_twice_no_mutation_authority",
        "endpoint": proof_approval.endpoint,
        "host": proof_approval.host,
        "port": proof_approval.port,
        "tls": proof_approval.tls,
        "midi_base_channel": proof_approval.midi_base_channel,
        "user_profile": proof_approval.user_profile,
        "input_channel": channel,
        "expected_route_interface": proof_approval.expected_interface,
        "observed_route_interface": proof_approval.expected_interface,
        "operator_attestation": {
            "model": proof_approval.operator_attested_model,
            "firmware": proof_approval.operator_attested_firmware,
            "show": proof_approval.operator_attested_show,
            "machine_verified": False,
        },
        "readbacks": {
            f"/ch/{channel}/name": {
                "status": "received",
                "address": f"/ch/{channel}/name",
                "value": proof_approval.expected_name,
                "confirmed_round_trips": 2,
            },
            f"/ch/{channel}/fdr": {
                "status": "received",
                "address": f"/ch/{channel}/fdr",
                "value": FakeDLiveClient.initial_fader_db,
                "confirmed_round_trips": 2,
            },
            f"/ch/{channel}/mute": {
                "status": "received",
                "address": f"/ch/{channel}/mute",
                "value": True,
                "confirmed_round_trips": 2,
            },
        },
        "midi_get_request_count": 6,
        "mutation_write_attempt_count": 0,
        "live_write": False,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "console_contacted": True,
        "teardown": {
            "disconnect_requested": True,
            "client_connected_after_teardown": False,
        },
    }
    artifact.update(artifact_overrides)
    path = (tmp_path / "readonly" / "probe.json").absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    confirmation_was_valid = (
        proof_approval.confirmation
        == build_dlive_fader_proof_confirmation(proof_approval)
    )
    capture = {
        "schema": "automixer.dlive_midi_capture.v1",
        "capture_id": "capture-1",
        "captured_at_utc": "2026-08-04T09:00:02+00:00",
        "capture_tool": "operator-packet-capture",
        "source_client": "allen_heath_dlive_director",
        "operator_owned": True,
        "acquired_by_automixer": False,
        "interface_opened_by_automixer": False,
        "direction": "client_to_selected_endpoint",
        "observation_point": "selected_endpoint_ingress",
        "endpoint": {
            "host": proof_approval.host,
            "endpoint": proof_approval.endpoint,
            "port": proof_approval.port,
            "tls": proof_approval.tls,
            "midi_base_channel": proof_approval.midi_base_channel,
            "user_profile": proof_approval.user_profile,
        },
        "bytes_hex": "bb63046217066f",
    }
    # The planned packet depends on channel and target. Build the accepted
    # report with production code, then bind its exact file digest.
    packet_client = DLiveClient(
        ip=proof_approval.host,
        endpoint=proof_approval.endpoint,
        port=proof_approval.port,
        tls=proof_approval.tls,
        midi_base_channel=proof_approval.midi_base_channel,
        user_profile=proof_approval.user_profile,
    )
    midi_channel, selector = packet_client._selector(
        proof_approval.input_channel,
        "input",
    )
    from dlive_client import build_nrpn, db_to_fader_value, FADER_PARAMETER_ID

    capture["bytes_hex"] = build_nrpn(
        midi_channel,
        selector,
        FADER_PARAMETER_ID,
        db_to_fader_value(proof_approval.target_db),
    ).hex()
    wire_report = review_dlive_fader_midi_capture(
        capture,
        host=proof_approval.host,
        endpoint=proof_approval.endpoint,
        port=proof_approval.port,
        tls=proof_approval.tls,
        midi_base_channel=proof_approval.midi_base_channel,
        user_profile=proof_approval.user_profile,
        input_channel=proof_approval.input_channel,
        target_db=proof_approval.target_db,
    )
    wire_path = (tmp_path / "wire" / "review.json").absolute()
    wire_path.parent.mkdir(parents=True, exist_ok=True)
    wire_path.write_text(json.dumps(wire_report, sort_keys=True) + "\n", encoding="utf-8")
    wire_digest = hashlib.sha256(wire_path.read_bytes()).hexdigest()

    bound = replace(
        proof_approval,
        readonly_artifact_path=str(path),
        readonly_artifact_sha256=digest,
        wire_evidence_path=str(wire_path),
        wire_evidence_sha256=wire_digest,
    )
    if confirmation_was_valid:
        bound = replace(
            bound,
            confirmation=build_dlive_fader_proof_confirmation(bound),
        )
    return bound


def run(tmp_path, proof_approval=None, *, artifact_overrides=None, **overrides):
    values = {
        "client_factory": FakeDLiveClient,
        "route_interface_resolver": lambda host: "en7",
        "wal_path": tmp_path / "runtime" / "dlive_wal.jsonl",
        "writer_lock_path": tmp_path / "runtime" / "console_writer.lock",
        "timeout_sec": 0.01,
    }
    values.update(overrides)
    bound_approval = bind_readonly_artifact(
        tmp_path,
        proof_approval or approval(),
        **(artifact_overrides or {}),
    )
    return run_dlive_fader_restore_proof(
        bound_approval,
        **values,
    )


def test_restore_only_proof_applies_reads_back_and_restores(tmp_path):
    receipt = run(tmp_path)
    client = FakeDLiveClient.instances[0]

    assert receipt["status"] == "proof_passed"
    assert receipt["reason"] == "apply_readback_exact_rollback_and_teardown_verified"
    assert client.sends == [("apply", 5, -9.0), ("rollback", 5, -6.0)]
    assert client.fader_db == -6.0
    assert receipt["mutation_write_attempt_count"] == 2
    assert receipt["standing_live_authority"] is False
    assert receipt["scoped_restore_only_authority_granted"] is False
    assert receipt["client_connected_after_teardown"] is False
    assert receipt["readonly_artifact"]["source_fader_db"] == -6.0
    assert receipt["wire_evidence"]["exact_wire_match"] is True
    assert receipt["wire_evidence"]["console_acceptance_proven"] is False

    records = DLiveProofWal(tmp_path / "runtime" / "dlive_wal.jsonl").records
    phases = [record["phase"] for record in records]
    assert phases.index("prepared") < phases.index("apply_send_boundary")
    assert phases.index("apply_send_boundary") < phases.index("apply_send_result")
    assert phases.index("apply_verified") < phases.index("rollback_send_boundary")
    assert phases[-1] == "proof_passed"
    assert records[1]["payload"]["send_may_have_occurred"] is True
    assert records[0]["payload"]["readonly_artifact"]["sha256"].startswith("sha256:")


def test_real_client_hook_allows_exact_rollback_above_later_ceiling(tmp_path):
    receipt = run(
        tmp_path,
        client_factory=LoopbackRealDLiveClient,
    )
    client = LoopbackRealDLiveClient.instances[0]
    assert receipt["status"] == "proof_passed"
    assert len(client.sent_values) == 2
    assert client.sent_values[0] < -8.5
    assert client.sent_values[1] > -7.0


def test_restored_state_is_not_proof_passed_until_disconnect_verifies(tmp_path):
    FakeDLiveClient.disconnect_mode = "raise"
    receipt = run(tmp_path)
    records = DLiveProofWal(tmp_path / "runtime" / "dlive_wal.jsonl").records
    assert receipt["status"] == "manual_review_required"
    assert receipt["reason"] == "proof_teardown_unverified"
    assert receipt["client_connected_after_teardown"] is True
    assert records[-1]["phase"] == "manual_review_required"


def test_wrong_route_blocks_before_client_and_wal(tmp_path):
    receipt = run(
        tmp_path,
        client_factory=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("client must not be constructed")
        ),
        route_interface_resolver=lambda host: "utun4",
    )
    assert receipt["reason"] == "dlive_route_interface_mismatch"
    assert receipt["hardware_contacted"] is False
    assert receipt["mutation_write_attempt_count"] == 0
    assert not (tmp_path / "runtime" / "dlive_wal.jsonl").exists()


@pytest.mark.parametrize(
    "artifact_overrides,reason",
    [
        ({"endpoint": "mixrack"}, "readonly_artifact_binding_mismatch:endpoint"),
        ({"user_profile": None}, "readonly_artifact_binding_mismatch:user_profile"),
        ({"mutation_write_attempt_count": 1}, "readonly_artifact_binding_mismatch:mutation_write_attempt_count"),
        ({"live_write": True}, "readonly_artifact_binding_mismatch:live_write"),
    ],
)
def test_readonly_artifact_binding_mismatch_blocks_before_route_and_client(
    tmp_path,
    artifact_overrides,
    reason,
):
    route_calls = []
    receipt = run(
        tmp_path,
        artifact_overrides=artifact_overrides,
        client_factory=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("client must not be constructed")
        ),
        route_interface_resolver=lambda host: route_calls.append(host) or "en7",
    )
    assert receipt["reason"] == reason
    assert route_calls == []
    assert receipt["hardware_contacted"] is False
    assert receipt["mutation_write_attempt_count"] == 0


def test_readonly_artifact_digest_mismatch_blocks_before_route_and_client(tmp_path):
    bound = bind_readonly_artifact(tmp_path, approval())
    mismatched = replace(bound, readonly_artifact_sha256="f" * 64)
    mismatched = replace(
        mismatched,
        confirmation=build_dlive_fader_proof_confirmation(mismatched),
    )
    route_calls = []
    receipt = run_dlive_fader_restore_proof(
        mismatched,
        client_factory=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("client must not be constructed")
        ),
        route_interface_resolver=lambda host: route_calls.append(host) or "en7",
        wal_path=tmp_path / "runtime" / "dlive_wal.jsonl",
        writer_lock_path=tmp_path / "runtime" / "console_writer.lock",
    )
    assert receipt["reason"] == "readonly_artifact_sha256_mismatch"
    assert route_calls == []
    assert receipt["mutation_write_attempt_count"] == 0


def test_route_change_after_lock_blocks_before_client(tmp_path):
    routes = iter(("en7", "utun4"))
    receipt = run(
        tmp_path,
        client_factory=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("client must not be constructed")
        ),
        route_interface_resolver=lambda host: next(routes),
    )
    assert receipt["reason"] == "dlive_route_interface_mismatch"
    assert [item["observed"] for item in receipt["route_checks"]] == ["en7", "utun4"]
    assert receipt["mutation_write_attempt_count"] == 0


def test_route_change_after_apply_boundary_is_zero_send(tmp_path):
    routes = iter(("en7", "en7", "utun4"))
    receipt = run(
        tmp_path,
        route_interface_resolver=lambda host: next(routes),
    )
    assert receipt["status"] == "blocked"
    assert receipt["reason"] == "dlive_route_interface_changed"
    assert receipt["mutation_write_attempt_count"] == 0
    assert FakeDLiveClient.instances[0].sends == []


def test_unmuted_or_wrong_name_blocks_with_zero_writes(tmp_path):
    FakeDLiveClient.initial_muted = False
    receipt = run(tmp_path)
    assert receipt["reason"] == "proof_input_channel_not_muted"
    assert receipt["mutation_write_attempt_count"] == 0
    assert FakeDLiveClient.instances[0].sends == []

    FakeDLiveClient.initial_muted = True
    FakeDLiveClient.expected_name = "Kick"
    receipt = run(tmp_path / "second")
    assert receipt["reason"] == "operator_expected_channel_name_mismatch"
    assert receipt["mutation_write_attempt_count"] == 0


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"confirmation": "yes"}, "artifact_target_bound_operator_confirmation_required"),
        ({"target_db": -3.0}, "fader_increase_not_allowed"),
        ({"target_db": -13.0}, "fader_cut_exceeds_operator_bound"),
        ({"pa_outputs_contained": False}, "pa_outputs_must_be_physically_contained"),
        ({"receive_midi_confirmed_enabled": False}, "proof_requires_receive_midi_enabled"),
        ({"security_state_attested": False}, "proof_requires_effective_security_state_attestation"),
        ({"security_mode_attested": "no_security"}, "proof_security_mode_mismatch"),
        (
            {
                "security_changed_for_setup": True,
                "reboot_after_security_change_confirmed": False,
            },
            "proof_requires_reboot_after_security_change",
        ),
    ],
)
def test_invalid_or_unbounded_approval_never_sends(tmp_path, changes, reason):
    receipt = run(tmp_path, approval(**changes))
    assert receipt["mutation_write_attempt_count"] == 0
    assert receipt["live_write"] is False
    if "exception_type" in receipt:
        assert receipt["exception_type"] == "ValueError"
    assert reason in receipt.get("reason", "") or receipt["reason"] == "proof_exception_recovery_attempted"
    if FakeDLiveClient.instances:
        assert FakeDLiveClient.instances[0].sends == []


def test_plaintext_transport_cannot_be_used_as_supervised_identity_or_mutation_proof(tmp_path):
    plaintext = approval(
        tls=False,
        port=51328,
        security_mode_attested="no_security",
    )

    receipt = run(tmp_path, plaintext)

    assert receipt["status"] == "blocked"
    assert "proof_requires_tls_authenticated_transport" in receipt["reason"]
    assert receipt["hardware_contacted"] is False
    assert receipt["mutation_write_attempt_count"] == 0
    assert FakeDLiveClient.instances == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_db", -8.0),
        ("expected_name", "Kick"),
        ("input_channel", 6),
        ("readonly_artifact_sha256", "f" * 64),
    ],
)
def test_operator_confirmation_is_bound_to_exact_proof_fields(
    tmp_path,
    field,
    value,
):
    bound = bind_readonly_artifact(tmp_path, approval())
    tampered = replace(bound, **{field: value})

    receipt = run_dlive_fader_restore_proof(
        tampered,
        client_factory=FakeDLiveClient,
        route_interface_resolver=lambda host: "en7",
        wal_path=tmp_path / "runtime" / "dlive_wal.jsonl",
        writer_lock_path=tmp_path / "runtime" / "console_writer.lock",
        timeout_sec=0.01,
    )

    assert receipt["reason"] == (
        "artifact_target_bound_operator_confirmation_required"
    )
    assert receipt["hardware_contacted"] is False
    assert receipt["mutation_write_attempt_count"] == 0
    assert FakeDLiveClient.instances == []


def test_required_confirmation_changes_with_artifact_endpoint_channel_or_target():
    base = approval()
    base_confirmation = build_dlive_fader_proof_confirmation(base)

    changed = (
        replace(base, readonly_artifact_sha256="f" * 64),
        replace(base, host="192.168.3.71"),
        replace(base, input_channel=6),
        replace(base, target_db=-8.0),
        replace(base, approval_id="approval-2"),
    )

    assert base_confirmation.startswith(
        "APPLY_AND_ROLLBACK_DLIVE_FADER_PROOF:sha256:"
    )
    assert all(
        build_dlive_fader_proof_confirmation(item) != base_confirmation
        for item in changed
    )


def test_no_transport_plan_validator_uses_exact_artifact_and_packets(tmp_path):
    bound = bind_readonly_artifact(tmp_path, approval())

    receipt = validate_dlive_fader_proof_plan(bound)

    assert receipt["status"] == "ready_for_operator_confirmation"
    assert receipt["readonly_artifact"]["source_fader_db"] == -6.0
    assert receipt["bounded_cut"]["target_db"] == -9.0
    assert receipt["apply_packet"]["packet_hex"]
    assert receipt["rollback_packet"]["packet_hex"]
    assert receipt["wire_evidence"]["exact_wire_match"] is True
    assert receipt["wire_evidence"]["wire_delivery_observed"] is True
    assert receipt["wire_evidence"]["console_acceptance_proven"] is False
    assert receipt["security_attestation"]["state_attested"] is True
    assert receipt["required_confirmation"].startswith(
        "APPLY_AND_ROLLBACK_DLIVE_FADER_PROOF:sha256:"
    )
    assert receipt["approval_id"] == bound.approval_id
    for field in (
        "operator_confirmation_recorded",
        "physical_conditions_attested",
        "route_checked",
        "console_contacted",
        "hardware_contacted",
        "midi_sent",
        "live_write",
        "execution_allowed",
        "live_authority_granted",
        "standing_live_authority",
    ):
        assert receipt[field] is False
    assert receipt["mutation_write_attempt_count"] == 0


def test_no_transport_plan_validator_rejects_artifact_digest_mismatch(tmp_path):
    bound = bind_readonly_artifact(tmp_path, approval())

    with pytest.raises(ValueError, match="readonly_artifact_sha256_mismatch"):
        validate_dlive_fader_proof_plan(
            replace(bound, readonly_artifact_sha256="f" * 64)
        )


def test_no_transport_plan_validator_requires_bound_wire_evidence(tmp_path):
    bound = bind_readonly_artifact(tmp_path, approval())
    missing = replace(
        bound,
        wire_evidence_path=str((tmp_path / "missing-wire.json").absolute()),
        wire_evidence_sha256="f" * 64,
    )
    missing = replace(
        missing,
        confirmation=build_dlive_fader_proof_confirmation(missing),
    )

    with pytest.raises(ValueError, match="wire_evidence_unavailable"):
        validate_dlive_fader_proof_plan(missing)


def test_no_transport_plan_validator_rejects_wire_operation_rebinding(tmp_path):
    bound = bind_readonly_artifact(tmp_path, approval())
    path = Path(bound.wire_evidence_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    report["operation"]["input_channel"] = 6
    path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    tampered = replace(
        bound,
        wire_evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    tampered = replace(
        tampered,
        confirmation=build_dlive_fader_proof_confirmation(tampered),
    )

    with pytest.raises(ValueError, match="wire_evidence_binding_mismatch:operation"):
        validate_dlive_fader_proof_plan(tampered)


def test_consumed_approval_id_cannot_start_a_second_proof(tmp_path):
    first = run(tmp_path)
    first_client_count = len(FakeDLiveClient.instances)

    second = run(tmp_path)

    assert first["status"] == "proof_passed"
    assert second["status"] == "blocked"
    assert second["reason"] == "operator_approval_already_consumed"
    assert second["hardware_contacted"] is False
    assert second["mutation_write_attempt_count"] == 0
    assert len(FakeDLiveClient.instances) == first_client_count


@pytest.mark.parametrize(
    "proof_approval,initial_fader,reason",
    [
        (
            approval(target_db=-5.5, max_cut_db=1.0),
            -6.0,
            "fader_increase_not_allowed",
        ),
        (
            approval(target_db=-12.25, max_cut_db=6.0),
            -6.0,
            "fader_cut_exceeds_operator_bound",
        ),
        (
            approval(target_db=-1.0, max_cut_db=2.0),
            0.5,
            "preimage_above_zero_db_not_supported",
        ),
    ],
)
def test_quantized_direction_and_exact_operator_cut_bound_fail_closed(
    tmp_path, proof_approval, initial_fader, reason
):
    FakeDLiveClient.initial_fader_db = initial_fader
    receipt = run(tmp_path, proof_approval)

    assert receipt["reason"] == reason
    assert receipt["mutation_write_attempt_count"] == 0
    assert receipt["live_write"] is False
    assert FakeDLiveClient.instances[0].sends == []


def test_negative_infinity_target_is_rejected_before_hardware_contact(tmp_path):
    receipt = run(
        tmp_path,
        approval(target_db=-54.0, max_cut_db=6.0),
        client_factory=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("client must not be constructed")
        ),
    )

    assert receipt["reason"] == "target_db_must_not_map_to_negative_infinity"
    assert receipt["hardware_contacted"] is False
    assert receipt["mutation_write_attempt_count"] == 0


def test_existing_writer_lock_blocks_before_client(tmp_path):
    lock_path = tmp_path / "runtime" / "console_writer.lock"
    holder = DLiveConsoleWriterLock(lock_path)
    assert holder.acquire() is True
    try:
        receipt = run(
            tmp_path,
            writer_lock_path=lock_path,
            client_factory=lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("client must not be constructed")
            ),
        )
    finally:
        holder.release()
    assert receipt["reason"] == "console_writer_conflict"
    assert receipt["mutation_write_attempt_count"] == 0


def test_exception_after_apply_attempts_exact_rollback(tmp_path):
    FakeDLiveClient.apply_mode = "raise_after_apply"
    receipt = run(tmp_path)
    client = FakeDLiveClient.instances[0]
    assert receipt["status"] == "restored_after_unverified_apply"
    assert client.sends == [("apply", 5, -9.0), ("rollback", 5, -6.0)]
    assert client.fader_db == -6.0
    assert receipt["write_state_unknown"] is False


def test_operator_emergency_after_apply_attempts_exact_rollback(tmp_path):
    FakeDLiveClient.apply_mode = "interrupt_after_apply"
    receipt = run(tmp_path)
    client = FakeDLiveClient.instances[0]
    assert receipt["status"] == "restored_after_unverified_apply"
    assert client.sends == [("apply", 5, -9.0), ("rollback", 5, -6.0)]
    assert client.fader_db == -6.0
    assert receipt["write_state_unknown"] is False


def test_operator_emergency_after_durable_boundary_is_terminal_zero_send(tmp_path):
    FakeDLiveClient.interrupt_on_fader_query_count = 2
    receipt = run(tmp_path)
    client = FakeDLiveClient.instances[0]
    records = DLiveProofWal(tmp_path / "runtime" / "dlive_wal.jsonl").records
    assert receipt["status"] == "blocked"
    assert receipt["reason"] == "exception_or_emergency_before_transport"
    assert receipt["mutation_write_attempt_count"] == 0
    assert receipt["write_state_unknown"] is False
    assert client.sends == []
    assert records[-1]["phase"] == "blocked_no_send"
    assert DLiveProofWal(
        tmp_path / "runtime" / "dlive_wal.jsonl"
    ).incomplete_transaction_ids() == ()


def test_operator_emergency_after_prepared_is_terminal_zero_send(tmp_path):
    calls = 0

    def interrupt_after_prepared():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("simulated_operator_emergency_stop")
        return time.time()

    receipt = run(tmp_path, now_fn=interrupt_after_prepared)
    client = FakeDLiveClient.instances[0]
    records = DLiveProofWal(tmp_path / "runtime" / "dlive_wal.jsonl").records
    assert receipt["status"] == "blocked"
    assert receipt["reason"] == "exception_or_emergency_before_apply_boundary"
    assert receipt["mutation_write_attempt_count"] == 0
    assert receipt["write_state_unknown"] is False
    assert client.sends == []
    assert records[-1]["phase"] == "blocked_no_send"
    assert DLiveProofWal(
        tmp_path / "runtime" / "dlive_wal.jsonl"
    ).incomplete_transaction_ids() == ()


def test_apply_mismatch_does_not_overwrite_external_value(tmp_path):
    FakeDLiveClient.apply_mode = "mismatch"
    receipt = run(tmp_path)
    client = FakeDLiveClient.instances[0]
    assert receipt["status"] == "manual_review_required"
    assert receipt["reason"] == "rollback_not_physically_verified"
    assert client.sends == [("apply", 5, -9.0)]
    assert receipt["write_state_unknown"] is True


def test_preapply_cas_change_blocks_with_zero_mutation_sends(tmp_path):
    FakeDLiveClient.change_on_fader_query = (2, -7.0)
    receipt = run(tmp_path)
    client = FakeDLiveClient.instances[0]
    assert receipt["status"] == "blocked"
    assert receipt["reason"] == "apply_identity_mute_or_preimage_cas_changed"
    assert receipt["mutation_write_attempt_count"] == 0
    assert client.sends == []
    assert DLiveProofWal(tmp_path / "runtime" / "dlive_wal.jsonl").records[-1]["phase"] == "blocked_no_send"


def test_target_above_configured_ceiling_blocks_before_wal_and_send(tmp_path):
    FakeDLiveClient.configured_ceiling_db = -12.0
    receipt = run(tmp_path)
    assert receipt["reason"] == "proof_target_above_configured_fader_ceiling"
    assert receipt["mutation_write_attempt_count"] == 0
    assert FakeDLiveClient.instances[0].sends == []
    assert not (tmp_path / "runtime" / "dlive_wal.jsonl").exists()


def test_rollback_cas_change_after_boundary_is_not_overwritten(tmp_path):
    FakeDLiveClient.change_on_fader_query = (5, -8.0)
    receipt = run(tmp_path)
    client = FakeDLiveClient.instances[0]
    assert receipt["status"] == "manual_review_required"
    assert client.sends == [("apply", 5, -9.0)]
    assert receipt["mutation_write_attempt_count"] == 1
    assert receipt["rollback_cas_mismatch_db"] == -8.0


def test_wal_rejects_tamper_and_sensitive_payload(tmp_path):
    path = tmp_path / "wal.jsonl"
    wal = DLiveProofWal(path)
    wal.append(transaction_id="tx", phase="blocked_no_send", payload={"reason": "test"})
    with pytest.raises(ValueError, match="sensitive"):
        wal.append(transaction_id="tx2", phase="blocked_no_send", payload={"password": "x"})

    record = json.loads(path.read_text(encoding="utf-8"))
    record["payload"]["reason"] = "tampered"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash_chain"):
        DLiveProofWal(path)


def test_wal_rejects_symlink_parent_and_hardlinked_file(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(OSError, match="path_component"):
        DLiveProofWal(linked / "wal.jsonl").append(
            transaction_id="tx",
            phase="blocked_no_send",
            payload={"reason": "test"},
        )

    original = tmp_path / "original.jsonl"
    original.write_text("", encoding="utf-8")
    hardlink = tmp_path / "hardlink.jsonl"
    os.link(original, hardlink)
    with pytest.raises(ValueError, match="regular_file"):
        DLiveProofWal(hardlink)


def test_incomplete_wal_blocks_new_hardware_contact(tmp_path):
    wal_path = tmp_path / "runtime" / "dlive_wal.jsonl"
    wal = DLiveProofWal(wal_path)
    wal.append(transaction_id="crashed", phase="apply_send_boundary", payload={"send": True})
    receipt = run(
        tmp_path,
        wal_path=wal_path,
        client_factory=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("client must not be constructed")
        ),
    )
    assert receipt["reason"] == "dlive_wal_recovery_required"
    assert receipt["incomplete_transaction_ids"] == ["crashed"]


def make_incomplete_wal(path, transaction_id="crashed"):
    public = approval().validate()
    wal = DLiveProofWal(path)
    wal.append(
        transaction_id=transaction_id,
        phase="prepared",
        payload={
            "approval_sha256": "sha256:" + "0" * 64,
            "endpoint": public["endpoint"],
            "expected_interface": "en7",
            "input_channel": 5,
            "expected_name": "Vocal",
            "operator_attestation": public["operator_attestation"],
            "channel_muted": True,
            "preimage_db": -6.0,
            "target_db": -9.0,
            "max_cut_db": 3.0,
            "apply_packet": {},
            "rollback_packet": {},
            "standing_live_authority": False,
        },
    )
    wal.append(
        transaction_id=transaction_id,
        phase="apply_send_boundary",
        payload={"send_may_have_occurred": True, "write_state_unknown": True},
    )
    return wal


def recover(tmp_path, **overrides):
    values = {
        "pa_outputs_contained": True,
        "input_confirmed_muted": True,
        "client_factory": FakeDLiveClient,
        "route_interface_resolver": lambda host: "en7",
        "wal_path": tmp_path / "runtime" / "dlive_wal.jsonl",
        "writer_lock_path": tmp_path / "runtime" / "console_writer.lock",
        "timeout_sec": 0.01,
    }
    values.update(overrides)
    return recover_dlive_fader_restore_proof("crashed", **values)


def test_recovery_restores_only_exact_wal_target(tmp_path):
    make_incomplete_wal(tmp_path / "runtime" / "dlive_wal.jsonl")
    FakeDLiveClient.initial_fader_db = -9.0
    receipt = recover(tmp_path)
    client = FakeDLiveClient.instances[0]
    assert receipt["status"] == "recovery_restored"
    assert receipt["reason"] == "exact_preimage_rollback_and_teardown_verified"
    assert client.sends == [("rollback", 5, -6.0)]
    assert client.fader_db == -6.0
    assert receipt["mutation_write_attempt_count"] == 1
    assert DLiveProofWal(tmp_path / "runtime" / "dlive_wal.jsonl").records[-1]["phase"] == "recovery_restored"


def test_recovery_observes_preimage_without_send(tmp_path):
    make_incomplete_wal(tmp_path / "runtime" / "dlive_wal.jsonl")
    FakeDLiveClient.initial_fader_db = -6.0
    receipt = recover(tmp_path)
    assert receipt["status"] == "recovery_restored"
    assert receipt["reason"] == "preimage_and_teardown_already_physically_restored"
    assert FakeDLiveClient.instances[0].sends == []
    assert receipt["live_write"] is False


def test_recovery_cas_mismatch_never_overwrites(tmp_path):
    make_incomplete_wal(tmp_path / "runtime" / "dlive_wal.jsonl")
    FakeDLiveClient.initial_fader_db = -7.5
    receipt = recover(tmp_path)
    assert receipt["status"] == "manual_review_required"
    assert receipt["reason"] == "recovery_cas_target_changed"
    assert FakeDLiveClient.instances[0].sends == []
    assert receipt["mutation_write_attempt_count"] == 0


def test_recovery_requires_fresh_physical_containment_before_contact(tmp_path):
    make_incomplete_wal(tmp_path / "runtime" / "dlive_wal.jsonl")
    receipt = recover(
        tmp_path,
        pa_outputs_contained=False,
        client_factory=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("client must not be constructed")
        ),
    )
    assert receipt["reason"] == "recovery_requires_physical_output_containment_and_muted_input"
    assert receipt["hardware_contacted"] is False
    assert receipt["mutation_write_attempt_count"] == 0
