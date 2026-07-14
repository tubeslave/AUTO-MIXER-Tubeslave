from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from m110_m006_restore import (
    M110_M006_FIXTURE_RELATIVE_PATH,
    M110_M006_FIXTURE_SHA256,
    M110_M006_RESTORE_STAGES,
    M110_M006_SOURCE_EVENT_SHA256,
    M110_M006_SOURCE_TRANSACTION_ID,
    build_m110_m006_restore_plan,
    load_m110_m006_fixture,
    m110_m006_restore_stage_operations,
)
from supervised_console_transaction import (
    AdapterStepResult,
    ConsoleTransactionOperation,
)
from supervised_family_proof_plan import (
    build_canonical_operation_bundle_from_prepared,
    canonical_bundle_scope,
)
from supervised_proof_capability import M110_M006_RESTORE_BUNDLE_KIND
from wing_supervised_console_adapters import (
    build_wing_supervised_family_adapters,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _operation(raw):
    return ConsoleTransactionOperation(
        operation_id=raw["operation_id"],
        family_id=raw["family_id"],
        write_kind=raw["write_kind"],
        target_value=raw["target_value"],
        channel=raw["channel"],
        parameter=raw["parameter"],
        metadata=raw["metadata"],
    )


def _safety_binding():
    return {
        "adapter_revision": "m110-m006-exact-preimage-restore.v1",
        "protocol_revision": "wing-osc-fw3.0.5",
        "runtime_instance_nonce": "1" * 32,
        "runtime_manifest_sha256": "sha256:" + "2" * 64,
        "source_head": "3" * 40,
        "source_tree": "4" * 40,
        "console_serial": "0100Q1V0604AAE214",
        "firmware_version": "3.0.5",
        "route_proof_id": "m110-route-1",
        "route_artifact_sha256": "sha256:" + "5" * 64,
        "capture_hardware_uid": "usb:5015:1291:214",
        "connection_generation": 6,
        "transport_generation": 7,
        "emergency_generation": 8,
    }


class PhysicalWing:
    def __init__(self, values):
        self.values = dict(values)
        self.sent = []

    def query_parameter(self, address, _timeout):
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


def test_m006_contract_is_hash_bound_to_exact_real_transaction():
    path = REPO_ROOT / M110_M006_FIXTURE_RELATIVE_PATH
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        M110_M006_FIXTURE_SHA256
    )
    fixture = load_m110_m006_fixture()
    assert fixture["operation_count"] == 51
    assert fixture["source"]["transaction_id"] == (
        M110_M006_SOURCE_TRANSACTION_ID
    )
    assert fixture["source"]["event_line_sha256"] == (
        M110_M006_SOURCE_EVENT_SHA256
    )


def test_m006_public_loaders_do_not_expose_mutable_cached_authority():
    fixture = load_m110_m006_fixture()
    fixture["operations"][0]["target_value"] = "tampered"
    assert load_m110_m006_fixture()["operations"][0]["target_value"] != (
        "tampered"
    )

    stages = m110_m006_restore_stage_operations()
    stages["foundation_level"][0]["metadata"][
        "m110_source_transaction_id"
    ] = "tampered"
    assert (
        m110_m006_restore_stage_operations()["foundation_level"][0][
            "metadata"
        ]["m110_source_transaction_id"]
        == M110_M006_SOURCE_TRANSACTION_ID
    )


def test_m006_stages_have_exact_finite_scope_and_changed_address_counts():
    expected = {
        "foundation_level": (23, 33, 19),
        "tonal_dynamics": (14, 63, 52),
        "spatial_balance": (14, 14, 6),
    }
    stages = m110_m006_restore_stage_operations()
    assert tuple(stages) == M110_M006_RESTORE_STAGES
    total_changed = set()
    for stage, (operation_count, scope_count, changed_count) in expected.items():
        operations = [_operation(raw) for raw in stages[stage]]
        adapters = build_wing_supervised_family_adapters(
            object(),
            fixed_preimage_restore_authorized=True,
        )
        prepared = [
            adapters[operation.family_id].prepare(operation)
            for operation in operations
        ]
        bundle = build_canonical_operation_bundle_from_prepared(
            bundle_kind=M110_M006_RESTORE_BUNDLE_KIND,
            operations=[
                {
                    "operation_id": operation.operation_id,
                    "family_id": operation.family_id,
                    "write_kind": operation.write_kind,
                    "channel": operation.channel,
                    "parameter": operation.parameter,
                }
                for operation in operations
            ],
            prepared_operations=prepared,
            safety_binding=_safety_binding(),
        )
        scope, _targets = canonical_bundle_scope(bundle)
        changed = {
            address
            for operation in operations
            for address in operation.metadata["m110_changed_addresses"]
        }
        assert len(operations) == operation_count
        assert len(scope) == len(set(scope)) == scope_count
        assert len(changed) == changed_count
        assert "/main/1/mute" not in scope
        total_changed.update(changed)
    assert len(total_changed) == 77


def test_m006_plan_role_changes_identity_but_not_reviewed_operations():
    proof = build_m110_m006_restore_plan(
        stage="foundation_level",
        role="proof",
        proposal_nonce="a" * 32,
    )
    commit = build_m110_m006_restore_plan(
        stage="foundation_level",
        role="commit",
        proposal_nonce="b" * 32,
    )
    assert proof["transaction_id"] != commit["transaction_id"]
    assert proof["operations"] == commit["operations"]
    assert proof["atomic_group_commit_required"] is True


def _tonal_eq_operation():
    raw = next(
        item
        for item in m110_m006_restore_stage_operations()["tonal_dynamics"]
        if item["write_kind"] == "eq"
    )
    return _operation(raw)


@pytest.mark.asyncio
async def test_fixed_restore_requires_private_grant_and_exact_cas_state():
    operation = _tonal_eq_operation()
    expected = dict(operation.metadata["m110_expected_current_values"])
    expected["/main/1/mute"] = 1
    expected[f"/ch/{operation.channel}/eq/on"] = 0
    wing = PhysicalWing(expected)

    unauthorized = build_wing_supervised_family_adapters(wing)[
        operation.family_id
    ]
    unauthorized_snapshot = await unauthorized.snapshot(operation)
    assert unauthorized.validate_snapshot(
        operation,
        unauthorized_snapshot,
    ) == "m110_fixed_preimage_restore_grant_required"

    authorized = build_wing_supervised_family_adapters(
        wing,
        fixed_preimage_restore_authorized=True,
        write_authority_current=lambda: True,
    )[operation.family_id]
    snapshot = await authorized.snapshot(operation)
    assert authorized.validate_snapshot(operation, snapshot) is None
    first_write = authorized.prepare(operation)["writes"][0]
    wing.values[first_write["address"]] = 12345.0
    changed_snapshot = await authorized.snapshot(operation)
    assert authorized.validate_snapshot(
        operation,
        changed_snapshot,
    ) == "m110_fixed_preimage_restore_cas_mismatch"


@pytest.mark.asyncio
@pytest.mark.parametrize("guard", ["main", "processor"])
async def test_fixed_restore_guard_loss_blocks_all_forward_sends(guard):
    operation = _tonal_eq_operation()
    values = dict(operation.metadata["m110_expected_current_values"])
    values["/main/1/mute"] = 1
    values[f"/ch/{operation.channel}/eq/on"] = 0
    wing = PhysicalWing(values)
    adapter = build_wing_supervised_family_adapters(
        wing,
        fixed_preimage_restore_authorized=True,
        write_authority_current=lambda: True,
    )[operation.family_id]
    snapshot = await adapter.snapshot(operation)
    if guard == "main":
        wing.values["/main/1/mute"] = 0
    else:
        wing.values[f"/ch/{operation.channel}/eq/on"] = 1

    result = await adapter.apply(operation, snapshot)

    assert result.success is False
    assert result.reason == "m110_fixed_preimage_restore_guard_changed"
    assert wing.sent == []


@pytest.mark.asyncio
async def test_fixed_restore_guard_loss_blocks_all_rollback_sends():
    operation = _tonal_eq_operation()
    old_values = dict(operation.metadata["m110_expected_current_values"])
    old_values["/main/1/mute"] = 1
    old_values[f"/ch/{operation.channel}/eq/on"] = 0
    wing = PhysicalWing(old_values)
    adapter = build_wing_supervised_family_adapters(
        wing,
        fixed_preimage_restore_authorized=True,
        write_authority_current=lambda: True,
    )[operation.family_id]
    snapshot = await adapter.snapshot(operation)
    writes = adapter.prepare(operation)["writes"]
    for write in writes:
        wing.values[write["address"]] = write["target_value"]
    wing.values["/main/1/mute"] = 0
    apply_result = AdapterStepResult(
        True,
        True,
        "synthetic_applied",
        {
            "writes": [
                {
                    "address": write["address"],
                    "value": write["target_value"],
                    "sent": True,
                }
                for write in writes
            ],
            "verified_values": {
                write["address"]: write["target_value"] for write in writes
            },
        },
    )

    result = await adapter.rollback(operation, snapshot, apply_result)

    assert result.success is False
    assert result.reason == (
        "m110_fixed_preimage_restore_rollback_guard_changed"
    )
    assert wing.sent == []
