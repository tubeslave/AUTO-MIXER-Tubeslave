from __future__ import annotations

import copy

import pytest

from supervised_family_proof_plan import (
    build_canonical_operation_bundle_from_prepared,
    canonical_bundle_scope,
    canonical_decimal_string,
    canonical_operation_bundle_bytes,
    canonical_operation_bundle_digest,
    validate_canonical_operation_bundle,
)


def _safety_binding():
    return {
        "adapter_revision": "wing-supervised-v1",
        "protocol_revision": "wing-fw-3.0.5",
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


def _bundle():
    return build_canonical_operation_bundle_from_prepared(
        bundle_kind="single_soundcheck_operation",
        operations=(
            {
                "operation_id": "op-1",
                "family_id": "eq_hpf",
                "write_kind": "eq",
                "channel": 1,
                "parameter": "eq_band_1_gain_db",
                "target_value": 1.25,
                "metadata": {"untrusted": "excluded"},
            },
        ),
        prepared_operations=(
            {
                "addresses": ("/ch/1/eq/1/g",),
                "write_count": 1,
                "writes": (
                    {
                        "address": "/ch/1/eq/1/g",
                        "target_value": 1.2500,
                        "tolerance": 0.05,
                    },
                ),
                "audible_state_reads": (
                    {
                        "address": "/ch/1/eq/on",
                        "expected_value": 1,
                        "tolerance": 0.0,
                    },
                ),
            },
        ),
        safety_binding=_safety_binding(),
    )


def test_decimal_canonicalization_is_finite_non_coercing_and_stable():
    assert canonical_decimal_string(1.2500) == "1.25"
    assert canonical_decimal_string(-0.0) == "0"
    assert canonical_decimal_string("100.000") == "100"
    with pytest.raises(ValueError):
        canonical_decimal_string(True)
    with pytest.raises(ValueError):
        canonical_decimal_string(float("nan"))


def test_bundle_is_server_prepared_strict_and_digest_stable():
    bundle = _bundle()
    assert bundle["operations"][0]["writes"][0] == {
        "address": "/ch/1/eq/1/g",
        "target_value": "1.25",
        "tolerance": "0.05",
    }
    assert "metadata" not in bundle["operations"][0]
    assert canonical_operation_bundle_bytes(bundle) == (
        canonical_operation_bundle_bytes(copy.deepcopy(bundle))
    )
    assert canonical_operation_bundle_digest(bundle).startswith("sha256:")
    assert canonical_bundle_scope(bundle) == (
        ("/ch/1/eq/1/g",),
        (("/ch/1/eq/1/g", '"1.25"'),),
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda bundle: bundle.update({"address": "/ch/2/eq/1/g"}),
        lambda bundle: bundle["operations"][0].update({"proof_run": True}),
        lambda bundle: bundle["operations"][0]["writes"][0].update(
            {"target_value": 1.25}
        ),
        lambda bundle: bundle["operations"][0]["writes"][0].update(
            {"tolerance": "0.050"}
        ),
        lambda bundle: bundle["safety_binding"].update(
            {"runtime_manifest_sha256": "2" * 64}
        ),
    ],
)
def test_bundle_near_misses_are_rejected(mutate):
    bundle = _bundle()
    mutate(bundle)
    with pytest.raises(ValueError):
        validate_canonical_operation_bundle(bundle)


def test_duplicate_write_address_and_client_supplied_prepared_shape_reject():
    with pytest.raises(ValueError, match="duplicate_address"):
        build_canonical_operation_bundle_from_prepared(
            bundle_kind="maintenance_dsp_reset_bundle",
            operations=(
                {
                    "operation_id": "one",
                    "family_id": "eq_hpf",
                    "write_kind": "eq",
                    "channel": 1,
                    "parameter": "eq-on",
                },
                {
                    "operation_id": "two",
                    "family_id": "eq_hpf",
                    "write_kind": "eq",
                    "channel": 1,
                    "parameter": "eq-on-again",
                },
            ),
            prepared_operations=(
                {
                    "addresses": ("/ch/1/eq/on",),
                    "write_count": 1,
                    "writes": (
                        {
                            "address": "/ch/1/eq/on",
                            "target_value": 0,
                            "tolerance": 0,
                        },
                    ),
                    "audible_state_reads": (),
                },
                {
                    "addresses": ("/ch/1/eq/on",),
                    "write_count": 1,
                    "writes": (
                        {
                            "address": "/ch/1/eq/on",
                            "target_value": 0,
                            "tolerance": 0,
                        },
                    ),
                    "audible_state_reads": (),
                },
            ),
            safety_binding=_safety_binding(),
        )

    prepared = {
        "addresses": ("/ch/1/eq/on",),
        "write_count": 1,
        "writes": (
            {
                "address": "/ch/1/eq/on",
                "target_value": 0,
                "tolerance": 0,
                "client_address": "/main/1/fdr",
            },
        ),
        "audible_state_reads": (),
    }
    with pytest.raises(ValueError, match="fields_mismatch"):
        build_canonical_operation_bundle_from_prepared(
            bundle_kind="single_soundcheck_operation",
            operations=(
                {
                    "operation_id": "one",
                    "family_id": "eq_hpf",
                    "write_kind": "eq",
                    "channel": 1,
                    "parameter": "eq-on",
                },
            ),
            prepared_operations=(prepared,),
            safety_binding=_safety_binding(),
        )
