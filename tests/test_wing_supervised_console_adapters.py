from __future__ import annotations

import asyncio
import threading

import pytest

from supervised_console_transaction import (
    AdapterStepResult,
    AudioRecheckResult,
    ConsoleTransactionOperation,
)
from wing_supervised_console_adapters import (
    WingSupervisedFamilyAdapter,
    audible_state_reads_for_operation,
    build_wing_supervised_family_adapters,
    parameter_writes_for_operation,
)


class FakeWing:
    def __init__(self):
        self.values = {"/ch/3/eq/on": 1}
        self.sent = []
        self.query_status = "received"
        self.fail_on_send_number = None
        self.raise_on_send_number = None
        self.after_send = None
        self.block_on_send_number = None
        self.block_started = threading.Event()
        self.block_release = threading.Event()

    def query_parameter(self, address, timeout):
        return {
            "status": self.query_status,
            "address": address,
            "value": self.values.get(address),
            "confirmed_round_trips": 2 if self.query_status == "received" else 0,
        }

    def send(self, address, value):
        self.sent.append((address, value))
        send_number = len(self.sent)
        if send_number == self.raise_on_send_number:
            raise RuntimeError("synthetic send failure")
        if send_number == self.fail_on_send_number:
            return False
        if send_number == self.block_on_send_number:
            self.block_started.set()
            self.block_release.wait(timeout=2.0)
        self.values[address] = value
        if callable(self.after_send):
            self.after_send(send_number, address, value)
        return True


def operation(family, kind, target, *, parameter="", channel=3):
    return ConsoleTransactionOperation(
        operation_id=f"{family}:{kind}",
        family_id=family,
        write_kind=kind,
        target_value=target,
        channel=channel,
        parameter=parameter,
    )


def test_parameter_translation_covers_all_generated_soundcheck_families():
    operations = [
        operation("gain_trim", "gain", -4.0),
        operation("fader", "fader", -12.0),
        operation("mute", "mute", True),
        operation("eq_hpf", "hpf", {"freq_hz": 90.0, "enabled": True}),
        operation("eq_hpf", "eq", {"band": 2, "freq_hz": 300.0, "gain_db": -2.0, "q": 1.1}),
        operation("gate_expander", "gate", {"threshold_db": -40.0, "range_db": 20.0, "attack_ms": 5.0, "hold_ms": 30.0, "release_ms": 120.0, "enabled": True}),
        operation("dynamics_compressor", "compressor", {"threshold_db": -18.0, "ratio": 3.0, "attack_ms": 20.0, "release_ms": 100.0, "makeup_db": 0.0, "enabled": True}),
        operation("pan_phase_delay", "pan", -25.0),
        operation("pan_phase_delay", "polarity", True),
        operation("pan_phase_delay", "delay", {"delay_ms": 1.2, "enabled": True}),
        operation("fx_sends_inserts", "send", {"send_bus": 2, "level_db": -18.0}),
        operation("bus_group_dca_matrix", "bus", {"bus_id": 1, "target_db": -6.0}, parameter="bus_1_fader_db", channel=None),
        operation("bus_group_dca_matrix", "dca", {"dca_id": 1, "target_db": -3.0}, parameter="dca_1_fader_db", channel=None),
        operation("master_output", "master", {"main_id": 1, "target_db": -8.0}, parameter="main_1_fader_db", channel=None),
    ]

    translated = [parameter_writes_for_operation(item) for item in operations]

    assert all(writes for writes in translated)
    assert [write.address for write in translated[4]] == [
        "/ch/3/eq/2f",
        "/ch/3/eq/2g",
        "/ch/3/eq/2q",
    ]
    assert [
        read.address for read in audible_state_reads_for_operation(operations[4])
    ] == ["/ch/3/eq/on"]
    assert translated[10][0].address == "/ch/3/send/2/lvl"
    assert translated[-1][0].address == "/main/1/fdr"


def test_m110_gate_processor_uses_exact_wing_fw305_enum_tokens():
    op = operation(
        "gate_expander",
        "gate",
        {
            "m110_processor_block": "gate",
            "model": "GATE",
            "mix_pct": 100.0,
            "gain_db": 0.0,
            "threshold_db": -40.0,
            "range_db": 20.0,
            "attack_ms": 5.0,
            "hold_ms": 30.0,
            "release_ms": 120.0,
            "accent_pct": 0.0,
            "ratio": "GATE",
            "sidechain": {
                "type": "OFF",
                "freq_hz": 1000.0,
                "q": 1.0,
                "source": "SELF",
                "tap": "IN",
            },
            "enabled": True,
        },
    )

    by_address = {
        write.address: write.value
        for write in parameter_writes_for_operation(op)
    }

    assert by_address["/ch/3/gate/ratio"] == "GATE"
    assert by_address["/ch/3/gatesc/type"] == "OFF"
    assert by_address["/ch/3/gatesc/src"] == "SELF"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("op", "old_values"),
    [
        (operation("fader", "fader", -12.0), {"/ch/3/fdr": -8.0}),
        (
            operation("gain_trim", "trim", -4.0),
            {"/ch/3/in/set/trim": 0.0},
        ),
        (
            operation(
                "fx_sends_inserts",
                "send",
                {"send_bus": 2, "level_db": -18.0},
            ),
            {"/ch/3/send/2/lvl": -12.0},
        ),
        (
            operation("dynamics_compressor", "compressor", -2.0),
            {"/ch/3/dyn/gain": 0.0},
        ),
        (
            operation(
                "bus_group_dca_matrix",
                "dca",
                {"dca_id": 1, "target_db": -6.0},
                parameter="dca_1_fader_db",
                channel=None,
            ),
            {"/dca/1/fdr": -3.0},
        ),
        (
            operation(
                "master_output",
                "master",
                {"main_id": 1, "target_db": -8.0},
                parameter="main_1_fader_db",
                channel=None,
            ),
            {"/main/1/fdr": -4.0},
        ),
        (operation("mute", "mute", True), {"/ch/3/mute": 0}),
        (
            operation(
                "eq_hpf",
                "eq",
                {
                    "band": 2,
                    "freq_hz": 300.0,
                    "gain_db": -2.0,
                    "q": 1.0,
                },
            ),
            {
                "/ch/3/eq/2f": 300.0,
                "/ch/3/eq/2g": 0.0,
                "/ch/3/eq/2q": 1.0,
            },
        ),
    ],
)
async def test_audio_failure_retention_proves_only_monotonic_targets_read_only(
    op,
    old_values,
):
    wing = FakeWing()
    writes = parameter_writes_for_operation(op)
    reads = audible_state_reads_for_operation(op)
    for write in writes:
        wing.values[write.address] = old_values.get(
            write.address,
            write.value,
        )
    for read in reads:
        wing.values[read.address] = read.value
    adapter = WingSupervisedFamilyAdapter(op.family_id, wing)
    snapshot = await adapter.snapshot(op)
    for write in writes:
        wing.values[write.address] = write.value

    result = await adapter.assess_audio_failure_retention(
        op,
        snapshot,
        AdapterStepResult(
            True,
            True,
            "applied",
            {
                "writes": [
                    {
                        "address": write.address,
                        "value": write.value,
                        "sent": True,
                    }
                    for write in writes
                ]
            },
        ),
    )

    assert result.success is True
    assert result.physical_verified is True
    assert result.evidence["retention_eligible"] is True
    assert result.evidence["target_physical_verified"] is True
    assert result.evidence["write_attempted"] is False
    assert wing.sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("op", "old_overrides"),
    [
        (operation("pan_phase_delay", "pan", -10.0), {"/ch/3/pan": 0.0}),
        (
            operation(
                "eq_hpf",
                "eq",
                {
                    "band": 2,
                    "freq_hz": 300.0,
                    "gain_db": -2.0,
                    "q": 1.0,
                },
            ),
            {
                "/ch/3/eq/2f": 200.0,
                "/ch/3/eq/2g": 0.0,
                "/ch/3/eq/2q": 1.0,
            },
        ),
        (
            operation(
                "dynamics_compressor",
                "compressor",
                {
                    "threshold_db": -18.0,
                    "ratio": 3.0,
                    "attack_ms": 20.0,
                    "release_ms": 100.0,
                    "makeup_db": -2.0,
                    "enabled": True,
                },
            ),
            {"/ch/3/dyn/thr": -12.0, "/ch/3/dyn/gain": 0.0},
        ),
    ],
)
async def test_audio_failure_retention_rejects_spatial_or_multi_parameter_changes(
    op,
    old_overrides,
):
    wing = FakeWing()
    writes = parameter_writes_for_operation(op)
    reads = audible_state_reads_for_operation(op)
    for write in writes:
        wing.values[write.address] = old_overrides.get(
            write.address,
            write.value,
        )
    for read in reads:
        wing.values[read.address] = read.value
    adapter = WingSupervisedFamilyAdapter(op.family_id, wing)
    snapshot = await adapter.snapshot(op)
    for write in writes:
        wing.values[write.address] = write.value

    result = await adapter.assess_audio_failure_retention(
        op,
        snapshot,
        AdapterStepResult(True, True, "applied"),
    )

    assert result.success is False
    assert result.evidence["retention_eligible"] is False
    assert result.evidence["write_attempted"] is False
    assert wing.sent == []


@pytest.mark.asyncio
async def test_audio_failure_retention_readback_failure_never_writes():
    op = operation("fader", "fader", -12.0)
    wing = FakeWing()
    wing.values["/ch/3/fdr"] = -8.0
    adapter = WingSupervisedFamilyAdapter(op.family_id, wing)
    snapshot = await adapter.snapshot(op)
    wing.values["/ch/3/fdr"] = -12.0
    wing.query_status = "timeout"

    result = await adapter.assess_audio_failure_retention(
        op,
        snapshot,
        AdapterStepResult(True, True, "applied"),
    )

    assert result.success is False
    assert result.evidence["retention_eligible"] is True
    assert result.evidence["target_physical_verified"] is False
    assert wing.sent == []


@pytest.mark.asyncio
async def test_processor_enable_runs_safety_immediately_before_and_after_enable():
    wing = FakeWing()
    op = ConsoleTransactionOperation(
        operation_id="m110:hpf:ch3",
        family_id="eq_hpf",
        write_kind="hpf",
        target_value={
            "m110_processor_block": "hpf",
            "freq_hz": 90.0,
            "slope_db_per_octave": "12",
            "enabled": True,
        },
        channel=3,
        parameter="m110_complete_hpf_block",
        metadata={"m110_processor_enable_bundle": True},
    )
    writes = parameter_writes_for_operation(op)
    for write in writes:
        wing.values[write.address] = 0
    events = []

    async def before_enable(operation):
        events.append(("before_enable", len(wing.sent), operation.operation_id))
        return True

    async def audio_recheck(operation, snapshot, apply_evidence):
        events.append(("post_enable", len(wing.sent), operation.operation_id))
        return AudioRecheckResult("passed")

    adapter = WingSupervisedFamilyAdapter(
        "eq_hpf",
        wing,
        audio_recheck=audio_recheck,
        write_authority_current=lambda: True,
        before_processor_enable=before_enable,
        processor_enable_authorized=True,
    )
    snapshot = await adapter.snapshot(op)

    result = await adapter.apply(op, snapshot)

    assert result.success is True
    assert events == [
        ("before_enable", len(writes) - 1, op.operation_id),
        ("post_enable", len(writes), op.operation_id),
    ]
    assert wing.sent[-1] == (writes[-1].address, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("op", "enable_address", "mutation_addresses"),
    [
        (
            operation(
                "bus_group_dca_matrix",
                "bus",
                {
                    "bus_id": 1,
                    "band": 2,
                    "freq_hz": 300.0,
                    "gain_db": -2.0,
                    "q": 1.0,
                },
                parameter="bus_1_eq_band_2",
                channel=None,
            ),
            "/bus/1/eq/on",
            ["/bus/1/eq/2f", "/bus/1/eq/2g", "/bus/1/eq/2q"],
        ),
        (
            operation(
                "bus_group_dca_matrix",
                "bus",
                {"bus_id": 1, "makeup_db": -2.0},
                parameter="bus_1_compressor_makeup_db",
                channel=None,
            ),
            "/bus/1/dyn/on",
            ["/bus/1/dyn/gain"],
        ),
    ],
)
async def test_bus_tonal_enable_is_read_only_and_off_state_blocks_before_send(
    op,
    enable_address,
    mutation_addresses,
):
    wing = FakeWing()
    writes = parameter_writes_for_operation(op)
    reads = audible_state_reads_for_operation(op)
    assert [write.address for write in writes] == mutation_addresses
    assert [read.address for read in reads] == [enable_address]
    for write in writes:
        wing.values[write.address] = write.value
    wing.values[enable_address] = 0
    adapter = WingSupervisedFamilyAdapter(op.family_id, wing)
    snapshot = await adapter.snapshot(op)

    applied = await adapter.apply(op, snapshot)

    assert snapshot["physical_verified"] is True
    assert snapshot["audible_state_verified"] is False
    assert applied.reason == "processing_enable_state_change_requires_separate_approval"
    assert wing.sent == []


def test_eq_boost_is_rejected_by_cut_only_live_adapter_contract():
    op = operation(
        "eq_hpf",
        "eq",
        {"band": 1, "freq_hz": 200.0, "gain_db": 2.0, "q": 1.0},
    )

    with pytest.raises(ValueError, match="eq_gain_db_out_of_range"):
        parameter_writes_for_operation(op)


@pytest.mark.parametrize(
    ("op", "reason"),
    [
        (
            operation(
                "eq_hpf",
                "eq",
                {"band": 2.9, "freq_hz": 200.0, "gain_db": -2.0, "q": 1.0},
            ),
            "invalid_eq_band",
        ),
        (
            operation(
                "fx_sends_inserts",
                "send",
                {"send_bus": True, "level_db": -18.0},
            ),
            "invalid_send_bus",
        ),
        (
            operation(
                "bus_group_dca_matrix",
                "dca",
                {"dca_id": 1.5, "target_db": -3.0},
                parameter="dca_1_fader_db",
                channel=None,
            ),
            "invalid_dca",
        ),
        (
            operation(
                "master_output",
                "master",
                {"main_id": False, "target_db": -8.0},
                parameter="main_1_fader_db",
                channel=None,
            ),
            "invalid_main",
        ),
    ],
)
def test_fractional_and_boolean_identifiers_are_rejected(op, reason):
    with pytest.raises(ValueError, match=reason):
        parameter_writes_for_operation(op)


def test_disabled_delay_still_writes_and_verifies_the_delay_value():
    op = operation(
        "pan_phase_delay",
        "delay",
        {"delay_ms": 1.2, "enabled": False},
    )

    writes = parameter_writes_for_operation(op)

    assert [write.address for write in writes] == [
        "/ch/3/in/set/dlymode",
        "/ch/3/in/set/dly",
        "/ch/3/in/set/dlyon",
    ]
    assert writes[1].value == 1.2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("family", "kind", "target", "old_values", "reason"),
    [
        (
            "eq_hpf",
            "hpf",
            {"freq_hz": 2000.0, "enabled": False},
            {"/ch/3/flt/lcf": 20.0, "/ch/3/flt/lc": 0},
            "hpf_frequency_delta_exceeds_one_octave",
        ),
        (
            "dynamics_compressor",
            "compressor",
            {
                "threshold_db": -18.0,
                "ratio": 100.0,
                "attack_ms": 20.0,
                "release_ms": 100.0,
                "makeup_db": 0.0,
                "enabled": False,
            },
            {
                "/ch/3/dyn/thr": -18.0,
                "/ch/3/dyn/ratio": "1.0",
                "/ch/3/dyn/att": 20.0,
                "/ch/3/dyn/rel": 100.0,
                "/ch/3/dyn/gain": 0.0,
                "/ch/3/dyn/on": 0,
            },
            "compressor_ratio_delta_exceeds_2",
        ),
    ],
)
async def test_extreme_processing_deltas_block_before_send(
    family,
    kind,
    target,
    old_values,
    reason,
):
    wing = FakeWing()
    wing.values.update(old_values)
    adapter = WingSupervisedFamilyAdapter(family, wing)
    op = operation(family, kind, target)
    snapshot = await adapter.snapshot(op)

    applied = await adapter.apply(op, snapshot)

    assert applied.success is False
    assert applied.reason == reason
    assert wing.sent == []


@pytest.mark.asyncio
async def test_snapshot_apply_verify_and_audio_recheck_use_physical_queries():
    wing = FakeWing()
    wing.values["/ch/3/fdr"] = -12.0

    async def audio(*_args):
        return AudioRecheckResult("passed", evidence={"true_peak_dbtp": -6.0})

    adapter = WingSupervisedFamilyAdapter("fader", wing, audio_recheck=audio)
    op = operation("fader", "fader", -14.0)

    snapshot = await adapter.snapshot(op)
    applied = await adapter.apply(op, snapshot)
    verified = await adapter.verify(op, snapshot, applied)
    checked = await adapter.audio_recheck(op, snapshot, applied)

    assert snapshot["physical_verified"] is True
    assert snapshot["values"] == {"/ch/3/fdr": -12.0}
    assert verified.success is True
    assert verified.physical_verified is True
    assert checked.passed is True


@pytest.mark.parametrize(
    "query_result",
    [
        {
            "status": "received",
            "address": "/ch/48/fdr",
            "value": -14.0,
            "confirmed_round_trips": 2,
        },
        {
            "status": "received",
            "value": -14.0,
            "confirmed_round_trips": 2,
        },
        {
            "status": "received",
            "address": "/ch/3/fdr",
            "value": -14.0,
            "confirmed_round_trips": 2.9,
        },
        {
            "status": "received",
            "address": "/ch/3/fdr",
            "value": -14.0,
            "confirmed_round_trips": "2",
        },
        {
            "status": "received",
            "address": "/ch/3/fdr",
            "value": -14.0,
            "confirmed_round_trips": True,
        },
        {
            "status": "received",
            "address": "/ch/3/fdr",
            "value": None,
            "confirmed_round_trips": 2,
        },
        {
            "status": "received",
            "address": "/ch/3/fdr",
            "value": float("nan"),
            "confirmed_round_trips": 2,
        },
        "invalid-response",
    ],
    ids=[
        "wrong-address",
        "missing-address",
        "fractional-round-trips",
        "string-round-trips",
        "boolean-round-trips",
        "missing-preimage",
        "non-finite-preimage",
        "non-mapping-response",
    ],
)
@pytest.mark.asyncio
async def test_physical_readback_contract_fails_closed_for_every_adapter_phase(
    query_result,
):
    wing = FakeWing()
    wing.query_parameter = lambda _address, _timeout: query_result
    adapter = WingSupervisedFamilyAdapter("fader", wing)
    op = operation("fader", "fader", -14.0)
    trusted_snapshot = {
        "physical_verified": True,
        "values": {"/ch/3/fdr": -12.0},
    }
    apply_result = AdapterStepResult(
        True,
        False,
        "wing_write_sent",
        {"writes": [{"address": "/ch/3/fdr", "sent": True}]},
    )

    snapshot = await adapter.snapshot(op)
    verified = await adapter.verify(op, trusted_snapshot, apply_result)
    rolled_back = await adapter.rollback(op, trusted_snapshot, apply_result)

    assert snapshot["physical_verified"] is False
    assert snapshot["values"] == {}
    assert verified.success is False
    assert verified.physical_verified is False
    assert rolled_back.success is False
    assert rolled_back.physical_verified is False
    assert rolled_back.reason == "rollback_cas_read_failed"
    assert wing.sent == []


@pytest.mark.asyncio
async def test_rollback_is_compare_and_swap_and_preserves_operator_change():
    wing = FakeWing()
    wing.values["/ch/3/fdr"] = -12.0
    adapter = WingSupervisedFamilyAdapter("fader", wing)
    op = operation("fader", "fader", -14.0)
    snapshot = await adapter.snapshot(op)
    applied = await adapter.apply(op, snapshot)
    wing.values["/ch/3/fdr"] = -9.0

    result = await adapter.rollback(op, snapshot, applied)

    assert result.success is False
    assert result.reason == "rollback_cas_target_changed"
    assert wing.values["/ch/3/fdr"] == -9.0


@pytest.mark.asyncio
async def test_rollback_restores_and_physically_verifies_original_values():
    wing = FakeWing()
    wing.values["/ch/3/fdr"] = -12.0
    adapter = WingSupervisedFamilyAdapter("fader", wing)
    op = operation("fader", "fader", -14.0)
    snapshot = await adapter.snapshot(op)
    applied = await adapter.apply(op, snapshot)

    result = await adapter.rollback(op, snapshot, applied)

    assert result.success is True
    assert result.physical_verified is True
    assert wing.values["/ch/3/fdr"] == -12.0
    assert result.evidence["initial_compare_queries"]["/ch/3/fdr"]["address"] == "/ch/3/fdr"
    assert result.evidence["per_address_cas_queries"][0]["address"] == "/ch/3/fdr"
    assert result.evidence["per_address_cas_queries"][0]["query"]["address"] == "/ch/3/fdr"


@pytest.mark.asyncio
async def test_eq_frequency_accepts_wing_log_scale_quantization_and_uses_verified_cas_anchor():
    wing = FakeWing()
    wing.values.update({
        "/ch/3/eq/2f": 300.0,
        "/ch/3/eq/2g": -2.0,
        "/ch/3/eq/2q": 1.0,
    })
    adapter = WingSupervisedFamilyAdapter("eq_hpf", wing)
    op = operation("eq_hpf", "eq", {"band": 2, "freq_hz": 375.0, "gain_db": -5.0, "q": 2.0})
    snapshot = await adapter.snapshot(op)
    applied = await adapter.apply(op, snapshot)
    wing.values["/ch/3/eq/2f"] = 374.0286865234375

    verified = await adapter.verify(op, snapshot, applied)
    anchor = AdapterStepResult(
        applied.success,
        applied.physical_verified,
        applied.reason,
        {**applied.evidence, "verified_values": dict(verified.evidence["values"])},
    )
    rolled_back = await adapter.rollback(op, snapshot, anchor)

    assert verified.success is True
    assert rolled_back.success is True
    assert wing.values["/ch/3/eq/2f"] == 300.0


@pytest.mark.asyncio
async def test_partial_multi_address_send_failure_restores_first_write():
    wing = FakeWing()
    wing.values.update({
        "/ch/3/eq/2f": 300.0,
        "/ch/3/eq/2g": -2.0,
        "/ch/3/eq/2q": 1.0,
    })
    adapter = WingSupervisedFamilyAdapter("eq_hpf", wing)
    op = operation(
        "eq_hpf",
        "eq",
        {"band": 2, "freq_hz": 375.0, "gain_db": -5.0, "q": 2.0},
    )
    snapshot = await adapter.snapshot(op)
    wing.fail_on_send_number = 2

    applied = await adapter.apply(op, snapshot)
    wing.fail_on_send_number = None
    rolled_back = await adapter.rollback(op, snapshot, applied)

    assert applied.success is False
    assert applied.reason == "wing_send_failed"
    assert rolled_back.success is True
    assert rolled_back.physical_verified is True
    assert wing.values == snapshot["values"]
    assert rolled_back.evidence["restore_addresses"] == ["/ch/3/eq/2f"]


@pytest.mark.asyncio
async def test_send_exception_treats_write_state_as_unknown_and_restores_safely():
    wing = FakeWing()
    wing.values.update({
        "/ch/3/eq/2f": 300.0,
        "/ch/3/eq/2g": -2.0,
        "/ch/3/eq/2q": 1.0,
    })
    adapter = WingSupervisedFamilyAdapter("eq_hpf", wing)
    op = operation(
        "eq_hpf",
        "eq",
        {"band": 2, "freq_hz": 375.0, "gain_db": -5.0, "q": 2.0},
    )
    snapshot = await adapter.snapshot(op)
    wing.raise_on_send_number = 2

    applied = await adapter.apply(op, snapshot)
    wing.raise_on_send_number = None
    rolled_back = await adapter.rollback(op, snapshot, applied)

    assert applied.success is False
    assert applied.reason == "wing_send_exception"
    assert applied.evidence["write_state_unknown"] is True
    assert rolled_back.success is True
    assert wing.values == snapshot["values"]


@pytest.mark.asyncio
async def test_apply_cas_preserves_operator_change_after_first_address_send():
    wing = FakeWing()
    wing.values.update({
        "/ch/3/eq/2f": 300.0,
        "/ch/3/eq/2g": -2.0,
        "/ch/3/eq/2q": 1.0,
    })
    adapter = WingSupervisedFamilyAdapter("eq_hpf", wing)
    op = operation(
        "eq_hpf",
        "eq",
        {"band": 2, "freq_hz": 375.0, "gain_db": -5.0, "q": 2.0},
    )
    snapshot = await adapter.snapshot(op)

    def operator_change(send_number, _address, _value):
        if send_number == 1:
            wing.values["/ch/3/eq/2g"] = -7.0

    wing.after_send = operator_change
    applied = await adapter.apply(op, snapshot)
    wing.after_send = None
    rolled_back = await adapter.rollback(op, snapshot, applied)

    assert applied.success is False
    assert applied.reason == "apply_cas_target_changed"
    assert wing.values["/ch/3/eq/2f"] == 300.0
    assert wing.values["/ch/3/eq/2g"] == -7.0
    assert rolled_back.success is False
    assert "/ch/3/eq/2g" in rolled_back.evidence["cas_mismatches"]


@pytest.mark.asyncio
async def test_send_exception_does_not_claim_never_attempted_later_address():
    wing = FakeWing()
    wing.values.update({
        "/ch/3/eq/2f": 300.0,
        "/ch/3/eq/2g": -2.0,
        "/ch/3/eq/2q": 1.0,
    })
    adapter = WingSupervisedFamilyAdapter("eq_hpf", wing)
    op = operation(
        "eq_hpf",
        "eq",
        {"band": 2, "freq_hz": 375.0, "gain_db": -5.0, "q": 2.0},
    )
    snapshot = await adapter.snapshot(op)
    wing.raise_on_send_number = 2
    applied = await adapter.apply(op, snapshot)
    wing.raise_on_send_number = None
    wing.values["/ch/3/eq/2q"] = 2.0

    rolled_back = await adapter.rollback(op, snapshot, applied)

    assert applied.evidence["writes"][-1]["sent"] is None
    assert wing.values["/ch/3/eq/2f"] == 300.0
    assert wing.values["/ch/3/eq/2q"] == 2.0
    assert "/ch/3/eq/2q" not in rolled_back.evidence["restore_addresses"]


@pytest.mark.asyncio
async def test_rollback_send_failure_continues_other_safe_addresses():
    wing = FakeWing()
    wing.values.update({
        "/ch/3/eq/2f": 300.0,
        "/ch/3/eq/2g": -2.0,
        "/ch/3/eq/2q": 1.0,
    })
    adapter = WingSupervisedFamilyAdapter("eq_hpf", wing)
    op = operation(
        "eq_hpf",
        "eq",
        {"band": 2, "freq_hz": 375.0, "gain_db": -5.0, "q": 2.0},
    )
    snapshot = await adapter.snapshot(op)
    applied = await adapter.apply(op, snapshot)
    wing.fail_on_send_number = 4

    rolled_back = await adapter.rollback(op, snapshot, applied)

    assert rolled_back.success is False
    assert wing.values["/ch/3/eq/2f"] == 300.0
    assert wing.values["/ch/3/eq/2g"] == -2.0
    assert wing.values["/ch/3/eq/2q"] == 2.0
    assert rolled_back.evidence["send_failures"] == [
        {"address": "/ch/3/eq/2q", "reason": "rollback_send_failed"}
    ]


@pytest.mark.asyncio
async def test_rollback_rechecks_each_address_and_preserves_late_operator_change():
    wing = FakeWing()
    wing.values.update({
        "/ch/3/eq/2f": 300.0,
        "/ch/3/eq/2g": -2.0,
        "/ch/3/eq/2q": 1.0,
    })
    adapter = WingSupervisedFamilyAdapter("eq_hpf", wing)
    op = operation(
        "eq_hpf",
        "eq",
        {"band": 2, "freq_hz": 375.0, "gain_db": -5.0, "q": 2.0},
    )
    snapshot = await adapter.snapshot(op)
    applied = await adapter.apply(op, snapshot)

    def operator_change_during_rollback(send_number, _address, _value):
        if send_number == 4:
            wing.values["/ch/3/eq/2g"] = -9.0

    wing.after_send = operator_change_during_rollback
    rolled_back = await adapter.rollback(op, snapshot, applied)

    assert rolled_back.success is False
    assert rolled_back.reason == "rollback_cas_target_changed"
    assert wing.values["/ch/3/eq/2g"] == -9.0
    assert "/ch/3/eq/2g" in rolled_back.evidence["cas_mismatches"]


@pytest.mark.asyncio
async def test_cancelled_inflight_send_is_journaled_then_recovered():
    wing = FakeWing()
    wing.values["/ch/3/fdr"] = -12.0
    wing.block_on_send_number = 1
    adapter = WingSupervisedFamilyAdapter("fader", wing)
    op = operation("fader", "fader", -14.0)
    snapshot = await adapter.snapshot(op)

    task = asyncio.create_task(adapter.apply(op, snapshot))
    started = await asyncio.to_thread(wing.block_started.wait, 1.0)
    assert started is True
    task.cancel()
    wing.block_release.set()
    applied = await task
    rolled_back = await adapter.rollback(op, snapshot, applied)

    assert applied.success is False
    assert applied.reason == "wing_send_cancelled"
    assert applied.evidence["writes"][0]["sent"] is True
    assert rolled_back.success is True
    assert wing.values["/ch/3/fdr"] == -12.0


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_blocking_send_to_finish():
    wing = FakeWing()
    wing.values["/ch/3/fdr"] = -12.0
    wing.block_on_send_number = 1
    adapter = WingSupervisedFamilyAdapter("fader", wing)
    op = operation("fader", "fader", -14.0)
    snapshot = await adapter.snapshot(op)

    task = asyncio.create_task(adapter.apply(op, snapshot))
    started = await asyncio.to_thread(wing.block_started.wait, 1.0)
    assert started is True
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0.05)
    assert task.done() is False

    wing.block_release.set()
    applied = await task

    assert applied.success is False
    assert applied.reason == "wing_send_cancelled"
    assert wing.values["/ch/3/fdr"] == -14.0


@pytest.mark.asyncio
async def test_stale_upward_fader_target_blocks_without_send():
    wing = FakeWing()
    wing.values["/ch/3/fdr"] = -60.0
    adapter = WingSupervisedFamilyAdapter("fader", wing)
    op = operation("fader", "fader", 0.0)
    snapshot = await adapter.snapshot(op)

    applied = await adapter.apply(op, snapshot)

    assert applied.success is False
    assert applied.reason == "fader_boost_not_allowed"
    assert wing.sent == []


def test_factory_registers_no_routing_or_scene_adapter():
    adapters = build_wing_supervised_family_adapters(FakeWing())

    assert "routing" not in adapters
    assert "scene_preset" not in adapters
    assert set(adapters) == {
        "fader",
        "gain_trim",
        "mute",
        "pan_phase_delay",
        "eq_hpf",
        "gate_expander",
        "dynamics_compressor",
        "fx_sends_inserts",
        "bus_group_dca_matrix",
        "master_output",
    }
