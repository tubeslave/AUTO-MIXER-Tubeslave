"""Explicit WING adapters for supervised soundcheck transaction families.

The adapter boundary performs physical two-round-trip reads through
``WingClient.query_parameter``.  It never trusts the WING state cache for a
snapshot, verification, or rollback compare-and-swap decision.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
from typing import Any, Awaitable, Callable, Dict, Iterable, Mapping, Sequence, Tuple

from compressor_adaptation import ratio_float_to_wing
from supervised_console_transaction import (
    AdapterStepResult,
    AudioRecheckResult,
    ConsoleTransactionOperation,
)


@dataclass(frozen=True)
class WingParameterWrite:
    address: str
    value: Any
    tolerance: float = 0.05


AudioRecheck = Callable[
    [ConsoleTransactionOperation, Mapping[str, Any], Mapping[str, Any]],
    Awaitable[AudioRecheckResult] | AudioRecheckResult,
]
WriteAuthorityCurrent = Callable[[], bool]
BeforeProcessorEnable = Callable[
    [ConsoleTransactionOperation], Awaitable[bool] | bool
]


class SafetyThreadCallCancelled(asyncio.CancelledError):
    """Cancellation delivered after the blocking worker outcome is known."""

    def __init__(self, *, outcome: Any = None, error: BaseException | None = None):
        super().__init__("blocking safety call completed after cancellation")
        self.outcome = outcome
        self.error = error


async def _safety_thread_call(function: Callable[..., Any], *args: Any) -> Any:
    """Wait for an in-flight blocking I/O call before propagating cancellation.

    ``asyncio.to_thread`` does not stop its worker when the awaiting task is
    cancelled. Waiting for the worker prevents rollback from racing a WING send
    that is still completing in the background.
    """
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancellation = cancellation or exc
            continue
        except BaseException:
            if cancellation is None:
                raise
            break

    outcome = None
    error = None
    try:
        outcome = task.result()
    except BaseException as exc:
        error = exc
    if cancellation is not None:
        raise SafetyThreadCallCancelled(outcome=outcome, error=error) from cancellation
    if error is not None:
        raise error
    return outcome


def _channel(operation: ConsoleTransactionOperation) -> int:
    if (
        isinstance(operation.channel, bool)
        or not isinstance(operation.channel, int)
        or operation.channel < 1
        or operation.channel > 48
    ):
        raise ValueError("channel_required")
    return operation.channel


def _mapping(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("mapping_target_required")
    return dict(value)


def _target_number(value: Any, minimum: float, maximum: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid_{name}") from exc
    if not math.isfinite(number) or number < minimum or number > maximum:
        raise ValueError(f"{name}_out_of_range")
    return number


def _target_bool(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int) and value in (0, 1):
        return value
    raise ValueError("invalid_boolean_target")


def _target_int(value: Any, minimum: int, maximum: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"invalid_{name}")
    if isinstance(value, int):
        number = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"invalid_{name}")
        number = int(value)
    elif isinstance(value, str):
        raw = value.strip()
        if not raw or not raw.lstrip("+-").isdigit():
            raise ValueError(f"invalid_{name}")
        number = int(raw)
    else:
        raise ValueError(f"invalid_{name}")
    if number < minimum or number > maximum:
        raise ValueError(f"{name}_out_of_range")
    return number


def _target_enum(value: Any, allowed: Iterable[str], name: str) -> str:
    text = str(value).strip()
    allowed_values = {str(item) for item in allowed}
    if text not in allowed_values:
        raise ValueError(f"invalid_{name}")
    return text


def _numeric_or_text_equal(expected: Any, actual: Any, tolerance: float) -> bool:
    if isinstance(expected, bool):
        return bool(actual) is expected
    try:
        expected_number = float(expected)
        actual_number = float(actual)
    except (TypeError, ValueError):
        return str(expected).strip().upper() == str(actual).strip().upper()
    return math.isfinite(actual_number) and abs(expected_number - actual_number) <= tolerance


def _physical_value_available(write: WingParameterWrite, value: Any) -> bool:
    """Require a usable physical preimage for verification and rollback."""

    if value is None:
        return False
    try:
        float(write.value)
    except (TypeError, ValueError):
        return isinstance(value, str) and bool(value.strip())
    try:
        actual_number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(actual_number)


def _ratio_text(value: Any) -> str:
    number = _target_number(value, 1.0, 100.0, "ratio")
    return ratio_float_to_wing(number)


def _channel_writes(operation: ConsoleTransactionOperation) -> Tuple[WingParameterWrite, ...]:
    channel = _channel(operation)
    target = operation.target_value
    base = f"/ch/{channel}"
    fixed_preimage_restore = bool(
        operation.metadata.get("m110_fixed_preimage_restore") is True
    )

    if operation.family_id == "gain_trim" and operation.write_kind in {"gain", "trim"}:
        return (WingParameterWrite(f"{base}/in/set/trim", _target_number(target, -18.0, 18.0, "trim_db")),)
    if operation.family_id == "fader" and operation.write_kind == "fader":
        return (WingParameterWrite(f"{base}/fdr", _target_number(target, -144.0, 0.0, "fader_db")),)
    if operation.family_id == "mute" and operation.write_kind == "mute":
        return (WingParameterWrite(f"{base}/mute", _target_bool(target), 0.0),)
    processor_block = (
        target.get("m110_processor_block")
        if isinstance(target, Mapping)
        else None
    )
    if operation.family_id == "eq_hpf" and operation.write_kind == "hpf":
        values = _mapping(target)
        hpf_hz = _target_number(values.get("freq_hz"), 20.0, 2000.0, "hpf_hz")
        if processor_block == "hpf":
            if set(values) != {
                "m110_processor_block",
                "freq_hz",
                "slope_db_per_octave",
                "enabled",
            }:
                raise ValueError("m110_hpf_processor_block_fields_invalid")
            return (
                WingParameterWrite(
                    f"{base}/flt/lcf",
                    hpf_hz,
                    max(1.0, hpf_hz * 0.005),
                ),
                WingParameterWrite(
                    f"{base}/flt/lcs",
                    _target_enum(
                        values.get("slope_db_per_octave"),
                        {"6", "12", "18", "24"},
                        "hpf_slope_db_per_octave",
                    ),
                    0.1,
                ),
                WingParameterWrite(
                    f"{base}/flt/lc",
                    _target_bool(values.get("enabled")),
                    0.0,
                ),
            )
        return (
            WingParameterWrite(f"{base}/flt/lcf", hpf_hz, max(1.0, hpf_hz * 0.005)),
            WingParameterWrite(f"{base}/flt/lc", _target_bool(values.get("enabled")), 0.0),
        )
    if operation.family_id == "eq_hpf" and operation.write_kind == "eq":
        values = _mapping(target)
        if processor_block == "eq":
            if set(values) != {
                "m110_processor_block",
                "model",
                "mix_pct",
                "low_shelf",
                "bands",
                "high_shelf",
                "enabled",
            }:
                raise ValueError("m110_eq_processor_block_fields_invalid")
            model = str(values.get("model") or "").strip().upper()
            if model != "STD":
                raise ValueError("m110_eq_processor_model_must_be_std")
            bands = values.get("bands")
            if not isinstance(bands, list) or len(bands) != 4:
                raise ValueError("m110_eq_processor_bands_must_cover_1_4")
            band_writes = []
            for expected_band, raw_band in enumerate(bands, start=1):
                band_values = _mapping(raw_band)
                if set(band_values) != {"band", "freq_hz", "gain_db", "q"}:
                    raise ValueError("m110_eq_processor_band_fields_invalid")
                band = _target_int(band_values.get("band"), 1, 4, "eq_band")
                if band != expected_band:
                    raise ValueError("m110_eq_processor_bands_not_canonical")
                eq_hz = _target_number(
                    band_values.get("freq_hz"),
                    20.0,
                    20000.0,
                    "eq_freq_hz",
                )
                band_writes.extend(
                    (
                        WingParameterWrite(
                            f"{base}/eq/{band}f",
                            eq_hz,
                            max(1.0, eq_hz * 0.005),
                        ),
                        WingParameterWrite(
                            f"{base}/eq/{band}g",
                            _target_number(
                                band_values.get("gain_db"),
                                -15.0,
                                0.0,
                                "eq_gain_db",
                            ),
                        ),
                        WingParameterWrite(
                            f"{base}/eq/{band}q",
                            _target_number(
                                band_values.get("q"),
                                0.44,
                                10.0,
                                "eq_q",
                            ),
                        ),
                    )
                )
            shelf_writes = []
            for shelf_name, suffix, minimum_hz in (
                ("low_shelf", "l", 20.0),
                ("high_shelf", "h", 50.0),
            ):
                shelf = _mapping(values.get(shelf_name))
                if set(shelf) != {"freq_hz", "gain_db", "q", "type"}:
                    raise ValueError("m110_eq_processor_shelf_fields_invalid")
                shelf_type = str(shelf.get("type") or "").strip().upper()
                if shelf_type not in {"PEQ", "SHV"}:
                    raise ValueError("m110_eq_processor_shelf_type_invalid")
                shelf_hz = _target_number(
                    shelf.get("freq_hz"),
                    minimum_hz,
                    20000.0,
                    f"eq_{shelf_name}_freq_hz",
                )
                shelf_writes.extend(
                    (
                        WingParameterWrite(
                            f"{base}/eq/{suffix}f",
                            shelf_hz,
                            max(1.0, shelf_hz * 0.005),
                        ),
                        WingParameterWrite(
                            f"{base}/eq/{suffix}g",
                            _target_number(
                                shelf.get("gain_db"),
                                -15.0,
                                0.0,
                                f"eq_{shelf_name}_gain_db",
                            ),
                        ),
                        WingParameterWrite(
                            f"{base}/eq/{suffix}q",
                            _target_number(
                                shelf.get("q"),
                                0.44,
                                10.0,
                                f"eq_{shelf_name}_q",
                            ),
                        ),
                        WingParameterWrite(
                            f"{base}/eq/{suffix}eq",
                            shelf_type,
                            0.0,
                        ),
                    )
                )
            return (
                WingParameterWrite(f"{base}/eq/mdl", model, 0.0),
                WingParameterWrite(
                    f"{base}/eq/mix",
                    _target_number(values.get("mix_pct"), 0.0, 100.0, "eq_mix_pct"),
                    0.5,
                ),
                *shelf_writes[:4],
                *band_writes,
                *shelf_writes[4:],
                WingParameterWrite(
                    f"{base}/eq/on",
                    _target_bool(values.get("enabled")),
                    0.0,
                ),
            )
        band = _target_int(values.get("band"), 1, 4, "eq_band")
        eq_hz = _target_number(values.get("freq_hz"), 20.0, 20000.0, "eq_freq_hz")
        return (
            WingParameterWrite(f"{base}/eq/{band}f", eq_hz, max(1.0, eq_hz * 0.005)),
            WingParameterWrite(
                f"{base}/eq/{band}g",
                _target_number(
                    values.get("gain_db"),
                    -15.0,
                    15.0 if fixed_preimage_restore else 0.0,
                    "eq_gain_db",
                ),
            ),
            WingParameterWrite(f"{base}/eq/{band}q", _target_number(values.get("q"), 0.44, 10.0, "eq_q")),
        )
    if operation.family_id == "gate_expander" and operation.write_kind == "gate":
        values = _mapping(target)
        attack_ms = _target_number(values.get("attack_ms"), 0.0, 120.0, "gate_attack_ms")
        hold_ms = _target_number(values.get("hold_ms"), 0.0, 2000.0, "gate_hold_ms")
        release_ms = _target_number(values.get("release_ms"), 4.0, 4000.0, "gate_release_ms")
        if processor_block == "gate":
            if set(values) != {
                "m110_processor_block",
                "model",
                "mix_pct",
                "gain_db",
                "threshold_db",
                "range_db",
                "attack_ms",
                "hold_ms",
                "release_ms",
                "accent_pct",
                "ratio",
                "sidechain",
                "enabled",
            }:
                raise ValueError("m110_gate_processor_block_fields_invalid")
            if str(values.get("model") or "").strip().upper() != "GATE":
                raise ValueError("m110_gate_processor_model_must_be_gate")
            ratio = str(values.get("ratio") or "").strip().upper()
            if ratio not in {"1:1.5", "1:2", "1:3", "1:4", "GATE"}:
                raise ValueError("m110_gate_processor_ratio_invalid")
            sidechain = _mapping(values.get("sidechain"))
            if set(sidechain) != {"type", "freq_hz", "q", "source", "tap"}:
                raise ValueError("m110_gate_sidechain_fields_invalid")
            if str(sidechain.get("type") or "").strip().upper() != "OFF":
                raise ValueError("m110_gate_sidechain_type_must_be_off")
            if str(sidechain.get("source") or "").strip().upper() != "SELF":
                raise ValueError("m110_gate_sidechain_source_must_be_self")
            if str(sidechain.get("tap") or "").strip().upper() != "IN":
                raise ValueError("m110_gate_sidechain_tap_must_be_in")
            return (
                WingParameterWrite(f"{base}/gate/mdl", "GATE", 0.0),
                WingParameterWrite(
                    f"{base}/gate/mix",
                    _target_number(
                        values.get("mix_pct"),
                        0.0,
                        100.0,
                        "gate_mix_pct",
                    ),
                    0.5,
                ),
                WingParameterWrite(
                    f"{base}/gate/gain",
                    _target_number(
                        values.get("gain_db"),
                        -6.0,
                        12.0,
                        "gate_gain_db",
                    ),
                    0.26,
                ),
                WingParameterWrite(f"{base}/gate/thr", _target_number(values.get("threshold_db"), -80.0, 0.0, "gate_threshold_db"), 0.26),
                WingParameterWrite(f"{base}/gate/range", _target_number(values.get("range_db"), 3.0, 60.0, "gate_range_db"), 0.51),
                WingParameterWrite(f"{base}/gate/att", attack_ms, max(0.5, attack_ms * 0.03)),
                WingParameterWrite(f"{base}/gate/hld", hold_ms, max(1.0, hold_ms * 0.03)),
                WingParameterWrite(f"{base}/gate/rel", release_ms, max(1.0, release_ms * 0.03)),
                WingParameterWrite(f"{base}/gate/acc", _target_number(values.get("accent_pct"), 0.0, 100.0, "gate_accent_pct"), 0.5),
                WingParameterWrite(f"{base}/gate/ratio", ratio, 0.0),
                WingParameterWrite(f"{base}/gatesc/type", "OFF", 0.0),
                WingParameterWrite(
                    f"{base}/gatesc/f",
                    _target_number(
                        sidechain.get("freq_hz"),
                        20.0,
                        20000.0,
                        "gate_sidechain_freq_hz",
                    ),
                    5.0,
                ),
                WingParameterWrite(
                    f"{base}/gatesc/q",
                    _target_number(
                        sidechain.get("q"),
                        0.44,
                        10.0,
                        "gate_sidechain_q",
                    ),
                ),
                WingParameterWrite(f"{base}/gatesc/src", "SELF", 0.0),
                WingParameterWrite(f"{base}/gatesc/tap", "IN", 0.0),
                WingParameterWrite(f"{base}/gate/on", _target_bool(values.get("enabled")), 0.0),
            )
        return (
            WingParameterWrite(f"{base}/gate/thr", _target_number(values.get("threshold_db"), -80.0, 0.0, "gate_threshold_db"), 0.26),
            WingParameterWrite(f"{base}/gate/range", _target_number(values.get("range_db"), 3.0, 60.0, "gate_range_db"), 0.51),
            WingParameterWrite(f"{base}/gate/att", attack_ms, max(0.5, attack_ms * 0.03)),
            WingParameterWrite(f"{base}/gate/hld", hold_ms, max(1.0, hold_ms * 0.03)),
            WingParameterWrite(f"{base}/gate/rel", release_ms, max(1.0, release_ms * 0.03)),
            WingParameterWrite(f"{base}/gate/on", _target_bool(values.get("enabled")), 0.0),
        )
    if operation.family_id == "dynamics_compressor" and operation.write_kind == "compressor":
        if isinstance(target, Mapping):
            values = _mapping(target)
            attack_ms = _target_number(values.get("attack_ms"), 0.0, 120.0, "compressor_attack_ms")
            release_ms = _target_number(values.get("release_ms"), 4.0, 4000.0, "compressor_release_ms")
            if processor_block == "compressor":
                if set(values) != {
                    "m110_processor_block",
                    "model",
                    "mix_pct",
                    "threshold_db",
                    "ratio",
                    "knee",
                    "detector",
                    "attack_ms",
                    "hold_ms",
                    "release_ms",
                    "envelope",
                    "auto",
                    "makeup_db",
                    "crossover",
                    "sidechain",
                    "enabled",
                }:
                    raise ValueError("m110_compressor_processor_block_fields_invalid")
                if str(values.get("model") or "").strip().upper() != "COMP":
                    raise ValueError("m110_compressor_model_must_be_comp")
                detector = str(values.get("detector") or "").strip().upper()
                envelope = str(values.get("envelope") or "").strip().upper()
                if detector not in {"PEAK", "RMS"}:
                    raise ValueError("m110_compressor_detector_invalid")
                if envelope not in {"LIN", "LOG"}:
                    raise ValueError("m110_compressor_envelope_invalid")
                crossover = _mapping(values.get("crossover"))
                if set(crossover) != {"depth_db", "type", "freq_hz", "q"}:
                    raise ValueError("m110_compressor_crossover_fields_invalid")
                if str(crossover.get("type") or "").strip().upper() != "OFF":
                    raise ValueError("m110_compressor_crossover_must_be_off")
                sidechain = _mapping(values.get("sidechain"))
                if set(sidechain) != {"type", "freq_hz", "q", "source", "tap"}:
                    raise ValueError("m110_compressor_sidechain_fields_invalid")
                if str(sidechain.get("type") or "").strip().upper() != "OFF":
                    raise ValueError("m110_compressor_sidechain_type_must_be_off")
                if str(sidechain.get("source") or "").strip().upper() != "SELF":
                    raise ValueError("m110_compressor_sidechain_source_must_be_self")
                if str(sidechain.get("tap") or "").strip().upper() != "IN":
                    raise ValueError("m110_compressor_sidechain_tap_must_be_in")
                return (
                    WingParameterWrite(f"{base}/dyn/mdl", "COMP", 0.0),
                    WingParameterWrite(f"{base}/dyn/mix", _target_number(values.get("mix_pct"), 0.0, 100.0, "compressor_mix_pct"), 0.5),
                    WingParameterWrite(f"{base}/dyn/thr", _target_number(values.get("threshold_db"), -60.0, 0.0, "compressor_threshold_db"), 0.26),
                    WingParameterWrite(f"{base}/dyn/ratio", _ratio_text(values.get("ratio")), 0.05),
                    WingParameterWrite(f"{base}/dyn/knee", _target_int(values.get("knee"), 0, 5, "compressor_knee"), 0.0),
                    WingParameterWrite(f"{base}/dyn/det", detector, 0.0),
                    WingParameterWrite(f"{base}/dyn/att", attack_ms, max(0.5, attack_ms * 0.03)),
                    WingParameterWrite(f"{base}/dyn/hld", _target_number(values.get("hold_ms"), 1.0, 200.0, "compressor_hold_ms"), 1.0),
                    WingParameterWrite(f"{base}/dyn/rel", release_ms, max(1.0, release_ms * 0.03)),
                    WingParameterWrite(f"{base}/dyn/env", envelope, 0.0),
                    WingParameterWrite(f"{base}/dyn/auto", _target_bool(values.get("auto")), 0.0),
                    WingParameterWrite(f"{base}/dyn/gain", _target_number(values.get("makeup_db"), -6.0, 6.0, "compressor_makeup_db"), 0.26),
                    WingParameterWrite(
                        f"{base}/dynxo/depth",
                        _target_number(
                            crossover.get("depth_db"),
                            0.0,
                            20.0,
                            "compressor_crossover_depth_db",
                        ),
                        0.26,
                    ),
                    WingParameterWrite(f"{base}/dynxo/type", "OFF", 0.0),
                    WingParameterWrite(
                        f"{base}/dynxo/f",
                        _target_number(
                            crossover.get("freq_hz"),
                            20.0,
                            20000.0,
                            "compressor_crossover_freq_hz",
                        ),
                        5.0,
                    ),
                    WingParameterWrite(
                        f"{base}/dynxo/q",
                        _target_number(
                            crossover.get("q"),
                            0.4,
                            4.0,
                            "compressor_crossover_q",
                        ),
                    ),
                    WingParameterWrite(f"{base}/dynsc/type", "OFF", 0.0),
                    WingParameterWrite(
                        f"{base}/dynsc/f",
                        _target_number(
                            sidechain.get("freq_hz"),
                            20.0,
                            20000.0,
                            "compressor_sidechain_freq_hz",
                        ),
                        5.0,
                    ),
                    WingParameterWrite(
                        f"{base}/dynsc/q",
                        _target_number(
                            sidechain.get("q"),
                            0.44,
                            10.0,
                            "compressor_sidechain_q",
                        ),
                    ),
                    WingParameterWrite(f"{base}/dynsc/src", "SELF", 0.0),
                    WingParameterWrite(f"{base}/dynsc/tap", "IN", 0.0),
                    WingParameterWrite(f"{base}/dyn/on", _target_bool(values.get("enabled")), 0.0),
                )
            return (
                WingParameterWrite(f"{base}/dyn/thr", _target_number(values.get("threshold_db"), -60.0, 0.0, "compressor_threshold_db"), 0.26),
                WingParameterWrite(f"{base}/dyn/ratio", _ratio_text(values.get("ratio")), 0.05),
                WingParameterWrite(f"{base}/dyn/att", attack_ms, max(0.5, attack_ms * 0.03)),
                WingParameterWrite(f"{base}/dyn/rel", release_ms, max(1.0, release_ms * 0.03)),
                WingParameterWrite(f"{base}/dyn/gain", _target_number(values.get("makeup_db", 0.0), -6.0, 12.0, "compressor_makeup_db"), 0.26),
                WingParameterWrite(f"{base}/dyn/on", _target_bool(values.get("enabled")), 0.0),
            )
        return (WingParameterWrite(f"{base}/dyn/gain", _target_number(target, -6.0, 12.0, "compressor_makeup_db")),)
    if operation.family_id == "pan_phase_delay" and operation.write_kind == "pan":
        return (WingParameterWrite(f"{base}/pan", _target_number(target, -100.0, 100.0, "pan"), 0.5),)
    if operation.family_id == "pan_phase_delay" and operation.write_kind == "polarity":
        return (WingParameterWrite(f"{base}/in/set/inv", _target_bool(target), 0.0),)
    if operation.family_id == "pan_phase_delay" and operation.write_kind == "delay":
        values = _mapping(target)
        enabled = _target_bool(values.get("enabled"))
        return (
            WingParameterWrite(f"{base}/in/set/dlymode", "MS", 0.0),
            WingParameterWrite(
                f"{base}/in/set/dly",
                _target_number(values.get("delay_ms"), 0.0, 500.0, "delay_ms"),
                0.2,
            ),
            WingParameterWrite(f"{base}/in/set/dlyon", enabled, 0.0),
        )
    if operation.family_id == "fx_sends_inserts" and operation.write_kind == "send":
        values = _mapping(target)
        send_bus = _target_int(values.get("send_bus"), 1, 16, "send_bus")
        return (WingParameterWrite(f"{base}/send/{send_bus}/lvl", _target_number(values.get("level_db"), -144.0, 0.0, "send_level_db")),)
    raise ValueError("unsupported_channel_operation")


def _bus_group_writes(operation: ConsoleTransactionOperation) -> Tuple[WingParameterWrite, ...]:
    values = _mapping(operation.target_value)
    if operation.write_kind == "dca":
        dca = _target_int(values.get("dca_id"), 1, 16, "dca")
        return (WingParameterWrite(f"/dca/{dca}/fdr", _target_number(values.get("target_db"), -144.0, 0.0, "dca_fader_db")),)
    bus = _target_int(values.get("bus_id"), 1, 16, "bus")
    base = f"/bus/{bus}"
    parameter = operation.parameter
    if parameter.endswith("_fader_db"):
        return (WingParameterWrite(f"{base}/fdr", _target_number(values.get("target_db"), -144.0, 0.0, "bus_fader_db")),)
    if "_eq_band_" in parameter:
        band = _target_int(values.get("band"), 1, 8, "bus_eq_band")
        eq_hz = _target_number(values.get("freq_hz"), 20.0, 20000.0, "bus_eq_freq_hz")
        return (
            WingParameterWrite(f"{base}/eq/{band}f", eq_hz, max(1.0, eq_hz * 0.005)),
            WingParameterWrite(f"{base}/eq/{band}g", _target_number(values.get("gain_db"), -15.0, 0.0, "bus_eq_gain_db")),
            WingParameterWrite(f"{base}/eq/{band}q", _target_number(values.get("q"), 0.44, 10.0, "bus_eq_q")),
        )
    if parameter.endswith("_compressor_makeup_db"):
        return (WingParameterWrite(f"{base}/dyn/gain", _target_number(values.get("makeup_db"), -6.0, 12.0, "bus_compressor_makeup_db")),)
    if parameter.endswith("_compressor"):
        attack_ms = _target_number(values.get("attack_ms"), 0.0, 120.0, "bus_compressor_attack_ms")
        release_ms = _target_number(values.get("release_ms"), 4.0, 4000.0, "bus_compressor_release_ms")
        return (
            WingParameterWrite(f"{base}/dyn/thr", _target_number(values.get("threshold_db"), -60.0, 0.0, "bus_compressor_threshold_db"), 0.26),
            WingParameterWrite(f"{base}/dyn/ratio", _ratio_text(values.get("ratio")), 0.05),
            WingParameterWrite(f"{base}/dyn/att", attack_ms, max(0.5, attack_ms * 0.03)),
            WingParameterWrite(f"{base}/dyn/rel", release_ms, max(1.0, release_ms * 0.03)),
            WingParameterWrite(f"{base}/dyn/gain", _target_number(values.get("makeup_db", 0.0), -6.0, 12.0, "bus_compressor_makeup_db"), 0.26),
            WingParameterWrite(f"{base}/dyn/on", _target_bool(values.get("enabled")), 0.0),
        )
    raise ValueError("unsupported_bus_group_operation")


def _master_writes(operation: ConsoleTransactionOperation) -> Tuple[WingParameterWrite, ...]:
    values = _mapping(operation.target_value)
    main = _target_int(values.get("main_id"), 1, 4, "main")
    if operation.write_kind != "master" or not operation.parameter.endswith("_fader_db"):
        raise ValueError("unsupported_master_operation")
    return (WingParameterWrite(f"/main/{main}/fdr", _target_number(values.get("target_db"), -144.0, 0.0, "master_fader_db")),)


def parameter_writes_for_operation(
    operation: ConsoleTransactionOperation,
) -> Tuple[WingParameterWrite, ...]:
    if operation.family_id == "bus_group_dca_matrix":
        return _bus_group_writes(operation)
    if operation.family_id == "master_output":
        return _master_writes(operation)
    return _channel_writes(operation)


def audible_state_reads_for_operation(
    operation: ConsoleTransactionOperation,
) -> Tuple[WingParameterWrite, ...]:
    """Return read-only processing-state probes required for an audible result.

    These addresses are deliberately not part of the mutation tuple.  Enabling
    an EQ or compressor is a separate operator-reviewed correction; a tonal
    child may proceed only when fresh physical readback proves that the relevant
    processor is already on.
    """

    if (
        isinstance(operation.target_value, Mapping)
        and operation.target_value.get("m110_processor_block")
        in {"hpf", "eq", "gate", "compressor"}
    ):
        return ()
    if operation.family_id == "eq_hpf" and operation.write_kind == "eq":
        return (
            WingParameterWrite(f"/ch/{_channel(operation)}/eq/on", 1, 0.0),
        )
    if operation.family_id == "bus_group_dca_matrix":
        values = _mapping(operation.target_value)
        if operation.write_kind == "dca":
            return ()
        bus = _target_int(values.get("bus_id"), 1, 16, "bus")
        if "_eq_band_" in operation.parameter:
            return (WingParameterWrite(f"/bus/{bus}/eq/on", 1, 0.0),)
        if operation.parameter.endswith("_compressor_makeup_db"):
            return (WingParameterWrite(f"/bus/{bus}/dyn/on", 1, 0.0),)
    if (
        operation.family_id == "dynamics_compressor"
        and operation.write_kind == "compressor"
        and not isinstance(operation.target_value, Mapping)
    ):
        return (
            WingParameterWrite(f"/ch/{_channel(operation)}/dyn/on", 1, 0.0),
        )
    return ()


class WingSupervisedFamilyAdapter:
    """One code-reviewed semantic family bound to one physical WING client."""

    def __init__(
        self,
        family_id: str,
        wing_client: Any,
        *,
        audio_recheck: AudioRecheck | None = None,
        write_authority_current: WriteAuthorityCurrent | None = None,
        before_processor_enable: BeforeProcessorEnable | None = None,
        processor_enable_authorized: bool = False,
        constructive_lane_authorized: bool = False,
        fixed_preimage_restore_authorized: bool = False,
        query_timeout: float = 0.5,
    ) -> None:
        self.family_id = family_id
        self.wing_client = wing_client
        self._audio_recheck = audio_recheck
        self._write_authority_current = write_authority_current
        self._before_processor_enable = before_processor_enable
        self._processor_enable_authorized = bool(processor_enable_authorized)
        self._constructive_lane_authorized = bool(
            constructive_lane_authorized
        )
        self._fixed_preimage_restore_authorized = bool(
            fixed_preimage_restore_authorized
        )
        self.query_timeout = max(0.05, float(query_timeout))

    def _writes(self, operation: ConsoleTransactionOperation) -> Tuple[WingParameterWrite, ...]:
        if operation.family_id != self.family_id:
            raise ValueError("family_adapter_identity_mismatch")
        return parameter_writes_for_operation(operation)

    def _audible_reads(
        self,
        operation: ConsoleTransactionOperation,
    ) -> Tuple[WingParameterWrite, ...]:
        if operation.metadata.get("m110_fixed_preimage_restore") is True:
            return ()
        return audible_state_reads_for_operation(operation)

    @staticmethod
    def _physical_guard_reads(
        operation: ConsoleTransactionOperation,
    ) -> Tuple[WingParameterWrite, ...]:
        # This executor-owned guard is deliberately outside the musical
        # canonical bundle address scope.  It is a read-only safety condition
        # repeated before every physical send, never a client target.
        if operation.metadata.get("m110_fixed_preimage_restore") is True:
            channel = operation.channel
            processor_guard = None
            if isinstance(channel, int) and not isinstance(channel, bool):
                base = f"/ch/{channel}"
                processor_suffix = {
                    "hpf": "/flt/lc",
                    "eq": "/eq/on",
                    "gate": "/gate/on",
                    "compressor": "/dyn/on",
                }.get(operation.write_kind)
                if processor_suffix is not None:
                    processor_guard = WingParameterWrite(
                        base + processor_suffix,
                        0,
                        0.0,
                    )
            return tuple(
                guard
                for guard in (
                    processor_guard,
                    # Main mute is intentionally queried last immediately
                    # before each fixed-restore send.
                    WingParameterWrite("/main/1/mute", 1, 0.0),
                )
                if guard is not None
            )
        return ()

    def prepare(self, operation: ConsoleTransactionOperation) -> Mapping[str, Any]:
        """Pure validation/address expansion performed before physical I/O."""
        writes = self._writes(operation)
        # Executor-owned physical guards are deliberately excluded from the
        # canonical musical address scope.  The same Main mute guard may be
        # required by every operation in an atomic restoration stage, whereas
        # canonical bundles reject duplicate client-controlled read addresses.
        audible_reads = self._audible_reads(operation)
        return {
            "addresses": tuple(write.address for write in writes),
            "write_count": len(writes),
            "writes": tuple(
                {
                    "address": write.address,
                    "target_value": write.value,
                    "tolerance": write.tolerance,
                }
                for write in writes
            ),
            "audible_state_reads": tuple(
                {
                    "address": read.address,
                    "expected_value": read.value,
                    "tolerance": read.tolerance,
                }
                for read in audible_reads
            ),
        }

    def write_authority_is_current(self) -> bool:
        return bool(
            self._write_authority_current is None
            or self._write_authority_current() is True
        )

    def validate_snapshot(
        self,
        operation: ConsoleTransactionOperation,
        snapshot: Mapping[str, Any],
    ) -> str | None:
        """Enforce conservative direction and delta limits on fresh state."""
        writes = self._writes(operation)
        audible_reads = (
            *self._audible_reads(operation),
            *self._physical_guard_reads(operation),
        )
        old_values = dict(snapshot.get("values") or {})
        if any(
            read.address not in old_values
            for read in (*writes, *audible_reads)
        ):
            return "apply_snapshot_incomplete"

        processor_block = bool(
            operation.metadata.get("m110_processor_enable_bundle") is True
            and isinstance(operation.target_value, Mapping)
            and operation.target_value.get("m110_processor_block")
            in {"hpf", "eq", "gate", "compressor"}
        )
        if processor_block:
            if not self._processor_enable_authorized:
                return "processor_enable_commit_once_grant_required"
            enable_writes = [
                write
                for write in writes
                if write.address.endswith(("/eq/on", "/gate/on", "/dyn/on", "/flt/lc"))
            ]
            if not (
                len(enable_writes) == 1
                and writes[-1].address == enable_writes[0].address
                and _numeric_or_text_equal(old_values[enable_writes[0].address], 0, 0.0)
                and _numeric_or_text_equal(enable_writes[0].value, 1, 0.0)
            ):
                return "processor_enable_block_must_start_bypassed_and_enable_last"
            return None

        fixed_preimage_restore = bool(
            operation.metadata.get("m110_fixed_preimage_restore") is True
        )
        if fixed_preimage_restore:
            if not self._fixed_preimage_restore_authorized:
                return "m110_fixed_preimage_restore_grant_required"
            expected_values = operation.metadata.get(
                "m110_expected_current_values"
            )
            if not isinstance(expected_values, Mapping) or set(
                expected_values
            ) != {write.address for write in writes}:
                return "m110_fixed_preimage_restore_expected_values_invalid"
            for write in writes:
                current = old_values[write.address]
                if not (
                    _numeric_or_text_equal(
                        expected_values[write.address],
                        current,
                        write.tolerance,
                    )
                    or _numeric_or_text_equal(
                        write.value,
                        current,
                        write.tolerance,
                    )
                ):
                    return "m110_fixed_preimage_restore_cas_mismatch"
            for read in audible_reads:
                if not _numeric_or_text_equal(
                    read.value,
                    old_values[read.address],
                    read.tolerance,
                ):
                    return (
                        "m110_fixed_preimage_restore_main_mute_required"
                        if read.address == "/main/1/mute"
                        else "m110_fixed_preimage_restore_processor_bypass_required"
                    )
            return None

        for read in audible_reads:
            if not _numeric_or_text_equal(
                read.value,
                old_values[read.address],
                read.tolerance,
            ):
                return "processing_enable_state_change_requires_separate_approval"

        for write in writes:
            address = write.address
            old_value = old_values[address]
            target = write.value
            try:
                old_number = float(old_value)
                target_number = float(target)
            except (TypeError, ValueError):
                old_number = None
                target_number = None

            if address.endswith("/mute"):
                if bool(old_value) and not bool(target):
                    return "unmute_requires_dedicated_operator_action"
                continue

            if address.endswith(("/eq/on", "/gate/on", "/dyn/on", "/flt/lc", "/in/set/dlyon")):
                if not _numeric_or_text_equal(old_value, target, 0.0):
                    return "processing_enable_state_change_requires_separate_approval"
                continue

            if old_number is None or target_number is None:
                continue

            if address.endswith("/in/set/trim"):
                if target_number > old_number + write.tolerance:
                    return "gain_up_requires_separate_headroom_and_source_proof"
                if old_number - target_number > 6.0:
                    return "gain_down_delta_exceeds_6db"
            elif address.endswith("/fdr"):
                if target_number > old_number + write.tolerance:
                    if not (
                        self._constructive_lane_authorized
                        and target_number - old_number <= 3.0
                    ):
                        return "fader_boost_not_allowed"
                maximum_cut = 6.0 if operation.family_id == "fader" else 3.0
                if old_number - target_number > maximum_cut:
                    return "fader_cut_delta_exceeded"
            elif "/send/" in address and address.endswith("/lvl"):
                if target_number > old_number + write.tolerance:
                    if not (
                        self._constructive_lane_authorized
                        and target_number - old_number <= 3.0
                    ):
                        return "send_level_increase_not_allowed"
                if old_number - target_number > 6.0:
                    return "send_cut_delta_exceeds_6db"
            elif "/eq/" in address and address.endswith("g"):
                if target_number > old_number + write.tolerance:
                    return "relative_eq_boost_not_allowed"
                if old_number - target_number > 3.0:
                    return "eq_cut_delta_exceeds_3db"
            elif address.endswith("/dyn/gain"):
                if target_number > old_number + write.tolerance:
                    if not (
                        self._constructive_lane_authorized
                        and target_number - old_number <= 3.0
                    ):
                        return "compressor_makeup_increase_not_allowed"
                if old_number - target_number > 3.0:
                    return "compressor_makeup_cut_delta_exceeds_3db"
            elif address.endswith("/pan") and abs(target_number - old_number) > 10.0:
                return "pan_delta_exceeds_10_percent"
            elif address.endswith("/in/set/dly") and abs(target_number - old_number) > 5.0:
                return "delay_delta_exceeds_5ms"
            elif address.endswith("/flt/lcf"):
                if old_number <= 0 or target_number <= 0:
                    return "hpf_frequency_invalid"
                if max(old_number, target_number) / min(old_number, target_number) > 2.0:
                    return "hpf_frequency_delta_exceeds_one_octave"
            elif address.endswith("/dyn/ratio") and abs(
                target_number - old_number
            ) > 2.0:
                return "compressor_ratio_delta_exceeds_2"
            elif address.endswith(("/gate/thr", "/dyn/thr")) and abs(
                target_number - old_number
            ) > 3.0:
                return "dynamics_threshold_delta_exceeds_3db"
        return None

    async def _physical_values(
        self,
        writes: Sequence[WingParameterWrite],
    ) -> tuple[bool, Dict[str, Any], Dict[str, Any]]:
        values: Dict[str, Any] = {}
        evidence: Dict[str, Any] = {}
        for write in writes:
            result = await _safety_thread_call(
                self.wing_client.query_parameter,
                write.address,
                self.query_timeout,
            )
            evidence[write.address] = (
                dict(result)
                if isinstance(result, Mapping)
                else {"status": "invalid_response", "response_type": type(result).__name__}
            )
            confirmed_round_trips = (
                result.get("confirmed_round_trips")
                if isinstance(result, Mapping)
                else None
            )
            if (
                not isinstance(result, Mapping)
                or result.get("status") != "received"
                or result.get("address") != write.address
                or isinstance(confirmed_round_trips, bool)
                or not isinstance(confirmed_round_trips, int)
                or confirmed_round_trips < 2
                or not _physical_value_available(write, result.get("value"))
            ):
                return False, values, evidence
            values[write.address] = result.get("value")
        return True, values, evidence

    async def snapshot(self, operation: ConsoleTransactionOperation) -> Mapping[str, Any]:
        writes = self._writes(operation)
        audible_reads = (
            *self._audible_reads(operation),
            *self._physical_guard_reads(operation),
        )
        verified, values, evidence = await self._physical_values(
            (*writes, *audible_reads)
        )
        return {
            "physical_verified": verified,
            "values": values,
            "queries": evidence,
            "audible_state_verified": bool(
                verified
                and all(
                    _numeric_or_text_equal(
                        read.value,
                        values.get(read.address),
                        read.tolerance,
                    )
                    for read in audible_reads
                )
            ),
        }

    async def apply(
        self,
        operation: ConsoleTransactionOperation,
        snapshot: Mapping[str, Any],
    ) -> AdapterStepResult:
        writes = self._writes(operation)
        audible_reads = (
            *self._audible_reads(operation),
            *self._physical_guard_reads(operation),
        )
        old_values = dict(snapshot.get("values") or {})
        if any(write.address not in old_values for write in writes):
            return AdapterStepResult(False, False, "apply_snapshot_incomplete")
        transition_error = self.validate_snapshot(operation, snapshot)
        if transition_error:
            return AdapterStepResult(False, False, transition_error)
        processor_block = bool(
            operation.metadata.get("m110_processor_enable_bundle") is True
            and isinstance(operation.target_value, Mapping)
            and operation.target_value.get("m110_processor_block")
            in {"hpf", "eq", "gate", "compressor"}
        )
        fixed_preimage_restore = bool(
            operation.metadata.get("m110_fixed_preimage_restore") is True
            and self._fixed_preimage_restore_authorized
        )
        enable_write = writes[-1] if processor_block else None
        enable_guard = (
            WingParameterWrite(enable_write.address, 0, 0.0)
            if enable_write is not None
            else None
        )
        sent = []
        verified_values: Dict[str, Any] = {}
        for write in writes:
            if processor_block and write is enable_write:
                # The enable send is authorized only after one final complete
                # sweep proves every parameter target while the block remains
                # physically bypassed. Per-parameter verification below is not
                # enough because an external writer could drift an early row.
                parameter_writes = tuple(writes[:-1])
                final_physical, final_values, final_evidence = (
                    await self._physical_values(
                        (*parameter_writes, enable_guard)
                    )
                )
                final_mismatches = [
                    spec.address
                    for spec in parameter_writes
                    if not _numeric_or_text_equal(
                        spec.value,
                        final_values.get(spec.address),
                        spec.tolerance,
                    )
                ]
                if not (
                    final_physical
                    and not final_mismatches
                    and _numeric_or_text_equal(
                        0,
                        final_values.get(enable_write.address),
                        0.0,
                    )
                ):
                    return AdapterStepResult(
                        False,
                        False,
                        "processor_parameters_not_verified_while_bypassed",
                        {
                            "writes": sent,
                            "verified_values": verified_values,
                            "mismatches": final_mismatches,
                            "bypass_guard_actual": final_values.get(
                                enable_write.address
                            ),
                            "final_parameter_queries": final_evidence,
                        },
                    )
                if self._before_processor_enable is None:
                    return AdapterStepResult(
                        False,
                        False,
                        "processor_enable_safety_callback_required",
                        {
                            "writes": sent,
                            "verified_values": verified_values,
                        },
                    )
                try:
                    enable_safe = self._before_processor_enable(operation)
                    if asyncio.iscoroutine(enable_safe) or hasattr(
                        enable_safe,
                        "__await__",
                    ):
                        enable_safe = await enable_safe
                except asyncio.CancelledError:
                    return AdapterStepResult(
                        False,
                        False,
                        "processor_enable_safety_cancelled",
                        {
                            "writes": sent,
                            "verified_values": verified_values,
                            "cancellation_requested": True,
                        },
                    )
                except Exception as exc:
                    return AdapterStepResult(
                        False,
                        False,
                        "processor_enable_safety_failed",
                        {
                            "writes": sent,
                            "verified_values": verified_values,
                            "exception_type": type(exc).__name__,
                        },
                    )
                if enable_safe is not True:
                    return AdapterStepResult(
                        False,
                        False,
                        "processor_enable_safety_not_verified",
                        {
                            "writes": sent,
                            "verified_values": verified_values,
                        },
                    )
            precondition_specs = (write, *audible_reads)
            if processor_block and write is not enable_write:
                precondition_specs = (write, enable_guard)
            precondition_physical, precondition_values, precondition_evidence = (
                await self._physical_values(precondition_specs)
            )
            current_value = precondition_values.get(write.address)
            if (
                not precondition_physical
                or not _numeric_or_text_equal(
                    old_values[write.address],
                    current_value,
                    write.tolerance,
                )
            ):
                return AdapterStepResult(
                    False,
                    False,
                    (
                        "apply_cas_read_failed"
                        if not precondition_physical
                        else "apply_cas_target_changed"
                    ),
                    {
                        "writes": sent,
                        "precondition_address": write.address,
                        "precondition_expected": old_values[write.address],
                        "precondition_actual": current_value,
                        "precondition_queries": precondition_evidence,
                        "verified_values": verified_values,
                    },
                )
            if processor_block and not _numeric_or_text_equal(
                0,
                precondition_values.get(enable_write.address),
                0.0,
            ):
                return AdapterStepResult(
                    False,
                    False,
                    "processor_bypass_changed_before_parameter_send",
                    {
                        "writes": sent,
                        "verified_values": verified_values,
                        "bypass_guard_actual": precondition_values.get(
                            enable_write.address
                        ),
                        "bypass_guard_queries": precondition_evidence,
                    },
                )
            if any(
                not _numeric_or_text_equal(
                    read.value,
                    precondition_values.get(read.address),
                    read.tolerance,
                )
                for read in audible_reads
            ):
                return AdapterStepResult(
                    False,
                    False,
                    (
                        "m110_fixed_preimage_restore_guard_changed"
                        if fixed_preimage_restore
                        else "processing_enable_state_change_requires_separate_approval"
                    ),
                    {
                        "writes": sent,
                        "audible_state_queries": precondition_evidence,
                    },
                )
            if fixed_preimage_restore and _numeric_or_text_equal(
                write.value,
                current_value,
                write.tolerance,
            ):
                sent.append(
                    {
                        "address": write.address,
                        "value": write.value,
                        "sent": False,
                        "skipped": True,
                        "reason": "m110_fixed_preimage_already_restored",
                    }
                )
                verified_values[write.address] = current_value
                continue
            if (
                self._write_authority_current is not None
                and self._write_authority_current() is not True
            ):
                return AdapterStepResult(
                    False,
                    False,
                    "write_authority_expired_before_send",
                    {"writes": sent, "verified_values": verified_values},
                )
            try:
                ok = await _safety_thread_call(
                    self.wing_client.send,
                    write.address,
                    write.value,
                )
            except SafetyThreadCallCancelled as exc:
                completed = exc.error is None and exc.outcome is not None
                sent.append(
                    {
                        "address": write.address,
                        "value": write.value,
                        "sent": bool(exc.outcome) if completed else None,
                        "cancellation_requested": True,
                        "error": type(exc.error).__name__ if exc.error else None,
                    }
                )
                return AdapterStepResult(
                    False,
                    False,
                    "wing_send_cancelled",
                    {
                        "writes": sent,
                        "write_state_unknown": not completed,
                        "cancellation_requested": True,
                        "verified_values": verified_values,
                    },
                )
            except Exception as exc:
                sent.append(
                    {
                        "address": write.address,
                        "value": write.value,
                        "sent": None,
                        "error": type(exc).__name__,
                    }
                )
                return AdapterStepResult(
                    False,
                    False,
                    "wing_send_exception",
                    {
                        "writes": sent,
                        "write_state_unknown": True,
                        "verified_values": verified_values,
                    },
                )
            sent.append({"address": write.address, "value": write.value, "sent": bool(ok)})
            if not ok:
                return AdapterStepResult(
                    False,
                    False,
                    "wing_send_failed",
                    {"writes": sent, "verified_values": verified_values},
                )
            if processor_block and write is enable_write:
                if self._audio_recheck is None:
                    return AdapterStepResult(
                        False,
                        False,
                        "processor_enable_immediate_audio_callback_required",
                        {
                            "writes": sent,
                            "verified_values": verified_values,
                        },
                    )
                try:
                    immediate_audio = self._audio_recheck(
                        operation,
                        snapshot,
                        {
                            "writes": list(sent),
                            "verified_values": dict(verified_values),
                            "processor_enable_immediate": True,
                        },
                    )
                    if asyncio.iscoroutine(immediate_audio) or hasattr(
                        immediate_audio,
                        "__await__",
                    ):
                        immediate_audio = await immediate_audio
                except asyncio.CancelledError:
                    return AdapterStepResult(
                        False,
                        False,
                        "processor_enable_immediate_audio_cancelled",
                        {
                            "writes": sent,
                            "verified_values": verified_values,
                            "cancellation_requested": True,
                        },
                    )
                except Exception as exc:
                    return AdapterStepResult(
                        False,
                        False,
                        "processor_enable_immediate_audio_exception",
                        {
                            "writes": sent,
                            "verified_values": verified_values,
                            "exception_type": type(exc).__name__,
                        },
                    )
                if not (
                    isinstance(immediate_audio, AudioRecheckResult)
                    and immediate_audio.status == "passed"
                ):
                    return AdapterStepResult(
                        False,
                        False,
                        "processor_enable_immediate_audio_failed",
                        {
                            "writes": sent,
                            "verified_values": verified_values,
                            "audio_status": getattr(
                                immediate_audio,
                                "status",
                                None,
                            ),
                            "audio_reason": getattr(
                                immediate_audio,
                                "reason",
                                None,
                            ),
                        },
                    )
            if processor_block and write is not enable_write:
                post_physical, post_values, post_evidence = (
                    await self._physical_values((write, enable_guard))
                )
                if not (
                    post_physical
                    and _numeric_or_text_equal(
                        write.value,
                        post_values.get(write.address),
                        write.tolerance,
                    )
                    and _numeric_or_text_equal(
                        0,
                        post_values.get(enable_write.address),
                        0.0,
                    )
                ):
                    return AdapterStepResult(
                        False,
                        False,
                        "processor_parameter_readback_failed_while_bypassed",
                        {
                            "writes": sent,
                            "verified_values": verified_values,
                            "parameter_address": write.address,
                            "parameter_expected": write.value,
                            "parameter_actual": post_values.get(write.address),
                            "bypass_guard_actual": post_values.get(
                                enable_write.address
                            ),
                            "parameter_queries": post_evidence,
                        },
                    )
                verified_values[write.address] = post_values.get(write.address)
        return AdapterStepResult(
            True,
            False,
            "wing_write_sent",
            {"writes": sent, "verified_values": verified_values},
        )

    async def verify(
        self,
        operation: ConsoleTransactionOperation,
        snapshot: Mapping[str, Any],
        apply_result: AdapterStepResult,
    ) -> AdapterStepResult:
        writes = self._writes(operation)
        audible_reads = (
            *self._audible_reads(operation),
            *self._physical_guard_reads(operation),
        )
        physical, values, evidence = await self._physical_values(
            (*writes, *audible_reads)
        )
        mismatches = []
        if physical:
            for write in writes:
                actual = values.get(write.address)
                if not _numeric_or_text_equal(write.value, actual, write.tolerance):
                    mismatches.append(
                        {"address": write.address, "expected": write.value, "actual": actual}
                    )
            for read in audible_reads:
                actual = values.get(read.address)
                if not _numeric_or_text_equal(
                    read.value,
                    actual,
                    read.tolerance,
                ):
                    mismatches.append(
                        {
                            "address": read.address,
                            "expected": read.value,
                            "actual": actual,
                            "read_only_audible_state": True,
                        }
                    )
        success = physical and not mismatches
        return AdapterStepResult(
            success,
            success,
            "physical_readback_verified" if success else "physical_readback_mismatch",
            {"values": values, "queries": evidence, "mismatches": mismatches},
        )

    def _audio_failure_retention_eligibility(
        self,
        operation: ConsoleTransactionOperation,
        snapshot: Mapping[str, Any],
        writes: Sequence[WingParameterWrite],
    ) -> tuple[bool, Dict[str, Any]]:
        """Classify whether retaining this target can only reduce level.

        This is deliberately narrower than normal live admission.  Processor
        enables, dynamics/time/spatial changes, and ambiguous multi-parameter
        operations are never retained after an audio-safety failure.
        """

        old_values = dict(snapshot.get("values") or {})
        evidence: Dict[str, Any] = {
            "retention_eligible": False,
            "target_physical_verified": False,
            "write_attempted": False,
            "changed_addresses": [],
            "attenuation_addresses": [],
        }
        if operation.metadata.get("m110_processor_enable_bundle") is True:
            evidence["eligibility_reason"] = "processor_enable_not_retainable"
            return False, evidence
        if any(write.address not in old_values for write in writes):
            evidence["eligibility_reason"] = "retention_snapshot_incomplete"
            return False, evidence

        changed: list[str] = []
        attenuation: list[str] = []
        for write in writes:
            address = write.address
            old_value = old_values[address]
            target = write.value
            if _numeric_or_text_equal(
                old_value,
                target,
                write.tolerance,
            ):
                continue
            changed.append(address)
            if address.endswith("/mute"):
                safe = bool(
                    len(writes) == 1
                    and _numeric_or_text_equal(old_value, 0, 0.0)
                    and _numeric_or_text_equal(target, 1, 0.0)
                )
            else:
                try:
                    old_number = float(old_value)
                    target_number = float(target)
                except (TypeError, ValueError):
                    safe = False
                else:
                    numeric_cut = bool(
                        math.isfinite(old_number)
                        and math.isfinite(target_number)
                        and target_number < old_number
                    )
                    fader_or_trim_or_send = bool(
                        address.endswith("/fdr")
                        or address.endswith("/in/set/trim")
                        or (
                            "/send/" in address
                            and address.endswith("/lvl")
                        )
                    )
                    makeup_only = bool(
                        len(writes) == 1
                        and address.endswith("/dyn/gain")
                    )
                    eq_gain_only = bool(
                        "/eq/" in address
                        and address.endswith("g")
                    )
                    safe = bool(
                        numeric_cut
                        and (
                            fader_or_trim_or_send
                            or makeup_only
                            or eq_gain_only
                        )
                    )
            if not safe:
                evidence.update(
                    {
                        "changed_addresses": changed,
                        "attenuation_addresses": attenuation,
                        "eligibility_reason": (
                            "audio_failure_target_not_monotonic_attenuation"
                        ),
                    }
                )
                return False, evidence
            attenuation.append(address)

        eligible = bool(changed and attenuation and changed == attenuation)
        evidence.update(
            {
                "retention_eligible": eligible,
                "changed_addresses": changed,
                "attenuation_addresses": attenuation,
                "eligibility_reason": (
                    "monotonic_attenuation_target"
                    if eligible
                    else "no_attenuation_change_to_retain"
                ),
            }
        )
        return eligible, evidence

    async def assess_audio_failure_retention(
        self,
        operation: ConsoleTransactionOperation,
        snapshot: Mapping[str, Any],
        apply_result: AdapterStepResult,
    ) -> AdapterStepResult:
        """Read-only proof that an eligible attenuation target is retained."""

        writes = self._writes(operation)
        eligible, eligibility = self._audio_failure_retention_eligibility(
            operation,
            snapshot,
            writes,
        )
        if not eligible:
            return AdapterStepResult(
                False,
                bool(snapshot.get("physical_verified") is True),
                str(
                    eligibility.get("eligibility_reason")
                    or "audio_failure_retention_ineligible"
                ),
                eligibility,
            )
        try:
            physical, current, queries = await self._physical_values(writes)
        except SafetyThreadCallCancelled as exc:
            return AdapterStepResult(
                False,
                False,
                "audio_failure_retention_read_cancelled",
                {
                    **eligibility,
                    "target_physical_verified": False,
                    "cancellation_requested": True,
                    "query_error": (
                        type(exc.error).__name__ if exc.error else None
                    ),
                },
            )
        except Exception as exc:
            return AdapterStepResult(
                False,
                False,
                "audio_failure_retention_read_exception",
                {
                    **eligibility,
                    "target_physical_verified": False,
                    "exception_type": type(exc).__name__,
                },
            )
        mismatches = [
            write.address
            for write in writes
            if not _numeric_or_text_equal(
                write.value,
                current.get(write.address),
                write.tolerance,
            )
        ] if physical else [write.address for write in writes]
        verified = bool(physical and not mismatches)
        return AdapterStepResult(
            verified,
            verified,
            (
                "audio_failure_safe_target_retained"
                if verified
                else "audio_failure_retention_target_unverified"
            ),
            {
                **eligibility,
                "target_physical_verified": verified,
                "values": current,
                "queries": queries,
                "mismatches": mismatches,
            },
        )

    async def rollback(
        self,
        operation: ConsoleTransactionOperation,
        snapshot: Mapping[str, Any],
        apply_result: AdapterStepResult,
    ) -> AdapterStepResult:
        writes = self._writes(operation)
        fixed_preimage_restore = bool(
            operation.metadata.get("m110_fixed_preimage_restore") is True
            and self._fixed_preimage_restore_authorized
        )
        physical_guards = (
            self._physical_guard_reads(operation)
            if fixed_preimage_restore
            else ()
        )
        if (
            operation.metadata.get("m110_processor_enable_bundle") is True
            and isinstance(operation.target_value, Mapping)
            and operation.target_value.get("m110_processor_block")
            in {"hpf", "eq", "gate", "compressor"}
        ):
            return await self._rollback_processor_block(
                operation,
                snapshot,
                apply_result,
                writes,
            )
        physical, current, compare_evidence = await self._physical_values(
            (*writes, *physical_guards)
        )
        if not physical:
            return AdapterStepResult(False, False, "rollback_cas_read_failed", {"queries": compare_evidence})
        initial_guard_mismatches = [
            guard.address
            for guard in physical_guards
            if not _numeric_or_text_equal(
                guard.value,
                current.get(guard.address),
                guard.tolerance,
            )
        ]
        if initial_guard_mismatches:
            return AdapterStepResult(
                False,
                False,
                "m110_fixed_preimage_restore_rollback_guard_changed",
                {
                    "queries": compare_evidence,
                    "guard_mismatches": initial_guard_mismatches,
                    "restore_addresses": [],
                },
            )
        old_values = dict(snapshot.get("values") or {})
        if any(write.address not in old_values for write in writes):
            return AdapterStepResult(False, False, "rollback_snapshot_incomplete")

        write_events = apply_result.evidence.get("writes")
        journal_candidate_addresses = {
            str(item.get("address"))
            for item in write_events
            if isinstance(item, Mapping) and item.get("sent") in {True, None}
        } if isinstance(write_events, Sequence) else set()
        write_state_unknown = bool(
            apply_result.evidence.get("write_state_unknown")
        )
        candidate_addresses = (
            {write.address for write in writes}
            if not isinstance(write_events, Sequence)
            else journal_candidate_addresses
        )
        verified_values = dict(apply_result.evidence.get("verified_values") or {})
        restore_addresses = []
        cas_mismatches = []
        for write in writes:
            address = write.address
            actual = current.get(address)
            old_value = old_values[address]
            if _numeric_or_text_equal(old_value, actual, write.tolerance):
                continue
            if address not in candidate_addresses:
                cas_mismatches.append(address)
                continue
            expected_current = verified_values.get(address, write.value)
            if _numeric_or_text_equal(
                expected_current,
                actual,
                write.tolerance,
            ):
                restore_addresses.append(address)
            else:
                cas_mismatches.append(address)
        send_failures = []
        per_address_cas_queries = []
        cas_read_failures = []
        rollback_guard_failures = []
        for write in reversed(writes):
            if write.address not in restore_addresses:
                continue
            physical_now, current_now, current_evidence = await self._physical_values(
                (write, *physical_guards)
            )
            per_address_cas_queries.append(
                {
                    "address": write.address,
                    "query": current_evidence.get(write.address),
                }
            )
            if not physical_now:
                cas_read_failures.append(write.address)
                continue
            guard_mismatches = [
                guard.address
                for guard in physical_guards
                if not _numeric_or_text_equal(
                    guard.value,
                    current_now.get(guard.address),
                    guard.tolerance,
                )
            ]
            if guard_mismatches:
                rollback_guard_failures.extend(guard_mismatches)
                break
            current_value = current_now.get(write.address)
            old_value = old_values[write.address]
            if _numeric_or_text_equal(old_value, current_value, write.tolerance):
                continue
            expected_current = verified_values.get(write.address, write.value)
            if not _numeric_or_text_equal(
                expected_current,
                current_value,
                write.tolerance,
            ):
                if write.address not in cas_mismatches:
                    cas_mismatches.append(write.address)
                continue
            try:
                ok = await _safety_thread_call(
                    self.wing_client.send,
                    write.address,
                    old_value,
                )
            except SafetyThreadCallCancelled as exc:
                send_failures.append(
                    {
                        "address": write.address,
                        "reason": "rollback_send_cancelled",
                        "send_completed": exc.error is None,
                        "send_result": exc.outcome,
                        "error": type(exc.error).__name__ if exc.error else None,
                    }
                )
                continue
            except Exception as exc:
                send_failures.append(
                    {
                        "address": write.address,
                        "exception_type": type(exc).__name__,
                        "reason": "rollback_send_exception",
                    }
                )
                continue
            if not ok:
                send_failures.append(
                    {
                        "address": write.address,
                        "reason": "rollback_send_failed",
                    }
                )

        physical, restored, restore_evidence = await self._physical_values(
            (*writes, *physical_guards)
        )
        final_guard_mismatches = [
            guard.address
            for guard in physical_guards
            if not _numeric_or_text_equal(
                guard.value,
                restored.get(guard.address),
                guard.tolerance,
            )
        ] if physical else [guard.address for guard in physical_guards]
        mismatches = []
        if physical:
            for write in writes:
                expected = old_values[write.address]
                actual = restored.get(write.address)
                if not _numeric_or_text_equal(expected, actual, write.tolerance):
                    mismatches.append(write.address)
        success = (
            physical
            and not mismatches
            and not cas_mismatches
            and not cas_read_failures
            and not send_failures
            and not rollback_guard_failures
            and not final_guard_mismatches
        )
        if success:
            reason = (
                "rollback_physical_verified"
                if restore_addresses
                else "rollback_not_required"
            )
        elif rollback_guard_failures or final_guard_mismatches:
            reason = "m110_fixed_preimage_restore_rollback_guard_changed"
        elif cas_mismatches:
            reason = "rollback_cas_target_changed"
        elif cas_read_failures:
            reason = "rollback_cas_read_failed"
        elif send_failures:
            reason = "rollback_send_failed"
        else:
            reason = "rollback_physical_mismatch"
        return AdapterStepResult(
            success,
            success,
            reason,
            {
                "restored": restored,
                "queries": restore_evidence,
                "initial_compare_queries": compare_evidence,
                "mismatches": mismatches,
                "cas_mismatches": cas_mismatches,
                "cas_read_failures": cas_read_failures,
                "per_address_cas_queries": per_address_cas_queries,
                "send_failures": send_failures,
                "rollback_guard_failures": rollback_guard_failures,
                "final_guard_mismatches": final_guard_mismatches,
                "restore_addresses": restore_addresses,
            },
        )

    async def _rollback_processor_block(
        self,
        operation: ConsoleTransactionOperation,
        snapshot: Mapping[str, Any],
        apply_result: AdapterStepResult,
        writes: Sequence[WingParameterWrite],
    ) -> AdapterStepResult:
        """Fail-safe processor rollback: bypass first, then restore params."""

        old_values = dict(snapshot.get("values") or {})
        if any(write.address not in old_values for write in writes):
            return AdapterStepResult(False, False, "rollback_snapshot_incomplete")
        enable_write = writes[-1]
        disable_events = []
        try:
            disable_ok = await _safety_thread_call(
                self.wing_client.send,
                enable_write.address,
                0,
            )
            disable_events.append(
                {
                    "address": enable_write.address,
                    "value": 0,
                    "sent": bool(disable_ok),
                }
            )
        except SafetyThreadCallCancelled as exc:
            disable_events.append(
                {
                    "address": enable_write.address,
                    "value": 0,
                    "sent": bool(exc.outcome) if exc.error is None else None,
                    "cancellation_requested": True,
                    "error": type(exc.error).__name__ if exc.error else None,
                }
            )
            disable_ok = False
        except Exception as exc:
            disable_events.append(
                {
                    "address": enable_write.address,
                    "value": 0,
                    "sent": None,
                    "error": type(exc).__name__,
                }
            )
            disable_ok = False
        bypass_physical, bypass_values, bypass_evidence = (
            await self._physical_values(
                (WingParameterWrite(enable_write.address, 0, 0.0),)
            )
        )
        bypass_verified = bool(
            bypass_physical
            and _numeric_or_text_equal(
                0,
                bypass_values.get(enable_write.address),
                0.0,
            )
        )
        if not bypass_verified:
            return AdapterStepResult(
                False,
                False,
                "processor_rollback_disable_not_verified",
                {
                    "disable_first": True,
                    "disable_events": disable_events,
                    "disable_queries": bypass_evidence,
                    "write_state_unknown": True,
                },
            )

        write_events = apply_result.evidence.get("writes")
        candidate_addresses = {
            str(item.get("address"))
            for item in write_events
            if isinstance(item, Mapping) and item.get("sent") in {True, None}
        } if isinstance(write_events, Sequence) else {
            write.address for write in writes
        }
        verified_values = dict(
            apply_result.evidence.get("verified_values") or {}
        )
        restore_events = []
        cas_mismatches = []
        cas_read_failures = []
        send_failures = []
        # Enable was already forced to the snapshot value (the private lane is
        # admitted only when that value is 0). Restore every touched parameter
        # individually so one failed query cannot delay the initial bypass.
        for write in reversed(tuple(writes[:-1])):
            if write.address not in candidate_addresses:
                continue
            physical_now, values_now, query_evidence = await self._physical_values(
                (write, WingParameterWrite(enable_write.address, 0, 0.0))
            )
            restore_events.append(
                {
                    "address": write.address,
                    "query": query_evidence.get(write.address),
                    "bypass_query": query_evidence.get(enable_write.address),
                }
            )
            if not (
                physical_now
                and _numeric_or_text_equal(
                    0,
                    values_now.get(enable_write.address),
                    0.0,
                )
            ):
                cas_read_failures.append(write.address)
                continue
            current_value = values_now.get(write.address)
            old_value = old_values[write.address]
            if _numeric_or_text_equal(
                old_value,
                current_value,
                write.tolerance,
            ):
                continue
            expected_current = verified_values.get(write.address, write.value)
            if not _numeric_or_text_equal(
                expected_current,
                current_value,
                write.tolerance,
            ):
                cas_mismatches.append(write.address)
                continue
            try:
                ok = await _safety_thread_call(
                    self.wing_client.send,
                    write.address,
                    old_value,
                )
            except SafetyThreadCallCancelled as exc:
                send_failures.append(
                    {
                        "address": write.address,
                        "reason": "rollback_send_cancelled",
                        "send_result": exc.outcome,
                        "error": type(exc.error).__name__ if exc.error else None,
                    }
                )
                continue
            except Exception as exc:
                send_failures.append(
                    {
                        "address": write.address,
                        "reason": "rollback_send_exception",
                        "exception_type": type(exc).__name__,
                    }
                )
                continue
            if not ok:
                send_failures.append(
                    {
                        "address": write.address,
                        "reason": "rollback_send_failed",
                    }
                )

        physical, restored, restore_evidence = await self._physical_values(writes)
        mismatches = []
        if physical:
            for write in writes:
                if not _numeric_or_text_equal(
                    old_values[write.address],
                    restored.get(write.address),
                    write.tolerance,
                ):
                    mismatches.append(write.address)
        success = bool(
            physical
            and not mismatches
            and not cas_mismatches
            and not cas_read_failures
            and not send_failures
        )
        return AdapterStepResult(
            success,
            success,
            (
                "rollback_physical_verified"
                if success
                else "processor_rollback_incomplete"
            ),
            {
                "disable_first": True,
                "disable_events": disable_events,
                "disable_queries": bypass_evidence,
                "restore_events": restore_events,
                "restored": restored,
                "queries": restore_evidence,
                "mismatches": mismatches,
                "cas_mismatches": cas_mismatches,
                "cas_read_failures": cas_read_failures,
                "send_failures": send_failures,
            },
        )

    async def audio_recheck(
        self,
        operation: ConsoleTransactionOperation,
        snapshot: Mapping[str, Any],
        apply_result: AdapterStepResult,
    ) -> AudioRecheckResult:
        if self._audio_recheck is None:
            return AudioRecheckResult(
                "unavailable",
                "audio_recheck_callback_required",
            )
        result = self._audio_recheck(operation, snapshot, apply_result.evidence)
        if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
            result = await result
        if not isinstance(result, AudioRecheckResult):
            raise TypeError("audio_recheck_must_return_AudioRecheckResult")
        return result


SUPPORTED_WING_SUPERVISED_FAMILIES = (
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
)


def build_wing_supervised_family_adapters(
    wing_client: Any,
    *,
    audio_recheck: AudioRecheck | None = None,
    write_authority_current: WriteAuthorityCurrent | None = None,
    before_processor_enable: BeforeProcessorEnable | None = None,
    processor_enable_authorized: bool = False,
    constructive_lane_authorized: bool = False,
    fixed_preimage_restore_authorized: bool = False,
    query_timeout: float = 0.5,
) -> Dict[str, WingSupervisedFamilyAdapter]:
    return {
        family_id: WingSupervisedFamilyAdapter(
            family_id,
            wing_client,
            audio_recheck=audio_recheck,
            write_authority_current=write_authority_current,
            before_processor_enable=before_processor_enable,
            processor_enable_authorized=processor_enable_authorized,
            constructive_lane_authorized=constructive_lane_authorized,
            fixed_preimage_restore_authorized=(
                fixed_preimage_restore_authorized
            ),
            query_timeout=query_timeout,
        )
        for family_id in SUPPORTED_WING_SUPERVISED_FAMILIES
    }
