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
    return f"{number:.1f}"


def _channel_writes(operation: ConsoleTransactionOperation) -> Tuple[WingParameterWrite, ...]:
    channel = _channel(operation)
    target = operation.target_value
    base = f"/ch/{channel}"

    if operation.family_id == "gain_trim" and operation.write_kind in {"gain", "trim"}:
        return (WingParameterWrite(f"{base}/in/set/trim", _target_number(target, -18.0, 18.0, "trim_db")),)
    if operation.family_id == "fader" and operation.write_kind == "fader":
        return (WingParameterWrite(f"{base}/fdr", _target_number(target, -144.0, 0.0, "fader_db")),)
    if operation.family_id == "mute" and operation.write_kind == "mute":
        return (WingParameterWrite(f"{base}/mute", _target_bool(target), 0.0),)
    if operation.family_id == "eq_hpf" and operation.write_kind == "hpf":
        values = _mapping(target)
        hpf_hz = _target_number(values.get("freq_hz"), 20.0, 2000.0, "hpf_hz")
        return (
            WingParameterWrite(f"{base}/flt/lcf", hpf_hz, max(1.0, hpf_hz * 0.005)),
            WingParameterWrite(f"{base}/flt/lc", _target_bool(values.get("enabled")), 0.0),
        )
    if operation.family_id == "eq_hpf" and operation.write_kind == "eq":
        values = _mapping(target)
        band = _target_int(values.get("band"), 1, 4, "eq_band")
        eq_hz = _target_number(values.get("freq_hz"), 20.0, 20000.0, "eq_freq_hz")
        return (
            WingParameterWrite(f"{base}/eq/{band}f", eq_hz, max(1.0, eq_hz * 0.005)),
            WingParameterWrite(f"{base}/eq/{band}g", _target_number(values.get("gain_db"), -15.0, 0.0, "eq_gain_db")),
            WingParameterWrite(f"{base}/eq/{band}q", _target_number(values.get("q"), 0.44, 10.0, "eq_q")),
        )
    if operation.family_id == "gate_expander" and operation.write_kind == "gate":
        values = _mapping(target)
        attack_ms = _target_number(values.get("attack_ms"), 0.0, 120.0, "gate_attack_ms")
        hold_ms = _target_number(values.get("hold_ms"), 0.0, 2000.0, "gate_hold_ms")
        release_ms = _target_number(values.get("release_ms"), 4.0, 4000.0, "gate_release_ms")
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

    if operation.family_id == "eq_hpf" and operation.write_kind == "eq":
        return (
            WingParameterWrite(f"/ch/{_channel(operation)}/eq/on", 1, 0.0),
        )
    if operation.family_id == "bus_group_dca_matrix":
        values = _mapping(operation.target_value)
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
        query_timeout: float = 0.5,
    ) -> None:
        self.family_id = family_id
        self.wing_client = wing_client
        self._audio_recheck = audio_recheck
        self._write_authority_current = write_authority_current
        self.query_timeout = max(0.05, float(query_timeout))

    def _writes(self, operation: ConsoleTransactionOperation) -> Tuple[WingParameterWrite, ...]:
        if operation.family_id != self.family_id:
            raise ValueError("family_adapter_identity_mismatch")
        return parameter_writes_for_operation(operation)

    def prepare(self, operation: ConsoleTransactionOperation) -> Mapping[str, Any]:
        """Pure validation/address expansion performed before physical I/O."""
        writes = self._writes(operation)
        audible_reads = audible_state_reads_for_operation(operation)
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
        audible_reads = audible_state_reads_for_operation(operation)
        old_values = dict(snapshot.get("values") or {})
        if any(
            read.address not in old_values
            for read in (*writes, *audible_reads)
        ):
            return "apply_snapshot_incomplete"

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
                    return "fader_boost_not_allowed"
                maximum_cut = 6.0 if operation.family_id == "fader" else 3.0
                if old_number - target_number > maximum_cut:
                    return "fader_cut_delta_exceeded"
            elif "/send/" in address and address.endswith("/lvl"):
                if target_number > old_number + write.tolerance:
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
        audible_reads = audible_state_reads_for_operation(operation)
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
        audible_reads = audible_state_reads_for_operation(operation)
        old_values = dict(snapshot.get("values") or {})
        if any(write.address not in old_values for write in writes):
            return AdapterStepResult(False, False, "apply_snapshot_incomplete")
        transition_error = self.validate_snapshot(operation, snapshot)
        if transition_error:
            return AdapterStepResult(False, False, transition_error)
        sent = []
        for write in writes:
            precondition_physical, precondition_values, precondition_evidence = (
                await self._physical_values((write, *audible_reads))
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
                    "processing_enable_state_change_requires_separate_approval",
                    {
                        "writes": sent,
                        "audible_state_queries": precondition_evidence,
                    },
                )
            if (
                self._write_authority_current is not None
                and self._write_authority_current() is not True
            ):
                return AdapterStepResult(
                    False,
                    False,
                    "write_authority_expired_before_send",
                    {"writes": sent},
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
                    },
                )
            sent.append({"address": write.address, "value": write.value, "sent": bool(ok)})
            if not ok:
                return AdapterStepResult(
                    False,
                    False,
                    "wing_send_failed",
                    {"writes": sent},
                )
        return AdapterStepResult(True, False, "wing_write_sent", {"writes": sent})

    async def verify(
        self,
        operation: ConsoleTransactionOperation,
        snapshot: Mapping[str, Any],
        apply_result: AdapterStepResult,
    ) -> AdapterStepResult:
        writes = self._writes(operation)
        audible_reads = audible_state_reads_for_operation(operation)
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

    async def rollback(
        self,
        operation: ConsoleTransactionOperation,
        snapshot: Mapping[str, Any],
        apply_result: AdapterStepResult,
    ) -> AdapterStepResult:
        writes = self._writes(operation)
        physical, current, compare_evidence = await self._physical_values(writes)
        if not physical:
            return AdapterStepResult(False, False, "rollback_cas_read_failed", {"queries": compare_evidence})
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
        for write in reversed(writes):
            if write.address not in restore_addresses:
                continue
            physical_now, current_now, current_evidence = await self._physical_values(
                (write,)
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

        physical, restored, restore_evidence = await self._physical_values(writes)
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
        )
        if success:
            reason = (
                "rollback_physical_verified"
                if restore_addresses
                else "rollback_not_required"
            )
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
                "restore_addresses": restore_addresses,
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
    query_timeout: float = 0.5,
) -> Dict[str, WingSupervisedFamilyAdapter]:
    return {
        family_id: WingSupervisedFamilyAdapter(
            family_id,
            wing_client,
            audio_recheck=audio_recheck,
            write_authority_current=write_authority_current,
            query_timeout=query_timeout,
        )
        for family_id in SUPPORTED_WING_SUPERVISED_FAMILIES
    }
