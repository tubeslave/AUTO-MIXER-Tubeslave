"""Authoritative live write seam for WING and future mixer adapters.

The old Automixer lets several controllers reach mixer writes through their own
policies. The new live runtime needs one explicit boundary: a proposed action is
authorized by the selected :class:`LiveMode`, resolved against fresh mixer
state when necessary, written through an injected hardware adapter, read back,
and recorded as a verified action.

This module intentionally does not know OSC addresses. WING/OSC/Dante-specific
translation belongs in adapters, while the mode policy and verification order
stay common and testable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Callable, Protocol

from .contracts import LiveMode, ProposedAction, VerifiedAction
from .safety_governor import authorize


class MixerWriteAdapter(Protocol):
    """Minimal contract required by the live control plane.

    ``read_value`` must be side-effect free. ``write_value`` is the only method
    in this contract allowed to mutate the mixer.
    """

    def read_value(self, action: ProposedAction) -> Any: ...

    def write_value(self, action: ProposedAction) -> Any: ...


@dataclass(frozen=True)
class WriteExecution:
    """Result of one authorization/write/readback cycle."""

    verified: VerifiedAction
    authorization_reason: str
    wrote: bool


def _matches_expected(expected: Any, actual: Any, *, tolerance: float) -> bool:
    """Compare a requested value with mixer readback without hiding mismatch.

    Mixer protocols often quantize floating-point values. Numeric readback is
    therefore compared with a small explicit tolerance; strings/bools and other
    typed values must match exactly.
    """

    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not (math.isfinite(float(expected)) and math.isfinite(float(actual))):
            return expected == actual
        return abs(float(expected) - float(actual)) <= tolerance
    return expected == actual


_RELATIVE_PARAMETERS = {
    "fader_delta_db": "fader_db",
    "eq_gain_delta_db": "eq_gain_db",
}


def _resolve_action(action: ProposedAction, before: Any) -> ProposedAction:
    """Resolve relative proposal semantics into a concrete mixer write.

    Directors reason naturally in bounded deltas for slow live moves. Hardware
    adapters receive absolute values only. Resolution happens after a fresh
    read, preventing a proposal such as ``-0.5`` from accidentally becoming an
    absolute fader or EQ-gain position.
    """

    resolved_parameter = _RELATIVE_PARAMETERS.get(action.parameter)
    if resolved_parameter is None:
        return action
    if isinstance(action.value, bool) or not isinstance(action.value, (int, float)):
        raise TypeError(f"{action.parameter} value must be numeric")
    if isinstance(before, bool) or not isinstance(before, (int, float)):
        raise TypeError(f"{action.parameter} requires numeric mixer readback")
    if action.parameter == "eq_gain_delta_db" and action.eq_locator is None:
        raise ValueError("eq_gain_delta_db requires an explicit EqBandLocator")
    delta = float(action.value)
    current = float(before)
    if not (math.isfinite(delta) and math.isfinite(current)):
        raise ValueError(f"{action.parameter} and current value must be finite")
    return replace(action, parameter=resolved_parameter, value=current + delta)


def _bounded_relative_action(action: ProposedAction) -> tuple[bool, str]:
    """Enforce the proposal's own max-step contract for relative moves."""

    if action.parameter not in _RELATIVE_PARAMETERS or action.max_step is None:
        return True, "within_step_bound"
    if isinstance(action.value, bool) or not isinstance(action.value, (int, float)):
        return False, "invalid_delta"
    if abs(float(action.value)) > abs(float(action.max_step)) + 1e-12:
        return False, "max_step_exceeded"
    return True, "within_step_bound"


class LiveControlPlane:
    """Single live-runtime authority for applying a proposed mixer action.

    BENCH_TEST deliberately bypasses the production allowlists/confidence gates
    in ``authorize`` so engineering decisions are visible on the physical WING.
    Read-before, proposal bounds, audit and readback verification still run.
    Production modes keep their policy checks. OBSERVE/PROPOSE/FREEZE never call
    ``write_value``.
    """

    def __init__(
        self,
        adapter: MixerWriteAdapter,
        *,
        audit_sink: Callable[[dict[str, Any]], None] | None = None,
        readback_tolerance: float = 0.02,
    ):
        if readback_tolerance < 0:
            raise ValueError("readback_tolerance must be >= 0")
        self._adapter = adapter
        self._audit_sink = audit_sink
        self._readback_tolerance = float(readback_tolerance)

    def _audit(self, payload: dict[str, Any]) -> None:
        if self._audit_sink is not None:
            self._audit_sink(payload)

    def execute(
        self,
        action: ProposedAction,
        mode: LiveMode,
        *,
        manual_freeze: bool = False,
    ) -> WriteExecution:
        """Authorize, resolve, write, read back and verify one action.

        The current value is read before authorization so blocked proposals are
        still auditable without mutating the console. Relative proposals are
        resolved against that fresh value. A successful transport write is not
        treated as success until readback matches the resolved target.
        """

        before = self._adapter.read_value(action)
        allowed, reason = authorize(action, mode, manual_freeze=manual_freeze)
        step_allowed, step_reason = _bounded_relative_action(action)
        if allowed and not step_allowed:
            allowed, reason = False, step_reason

        if not allowed:
            verified = VerifiedAction(
                proposal=action,
                before=before,
                after=before,
                readback=before,
                accepted=False,
                rollback_value=None,
            )
            self._audit(
                {
                    "event": "live_write_blocked",
                    "mode": mode.value,
                    "reason": reason,
                    "action": asdict(action),
                    "before": before,
                }
            )
            return WriteExecution(verified=verified, authorization_reason=reason, wrote=False)

        resolved = _resolve_action(action, before)
        self._adapter.write_value(resolved)
        readback = self._adapter.read_value(resolved)
        accepted = _matches_expected(
            resolved.value,
            readback,
            tolerance=self._readback_tolerance,
        )
        verified = VerifiedAction(
            proposal=action,
            before=before,
            after=resolved.value,
            readback=readback,
            accepted=accepted,
            rollback_value=before if action.reversible else None,
        )
        self._audit(
            {
                "event": "live_write_verified" if accepted else "live_write_mismatch",
                "mode": mode.value,
                "reason": reason,
                "action": asdict(action),
                "resolved_action": asdict(resolved),
                "before": before,
                "readback": readback,
                "accepted": accepted,
            }
        )
        return WriteExecution(verified=verified, authorization_reason=reason, wrote=True)
