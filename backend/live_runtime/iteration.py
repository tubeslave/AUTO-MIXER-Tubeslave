"""One-hypothesis live iteration with Critic verification and verified rollback.

This module transfers the studio autonomous-iteration pattern into the live
runtime without importing studio editing/mastering behavior.  It deliberately
owns no mixer transport: writes and restoration stay behind the runtime service
or control-plane interface supplied by ``IterationRuntime``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import time
from typing import Any, Callable, Protocol

from .control_plane import RollbackExecution, WriteExecution
from .decision_engine import LiveHypothesis, verify as default_critic


class IterationRuntime(Protocol):
    """Minimal live boundary required by the iteration coordinator."""

    def execute_action(self, action, *, manual_freeze: bool = False) -> WriteExecution: ...

    def rollback_action(
        self,
        verified,
        *,
        manual_freeze: bool = False,
    ) -> RollbackExecution: ...


class IterationPhase(str, Enum):
    APPLY_BLOCKED = "apply_blocked"
    VERIFY_PENDING = "verify_pending"
    VERIFY_WAIT = "verify_wait"
    KEPT = "kept"
    ROLLED_BACK = "rolled_back"
    HOLD = "hold"


@dataclass(frozen=True)
class IterationResult:
    phase: IterationPhase
    hypothesis: LiveHypothesis
    execution: WriteExecution
    critic: dict[str, Any] | None = None
    rollback: RollbackExecution | None = None
    reason: str | None = None


@dataclass
class _ActiveIteration:
    hypothesis: LiveHypothesis
    execution: WriteExecution
    before_metrics: dict[str, Any]
    applied_at_s: float


class IterationCoordinator:
    """Allow exactly one live hypothesis to be in flight at a time.

    A hypothesis is applied through the authoritative live runtime, observed for
    a short configurable window, then judged by the live Critic.  Regression is
    restored through the already-verified rollback path.  If restoration fails,
    or the operator takes control during verification, the coordinator enters a
    HOLD state and refuses further autonomous iterations for the session.

    The coordinator never sleeps.  The realtime loop supplies new metrics and
    calls :meth:`verify`; calls made before the verification window expires
    return ``VERIFY_WAIT``.  This keeps timing causal and testable instead of
    blocking the audio/control loop.
    """

    def __init__(
        self,
        runtime: IterationRuntime,
        *,
        critic: Callable[[dict[str, Any], dict[str, Any], LiveHypothesis], dict[str, Any]] = default_critic,
        verification_window_s: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        audit_sink: Callable[[dict[str, Any]], None] | None = None,
    ):
        if verification_window_s < 0:
            raise ValueError("verification_window_s must be >= 0")
        self._runtime = runtime
        self._critic = critic
        self._verification_window_s = float(verification_window_s)
        self._clock = clock
        self._audit_sink = audit_sink
        self._active: _ActiveIteration | None = None
        self._hold_reason: str | None = None

    @property
    def has_active_hypothesis(self) -> bool:
        return self._active is not None

    @property
    def hold_reason(self) -> str | None:
        return self._hold_reason

    def _audit(self, payload: dict[str, Any]) -> None:
        if self._audit_sink is not None:
            self._audit_sink(dict(payload))

    def _ensure_available(self) -> None:
        if self._hold_reason is not None:
            raise RuntimeError(f"Live iteration is on HOLD: {self._hold_reason}")
        if self._active is not None:
            raise RuntimeError("A live hypothesis is already awaiting verification")

    def _hold(
        self,
        active: _ActiveIteration,
        *,
        reason: str,
        critic: dict[str, Any] | None = None,
        rollback: RollbackExecution | None = None,
    ) -> IterationResult:
        self._active = None
        self._hold_reason = reason
        result = IterationResult(
            phase=IterationPhase.HOLD,
            hypothesis=active.hypothesis,
            execution=active.execution,
            critic=critic,
            rollback=rollback,
            reason=reason,
        )
        self._audit(
            {
                "event": "live_iteration_hold",
                "reason": reason,
                "hypothesis": active.hypothesis.name,
                "critic": critic,
                "rollback_restored": rollback.restored if rollback is not None else None,
            }
        )
        return result

    def start(
        self,
        hypothesis: LiveHypothesis,
        before_metrics: dict[str, Any],
        *,
        manual_freeze: bool = False,
    ) -> IterationResult:
        """Apply one hypothesis or fail closed before a verify window begins."""

        self._ensure_available()
        execution = self._runtime.execute_action(
            hypothesis.action,
            manual_freeze=manual_freeze,
        )

        if not execution.wrote:
            result = IterationResult(
                phase=IterationPhase.APPLY_BLOCKED,
                hypothesis=hypothesis,
                execution=execution,
                reason=execution.authorization_reason,
            )
            self._audit(
                {
                    "event": "live_iteration_apply_blocked",
                    "hypothesis": hypothesis.name,
                    "reason": execution.authorization_reason,
                }
            )
            return result

        active = _ActiveIteration(
            hypothesis=hypothesis,
            execution=execution,
            before_metrics=dict(before_metrics),
            applied_at_s=float(self._clock()),
        )

        # A transport write whose immediate readback did not match must never be
        # handed to the perceptual Critic as a valid experimental state. Restore
        # it first using the captured pre-write value.
        if not execution.verified.accepted:
            rollback = self._runtime.rollback_action(
                execution.verified,
                manual_freeze=manual_freeze,
            )
            if rollback.restored:
                result = IterationResult(
                    phase=IterationPhase.ROLLED_BACK,
                    hypothesis=hypothesis,
                    execution=execution,
                    rollback=rollback,
                    reason="apply_readback_mismatch",
                )
                self._audit(
                    {
                        "event": "live_iteration_apply_mismatch_rolled_back",
                        "hypothesis": hypothesis.name,
                    }
                )
                return result
            return self._hold(
                active,
                reason="apply_readback_mismatch_rollback_failed",
                rollback=rollback,
            )

        self._active = active
        result = IterationResult(
            phase=IterationPhase.VERIFY_PENDING,
            hypothesis=hypothesis,
            execution=execution,
        )
        self._audit(
            {
                "event": "live_iteration_verify_pending",
                "hypothesis": hypothesis.name,
                "verify_metric": hypothesis.verify_metric,
                "verification_window_s": self._verification_window_s,
                "action": asdict(hypothesis.action),
            }
        )
        return result

    def verify(
        self,
        after_metrics: dict[str, Any],
        *,
        manual_freeze: bool = False,
    ) -> IterationResult:
        """Judge the active hypothesis after its verification window.

        Operator intervention has priority over autonomy: if the Critic reports
        ``operator_took_control``, the coordinator does not fight the operator by
        issuing rollback.  It enters HOLD and requires a new live session/control
        decision before autonomous iteration can continue.
        """

        if self._hold_reason is not None:
            raise RuntimeError(f"Live iteration is on HOLD: {self._hold_reason}")
        active = self._active
        if active is None:
            raise RuntimeError("No live hypothesis is awaiting verification")

        elapsed = float(self._clock()) - active.applied_at_s
        if elapsed < self._verification_window_s:
            return IterationResult(
                phase=IterationPhase.VERIFY_WAIT,
                hypothesis=active.hypothesis,
                execution=active.execution,
                reason="verification_window_open",
            )

        critic = dict(self._critic(
            active.before_metrics,
            dict(after_metrics),
            active.hypothesis,
        ))
        failures = list(critic.get("failures") or [])

        # Manual touch priority is a live safety invariant.  Rollback after an
        # operator has taken control could undo the operator's corrective move.
        if "operator_took_control" in failures:
            return self._hold(
                active,
                reason="operator_took_control",
                critic=critic,
            )

        if bool(critic.get("accept")):
            self._active = None
            result = IterationResult(
                phase=IterationPhase.KEPT,
                hypothesis=active.hypothesis,
                execution=active.execution,
                critic=critic,
            )
            self._audit(
                {
                    "event": "live_iteration_kept",
                    "hypothesis": active.hypothesis.name,
                    "critic": critic,
                }
            )
            return result

        rollback = self._runtime.rollback_action(
            active.execution.verified,
            manual_freeze=manual_freeze,
        )
        if rollback.restored:
            self._active = None
            result = IterationResult(
                phase=IterationPhase.ROLLED_BACK,
                hypothesis=active.hypothesis,
                execution=active.execution,
                critic=critic,
                rollback=rollback,
                reason="critic_rejected",
            )
            self._audit(
                {
                    "event": "live_iteration_rolled_back",
                    "hypothesis": active.hypothesis.name,
                    "critic": critic,
                }
            )
            return result

        return self._hold(
            active,
            reason="critic_rejected_rollback_failed",
            critic=critic,
            rollback=rollback,
        )
