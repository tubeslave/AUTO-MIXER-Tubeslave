"""Executor for Safety Gate-approved v2 action plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

from automixer.decision.models import (
    ACTION_NO_ACTION,
    ActionDecision,
    ActionPlan,
    jsonable,
)
from automixer.safety import SafetyGate, SafetyGateResult


@dataclass(frozen=True)
class ExecutionResult:
    """Result of attempting to execute a v2 plan."""

    safety: SafetyGateResult
    sent: list[Dict[str, Any]] = field(default_factory=list)
    recommended_only: list[Dict[str, Any]] = field(default_factory=list)
    blocked: list[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "safety": self.safety.to_dict(),
            "sent": jsonable(self.sent),
            "recommended_only": jsonable(self.recommended_only),
            "blocked": jsonable(self.blocked),
        }


class ActionPlanExecutor:
    """Translate gated actions into proposal-only execution results.

    Decision Engine v2 has no canonical supervised-write transaction or
    operation-specific authority binding. It therefore always evaluates and
    records actions as dry-run recommendations and never calls a mixer write
    method. Live writes remain owned by the repository's supervised console
    transaction paths.
    """

    def __init__(
        self,
        mixer_client: Any | None,
        safety_gate: SafetyGate,
        *,
        logger: Any | None = None,
        dry_run: bool = False,
    ):
        self.mixer_client = mixer_client
        self.safety_gate = safety_gate
        self.logger = logger
        self.dry_run = dry_run

    def execute(
        self,
        plan: ActionPlan,
        *,
        current_state: Mapping[str, Any] | None = None,
        live_mode: bool = True,
    ) -> ExecutionResult:
        live_dispatch_requested = not self.dry_run
        safety = self.safety_gate.evaluate_plan(
            plan,
            current_state=current_state,
            live_mode=live_mode,
            dry_run=True,
        )
        sent: list[Dict[str, Any]] = []
        recommended_only: list[Dict[str, Any]] = []
        blocked = list(safety.blocked)

        for decision in safety.allowed_plan.decisions:
            if decision.action_type == ACTION_NO_ACTION:
                recommended_only.append({"decision": decision.to_dict(), "reason": "no_action"})
                continue
            reason = (
                "live_write_authority_unavailable"
                if live_dispatch_requested
                else "dry_run"
            )
            recommended_only.append({"decision": decision.to_dict(), "reason": reason})
            self._log("recommended", decision, reason)

        return ExecutionResult(
            safety=safety,
            sent=sent,
            recommended_only=recommended_only,
            blocked=blocked,
        )

    def _log(self, event_type: str, decision: ActionDecision, message: str) -> None:
        if self.logger is None:
            return
        log_method = getattr(self.logger, "log_executor", None)
        if log_method is not None:
            log_method(event_type=event_type, decision=decision, message=message)
