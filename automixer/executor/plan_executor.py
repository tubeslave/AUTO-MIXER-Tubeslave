"""Executor for Safety Gate-approved v2 action plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

from automixer.decision.models import (
    ACTION_COMPRESSION,
    ACTION_EQ,
    ACTION_GAIN,
    ACTION_NO_ACTION,
    ACTION_PAN,
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
    """Translate gated actions into mixer-client calls.

    The Executor never bypasses Safety Gate. In dry-run mode it records what
    would have been sent and performs no mixer write.
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
        safety = self.safety_gate.evaluate_plan(
            plan,
            current_state=current_state,
            live_mode=live_mode,
            dry_run=self.dry_run,
        )
        sent: list[Dict[str, Any]] = []
        recommended_only: list[Dict[str, Any]] = []
        blocked = list(safety.blocked)

        for decision in safety.allowed_plan.decisions:
            if decision.action_type == ACTION_NO_ACTION:
                recommended_only.append({"decision": decision.to_dict(), "reason": "no_action"})
                continue
            send_to_executor = bool(decision.metadata.get("safety_gate", {}).get("send_to_executor"))
            if not send_to_executor or self.dry_run or safety.dry_run:
                recommended_only.append({"decision": decision.to_dict(), "reason": "dry_run"})
                self._log("recommended", decision, "dry_run")
                continue
            ok, message = self._dispatch(decision)
            payload = {"decision": decision.to_dict(), "message": message}
            if ok:
                sent.append(payload)
                self._log("sent", decision, message)
            else:
                blocked.append({**payload, "reason": "executor_dispatch_failed"})
                self._log("blocked", decision, message)

        return ExecutionResult(
            safety=safety,
            sent=sent,
            recommended_only=recommended_only,
            blocked=blocked,
        )

    def _dispatch(self, decision: ActionDecision) -> tuple[bool, str]:
        if self.mixer_client is None:
            return False, "no_mixer_client"
        if decision.action_type == ACTION_GAIN:
            return self._dispatch_gain(decision)
        if decision.action_type == ACTION_EQ:
            return self._dispatch_eq(decision)
        if decision.action_type == ACTION_COMPRESSION:
            return self._dispatch_compression(decision)
        if decision.action_type == ACTION_PAN:
            return self._dispatch_pan(decision)
        return False, "unsupported_action_type"

    def _dispatch_gain(self, decision: ActionDecision) -> tuple[bool, str]:
        channel = _channel_id(decision)
        if channel is None:
            return False, "missing_channel_id"
        gain_db = float(decision.parameters.get("gain_db", 0.0))
        if hasattr(self.mixer_client, "get_fader") and hasattr(self.mixer_client, "set_fader"):
            current = float(self.mixer_client.get_fader(channel))
            target = max(-144.0, min(0.0, current + gain_db))
            result = self.mixer_client.set_fader(channel, target)
            return bool(result is not False), f"set_fader channel={channel} target_db={target:.2f}"
        if hasattr(self.mixer_client, "set_gain"):
            result = self.mixer_client.set_gain(channel, gain_db)
            return bool(result is not False), f"set_gain channel={channel} gain_db={gain_db:.2f}"
        return False, "mixer_has_no_gain_or_fader_method"

    def _dispatch_eq(self, decision: ActionDecision) -> tuple[bool, str]:
        channel = _channel_id(decision)
        if channel is None:
            return False, "missing_channel_id"
        if not hasattr(self.mixer_client, "set_eq_band"):
            return False, "mixer_has_no_set_eq_band"
        params = decision.parameters
        result = self.mixer_client.set_eq_band(
            channel,
            int(params.get("band", 2)),
            float(params.get("frequency_hz", 1000.0)),
            float(params.get("gain_db", 0.0)),
            float(params.get("q", 1.0)),
        )
        return bool(result is not False), f"set_eq_band channel={channel}"

    def _dispatch_compression(self, decision: ActionDecision) -> tuple[bool, str]:
        channel = _channel_id(decision)
        if channel is None:
            return False, "missing_channel_id"
        if not hasattr(self.mixer_client, "set_compressor"):
            return False, "mixer_has_no_set_compressor"
        params = decision.parameters
        try:
            result = self.mixer_client.set_compressor(
                channel,
                threshold_db=float(params.get("threshold_db", -18.0)),
                ratio=float(params.get("ratio", 2.0)),
                attack_ms=float(params.get("attack_ms", 15.0)),
                release_ms=float(params.get("release_ms", 160.0)),
                makeup_db=float(params.get("makeup_db", 0.0)),
                enabled=True,
            )
        except TypeError:
            result = self.mixer_client.set_compressor(
                channel,
                threshold=float(params.get("threshold_db", -18.0)),
                ratio=str(params.get("ratio", 2.0)),
                attack=float(params.get("attack_ms", 15.0)),
                release=float(params.get("release_ms", 160.0)),
                gain=float(params.get("makeup_db", 0.0)),
            )
        return bool(result is not False), f"set_compressor channel={channel}"

    def _dispatch_pan(self, decision: ActionDecision) -> tuple[bool, str]:
        channel = _channel_id(decision)
        if channel is None:
            return False, "missing_channel_id"
        if not hasattr(self.mixer_client, "set_pan"):
            return False, "mixer_has_no_set_pan"
        pan = float(decision.parameters.get("pan", 0.0))
        result = self.mixer_client.set_pan(channel, pan)
        return bool(result is not False), f"set_pan channel={channel} pan={pan:.3f}"

    def _log(self, event_type: str, decision: ActionDecision, message: str) -> None:
        if self.logger is None:
            return
        log_method = getattr(self.logger, "log_executor", None)
        if log_method is not None:
            log_method(event_type=event_type, decision=decision, message=message)


def _channel_id(decision: ActionDecision) -> int | None:
    value = decision.parameters.get("channel_id")
    if value is None and decision.target.startswith("channel:"):
        value = decision.target.split(":", 1)[1]
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
