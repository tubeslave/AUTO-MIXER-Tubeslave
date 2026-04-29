"""Dry-run-first OSC command queue for `production_mix_v1`."""

from __future__ import annotations

from typing import Any

from .models import (
    ACTION_COMPRESSION,
    ACTION_EQ,
    ACTION_FX_SEND,
    ACTION_GAIN,
    ACTION_PAN,
    ACTION_PARALLEL_COMPRESSION,
    MixAction,
    MixCandidate,
    OSCCommand,
    QueueFlushResult,
    SafetyResult,
)


class OSCCommandQueue:
    """Queue semantic OSC commands. It never sends unless explicitly flushed live."""

    def __init__(self, *, dry_run: bool = True):
        self.dry_run = bool(dry_run)
        self._commands: list[OSCCommand] = []

    @property
    def commands(self) -> list[OSCCommand]:
        return list(self._commands)

    def enqueue_candidate(self, candidate: MixCandidate, safety: SafetyResult) -> list[OSCCommand]:
        """Queue commands for Safety Governor-approved actions only."""
        if not safety.passed:
            return []
        created = []
        for action in safety.allowed_actions:
            commands = self._commands_for_action(action)
            self._commands.extend(commands)
            created.extend(commands)
        return created

    def flush(self, sender: Any | None = None) -> QueueFlushResult:
        """Dry-run or send queued commands through an explicitly injected sender."""
        if self.dry_run:
            return QueueFlushResult(
                dry_run=True,
                would_send=[command.to_dict() for command in self._commands],
                warnings=["dry_run_enabled_no_osc_sent"],
            )
        if sender is None:
            return QueueFlushResult(
                dry_run=False,
                blocked=[command.to_dict() for command in self._commands],
                warnings=["no_sender_injected_no_osc_sent"],
            )

        sent: list[dict[str, Any]] = []
        blocked: list[dict[str, Any]] = []
        for command in self._commands:
            try:
                if command.dry_run_only:
                    blocked.append(
                        {
                            "command": command.to_dict(),
                            "reason": "command_requires_adapter_translation",
                        }
                    )
                    continue
                if hasattr(sender, "send_message"):
                    sender.send_message(command.address, *command.args)
                elif hasattr(sender, "send"):
                    sender.send(command.address, command.args)
                else:
                    raise TypeError("sender has no send_message/send method")
                sent.append(command.to_dict())
            except Exception as exc:
                blocked.append({"command": command.to_dict(), "reason": str(exc)})
        return QueueFlushResult(dry_run=False, sent=sent, blocked=blocked)

    def _commands_for_action(self, action: MixAction) -> list[OSCCommand]:
        channel = int(action.channel_id or 0)
        if action.action_type == ACTION_GAIN:
            if action.channel_id is None:
                return [
                    OSCCommand(
                        address="/main/mix/fader",
                        args=[float(action.parameters.get("gain_db", 0.0))],
                        action=action.to_dict(),
                        dry_run_only=True,
                        rationale=(
                            "Semantic mix-bus linear trim queued for offline/live adapter; "
                            "not a raw absolute WING fader value."
                        ),
                    )
                ]
            return [
                OSCCommand(
                    address=f"/ch/{channel}/mix/fader",
                    args=[float(action.parameters.get("gain_db", 0.0))],
                    action=action.to_dict(),
                    dry_run_only=True,
                    rationale=(
                        "Semantic gain delta queued for OSC adapter conversion; not raw WING fader."
                    ),
                )
            ]
        if action.action_type == ACTION_EQ:
            if action.channel_id is None:
                band = int(action.parameters.get("band", 4))
                return [
                    OSCCommand(
                        address=f"/main/eq/{band}/f",
                        args=[float(action.parameters.get("frequency_hz", 10000.0))],
                        action=action.to_dict(),
                        dry_run_only=True,
                        rationale=(
                            "Semantic mix-bus EQ queued for offline/live adapter; not raw console state."
                        ),
                    ),
                    OSCCommand(
                        address=f"/main/eq/{band}/g",
                        args=[float(action.parameters.get("gain_db", 0.0))],
                        action=action.to_dict(),
                        dry_run_only=True,
                        rationale=(
                            "Semantic mix-bus EQ queued for offline/live adapter; not raw console state."
                        ),
                    ),
                ]
            band = int(action.parameters.get("band", 2))
            return [
                OSCCommand(
                    address=f"/ch/{channel}/eq/on",
                    args=[1],
                    action=action.to_dict(),
                    dry_run_only=False,
                    rationale=action.rationale,
                ),
                OSCCommand(
                    address=f"/ch/{channel}/eq/{band}/f",
                    args=[float(action.parameters.get("frequency_hz", 1000.0))],
                    action=action.to_dict(),
                    dry_run_only=False,
                    rationale=action.rationale,
                ),
                OSCCommand(
                    address=f"/ch/{channel}/eq/{band}/g",
                    args=[float(action.parameters.get("gain_db", 0.0))],
                    action=action.to_dict(),
                    dry_run_only=False,
                    rationale=action.rationale,
                ),
                OSCCommand(
                    address=f"/ch/{channel}/eq/{band}/q",
                    args=[float(action.parameters.get("q", 1.0))],
                    action=action.to_dict(),
                    dry_run_only=False,
                    rationale=action.rationale,
                ),
            ]
        if action.action_type == ACTION_COMPRESSION:
            return [
                OSCCommand(
                    address=f"/ch/{channel}/dyn/on",
                    args=[1],
                    action=action.to_dict(),
                    dry_run_only=False,
                    rationale=action.rationale,
                ),
                OSCCommand(
                    address=f"/ch/{channel}/dyn/thr26",
                    args=[float(action.parameters.get("threshold_db", -18.0))],
                    action=action.to_dict(),
                    dry_run_only=False,
                    rationale=action.rationale,
                ),
            ]
        if action.action_type == ACTION_PARALLEL_COMPRESSION:
            return [
                OSCCommand(
                    address=f"/ch/{channel}/dyn/parallel",
                    args=[
                        float(action.parameters.get("threshold_db", -24.0)),
                        float(action.parameters.get("ratio", 3.0)),
                        float(action.parameters.get("wet_mix_percent", 20.0)),
                    ],
                    action=action.to_dict(),
                    dry_run_only=True,
                    rationale=(
                        "Semantic parallel compression recipe requires adapter translation before live use."
                    ),
                )
            ]
        if action.action_type == ACTION_FX_SEND:
            recipe = str(action.parameters.get("fx_slot", action.parameters.get("recipe", "FX15")))
            return [
                OSCCommand(
                    address=f"/ch/{channel}/send/{recipe}",
                    args=[float(action.parameters.get("send_level_db", -18.0))],
                    action=action.to_dict(),
                    dry_run_only=True,
                    rationale=(
                        "Semantic FX send/depth command queued for future OSC adapter; dry-run by default."
                    ),
                )
            ]
        if action.action_type == ACTION_PAN:
            pan = float(action.parameters.get("pan", 0.0))
            return [
                OSCCommand(
                    address=f"/ch/{channel}/mix/pan",
                    args=[0.5 + pan * 0.5],
                    action=action.to_dict(),
                    dry_run_only=False,
                    rationale=action.rationale,
                )
            ]
        return []
