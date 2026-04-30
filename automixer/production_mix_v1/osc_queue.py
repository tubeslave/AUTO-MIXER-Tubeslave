"""OSC command queue. Queueing is never sending."""

from __future__ import annotations

from .models import MixCandidate, OSCCommand, QueueFlushResult, SafetyResult


class OSCCommandQueue:
    def __init__(self, *, dry_run: bool = True):
        self.dry_run = bool(dry_run)
        self._commands: list[OSCCommand] = []

    def enqueue_candidate(self, candidate: MixCandidate, safety: SafetyResult) -> None:
        if not safety.passed:
            return
        for action in safety.allowed_actions:
            if action.channel_id is None:
                continue
            self._commands.append(
                OSCCommand(
                    address=f"/ch/{int(action.channel_id):02d}/{action.action_type}",
                    args=[action.parameters],
                    action=action.to_dict(),
                    dry_run_only=True,
                    rationale=action.rationale,
                )
            )

    def flush(self, sender=None) -> QueueFlushResult:
        if self.dry_run or sender is None:
            return QueueFlushResult(dry_run=self.dry_run, would_send=[cmd.to_dict() for cmd in self._commands], warnings=["dry_run_enabled_no_osc_sent"] if self._commands else [])
        sent = []
        for cmd in self._commands:
            sender.send(cmd.address, *cmd.args)
            sent.append(cmd.to_dict())
        return QueueFlushResult(dry_run=False, sent=sent)
