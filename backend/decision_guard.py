"""Stateful decision gate. Proposed/rejected/failed writes are not applied state."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import math
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class GuardConfig:
    min_improvement: float = 0.02
    uncertainty_penalty: float = 1.0
    max_uncertainty: float = 0.5
    confirmations: int = 2
    reversal_hold_sec: float = 15.0
    reversal_improvement_multiplier: float = 2.0
    confirmation_window_sec: float = 10.0
    max_keys: int = 2048

    def __post_init__(self) -> None:
        numbers = (self.min_improvement, self.uncertainty_penalty, self.max_uncertainty,
                   self.reversal_hold_sec, self.reversal_improvement_multiplier,
                   self.confirmation_window_sec)
        if not all(math.isfinite(x) and x >= 0 for x in numbers):
            raise ValueError("Invalid guard limits")
        if (self.min_improvement <= 0 or self.confirmations < 1 or self.max_keys < 1
                or self.reversal_improvement_multiplier < 1 or self.confirmation_window_sec <= 0):
            raise ValueError("Invalid guard policy")


@dataclass
class ReviewDecision:
    allowed: bool
    reason: str
    report: dict[str, Any] = field(default_factory=dict)
    _commit: Callable[[], None] | None = field(default=None, repr=False)

    def commit(self) -> None:
        """Call once, only after success (or explicitly simulated success)."""
        if self.allowed and self._commit is not None:
            callback, self._commit = self._commit, None
            callback()


class DecisionGuard:
    def __init__(self, config: GuardConfig | None = None):
        self.config = config or GuardConfig()
        self._pending: OrderedDict[tuple, tuple] = OrderedDict()
        self._applied: OrderedDict[tuple, tuple] = OrderedDict()
        self._last_clock = -math.inf

    def _bound_memory(self) -> None:
        for memory in (self._pending, self._applied):
            while len(memory) > self.config.max_keys:
                memory.popitem(last=False)

    def review(
        self, key: tuple, *, changes: Mapping[str, float], before: float, after: float,
        uncertainty: float, frame_id: str, now: float, signature: str = "",
    ) -> ReviewDecision:
        """Changes are expressed in units of each parameter's deadband.

        A fresh audio frame is required per confirmation. Reversal history is
        tracked per physical parameter, not merely by an action's class name.
        """
        report: dict[str, Any] = {}

        def reject(reason: str, reset: bool = True) -> ReviewDecision:
            if reset:
                self._pending.pop(key, None)
            return ReviewDecision(False, reason, report)

        values = (before, after, uncertainty, now, *changes.values())
        if (not all(math.isfinite(x) for x in values) or min(before, after, uncertainty) < 0
                or not frame_id or now < self._last_clock):
            return reject("invalid_evidence")
        self._last_clock = now
        directions = {name: (1 if delta > 0 else -1)
                      for name, delta in changes.items() if abs(delta) >= 1.0}
        if not directions:
            return reject("deadband")
        improvement = before - after
        net = improvement - self.config.uncertainty_penalty * uncertainty
        report.update(before=before, after=after, improvement=improvement,
                      uncertainty=uncertainty, net_improvement=net)
        if uncertainty > self.config.max_uncertainty:
            return reject("uncertainty_too_high")
        if improvement <= 0 or net < self.config.min_improvement:
            return reject("no_significant_improvement")
        reverses = False
        for parameter, direction in directions.items():
            previous = self._applied.get((key, parameter))
            if previous and previous[0] != direction:
                reverses = True
                if now - previous[1] < self.config.reversal_hold_sec:
                    return reject("reversal_hold")
        if reverses and net < (self.config.min_improvement
                               * self.config.reversal_improvement_multiplier):
            return reject("reversal_hysteresis")
        token = (tuple(sorted(directions.items())), signature)
        pending = self._pending.get(key)
        if pending and pending[0] == token and now - pending[3] <= self.config.confirmation_window_sec:
            frames = pending[2] | {frame_id}
        else:
            frames = {frame_id}
        # There is no reason to retain unbounded frame hashes after confirmation.
        if len(frames) > self.config.confirmations:
            frames = {frame_id, *sorted(frames - {frame_id})[:self.config.confirmations - 1]}
        count = len(frames)
        self._pending[key] = (token, count, frames, now)
        self._pending.move_to_end(key)
        self._bound_memory()
        report["confirmations"] = count
        if count < self.config.confirmations:
            return reject("awaiting_fresh_confirmation", reset=False)

        def commit() -> None:
            for parameter, direction in directions.items():
                item_key = (key, parameter)
                self._applied[item_key] = (direction, now)
                self._applied.move_to_end(item_key)
            self._pending.pop(key, None)
            self._bound_memory()

        return ReviewDecision(True, "approved", report, commit)
