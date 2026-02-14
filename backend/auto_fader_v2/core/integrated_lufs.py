"""Rolling integrated LUFS helpers for real-time control."""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Tuple


@dataclass
class ChannelLufsWindow:
    """Per-channel LUFS history for rolling integrated calculation."""

    samples: Deque[Tuple[float, float]] = field(default_factory=deque)  # (timestamp, lufs)


class RollingIntegratedLufs:
    """
    Calculates rolling integrated LUFS over a fixed time window.

    This is a practical approximation for real-time control:
    - keep last `window_seconds` LUFS samples per channel
    - average in linear domain
    - convert back to LUFS
    """

    def __init__(self, window_seconds: float = 3.0):
        self.window_seconds = max(0.5, float(window_seconds))
        self._channels: Dict[int, ChannelLufsWindow] = {}

    def reset(self) -> None:
        """Reset all channel windows."""
        self._channels.clear()

    def _prune(self, window: ChannelLufsWindow, now: float) -> None:
        cutoff = now - self.window_seconds
        while window.samples and window.samples[0][0] < cutoff:
            window.samples.popleft()

    def update(self, channel_id: int, lufs_value: float, now: float) -> float:
        """
        Add LUFS sample and return rolling integrated LUFS for channel.

        Args:
            channel_id: Mixer channel id
            lufs_value: Current LUFS sample (momentary/short/etc)
            now: Unix timestamp in seconds
        """
        if channel_id not in self._channels:
            self._channels[channel_id] = ChannelLufsWindow()

        window = self._channels[channel_id]
        window.samples.append((now, float(lufs_value)))
        self._prune(window, now)

        if not window.samples:
            return -100.0

        linear_values = [10.0 ** (value / 10.0) for _, value in window.samples]
        mean_linear = sum(linear_values) / max(len(linear_values), 1)
        if mean_linear <= 1e-12:
            return -100.0
        return 10.0 * math.log10(mean_linear)

    def get(self, channel_id: int, fallback: float = -100.0) -> float:
        """Get current rolling integrated LUFS without adding new sample."""
        window = self._channels.get(channel_id)
        if not window or not window.samples:
            return fallback

        linear_values = [10.0 ** (value / 10.0) for _, value in window.samples]
        mean_linear = sum(linear_values) / max(len(linear_values), 1)
        if mean_linear <= 1e-12:
            return fallback
        return 10.0 * math.log10(mean_linear)
