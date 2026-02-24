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
    - apply absolute gate (-70 LUFS)
    - apply relative gate (-10 dB from gated mean)
    - average in linear domain and convert back to LUFS
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
        return self._integrated_from_samples(window.samples, fallback=-100.0)

    def get(self, channel_id: int, fallback: float = -100.0) -> float:
        """Get current rolling integrated LUFS without adding new sample."""
        window = self._channels.get(channel_id)
        if not window or not window.samples:
            return fallback

        return self._integrated_from_samples(window.samples, fallback=fallback)

    @staticmethod
    def _integrated_from_samples(
        samples: Deque[Tuple[float, float]],
        fallback: float = -100.0,
    ) -> float:
        """BS.1770-style two-stage gating over cached LUFS samples."""
        values = [value for _, value in samples]
        if not values:
            return fallback

        # Stage 1: absolute gate
        abs_gated = [v for v in values if v > -70.0]
        if not abs_gated:
            return fallback

        abs_linear = [10.0 ** (v / 10.0) for v in abs_gated]
        abs_mean_linear = sum(abs_linear) / max(len(abs_linear), 1)
        if abs_mean_linear <= 1e-12:
            return fallback

        abs_mean_lufs = 10.0 * math.log10(abs_mean_linear)

        # Stage 2: relative gate
        rel_threshold = abs_mean_lufs - 10.0
        rel_gated = [v for v in abs_gated if v > rel_threshold]
        if not rel_gated:
            return abs_mean_lufs

        rel_linear = [10.0 ** (v / 10.0) for v in rel_gated]
        rel_mean_linear = sum(rel_linear) / max(len(rel_linear), 1)
        if rel_mean_linear <= 1e-12:
            return fallback

        return 10.0 * math.log10(rel_mean_linear)
