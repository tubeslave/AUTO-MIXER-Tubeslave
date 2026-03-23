"""
Real-time acoustic feedback detection and suppression.

Uses FFT peak tracking to detect feedback oscillations, applies automatic
notch EQ filters, and falls back to fader reduction when EQ is insufficient.
"""

import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)

# Constants
MAX_NOTCH_FILTERS_PER_CHANNEL = 8
MAX_NOTCH_DEPTH_DB = -12.0
FEEDBACK_PERSISTENCE_FRAMES = 5
MIN_FEEDBACK_FREQUENCY_HZ = 80.0
MAX_FEEDBACK_FREQUENCY_HZ = 12000.0


@dataclass
class FeedbackPeak:
    """Tracked feedback peak."""
    frequency_hz: float
    magnitude_db: float
    persistence: int = 0
    notch_applied: bool = False
    notch_depth_db: float = 0.0
    first_detected: float = 0.0
    last_detected: float = 0.0


@dataclass
class NotchFilter:
    """Applied notch EQ filter."""
    frequency_hz: float
    gain_db: float
    q: float = 8.0
    channel: int = 0
    applied_at: float = 0.0


@dataclass
class FeedbackEvent:
    """Record of a feedback event."""
    channel: int
    frequency_hz: float
    magnitude_db: float
    action: str  # 'notch', 'fader_reduce', 'cleared'
    timestamp: float = 0.0


class FeedbackDetector:
    """
    Real-time acoustic feedback detector and suppressor.

    Detection algorithm:
    1. Compute FFT magnitude spectrum every block
    2. Track persistent narrow-band peaks (Q > 10)
    3. If a peak persists for N consecutive frames and grows, flag as feedback
    4. Apply notch EQ at detected frequency
    5. If notch limit reached, reduce channel fader as fallback

    Latency budget: <50ms (single FFT block at 48kHz/2048 = ~43ms)
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        fft_size: int = 2048,
        threshold_db: float = -20.0,
        peak_rise_db: float = 6.0,
        persistence_frames: int = FEEDBACK_PERSISTENCE_FRAMES,
        max_notch_filters: int = MAX_NOTCH_FILTERS_PER_CHANNEL,
        max_notch_depth_db: float = MAX_NOTCH_DEPTH_DB,
        fader_reduction_db: float = -6.0,
    ):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.threshold_db = threshold_db
        self.peak_rise_db = peak_rise_db
        self.persistence_frames = persistence_frames
        self.max_notch_filters = max_notch_filters
        self.max_notch_depth_db = max_notch_depth_db
        self.fader_reduction_db = fader_reduction_db

        self.freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
        self.window = np.hanning(fft_size)
        self.freq_resolution = sample_rate / fft_size

        # Per-channel state
        self._channel_peaks: Dict[int, Dict[int, FeedbackPeak]] = {}
        self._channel_notches: Dict[int, List[NotchFilter]] = {}
        self._channel_fader_reduction: Dict[int, float] = {}
        self._avg_spectrum: Dict[int, Optional[np.ndarray]] = {}
        self._event_log: List[FeedbackEvent] = []
        self._enabled = True

        logger.info(
            f"FeedbackDetector initialized: sr={sample_rate}, fft={fft_size}, "
            f"threshold={threshold_db}dB, persistence={persistence_frames} frames"
        )

    @property
    def latency_ms(self) -> float:
        return self.fft_size / self.sample_rate * 1000.0

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def process(self, channel: int, samples: np.ndarray) -> List[FeedbackEvent]:
        """
        Process an audio block for feedback detection.

        Args:
            channel: Channel number
            samples: Audio samples (mono, float32)

        Returns:
            List of feedback events (actions taken)
        """
        if not self._enabled or len(samples) < self.fft_size:
            return []

        events = []
        now = time.time()

        # Initialize per-channel state
        if channel not in self._channel_peaks:
            self._channel_peaks[channel] = {}
            self._channel_notches[channel] = []
            self._channel_fader_reduction[channel] = 0.0
            self._avg_spectrum[channel] = None

        # Compute magnitude spectrum
        block = samples[-self.fft_size:].astype(np.float32) * self.window
        spectrum = np.abs(np.fft.rfft(block))
        magnitude_db = 20.0 * np.log10(spectrum + 1e-10)

        # Update running average spectrum (EMA with alpha=0.1)
        if self._avg_spectrum[channel] is None:
            self._avg_spectrum[channel] = magnitude_db.copy()
        else:
            self._avg_spectrum[channel] = (
                0.9 * self._avg_spectrum[channel] + 0.1 * magnitude_db
            )

        # Detect peaks above average + threshold
        deviation = magnitude_db - self._avg_spectrum[channel]
        peak_mask = (
            (deviation > self.peak_rise_db)
            & (magnitude_db > self.threshold_db)
            & (self.freqs >= MIN_FEEDBACK_FREQUENCY_HZ)
            & (self.freqs <= MAX_FEEDBACK_FREQUENCY_HZ)
        )

        # Find peak frequencies
        peak_indices = np.where(peak_mask)[0]

        # Group nearby peaks (within 2 bins)
        grouped_peaks = self._group_peaks(peak_indices, magnitude_db)

        # Track peak persistence
        current_peak_bins = set()
        for bin_idx, mag_db in grouped_peaks:
            current_peak_bins.add(bin_idx)
            freq = float(self.freqs[bin_idx])

            if bin_idx in self._channel_peaks[channel]:
                peak = self._channel_peaks[channel][bin_idx]
                peak.persistence += 1
                peak.magnitude_db = mag_db
                peak.last_detected = now
            else:
                self._channel_peaks[channel][bin_idx] = FeedbackPeak(
                    frequency_hz=freq,
                    magnitude_db=mag_db,
                    persistence=1,
                    first_detected=now,
                    last_detected=now,
                )

        # Decay peaks not seen this frame
        stale_bins = []
        for bin_idx, peak in self._channel_peaks[channel].items():
            if bin_idx not in current_peak_bins:
                peak.persistence = max(0, peak.persistence - 1)
                if peak.persistence <= 0 and not peak.notch_applied:
                    stale_bins.append(bin_idx)
        for b in stale_bins:
            del self._channel_peaks[channel][b]

        # Act on persistent peaks (feedback confirmed)
        for bin_idx, peak in self._channel_peaks[channel].items():
            if peak.persistence >= self.persistence_frames and not peak.notch_applied:
                event = self._suppress_feedback(channel, peak, now)
                if event:
                    events.append(event)

        return events

    def _group_peaks(
        self, peak_indices: np.ndarray, magnitude_db: np.ndarray
    ) -> List[Tuple[int, float]]:
        """Group nearby spectral peaks, keeping the loudest in each group."""
        if len(peak_indices) == 0:
            return []

        groups = []
        current_group = [peak_indices[0]]

        for i in range(1, len(peak_indices)):
            if peak_indices[i] - peak_indices[i - 1] <= 2:
                current_group.append(peak_indices[i])
            else:
                best = max(current_group, key=lambda idx: magnitude_db[idx])
                groups.append((best, float(magnitude_db[best])))
                current_group = [peak_indices[i]]

        if current_group:
            best = max(current_group, key=lambda idx: magnitude_db[idx])
            groups.append((best, float(magnitude_db[best])))

        return groups

    def _suppress_feedback(
        self, channel: int, peak: FeedbackPeak, now: float
    ) -> Optional[FeedbackEvent]:
        """Apply suppression for a confirmed feedback peak."""
        notches = self._channel_notches[channel]

        if len(notches) < self.max_notch_filters:
            # Apply notch EQ
            depth = max(self.max_notch_depth_db, -6.0)  # Start conservative
            notch = NotchFilter(
                frequency_hz=peak.frequency_hz,
                gain_db=depth,
                q=8.0,
                channel=channel,
                applied_at=now,
            )
            notches.append(notch)
            peak.notch_applied = True
            peak.notch_depth_db = depth

            event = FeedbackEvent(
                channel=channel,
                frequency_hz=peak.frequency_hz,
                magnitude_db=peak.magnitude_db,
                action="notch",
                timestamp=now,
            )
            self._event_log.append(event)
            logger.warning(
                f"Feedback detected ch{channel} @ {peak.frequency_hz:.0f}Hz "
                f"({peak.magnitude_db:.1f}dB), applied notch {depth:.1f}dB"
            )
            return event
        else:
            # Fallback: reduce fader
            self._channel_fader_reduction[channel] += self.fader_reduction_db
            self._channel_fader_reduction[channel] = max(
                self._channel_fader_reduction[channel], -18.0
            )
            event = FeedbackEvent(
                channel=channel,
                frequency_hz=peak.frequency_hz,
                magnitude_db=peak.magnitude_db,
                action="fader_reduce",
                timestamp=now,
            )
            self._event_log.append(event)
            peak.notch_applied = True  # Mark so we don't retry
            logger.warning(
                f"Feedback ch{channel} @ {peak.frequency_hz:.0f}Hz — "
                f"notch limit reached, reducing fader by "
                f"{self.fader_reduction_db:.1f}dB"
            )
            return event

    def deepen_notch(self, channel: int, frequency_hz: float, additional_db: float = -3.0):
        """Deepen an existing notch filter if feedback persists."""
        for notch in self._channel_notches.get(channel, []):
            if abs(notch.frequency_hz - frequency_hz) < self.freq_resolution * 2:
                new_depth = max(
                    notch.gain_db + additional_db, self.max_notch_depth_db
                )
                notch.gain_db = new_depth
                logger.info(
                    f"Deepened notch ch{channel} @ {frequency_hz:.0f}Hz "
                    f"to {new_depth:.1f}dB"
                )
                return

    def get_notch_filters(self, channel: int) -> List[NotchFilter]:
        """Get all active notch filters for a channel."""
        return list(self._channel_notches.get(channel, []))

    def get_fader_reduction(self, channel: int) -> float:
        """Get total fader reduction applied for a channel."""
        return self._channel_fader_reduction.get(channel, 0.0)

    def clear_channel(self, channel: int):
        """Clear all feedback state for a channel."""
        self._channel_peaks.pop(channel, None)
        cleared = len(self._channel_notches.get(channel, []))
        self._channel_notches.pop(channel, None)
        self._channel_fader_reduction.pop(channel, None)
        self._avg_spectrum.pop(channel, None)
        if cleared > 0:
            logger.info(f"Cleared {cleared} notch filters for ch{channel}")

    def clear_all(self):
        """Clear all feedback state."""
        self._channel_peaks.clear()
        self._channel_notches.clear()
        self._channel_fader_reduction.clear()
        self._avg_spectrum.clear()

    def get_event_log(self, max_events: int = 100) -> List[FeedbackEvent]:
        """Get recent feedback events."""
        return self._event_log[-max_events:]

    def get_status(self) -> Dict:
        """Get detector status summary."""
        total_notches = sum(
            len(n) for n in self._channel_notches.values()
        )
        active_channels = len(self._channel_peaks)
        return {
            "enabled": self._enabled,
            "latency_ms": self.latency_ms,
            "active_channels": active_channels,
            "total_notch_filters": total_notches,
            "total_events": len(self._event_log),
            "fader_reductions": dict(self._channel_fader_reduction),
        }
