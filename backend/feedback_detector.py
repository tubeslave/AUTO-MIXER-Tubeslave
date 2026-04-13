"""
Real-time acoustic feedback detection and suppression.

Uses FFT peak tracking to detect feedback oscillations, applies automatic
notch EQ filters, and falls back to fader reduction when EQ is insufficient.

This module has absolute priority for safety — it can reduce fader levels
without approval from other agents.
"""

import math
import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)

MAX_NOTCH_FILTERS = 8
MAX_NOTCH_DEPTH_DB = -12.0
MIN_FEEDBACK_FREQ_HZ = 80.0
MAX_FEEDBACK_FREQ_HZ = 12000.0


# ── Helper functions ─────────────────────────────────────────────

def _freq_close(
    freq_a: float, freq_b: float, tolerance_cents: float = 50.0
) -> bool:
    """Check if two frequencies are close within *tolerance_cents*."""
    if freq_a <= 0 or freq_b <= 0:
        return False
    ratio = freq_a / freq_b
    if ratio <= 0:
        return False
    cents = abs(1200.0 * math.log2(ratio))
    return cents <= tolerance_cents


def _q_for_feedback(confidence: float) -> float:
    """Compute notch Q factor from detection confidence (0..1).

    Higher confidence → narrower (higher Q) notch.
    Range: 4.0 (low confidence) → 10.0 (high confidence).
    """
    confidence = max(0.0, min(1.0, confidence))
    return 4.0 + 6.0 * confidence


def _find_spectral_peaks(
    magnitude_db: np.ndarray,
    height: float = -30.0,
    distance: int = 5,
    prominence: float = 6.0,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Find spectral peaks using simple local-maximum detection.

    Returns (peak_indices, properties_dict) similar to scipy.signal.find_peaks.
    """
    n = len(magnitude_db)
    peaks = []
    prominences = []

    for i in range(distance, n - distance):
        if magnitude_db[i] < height:
            continue
        is_peak = True
        for d in range(1, distance + 1):
            if magnitude_db[i] <= magnitude_db[i - d] or \
               magnitude_db[i] <= magnitude_db[i + d]:
                is_peak = False
                break
        if is_peak:
            left_min = np.min(magnitude_db[max(0, i - distance * 3):i])
            right_min = np.min(magnitude_db[i + 1:min(n, i + distance * 3 + 1)])
            prom = magnitude_db[i] - max(left_min, right_min)
            if prom >= prominence:
                peaks.append(i)
                prominences.append(prom)

    return (
        np.array(peaks, dtype=int),
        {"prominences": np.array(prominences)},
    )


# ── Data classes ─────────────────────────────────────────────────

@dataclass
class FeedbackPeak:
    """Tracked feedback peak (legacy compat)."""
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
    frequency: float
    gain_db: float
    q_factor: float = 8.0
    slot_index: int = 0
    applied_at: float = 0.0

    @property
    def frequency_hz(self) -> float:
        return self.frequency

    @property
    def channel(self) -> int:
        return 0

    @property
    def q(self) -> float:
        return self.q_factor


@dataclass
class FeedbackEvent:
    """Record of a feedback event."""
    channel: int
    frequency_hz: float
    magnitude_db: float
    action: str  # 'notch', 'fader_reduce', 'cleared'
    confidence: float = 0.0
    timestamp: float = 0.0
    notch_filter: Optional[NotchFilter] = None


@dataclass
class FeedbackDetectorConfig:
    """Configuration for the feedback detector."""
    sample_rate: int = 48000
    fft_size: int = 2048
    persistence_frames: int = 6
    min_confidence: float = 0.5
    peak_height_db: float = -20.0
    peak_prominence_db: float = 6.0
    peak_distance_bins: int = 5
    max_notch_filters: int = MAX_NOTCH_FILTERS
    max_notch_depth_db: float = MAX_NOTCH_DEPTH_DB
    fader_reduction_step_db: float = -3.0
    fader_reduction_floor_db: float = -18.0
    notch_stale_age_sec: float = 30.0
    freq_tolerance_cents: float = 50.0


@dataclass
class _PeakState:
    """Internal tracked peak state."""
    frequency: float
    magnitude_db: float
    persistence: int = 0
    confidence: float = 0.0
    magnitude_history: List[float] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0

    @property
    def age_sec(self) -> float:
        return self.last_seen - self.first_seen

    def is_growing(self) -> bool:
        if len(self.magnitude_history) < 3:
            return False
        recent = self.magnitude_history[-3:]
        return all(recent[i] < recent[i + 1] for i in range(len(recent) - 1))


class _ChannelFeedbackState:
    """Per-channel feedback tracking state."""

    def __init__(self, channel_id: int, config: FeedbackDetectorConfig):
        self.channel_id = channel_id
        self.config = config
        self.tracked_peaks: Dict[int, _PeakState] = {}
        self.notch_filters: List[NotchFilter] = []
        self.fader_reduction_db: float = 0.0
        self.avg_spectrum: Optional[np.ndarray] = None
        self.events: List[FeedbackEvent] = []
        self._next_slot = 0

    def add_notch(
        self, frequency: float, confidence: float, now: float = 0.0
    ) -> Optional[NotchFilter]:
        """Add or deepen a notch filter at *frequency*.

        Returns the NotchFilter, or None if max reached.
        """
        for nf in self.notch_filters:
            if _freq_close(nf.frequency, frequency, self.config.freq_tolerance_cents):
                old_gain = nf.gain_db
                nf.gain_db = max(
                    self.config.max_notch_depth_db,
                    nf.gain_db - 1.5,
                )
                nf.applied_at = now
                return nf

        if len(self.notch_filters) >= self.config.max_notch_filters:
            return None

        q = _q_for_feedback(confidence)
        nf = NotchFilter(
            frequency=frequency,
            gain_db=-3.0,
            q_factor=q,
            slot_index=self._next_slot,
            applied_at=now,
        )
        self._next_slot += 1
        self.notch_filters.append(nf)
        return nf

    def release_notch(self, slot_index: int) -> bool:
        """Remove notch filter by slot index."""
        for i, nf in enumerate(self.notch_filters):
            if nf.slot_index == slot_index:
                self.notch_filters.pop(i)
                return True
        return False

    def apply_fader_reduction(
        self, step_db: float = -3.0, floor_db: float = -18.0
    ) -> float:
        """Apply additional fader reduction. Returns cumulative reduction."""
        self.fader_reduction_db = max(
            floor_db, self.fader_reduction_db + step_db
        )
        return self.fader_reduction_db

    def get_stale_notches(
        self, now: float, max_age_sec: float = 30.0
    ) -> List[NotchFilter]:
        """Get notch filters older than *max_age_sec*."""
        return [
            nf for nf in self.notch_filters
            if (now - nf.applied_at) > max_age_sec
        ]


# ── Main detector ────────────────────────────────────────────────

class FeedbackDetector:
    """
    Real-time acoustic feedback detector and suppressor.

    Detection algorithm:
    1. Compute FFT magnitude spectrum every block
    2. Track persistent narrow-band peaks
    3. If a peak persists for N frames and is growing, flag as feedback
    4. Apply notch EQ at detected frequency
    5. If notch limit reached, reduce channel fader as fallback
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        fft_size: int = 2048,
        threshold_db: float = -20.0,
        peak_rise_db: float = 6.0,
        persistence_frames: int = 5,
        max_notch_filters: int = MAX_NOTCH_FILTERS,
        max_notch_depth_db: float = MAX_NOTCH_DEPTH_DB,
        fader_reduction_db: float = -6.0,
        config: Optional[FeedbackDetectorConfig] = None,
    ):
        if config:
            self.config = config
        else:
            self.config = FeedbackDetectorConfig(
                sample_rate=sample_rate,
                fft_size=fft_size,
                persistence_frames=persistence_frames,
                peak_height_db=threshold_db,
                peak_prominence_db=peak_rise_db,
                max_notch_filters=max_notch_filters,
                max_notch_depth_db=max_notch_depth_db,
                fader_reduction_step_db=-abs(fader_reduction_db) if fader_reduction_db < 0 else -3.0,
            )

        self.freqs = np.fft.rfftfreq(self.config.fft_size, 1.0 / self.config.sample_rate)
        self.window = np.hanning(self.config.fft_size)
        self.freq_resolution = self.config.sample_rate / self.config.fft_size
        self._channel_states: Dict[int, _ChannelFeedbackState] = {}
        self._enabled = True

        logger.info(
            f"FeedbackDetector initialized: sr={self.config.sample_rate}, "
            f"fft={self.config.fft_size}, persistence={self.config.persistence_frames}"
        )

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def fft_size(self) -> int:
        return self.config.fft_size

    @property
    def latency_ms(self) -> float:
        return self.config.fft_size / self.config.sample_rate * 1000.0

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def _get_or_create_state(self, channel: int) -> _ChannelFeedbackState:
        if channel not in self._channel_states:
            self._channel_states[channel] = _ChannelFeedbackState(
                channel_id=channel, config=self.config
            )
        return self._channel_states[channel]

    def process_audio(
        self, channel: int, samples: np.ndarray
    ) -> Optional[FeedbackEvent]:
        """Process an audio block.

        Returns a FeedbackEvent if action was taken, None otherwise.
        """
        if not self._enabled or len(samples) < self.config.fft_size:
            return None

        now = time.time()
        state = self._get_or_create_state(channel)

        block = samples[-self.config.fft_size:].astype(np.float32) * self.window
        spectrum = np.abs(np.fft.rfft(block))
        magnitude_db = 20.0 * np.log10(spectrum + 1e-10)

        if state.avg_spectrum is None:
            state.avg_spectrum = magnitude_db.copy()
        else:
            state.avg_spectrum = 0.9 * state.avg_spectrum + 0.1 * magnitude_db

        peak_indices, props = _find_spectral_peaks(
            magnitude_db,
            height=self.config.peak_height_db,
            distance=self.config.peak_distance_bins,
            prominence=self.config.peak_prominence_db,
        )

        if len(peak_indices) == 0:
            for pid in list(state.tracked_peaks.keys()):
                peak = state.tracked_peaks[pid]
                peak.persistence = max(0, peak.persistence - 1)
                if peak.persistence <= 0:
                    del state.tracked_peaks[pid]
            return None

        freq_mask = np.array([
            MIN_FEEDBACK_FREQ_HZ <= self.freqs[idx] <= MAX_FEEDBACK_FREQ_HZ
            for idx in peak_indices
        ])
        peak_indices = peak_indices[freq_mask]

        current_bins = set()
        for idx in peak_indices:
            current_bins.add(int(idx))
            freq = float(self.freqs[idx])
            mag = float(magnitude_db[idx])

            matched = False
            for pid, peak in state.tracked_peaks.items():
                if _freq_close(peak.frequency, freq, self.config.freq_tolerance_cents):
                    peak.persistence += 1
                    peak.magnitude_db = mag
                    peak.magnitude_history.append(mag)
                    if len(peak.magnitude_history) > 20:
                        peak.magnitude_history = peak.magnitude_history[-20:]
                    peak.last_seen = now
                    matched = True
                    break

            if not matched:
                state.tracked_peaks[int(idx)] = _PeakState(
                    frequency=freq,
                    magnitude_db=mag,
                    persistence=1,
                    magnitude_history=[mag],
                    first_seen=now,
                    last_seen=now,
                )

        stale = [pid for pid, p in state.tracked_peaks.items()
                 if pid not in current_bins]
        for pid in stale:
            peak = state.tracked_peaks[pid]
            peak.persistence = max(0, peak.persistence - 1)
            if peak.persistence <= 0:
                del state.tracked_peaks[pid]

        for pid, peak in state.tracked_peaks.items():
            if peak.persistence >= self.config.persistence_frames and peak.is_growing():
                confidence = min(
                    1.0,
                    peak.persistence / (self.config.persistence_frames * 2),
                )
                if confidence < self.config.min_confidence:
                    continue

                nf = state.add_notch(peak.frequency, confidence, now)
                if nf is not None:
                    event = FeedbackEvent(
                        channel=channel,
                        frequency_hz=peak.frequency,
                        magnitude_db=peak.magnitude_db,
                        action="notch",
                        confidence=confidence,
                        timestamp=now,
                        notch_filter=nf,
                    )
                    state.events.append(event)
                    return event
                else:
                    reduction = state.apply_fader_reduction(
                        step_db=self.config.fader_reduction_step_db,
                        floor_db=self.config.fader_reduction_floor_db,
                    )
                    event = FeedbackEvent(
                        channel=channel,
                        frequency_hz=peak.frequency,
                        magnitude_db=peak.magnitude_db,
                        action="fader_reduce",
                        confidence=confidence,
                        timestamp=now,
                    )
                    state.events.append(event)
                    return event

        return None

    def process(
        self, channel: int, samples: np.ndarray
    ) -> List[FeedbackEvent]:
        """Legacy API: process audio and return list of events."""
        event = self.process_audio(channel, samples)
        return [event] if event else []

    def get_active_notches(self, channel: int) -> List[NotchFilter]:
        state = self._channel_states.get(channel)
        if state is None:
            return []
        return list(state.notch_filters)

    def get_fader_reduction(self, channel: int) -> float:
        state = self._channel_states.get(channel)
        if state is None:
            return 0.0
        return state.fader_reduction_db

    def get_channel_state(self, channel: int) -> Optional[_ChannelFeedbackState]:
        return self._channel_states.get(channel)

    def reset_channel(self, channel: int):
        self._channel_states.pop(channel, None)

    def reset_all(self):
        self._channel_states.clear()

    def get_diagnostics(self, channel: int) -> Dict[str, Any]:
        state = self._channel_states.get(channel)
        if state is None:
            return {
                "channel_id": channel,
                "tracked_peaks": 0,
                "notch_filters": 0,
                "fader_reduction_db": 0.0,
                "events": 0,
            }
        return {
            "channel_id": channel,
            "tracked_peaks": len(state.tracked_peaks),
            "notch_filters": len(state.notch_filters),
            "fader_reduction_db": state.fader_reduction_db,
            "events": len(state.events),
            "peak_details": [
                {
                    "frequency": p.frequency,
                    "magnitude_db": p.magnitude_db,
                    "persistence": p.persistence,
                    "growing": p.is_growing(),
                }
                for p in state.tracked_peaks.values()
            ],
        }

    def generate_osc_commands(self, channel: int) -> List[Tuple[str, Any]]:
        """Generate WING OSC commands for active notch filters."""
        state = self._channel_states.get(channel)
        if state is None or not state.notch_filters:
            return []

        commands = []
        commands.append((f"/ch/{channel}/eq/on", 1))
        for nf in state.notch_filters:
            slot = (nf.slot_index % 4) + 1
            commands.append((f"/ch/{channel}/eq/{slot}f", nf.frequency))
            commands.append((f"/ch/{channel}/eq/{slot}g", nf.gain_db))
            commands.append((f"/ch/{channel}/eq/{slot}q", nf.q_factor))
        return commands

    def generate_reset_osc_commands(self, channel: int) -> List[Tuple[str, Any]]:
        """Generate OSC commands to reset EQ filters."""
        commands = []
        for band in range(1, 5):
            commands.append((f"/ch/{channel}/eq/{band}g", 0.0))
            commands.append((f"/ch/{channel}/eq/{band}f", 1000.0))
            commands.append((f"/ch/{channel}/eq/{band}q", 1.0))
        for band in range(1, 4):
            commands.append((f"/ch/{channel}/peq/{band}g", 0.0))
            commands.append((f"/ch/{channel}/peq/{band}f", 1000.0))
            commands.append((f"/ch/{channel}/peq/{band}q", 1.0))
        return commands
