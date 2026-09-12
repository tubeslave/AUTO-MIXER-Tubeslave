"""Conservative real-time acoustic feedback detection.

Detection is intentionally separated from transport.  The detector works in
amplitude-normalised dBFS, tracks frequency-stable narrow-band peaks, sanitises
invalid samples, and rate-limits repeated mitigation events.  It never sends
OSC itself; command generation remains an explicit compatibility API.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

MAX_NOTCH_FILTERS = 8
MAX_NOTCH_DEPTH_DB = -12.0
MIN_FEEDBACK_FREQ_HZ = 80.0
MAX_FEEDBACK_FREQ_HZ = 12000.0
EPS = 1e-12


def _freq_close(freq_a: float, freq_b: float, tolerance_cents: float = 50.0) -> bool:
    if freq_a <= 0.0 or freq_b <= 0.0 or not np.isfinite(freq_a) or not np.isfinite(freq_b):
        return False
    return abs(1200.0 * math.log2(freq_a / freq_b)) <= tolerance_cents


def _q_for_feedback(confidence: float) -> float:
    confidence = float(np.clip(confidence if np.isfinite(confidence) else 0.0, 0.0, 1.0))
    return 4.0 + 6.0 * confidence


def _find_spectral_peaks(
    magnitude_db: np.ndarray,
    height: float = -30.0,
    distance: int = 5,
    prominence: float = 6.0,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    data = np.nan_to_num(np.asarray(magnitude_db, dtype=np.float64), nan=-240.0, posinf=-240.0, neginf=-240.0)
    n = data.size
    distance = max(1, int(distance))
    peaks: List[int] = []
    prominences: List[float] = []
    for i in range(distance, max(distance, n - distance)):
        if data[i] < height:
            continue
        if any(data[i] <= data[i - d] or data[i] <= data[i + d] for d in range(1, distance + 1)):
            continue
        lo = max(0, i - 3 * distance)
        hi = min(n, i + 3 * distance + 1)
        left = data[lo:i]
        right = data[i + 1:hi]
        if left.size == 0 or right.size == 0:
            continue
        local_floor = max(float(np.min(left)), float(np.min(right)))
        prom = float(data[i] - local_floor)
        if prom >= prominence:
            peaks.append(i)
            prominences.append(prom)
    return np.asarray(peaks, dtype=int), {"prominences": np.asarray(prominences, dtype=np.float64)}


@dataclass
class FeedbackPeak:
    frequency_hz: float
    magnitude_db: float
    persistence: int = 0
    notch_applied: bool = False
    notch_depth_db: float = 0.0
    first_detected: float = 0.0
    last_detected: float = 0.0


@dataclass
class NotchFilter:
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
    channel: int
    frequency_hz: float
    magnitude_db: float
    action: str
    confidence: float = 0.0
    timestamp: float = 0.0
    notch_filter: Optional[NotchFilter] = None


@dataclass
class FeedbackDetectorConfig:
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
    stable_level_std_db: float = 1.5
    action_cooldown_frames: int = 8
    max_console_notches: int = 4


@dataclass
class _PeakState:
    frequency: float
    magnitude_db: float
    persistence: int = 0
    confidence: float = 0.0
    magnitude_history: List[float] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    last_action_frame: int = -1000000

    @property
    def age_sec(self) -> float:
        return self.last_seen - self.first_seen

    def is_growing(self) -> bool:
        if len(self.magnitude_history) < 3:
            return False
        recent = self.magnitude_history[-3:]
        return all(recent[i] < recent[i + 1] for i in range(len(recent) - 1))

    def is_stable(self, max_std_db: float = 1.5) -> bool:
        if len(self.magnitude_history) < 4:
            return False
        recent = np.asarray(self.magnitude_history[-6:], dtype=np.float64)
        return bool(np.all(np.isfinite(recent)) and float(np.std(recent)) <= max_std_db)


class _ChannelFeedbackState:
    def __init__(self, channel_id: int, config: FeedbackDetectorConfig):
        self.channel_id = channel_id
        self.config = config
        self.tracked_peaks: Dict[int, _PeakState] = {}
        self.notch_filters: List[NotchFilter] = []
        self.fader_reduction_db = 0.0
        self.avg_spectrum: Optional[np.ndarray] = None
        self.events: List[FeedbackEvent] = []
        self._next_slot = 0
        self.frame_index = 0

    def add_notch(self, frequency: float, confidence: float, now: float = 0.0) -> Optional[NotchFilter]:
        for nf in self.notch_filters:
            if _freq_close(nf.frequency, frequency, self.config.freq_tolerance_cents):
                nf.gain_db = max(self.config.max_notch_depth_db, nf.gain_db - 1.5)
                nf.applied_at = now
                return nf
        if len(self.notch_filters) >= self.config.max_notch_filters:
            return None
        nf = NotchFilter(
            frequency=float(frequency),
            gain_db=-3.0,
            q_factor=_q_for_feedback(confidence),
            slot_index=self._next_slot,
            applied_at=now,
        )
        self._next_slot += 1
        self.notch_filters.append(nf)
        return nf

    def release_notch(self, slot_index: int) -> bool:
        for index, nf in enumerate(self.notch_filters):
            if nf.slot_index == slot_index:
                self.notch_filters.pop(index)
                return True
        return False

    def apply_fader_reduction(self, step_db: float = -3.0, floor_db: float = -18.0) -> float:
        self.fader_reduction_db = max(float(floor_db), self.fader_reduction_db + float(step_db))
        return self.fader_reduction_db

    def get_stale_notches(self, now: float, max_age_sec: float = 30.0) -> List[NotchFilter]:
        return [nf for nf in self.notch_filters if now - nf.applied_at > max_age_sec]


class FeedbackDetector:
    """Track persistent narrow-band oscillations while rejecting broadband noise."""

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
        self.config = config or FeedbackDetectorConfig(
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
        self.window = np.hanning(self.config.fft_size).astype(np.float64)
        self._window_amplitude_norm = max(float(np.sum(self.window)) / 2.0, EPS)
        self.freq_resolution = self.config.sample_rate / self.config.fft_size
        self._channel_states: Dict[int, _ChannelFeedbackState] = {}
        self._enabled = True

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
            self._channel_states[channel] = _ChannelFeedbackState(channel, self.config)
        return self._channel_states[channel]

    def _spectrum_dbfs(self, samples: np.ndarray) -> np.ndarray:
        block = np.asarray(samples[-self.config.fft_size:], dtype=np.float64)
        block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
        block = block - float(np.mean(block))
        spectrum = np.abs(np.fft.rfft(block * self.window)) / self._window_amplitude_norm
        return 20.0 * np.log10(np.maximum(spectrum, 1e-12))

    def _match_peak(self, state: _ChannelFeedbackState, frequency: float, used: set[int]) -> Optional[int]:
        candidates = [
            (pid, abs(math.log(max(peak.frequency, EPS) / max(frequency, EPS))))
            for pid, peak in state.tracked_peaks.items()
            if pid not in used and _freq_close(peak.frequency, frequency, self.config.freq_tolerance_cents)
        ]
        return min(candidates, key=lambda item: item[1])[0] if candidates else None

    def process_audio(self, channel: int, samples: np.ndarray) -> Optional[FeedbackEvent]:
        if not self._enabled or len(samples) < self.config.fft_size:
            return None
        state = self._get_or_create_state(channel)
        state.frame_index += 1
        now = time.time()
        magnitude_db = self._spectrum_dbfs(samples)
        if state.avg_spectrum is None:
            state.avg_spectrum = magnitude_db.copy()
        else:
            state.avg_spectrum = 0.9 * state.avg_spectrum + 0.1 * magnitude_db

        peak_indices, properties = _find_spectral_peaks(
            magnitude_db,
            height=self.config.peak_height_db,
            distance=self.config.peak_distance_bins,
            prominence=self.config.peak_prominence_db,
        )
        prominences = properties.get("prominences", np.zeros(len(peak_indices)))
        candidates = []
        for idx, prominence in zip(peak_indices, prominences):
            freq = float(self.freqs[idx])
            if MIN_FEEDBACK_FREQ_HZ <= freq <= MAX_FEEDBACK_FREQ_HZ:
                candidates.append((int(idx), freq, float(magnitude_db[idx]), float(prominence)))

        used_tracks: set[int] = set()
        observed_tracks: set[int] = set()
        for bin_index, freq, mag, prominence in candidates:
            track_id = self._match_peak(state, freq, used_tracks)
            if track_id is None:
                track_id = bin_index
                while track_id in state.tracked_peaks:
                    track_id += self.config.fft_size + 1
                state.tracked_peaks[track_id] = _PeakState(
                    frequency=freq,
                    magnitude_db=mag,
                    persistence=1,
                    magnitude_history=[mag],
                    first_seen=now,
                    last_seen=now,
                )
            else:
                peak = state.tracked_peaks[track_id]
                peak.frequency = 0.8 * peak.frequency + 0.2 * freq
                peak.magnitude_db = mag
                peak.persistence += 1
                peak.magnitude_history.append(mag)
                peak.magnitude_history = peak.magnitude_history[-20:]
                peak.last_seen = now
            peak = state.tracked_peaks[track_id]
            # Prominence acts as tonality evidence. Store a conservative
            # confidence component without changing the legacy dataclass API.
            persistence_score = min(1.0, peak.persistence / max(1.0, 2.0 * self.config.persistence_frames))
            prominence_score = float(np.clip(prominence / max(12.0, 2.0 * self.config.peak_prominence_db), 0.0, 1.0))
            peak.confidence = 0.55 * persistence_score + 0.45 * prominence_score
            used_tracks.add(track_id)
            observed_tracks.add(track_id)

        for track_id in list(state.tracked_peaks):
            if track_id in observed_tracks:
                continue
            peak = state.tracked_peaks[track_id]
            peak.persistence -= 1
            if peak.persistence <= 0:
                del state.tracked_peaks[track_id]

        eligible: List[_PeakState] = []
        for peak in state.tracked_peaks.values():
            persistent = peak.persistence >= self.config.persistence_frames
            tonal_evolution = peak.is_growing() or peak.is_stable(self.config.stable_level_std_db)
            cooled_down = state.frame_index - peak.last_action_frame >= self.config.action_cooldown_frames
            if persistent and tonal_evolution and cooled_down and peak.confidence >= self.config.min_confidence:
                eligible.append(peak)
        if not eligible:
            return None

        # Prefer the most persistent and most confident oscillation, not simply
        # the loudest FFT bin.
        peak = max(eligible, key=lambda item: (item.confidence, item.persistence, item.magnitude_db))
        confidence = float(np.clip(peak.confidence, 0.0, 1.0))
        nf = state.add_notch(peak.frequency, confidence, now)
        peak.last_action_frame = state.frame_index
        if nf is not None:
            event = FeedbackEvent(
                channel=channel,
                frequency_hz=float(peak.frequency),
                magnitude_db=float(peak.magnitude_db),
                action="notch",
                confidence=confidence,
                timestamp=now,
                notch_filter=nf,
            )
        else:
            state.apply_fader_reduction(
                step_db=self.config.fader_reduction_step_db,
                floor_db=self.config.fader_reduction_floor_db,
            )
            event = FeedbackEvent(
                channel=channel,
                frequency_hz=float(peak.frequency),
                magnitude_db=float(peak.magnitude_db),
                action="fader_reduce",
                confidence=confidence,
                timestamp=now,
            )
        state.events.append(event)
        return event

    def process(self, channel: int, samples: np.ndarray) -> List[FeedbackEvent]:
        event = self.process_audio(channel, samples)
        return [event] if event else []

    def get_active_notches(self, channel: int) -> List[NotchFilter]:
        state = self._channel_states.get(channel)
        return [] if state is None else list(state.notch_filters)

    def get_fader_reduction(self, channel: int) -> float:
        state = self._channel_states.get(channel)
        return 0.0 if state is None else state.fader_reduction_db

    def get_channel_state(self, channel: int) -> Optional[_ChannelFeedbackState]:
        return self._channel_states.get(channel)

    def reset_channel(self, channel: int):
        self._channel_states.pop(channel, None)

    def reset_all(self):
        self._channel_states.clear()

    def get_diagnostics(self, channel: int) -> Dict[str, Any]:
        state = self._channel_states.get(channel)
        if state is None:
            return {"channel_id": channel, "tracked_peaks": 0, "notch_filters": 0, "fader_reduction_db": 0.0, "events": 0}
        return {
            "channel_id": channel,
            "tracked_peaks": len(state.tracked_peaks),
            "notch_filters": len(state.notch_filters),
            "fader_reduction_db": state.fader_reduction_db,
            "events": len(state.events),
            "peak_details": [
                {
                    "frequency": float(p.frequency),
                    "magnitude_db": float(p.magnitude_db),
                    "persistence": p.persistence,
                    "growing": p.is_growing(),
                    "stable": p.is_stable(self.config.stable_level_std_db),
                    "confidence": float(p.confidence),
                }
                for p in state.tracked_peaks.values()
            ],
        }

    def generate_osc_commands(self, channel: int) -> List[Tuple[str, Any]]:
        state = self._channel_states.get(channel)
        if state is None or not state.notch_filters:
            return []
        # WING main EQ exposes four parametric slots on this compatibility path.
        # Never alias more internal filters onto the same slots with modulo.
        active = sorted(state.notch_filters, key=lambda nf: nf.applied_at, reverse=True)[: self.config.max_console_notches]
        active.reverse()
        commands: List[Tuple[str, Any]] = [(f"/ch/{channel}/eq/on", 1)]
        for slot, nf in enumerate(active, start=1):
            commands.extend([
                (f"/ch/{channel}/eq/{slot}f", nf.frequency),
                (f"/ch/{channel}/eq/{slot}g", nf.gain_db),
                (f"/ch/{channel}/eq/{slot}q", nf.q_factor),
            ])
        return commands

    def generate_reset_osc_commands(self, channel: int) -> List[Tuple[str, Any]]:
        commands: List[Tuple[str, Any]] = []
        for band in range(1, 5):
            commands.extend([
                (f"/ch/{channel}/eq/{band}g", 0.0),
                (f"/ch/{channel}/eq/{band}f", 1000.0),
                (f"/ch/{channel}/eq/{band}q", 1.0),
            ])
        for band in range(1, 4):
            commands.extend([
                (f"/ch/{channel}/peq/{band}g", 0.0),
                (f"/ch/{channel}/peq/{band}f", 1000.0),
                (f"/ch/{channel}/peq/{band}q", 1.0),
            ])
        return commands
