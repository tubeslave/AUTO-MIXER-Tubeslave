"""
Real-Time Acoustic Feedback Detector for Behringer Wing Rack.

Detects feedback via FFT peak tracking and persistent-peak classification.
When feedback is confirmed, applies automatic notch EQ filters or fader
reduction as a fallback.  Designed for <50 ms latency at 48 kHz.

Dependencies: numpy (required), scipy (optional — falls back to a simple
peak-finder when scipy.signal is unavailable).
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.signal import find_peaks as _scipy_find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Wing OSC EQ band mapping
# The Wing has: low shelf (l), bands 1-4 (parametric), high shelf (h).
# We repurpose bands 1-4 as notch filters and, if more are needed, use the
# 3-band pre-EQ (peq) for up to 8 total notch slots per channel.
# ---------------------------------------------------------------------------

# Maximum number of notch filters we will deploy per channel
MAX_NOTCH_FILTERS = 8

# Maximum cut per notch (dB, negative value applied)
MAX_NOTCH_DEPTH_DB = -12.0

# Wing EQ band slots available for notch use (main EQ bands 1-4, then PEQ 1-3,
# then main EQ low shelf as last resort).
_MAIN_EQ_BANDS = [
    {"gain": "/ch/{ch}/eq/{band}g", "freq": "/ch/{ch}/eq/{band}f", "q": "/ch/{ch}/eq/{band}q", "band": b}
    for b in range(1, 5)
]

_PEQ_BANDS = [
    {"gain": "/ch/{ch}/peq/{band}g", "freq": "/ch/{ch}/peq/{band}f", "q": "/ch/{ch}/peq/{band}q", "band": b}
    for b in range(1, 4)
]

# 8th slot: main EQ low shelf repurposed as narrow PEQ
_LOW_SHELF_SLOT = {
    "gain": "/ch/{ch}/eq/lg",
    "freq": "/ch/{ch}/eq/lf",
    "q": "/ch/{ch}/eq/lq",
    "type_addr": "/ch/{ch}/eq/leq",  # set to PEQ mode
    "band": "l",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NotchFilter:
    """Represents a single notch EQ filter applied to suppress feedback."""
    frequency: float       # Centre frequency in Hz
    gain_db: float         # Negative gain (cut) in dB, clamped to MAX_NOTCH_DEPTH_DB
    q_factor: float        # Q / bandwidth control (higher = narrower)
    slot_index: int = 0    # Which EQ band slot this occupies (0-7)
    applied_at: float = 0.0  # time.monotonic() when first applied


@dataclass
class FeedbackEvent:
    """Returned by process_audio() when feedback is detected or acted upon."""
    channel_id: int
    frequency_hz: float
    gain_db: float         # The notch depth applied (negative dB)
    confidence: float      # 0.0 – 1.0 confidence that this is feedback
    action: str            # 'notch' or 'fader_reduce'
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class _PeakState:
    """Internal tracking of a spectral peak across frames."""
    frequency: float
    magnitude_db: float
    frame_count: int = 1           # Number of consecutive frames the peak persists
    first_seen: float = 0.0        # time.monotonic()
    last_seen: float = 0.0
    magnitude_history: List[float] = field(default_factory=list)

    @property
    def age_sec(self) -> float:
        return self.last_seen - self.first_seen

    def is_growing(self, window: int = 4) -> bool:
        """True if the magnitude trend is upward over the last *window* frames."""
        hist = self.magnitude_history[-window:]
        if len(hist) < 2:
            return False
        diffs = [hist[i + 1] - hist[i] for i in range(len(hist) - 1)]
        return sum(1 for d in diffs if d > 0) > len(diffs) / 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_peaks_fallback(magnitude_db: np.ndarray, height: float,
                         distance: int, prominence: float) -> Tuple[np.ndarray, Dict]:
    """Simple peak finder used when scipy is not available."""
    peaks = []
    prominences = []
    n = len(magnitude_db)
    for i in range(1, n - 1):
        if magnitude_db[i] < height:
            continue
        left_ok = all(magnitude_db[i] >= magnitude_db[max(0, i - d)] for d in range(1, distance + 1) if i - d >= 0)
        right_ok = all(magnitude_db[i] >= magnitude_db[min(n - 1, i + d)] for d in range(1, distance + 1) if i + d < n)
        if left_ok and right_ok:
            # Estimate prominence as difference from the average of the two nearest troughs
            left_min = np.min(magnitude_db[max(0, i - distance):i]) if i > 0 else magnitude_db[i]
            right_min = np.min(magnitude_db[i + 1:min(n, i + distance + 1)]) if i < n - 1 else magnitude_db[i]
            prom = magnitude_db[i] - max(left_min, right_min)
            if prom >= prominence:
                peaks.append(i)
                prominences.append(prom)
    return np.array(peaks, dtype=int), {"prominences": np.array(prominences)}


def _find_spectral_peaks(magnitude_db: np.ndarray, height: float = -30.0,
                         distance: int = 5, prominence: float = 6.0
                         ) -> Tuple[np.ndarray, Dict]:
    """Return indices of spectral peaks that exceed thresholds."""
    if HAS_SCIPY:
        return _scipy_find_peaks(magnitude_db, height=height,
                                 distance=distance, prominence=prominence)
    return _find_peaks_fallback(magnitude_db, height, distance, prominence)


def _freq_close(f1: float, f2: float, tolerance_cents: float = 100.0) -> bool:
    """Return True if two frequencies are within *tolerance_cents* of each other."""
    if f1 <= 0 or f2 <= 0:
        return False
    ratio = f1 / f2
    cents = abs(1200.0 * np.log2(ratio))
    return cents <= tolerance_cents


def _q_for_feedback(confidence: float) -> float:
    """Return a Q factor for the notch: narrow (high Q) for high-confidence feedback."""
    # Range: Q 8 (tight) at confidence 1.0  →  Q 4 (wider) at confidence 0.5
    return float(np.clip(4.0 + 4.0 * confidence, 4.0, 10.0))


# ---------------------------------------------------------------------------
# Channel state container
# ---------------------------------------------------------------------------

class _ChannelFeedbackState:
    """Per-channel state for peak tracking, notch filters, and fader reduction."""

    def __init__(self, channel_id: int, config: "FeedbackDetectorConfig"):
        self.channel_id = channel_id
        self.config = config
        self.tracked_peaks: List[_PeakState] = []
        self.notch_filters: List[NotchFilter] = []
        self.fader_reduction_db: float = 0.0
        self._prev_spectrum: Optional[np.ndarray] = None

    # -- Peak tracking -----------------------------------------------------

    def update_peaks(self, frequencies: np.ndarray, magnitude_db: np.ndarray,
                     now: float) -> List[_PeakState]:
        """Find spectral peaks, match against existing tracked peaks, return newly confirmed feedback peaks."""
        peak_indices, properties = _find_spectral_peaks(
            magnitude_db,
            height=self.config.peak_height_db,
            distance=self.config.peak_distance_bins,
            prominence=self.config.peak_prominence_db,
        )
        if len(peak_indices) == 0:
            # Decay existing peaks that were not seen this frame
            self._decay_peaks(now)
            return []

        peak_freqs = frequencies[peak_indices]
        peak_mags = magnitude_db[peak_indices]

        matched_indices: set = set()
        newly_confirmed: List[_PeakState] = []

        # Match detected peaks to existing tracked peaks
        for pf, pm in zip(peak_freqs, peak_mags):
            matched = False
            for tp in self.tracked_peaks:
                if _freq_close(pf, tp.frequency, self.config.match_tolerance_cents):
                    tp.frequency = float(pf)
                    tp.magnitude_db = float(pm)
                    tp.frame_count += 1
                    tp.last_seen = now
                    tp.magnitude_history.append(float(pm))
                    # Trim history
                    if len(tp.magnitude_history) > 64:
                        tp.magnitude_history = tp.magnitude_history[-64:]
                    matched = True
                    matched_indices.add(id(tp))
                    # Check confirmation threshold
                    if (tp.frame_count >= self.config.persistence_frames
                            and tp.is_growing()):
                        newly_confirmed.append(tp)
                    break
            if not matched:
                new_peak = _PeakState(
                    frequency=float(pf),
                    magnitude_db=float(pm),
                    frame_count=1,
                    first_seen=now,
                    last_seen=now,
                    magnitude_history=[float(pm)],
                )
                self.tracked_peaks.append(new_peak)

        self._decay_peaks(now)
        return newly_confirmed

    def _decay_peaks(self, now: float) -> None:
        """Remove peaks that have not been seen for too long."""
        timeout = self.config.peak_timeout_sec
        self.tracked_peaks = [
            p for p in self.tracked_peaks
            if (now - p.last_seen) < timeout
        ]

    # -- Notch management --------------------------------------------------

    def add_notch(self, frequency: float, confidence: float, now: float) -> Optional[NotchFilter]:
        """Add or deepen a notch filter at *frequency*.  Returns the filter if created/updated."""
        # Check if a notch already exists near this frequency
        for nf in self.notch_filters:
            if _freq_close(nf.frequency, frequency, tolerance_cents=80.0):
                # Deepen existing notch (up to limit)
                deeper = max(nf.gain_db - 3.0, MAX_NOTCH_DEPTH_DB)
                if deeper < nf.gain_db:
                    nf.gain_db = deeper
                    logger.info(
                        "Ch %d: deepened notch @ %.0f Hz to %.1f dB",
                        self.channel_id, nf.frequency, nf.gain_db,
                    )
                return nf

        if len(self.notch_filters) >= MAX_NOTCH_FILTERS:
            logger.warning(
                "Ch %d: max notch filters (%d) reached, cannot add @ %.0f Hz",
                self.channel_id, MAX_NOTCH_FILTERS, frequency,
            )
            return None

        slot = len(self.notch_filters)
        initial_gain = max(-6.0 * confidence, MAX_NOTCH_DEPTH_DB)
        q = _q_for_feedback(confidence)
        nf = NotchFilter(
            frequency=frequency,
            gain_db=initial_gain,
            q_factor=q,
            slot_index=slot,
            applied_at=now,
        )
        self.notch_filters.append(nf)
        logger.info(
            "Ch %d: new notch #%d @ %.0f Hz, gain %.1f dB, Q %.1f",
            self.channel_id, slot, frequency, initial_gain, q,
        )
        return nf

    def should_reduce_fader(self) -> bool:
        """Return True if we've exhausted notch filters and feedback persists."""
        if len(self.notch_filters) < MAX_NOTCH_FILTERS:
            return False
        # Check if any tracked peak is still growing despite all notches
        return any(
            p.frame_count >= self.config.persistence_frames and p.is_growing()
            for p in self.tracked_peaks
        )

    def apply_fader_reduction(self, step_db: float = -3.0, floor_db: float = -18.0) -> float:
        """Reduce the fader level by *step_db* (negative).  Returns total reduction."""
        self.fader_reduction_db = max(self.fader_reduction_db + step_db, floor_db)
        logger.warning(
            "Ch %d: fader reduced to %.1f dB",
            self.channel_id, self.fader_reduction_db,
        )
        return self.fader_reduction_db

    def release_notch(self, index: int) -> bool:
        """Release (remove) a notch filter by slot index.  Returns True if removed."""
        for i, nf in enumerate(self.notch_filters):
            if nf.slot_index == index:
                self.notch_filters.pop(i)
                # Re-index remaining slots
                for j, remaining in enumerate(self.notch_filters):
                    remaining.slot_index = j
                logger.info("Ch %d: released notch slot %d", self.channel_id, index)
                return True
        return False

    def get_stale_notches(self, now: float, max_age_sec: float = 30.0) -> List[NotchFilter]:
        """Return notch filters that have been active longer than *max_age_sec* and whose
        source frequency is no longer among tracked peaks."""
        stale = []
        active_freqs = {p.frequency for p in self.tracked_peaks if p.frame_count >= 2}
        for nf in self.notch_filters:
            age = now - nf.applied_at
            if age < max_age_sec:
                continue
            still_active = any(
                _freq_close(nf.frequency, af, tolerance_cents=80.0)
                for af in active_freqs
            )
            if not still_active:
                stale.append(nf)
        return stale


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FeedbackDetectorConfig:
    """All tuneable parameters for the feedback detector."""
    sample_rate: int = 48000
    fft_size: int = 2048
    hop_size: int = 1024              # Overlap: fft_size - hop_size samples
    # Peak detection
    peak_height_db: float = -20.0     # Minimum magnitude for a peak to be considered
    peak_distance_bins: int = 4       # Minimum distance between peaks in FFT bins
    peak_prominence_db: float = 8.0   # Minimum prominence above neighbours
    # Persistence
    persistence_frames: int = 6       # Frames a peak must persist before it's feedback
    match_tolerance_cents: float = 100.0  # Cents tolerance for matching peaks across frames
    peak_timeout_sec: float = 0.5     # Remove unmatched peaks after this duration
    # Notch
    max_notch_filters: int = MAX_NOTCH_FILTERS
    max_notch_depth_db: float = MAX_NOTCH_DEPTH_DB
    # Fader fallback
    fader_reduce_step_db: float = -3.0
    fader_reduce_floor_db: float = -18.0
    # Auto-release
    notch_release_age_sec: float = 30.0  # Release stale notches after this duration
    # Confidence
    min_confidence: float = 0.5       # Minimum confidence to act


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class FeedbackDetector:
    """
    Real-time acoustic feedback detector and suppressor.

    Usage::

        detector = FeedbackDetector()
        event = detector.process_audio(channel_id=1, samples=audio_block)
        if event:
            osc_cmds = detector.generate_osc_commands(channel_id=1)
            for addr, value in osc_cmds:
                wing_client.send(addr, value)
    """

    def __init__(self, config: Optional[FeedbackDetectorConfig] = None):
        self.config = config or FeedbackDetectorConfig()
        self._channels: Dict[int, _ChannelFeedbackState] = {}
        # Pre-compute FFT window and frequency bins
        self._window = np.hanning(self.config.fft_size).astype(np.float32)
        self._freqs = np.fft.rfftfreq(self.config.fft_size, 1.0 / self.config.sample_rate)
        # Per-channel input buffers for overlapping FFT frames
        self._buffers: Dict[int, np.ndarray] = {}
        logger.info(
            "FeedbackDetector initialised: sr=%d, fft=%d, hop=%d, persist=%d frames",
            self.config.sample_rate, self.config.fft_size,
            self.config.hop_size, self.config.persistence_frames,
        )

    # -- Public API --------------------------------------------------------

    def process_audio(self, channel_id: int, samples: np.ndarray) -> Optional[FeedbackEvent]:
        """
        Process a block of audio samples for the given channel.

        Args:
            channel_id: Mixer channel number (1-40).
            samples: 1-D numpy array of float32 audio samples (mono).

        Returns:
            A ``FeedbackEvent`` if feedback was detected and acted upon,
            otherwise ``None``.
        """
        now = time.monotonic()
        samples = np.asarray(samples, dtype=np.float32)
        state = self._get_or_create_state(channel_id)

        # Append to rolling buffer
        buf = self._buffers.get(channel_id, np.array([], dtype=np.float32))
        buf = np.concatenate([buf, samples])

        # Process as many full FFT frames as possible
        event: Optional[FeedbackEvent] = None
        fft_size = self.config.fft_size
        hop = self.config.hop_size

        while len(buf) >= fft_size:
            frame = buf[:fft_size]
            buf = buf[hop:]  # advance by hop_size (overlap)
            result = self._process_frame(state, frame, now)
            if result is not None:
                event = result  # Keep the latest event from this batch

        self._buffers[channel_id] = buf

        # Periodic stale-notch release
        self._auto_release_stale(state, now)

        return event

    def get_channel_state(self, channel_id: int) -> Optional[_ChannelFeedbackState]:
        """Return the internal state for a channel, or None."""
        return self._channels.get(channel_id)

    def get_active_notches(self, channel_id: int) -> List[NotchFilter]:
        """Return the list of currently active notch filters for a channel."""
        state = self._channels.get(channel_id)
        if state is None:
            return []
        return list(state.notch_filters)

    def get_fader_reduction(self, channel_id: int) -> float:
        """Return the cumulative fader reduction (dB) for a channel.  0.0 = no reduction."""
        state = self._channels.get(channel_id)
        if state is None:
            return 0.0
        return state.fader_reduction_db

    def reset_channel(self, channel_id: int) -> None:
        """Clear all feedback state and notch filters for a channel."""
        self._channels.pop(channel_id, None)
        self._buffers.pop(channel_id, None)
        logger.info("FeedbackDetector: channel %d state reset", channel_id)

    def reset_all(self) -> None:
        """Clear state for all channels."""
        self._channels.clear()
        self._buffers.clear()
        logger.info("FeedbackDetector: all state reset")

    def generate_osc_commands(self, channel_id: int) -> List[Tuple[str, float]]:
        """
        Generate Behringer Wing OSC commands to apply the current notch filters
        and fader reduction for *channel_id*.

        Returns:
            List of (osc_address, value) tuples ready to send.
        """
        state = self._channels.get(channel_id)
        if state is None:
            return []

        commands: List[Tuple[str, float]] = []
        ch = channel_id

        # Ensure main EQ is on
        commands.append((f"/ch/{ch}/eq/on", 1.0))

        for nf in state.notch_filters:
            slot = nf.slot_index
            if slot < 4:
                # Main EQ bands 1-4
                band_num = slot + 1
                commands.append((f"/ch/{ch}/eq/{band_num}f", nf.frequency))
                commands.append((f"/ch/{ch}/eq/{band_num}g", nf.gain_db))
                commands.append((f"/ch/{ch}/eq/{band_num}q", nf.q_factor))
            elif slot < 7:
                # Pre-EQ bands 1-3
                peq_num = slot - 3  # slots 4,5,6 → peq bands 1,2,3
                commands.append((f"/ch/{ch}/peq/on", 1.0))
                commands.append((f"/ch/{ch}/peq/{peq_num}f", nf.frequency))
                commands.append((f"/ch/{ch}/peq/{peq_num}g", nf.gain_db))
                commands.append((f"/ch/{ch}/peq/{peq_num}q", nf.q_factor))
            elif slot == 7:
                # 8th slot: low shelf repurposed to PEQ mode
                commands.append((f"/ch/{ch}/eq/leq", "PEQ"))
                commands.append((f"/ch/{ch}/eq/lf", nf.frequency))
                commands.append((f"/ch/{ch}/eq/lg", nf.gain_db))
                commands.append((f"/ch/{ch}/eq/lq", nf.q_factor))

        # Fader reduction (if any)
        if state.fader_reduction_db < 0:
            commands.append((f"/ch/{ch}/fdr", state.fader_reduction_db))

        return commands

    def generate_reset_osc_commands(self, channel_id: int) -> List[Tuple[str, float]]:
        """
        Generate OSC commands to reset (zero out) all notch filters previously
        applied to *channel_id*.  Call this after ``reset_channel()``.
        """
        commands: List[Tuple[str, float]] = []
        ch = channel_id

        # Reset main EQ bands 1-4 gain to 0
        for band in range(1, 5):
            commands.append((f"/ch/{ch}/eq/{band}g", 0.0))

        # Reset pre-EQ bands 1-3 gain to 0
        for band in range(1, 4):
            commands.append((f"/ch/{ch}/peq/{band}g", 0.0))

        # Reset low shelf to shelf mode and 0 gain
        commands.append((f"/ch/{ch}/eq/leq", "SHV"))
        commands.append((f"/ch/{ch}/eq/lg", 0.0))

        return commands

    # -- Internal ----------------------------------------------------------

    def _get_or_create_state(self, channel_id: int) -> _ChannelFeedbackState:
        if channel_id not in self._channels:
            self._channels[channel_id] = _ChannelFeedbackState(channel_id, self.config)
        return self._channels[channel_id]

    def _process_frame(self, state: _ChannelFeedbackState, frame: np.ndarray,
                       now: float) -> Optional[FeedbackEvent]:
        """Analyse one FFT frame, update peak tracking, act if needed."""
        # Windowed FFT
        windowed = frame * self._window
        spectrum = np.fft.rfft(windowed)
        magnitude = np.abs(spectrum)
        # Convert to dB, avoiding log of zero
        magnitude_db = 20.0 * np.log10(magnitude + 1e-10)

        # Update peak tracking
        confirmed = state.update_peaks(self._freqs, magnitude_db, now)

        if not confirmed:
            return None

        # Pick the strongest confirmed feedback peak
        strongest = max(confirmed, key=lambda p: p.magnitude_db)
        confidence = self._compute_confidence(strongest, magnitude_db)

        if confidence < self.config.min_confidence:
            return None

        # Decide action: notch or fader reduce
        if state.should_reduce_fader():
            reduction = state.apply_fader_reduction(
                self.config.fader_reduce_step_db,
                self.config.fader_reduce_floor_db,
            )
            return FeedbackEvent(
                channel_id=state.channel_id,
                frequency_hz=strongest.frequency,
                gain_db=reduction,
                confidence=confidence,
                action="fader_reduce",
            )

        nf = state.add_notch(strongest.frequency, confidence, now)
        if nf is None:
            # All slots full but should_reduce_fader was False — edge case
            return None

        return FeedbackEvent(
            channel_id=state.channel_id,
            frequency_hz=nf.frequency,
            gain_db=nf.gain_db,
            confidence=confidence,
            action="notch",
        )

    def _compute_confidence(self, peak: _PeakState, magnitude_db: np.ndarray) -> float:
        """
        Compute a 0–1 confidence that the peak is genuine feedback.

        Factors:
        1. Persistence (more frames → higher confidence).
        2. Narrowness: feedback peaks are typically very narrow.  We measure
           how far the peak protrudes above the local spectral floor.
        3. Growth: a rising magnitude trend is a strong indicator.
        """
        # 1. Persistence score (saturates at 2x the threshold)
        max_persist = self.config.persistence_frames * 2
        persist_score = min(peak.frame_count / max_persist, 1.0)

        # 2. Prominence score
        # Find the bin closest to the peak frequency
        bin_idx = int(round(peak.frequency * self.config.fft_size / self.config.sample_rate))
        bin_idx = np.clip(bin_idx, 0, len(magnitude_db) - 1)
        # Local floor: average of bins ±20 around the peak, excluding ±3
        lo = max(0, bin_idx - 20)
        hi = min(len(magnitude_db), bin_idx + 21)
        exclude_lo = max(0, bin_idx - 3)
        exclude_hi = min(len(magnitude_db), bin_idx + 4)
        neighbourhood = np.concatenate([
            magnitude_db[lo:exclude_lo],
            magnitude_db[exclude_hi:hi],
        ])
        if len(neighbourhood) > 0:
            local_floor = float(np.median(neighbourhood))
            prominence = peak.magnitude_db - local_floor
        else:
            prominence = 0.0
        # Map prominence 6–20 dB to 0–1
        prominence_score = float(np.clip((prominence - 6.0) / 14.0, 0.0, 1.0))

        # 3. Growth score
        growth_score = 1.0 if peak.is_growing() else 0.3

        # Weighted combination
        confidence = 0.35 * persist_score + 0.35 * prominence_score + 0.30 * growth_score
        return float(np.clip(confidence, 0.0, 1.0))

    def _auto_release_stale(self, state: _ChannelFeedbackState, now: float) -> None:
        """Release notch filters whose source peaks have disappeared."""
        stale = state.get_stale_notches(now, self.config.notch_release_age_sec)
        for nf in stale:
            state.release_notch(nf.slot_index)
            logger.info(
                "Ch %d: auto-released stale notch @ %.0f Hz (slot %d)",
                state.channel_id, nf.frequency, nf.slot_index,
            )

    # -- Diagnostics -------------------------------------------------------

    def get_diagnostics(self, channel_id: int) -> Dict:
        """Return a diagnostic dict for the given channel (useful for UI)."""
        state = self._channels.get(channel_id)
        if state is None:
            return {"channel_id": channel_id, "tracked_peaks": 0, "notches": 0,
                    "fader_reduction_db": 0.0}
        return {
            "channel_id": channel_id,
            "tracked_peaks": len(state.tracked_peaks),
            "peaks": [
                {
                    "freq_hz": round(p.frequency, 1),
                    "mag_db": round(p.magnitude_db, 1),
                    "frames": p.frame_count,
                    "growing": p.is_growing(),
                }
                for p in state.tracked_peaks
            ],
            "notches": [
                {
                    "slot": nf.slot_index,
                    "freq_hz": round(nf.frequency, 1),
                    "gain_db": round(nf.gain_db, 1),
                    "q": round(nf.q_factor, 2),
                }
                for nf in state.notch_filters
            ],
            "fader_reduction_db": round(state.fader_reduction_db, 1),
        }
