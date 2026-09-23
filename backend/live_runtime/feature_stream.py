"""Realtime USB feature ingress for the canonical live runtime.

The transport layer may deliver all 48 WING USB input channels, but raw input
stems do not contain authoritative Main-bus metering.  This module therefore
keeps channel evidence and Main evidence separate until an explicit Main meter
sample is supplied.  It never fabricates a Main by summing input stems.

The feature math intentionally extracts only evidence.  Musical decisions stay
in :mod:`live_runtime.decision_engine` and mutations stay behind the live
control plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .contracts import ChannelFeatures, MixFeatures


USB_SAMPLE_RATE = 48_000
USB_CHANNEL_COUNT = 48
# Migrated evidence primitive from legacy activity detection.  The old module
# used a -50 dB loudness threshold; the live runtime owns the evidence contract
# now and does not import the legacy decision tree.
ACTIVITY_THRESHOLD_DBFS = -50.0
_SILENCE_FLOOR_DBFS = -120.0


@dataclass(frozen=True)
class MainFeatureEvidence:
    """Authoritative Main-bus meter evidence from mixer/readback or a real tap."""

    rms_dbfs: float
    peak_dbfs: float
    crest_db: float
    timestamp_s: float = 0.0


@dataclass(frozen=True)
class UsbChannelFeatureFrame:
    """One channel-only realtime evidence frame from a 48-channel USB block."""

    channels: list[ChannelFeatures]
    timestamp_s: float
    sample_rate: int = USB_SAMPLE_RATE
    source_channels: int = USB_CHANNEL_COUNT


def _amp_to_db(value: float) -> float:
    if value <= 1e-6:
        return _SILENCE_FLOOR_DBFS
    return float(max(_SILENCE_FLOOR_DBFS, 20.0 * np.log10(value)))


def _spectral_evidence(samples: np.ndarray, sample_rate: int) -> tuple[float | None, float | None, float | None]:
    """Return centroid, low-mid ratio dB and normalized harshness evidence.

    This is a realtime extraction of the already-used repository spectral
    primitives: Hann-window FFT, energy bands and a 2-8 kHz harshness proxy.
    It deliberately returns no evidence for near-silence instead of producing
    meaningless FFT descriptors from a padded zero frame.
    """
    rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
    if rms <= 1e-6:
        return None, None, None

    n_fft = 1
    target = max(256, min(4096, int(samples.size)))
    while n_fft < target:
        n_fft <<= 1
    if samples.size < n_fft:
        data = np.pad(samples, (0, n_fft - samples.size))
    else:
        data = samples[-n_fft:]

    window = np.hanning(n_fft).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(data * window)).astype(np.float64)
    power = np.square(spectrum)
    total_power = float(np.sum(power))
    total_mag = float(np.sum(spectrum))
    if total_power <= 1e-18 or total_mag <= 1e-12:
        return None, None, None

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    centroid = float(np.sum(freqs * spectrum) / total_mag)

    def band_power(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.sum(power[mask])) if np.any(mask) else 0.0

    low_mid = band_power(250.0, 500.0)
    reference = band_power(500.0, 2_000.0)
    low_mid_ratio_db = 10.0 * np.log10((low_mid + 1e-18) / (reference + 1e-18))

    high_mid = band_power(2_000.0, 4_000.0) / total_power
    presence = band_power(4_000.0, 8_000.0) / total_power
    harshness = float(np.clip(high_mid + 0.6 * presence, 0.0, 1.0))
    return centroid, float(low_mid_ratio_db), harshness


class Usb48FeatureExtractor:
    """Extract evidence from one `[frames, 48]` float USB input block.

    The extractor is intentionally stateless.  It may run on a decimated
    analysis cadence instead of inside the audio callback.  Channel numbers are
    1-based to match the mixer/control contracts.
    """

    def __init__(
        self,
        *,
        sample_rate: int = USB_SAMPLE_RATE,
        channel_count: int = USB_CHANNEL_COUNT,
        activity_threshold_dbfs: float = ACTIVITY_THRESHOLD_DBFS,
    ) -> None:
        if sample_rate != USB_SAMPLE_RATE:
            raise ValueError("USB live MVP requires 48000 Hz capture")
        if channel_count != USB_CHANNEL_COUNT:
            raise ValueError("USB live MVP requires exactly 48 capture channels")
        self.sample_rate = sample_rate
        self.channel_count = channel_count
        self.activity_threshold_dbfs = float(activity_threshold_dbfs)

    def extract(
        self,
        block: np.ndarray,
        *,
        selected_channels: Sequence[int] | None = None,
        channel_names: Mapping[int, str] | None = None,
        timestamp_s: float = 0.0,
    ) -> UsbChannelFeatureFrame:
        data = np.asarray(block, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError("USB audio block must have shape [frames, channels]")
        if data.shape[0] <= 0:
            raise ValueError("USB audio block must contain at least one frame")
        if data.shape[1] != self.channel_count:
            raise ValueError(
                f"USB audio block must contain exactly {self.channel_count} channels; got {data.shape[1]}"
            )
        if not np.all(np.isfinite(data)):
            raise ValueError("USB audio block contains NaN or infinity")

        selected = list(selected_channels) if selected_channels is not None else list(range(1, self.channel_count + 1))
        if len(set(selected)) != len(selected):
            raise ValueError("selected_channels contains duplicates")
        for channel in selected:
            if channel < 1 or channel > self.channel_count:
                raise ValueError(f"USB channel {channel} is outside 1..{self.channel_count}")

        names = channel_names or {}
        features: list[ChannelFeatures] = []
        for channel in selected:
            samples = data[:, channel - 1]
            peak = float(np.max(np.abs(samples)))
            rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
            peak_db = _amp_to_db(peak)
            rms_db = _amp_to_db(rms)
            crest_db = float(max(0.0, peak_db - rms_db)) if rms_db > _SILENCE_FLOOR_DBFS else 0.0
            centroid, low_mid_ratio, harshness = _spectral_evidence(samples, self.sample_rate)

            features.append(
                ChannelFeatures(
                    channel=channel,
                    name=str(names.get(channel, f"CH {channel}")),
                    rms_dbfs=rms_db,
                    peak_dbfs=peak_db,
                    crest_db=crest_db,
                    spectral_centroid_hz=centroid,
                    low_mid_ratio_db=low_mid_ratio,
                    harshness=harshness,
                    activity=1.0 if rms_db >= self.activity_threshold_dbfs else 0.0,
                    confidence=1.0,
                )
            )

        return UsbChannelFeatureFrame(
            channels=features,
            timestamp_s=float(timestamp_s),
            sample_rate=self.sample_rate,
            source_channels=self.channel_count,
        )


def assemble_mix_features(
    frame: UsbChannelFeatureFrame,
    main: MainFeatureEvidence,
    *,
    max_main_age_s: float = 0.250,
) -> MixFeatures:
    """Join USB channel evidence with explicit, time-coherent Main evidence.

    A separate Main sample is mandatory because pre-console USB input stems do
    not describe post-console Main level/headroom.  Reject stale or non-finite
    evidence instead of synthesizing a Main from the captured channels.
    """
    values = (main.rms_dbfs, main.peak_dbfs, main.crest_db, main.timestamp_s)
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("Main evidence contains NaN or infinity")
    if max_main_age_s < 0:
        raise ValueError("max_main_age_s must be >= 0")
    if abs(float(frame.timestamp_s) - float(main.timestamp_s)) > max_main_age_s:
        raise ValueError("Main evidence is stale relative to the USB feature frame")

    return MixFeatures(
        channels=list(frame.channels),
        main_rms_dbfs=float(main.rms_dbfs),
        main_peak_dbfs=float(main.peak_dbfs),
        main_crest_db=float(main.crest_db),
        timestamp_s=float(frame.timestamp_s),
    )
