"""Standards-based mastering measurements and true-peak safety helpers.

This module is deliberately side-effect free so the STUDIO/offline pipeline can
measure a candidate before deciding whether it is eligible for human review.
It reuses the project's existing ITU-R BS.1770 K-weighting and 4x true-peak
meter instead of adding another DSP implementation or dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from lufs_gain_staging import KWeightingFilter, TruePeakMeter


ABSOLUTE_GATE_LUFS = -70.0
RELATIVE_GATE_LU = -10.0
BLOCK_SECONDS = 0.400
HOP_SECONDS = 0.100
SILENCE_FLOOR_DB = -100.0
TRUE_PEAK_TOLERANCE_DB = 0.02


@dataclass(frozen=True)
class MasteringSafetyMeasurement:
    """Objective level evidence for one offline master candidate."""

    integrated_lufs: float
    sample_peak_dbfs: float
    true_peak_dbtp: float
    rms_dbfs: float
    crest_factor_db: float


class StudioMasteringMeter:
    """Measure offline mastering candidates with project-standard meters."""

    def __init__(self, sample_rate: int = 48_000):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.sample_rate = int(sample_rate)

    @staticmethod
    def _as_samples_channels(audio: np.ndarray) -> np.ndarray:
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim == 1:
            return arr[:, None]
        if arr.ndim != 2:
            raise ValueError("audio must be mono or 2-D multichannel")
        if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
            arr = arr.T
        return arr.astype(np.float32, copy=False)

    def _k_weighted(self, audio: np.ndarray) -> np.ndarray:
        arr = self._as_samples_channels(audio)
        if len(arr) == 0:
            return arr
        channels = []
        for index in range(arr.shape[1]):
            meter_filter = KWeightingFilter(self.sample_rate)
            channels.append(meter_filter.process(arr[:, index]))
        return np.column_stack(channels).astype(np.float64, copy=False)

    def integrated_lufs(self, audio: np.ndarray) -> float:
        """Return gated integrated loudness using BS.1770-style 400 ms blocks."""
        weighted = self._k_weighted(audio)
        if len(weighted) == 0 or not np.any(np.abs(weighted) > 1e-12):
            return SILENCE_FLOOR_DB

        block_len = max(1, int(round(BLOCK_SECONDS * self.sample_rate)))
        hop_len = max(1, int(round(HOP_SECONDS * self.sample_rate)))

        if len(weighted) < block_len:
            padded = np.zeros((block_len, weighted.shape[1]), dtype=np.float64)
            padded[: len(weighted)] = weighted
            weighted = padded

        starts = list(range(0, len(weighted) - block_len + 1, hop_len))
        if not starts:
            starts = [0]
        final_start = len(weighted) - block_len
        if starts[-1] != final_start:
            starts.append(final_start)

        block_powers = []
        block_loudness = []
        for start in starts:
            block = weighted[start : start + block_len]
            # L/R/C channels have unit weighting. Offline STUDIO currently
            # targets mono/stereo, so do not guess surround channel roles.
            power = float(np.sum(np.mean(np.square(block), axis=0)))
            loudness = -0.691 + 10.0 * np.log10(max(power, 1e-20))
            block_powers.append(power)
            block_loudness.append(loudness)

        powers = np.asarray(block_powers, dtype=np.float64)
        loudness = np.asarray(block_loudness, dtype=np.float64)
        absolute_mask = loudness >= ABSOLUTE_GATE_LUFS
        if not np.any(absolute_mask):
            return SILENCE_FLOOR_DB

        ungated_power = float(np.mean(powers[absolute_mask]))
        ungated_lufs = -0.691 + 10.0 * np.log10(max(ungated_power, 1e-20))
        relative_threshold = ungated_lufs + RELATIVE_GATE_LU
        gated_mask = absolute_mask & (loudness >= relative_threshold)
        if not np.any(gated_mask):
            return float(ungated_lufs)

        gated_power = float(np.mean(powers[gated_mask]))
        return float(-0.691 + 10.0 * np.log10(max(gated_power, 1e-20)))

    def true_peak_dbtp(self, audio: np.ndarray) -> float:
        """Return maximum 4x-oversampled true peak across channels."""
        arr = self._as_samples_channels(audio)
        if len(arr) == 0 or not np.any(np.abs(arr) > 0.0):
            return SILENCE_FLOOR_DB

        peaks = []
        # Flush the short FIR tail so peaks near the end are not hidden by
        # interpolation filter delay.
        tail = np.zeros(16, dtype=np.float32)
        for index in range(arr.shape[1]):
            meter = TruePeakMeter(self.sample_rate)
            channel = np.concatenate((arr[:, index], tail))
            peaks.append(meter.process(channel))
        return float(max(peaks))

    def measure(self, audio: np.ndarray) -> MasteringSafetyMeasurement:
        arr = self._as_samples_channels(audio)
        if len(arr) == 0:
            return MasteringSafetyMeasurement(
                integrated_lufs=SILENCE_FLOOR_DB,
                sample_peak_dbfs=SILENCE_FLOOR_DB,
                true_peak_dbtp=SILENCE_FLOOR_DB,
                rms_dbfs=SILENCE_FLOOR_DB,
                crest_factor_db=0.0,
            )

        sample_peak = float(np.max(np.abs(arr)))
        sample_peak_dbfs = (
            20.0 * np.log10(sample_peak) if sample_peak > 0.0 else SILENCE_FLOOR_DB
        )
        rms = float(np.sqrt(np.mean(np.square(arr.astype(np.float64)))))
        rms_dbfs = 20.0 * np.log10(rms) if rms > 0.0 else SILENCE_FLOOR_DB
        true_peak = self.true_peak_dbtp(arr)
        return MasteringSafetyMeasurement(
            integrated_lufs=self.integrated_lufs(arr),
            sample_peak_dbfs=float(sample_peak_dbfs),
            true_peak_dbtp=true_peak,
            rms_dbfs=float(rms_dbfs),
            crest_factor_db=float(true_peak - rms_dbfs) if rms > 0.0 else 0.0,
        )

    def limit_true_peak(
        self,
        audio: np.ndarray,
        ceiling_dbtp: float = -1.0,
    ) -> Tuple[np.ndarray, float]:
        """Apply the minimum static attenuation required by the true-peak ceiling.

        This is a safety guard, not a creative limiter. It never adds gain and
        therefore cannot be used to chase a loudness target.
        """
        arr = self._as_samples_channels(audio)
        original_was_mono = np.asarray(audio).ndim == 1
        if len(arr) == 0:
            result = arr[:, 0] if original_was_mono else arr
            return result.astype(np.float32, copy=False), 0.0

        current_peak = self.true_peak_dbtp(arr)
        if current_peak <= ceiling_dbtp + TRUE_PEAK_TOLERANCE_DB:
            result = arr[:, 0] if original_was_mono else arr
            return result.astype(np.float32, copy=True), 0.0

        reduction_db = float(current_peak - ceiling_dbtp)
        gain = np.float32(10.0 ** (-reduction_db / 20.0))
        limited = (arr * gain).astype(np.float32)

        measured_peak = self.true_peak_dbtp(limited)
        if measured_peak > ceiling_dbtp + TRUE_PEAK_TOLERANCE_DB:
            correction_db = float(measured_peak - ceiling_dbtp)
            limited *= np.float32(10.0 ** (-correction_db / 20.0))
            reduction_db += correction_db

        result = limited[:, 0] if original_was_mono else limited
        return result.astype(np.float32, copy=False), reduction_db
