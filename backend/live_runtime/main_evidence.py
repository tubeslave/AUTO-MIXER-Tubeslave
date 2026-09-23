"""Authoritative Main-bus evidence sources for the canonical live runtime.

The first concrete provider uses a *real post-console* Main tap routed back into
reserved WING USB capture channels.  It deliberately measures only that tap and
never estimates Main level by summing input stems.

The provider consumes the same coherent 48-channel snapshot used by the live
feature bridge, so Main timing is identical to the channel feature frame.  The
reserved tap channels are exposed explicitly so they can be excluded from
channel-level musical decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .feature_stream import MainFeatureEvidence, USB_CHANNEL_COUNT


_SILENCE_FLOOR_DBFS = -120.0


def _amp_to_db(value: float) -> float:
    if value <= 1e-6:
        return _SILENCE_FLOOR_DBFS
    return float(max(_SILENCE_FLOOR_DBFS, 20.0 * math.log10(value)))


def _capture_channel(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if not 1 <= value <= USB_CHANNEL_COUNT:
        raise ValueError(f"{label} must be inside 1..{USB_CHANNEL_COUNT}")
    return value


@dataclass(frozen=True)
class PostConsoleMainTap:
    """Capture slots carrying an explicitly routed post-console Main tap.

    ``right_channel`` may be omitted for a mono Main.  A stereo tap must occupy
    two distinct USB capture slots.
    """

    left_channel: int
    right_channel: int | None = None

    def __post_init__(self) -> None:
        left = _capture_channel(self.left_channel, label="left_channel")
        object.__setattr__(self, "left_channel", left)
        if self.right_channel is not None:
            right = _capture_channel(self.right_channel, label="right_channel")
            if right == left:
                raise ValueError("stereo Main tap channels must be distinct")
            object.__setattr__(self, "right_channel", right)

    @property
    def channels(self) -> tuple[int, ...]:
        if self.right_channel is None:
            return (self.left_channel,)
        return (self.left_channel, self.right_channel)


class PostConsoleMainTapEvidenceProvider:
    """Measure RMS/peak/crest from a real post-console Main USB return.

    The WING routing itself is external to this class and must be verified by
    patch/readback during soundcheck startup.  This class only turns the routed
    audio samples into evidence.  It therefore stays a measurement primitive,
    not a routing or decision policy.
    """

    def __init__(
        self,
        left_channel: int,
        right_channel: int | None = None,
    ) -> None:
        self.tap = PostConsoleMainTap(left_channel, right_channel)

    @property
    def reserved_capture_channels(self) -> tuple[int, ...]:
        """Capture slots that must not be treated as controllable input stems."""
        return self.tap.channels

    def from_block(
        self,
        block: np.ndarray,
        capture_timestamp_s: float,
    ) -> MainFeatureEvidence:
        data = np.asarray(block, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError("Main tap block must have shape [frames, channels]")
        if data.shape[0] <= 0:
            raise ValueError("Main tap block must contain at least one frame")
        if data.shape[1] != USB_CHANNEL_COUNT:
            raise ValueError(
                f"Main tap block must contain exactly {USB_CHANNEL_COUNT} channels; got {data.shape[1]}"
            )
        if not np.all(np.isfinite(data)):
            raise ValueError("Main tap block contains NaN or infinity")
        if not math.isfinite(float(capture_timestamp_s)):
            raise ValueError("Main tap timestamp must be finite")

        indices = [channel - 1 for channel in self.tap.channels]
        tapped = data[:, indices].astype(np.float64, copy=False)
        peak = float(np.max(np.abs(tapped)))
        rms = float(np.sqrt(np.mean(np.square(tapped), dtype=np.float64)))
        peak_db = _amp_to_db(peak)
        rms_db = _amp_to_db(rms)
        crest_db = float(max(0.0, peak_db - rms_db)) if rms_db > _SILENCE_FLOOR_DBFS else 0.0

        return MainFeatureEvidence(
            rms_dbfs=rms_db,
            peak_dbfs=peak_db,
            crest_db=crest_db,
            timestamp_s=float(capture_timestamp_s),
        )
