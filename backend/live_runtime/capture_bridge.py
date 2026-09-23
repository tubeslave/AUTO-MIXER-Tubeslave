"""Bridge the validated legacy AudioCapture transport into ``live_runtime``.

Only transport is reused here.  The legacy capture service owns device access and
per-channel ring buffers; this module snapshots those buffers at the callback
boundary, moves all FFT/Director work off the realtime audio callback, joins an
explicit Main-bus meter sample, and feeds the canonical LiveSoundcheckService.

Main evidence may come from an external authoritative meter provider or from a
real post-console Main tap contained in the same coherent USB snapshot.  Main
tap slots are excluded from channel-level musical analysis.  No Main level is
synthesized from input stems and no legacy ``auto_*`` decision policy is imported.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from .feature_stream import (
    MainFeatureEvidence,
    USB_CHANNEL_COUNT,
    USB_SAMPLE_RATE,
    Usb48FeatureExtractor,
    assemble_mix_features,
)


class AudioCaptureReader(Protocol):
    """Narrow KEEP_CORE surface required from :mod:`backend.audio_capture`."""

    num_channels: int
    sample_rate: int

    def subscribe(self, name: str, callback: Callable[[], None]) -> None: ...

    def unsubscribe(self, name: str) -> None: ...

    def get_buffer(self, channel: int, num_samples: int = 0) -> np.ndarray: ...


class MainEvidenceProvider(Protocol):
    """Return externally measured Main-bus evidence aligned to a capture timestamp."""

    def __call__(self, capture_timestamp_s: float) -> MainFeatureEvidence | None: ...


class SnapshotMainEvidenceProvider(Protocol):
    """Measure Main from reserved channels in the same coherent USB snapshot."""

    @property
    def reserved_capture_channels(self) -> tuple[int, ...]: ...

    def from_block(
        self,
        block: np.ndarray,
        capture_timestamp_s: float,
    ) -> MainFeatureEvidence | None: ...


class FeatureSnapshotConsumer(Protocol):
    """Minimal LiveSoundcheckService surface used by the pump."""

    def process_feature_snapshot(
        self,
        features: Any,
        roles: dict[int, str],
        **kwargs: Any,
    ) -> Any: ...


@dataclass(frozen=True)
class CaptureBridgeStatus:
    running: bool
    snapshots_captured: int
    snapshots_processed: int
    snapshots_replaced: int
    incomplete_windows: int
    missing_main_evidence: int
    processing_failures: int
    last_error: str | None


@dataclass(frozen=True)
class _CaptureSnapshot:
    audio: np.ndarray
    timestamp_s: float


class LiveAudioCaptureBridge:
    """Decimated 48-channel capture pump for the canonical live runtime.

    ``AudioCapture`` invokes named subscribers only after it has written every
    channel for the current input callback.  The subscriber therefore performs
    one bounded raw-memory snapshot while still at that boundary.  It does *not*
    run FFT analysis, Directors, Critics or WING writes there.  A dedicated
    worker consumes at most the newest pending snapshot, so analysis backlog can
    never grow without bound.

    The bridge owns neither the audio device nor mixer transport lifecycle.  It
    only subscribes/unsubscribes from an already configured capture service.
    Exactly one authoritative Main evidence source must be configured.
    """

    def __init__(
        self,
        capture: AudioCaptureReader,
        service: FeatureSnapshotConsumer,
        *,
        roles: Mapping[int, str],
        main_evidence_provider: MainEvidenceProvider | None = None,
        snapshot_main_evidence_provider: SnapshotMainEvidenceProvider | None = None,
        channel_names: Mapping[int, str] | None = None,
        window_frames: int = 2048,
        analysis_interval_s: float = 0.100,
        max_main_age_s: float = 0.250,
        subscriber_name: str = "live_runtime_feature_bridge",
        wall_clock: Callable[[], float] = time.time,
        interval_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if int(capture.sample_rate) != USB_SAMPLE_RATE:
            raise ValueError("live capture bridge requires 48000 Hz AudioCapture")
        if int(capture.num_channels) != USB_CHANNEL_COUNT:
            raise ValueError("live capture bridge requires exactly 48 AudioCapture channels")
        if window_frames <= 0:
            raise ValueError("window_frames must be > 0")
        if analysis_interval_s < 0:
            raise ValueError("analysis_interval_s must be >= 0")
        if max_main_age_s < 0:
            raise ValueError("max_main_age_s must be >= 0")
        if not subscriber_name:
            raise ValueError("subscriber_name must not be empty")
        if (main_evidence_provider is None) == (snapshot_main_evidence_provider is None):
            raise ValueError("configure exactly one authoritative Main evidence provider")

        reserved_channels: tuple[int, ...] = ()
        if snapshot_main_evidence_provider is not None:
            reserved_channels = tuple(snapshot_main_evidence_provider.reserved_capture_channels)
            if len(set(reserved_channels)) != len(reserved_channels):
                raise ValueError("Main evidence provider reserves duplicate capture channels")
            for channel in reserved_channels:
                if isinstance(channel, bool) or not isinstance(channel, int):
                    raise TypeError("reserved Main capture channels must be integers")
                if not 1 <= channel <= USB_CHANNEL_COUNT:
                    raise ValueError(
                        f"reserved Main capture channel {channel} is outside 1..{USB_CHANNEL_COUNT}"
                    )

        self._capture = capture
        self._service = service
        self._roles = {int(channel): str(role) for channel, role in roles.items()}
        self._main_evidence_provider = main_evidence_provider
        self._snapshot_main_evidence_provider = snapshot_main_evidence_provider
        self._analysis_channels = tuple(
            channel
            for channel in range(1, USB_CHANNEL_COUNT + 1)
            if channel not in set(reserved_channels)
        )
        if not self._analysis_channels:
            raise ValueError("Main evidence provider cannot reserve every capture channel")
        self._channel_names = dict(channel_names or {})
        self._window_frames = int(window_frames)
        self._analysis_interval_s = float(analysis_interval_s)
        self._max_main_age_s = float(max_main_age_s)
        self._subscriber_name = subscriber_name
        self._wall_clock = wall_clock
        self._interval_clock = interval_clock
        self._extractor = Usb48FeatureExtractor()

        self._condition = threading.Condition()
        self._pending: _CaptureSnapshot | None = None
        self._running = False
        self._worker: threading.Thread | None = None
        self._last_capture_interval_s: float | None = None

        self._snapshots_captured = 0
        self._snapshots_processed = 0
        self._snapshots_replaced = 0
        self._incomplete_windows = 0
        self._missing_main_evidence = 0
        self._processing_failures = 0
        self._last_error: str | None = None

    @property
    def running(self) -> bool:
        with self._condition:
            return self._running

    def start(self) -> None:
        """Subscribe to AudioCapture and start the non-realtime worker."""
        with self._condition:
            if self._running:
                return
            self._running = True
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="live-audio-feature-worker",
                daemon=True,
            )
            self._worker.start()
        try:
            self._capture.subscribe(self._subscriber_name, self._on_audio_ready)
        except Exception:
            with self._condition:
                self._running = False
                self._condition.notify_all()
            worker = self._worker
            if worker is not None:
                worker.join(timeout=1.0)
            self._worker = None
            raise

    def stop(self, *, timeout_s: float = 1.0) -> None:
        """Detach from capture and stop analysis without stopping the audio device."""
        self._capture.unsubscribe(self._subscriber_name)
        with self._condition:
            if not self._running and self._worker is None:
                return
            self._running = False
            self._pending = None
            self._condition.notify_all()
            worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=max(0.0, float(timeout_s)))
        with self._condition:
            if self._worker is worker:
                self._worker = None

    def _on_audio_ready(self) -> None:
        """Copy one coherent raw window; never run analysis on the audio callback."""
        interval_now = float(self._interval_clock())
        with self._condition:
            if not self._running:
                return
            previous = self._last_capture_interval_s
            if previous is not None and interval_now - previous < self._analysis_interval_s:
                return
            self._last_capture_interval_s = interval_now

        channels: list[np.ndarray] = []
        for channel in range(1, USB_CHANNEL_COUNT + 1):
            samples = np.asarray(
                self._capture.get_buffer(channel, self._window_frames),
                dtype=np.float32,
            )
            if samples.ndim != 1 or samples.size != self._window_frames:
                with self._condition:
                    self._incomplete_windows += 1
                return
            channels.append(samples.copy())

        block = np.column_stack(channels).astype(np.float32, copy=False)
        snapshot = _CaptureSnapshot(audio=block, timestamp_s=float(self._wall_clock()))
        with self._condition:
            if not self._running:
                return
            if self._pending is not None:
                self._snapshots_replaced += 1
            self._pending = snapshot
            self._snapshots_captured += 1
            self._condition.notify()

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while self._running and self._pending is None:
                    self._condition.wait()
                if not self._running:
                    return
                snapshot = self._pending
                self._pending = None
            if snapshot is not None:
                self._process_snapshot(snapshot)

    def _process_snapshot(self, snapshot: _CaptureSnapshot) -> Any | None:
        try:
            if self._snapshot_main_evidence_provider is not None:
                main = self._snapshot_main_evidence_provider.from_block(
                    snapshot.audio,
                    snapshot.timestamp_s,
                )
            else:
                provider = self._main_evidence_provider
                if provider is None:  # pragma: no cover - constructor enforces XOR
                    raise RuntimeError("authoritative Main evidence provider is not configured")
                main = provider(snapshot.timestamp_s)

            if main is None:
                with self._condition:
                    self._missing_main_evidence += 1
                    self._last_error = "missing_main_evidence"
                return None

            frame = self._extractor.extract(
                snapshot.audio,
                selected_channels=self._analysis_channels,
                channel_names=self._channel_names,
                timestamp_s=snapshot.timestamp_s,
            )
            features = assemble_mix_features(
                frame,
                main,
                max_main_age_s=self._max_main_age_s,
            )
            result = self._service.process_feature_snapshot(features, dict(self._roles))
            with self._condition:
                self._snapshots_processed += 1
                self._last_error = None
            return result
        except Exception as exc:
            with self._condition:
                self._processing_failures += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
            return None

    def status(self) -> CaptureBridgeStatus:
        with self._condition:
            return CaptureBridgeStatus(
                running=self._running,
                snapshots_captured=self._snapshots_captured,
                snapshots_processed=self._snapshots_processed,
                snapshots_replaced=self._snapshots_replaced,
                incomplete_windows=self._incomplete_windows,
                missing_main_evidence=self._missing_main_evidence,
                processing_failures=self._processing_failures,
                last_error=self._last_error,
            )
