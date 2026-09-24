"""Canonical live audio-device selection and AudioCapture lifecycle ownership.

This module extracts only validated transport plumbing from the legacy
``AutoSoundcheckEngine``.  It contains no channel classification, musical
presets, AutoFOH policy, Director logic or mixer writes.

The live runtime is deliberately stricter than the legacy fallback path:
requested stream width and sample rate are explicit contracts, a device with
insufficient inputs is rejected instead of silently shrinking the stream, and a
real-device start that falls back to SILENCE is treated as a startup failure.
That matters for the USB 48x48 path because ``LiveAudioCaptureBridge`` requires
an exact 48-channel / 48 kHz stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from audio_capture import AudioCapture, AudioSourceType
from audio_device_scanner import (
    AudioDevice,
    AudioProtocol,
    scan_audio_devices,
    select_best_device,
)


class LiveAudioCaptureError(RuntimeError):
    """Fail-closed live capture setup or lifecycle error."""


@dataclass(frozen=True)
class LiveAudioCaptureConfig:
    """Explicit audio-plane contract for one live session."""

    audio_device_name: str | None = None
    num_channels: int = 48
    sample_rate: int = 48_000
    block_size: int = 1024
    buffer_seconds: float = 5.0
    required_channel_ids: Sequence[int] = ()

    def __post_init__(self) -> None:
        if isinstance(self.num_channels, bool) or int(self.num_channels) < 2:
            raise ValueError("num_channels must be an integer >= 2")
        if isinstance(self.sample_rate, bool) or int(self.sample_rate) <= 0:
            raise ValueError("sample_rate must be > 0")
        if isinstance(self.block_size, bool) or int(self.block_size) <= 0:
            raise ValueError("block_size must be > 0")
        if isinstance(self.buffer_seconds, bool) or float(self.buffer_seconds) <= 0:
            raise ValueError("buffer_seconds must be > 0")

        num_channels = int(self.num_channels)
        required: list[int] = []
        for raw in self.required_channel_ids:
            if isinstance(raw, bool):
                raise TypeError("required_channel_ids must contain integers")
            try:
                channel = int(raw)
            except (TypeError, ValueError) as exc:
                raise TypeError("required_channel_ids must contain integers") from exc
            if channel < 1 or channel > num_channels:
                raise ValueError(
                    f"required capture channel {channel} is outside 1..{num_channels}"
                )
            required.append(channel)

        name = str(self.audio_device_name or "").strip() or None
        object.__setattr__(self, "audio_device_name", name)
        object.__setattr__(self, "num_channels", num_channels)
        object.__setattr__(self, "sample_rate", int(self.sample_rate))
        object.__setattr__(self, "block_size", int(self.block_size))
        object.__setattr__(self, "buffer_seconds", float(self.buffer_seconds))
        object.__setattr__(self, "required_channel_ids", tuple(sorted(set(required))))


@dataclass(frozen=True)
class LiveAudioCaptureStatus:
    running: bool
    requested_channels: int
    effective_channels: int
    sample_rate: int
    device_index: int | None
    device_name: str | None
    protocol: str | None
    source_type: str | None


def _preferred_protocol(name: str | None) -> AudioProtocol | None:
    lowered = str(name or "").lower()
    if "soundgrid" in lowered or "waves" in lowered:
        return AudioProtocol.SOUNDGRID
    if "dante" in lowered:
        return AudioProtocol.DANTE
    return None


def _source_type_for_protocol(protocol: AudioProtocol) -> AudioSourceType:
    if protocol is AudioProtocol.SOUNDGRID:
        return AudioSourceType.SOUNDGRID
    if protocol is AudioProtocol.DANTE:
        return AudioSourceType.DANTE
    return AudioSourceType.SOUNDDEVICE


class LiveAudioCaptureSession:
    """Own one canonical live ``AudioCapture`` from selection through teardown.

    The constructor accepts narrow injectable scan/select/capture primitives so
    CI can exercise the whole ownership contract without opening audio hardware.
    """

    def __init__(
        self,
        config: LiveAudioCaptureConfig,
        *,
        scan_devices: Callable[[], list[AudioDevice]] = scan_audio_devices,
        select_device: Callable[..., AudioDevice | None] = select_best_device,
        capture_factory: Callable[..., Any] = AudioCapture,
        audit_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if not isinstance(config, LiveAudioCaptureConfig):
            raise TypeError("config must be LiveAudioCaptureConfig")
        self.config = config
        self._scan_devices = scan_devices
        self._select_device = select_device
        self._capture_factory = capture_factory
        self._audit_sink = audit_sink

        self._capture: Any | None = None
        self._devices: tuple[AudioDevice, ...] = ()
        self._selected_device: AudioDevice | None = None
        self._source_type: AudioSourceType | None = None
        self._running = False

    @property
    def capture(self) -> Any | None:
        return self._capture

    @property
    def devices(self) -> tuple[AudioDevice, ...]:
        return self._devices

    @property
    def selected_device(self) -> AudioDevice | None:
        return self._selected_device

    @property
    def running(self) -> bool:
        return self._running

    def _audit(self, event: str, **payload: Any) -> None:
        if self._audit_sink is not None:
            self._audit_sink({"event": event, **payload})

    def _select(self) -> AudioDevice:
        devices = tuple(self._scan_devices() or ())
        self._devices = devices
        preferred = _preferred_protocol(self.config.audio_device_name)
        best = self._select_device(
            list(devices),
            preferred_protocol=preferred,
            preferred_name=self.config.audio_device_name,
            min_channels=self.config.num_channels,
        )
        if best is None:
            raise LiveAudioCaptureError(
                "No live audio input device satisfies the explicit capture contract"
            )
        if int(best.max_input_channels) < self.config.num_channels:
            raise LiveAudioCaptureError(
                f"Audio device {best.name!r} exposes {best.max_input_channels} inputs; "
                f"{self.config.num_channels} are required"
            )
        return best

    def start(self) -> Any:
        """Select, construct and start capture exactly once.

        A running session is idempotent. A failed startup releases any partially
        constructed capture and leaves the session stopped.
        """
        if self._running and self._capture is not None:
            return self._capture

        capture: Any | None = None
        best: AudioDevice | None = None
        try:
            best = self._select()
            source_type = _source_type_for_protocol(best.protocol)
            capture = self._capture_factory(
                num_channels=self.config.num_channels,
                sample_rate=self.config.sample_rate,
                buffer_seconds=self.config.buffer_seconds,
                block_size=self.config.block_size,
                source_type=source_type,
                device_name=best.index,
            )
            capture.start()

            actual_channels = int(getattr(capture, "num_channels", 0))
            actual_rate = int(getattr(capture, "sample_rate", 0))
            actual_source = getattr(capture, "source_type", source_type)
            if actual_channels != self.config.num_channels:
                raise LiveAudioCaptureError(
                    f"AudioCapture opened {actual_channels} channels; "
                    f"{self.config.num_channels} are required"
                )
            if actual_rate != self.config.sample_rate:
                raise LiveAudioCaptureError(
                    f"AudioCapture opened {actual_rate} Hz; "
                    f"{self.config.sample_rate} Hz is required"
                )
            if (
                source_type is not AudioSourceType.SILENCE
                and actual_source is AudioSourceType.SILENCE
            ):
                raise LiveAudioCaptureError(
                    "Real live audio capture fell back to SILENCE"
                )

            self._capture = capture
            self._selected_device = best
            self._source_type = actual_source
            self._running = True
            self._audit(
                "live_audio_capture_started",
                device_index=int(best.index),
                device_name=str(best.name),
                protocol=best.protocol.value,
                channels=actual_channels,
                sample_rate=actual_rate,
                source_type=getattr(actual_source, "value", str(actual_source)),
            )
            return capture
        except Exception as exc:
            if capture is not None:
                try:
                    capture.stop()
                except Exception:
                    pass
            self._capture = None
            self._selected_device = best
            self._source_type = None
            self._running = False
            self._audit(
                "live_audio_capture_start_failed",
                reason=f"{type(exc).__name__}: {exc}",
            )
            if isinstance(exc, LiveAudioCaptureError):
                raise
            raise LiveAudioCaptureError(
                f"Live audio capture startup failed: {type(exc).__name__}: {exc}"
            ) from exc

    def stop(self) -> bool:
        """Stop and release the owned capture; safe to call repeatedly."""
        capture = self._capture
        if capture is None:
            self._running = False
            return False

        self._capture = None
        self._running = False
        try:
            capture.stop()
        except Exception as exc:
            self._audit(
                "live_audio_capture_stop_failed",
                reason=f"{type(exc).__name__}: {exc}",
            )
            raise LiveAudioCaptureError(
                f"Live audio capture stop failed: {type(exc).__name__}: {exc}"
            ) from exc

        self._audit("live_audio_capture_stopped")
        return True

    def status(self) -> LiveAudioCaptureStatus:
        capture = self._capture
        device = self._selected_device
        source = self._source_type
        return LiveAudioCaptureStatus(
            running=self._running,
            requested_channels=self.config.num_channels,
            effective_channels=int(getattr(capture, "num_channels", 0)) if capture else 0,
            sample_rate=int(getattr(capture, "sample_rate", self.config.sample_rate)),
            device_index=int(device.index) if device is not None else None,
            device_name=str(device.name) if device is not None else None,
            protocol=device.protocol.value if device is not None else None,
            source_type=getattr(source, "value", str(source)) if source is not None else None,
        )
