"""Temporary migration seam for externally owned live AudioCapture.

The canonical LIVE runtime owns the physical AudioCapture through
``LiveAudioCaptureSession``.  During the repository renovation the legacy
``AutoSoundcheckEngine`` still needs to *use* that capture for connection and
analysis plumbing, but it must not create, start, or stop a second physical
audio stream.

This module is deliberately infrastructure-only.  It contains no musical
policy and must disappear once ``AutoSoundcheckEngine`` is no longer a runtime
dependency of LIVE/SOUNDCHECK.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


AuditSink = Optional[Callable[[dict[str, Any]], None]]


class LegacyAudioCaptureSeamError(RuntimeError):
    """Raised when non-owning capture hand-off cannot be proven safe."""


@dataclass(frozen=True)
class LegacyAudioCaptureSeamStatus:
    bound: bool
    engine_uses_proxy: bool
    physical_capture_id: int


class _NonOwningAudioCaptureProxy:
    """Delegate reads/analysis to a capture while suppressing lifecycle calls."""

    def __init__(self, capture: Any, audit_sink: AuditSink = None):
        self._capture = capture
        self._audit_sink = audit_sink

    @property
    def physical_capture(self) -> Any:
        return self._capture

    def _audit(self, event: str) -> None:
        if self._audit_sink is not None:
            self._audit_sink(
                {
                    "event": event,
                    "physical_capture_id": id(self._capture),
                }
            )

    def start(self, *args: Any, **kwargs: Any) -> None:
        # The external LiveAudioCaptureSession already owns/opened the stream.
        self._audit("legacy_external_audio_capture_start_suppressed")
        return None

    def stop(self, *args: Any, **kwargs: Any) -> None:
        # Legacy teardown must never close a stream owned by live_runtime.
        self._audit("legacy_external_audio_capture_stop_suppressed")
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._capture, name)


class LegacyExternalAudioCaptureSeam:
    """Bind one externally owned capture to a legacy soundcheck engine.

    Contract:
    - caller starts the physical capture before :meth:`bind`;
    - the legacy engine may read/analyse through ``engine.audio_capture``;
    - legacy ``_start_audio()`` is replaced by a migration-only no-op proof;
    - legacy ``audio_capture.start()/stop()`` calls are suppressed by a proxy;
    - caller detaches after legacy teardown and then stops the canonical
      ``LiveAudioCaptureSession``.

    The seam is intentionally explicit and fail-closed.  It refuses to replace
    an already populated legacy ``audio_capture`` because that would make
    physical stream ownership ambiguous.
    """

    def __init__(
        self,
        engine: Any,
        capture: Any,
        *,
        audit_sink: AuditSink = None,
    ) -> None:
        if engine is None:
            raise ValueError("engine is required")
        if capture is None:
            raise ValueError("capture is required")
        if not callable(getattr(engine, "_start_audio", None)):
            raise LegacyAudioCaptureSeamError(
                "Legacy engine does not expose callable _start_audio migration seam"
            )

        self._engine = engine
        self._capture = capture
        self._audit_sink = audit_sink
        self._proxy = _NonOwningAudioCaptureProxy(capture, audit_sink=audit_sink)
        self._original_start_audio: Optional[Callable[..., Any]] = None
        self._bound = False

    @property
    def bound(self) -> bool:
        return self._bound

    @property
    def proxy(self) -> Any:
        return self._proxy

    def _audit(self, event: str) -> None:
        if self._audit_sink is not None:
            self._audit_sink(
                {
                    "event": event,
                    "physical_capture_id": id(self._capture),
                }
            )

    def bind(self) -> bool:
        """Give legacy code non-owning access without opening another stream."""
        if self._bound:
            return False

        existing = getattr(self._engine, "audio_capture", None)
        if existing is not None:
            raise LegacyAudioCaptureSeamError(
                "Legacy engine already has audio_capture; refusing ambiguous ownership"
            )

        original = getattr(self._engine, "_start_audio", None)
        if not callable(original):
            raise LegacyAudioCaptureSeamError(
                "Legacy engine _start_audio is no longer callable"
            )

        self._original_start_audio = original
        self._engine.audio_capture = self._proxy

        def _use_external_capture() -> bool:
            if getattr(self._engine, "audio_capture", None) is not self._proxy:
                raise LegacyAudioCaptureSeamError(
                    "External capture proxy was displaced before legacy audio startup"
                )
            self._audit("legacy_external_audio_capture_start_bypassed")
            return True

        # Instance-level override is intentional and temporary: the legacy
        # class remains frozen while LIVE runtime severs its ownership duties.
        self._engine._start_audio = _use_external_capture
        self._bound = True
        self._audit("legacy_external_audio_capture_bound")
        return True

    def detach(self) -> bool:
        """Restore the frozen legacy method without touching the physical stream."""
        if not self._bound:
            return False

        if getattr(self._engine, "audio_capture", None) is not self._proxy:
            raise LegacyAudioCaptureSeamError(
                "Legacy audio_capture changed while external ownership was bound"
            )
        if self._original_start_audio is None:
            raise LegacyAudioCaptureSeamError("Original legacy _start_audio was lost")

        self._engine.audio_capture = None
        self._engine._start_audio = self._original_start_audio
        self._original_start_audio = None
        self._bound = False
        self._audit("legacy_external_audio_capture_detached")
        return True

    def status(self) -> LegacyAudioCaptureSeamStatus:
        return LegacyAudioCaptureSeamStatus(
            bound=self._bound,
            engine_uses_proxy=(getattr(self._engine, "audio_capture", None) is self._proxy),
            physical_capture_id=id(self._capture),
        )
