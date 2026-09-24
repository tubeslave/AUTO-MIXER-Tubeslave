"""Canonical LIVE/SOUNDCHECK service with single-owner hardware lifecycle.

The configured production path now owns mixer, audio capture, startup gates and
session lifecycle entirely inside ``backend/live_runtime``.  It does not
construct or start ``AutoSoundcheckEngine``.  Sessions without an explicit
capture bridge retain the frozen legacy path temporarily as compatibility
proof while references are severed.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable

from .audio_capture_session import LiveAudioCaptureConfig, LiveAudioCaptureSession
from .contracts import SoundcheckState
from .legacy_mixer_seam import LegacyExternalMixerSeam
from .mixer_session import LiveMixerConfig, LiveMixerSession
from .service_core import LiveCaptureBridgeConfig, LiveSnapshotResult, LiveStartRequest
from .service_core import LiveSoundcheckService as _CoreLiveSoundcheckService
from .session_lifecycle import LiveSessionLifecycle

__all__ = [
    "LiveCaptureBridgeConfig",
    "LiveSnapshotResult",
    "LiveStartRequest",
    "LiveSoundcheckService",
]


class LiveSoundcheckService(_CoreLiveSoundcheckService):
    """Authoritative live service owning canonical mixer/audio/session lifecycle.

    Sessions with an explicit ``capture_bridge`` are the migrated production
    composition.  They never construct the legacy decision engine.  Sessions
    without it retain the legacy path temporarily for compatibility evidence
    only; that path is not expanded with new decision features.
    """

    def __init__(
        self,
        *args: Any,
        mixer_session_factory: Callable[..., LiveMixerSession] | None = None,
        legacy_mixer_seam_factory: Callable[..., LegacyExternalMixerSeam] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._mixer_session_factory = mixer_session_factory
        self._legacy_mixer_seam_factory = legacy_mixer_seam_factory
        self._mixer_session: LiveMixerSession | Any | None = None
        self._legacy_mixer_seam: LegacyExternalMixerSeam | Any | None = None

    @staticmethod
    def _is_canonical_session_handle(value: Any) -> bool:
        return isinstance(value, LiveSessionLifecycle)

    def _build_owned_mixer(self) -> Any:
        if self._request is None or self._request.capture_bridge is None:
            raise RuntimeError("Canonical mixer ownership requires configured capture_bridge")
        if self._engine is None:
            raise RuntimeError("Canonical live session lifecycle must exist before mixer start")
        if self._mixer_session is not None or self._legacy_mixer_seam is not None:
            raise RuntimeError("Canonical mixer ownership is already established")

        config = LiveMixerConfig(
            mixer_type=self._request.mixer_type,
            mixer_ip=self._request.mixer_ip,
            mixer_port=self._request.mixer_port,
            auto_discover=True,
        )
        session_factory = self._mixer_session_factory or LiveMixerSession
        session = session_factory(config, audit_sink=self._record_control_audit)
        client = session.start()
        seam = None
        try:
            if not self._is_canonical_session_handle(self._engine):
                seam_factory = self._legacy_mixer_seam_factory or LegacyExternalMixerSeam
                seam = seam_factory(
                    self._engine,
                    session,
                    audit_sink=self._record_control_audit,
                )
                seam.bind()
        except Exception:
            session.stop()
            raise

        self._mixer_session = session
        self._legacy_mixer_seam = seam
        target = session.target
        self._record_control_audit(
            {
                "event": "live_service_mixer_owned",
                "mixer_type": getattr(target, "mixer_type", None),
                "ip": getattr(target, "ip", None),
                "port": getattr(target, "port", None),
                "legacy_seam_bound": seam is not None,
            }
        )
        return client

    def _release_owned_mixer(self) -> list[BaseException]:
        """Detach any legacy non-owner, then disconnect the physical mixer once."""
        errors: list[BaseException] = []
        seam = self._legacy_mixer_seam
        session = self._mixer_session
        self._legacy_mixer_seam = None
        self._mixer_session = None

        if seam is not None:
            try:
                seam.detach()
            except BaseException as exc:
                errors.append(exc)
                self._record_control_audit(
                    {
                        "event": "live_service_mixer_seam_detach_failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
        if session is not None:
            try:
                session.stop()
            except BaseException as exc:
                errors.append(exc)
                self._record_control_audit(
                    {
                        "event": "live_service_mixer_stop_failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
        if not errors and (seam is not None or session is not None):
            self._record_control_audit({"event": "live_service_mixer_released"})
        return errors

    def _build_owned_audio_capture(self) -> Any:
        """Create the canonical physical stream without binding a legacy consumer."""
        if self._request is None or self._request.capture_bridge is None:
            raise RuntimeError("Canonical audio capture requires configured capture_bridge")
        if self._engine is None:
            raise RuntimeError("Canonical live session lifecycle must exist before audio start")
        if self._audio_capture_session is not None or self._legacy_audio_capture_seam is not None:
            raise RuntimeError("Canonical audio capture ownership is already established")

        config = self._request.capture_bridge
        required_channels = set(config.roles)
        required_channels.update(config.patch_contract.tap.channels)
        capture_config = LiveAudioCaptureConfig(
            audio_device_name=self._request.audio_device_name,
            num_channels=self._request.num_channels,
            sample_rate=48_000,
            required_channel_ids=tuple(sorted(required_channels)),
        )
        session_factory = self._audio_capture_session_factory or LiveAudioCaptureSession
        session = session_factory(capture_config, audit_sink=self._record_control_audit)
        capture = session.start()

        self._audio_capture_session = session
        self._legacy_audio_capture_seam = None
        self._record_control_audit(
            {
                "event": "live_service_audio_capture_owned",
                "channels": self._request.num_channels,
                "sample_rate": 48_000,
                "required_channel_ids": list(sorted(required_channels)),
                "legacy_seam_bound": False,
            }
        )
        return capture

    def _active_wing_transport(self) -> Any:
        """Return the live-runtime-owned physical WING client when migrated."""
        if self._engine is None or self._request is None:
            raise RuntimeError("Live soundcheck is not running")

        mixer_type = str(self._request.mixer_type or "").strip().lower().replace("-", "_")
        if mixer_type not in {"wing", "wing_rack", "behringer_wing"}:
            raise NotImplementedError(
                f"Authoritative live control adapter is not migrated for mixer type {self._request.mixer_type!r}"
            )

        session = self._mixer_session
        if session is not None:
            if not bool(getattr(session, "connected", False)):
                raise RuntimeError("Canonical LiveMixerSession is not connected")
            client = getattr(session, "client", None)
        else:
            return super()._active_wing_transport()

        if client is None:
            raise RuntimeError("Physical WING transport is not ready")
        if not hasattr(client, "send") or not hasattr(client, "subscribe"):
            raise TypeError("Active WING transport does not expose send/subscribe")
        return client

    def start(
        self,
        request: LiveStartRequest,
        *,
        on_state_change=None,
        on_channel_update=None,
        on_observation=None,
    ) -> Any:
        """Start one session; configured live paths never run the legacy engine."""
        if self.is_active():
            raise RuntimeError("Live soundcheck engine already running")

        self._reset_control_plane()
        self._reset_iteration()
        self._reset_startup()
        self._control_audit.clear()
        self._capture_bridge = None
        self._capture_bridge_error = None
        self._audio_capture_session = None
        self._legacy_audio_capture_seam = None
        self._mixer_session = None
        self._legacy_mixer_seam = None
        self._lifecycle_stopping = False

        self._request = request
        self._soundcheck_state = SoundcheckState.DISCOVER

        if request.capture_bridge is not None:
            lifecycle = LiveSessionLifecycle(
                on_state_change=on_state_change,
                audit_sink=self._record_control_audit,
                stop_callback=self.stop,
            )
            self._engine = lifecycle
            lifecycle.begin_start()
            try:
                self._build_owned_mixer()
                self._build_owned_audio_capture()
                self._try_start_capture_bridge()
                lifecycle.mark_running()
            except Exception as exc:
                self._lifecycle_stopping = True
                try:
                    lifecycle.mark_error(exc)
                    self._stop_capture_bridge()
                    self._release_owned_audio_capture()
                    self._release_owned_mixer()
                finally:
                    self._engine = None
                    self._request = None
                    self._reset_control_plane()
                    self._reset_iteration()
                    self._reset_startup()
                    self._lifecycle_stopping = False
                raise
            return lifecycle

        def lifecycle_state_change(state, message):
            if on_state_change is not None:
                on_state_change(state, message)

        engine = self.create_engine(
            request,
            on_state_change=lifecycle_state_change,
            on_channel_update=on_channel_update,
            on_observation=on_observation,
        )
        self._engine = engine
        try:
            engine.start_async()
        except Exception:
            self._lifecycle_stopping = True
            try:
                try:
                    engine.stop()
                except Exception as cleanup_exc:
                    self._record_control_audit(
                        {
                            "event": "live_legacy_engine_start_cleanup_failed",
                            "reason": f"{type(cleanup_exc).__name__}: {cleanup_exc}",
                        }
                    )
            finally:
                self._engine = None
                self._request = None
                self._reset_control_plane()
                self._reset_iteration()
                self._reset_startup()
                self._lifecycle_stopping = False
            raise
        return engine

    def stop(self) -> bool:
        """Stop canonical consumers/owners or the frozen legacy compatibility path."""
        engine = self._engine
        if engine is None:
            return False

        canonical = self._is_canonical_session_handle(engine)
        self._lifecycle_stopping = True
        first_error: BaseException | None = None
        if canonical:
            engine.begin_stop()
        try:
            self._stop_capture_bridge()
            if not canonical:
                try:
                    engine.stop()
                except BaseException as exc:
                    first_error = exc
                    self._record_control_audit(
                        {
                            "event": "live_legacy_engine_stop_failed",
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
            for exc in self._release_owned_audio_capture():
                if first_error is None:
                    first_error = exc
            for exc in self._release_owned_mixer():
                if first_error is None:
                    first_error = exc

            if canonical:
                if first_error is None:
                    engine.mark_stopped()
                else:
                    engine.mark_error(first_error)
        finally:
            self._engine = None
            self._request = None
            self._reset_control_plane()
            self._reset_iteration()
            self._reset_startup()
            self._lifecycle_stopping = False
        if first_error is not None:
            raise first_error
        return True

    def get_status(self) -> dict[str, Any]:
        """Expose physical ownership without probing any legacy write-blocking proxy."""
        session = self._mixer_session
        if session is None:
            status = super().get_status()
            status["mixer_transport_owned_by_live_runtime"] = False
            status["legacy_engine_attached"] = bool(
                self._engine is not None
                and not self._is_canonical_session_handle(self._engine)
            )
            return status

        engine = self._engine
        if engine is None:
            return {
                "state": "idle",
                "mixer_connected": False,
                "audio_running": False,
                "control_plane_ready": False,
                "control_audit_count": len(self._control_audit),
                "iteration_active": False,
                "iteration_hold_reason": None,
                "soundcheck_state": "idle",
                "patch_verify_verified": None,
                "patch_verify_reason": None,
                "patch_verify_physical_source": None,
                "capture_bridge_configured": False,
                "capture_bridge_running": False,
                "capture_bridge_error": self._capture_bridge_error,
                "capture_bridge": None,
                "audio_capture_owned_by_live_runtime": False,
                "mixer_transport_owned_by_live_runtime": False,
                "legacy_engine_attached": False,
            }

        raw = engine.get_status() if hasattr(engine, "get_status") else {}
        status = dict(raw or {})
        mode = self.active_mode
        if mode is not None:
            status["mode"] = mode.value

        iteration_active = bool(self._iteration and self._iteration.has_active_hypothesis)
        iteration_hold_reason = self._iteration.hold_reason if self._iteration else None
        soundcheck_state = (
            SoundcheckState.HOLD.value
            if iteration_hold_reason is not None
            else SoundcheckState.VERIFY.value
            if iteration_active
            else self._soundcheck_state.value
            if self._soundcheck_state is not None
            else "idle"
        )
        patch_result = self._last_patch_verify
        bridge = self._capture_bridge
        bridge_status = None
        if bridge is not None and hasattr(bridge, "status"):
            try:
                bridge_status = asdict(bridge.status())
            except (TypeError, ValueError):
                bridge_status = None
        bridge_running = bool(bridge_status.get("running")) if bridge_status else bridge is not None
        bridge_configured = bool(self._request and self._request.capture_bridge is not None)
        audio_owned = self._audio_capture_session is not None

        connected = bool(getattr(session, "connected", False))
        client = getattr(session, "client", None)
        mixer_type = str(self._request.mixer_type if self._request else "").strip().lower().replace("-", "_")
        status["mixer_connected"] = connected
        status["control_plane_ready"] = (
            connected
            and mixer_type in {"wing", "wing_rack", "behringer_wing"}
            and client is not None
            and hasattr(client, "send")
            and hasattr(client, "subscribe")
        )
        status["control_audit_count"] = len(self._control_audit)
        status["iteration_active"] = iteration_active
        status["iteration_hold_reason"] = iteration_hold_reason
        status["soundcheck_state"] = soundcheck_state
        status["patch_verify_verified"] = patch_result.verified if patch_result else None
        status["patch_verify_reason"] = patch_result.reason if patch_result else None
        status["patch_verify_physical_source"] = (
            patch_result.physical_evidence.source
            if patch_result and patch_result.physical_evidence is not None
            else None
        )
        status["capture_bridge_configured"] = bridge_configured
        status["capture_bridge_running"] = bridge_running
        status["capture_bridge_error"] = self._capture_bridge_error
        status["capture_bridge"] = bridge_status
        status["audio_capture_owned_by_live_runtime"] = audio_owned
        status["mixer_transport_owned_by_live_runtime"] = True
        status["legacy_engine_attached"] = not self._is_canonical_session_handle(engine)
        return status
