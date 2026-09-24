"""Canonical LIVE/SOUNDCHECK service with single-owner hardware lifecycle.

The existing live decision/startup/audio lifecycle remains in ``service_core``.
This facade completes the next migration seam by making ``LiveMixerSession``
the owner of physical mixer discovery/connect/disconnect whenever the explicit
canonical capture composition is enabled. The frozen legacy engine receives
only ``LegacyExternalMixerSeam``'s read-only proxy; it cannot create a second
connection or recover write authority, including in BENCH_TEST.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable

from .contracts import SoundcheckState
from .legacy_mixer_seam import LegacyExternalMixerSeam
from .mixer_session import LiveMixerConfig, LiveMixerSession
from .service_core import LiveCaptureBridgeConfig, LiveSnapshotResult, LiveStartRequest
from .service_core import LiveSoundcheckService as _CoreLiveSoundcheckService

__all__ = [
    "LiveCaptureBridgeConfig",
    "LiveSnapshotResult",
    "LiveStartRequest",
    "LiveSoundcheckService",
]


class LiveSoundcheckService(_CoreLiveSoundcheckService):
    """Authoritative live service owning canonical mixer and audio sessions.

    Sessions with an explicit ``capture_bridge`` are the migrated production
    composition and therefore receive single-owner mixer + audio lifecycle.
    Sessions without it retain the legacy path temporarily for compatibility
    evidence only; that path is not expanded with new decision features.
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

    def _build_owned_mixer(self) -> Any:
        if self._request is None or self._request.capture_bridge is None:
            raise RuntimeError("Canonical mixer ownership requires configured capture_bridge")
        if self._engine is None:
            raise RuntimeError("Legacy compatibility engine must exist before mixer hand-off")
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
        try:
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
            }
        )
        return client

    def _release_owned_mixer(self) -> list[BaseException]:
        """Detach the legacy non-owner, then disconnect the physical mixer once."""
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
        """Start one session with canonical mixer+audio ownership when configured."""
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

        def lifecycle_state_change(state, message):
            self._try_start_capture_bridge()
            if on_state_change is not None:
                on_state_change(state, message)

        engine = self.create_engine(
            request,
            on_state_change=lifecycle_state_change,
            on_channel_update=on_channel_update,
            on_observation=on_observation,
        )
        self._engine = engine
        self._request = request
        self._soundcheck_state = SoundcheckState.DISCOVER
        legacy_start_attempted = False
        try:
            if request.capture_bridge is not None:
                self._build_owned_mixer()
                self._build_owned_audio_capture()
            legacy_start_attempted = True
            engine.start_async()
            self._try_start_capture_bridge()
        except Exception:
            self._lifecycle_stopping = True
            try:
                self._stop_capture_bridge()
                if legacy_start_attempted:
                    try:
                        engine.stop()
                    except Exception as cleanup_exc:
                        self._record_control_audit(
                            {
                                "event": "live_legacy_engine_start_cleanup_failed",
                                "reason": f"{type(cleanup_exc).__name__}: {cleanup_exc}",
                            }
                        )
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
        return engine

    def stop(self) -> bool:
        """Stop bridge/legacy consumers before canonical audio and mixer owners."""
        engine = self._engine
        if engine is None:
            return False

        self._lifecycle_stopping = True
        first_error: BaseException | None = None
        try:
            self._stop_capture_bridge()
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
        """Expose physical ownership without probing the legacy write-blocking proxy."""
        session = self._mixer_session
        if session is None:
            status = super().get_status()
            status["mixer_transport_owned_by_live_runtime"] = False
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
        return status
