"""Authoritative live soundcheck construction and lifecycle seam.

This module is the only new-live layer allowed to construct the legacy
``AutoSoundcheckEngine`` while its useful hardware/audio plumbing is migrated
behind ``live_runtime`` interfaces. Callers depend on this service rather than
constructing, starting, stopping or inspecting the legacy decision engine
independently.

The compatibility bridge is temporary. It removes parallel decision authority
from the UI/transport layer without forcing a flag-day rewrite of WING/audio
capture.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import threading
from typing import Any, Callable, Mapping

from .audio_capture_session import LiveAudioCaptureConfig, LiveAudioCaptureSession
from .capture_bridge import LiveAudioCaptureBridge
from .contracts import (
    EqBandLocator,
    LiveMode,
    MixFeatures,
    ProposedAction,
    SoundcheckState,
    VerifiedAction,
)
from .control_plane import LiveControlPlane, RollbackExecution, WriteExecution
from .decision_engine import LiveHypothesis, propose_one
from .eq_locator import EqTargetEvidence, RealtimeEqLocatorSelector
from .feature_stream import MainFeatureEvidence
from .iteration import IterationCoordinator, IterationPhase, IterationResult
from .legacy_audio_capture_seam import LegacyExternalAudioCaptureSeam
from .main_evidence import PostConsoleMainTapEvidenceProvider
from .patch_startup import MainTapPatchStartupCoordinator, PatchVerifyStartupResult
from .patch_verify import MainTapPatchContract
from .wing_adapter import WingWriteAdapter


@dataclass(frozen=True)
class LiveCaptureBridgeConfig:
    """Explicit production composition for the service-owned capture bridge.

    The patch contract is the single source of truth for reserved post-console
    Main capture slots. The service derives the snapshot Main provider from
    that contract, preventing callers from accidentally composing a mismatched
    provider/route pair. Channel roles remain explicit evidence supplied by
    the composition root; no legacy classifier or ``auto_*`` policy is reused.
    """

    patch_contract: MainTapPatchContract
    roles: Mapping[int, str]
    channel_names: Mapping[int, str] | None = None
    window_frames: int = 2048
    analysis_interval_s: float = 0.100

    def __post_init__(self) -> None:
        if not isinstance(self.patch_contract, MainTapPatchContract):
            raise TypeError("patch_contract must be MainTapPatchContract")
        if isinstance(self.window_frames, bool) or int(self.window_frames) <= 0:
            raise ValueError("window_frames must be > 0")
        if isinstance(self.analysis_interval_s, bool) or float(self.analysis_interval_s) < 0.0:
            raise ValueError("analysis_interval_s must be >= 0")
        object.__setattr__(self, "window_frames", int(self.window_frames))
        object.__setattr__(self, "analysis_interval_s", float(self.analysis_interval_s))
        object.__setattr__(
            self,
            "roles",
            {int(channel): str(role) for channel, role in dict(self.roles).items()},
        )
        object.__setattr__(
            self,
            "channel_names",
            {int(channel): str(name) for channel, name in dict(self.channel_names or {}).items()},
        )


@dataclass(frozen=True)
class LiveStartRequest:
    mixer_type: str
    mixer_ip: str
    mixer_port: int
    audio_device_name: str
    num_channels: int
    selected_channels: list[int]
    mode: LiveMode = LiveMode.OBSERVE
    capture_bridge: LiveCaptureBridgeConfig | None = None


@dataclass(frozen=True)
class LiveSnapshotResult:
    """One causal realtime snapshot step through the live soundcheck loop."""

    state: SoundcheckState
    hypothesis: LiveHypothesis | None = None
    iteration: IterationResult | None = None
    reason: str | None = None


class LiveSoundcheckService:
    """Authoritative lifecycle, decision-iteration and live-control seam.

    ``AutoSoundcheckEngine`` remains an ADAPT dependency because it still owns
    useful mixer discovery, connection and readback plumbing. New callers do
    not own that engine directly: they start/stop/query this service.

    WING mutations owned by the new architecture are executed through
    :class:`LiveControlPlane`. One-hypothesis proposal/verification is owned by
    :class:`IterationCoordinator`; the legacy engine never receives decision
    authority from this service. The autonomous feature loop is additionally
    gated by the canonical startup state machine: DISCOVER -> PATCH_VERIFY ->
    LISTEN/HOLD.

    When ``LiveStartRequest.capture_bridge`` is configured, this service owns
    the physical ``AudioCapture`` through ``LiveAudioCaptureSession``. A
    temporary non-owning seam lets the frozen legacy engine consume that stream
    without opening or closing a second device. Teardown is deliberately
    ordered: feature bridge -> legacy engine -> seam detach -> physical capture.
    """

    legacy_bridge = True

    def __init__(
        self,
        engine_factory: Callable[..., Any] | None = None,
        *,
        audit_sink: Callable[[dict[str, Any]], None] | None = None,
        iteration_verification_window_s: float = 1.0,
        patch_startup_factory: Callable[..., MainTapPatchStartupCoordinator] | None = None,
        capture_bridge_factory: Callable[..., LiveAudioCaptureBridge] | None = None,
        audio_capture_session_factory: Callable[..., LiveAudioCaptureSession] | None = None,
        legacy_audio_capture_seam_factory: Callable[..., LegacyExternalAudioCaptureSeam] | None = None,
    ):
        if iteration_verification_window_s < 0:
            raise ValueError("iteration_verification_window_s must be >= 0")
        self._engine_factory = engine_factory
        self._patch_startup_factory = patch_startup_factory
        self._capture_bridge_factory = capture_bridge_factory
        self._audio_capture_session_factory = audio_capture_session_factory
        self._legacy_audio_capture_seam_factory = legacy_audio_capture_seam_factory
        self._engine: Any | None = None
        self._request: LiveStartRequest | None = None
        self._control_plane: LiveControlPlane | None = None
        self._control_adapter: WingWriteAdapter | None = None
        self._control_transport: Any | None = None
        self._control_audit: list[dict[str, Any]] = []
        self._external_audit_sink = audit_sink
        self._iteration_verification_window_s = float(iteration_verification_window_s)
        self._iteration: IterationCoordinator | None = None
        self._iteration_hypothesis: LiveHypothesis | None = None
        self._soundcheck_state: SoundcheckState | None = None
        self._last_patch_verify: PatchVerifyStartupResult | None = None
        self._capture_bridge: LiveAudioCaptureBridge | Any | None = None
        self._capture_bridge_error: str | None = None
        self._capture_bridge_lock = threading.RLock()
        self._audio_capture_session: LiveAudioCaptureSession | Any | None = None
        self._legacy_audio_capture_seam: LegacyExternalAudioCaptureSeam | Any | None = None
        self._lifecycle_stopping = False

    @property
    def active_engine(self) -> Any | None:
        """Temporary compatibility view for legacy server cleanup/readback."""
        return self._engine

    @property
    def active_mode(self) -> LiveMode | None:
        return self._request.mode if self._request else None

    @property
    def control_audit_events(self) -> list[dict[str, Any]]:
        """Return a copy of new-runtime control decisions for HIL inspection."""
        return [dict(event) for event in self._control_audit]

    @staticmethod
    def _legacy_flags(mode: LiveMode) -> tuple[bool, bool]:
        """Map new runtime modes onto the temporary legacy engine flags."""
        if mode in (LiveMode.OBSERVE, LiveMode.PROPOSE, LiveMode.FREEZE):
            return True, False
        return False, True

    def _factory(self) -> Callable[..., Any]:
        factory = self._engine_factory
        if factory is None:
            # Sole compatibility import to delete once discovery/connection moves.
            from auto_soundcheck_engine import AutoSoundcheckEngine

            factory = AutoSoundcheckEngine
        return factory

    def _record_control_audit(self, payload: dict[str, Any]) -> None:
        event = dict(payload)
        self._control_audit.append(event)
        if self._external_audit_sink is not None:
            self._external_audit_sink(event)

    def _reset_control_plane(self) -> None:
        self._control_plane = None
        self._control_adapter = None
        self._control_transport = None

    def _reset_iteration(self) -> None:
        self._iteration = None
        self._iteration_hypothesis = None

    def _reset_startup(self) -> None:
        self._soundcheck_state = None
        self._last_patch_verify = None

    def _build_owned_audio_capture(self) -> Any:
        if self._request is None or self._request.capture_bridge is None:
            raise RuntimeError("Canonical audio capture requires configured capture_bridge")
        if self._engine is None:
            raise RuntimeError("Legacy connection engine must exist before capture hand-off")
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
        try:
            seam_factory = self._legacy_audio_capture_seam_factory or LegacyExternalAudioCaptureSeam
            seam = seam_factory(
                self._engine,
                capture,
                audit_sink=self._record_control_audit,
            )
            seam.bind()
        except Exception:
            session.stop()
            raise

        self._audio_capture_session = session
        self._legacy_audio_capture_seam = seam
        self._record_control_audit(
            {
                "event": "live_service_audio_capture_owned",
                "channels": self._request.num_channels,
                "sample_rate": 48_000,
                "required_channel_ids": list(sorted(required_channels)),
            }
        )
        return capture

    def _release_owned_audio_capture(self) -> list[BaseException]:
        """Detach legacy non-owner, then stop the physical stream exactly once."""
        errors: list[BaseException] = []
        seam = self._legacy_audio_capture_seam
        session = self._audio_capture_session
        self._legacy_audio_capture_seam = None
        self._audio_capture_session = None

        if seam is not None:
            try:
                seam.detach()
            except BaseException as exc:
                errors.append(exc)
                self._record_control_audit(
                    {
                        "event": "live_service_audio_capture_seam_detach_failed",
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
                        "event": "live_service_audio_capture_stop_failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
        if not errors and (seam is not None or session is not None):
            self._record_control_audit({"event": "live_service_audio_capture_released"})
        return errors

    def _build_capture_bridge(self, capture: Any) -> LiveAudioCaptureBridge | Any:
        if self._request is None or self._request.capture_bridge is None:
            raise RuntimeError("Live capture bridge is not configured")
        config = self._request.capture_bridge
        tap = config.patch_contract.tap
        main_provider = PostConsoleMainTapEvidenceProvider(
            tap.left_channel,
            tap.right_channel,
        )
        factory = self._capture_bridge_factory or LiveAudioCaptureBridge
        return factory(
            capture,
            self,
            roles=config.roles,
            channel_names=config.channel_names,
            snapshot_main_evidence_provider=main_provider,
            patch_contract=config.patch_contract,
            window_frames=config.window_frames,
            analysis_interval_s=config.analysis_interval_s,
        )

    def _try_start_capture_bridge(self) -> bool:
        """Attach the canonical feature bridge once an AudioCapture exists."""
        with self._capture_bridge_lock:
            if self._lifecycle_stopping:
                return False
            if self._capture_bridge is not None:
                return True
            if self._capture_bridge_error is not None:
                return False
            if self._engine is None or self._request is None:
                return False
            config = self._request.capture_bridge
            if config is None:
                return False

            capture = None
            if self._audio_capture_session is not None:
                capture = getattr(self._audio_capture_session, "capture", None)
            if capture is None:
                capture = getattr(self._engine, "audio_capture", None)
            if capture is None:
                return False
            physical_capture = getattr(capture, "physical_capture", capture)

            try:
                bridge = self._build_capture_bridge(physical_capture)
                bridge.start()
            except Exception as exc:
                self._capture_bridge_error = f"{type(exc).__name__}: {exc}"
                previous = self._soundcheck_state
                self._soundcheck_state = SoundcheckState.HOLD
                self._record_control_audit(
                    {
                        "event": "live_capture_bridge_start_failed",
                        "state_before": previous.value if previous is not None else None,
                        "state_after": SoundcheckState.HOLD.value,
                        "reason": self._capture_bridge_error,
                    }
                )
                return False

            self._capture_bridge = bridge
            self._record_control_audit(
                {
                    "event": "live_capture_bridge_started",
                    "reserved_main_channels": list(config.patch_contract.tap.channels),
                    "role_count": len(config.roles),
                    "window_frames": config.window_frames,
                    "analysis_interval_s": config.analysis_interval_s,
                }
            )
            return True

    def _stop_capture_bridge(self) -> bool:
        with self._capture_bridge_lock:
            bridge = self._capture_bridge
            self._capture_bridge = None
        if bridge is None:
            return False
        try:
            bridge.stop()
        except Exception as exc:
            self._capture_bridge_error = f"{type(exc).__name__}: {exc}"
            self._record_control_audit(
                {
                    "event": "live_capture_bridge_stop_failed",
                    "reason": self._capture_bridge_error,
                }
            )
            return False
        self._record_control_audit({"event": "live_capture_bridge_stopped"})
        return True

    def _active_wing_transport(self) -> Any:
        """Return the physical WING client owned by the migration bridge."""
        if self._engine is None or self._request is None:
            raise RuntimeError("Live soundcheck is not running")

        mixer_type = str(self._request.mixer_type or "").strip().lower().replace("-", "_")
        if mixer_type not in {"wing", "wing_rack", "behringer_wing"}:
            raise NotImplementedError(
                f"Authoritative live control adapter is not migrated for mixer type {self._request.mixer_type!r}"
            )

        client = getattr(self._engine, "_real_mixer_client", None)
        if client is None:
            client = getattr(self._engine, "mixer_client", None)
        if client is None:
            raise RuntimeError("Physical WING transport is not ready")
        if not hasattr(client, "send") or not hasattr(client, "subscribe"):
            raise TypeError("Active WING transport does not expose send/subscribe")
        return client

    def _active_wing_adapter(self) -> WingWriteAdapter:
        """Return one shared adapter for read-only evidence and control writes."""
        client = self._active_wing_transport()
        if self._control_adapter is None or self._control_transport is not client:
            if (
                self._control_transport is not None
                and self._control_transport is not client
                and self._iteration is not None
                and self._iteration.has_active_hypothesis
            ):
                raise RuntimeError("Physical WING transport changed during an active live iteration")
            self._control_transport = client
            self._control_adapter = WingWriteAdapter(client)
            self._control_plane = None
        return self._control_adapter

    def _active_control_plane(self) -> LiveControlPlane:
        adapter = self._active_wing_adapter()
        if self._control_plane is None:
            self._control_plane = LiveControlPlane(
                adapter,
                audit_sink=self._record_control_audit,
            )
        return self._control_plane

    def _active_iteration_coordinator(self) -> IterationCoordinator:
        if self._request is None:
            raise RuntimeError("Live soundcheck is not running")
        if self._iteration is None:
            self._iteration = IterationCoordinator(
                self,
                verification_window_s=self._iteration_verification_window_s,
                audit_sink=self._record_control_audit,
            )
        return self._iteration

    def _build_patch_startup_coordinator(self) -> MainTapPatchStartupCoordinator:
        if self._request is None:
            raise RuntimeError("Live soundcheck is not running")
        adapter = self._active_wing_adapter()
        if self._patch_startup_factory is not None:
            coordinator = self._patch_startup_factory(
                adapter,
                self._request.mixer_ip,
                self._record_control_audit,
            )
            if not isinstance(coordinator, MainTapPatchStartupCoordinator):
                raise TypeError(
                    "patch_startup_factory must return MainTapPatchStartupCoordinator"
                )
            return coordinator
        return MainTapPatchStartupCoordinator.for_wing(
            adapter,
            self._request.mixer_ip,
            audit_sink=self._record_control_audit,
        )

    def verify_main_tap_patch(
        self,
        contract: MainTapPatchContract,
        tap_evidence: MainFeatureEvidence,
    ) -> PatchVerifyStartupResult:
        """Run the read-only Main PATCH_VERIFY gate for the active WING session."""
        if self._engine is None or self._request is None:
            raise RuntimeError("Live soundcheck is not running")
        if self._soundcheck_state not in (
            SoundcheckState.DISCOVER,
            SoundcheckState.PATCH_VERIFY,
        ):
            current = self._soundcheck_state.value if self._soundcheck_state else "idle"
            raise RuntimeError(
                "Main PATCH_VERIFY may run only from DISCOVER/PATCH_VERIFY; "
                f"current state is {current}"
            )

        if self._soundcheck_state is SoundcheckState.DISCOVER:
            self._soundcheck_state = SoundcheckState.PATCH_VERIFY
            self._record_control_audit(
                {
                    "event": "live_state_transition",
                    "state_before": SoundcheckState.DISCOVER.value,
                    "state_after": SoundcheckState.PATCH_VERIFY.value,
                    "reason": "startup_patch_verify",
                }
            )

        coordinator = self._build_patch_startup_coordinator()
        result = coordinator.run(
            SoundcheckState.PATCH_VERIFY,
            contract,
            tap_evidence,
        )
        self._last_patch_verify = result
        self._soundcheck_state = result.state
        return result

    def select_eq_locator(
        self,
        channel: int,
        evidence: EqTargetEvidence,
    ) -> EqBandLocator | None:
        """Resolve realtime spectral evidence against fresh physical WING bands."""
        selector = RealtimeEqLocatorSelector(self._active_wing_adapter())
        locator = selector.select(channel, evidence)
        self._record_control_audit(
            {
                "event": "live_eq_locator_selected" if locator is not None else "live_eq_locator_unresolved",
                "channel": channel,
                "evidence": asdict(evidence),
                "locator": asdict(locator) if locator is not None else None,
            }
        )
        return locator

    def propose_hypothesis(
        self,
        features: MixFeatures,
        roles: dict[int, str],
        *,
        masking: dict[tuple[int, int], float] | None = None,
        eq_evidence: dict[tuple[int, str], EqTargetEvidence] | None = None,
    ) -> LiveHypothesis | None:
        """Compose realtime evidence, physical band selection and the Director."""
        locators: dict[tuple[int, str], EqBandLocator] = {}
        for key, evidence in (eq_evidence or {}).items():
            channel, _intent = key
            locator = self.select_eq_locator(channel, evidence)
            if locator is not None:
                locators[key] = locator
        return propose_one(
            features,
            roles,
            masking=masking,
            eq_locators=locators,
        )

    @staticmethod
    def _snapshot_metrics(
        features: MixFeatures,
        hypothesis: LiveHypothesis,
        verification_metrics: dict[str, Any] | None,
        *,
        operator_took_control: bool,
    ) -> dict[str, Any]:
        """Build Critic evidence from the actual sequential feature snapshot."""
        metrics = dict(verification_metrics or {})
        metrics["main_peak_dbfs"] = float(features.main_peak_dbfs)
        metrics["operator_touch"] = bool(
            operator_took_control or metrics.get("operator_touch", False)
        )

        if hypothesis.verify_metric == "harshness":
            target = hypothesis.target
            if target.startswith("ch:"):
                try:
                    channel = int(target.split(":", 1)[1])
                except ValueError:
                    channel = -1
                for item in features.channels:
                    if item.channel == channel and item.harshness is not None:
                        metrics["harshness"] = float(item.harshness)
                        break
        return metrics

    @staticmethod
    def _state_for_iteration(result: IterationResult) -> SoundcheckState:
        if result.phase in (IterationPhase.VERIFY_PENDING, IterationPhase.VERIFY_WAIT):
            return SoundcheckState.VERIFY
        if result.phase is IterationPhase.HOLD:
            return SoundcheckState.HOLD
        if result.phase is IterationPhase.APPLY_BLOCKED:
            return SoundcheckState.PROPOSE
        return SoundcheckState.LISTEN

    def process_feature_snapshot(
        self,
        features: MixFeatures,
        roles: dict[int, str],
        *,
        masking: dict[tuple[int, int], float] | None = None,
        eq_evidence: dict[tuple[int, str], EqTargetEvidence] | None = None,
        verification_metrics: dict[str, Any] | None = None,
        operator_took_control: bool = False,
        manual_freeze: bool = False,
    ) -> LiveSnapshotResult:
        """Advance the canonical one-hypothesis live loop by one feature frame."""
        if self._request is None or self._engine is None:
            raise RuntimeError("Live soundcheck is not running")
        startup_state = self._soundcheck_state
        if startup_state not in (
            SoundcheckState.LISTEN,
            SoundcheckState.PROPOSE,
            SoundcheckState.APPLY,
            SoundcheckState.VERIFY,
        ):
            if startup_state is None:
                raise RuntimeError("Live soundcheck startup state is not initialized")
            return LiveSnapshotResult(
                state=startup_state,
                reason=f"startup_state_blocked:{startup_state.value}",
            )

        coordinator = self._active_iteration_coordinator()
        if coordinator.hold_reason is not None:
            self._soundcheck_state = SoundcheckState.HOLD
            return LiveSnapshotResult(
                state=SoundcheckState.HOLD,
                reason=coordinator.hold_reason,
            )

        if coordinator.has_active_hypothesis:
            hypothesis = self._iteration_hypothesis
            if hypothesis is None:
                raise RuntimeError("Live iteration hypothesis ownership invariant failed")
            metrics = self._snapshot_metrics(
                features,
                hypothesis,
                verification_metrics,
                operator_took_control=operator_took_control,
            )
            result = coordinator.verify(metrics, manual_freeze=manual_freeze)
            state = self._state_for_iteration(result)
            self._soundcheck_state = state
            if result.phase not in (IterationPhase.VERIFY_PENDING, IterationPhase.VERIFY_WAIT):
                self._iteration_hypothesis = None
            return LiveSnapshotResult(
                state=state,
                hypothesis=hypothesis,
                iteration=result,
                reason=result.reason,
            )

        hypothesis = self.propose_hypothesis(
            features,
            roles,
            masking=masking,
            eq_evidence=eq_evidence,
        )
        if hypothesis is None:
            self._soundcheck_state = SoundcheckState.LISTEN
            return LiveSnapshotResult(state=SoundcheckState.LISTEN, reason="no_hypothesis")

        metrics = self._snapshot_metrics(
            features,
            hypothesis,
            verification_metrics,
            operator_took_control=operator_took_control,
        )
        result = coordinator.start(
            hypothesis,
            metrics,
            manual_freeze=manual_freeze,
        )
        if result.phase is IterationPhase.VERIFY_PENDING:
            self._iteration_hypothesis = hypothesis
        else:
            self._iteration_hypothesis = None
        state = self._state_for_iteration(result)
        self._soundcheck_state = state
        return LiveSnapshotResult(
            state=state,
            hypothesis=hypothesis,
            iteration=result,
            reason=result.reason,
        )

    def execute_action(
        self,
        action: ProposedAction,
        *,
        manual_freeze: bool = False,
    ) -> WriteExecution:
        """Execute one new-runtime action through the physical WING boundary."""
        if self._request is None:
            raise RuntimeError("Live soundcheck is not running")
        return self._active_control_plane().execute(
            action,
            self._request.mode,
            manual_freeze=manual_freeze,
        )

    def rollback_action(
        self,
        verified: VerifiedAction,
        *,
        manual_freeze: bool = False,
    ) -> RollbackExecution:
        """Restore an action through the same physical WING/readback boundary."""
        if self._request is None:
            raise RuntimeError("Live soundcheck is not running")
        return self._active_control_plane().rollback(
            verified,
            self._request.mode,
            manual_freeze=manual_freeze,
        )

    def create_engine(
        self,
        request: LiveStartRequest,
        *,
        on_state_change=None,
        on_channel_update=None,
        on_observation=None,
    ) -> Any:
        """Construct an engine without starting it."""
        observe_only, auto_apply = self._legacy_flags(request.mode)
        engine = self._factory()(
            mixer_type=request.mixer_type,
            mixer_ip=request.mixer_ip,
            mixer_port=request.mixer_port,
            audio_device_name=request.audio_device_name,
            num_channels=request.num_channels,
            selected_channels=request.selected_channels,
            observe_only=observe_only,
            auto_apply=auto_apply,
            on_state_change=on_state_change,
            on_channel_update=on_channel_update,
            on_observation=on_observation,
        )
        setattr(engine, "live_runtime_mode", request.mode.value)
        return engine

    def is_active(self) -> bool:
        engine = self._engine
        if engine is None:
            return False
        state = getattr(getattr(engine, "state", None), "value", None)
        return state not in ("stopped", "error")

    def start(
        self,
        request: LiveStartRequest,
        *,
        on_state_change=None,
        on_channel_update=None,
        on_observation=None,
    ) -> Any:
        """Start one live session with single-owner audio when bridge is configured."""
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
        """Stop bridge, legacy non-owner, seam, then the physical capture."""
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
            release_errors = self._release_owned_audio_capture()
            if first_error is None and release_errors:
                first_error = release_errors[0]
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
        """Return a transport-safe live status without exposing engine ownership."""
        engine = self._engine
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
            }

        raw = engine.get_status() if hasattr(engine, "get_status") else {}
        status = dict(raw or {})
        mode = self.active_mode
        if mode is not None:
            status["mode"] = mode.value

        transport = getattr(engine, "_real_mixer_client", None)
        if transport is None:
            transport = getattr(engine, "mixer_client", None)
        mixer_type = str(self._request.mixer_type if self._request else "").strip().lower().replace("-", "_")
        status["control_plane_ready"] = (
            mixer_type in {"wing", "wing_rack", "behringer_wing"}
            and transport is not None
            and hasattr(transport, "send")
            and hasattr(transport, "subscribe")
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
        return status
