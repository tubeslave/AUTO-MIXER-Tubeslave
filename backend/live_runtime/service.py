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
from typing import Any, Callable

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
from .iteration import IterationCoordinator, IterationPhase, IterationResult
from .wing_adapter import WingWriteAdapter


@dataclass(frozen=True)
class LiveStartRequest:
    mixer_type: str
    mixer_ip: str
    mixer_port: int
    audio_device_name: str
    num_channels: int
    selected_channels: list[int]
    mode: LiveMode = LiveMode.OBSERVE


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
    useful mixer discovery, audio capture, readback and action logging. New
    callers do not own that engine directly: they start/stop/query this service.

    WING mutations owned by the new architecture are executed through
    :class:`LiveControlPlane`. One-hypothesis proposal/verification is owned by
    :class:`IterationCoordinator`; the legacy engine never receives decision
    authority from this service.
    """

    legacy_bridge = True

    def __init__(
        self,
        engine_factory: Callable[..., Any] | None = None,
        *,
        audit_sink: Callable[[dict[str, Any]], None] | None = None,
        iteration_verification_window_s: float = 1.0,
    ):
        if iteration_verification_window_s < 0:
            raise ValueError("iteration_verification_window_s must be >= 0")
        self._engine_factory = engine_factory
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
            # Sole compatibility import to delete when live_runtime owns capture.
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
        """Build Critic evidence from the actual sequential feature snapshot.

        Metrics already calculated by the realtime analyzer can be supplied in
        ``verification_metrics``. Values directly represented by ``MixFeatures``
        are always grounded here. If a hypothesis needs a metric that is absent
        from both sources, the Critic fails closed and the verified rollback path
        is used rather than inventing a proxy measurement.
        """
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
        """Advance the canonical one-hypothesis live loop by one feature frame.

        When a hypothesis is already in flight, this snapshot can only verify it;
        no second proposal is evaluated. A terminal KEEP/rollback result returns
        to LISTEN. Operator takeover or rollback failure propagates HOLD. A new
        proposal is considered only on a later snapshot.
        """
        coordinator = self._active_iteration_coordinator()
        if coordinator.hold_reason is not None:
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
        return LiveSnapshotResult(
            state=self._state_for_iteration(result),
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
        """Construct and start exactly one live engine instance."""
        if self.is_active():
            raise RuntimeError("Live soundcheck engine already running")

        self._reset_control_plane()
        self._reset_iteration()
        self._control_audit.clear()
        engine = self.create_engine(
            request,
            on_state_change=on_state_change,
            on_channel_update=on_channel_update,
            on_observation=on_observation,
        )
        self._engine = engine
        self._request = request
        try:
            engine.start_async()
        except Exception:
            self._engine = None
            self._request = None
            self._reset_control_plane()
            self._reset_iteration()
            raise
        return engine

    def stop(self) -> bool:
        """Stop the active engine and release lifecycle/control ownership."""
        engine = self._engine
        self._engine = None
        self._request = None
        self._reset_control_plane()
        self._reset_iteration()
        if engine is None:
            return False
        engine.stop()
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
            else SoundcheckState.LISTEN.value
        )

        if engine is None:
            return {
                "state": "idle",
                "mixer_connected": False,
                "audio_running": False,
                "control_plane_ready": False,
                "control_audit_count": len(self._control_audit),
                "iteration_active": False,
                "iteration_hold_reason": None,
                "soundcheck_state": SoundcheckState.LISTEN.value,
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
        return status
