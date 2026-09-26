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

from .contracts import EqBandLocator, LiveMode, MixFeatures, ProposedAction, VerifiedAction
from .control_plane import LiveControlPlane, RollbackExecution, WriteExecution
from .decision_engine import LiveHypothesis, propose_one
from .eq_locator import EqTargetEvidence, RealtimeEqLocatorSelector
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


class LiveSoundcheckService:
    """Authoritative lifecycle and live-control composition seam.

    ``AutoSoundcheckEngine`` remains a MIGRATE dependency because it still owns
    useful mixer discovery, audio capture, readback and action logging. New
    callers do not own that engine directly: they start/stop/query this service.

    WING mutations owned by the new architecture are executed through
    :class:`LiveControlPlane`. The service deliberately obtains the physical
    WING transport from the compatibility engine but never delegates decision
    policy to it.
    """

    legacy_bridge = True

    def __init__(
        self,
        engine_factory: Callable[..., Any] | None = None,
        *,
        audit_sink: Callable[[dict[str, Any]], None] | None = None,
    ):
        self._engine_factory = engine_factory
        self._engine: Any | None = None
        self._request: LiveStartRequest | None = None
        self._control_plane: LiveControlPlane | None = None
        self._control_adapter: WingWriteAdapter | None = None
        self._control_transport: Any | None = None
        self._control_audit: list[dict[str, Any]] = []
        self._external_audit_sink = audit_sink

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
        """Map new runtime modes onto the temporary legacy engine flags.

        OBSERVE/PROPOSE must not write. BENCH_TEST and production write modes
        may write; detailed authorization belongs to live_runtime policy, not UI
        handlers. The bridge never infers a mode from mixer presence.
        """
        if mode in (LiveMode.OBSERVE, LiveMode.PROPOSE, LiveMode.FREEZE):
            return True, False
        return False, True

    def _factory(self) -> Callable[..., Any]:
        factory = self._engine_factory
        if factory is None:
            # Deliberately lazy and isolated: this is the sole compatibility
            # import to delete when the new live runtime fully owns the loop.
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

    def _active_wing_transport(self) -> Any:
        """Return the physical WING client owned by the migration bridge.

        In observe mode the legacy engine may expose an ``ObservationMixerClient``
        as ``mixer_client``. ``_real_mixer_client`` is therefore preferred. A
        local observation wrapper must never be used to prove physical readback.
        """
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
        """Return one shared adapter for read-only evidence and control writes.

        EQ band selection and mutation must observe the same physical transport.
        Reusing one adapter also reuses its callback subscriptions while every
        query still requires a fresh inbound WING response.
        """
        client = self._active_wing_transport()
        if self._control_adapter is None or self._control_transport is not client:
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

    def select_eq_locator(
        self,
        channel: int,
        evidence: EqTargetEvidence,
    ) -> EqBandLocator | None:
        """Resolve realtime spectral evidence against fresh physical WING bands.

        This is read-only. A low-confidence or unmatched target returns ``None``
        and therefore cannot become a hardware-actionable EQ proposal.
        """
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
        """Compose realtime evidence, physical band selection and the Director.

        ``eq_evidence`` keys are ``(channel, intent)`` where intent matches the
        Director's evidence names such as ``harshness`` or ``vocal_masking``.
        The service resolves only supplied evidence; it never invents a PEQ band
        from an instrument preset or legacy AutoEQ policy.
        """
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

    def execute_action(
        self,
        action: ProposedAction,
        *,
        manual_freeze: bool = False,
    ) -> WriteExecution:
        """Execute one new-runtime action through the physical WING boundary.

        Mode selection always comes from the explicit active ``LiveStartRequest``.
        BENCH_TEST therefore exercises the same write/readback path as production
        while bypassing production action allowlists. OBSERVE/PROPOSE/FREEZE are
        still read-before + audit only and never mutate the console.
        """
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
        """Restore an action through the same physical WING/readback boundary.

        The rollback value must have been captured by a prior reversible
        ``execute_action`` result. The current active mode remains authoritative:
        OBSERVE/PROPOSE/FREEZE cannot mutate the console, while write-capable
        modes may restore the captured pre-write state.
        """
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
        """Construct an engine without starting it.

        Kept for focused bridge tests and staged migration. Runtime callers
        should use :meth:`start` so lifecycle ownership stays in live_runtime.
        """
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
            raise
        return engine

    def stop(self) -> bool:
        """Stop the active engine and release lifecycle/control ownership."""
        engine = self._engine
        self._engine = None
        self._request = None
        self._reset_control_plane()
        if engine is None:
            return False
        engine.stop()
        return True

    def get_status(self) -> dict[str, Any]:
        """Return a transport-safe live status without exposing engine ownership."""
        engine = self._engine
        if engine is None:
            return {
                "state": "idle",
                "mixer_connected": False,
                "audio_running": False,
                "control_plane_ready": False,
                "control_audit_count": len(self._control_audit),
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
        return status
