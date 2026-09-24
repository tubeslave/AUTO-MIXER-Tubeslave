"""Read-only startup orchestration for the canonical live PATCH_VERIFY state.

The soundcheck FSM must not advance from PATCH_VERIFY merely because the USB
capture slots are configured.  This coordinator proves the Main return in a
strict causal order:

1. fresh physical WING route readback for every reserved USB slot;
2. one independent native WING Main-meter observation;
3. level-coherence between that physical meter and the post-console USB tap.

A complete proof advances PATCH_VERIFY -> LISTEN.  Any expected transport,
protocol or evidence failure advances PATCH_VERIFY -> HOLD.  This module never
repairs routing and never writes mixer state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol

from .contracts import SoundcheckState
from .feature_stream import MainFeatureEvidence
from .patch_verify import (
    MainTapLevelCoherencePolicy,
    MainTapLevelCoherenceVerifier,
    MainTapPatchContract,
    MainTapPatchGateVerification,
    MainTapPatchVerification,
    MainTapPatchVerifier,
    PhysicalMainMeterEvidence,
)
from .wing_adapter import WingWriteAdapter
from .wing_main_meter import WingNativeMainMeterProvider


class PhysicalMainMeterReader(Protocol):
    """Minimal read-only meter boundary required by startup PATCH_VERIFY."""

    def read(self) -> PhysicalMainMeterEvidence: ...


@dataclass(frozen=True)
class PatchVerifyStartupResult:
    """One terminal startup PATCH_VERIFY transition."""

    state: SoundcheckState
    verification: MainTapPatchGateVerification
    physical_evidence: PhysicalMainMeterEvidence | None = None

    @property
    def verified(self) -> bool:
        return self.verification.verified

    @property
    def reason(self) -> str:
        return self.verification.reason


class MainTapPatchStartupCoordinator:
    """Own the ordered, fail-closed PATCH_VERIFY startup gate.

    The independent physical meter is intentionally read *after* the fresh
    routing proof succeeds.  Apart from avoiding needless native-meter traffic,
    this ordering makes it impossible for a good meter sample to bless a stale
    or incorrectly routed USB return.
    """

    def __init__(
        self,
        route_verifier: MainTapPatchVerifier,
        meter_provider: PhysicalMainMeterReader,
        *,
        level_verifier: MainTapLevelCoherenceVerifier | None = None,
        audit_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._route_verifier = route_verifier
        self._meter_provider = meter_provider
        self._level_verifier = level_verifier or MainTapLevelCoherenceVerifier()
        self._audit_sink = audit_sink

    @classmethod
    def for_wing(
        cls,
        adapter: WingWriteAdapter,
        host: str,
        *,
        main: int = 1,
        meter_timeout_s: float = 0.35,
        level_policy: MainTapLevelCoherencePolicy | None = None,
        audit_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> "MainTapPatchStartupCoordinator":
        """Build the production read-only WING PATCH_VERIFY composition."""

        return cls(
            MainTapPatchVerifier(adapter),
            WingNativeMainMeterProvider(
                host,
                main=main,
                timeout_s=meter_timeout_s,
            ),
            level_verifier=MainTapLevelCoherenceVerifier(level_policy),
            audit_sink=audit_sink,
        )

    @staticmethod
    def _failed_gate(
        reason: str,
        route: MainTapPatchVerification,
    ) -> MainTapPatchGateVerification:
        return MainTapPatchGateVerification(
            verified=False,
            reason=reason,
            route=route,
            level=None,
        )

    def _emit(
        self,
        result: PatchVerifyStartupResult,
    ) -> None:
        if self._audit_sink is None:
            return
        level = result.verification.level
        self._audit_sink(
            {
                "event": "live_patch_verify_complete",
                "state_before": SoundcheckState.PATCH_VERIFY.value,
                "state_after": result.state.value,
                "verified": result.verified,
                "reason": result.reason,
                "physical_source": (
                    result.physical_evidence.source
                    if result.physical_evidence is not None
                    else None
                ),
                "route_observed": [
                    asdict(item) for item in result.verification.route.observed
                ],
                "level": asdict(level) if level is not None else None,
            }
        )

    def run(
        self,
        state: SoundcheckState,
        contract: MainTapPatchContract,
        tap_evidence: MainFeatureEvidence,
    ) -> PatchVerifyStartupResult:
        """Execute exactly one PATCH_VERIFY attempt.

        Wrong-state invocation is a programming error and does not touch either
        WING transport.  Expected proof failures are terminal for this attempt
        and return HOLD rather than raising into the realtime loop.
        """

        if state is not SoundcheckState.PATCH_VERIFY:
            raise RuntimeError(
                "Main tap startup verification may run only in PATCH_VERIFY; "
                f"got {state.value if isinstance(state, SoundcheckState) else state!r}"
            )
        if not isinstance(contract, MainTapPatchContract):
            raise TypeError("contract must be MainTapPatchContract")
        if not isinstance(tap_evidence, MainFeatureEvidence):
            raise TypeError("tap_evidence must be MainFeatureEvidence")

        route = self._route_verifier.verify(contract)
        if not route.verified:
            result = PatchVerifyStartupResult(
                state=SoundcheckState.HOLD,
                verification=self._failed_gate(
                    f"route_failed: {route.reason}",
                    route,
                ),
            )
            self._emit(result)
            return result

        try:
            physical = self._meter_provider.read()
            if not isinstance(physical, PhysicalMainMeterEvidence):
                raise TypeError(
                    "physical Main meter provider returned "
                    f"{type(physical).__name__}, expected PhysicalMainMeterEvidence"
                )
        except (RuntimeError, TimeoutError, OSError, TypeError, ValueError) as exc:
            result = PatchVerifyStartupResult(
                state=SoundcheckState.HOLD,
                verification=self._failed_gate(
                    f"meter_failed: {type(exc).__name__}: {exc}",
                    route,
                ),
            )
            self._emit(result)
            return result

        level = self._level_verifier.verify(tap_evidence, physical)
        if not level.verified:
            result = PatchVerifyStartupResult(
                state=SoundcheckState.HOLD,
                verification=MainTapPatchGateVerification(
                    verified=False,
                    reason=f"level_failed: {level.reason}",
                    route=route,
                    level=level,
                ),
                physical_evidence=physical,
            )
            self._emit(result)
            return result

        result = PatchVerifyStartupResult(
            state=SoundcheckState.LISTEN,
            verification=MainTapPatchGateVerification(
                verified=True,
                reason="verified",
                route=route,
                level=level,
            ),
            physical_evidence=physical,
        )
        self._emit(result)
        return result
