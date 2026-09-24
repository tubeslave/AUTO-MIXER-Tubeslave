"""Fail-closed PATCH_VERIFY for the authoritative post-console Main USB tap.

PATCH_VERIFY has two independent proof layers:
1. fresh WING routing readback proves that every reserved USB capture slot is
   fed by the explicitly declared MAIN source;
2. level-coherence compares the returned post-console tap with an independent
   physical Main meter sample in the same short time window.

This module is deliberately read-only.  It never repairs routing, writes mixer
state, or invents a Main signal by summing input stems.  A concrete physical
meter provider is intentionally kept outside this module until its WING meter
endpoint/transport has been validated on hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from .feature_stream import ACTIVITY_THRESHOLD_DBFS, MainFeatureEvidence
from .main_evidence import PostConsoleMainTap
from .wing_adapter import OutputRouteReadback, WingWriteAdapter


@dataclass(frozen=True)
class MainTapRouteExpectation:
    """Expected WING source for one reserved USB capture slot."""

    usb_slot: int
    source_channel: int
    source_group: str = "MAIN"

    def __post_init__(self) -> None:
        if isinstance(self.usb_slot, bool) or not isinstance(self.usb_slot, int):
            raise TypeError("usb_slot must be an integer")
        if not 1 <= self.usb_slot <= 48:
            raise ValueError(f"usb_slot out of range: {self.usb_slot}")
        if not isinstance(self.source_group, str):
            raise TypeError("source_group must be a string")
        normalized_group = self.source_group.strip().upper()
        if normalized_group != "MAIN":
            raise ValueError(
                "post-console Main tap expectation must use WING source_group MAIN"
            )
        object.__setattr__(self, "source_group", normalized_group)
        if isinstance(self.source_channel, bool) or not isinstance(self.source_channel, int):
            raise TypeError("source_channel must be an integer")
        if not 1 <= self.source_channel <= 4:
            raise ValueError(f"WING MAIN source channel out of range: {self.source_channel}")


@dataclass(frozen=True)
class MainTapPatchContract:
    """Exact routing proof required for a configured post-console Main tap."""

    tap: PostConsoleMainTap
    routes: tuple[MainTapRouteExpectation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.tap, PostConsoleMainTap):
            raise TypeError("tap must be PostConsoleMainTap")
        routes = tuple(self.routes)
        if not routes:
            raise ValueError("Main tap PATCH_VERIFY requires at least one route")
        expected_slots = set(self.tap.channels)
        route_slots = [route.usb_slot for route in routes]
        if len(route_slots) != len(set(route_slots)):
            raise ValueError("Main tap PATCH_VERIFY routes contain duplicate USB slots")
        if set(route_slots) != expected_slots:
            raise ValueError(
                "Main tap PATCH_VERIFY routes must exactly cover reserved tap slots: "
                f"tap={sorted(expected_slots)}, routes={sorted(route_slots)}"
            )
        object.__setattr__(self, "routes", routes)


@dataclass(frozen=True)
class MainTapPatchVerification:
    verified: bool
    reason: str
    observed: tuple[OutputRouteReadback, ...]


class MainTapPatchVerifier:
    """Read-only WING routing verifier for PATCH_VERIFY.

    A successful result means the configured USB slots fresh-read as the exact
    declared MAIN routes.  It is one half of the complete Main tap gate; level
    coherence against independent physical evidence is still required.
    """

    def __init__(self, adapter: WingWriteAdapter):
        self._adapter = adapter

    def verify(self, contract: MainTapPatchContract) -> MainTapPatchVerification:
        observed: list[OutputRouteReadback] = []
        for expected in contract.routes:
            try:
                route = self._adapter.read_output_route("USB", expected.usb_slot)
            except (RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                return MainTapPatchVerification(
                    verified=False,
                    reason=(
                        "readback_failed: "
                        f"USB {expected.usb_slot}: {type(exc).__name__}: {exc}"
                    ),
                    observed=tuple(observed),
                )

            observed.append(route)
            if (
                route.source_group != expected.source_group
                or route.source_channel != expected.source_channel
            ):
                return MainTapPatchVerification(
                    verified=False,
                    reason=(
                        "route_mismatch: "
                        f"USB {expected.usb_slot} expected "
                        f"{expected.source_group} {expected.source_channel}, read "
                        f"{route.source_group} {route.source_channel}"
                    ),
                    observed=tuple(observed),
                )

        return MainTapPatchVerification(
            verified=True,
            reason="verified",
            observed=tuple(observed),
        )


@dataclass(frozen=True)
class PhysicalMainMeterEvidence:
    """One independent physical Main-meter observation in dBFS.

    The evidence producer is deliberately external to PATCH_VERIFY.  This keeps
    the proof contract usable while preventing an unverified WING meter address
    from becoming production authority merely because it was guessed in code.
    """

    peak_dbfs: float
    timestamp_s: float
    source: str
    rms_dbfs: float | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("peak_dbfs", self.peak_dbfs),
            ("timestamp_s", self.timestamp_s),
        ):
            if isinstance(value, bool):
                raise TypeError(f"{label} must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{label} must be finite")
            object.__setattr__(self, label, number)

        if self.rms_dbfs is not None:
            if isinstance(self.rms_dbfs, bool):
                raise TypeError("rms_dbfs must be numeric when provided")
            rms = float(self.rms_dbfs)
            if not math.isfinite(rms):
                raise ValueError("rms_dbfs must be finite when provided")
            object.__setattr__(self, "rms_dbfs", rms)

        if not isinstance(self.source, str):
            raise TypeError("source must be a string")
        source = self.source.strip()
        if not source:
            raise ValueError("source must name the independent Main meter")
        object.__setattr__(self, "source", source)


@dataclass(frozen=True)
class MainTapLevelCoherencePolicy:
    """Fail-closed tolerances for comparing the USB Main tap to physical meter evidence.

    Defaults are conservative software gate values, not claimed WING meter
    calibration constants.  HIL may tighten them once physical meter ballistics
    and latency are measured.
    """

    max_timestamp_skew_s: float = 0.100
    max_peak_delta_db: float = 1.5
    max_rms_delta_db: float = 1.5
    min_active_peak_dbfs: float = ACTIVITY_THRESHOLD_DBFS
    require_rms: bool = False

    def __post_init__(self) -> None:
        for label in (
            "max_timestamp_skew_s",
            "max_peak_delta_db",
            "max_rms_delta_db",
            "min_active_peak_dbfs",
        ):
            value = getattr(self, label)
            if isinstance(value, bool):
                raise TypeError(f"{label} must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{label} must be finite")
            object.__setattr__(self, label, number)
        if self.max_timestamp_skew_s < 0.0:
            raise ValueError("max_timestamp_skew_s must be >= 0")
        if self.max_peak_delta_db < 0.0:
            raise ValueError("max_peak_delta_db must be >= 0")
        if self.max_rms_delta_db < 0.0:
            raise ValueError("max_rms_delta_db must be >= 0")
        if not isinstance(self.require_rms, bool):
            raise TypeError("require_rms must be bool")


@dataclass(frozen=True)
class MainTapLevelVerification:
    verified: bool
    reason: str
    physical_source: str
    timestamp_skew_s: float | None = None
    peak_delta_db: float | None = None
    rms_delta_db: float | None = None


class MainTapLevelCoherenceVerifier:
    """Compare post-console USB evidence with an independent physical Main meter."""

    def __init__(self, policy: MainTapLevelCoherencePolicy | None = None) -> None:
        self.policy = policy or MainTapLevelCoherencePolicy()

    def verify(
        self,
        tap_evidence: MainFeatureEvidence,
        physical_evidence: PhysicalMainMeterEvidence,
    ) -> MainTapLevelVerification:
        if not isinstance(tap_evidence, MainFeatureEvidence):
            raise TypeError("tap_evidence must be MainFeatureEvidence")
        if not isinstance(physical_evidence, PhysicalMainMeterEvidence):
            raise TypeError("physical_evidence must be PhysicalMainMeterEvidence")

        tap_values = {
            "rms_dbfs": tap_evidence.rms_dbfs,
            "peak_dbfs": tap_evidence.peak_dbfs,
            "crest_db": tap_evidence.crest_db,
            "timestamp_s": tap_evidence.timestamp_s,
        }
        for label, value in tap_values.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                return MainTapLevelVerification(
                    verified=False,
                    reason=f"invalid_tap_evidence: {label} is not numeric",
                    physical_source=physical_evidence.source,
                )
            if not math.isfinite(number):
                return MainTapLevelVerification(
                    verified=False,
                    reason=f"invalid_tap_evidence: {label} is not finite",
                    physical_source=physical_evidence.source,
                )

        skew = abs(float(tap_evidence.timestamp_s) - physical_evidence.timestamp_s)
        if skew > self.policy.max_timestamp_skew_s:
            return MainTapLevelVerification(
                verified=False,
                reason=(
                    f"timestamp_skew: {skew:.6f}s exceeds "
                    f"{self.policy.max_timestamp_skew_s:.6f}s"
                ),
                physical_source=physical_evidence.source,
                timestamp_skew_s=skew,
            )

        if (
            float(tap_evidence.peak_dbfs) < self.policy.min_active_peak_dbfs
            or physical_evidence.peak_dbfs < self.policy.min_active_peak_dbfs
        ):
            return MainTapLevelVerification(
                verified=False,
                reason=(
                    "signal_too_low: "
                    f"tap_peak={float(tap_evidence.peak_dbfs):.2f}dBFS, "
                    f"physical_peak={physical_evidence.peak_dbfs:.2f}dBFS, "
                    f"minimum={self.policy.min_active_peak_dbfs:.2f}dBFS"
                ),
                physical_source=physical_evidence.source,
                timestamp_skew_s=skew,
            )

        peak_delta = abs(float(tap_evidence.peak_dbfs) - physical_evidence.peak_dbfs)
        if peak_delta > self.policy.max_peak_delta_db:
            return MainTapLevelVerification(
                verified=False,
                reason=(
                    f"peak_mismatch: {peak_delta:.2f}dB exceeds "
                    f"{self.policy.max_peak_delta_db:.2f}dB"
                ),
                physical_source=physical_evidence.source,
                timestamp_skew_s=skew,
                peak_delta_db=peak_delta,
            )

        if physical_evidence.rms_dbfs is None:
            if self.policy.require_rms:
                return MainTapLevelVerification(
                    verified=False,
                    reason="rms_missing: policy requires independent RMS evidence",
                    physical_source=physical_evidence.source,
                    timestamp_skew_s=skew,
                    peak_delta_db=peak_delta,
                )
            rms_delta = None
        else:
            rms_delta = abs(float(tap_evidence.rms_dbfs) - physical_evidence.rms_dbfs)
            if rms_delta > self.policy.max_rms_delta_db:
                return MainTapLevelVerification(
                    verified=False,
                    reason=(
                        f"rms_mismatch: {rms_delta:.2f}dB exceeds "
                        f"{self.policy.max_rms_delta_db:.2f}dB"
                    ),
                    physical_source=physical_evidence.source,
                    timestamp_skew_s=skew,
                    peak_delta_db=peak_delta,
                    rms_delta_db=rms_delta,
                )

        return MainTapLevelVerification(
            verified=True,
            reason="verified",
            physical_source=physical_evidence.source,
            timestamp_skew_s=skew,
            peak_delta_db=peak_delta,
            rms_delta_db=rms_delta,
        )


@dataclass(frozen=True)
class MainTapPatchGateVerification:
    verified: bool
    reason: str
    route: MainTapPatchVerification
    level: MainTapLevelVerification | None


class MainTapPatchGateVerifier:
    """Complete read-only Main tap gate: exact route proof plus level coherence."""

    def __init__(
        self,
        route_verifier: MainTapPatchVerifier,
        level_verifier: MainTapLevelCoherenceVerifier | None = None,
    ) -> None:
        self._route_verifier = route_verifier
        self._level_verifier = level_verifier or MainTapLevelCoherenceVerifier()

    def verify(
        self,
        contract: MainTapPatchContract,
        tap_evidence: MainFeatureEvidence,
        physical_evidence: PhysicalMainMeterEvidence,
    ) -> MainTapPatchGateVerification:
        route = self._route_verifier.verify(contract)
        if not route.verified:
            return MainTapPatchGateVerification(
                verified=False,
                reason=f"route_failed: {route.reason}",
                route=route,
                level=None,
            )

        level = self._level_verifier.verify(tap_evidence, physical_evidence)
        if not level.verified:
            return MainTapPatchGateVerification(
                verified=False,
                reason=f"level_failed: {level.reason}",
                route=route,
                level=level,
            )

        return MainTapPatchGateVerification(
            verified=True,
            reason="verified",
            route=route,
            level=level,
        )


def main_tap_patch_contract(
    tap: PostConsoleMainTap,
    routes: Iterable[MainTapRouteExpectation],
) -> MainTapPatchContract:
    """Small constructor for config/FSM layers without hidden defaults."""
    return MainTapPatchContract(tap=tap, routes=tuple(routes))
