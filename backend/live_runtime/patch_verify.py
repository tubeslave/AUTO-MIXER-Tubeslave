"""Fail-closed PATCH_VERIFY for the authoritative post-console Main USB tap.

The verifier proves only routing identity in this module: each reserved capture
slot must be fed by the explicitly declared WING USB output route and every
route must fresh-read as a MAIN source. It never mutates routing. Physical
level-coherence against an independent Main meter remains a separate HIL gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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
    declared MAIN routes. It does not yet prove signal-level coherence, so its
    result must not be treated as complete HIL proof by itself.
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


def main_tap_patch_contract(
    tap: PostConsoleMainTap,
    routes: Iterable[MainTapRouteExpectation],
) -> MainTapPatchContract:
    """Small constructor for config/FSM layers without hidden defaults."""
    return MainTapPatchContract(tap=tap, routes=tuple(routes))
