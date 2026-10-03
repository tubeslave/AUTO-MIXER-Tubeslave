"""Realtime selection of an existing WING PEQ band from explicit evidence.

The new live runtime must never translate a vague musical statement directly
into an arbitrary hardware EQ slot.  This module keeps band selection separate
from the Director: realtime analysis supplies an explicit spectral target, the
selector reads the *current* physical WING band frequency/Q values through a
read-only adapter, and returns an :class:`EqBandLocator` only when an existing
band is already suitable for a gain-only correction.

This first slice deliberately does not move EQ frequency or Q.  If no current
band is close enough to the evidence target, selection fails closed and the
Director remains non-actionable for that EQ idea.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, Sequence

from .contracts import EqBandLocator


class EqBandStateReader(Protocol):
    """Read-only source of fresh physical EQ band fingerprints."""

    def read_eq_locators(self, channel: int) -> list[EqBandLocator]: ...


@dataclass(frozen=True)
class EqTargetEvidence:
    """Spectral evidence required before selecting a physical PEQ band.

    ``center_frequency_hz`` comes from realtime analysis, not from a hard-coded
    instrument preset.  Optional Q constraints describe which already-existing
    bands are acceptable for a gain-only move; they do not authorize changing
    the console band's frequency or Q.
    """

    center_frequency_hz: float
    confidence: float
    max_octave_distance: float = 0.5
    preferred_q: float | None = None
    min_q: float | None = None
    max_q: float | None = None
    source: str = "realtime_features"

    def __post_init__(self) -> None:
        center = float(self.center_frequency_hz)
        confidence = float(self.confidence)
        distance = float(self.max_octave_distance)
        if not math.isfinite(center) or center <= 0:
            raise ValueError("center_frequency_hz must be finite and > 0")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be within 0..1")
        if not math.isfinite(distance) or distance < 0:
            raise ValueError("max_octave_distance must be finite and >= 0")
        if self.preferred_q is not None:
            preferred_q = float(self.preferred_q)
            if not math.isfinite(preferred_q) or preferred_q <= 0:
                raise ValueError("preferred_q must be finite and > 0")
        if self.min_q is not None:
            min_q = float(self.min_q)
            if not math.isfinite(min_q) or min_q <= 0:
                raise ValueError("min_q must be finite and > 0")
        if self.max_q is not None:
            max_q = float(self.max_q)
            if not math.isfinite(max_q) or max_q <= 0:
                raise ValueError("max_q must be finite and > 0")
        if self.min_q is not None and self.max_q is not None:
            if float(self.min_q) > float(self.max_q):
                raise ValueError("min_q cannot exceed max_q")
        if not str(self.source).strip():
            raise ValueError("source must be non-empty")


def _valid_band(locator: EqBandLocator) -> bool:
    try:
        frequency = float(locator.frequency_hz)
        q = float(locator.q)
    except (TypeError, ValueError):
        return False
    return (
        1 <= int(locator.band) <= 4
        and math.isfinite(frequency)
        and frequency > 0
        and math.isfinite(q)
        and q > 0
    )


def select_existing_eq_band(
    bands: Sequence[EqBandLocator],
    evidence: EqTargetEvidence,
    *,
    min_confidence: float = 0.7,
) -> EqBandLocator | None:
    """Choose the best *existing* band for a gain-only live EQ proposal.

    Frequency distance is measured in octaves so the criterion behaves
    consistently across the spectrum.  Preferred-Q is only a tie/quality term;
    explicit min/max Q constraints are hard eligibility gates.  The function
    never invents a band, frequency or Q and never mutates mixer state.
    """

    threshold = float(min_confidence)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("min_confidence must be within 0..1")
    if float(evidence.confidence) < threshold:
        return None

    center = float(evidence.center_frequency_hz)
    maximum_distance = float(evidence.max_octave_distance)
    candidates: list[tuple[float, float, int, EqBandLocator]] = []

    for locator in bands:
        if not _valid_band(locator):
            continue
        frequency = float(locator.frequency_hz)
        q = float(locator.q)
        if evidence.min_q is not None and q < float(evidence.min_q):
            continue
        if evidence.max_q is not None and q > float(evidence.max_q):
            continue

        octave_distance = abs(math.log2(frequency / center))
        if octave_distance > maximum_distance:
            continue

        q_distance = 0.0
        if evidence.preferred_q is not None:
            q_distance = abs(math.log2(q / float(evidence.preferred_q)))

        # Frequency is the primary evidence. Q is a deliberately smaller
        # secondary preference because this slice does not move Q itself.
        score = octave_distance + 0.2 * q_distance
        candidates.append((score, octave_distance, int(locator.band), locator))

    if not candidates:
        return None
    candidates.sort(key=lambda row: (row[0], row[1], row[2]))
    return candidates[0][3]


class RealtimeEqLocatorSelector:
    """Compose fresh console state with spectral evidence."""

    def __init__(self, reader: EqBandStateReader, *, min_confidence: float = 0.7):
        threshold = float(min_confidence)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("min_confidence must be within 0..1")
        self._reader = reader
        self._min_confidence = threshold

    def select(self, channel: int, evidence: EqTargetEvidence) -> EqBandLocator | None:
        """Return a fresh physical band locator, or ``None`` when unsafe/unclear."""
        if float(evidence.confidence) < self._min_confidence:
            return None
        bands = self._reader.read_eq_locators(channel)
        return select_existing_eq_band(
            bands,
            evidence,
            min_confidence=self._min_confidence,
        )
