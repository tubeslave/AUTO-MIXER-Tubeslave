"""Sequential stem blending policy inspired by Yeh et al. (2026).

This module does NOT implement the paper's latent flow-matching model. It makes
its most useful orchestration principle available to the existing Automixer:
each new stem is evaluated against the current growing submix, never in
isolation. The policy is deliberately advisory/shadow-safe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


DEFAULT_ROLE_ORDER = (
    "drums",
    "bass",
    "rhythm_guitar",
    "harmonic_support",
    "lead_guitar",
    "lead_vocal",
    "backing_vocal",
    "fx",
)


@dataclass(frozen=True)
class StemContext:
    stem_id: str
    role: str
    integrated_loudness_lufs: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BlendStep:
    index: int
    stem: StemContext
    prior_stem_ids: tuple[str, ...]
    requires_submix_context: bool = True
    auto_apply: bool = False


def _rank(role: str) -> int:
    try:
        return DEFAULT_ROLE_ORDER.index(role)
    except ValueError:
        return DEFAULT_ROLE_ORDER.index("harmonic_support")


def build_blend_plan(stems: Iterable[StemContext]) -> list[BlendStep]:
    """Return a deterministic, inspectable growing-submix plan.

    Unknown roles are placed with harmonic support. No DSP is applied here.
    Every step explicitly carries the IDs of stems already admitted to the
    submix so downstream critics/proposal agents can compare candidate changes
    in context.
    """
    ordered = sorted(stems, key=lambda s: (_rank(s.role), s.stem_id))
    prior: list[str] = []
    plan: list[BlendStep] = []
    for i, stem in enumerate(ordered):
        plan.append(BlendStep(i, stem, tuple(prior)))
        prior.append(stem.stem_id)
    return plan


def acceptance_requirements() -> dict[str, object]:
    """Shared guardrails for each proposed blend operation."""
    return {
        "compare_in_full_submix": True,
        "level_matched_ab": True,
        "protect": [
            "lead_vocal_intelligibility",
            "kick_bass_anchor",
            "section_dynamics",
            "mono_compatibility",
        ],
        "reject_if": [
            "full_mix_preference_drops",
            "later_stems_require_systematic_compensation",
            "masking_outliers_increase",
        ],
        "auto_apply": False,
    }
