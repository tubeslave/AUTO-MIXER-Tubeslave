"""KVMS v0.2: deterministic, OFFLINE perceptual-scene research proxy.

This package does not claim to predict listener preference, perceptual masking
thresholds, or neural activity. It contains no mixer/OSC/device interfaces.

Inputs: licensed, time-aligned, dry mono stems. Do NOT treat live microphone
channels with unverified bleed as clean stems. Role annotations are explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Mapping

import numpy as np


ALLOWED_ROLES = {"foreground", "support", "blend", "unknown"}
ALLOWED_ATTRIBUTION = {"TARGET", "BLEED", "TARGET+BLEED", "UNKNOWN"}


@dataclass(frozen=True)
class SceneContext:
    section: str
    target: str
    role: str
    attribution: str = "UNKNOWN"
    attribution_confidence: float = 0.0
    annotation_source: str = "unknown"


@dataclass(frozen=True)
class ScoreResult:
    status: str
    reason: str
    section: str
    target: str
    target_activity_rms_dbfs: float | None = None
    overlap_proxy: float | None = None
    # 0 = no shared target energy; 1 = target energy deeply overlapped by masker.
    # This is NOT a calibrated psychoacoustic masking percentage.

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Proposal:
    status: str
    action: str
    target: str
    masker: str
    section: str
    delta_db: float
    baseline_overlap_proxy: float | None
    candidate_overlap_proxy: float | None
    measured_delta_proxy: float | None
    reason: str
    auto_apply: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _valid_audio(stems: Mapping[str, np.ndarray], sample_rate: int) -> dict[str, np.ndarray]:
    if not (8000 <= sample_rate <= 192000):
        raise ValueError("unsupported sample_rate")
    if not stems:
        raise ValueError("empty stems")
    out = {}
    lengths = set()
    for name, arr in stems.items():
        if not isinstance(name, str) or not name:
            raise ValueError("invalid track name")
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim != 1 or not np.all(np.isfinite(a)):
            raise ValueError("stems must be finite, mono vectors")
        if len(a) < 2048:
            raise ValueError("at least 2048 samples are required per stem")
        out[name] = a
        lengths.add(len(a))
    if len(lengths) != 1:
        raise ValueError("stems must be sample-aligned and have equal lengths")
    return out


def _rms_dbfs(a: np.ndarray) -> float:
    v = float(np.sqrt(np.mean(a * a)))
    return 20 * math.log10(max(v, 1e-12))


def _stft_power(a: np.ndarray, *, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    """Hann windowed time/frequency-bin power, no perceptual filterbank."""
    window = np.hanning(n_fft)
    frames = np.lib.stride_tricks.sliding_window_view(a, n_fft)[::hop]
    return np.abs(np.fft.rfft(frames * window, axis=-1)) ** 2


def _validate_context(context: SceneContext) -> str | None:
    if context.role not in ALLOWED_ROLES:
        return "invalid role"
    if context.attribution not in ALLOWED_ATTRIBUTION:
        return "invalid attribution label"
    if not (0 <= context.attribution_confidence <= 1):
        return "invalid attribution confidence"
    if not context.section or not context.target:
        return "missing section or target"
    return None


def score_scene(
    stems: Mapping[str, np.ndarray], context: SceneContext, sample_rate: int = 48000,
    *, activity_floor_dbfs: float = -60.0, min_confidence: float = 0.90,
) -> ScoreResult:
    """Evaluate a single foreground target against other aligned mono stems.

    Overlap proxy: target-energy-weighted fraction of competing STFT power.
    No claim is made about auditory masking, objective clarity, or preference.
    """
    audio = _valid_audio(stems, sample_rate)
    problem = _validate_context(context)
    if problem:
        return ScoreResult("abstain", problem, context.section, context.target)
    if context.target not in audio:
        return ScoreResult("abstain", "target missing", context.section, context.target)
    if context.role != "foreground":
        return ScoreResult("abstain", "non-foreground role: no separation objective", context.section, context.target)
    if context.attribution != "TARGET" or context.attribution_confidence < min_confidence:
        return ScoreResult("abstain", "unverified target source attribution / bleed", context.section, context.target)
    target = audio[context.target]
    activity = _rms_dbfs(target)
    if activity <= activity_floor_dbfs:
        return ScoreResult("abstain", "target inactive or too quiet", context.section, context.target, activity)
    maskers = [a for name, a in audio.items() if name != context.target]
    if not maskers:
        return ScoreResult("valid", "no competing stems", context.section, context.target, activity, 0.0)
    # Sum masker powers, not complex waveforms: phases differ between sources.
    T = _stft_power(target)
    M = sum((_stft_power(m) for m in maskers), np.zeros_like(T))
    if not np.all(np.isfinite(T)) or not np.all(np.isfinite(M)):
        return ScoreResult("abstain", "nonfinite spectrum", context.section, context.target, activity)
    # Gate STFT bins where target contributes negligible energy.
    t_max = float(np.max(T))
    if t_max <= 0:
        return ScoreResult("abstain", "target has zero spectral energy", context.section, context.target, activity)
    weights = np.where(T >= (t_max * 1e-5), T, 0.0)
    denom = float(np.sum(weights))
    if denom <= 0:
        return ScoreResult("abstain", "no reliable target bins", context.section, context.target, activity)
    overlap = float(np.sum(weights * M / np.maximum(T + M, 1e-20)) / denom)
    return ScoreResult("valid", "engineering overlap proxy only", context.section, context.target, activity, min(1., max(0., overlap)))


def propose_masker_gain_reduction(
    stems: Mapping[str, np.ndarray], context: SceneContext,
    masker: str, sample_rate: int = 48000, *, delta_db: float = -1.0,
    min_proxy_improvement: float = 0.001, attribution_labels: Mapping[str, str] | None = None,
) -> Proposal:
    """Generate an advisory candidate; no DSP write or auto approval possible.

    For first pilot, only attenuate a VERIFIED dry masker, at most 1 dB.
    """
    audio = _valid_audio(stems, sample_rate)
    def abstain(reason: str, b: float | None = None, c: float | None = None) -> Proposal:
        return Proposal("abstain", "masker_gain", context.target, masker, context.section, 0.0, b, c, None, reason)
    if masker not in audio or masker == context.target:
        return abstain("invalid masker")
    if attribution_labels is None or attribution_labels.get(masker) != "TARGET":
        return abstain("masker source attribution not verified")
    if not math.isfinite(delta_db) or not (-1.0 <= delta_db < 0.0):
        return abstain("candidate gain outside offline pilot bound [-1, 0) dB")
    base = score_scene(audio, context, sample_rate)
    if base.status != "valid" or base.overlap_proxy is None:
        return abstain(f"baseline: {base.reason}")
    changed = {k: v.copy() for k, v in audio.items()}
    changed[masker] = changed[masker] * (10 ** (delta_db / 20.0))
    cand = score_scene(changed, context, sample_rate)
    if cand.status != "valid" or cand.overlap_proxy is None:
        return abstain(f"candidate: {cand.reason}", base.overlap_proxy)
    change = base.overlap_proxy - cand.overlap_proxy
    if change < min_proxy_improvement:
        return abstain("insufficient improvement in engineering overlap proxy", base.overlap_proxy, cand.overlap_proxy)
    return Proposal(
        "candidate_shadow", "masker_gain", context.target, masker, context.section,
        float(delta_db), base.overlap_proxy, cand.overlap_proxy, change,
        "spectral overlap proxy decreased; NO validated perceptual or musical benefit",
    )
