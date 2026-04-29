"""End-to-end analysis pipeline for loaded offline audio."""

from __future__ import annotations

from typing import Dict

import numpy as np

from mix_agent.models import MixAnalysis, TrackMetrics

from .artifacts import artifact_summary
from .dynamics import compute_dynamics_metrics
from .loader import LoadedAudioContext
from .loudness import compute_level_metrics
from .masking import compute_masking_matrix, compute_stem_relationships
from .reference import compare_reference
from .spectral import compute_spectral_metrics
from .stereo import compute_stereo_metrics


def _track_metrics(name: str, role: str, audio: np.ndarray, sample_rate: int) -> TrackMetrics:
    level, limitations = compute_level_metrics(audio, sample_rate)
    spectral = compute_spectral_metrics(audio, sample_rate)
    dynamics = compute_dynamics_metrics(audio, sample_rate)
    stereo = compute_stereo_metrics(audio, sample_rate)
    artifacts = artifact_summary(level, stereo)
    limitations.extend(stereo.get("limitations", []))
    return TrackMetrics(
        name=name,
        role=role,
        sample_rate=sample_rate,
        duration_sec=round(len(audio) / sample_rate, 3),
        level=level,
        spectral=spectral,
        dynamics=dynamics,
        stereo=stereo,
        artifacts=artifacts,
        limitations=limitations,
    )


def analyze_loaded_context(loaded: LoadedAudioContext) -> MixAnalysis:
    """Analyze mix, stems, masking and optional reference."""
    mix_metrics = _track_metrics("mix", "mix", loaded.mix_audio, loaded.mix_sample_rate)
    stem_metrics: Dict[str, TrackMetrics] = {}
    roles = {}
    for name, audio in loaded.stems.items():
        role = loaded.stem_info[name].role
        roles[name] = role
        stem_metrics[name] = _track_metrics(name, role, audio, loaded.mix_sample_rate)

    masking_matrix = (
        compute_masking_matrix(loaded.stems, loaded.mix_sample_rate)
        if loaded.stems
        else {}
    )
    relationships = (
        compute_stem_relationships(loaded.stems, roles, loaded.mix_sample_rate)
        if loaded.stems
        else {
            "limitation": "No stems supplied; stem-aware masking and balance are limited.",
        }
    )
    reference = compare_reference(
        loaded.mix_audio,
        loaded.reference_audio,
        loaded.mix_sample_rate,
    )
    limitations = list(loaded.context.limitations)
    if not loaded.stems:
        limitations.append("No stems supplied; analysis is limited to the stereo mix.")
    limitations.extend(mix_metrics.limitations)
    limitations.extend(reference.limitations)
    return MixAnalysis(
        context=loaded.context,
        mix=mix_metrics,
        stems=stem_metrics,
        masking_matrix=masking_matrix,
        stem_relationships=relationships,
        reference=reference,
        limitations=limitations,
    )
