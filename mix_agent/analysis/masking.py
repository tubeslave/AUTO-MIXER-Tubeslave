"""Stem-aware spectral overlap and masking proxies."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from .loudness import amp_to_db
from .spectral import band_powers


def _normalize(values: Mapping[str, float]) -> np.ndarray:
    vector = np.asarray([float(values[key]) for key in sorted(values)], dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-12)


def compute_masking_matrix(
    stems: Mapping[str, np.ndarray],
    sample_rate: int,
) -> Dict[str, Dict[str, float]]:
    """Estimate broad-band spectral overlap between stems."""
    powers = {name: band_powers(audio, sample_rate) for name, audio in stems.items()}
    normalized = {name: _normalize(values) for name, values in powers.items()}
    matrix: Dict[str, Dict[str, float]] = {}
    for a, vec_a in normalized.items():
        matrix[a] = {}
        for b, vec_b in normalized.items():
            if a == b:
                matrix[a][b] = 1.0
            else:
                matrix[a][b] = round(float(np.dot(vec_a, vec_b)), 6)
    return matrix


def compute_stem_relationships(
    stems: Mapping[str, np.ndarray],
    roles: Mapping[str, str],
    sample_rate: int,
) -> Dict[str, Any]:
    """Compute simple role-aware balance and conflict proxies."""
    levels = {
        name: amp_to_db(float(np.sqrt(np.mean(np.square(audio)) + 1e-12)))
        for name, audio in stems.items()
    }
    matrix = compute_masking_matrix(stems, sample_rate) if stems else {}

    vocal_names = [
        name
        for name, role in roles.items()
        if role in {"lead_vocal", "vocal"} or "vocal" in role
    ]
    accompaniment = [name for name in stems if name not in vocal_names]
    vocal_to_acc = None
    if vocal_names and accompaniment:
        vocal_level = max(levels[name] for name in vocal_names)
        acc_level = float(np.mean([levels[name] for name in accompaniment]))
        vocal_to_acc = round(vocal_level - acc_level, 3)

    kick_names = [name for name, role in roles.items() if role == "kick"]
    bass_names = [name for name, role in roles.items() if role == "bass"]
    kick_bass = None
    if kick_names and bass_names:
        pairs = [
            matrix.get(kick, {}).get(bass, 0.0)
            for kick in kick_names
            for bass in bass_names
        ]
        kick_bass = {
            "spectral_overlap": round(float(max(pairs)), 6) if pairs else 0.0,
            "kick_to_bass_level_db": round(
                max(levels[kick] for kick in kick_names) - max(levels[bass] for bass in bass_names),
                3,
            ),
        }

    highest_conflicts = []
    for a, row in matrix.items():
        for b, score in row.items():
            if a < b:
                highest_conflicts.append({"a": a, "b": b, "overlap": score})
    highest_conflicts.sort(key=lambda item: item["overlap"], reverse=True)
    return {
        "stem_loudness_dbfs": {name: round(value, 3) for name, value in levels.items()},
        "relative_vocal_to_accompaniment_db": vocal_to_acc,
        "kick_to_bass_relationship": kick_bass,
        "stem_spectral_overlap_top": highest_conflicts[:10],
        "bark_erb_masking_proxy": matrix,
        "transient_conflict_matrix": {},
        "panorama_distribution": {},
    }
