"""Embedding distance metrics used by the perceptual evaluator."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def _as_vector(value: Iterable[float] | np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def _align_vectors(
    first: Iterable[float] | np.ndarray,
    second: Iterable[float] | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    a = _as_vector(first)
    b = _as_vector(second)
    if a.size == b.size:
        return a, b
    n = min(a.size, b.size)
    if n <= 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    return a[:n], b[:n]


def embedding_mse(
    first: Iterable[float] | np.ndarray,
    second: Iterable[float] | np.ndarray,
) -> float:
    """Return mean squared error between two embedding vectors."""

    a, b = _align_vectors(first, second)
    if a.size == 0:
        return 0.0
    diff = a - b
    return float(np.mean(diff * diff))


def cosine_distance(
    first: Iterable[float] | np.ndarray,
    second: Iterable[float] | np.ndarray,
) -> float:
    """Return cosine distance in the range 0..2 for aligned embeddings."""

    a, b = _align_vectors(first, second)
    if a.size == 0:
        return 0.0
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= 1e-12 and norm_b <= 1e-12:
        return 0.0
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 1.0
    similarity = float(np.dot(a, b) / (norm_a * norm_b))
    similarity = max(-1.0, min(1.0, similarity))
    return float(1.0 - similarity)


def fad_like_distance(
    reference_embeddings: Iterable[Iterable[float]] | np.ndarray,
    candidate_embeddings: Iterable[Iterable[float]] | np.ndarray,
) -> float:
    """Compute a safe Fréchet-audio-distance-like batch metric.

    This intentionally avoids scipy. For small batches or singular covariance
    matrices the value is still deterministic, but should be interpreted as an
    offline diagnostic rather than a publication-grade FAD implementation.
    """

    ref = np.asarray(reference_embeddings, dtype=np.float64)
    cand = np.asarray(candidate_embeddings, dtype=np.float64)
    if ref.ndim == 1:
        ref = ref.reshape(1, -1)
    if cand.ndim == 1:
        cand = cand.reshape(1, -1)
    if ref.size == 0 or cand.size == 0:
        return 0.0

    dim = min(ref.shape[1], cand.shape[1])
    ref = np.nan_to_num(ref[:, :dim], nan=0.0, posinf=0.0, neginf=0.0)
    cand = np.nan_to_num(cand[:, :dim], nan=0.0, posinf=0.0, neginf=0.0)

    mu_ref = np.mean(ref, axis=0)
    mu_cand = np.mean(cand, axis=0)
    diff = mu_ref - mu_cand

    if ref.shape[0] < 2:
        cov_ref = np.zeros((dim, dim), dtype=np.float64)
    else:
        cov_ref = np.atleast_2d(np.cov(ref, rowvar=False))
    if cand.shape[0] < 2:
        cov_cand = np.zeros((dim, dim), dtype=np.float64)
    else:
        cov_cand = np.atleast_2d(np.cov(cand, rowvar=False))

    product = cov_ref @ cov_cand
    try:
        eigvals = np.linalg.eigvals(product)
        trace_sqrt = float(np.sum(np.sqrt(np.clip(np.real(eigvals), 0.0, None))))
    except Exception:
        trace_sqrt = 0.0

    distance = float(
        np.dot(diff, diff)
        + np.trace(cov_ref)
        + np.trace(cov_cand)
        - 2.0 * trace_sqrt
    )
    return max(0.0, distance)


def normalized_similarity_from_distances(mse: float, cosine: float) -> float:
    """Map MSE and cosine distance to a conservative 0..1 similarity score."""

    mse_score = 1.0 / (1.0 + max(0.0, float(mse)))
    cosine_score = 1.0 - min(max(float(cosine), 0.0), 2.0) / 2.0
    return float(max(0.0, min(1.0, 0.5 * mse_score + 0.5 * cosine_score)))
