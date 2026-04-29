"""Perceptual shadow evaluation for AUTO-MIXER-Tubeslave."""

from .embedding_backend import (
    EmbeddingBackend,
    LightweightEmbeddingBackend,
    MERTEmbeddingBackend,
    create_embedding_backend,
)
from .metrics import (
    cosine_distance,
    embedding_mse,
    fad_like_distance,
)
from .perceptual_evaluator import (
    PerceptualConfig,
    PerceptualEvaluationResult,
    PerceptualEvaluator,
)
from .reference_store import ReferenceStore
from .reward import RewardSignal

__all__ = [
    "EmbeddingBackend",
    "LightweightEmbeddingBackend",
    "MERTEmbeddingBackend",
    "PerceptualConfig",
    "PerceptualEvaluationResult",
    "PerceptualEvaluator",
    "ReferenceStore",
    "RewardSignal",
    "cosine_distance",
    "create_embedding_backend",
    "embedding_mse",
    "fad_like_distance",
]
