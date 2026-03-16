"""Channel classifier - classifies instrument type from acoustic features.

Uses the ML-based CNN classifier (backend.ml.channel_classifier) as primary,
with spectral-centroid stub as fallback when ML is unavailable.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try to load the ML classifier
_ml_classifier_class = None
_ML_INSTRUMENT_CLASSES = None
try:
    from backend.ml.channel_classifier import (
        ChannelClassifier as MLChannelClassifier,
        INSTRUMENT_CLASSES as ML_INSTRUMENT_CLASSES,
    )
    _ml_classifier_class = MLChannelClassifier
    _ML_INSTRUMENT_CLASSES = ML_INSTRUMENT_CLASSES
except ImportError:
    logger.info("ML channel classifier unavailable, using spectral-centroid fallback")


def _stub_classify(metrics: Any = None, features: Any = None) -> str:
    """Original spectral-centroid stub (4 drum types)."""
    if metrics is not None:
        centroid = getattr(metrics, 'spectral_centroid', 0)
        if centroid < 200:
            return 'kick'
        if centroid < 800:
            return 'tom'
        if centroid < 2000:
            return 'snare'
        if centroid < 4000:
            return 'hihat'
    return 'unknown'


class ChannelClassifier:
    """Classifies channel instrument type from metrics/features.

    Uses the 24-class ML CNN classifier when available, falling back to
    a simple spectral-centroid heuristic for 4 drum types.
    """

    def __init__(self, model_path: Optional[str] = None, use_ml: bool = True):
        self._ml_classifier = None
        if use_ml and _ml_classifier_class is not None:
            try:
                self._ml_classifier = _ml_classifier_class(model_path=model_path)
                logger.info("ML channel classifier loaded (24 classes)")
            except Exception as e:
                logger.warning(f"Failed to initialize ML classifier, using fallback: {e}")
                self._ml_classifier = None

    @property
    def has_ml(self) -> bool:
        return self._ml_classifier is not None

    def classify(self, metrics: Any = None, features: Any = None,
                 audio: Optional[np.ndarray] = None) -> str:
        """Return top-1 instrument type string (e.g. 'kick', 'snare').

        If ML classifier is available and ``audio`` is provided, uses the
        CNN classifier.  Otherwise falls back to the spectral-centroid stub.
        """
        # Try ML path first
        if self._ml_classifier is not None and audio is not None:
            try:
                top = self._ml_classifier.classify_top_k(audio, k=1)
                if top:
                    label, confidence = top[0]
                    if confidence > 0.1:
                        return label
            except Exception as e:
                logger.debug(f"ML classify failed, falling back to stub: {e}")

        # Fallback to stub
        return _stub_classify(metrics, features)

    def classify_detailed(self, audio: Optional[np.ndarray] = None,
                          metrics: Any = None,
                          features: Any = None,
                          k: int = 3) -> List[Tuple[str, float]]:
        """Return top-k classifications with confidence scores.

        Returns list of (label, confidence) tuples.
        Falls back to stub result with confidence 1.0 if ML unavailable.
        """
        if self._ml_classifier is not None and audio is not None:
            try:
                return self._ml_classifier.classify_top_k(audio, k=k)
            except Exception as e:
                logger.debug(f"ML classify_detailed failed: {e}")

        label = _stub_classify(metrics, features)
        return [(label, 1.0)]

    def classify_all(self, audio: np.ndarray) -> Dict[str, float]:
        """Return full probability distribution over all classes.

        Only available with ML classifier. Returns empty dict otherwise.
        """
        if self._ml_classifier is not None:
            try:
                return self._ml_classifier.classify(audio)
            except Exception as e:
                logger.debug(f"ML classify_all failed: {e}")
        return {}
