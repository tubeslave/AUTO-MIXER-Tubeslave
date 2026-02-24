"""Channel classifier - classifies instrument type from acoustic features."""

from typing import Dict, Any


class ChannelClassifier:
    """Classifies channel instrument type from metrics/features."""

    def __init__(self):
        pass

    def classify(self, metrics: Any = None, features: Any = None) -> str:
        """Return instrument type string (e.g. 'kick', 'snare')."""
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
