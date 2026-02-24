"""ML data collector for training/fine-tuning."""

from typing import Dict, Any


class MLDataCollector:
    """Collects metrics for ML training (stub)."""

    def __init__(self):
        self._samples: list = []

    def add_sample(self, channel_id: int, metrics: Any, adjustment: float):
        """Add a sample for training."""
        pass

    def reset(self):
        """Clear collected data."""
        self._samples.clear()
