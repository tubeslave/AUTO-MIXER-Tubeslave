"""Static balancer - collects samples and calculates one-time balance result."""

import time
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class StaticBalancer:
    """Collects LUFS samples and calculates static balance adjustments."""

    def __init__(self):
        self.is_collecting = False
        self.collection_duration = 15.0
        self.collection_start_time = 0.0
        self.balance_result: Dict[int, float] = {}
        self._samples: Dict[int, List[float]] = {}

    def start_collection(self, duration: float):
        """Start collecting samples for balance calculation."""
        self.is_collecting = True
        self.collection_duration = duration
        self.collection_start_time = time.time()
        self._samples.clear()
        self.balance_result = {}

    def is_collection_complete(self) -> bool:
        """True if collection time has elapsed."""
        return time.time() - self.collection_start_time >= self.collection_duration

    def collect_sample(self, ch_id: int, lufs: float, spectral_centroid: float = 0, is_active: bool = True):
        """Add a sample for a channel."""
        if ch_id not in self._samples:
            self._samples[ch_id] = []
        self._samples[ch_id].append(lufs)
        if len(self._samples[ch_id]) > 200:
            self._samples[ch_id].pop(0)

    def calculate_statistics(self):
        """Compute statistics from collected samples."""
        pass

    def calculate_balance(
        self,
        reference_channels: List[int],
        instrument_types: Dict[int, str],
        instrument_offsets: Dict[str, float],
    ) -> Dict:
        """Calculate balance adjustments. Returns result dict."""
        import numpy as np
        self.balance_result = {}
        if not self._samples:
            return {"adjustments": {}, "message": "No samples collected"}

        target = -18.0
        for ch_id, samples in self._samples.items():
            if ch_id in reference_channels:
                self.balance_result[ch_id] = 0.0
                continue
            if not samples:
                continue
            avg = float(np.median(samples))
            offset = instrument_offsets.get(instrument_types.get(ch_id, "unknown"), -7.0)
            tgt = target + offset
            self.balance_result[ch_id] = tgt - avg

        return {
            "adjustments": self.balance_result,
            "message": f"Balanced {len(self.balance_result)} channels",
        }
