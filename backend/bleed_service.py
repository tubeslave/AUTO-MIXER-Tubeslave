"""
Centralized bleed detection service for all modules.

Provides unified access to BleedDetector and BleedCompensator from core/.
All modules (Gain Staging, Phase Alignment, Auto EQ, Auto Compressor, Auto Fader)
use this service to get bleed information and compensated levels.
"""

import logging
from typing import Dict, Optional

try:
    from core.bleed_detector import BleedDetector, BleedInfo
    from core.bleed_compensator import get_compensated_level

    BLEED_AVAILABLE = True
except ImportError:
    BleedDetector = None
    BleedInfo = None
    get_compensated_level = None
    BLEED_AVAILABLE = False

logger = logging.getLogger(__name__)


class BleedService:
    """
    Centralized service for bleed detection and compensation.

    Maintains a single BleedDetector instance and provides methods for:
    - Updating bleed detection with current channel metrics
    - Getting bleed information for specific channels
    - Getting compensated levels (own signal estimate)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize bleed service.

        Args:
            config: Configuration dict with bleed_protection settings
        """
        if not BLEED_AVAILABLE:
            logger.warning("BleedDetector not available - bleed detection disabled")
            self.enabled = False
            self.bleed_detector = None
            self.compensation_factor_db = 6.0
            self.compensation_mode = "linear"
            return

        config = config or {}
        bleed_cfg = (
            config.get("bleed_protection")
            or config.get("automation", {}).get("bleed_protection")
            or {}
        )

        self.enabled = bleed_cfg.get("enabled", True)
        self.compensation_factor_db = bleed_cfg.get("compensation_factor_db", 6.0)
        self.compensation_mode = bleed_cfg.get("compensation_mode", "band_level")

        if self.enabled:
            self.bleed_detector = BleedDetector(bleed_cfg)
            logger.info(
                f"BleedService initialized: enabled={self.enabled}, "
                f"compensation_factor={self.compensation_factor_db}dB, "
                f"mode={self.compensation_mode}"
            )
        else:
            self.bleed_detector = None
            logger.info("BleedService initialized but disabled")

        # Cached results from last update
        self.bleed_results: Dict[int, BleedInfo] = {}
        self.all_channel_levels: Dict[int, float] = {}
        self.all_channel_centroids: Dict[int, float] = {}
        self.all_channel_metrics: Dict[int, object] = {}

    def configure(self, instrument_types: Dict[int, str]):
        """
        Configure instrument types for bleed detection.

        Args:
            instrument_types: Dict mapping channel_id -> instrument_type
        """
        if self.bleed_detector:
            self.bleed_detector.configure(instrument_types)
            logger.debug(
                f"BleedService configured with {len(instrument_types)} instrument types"
            )

    def reset(self):
        """Clear temporal history in bleed detector."""
        if self.bleed_detector:
            self.bleed_detector.reset()

    def update(
        self,
        all_channel_levels: Dict[int, float],
        all_channel_centroids: Dict[int, float],
        all_channel_metrics: Optional[Dict[int, object]] = None,
    ):
        """
        Update bleed detection for all channels.

        Should be called periodically (e.g., every 100-200ms) with current metrics.
        Results are cached and can be retrieved via get_bleed_info().

        Args:
            all_channel_levels: Dict[channel_id, lufs_level]
            all_channel_centroids: Dict[channel_id, spectral_centroid]
            all_channel_metrics: Dict[channel_id, metrics_object] with band_energy_* attributes
        """
        if not self.enabled or not self.bleed_detector:
            self.bleed_results = {}
            return

        all_channel_metrics = all_channel_metrics or {}

        # Cache metrics for compensation
        self.all_channel_levels = all_channel_levels.copy()
        self.all_channel_centroids = all_channel_centroids.copy()
        self.all_channel_metrics = all_channel_metrics.copy()

        # Detect bleed for each channel
        self.bleed_results = {}
        for channel_id in all_channel_levels.keys():
            if channel_id not in all_channel_centroids:
                continue

            bleed_info = self.bleed_detector.detect_bleed(
                channel_id=channel_id,
                current_lufs=all_channel_levels[channel_id],
                spectral_centroid=all_channel_centroids[channel_id],
                all_channel_levels=all_channel_levels,
                all_channel_centroids=all_channel_centroids,
                all_channel_metrics=all_channel_metrics,
            )

            self.bleed_results[channel_id] = bleed_info

    def get_bleed_info(self, channel_id: int) -> Optional[BleedInfo]:
        """
        Get bleed information for a specific channel.

        Args:
            channel_id: Channel ID

        Returns:
            BleedInfo if available, None otherwise
        """
        if not self.enabled:
            return None

        return self.bleed_results.get(channel_id)

    def get_all_bleed(self) -> Dict[int, BleedInfo]:
        """
        Get bleed information for all channels.

        Returns:
            Dict[channel_id, BleedInfo]
        """
        if not self.enabled:
            return {}

        return self.bleed_results.copy()

    def get_compensated_level(self, channel_id: int, raw_lufs: float) -> float:
        """
        Get compensated level (own signal estimate) for a channel.

        Args:
            channel_id: Channel ID
            raw_lufs: Measured LUFS (including bleed)

        Returns:
            Compensated LUFS (own signal estimate), or raw_lufs if no bleed detected
        """
        if not self.enabled or not BLEED_AVAILABLE:
            return raw_lufs

        bleed_info = self.get_bleed_info(channel_id)
        if not bleed_info or bleed_info.bleed_ratio <= 0:
            return raw_lufs

        channel_metrics = self.all_channel_metrics.get(channel_id)
        source_metrics = None
        if bleed_info.bleed_source_channel:
            source_metrics = self.all_channel_metrics.get(
                bleed_info.bleed_source_channel
            )

        return get_compensated_level(
            raw_lufs=raw_lufs,
            bleed_info=bleed_info,
            channel_metrics=channel_metrics,
            source_metrics=source_metrics,
            compensation_factor_db=self.compensation_factor_db,
            compensation_mode=self.compensation_mode,
        )
