"""Activity detection - channels below threshold are considered inactive/silent."""

# LUFS threshold: channels below this are "inactive" (silent or bleed-only)
ACTIVITY_THRESHOLD_LUFS = -50.0


class ActivityDetector:
    """Detects if a channel has meaningful signal (vs silence/bleed)."""

    def __init__(self, threshold_db: float = ACTIVITY_THRESHOLD_LUFS):
        self.threshold_db = threshold_db

    def is_active(self, lufs: float) -> bool:
        """True if channel level is above activity threshold."""
        return lufs >= self.threshold_db
