"""Offline-only spectral EQ method inspired by observed smartEQ4 architecture."""

from .method import (
    EQBand,
    OfflineSmartEQConfig,
    OfflineSmartEQReport,
    apply_offline_smarteq_eq,
    design_offline_smarteq_eq,
)

__all__ = [
    "EQBand",
    "OfflineSmartEQConfig",
    "OfflineSmartEQReport",
    "apply_offline_smarteq_eq",
    "design_offline_smarteq_eq",
]
