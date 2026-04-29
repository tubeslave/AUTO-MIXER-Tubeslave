"""Opt-in production mixing pipeline v1.

The package is intentionally isolated from legacy defaults. It can run offline
dry-run smoke tests today and provides a safe command-queue boundary for future
live OSC integration.
"""

from .config import ProductionMixConfig, load_production_mix_config
from .pipeline import ProductionMixPipeline, run_production_mix_v1

__all__ = [
    "ProductionMixConfig",
    "ProductionMixPipeline",
    "load_production_mix_config",
    "run_production_mix_v1",
]
