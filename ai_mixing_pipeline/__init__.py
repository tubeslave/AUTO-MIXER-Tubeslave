"""Offline-only AI mixing pipeline.

The package is intentionally separate from live OSC/MIDI control paths.  It
loads files, renders local candidates, evaluates them, and writes reports.
"""

from .config import load_roles_config
from .models import CandidateAction, MixCandidate

__all__ = ["CandidateAction", "MixCandidate", "load_roles_config"]
