"""Concert and soundcheck learning telemetry package.

The telemetry package is local-disk and read-only with respect to mixer
hardware. It records normalized events for future training/evaluation, but does
not send OSC/MIDI/WING writes.
"""

from .schemas import SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION"]
