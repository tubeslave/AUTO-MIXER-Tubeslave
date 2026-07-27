"""Read-only System Inspector.

Phase 0 is intentionally limited to validated offline snapshot replay.  The
package must not import mixer transports, network scanners, CoreAudio capture,
or any actuation module.
"""

from .engine import InspectorEngine, inspect_snapshot_bundle
from .model import Finding, Severity, SystemGraph
from .safety import (
    INSPECTOR_READ_ONLY,
    InspectorWriteProhibited,
    assert_read_only_mode,
    prohibit_write,
)

__all__ = [
    "Finding",
    "INSPECTOR_READ_ONLY",
    "InspectorEngine",
    "InspectorWriteProhibited",
    "Severity",
    "SystemGraph",
    "assert_read_only_mode",
    "inspect_snapshot_bundle",
    "prohibit_write",
]

