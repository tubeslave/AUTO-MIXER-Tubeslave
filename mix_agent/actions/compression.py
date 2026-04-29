"""Compression action placeholder module.

Real compression is intentionally not faked by the generic mix-agent layer.
Backend/live usage should map compressor recommendations through
``AutoFOHSafetyController`` only when a concrete processor is available.
"""

SUPPORTED = False
