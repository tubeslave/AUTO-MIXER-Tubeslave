"""Backend package bootstrap for repo-root imports.

Importing ``backend.*`` from the repository root must keep legacy sibling
imports working until the backend is fully converted to package-relative
imports. Adding the backend directory to ``sys.path`` here gives callers a
supported repo-root entrypoint without requiring ``PYTHONPATH`` tweaks.
"""

from __future__ import annotations

import sys
from pathlib import Path


_BACKEND_DIR = Path(__file__).resolve().parent
_backend_dir_str = str(_BACKEND_DIR)
if _backend_dir_str not in sys.path:
    sys.path.insert(0, _backend_dir_str)
