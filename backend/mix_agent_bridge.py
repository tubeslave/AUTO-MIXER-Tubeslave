"""Backend import shim for the root ``mix_agent`` live-console bridge."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mix_agent.backend_bridge import MixAgentBackendBridge
from mix_agent.models import BackendBridgeResult, BackendChannelSnapshot

__all__ = ["BackendBridgeResult", "BackendChannelSnapshot", "MixAgentBackendBridge"]
