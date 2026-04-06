"""Handler modules for AutoMixerServer message dispatch.

Each sub-module exposes a ``register_handlers(server)`` function that returns
a ``dict[str, Callable]`` mapping message-type strings to async handler
coroutines with signature ``(websocket, data) -> None``.
"""

from .audio_handlers import register_handlers as _audio
from .connection_handlers import register_handlers as _connection
from .mixer_handlers import register_handlers as _mixer
from .routing_handlers import register_handlers as _routing
from .snapshot_handlers import register_handlers as _snapshot
from .voice_handlers import register_handlers as _voice
from .gain_staging_handlers import register_handlers as _gain_staging
from .channel_scan_handlers import register_handlers as _channel_scan
from .eq_handlers import register_handlers as _eq
from .phase_handlers import register_handlers as _phase
from .fader_handlers import register_handlers as _fader
from .automation_handlers import register_handlers as _automation
from .soundcheck_handlers import register_handlers as _soundcheck
from .compressor_handlers import register_handlers as _compressor
from .feedback_handlers import register_handlers as _feedback
from .agent_handlers import register_handlers as _agent

_ALL_REGISTRARS = [
    _audio, _connection, _mixer, _routing, _snapshot, _voice,
    _gain_staging, _channel_scan, _eq, _phase, _fader,
    _automation, _soundcheck, _compressor, _feedback, _agent,
]


def register_all_handlers(server):
    """Collect every handler mapping and return a single dispatch dict."""
    dispatch = {}
    for registrar in _ALL_REGISTRARS:
        dispatch.update(registrar(server))
    return dispatch
