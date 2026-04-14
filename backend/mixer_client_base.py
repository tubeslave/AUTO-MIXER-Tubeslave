"""
Abstract base class for mixer clients.

Defines the common interface that WingClient, DLiveClient, and
any future mixer implementations must satisfy.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional


class MixerClientBase(ABC):
    """Abstract base class for all mixer clients."""

    @abstractmethod
    def connect(self, timeout: float = 5.0) -> bool:
        """Connect to the mixer. Returns True on success."""
        ...

    @abstractmethod
    def disconnect(self):
        """Disconnect from the mixer."""
        ...

    # Subclasses must provide an ``is_connected`` attribute (bool).

    @abstractmethod
    def send(self, address: str, *args):
        """Send a command to the mixer (OSC address or logical address)."""
        ...

    @abstractmethod
    def subscribe(self, address: str, callback: Callable):
        """Subscribe to parameter changes from the mixer."""
        ...

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Return current known mixer state."""
        ...

    # ── Faders ──────────────────────────────────────────────────
    @abstractmethod
    def set_fader(self, channel: int, value_db: float):
        """Set fader level in dB."""
        ...

    @abstractmethod
    def get_fader(self, channel: int) -> float:
        """Get fader level in dB."""
        ...

    # ── Mutes ───────────────────────────────────────────────────
    @abstractmethod
    def set_mute(self, channel: int, muted: bool):
        ...

    @abstractmethod
    def get_mute(self, channel: int) -> bool:
        ...

    # ── Gain / Preamp ──────────────────────────────────────────
    @abstractmethod
    def set_gain(self, channel: int, value_db: float):
        ...

    # ── EQ ──────────────────────────────────────────────────────
    @abstractmethod
    def set_eq_band(self, channel: int, band: int, freq: float, gain: float, q: float):
        ...

    # ── HPF ─────────────────────────────────────────────────────
    def set_hpf(self, channel: int, freq: float, enabled: bool = True):
        """Set high-pass filter. Optional — not all mixers support this."""
        ...

    # ── Channel names ──────────────────────────────────────────
    def get_channel_name(self, channel: int) -> str:
        """Get channel name. Returns generic name if not available."""
        return f"Ch {channel}"

    # ── Channel reset ──────────────────────────────────────────
    def reset_channel_eq(self, channel: int):
        """Reset EQ to flat (all bands gain=0). Override in subclass."""
        ...

    def reset_channel_hpf(self, channel: int):
        """Reset HPF to off. Override in subclass."""
        ...

    def reset_channel_processing(self, channel: int):
        """Reset all processing to neutral. Override in subclass."""
        self.reset_channel_eq(channel)
        self.reset_channel_hpf(channel)

    def get_channel_settings(self, channel: int) -> Dict[str, Any]:
        """Read current channel settings. Override in subclass."""
        return {"channel": channel}

    # ── Scene / Snap ────────────────────────────────────────────
    @abstractmethod
    def recall_scene(self, scene_number: int):
        ...


def create_mixer_client(mixer_type: str, config: dict) -> MixerClientBase:
    """
    Factory: create the right mixer client based on *mixer_type*.

    Parameters
    ----------
    mixer_type : str
        "wing" or "dlive"
    config : dict
        Mixer-specific configuration (ip, port, etc.)

    Returns
    -------
    MixerClientBase
    """
    mixer_type = mixer_type.lower().strip()

    if mixer_type == "wing":
        # Lazy import to avoid circular deps
        from wing_client import WingClient
        ip = config.get("ip", "192.168.1.1")
        port = config.get("port", 2223)
        return WingClient(ip=ip, port=port)

    if mixer_type == "dlive":
        from dlive_client import DLiveClient
        ip = config.get("ip", "192.168.3.70")
        port = config.get("port", 51328)
        tls = config.get("tls", False)
        midi_channel = config.get("midi_base_channel", 0)
        return DLiveClient(ip=ip, port=port, tls=tls, midi_base_channel=midi_channel)

    raise ValueError(f"Unknown mixer type: {mixer_type!r}. Supported: 'wing', 'dlive'.")
