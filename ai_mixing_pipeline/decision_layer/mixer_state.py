"""Small serializable mixer-state helpers for offline correction runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

from ai_mixing_pipeline.audio_utils import audio_files


def infer_channel_role(name: str) -> str:
    label = name.lower().replace("_", " ").replace("-", " ")
    if "kick" in label:
        return "kick"
    if "snare" in label:
        return "snare"
    if any(token in label for token in ("drum", "tom", "oh ", "overhead")):
        return "drums"
    if "bass" in label:
        return "bass"
    if any(token in label for token in ("vocal", "vox", "voice")):
        return "vocal"
    if any(token in label for token in ("guitar", "gtr")):
        return "guitars"
    if any(token in label for token in ("keys", "piano", "synth")):
        return "keys"
    return "unknown"


def load_channel_map(input_dir: str | Path) -> dict[str, Any]:
    path = Path(input_dir).expanduser() / "config" / "channel_map.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def role_from_channel_map(path: Path, channel_map: dict[str, Any]) -> str:
    for key in (path.name, path.stem):
        item = channel_map.get(key)
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for role_key in ("role", "source_role", "stem_role", "instrument"):
                if item.get(role_key):
                    return str(item[role_key])
    channels = channel_map.get("channels")
    if isinstance(channels, list):
        for item in channels:
            if isinstance(item, dict) and item.get("file") in {path.name, path.stem}:
                return str(item.get("role") or item.get("instrument") or "unknown")
    return infer_channel_role(path.stem)


@dataclass
class ChannelState:
    channel_id: str
    path: str
    role: str = "unknown"
    gain_db: float = 0.0
    pan: float = 0.0
    muted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "path": self.path,
            "role": self.role,
            "gain_db": self.gain_db,
            "pan": self.pan,
            "muted": self.muted,
        }


@dataclass
class MixerState:
    sample_rate: int = 48000
    channels: dict[str, ChannelState] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "channels": {key: value.to_dict() for key, value in self.channels.items()},
            "metadata": dict(self.metadata),
            "osc_midi_sent": False,
        }


def discover_multitrack(input_dir: str | Path) -> tuple[Path, dict[str, Any], list[Path]]:
    root = Path(input_dir).expanduser()
    multitrack_dir = root / "multitrack"
    files = audio_files(multitrack_dir)
    if not files:
        raise ValueError(f"No supported audio files found in {multitrack_dir}")
    return multitrack_dir, load_channel_map(root), files
