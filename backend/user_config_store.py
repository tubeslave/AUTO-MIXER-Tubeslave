"""Helpers for loading, normalizing, and saving user-facing config."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


def get_user_config_path(repo_root: str) -> str:
    """Return the canonical path to the persisted user config."""
    return os.path.join(repo_root, "config", "user_config.json")


def normalize_mixer_user_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize mixer settings so frontend aliases and backend keys stay aligned."""
    if not isinstance(settings, dict):
        return {}

    normalized = dict(settings)
    mixer_type = normalized.get("mixerType") or normalized.get("type") or "wing"

    if mixer_type == "wing":
        ip = normalized.get("mixerIp") or normalized.get("ip") or "192.168.1.102"
        send_port = normalized.get("mixerSendPort") or normalized.get("send_port") or 2222
        receive_port = normalized.get("mixerReceivePort") or normalized.get("receive_port")
        port = normalized.get("mixerPort") or normalized.get("port") or receive_port or 2223
        receive_port = receive_port or port

        normalized.update({
            "mixerType": "wing",
            "type": "wing",
            "mixerIp": ip,
            "ip": ip,
            "mixerPort": port,
            "port": port,
            "mixerSendPort": send_port,
            "send_port": send_port,
            "mixerReceivePort": receive_port,
            "receive_port": receive_port,
        })
        return normalized

    ip = normalized.get("dliveIp") or normalized.get("mixerIp") or normalized.get("ip") or "192.168.3.70"
    port = normalized.get("dlivePort") or normalized.get("mixerPort") or normalized.get("port") or 51328
    tls = normalized.get("dliveTls")
    if tls is None:
        tls = normalized.get("tls", False)
    midi_base_channel = normalized.get("dliveMidiChannel")
    if midi_base_channel is None:
        midi_base_channel = normalized.get("midi_base_channel", 0)

    normalized.update({
        "mixerType": "dlive",
        "type": "dlive",
        "mixerIp": ip,
        "dliveIp": ip,
        "ip": ip,
        "mixerPort": port,
        "dlivePort": port,
        "port": port,
        "dliveTls": tls,
        "tls": tls,
        "dliveMidiChannel": midi_base_channel,
        "midi_base_channel": midi_base_channel,
    })
    return normalized


def normalize_user_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize persisted config sections for frontend/backend compatibility."""
    if not isinstance(data, dict):
        return {}

    normalized = dict(data)
    if "mixer" in normalized:
        normalized["mixer"] = normalize_mixer_user_settings(normalized["mixer"])
    return normalized


def load_user_config(path: str) -> Dict[str, Any]:
    """Load and normalize user config from disk."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("User config loaded from %s", path)
        return normalize_user_config(data)
    except Exception as exc:
        logger.warning("Failed to load user config: %s", exc)
        return {}


def save_user_config(path: str, section: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    """Save one normalized section back to the user config file."""
    existing = load_user_config(path)
    existing[section] = normalize_mixer_user_settings(settings) if section == "mixer" else settings

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    logger.info("User config saved to %s: section=%s", path, section)
    return existing
