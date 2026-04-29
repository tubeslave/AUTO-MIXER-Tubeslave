"""Tests for backend/user_config_store.py."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from user_config_store import (
    get_user_config_path,
    load_user_config,
    normalize_mixer_user_settings,
    normalize_user_config,
    save_user_config,
)


def test_get_user_config_path_joins_repo_root():
    path = get_user_config_path("/tmp/repo")
    assert path == "/tmp/repo/config/user_config.json"


def test_normalize_wing_settings_from_frontend_aliases():
    normalized = normalize_mixer_user_settings({
        "mixerType": "wing",
        "mixerIp": "10.0.0.5",
        "mixerPort": 2223,
    })
    assert normalized["type"] == "wing"
    assert normalized["ip"] == "10.0.0.5"
    assert normalized["send_port"] == 2222
    assert normalized["receive_port"] == 2223


def test_normalize_dlive_settings_from_backend_keys():
    normalized = normalize_mixer_user_settings({
        "type": "dlive",
        "ip": "192.168.3.70",
        "port": 51328,
        "tls": True,
        "midi_base_channel": 3,
    })
    assert normalized["mixerType"] == "dlive"
    assert normalized["dliveIp"] == "192.168.3.70"
    assert normalized["dlivePort"] == 51328
    assert normalized["dliveTls"] is True
    assert normalized["dliveMidiChannel"] == 3


def test_normalize_user_config_only_touches_mixer_section():
    normalized = normalize_user_config({
        "mixer": {"mixerType": "wing", "mixerIp": "10.0.0.5", "mixerPort": 2223},
        "gainStaging": {"targetLufs": -18},
    })
    assert normalized["mixer"]["ip"] == "10.0.0.5"
    assert normalized["gainStaging"] == {"targetLufs": -18}


def test_save_and_load_user_config_roundtrip(tmp_path):
    path = tmp_path / "config" / "user_config.json"
    save_user_config(str(path), "mixer", {
        "mixerType": "wing",
        "mixerIp": "10.0.0.9",
        "mixerPort": 2223,
    })
    save_user_config(str(path), "gainStaging", {"targetLufs": -20})

    loaded = load_user_config(str(path))
    assert loaded["mixer"]["ip"] == "10.0.0.9"
    assert loaded["mixer"]["receive_port"] == 2223
    assert loaded["gainStaging"]["targetLufs"] == -20

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["mixer"]["type"] == "wing"
