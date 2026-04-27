"""Tests for shared offline artifact locations."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from output_paths import ai_logs_path, ai_mixing_path, ensure_parent_dir


def test_ai_mixing_and_logs_paths_use_desktop_folders():
    audio_path = ai_mixing_path("mix.wav")
    log_path = ai_logs_path("mix.jsonl")

    assert "Ai MIXING" in audio_path.parts
    assert audio_path.name == "mix.wav"
    assert "Ai LOGS" in log_path.parts
    assert log_path.name == "mix.jsonl"


def test_ensure_parent_dir_expands_and_creates(tmp_path):
    target = tmp_path / "nested" / "report.json"

    result = ensure_parent_dir(target)

    assert result == target
    assert target.parent.exists()
