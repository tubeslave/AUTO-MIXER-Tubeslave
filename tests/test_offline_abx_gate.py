from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "offline_abx_gate.py"
SPEC = importlib.util.spec_from_file_location("offline_abx_gate", MODULE_PATH)
offline_abx_gate = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = offline_abx_gate
SPEC.loader.exec_module(offline_abx_gate)


def _choices(preferred: str, preferred_count: int, other_count: int):
    other = "A" if preferred == "B" else "B"
    return [{"trial_id": f"p{i}", "choice": preferred} for i in range(preferred_count)] + [
        {"trial_id": f"o{i}", "choice": other} for i in range(other_count)
    ]


def test_abx_gate_passes_loudness_peak_and_preference():
    payload = {
        "schema_version": "automixer-abx-v1",
        "seed": 4242,
        "expected_preferred_variant": "B",
        "clips": [
            {"clip_id": "a", "variant": "A", "lufs": -14.12, "peak_dbfs": -1.4, "sha256": "a" * 64},
            {"clip_id": "b", "variant": "B", "lufs": -14.03, "peak_dbfs": -0.7, "sha256": "b" * 64},
        ],
        "trials": _choices("B", preferred_count=25, other_count=5),
    }

    result = offline_abx_gate.evaluate_abx_artifact(payload)

    assert result["overall_pass"] is True
    assert result["technical_pass"] is True
    assert result["preference_pass"] is True
    assert result["loudness"]["delta_lu"] == 0.09
    assert result["trials"]["preferred_count"] == 25
    assert result["trials"]["preference_claim_allowed"] is True
    assert result["offline_only"] is True
    assert result["live_write"] is False


def test_abx_gate_blocks_peak_and_lufs_mismatch():
    payload = {
        "seed": 7,
        "clips": [
            {"clip_id": "a", "variant": "A", "lufs": -14.0, "peak_dbfs": -1.0, "sha256": "a" * 64},
            {"clip_id": "b", "variant": "B", "lufs": -13.1, "peak_dbfs": 0.0, "sha256": "b" * 64},
        ],
        "trials": _choices("B", preferred_count=20, other_count=10),
    }

    result = offline_abx_gate.evaluate_abx_artifact(payload)
    codes = {failure["code"] for failure in result["failures"]}

    assert result["overall_pass"] is False
    assert "lufs_mismatch" in codes
    assert "peak_ceiling_exceeded" in codes


def test_abx_gate_requires_seed_and_clip_checksums_by_default():
    payload = {
        "clips": [
            {"clip_id": "a", "variant": "A", "lufs": -14.0, "peak_dbfs": -1.0},
            {"clip_id": "b", "variant": "B", "lufs": -14.1, "peak_dbfs": -1.2, "sha256": "b" * 64},
        ],
    }

    result = offline_abx_gate.evaluate_abx_artifact(payload)
    codes = {failure["code"] for failure in result["failures"]}

    assert "missing_seed" in codes
    assert "missing_clip_checksum" in codes
    assert result["technical_pass"] is False


def test_abx_gate_can_pass_technical_artifact_without_preference_claim():
    payload = {
        "seed": 11,
        "clips": [
            {"clip_id": "a", "variant": "A", "lufs": -16.0, "peak_dbfs": -3.0, "sha256": "a" * 64},
            {"clip_id": "b", "variant": "B", "lufs": -15.8, "peak_dbfs": -2.8, "sha256": "b" * 64},
        ],
    }

    result = offline_abx_gate.evaluate_abx_artifact(payload)

    assert result["overall_pass"] is True
    assert result["technical_pass"] is True
    assert result["preference_pass"] is False
    assert result["trials"]["n"] == 0
    assert result["warnings"][0]["code"] == "no_abx_trials"


def test_abx_gate_fails_overall_when_preference_is_required():
    payload = {
        "seed": 11,
        "clips": [
            {"clip_id": "a", "variant": "A", "lufs": -16.0, "peak_dbfs": -3.0, "sha256": "a" * 64},
            {"clip_id": "b", "variant": "B", "lufs": -15.8, "peak_dbfs": -2.8, "sha256": "b" * 64},
        ],
        "trials": _choices("B", preferred_count=16, other_count=14),
    }
    config = offline_abx_gate.ABXGateConfig(require_preference=True)

    result = offline_abx_gate.evaluate_abx_artifact(payload, config)

    assert result["overall_pass"] is False
    assert "preference_gate_failed" in {failure["code"] for failure in result["failures"]}


def test_abx_gate_source_has_no_forbidden_live_write_calls():
    source = MODULE_PATH.read_text(encoding="utf-8")

    forbidden = [
        "set_fader(",
        "set_gain(",
        "set_eq(",
        "set_compressor(",
        "route_output(",
        "load_snap(",
        "save_snap(",
        "reset(",
        "mute(",
    ]
    assert not any(token in source for token in forbidden)
