from __future__ import annotations

import importlib.util
import json
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


def test_abx_gate_accepts_attached_safety_audit_when_required():
    payload = {
        "seed": 42,
        "clips": [
            {"clip_id": "a", "variant": "A", "lufs": -16.0, "peak_dbfs": -3.0, "sha256": "a" * 64},
            {"clip_id": "b", "variant": "B", "lufs": -15.8, "peak_dbfs": -2.8, "sha256": "b" * 64},
        ],
        "safety_audits": [
            {
                "name": "autoeq_audit_validation",
                "schema_version": "0.1",
                "passed": True,
                "artifact": "artifacts/offline/autoeq_audit_report.json",
                "target_render_sha256": "b" * 64,
            }
        ],
    }
    config = offline_abx_gate.ABXGateConfig(require_safety_audits=True)

    result = offline_abx_gate.evaluate_abx_artifact(payload, config)

    assert result["overall_pass"] is True
    assert result["technical_pass"] is True
    assert result["safety_audits"]["required"] is True
    assert result["safety_audits"]["passed"] == ["autoeq_audit_validation"]
    assert result["safety_audits"]["with_provenance"] == ["autoeq_audit_validation"]


def test_abx_gate_requires_safety_audit_render_provenance_when_required():
    payload = {
        "seed": 42,
        "clips": [
            {"clip_id": "a", "variant": "A", "lufs": -16.0, "peak_dbfs": -3.0, "sha256": "a" * 64},
            {"clip_id": "b", "variant": "B", "lufs": -15.8, "peak_dbfs": -2.8, "sha256": "b" * 64},
        ],
        "safety_audits": [{"name": "autoeq_audit_validation", "passed": True}],
    }
    config = offline_abx_gate.ABXGateConfig(require_safety_audits=True)

    result = offline_abx_gate.evaluate_abx_artifact(payload, config)

    assert result["overall_pass"] is False
    assert result["technical_pass"] is False
    assert "safety_audit_missing_provenance" in {failure["code"] for failure in result["failures"]}
    assert result["safety_audits"]["missing_provenance"] == ["autoeq_audit_validation"]


def test_abx_gate_rejects_safety_audit_for_different_render_when_required():
    payload = {
        "seed": 42,
        "clips": [
            {"clip_id": "a", "variant": "A", "lufs": -16.0, "peak_dbfs": -3.0, "sha256": "a" * 64},
            {"clip_id": "b", "variant": "B", "lufs": -15.8, "peak_dbfs": -2.8, "sha256": "b" * 64},
        ],
        "safety_audits": [
            {
                "name": "autoeq_audit_validation",
                "passed": True,
                "target_render_sha256": "c" * 64,
            }
        ],
    }
    config = offline_abx_gate.ABXGateConfig(require_safety_audits=True)

    result = offline_abx_gate.evaluate_abx_artifact(payload, config)

    assert result["overall_pass"] is False
    assert result["technical_pass"] is False
    assert "safety_audit_render_mismatch" in {failure["code"] for failure in result["failures"]}
    assert result["safety_audits"]["mismatched_provenance"] == ["autoeq_audit_validation"]


def test_abx_gate_blocks_failed_or_missing_safety_audit_status():
    payload = {
        "seed": 42,
        "clips": [
            {"clip_id": "a", "variant": "A", "lufs": -16.0, "peak_dbfs": -3.0, "sha256": "a" * 64},
            {"clip_id": "b", "variant": "B", "lufs": -15.8, "peak_dbfs": -2.8, "sha256": "b" * 64},
        ],
        "safety_audits": [
            {"name": "autoeq_audit_validation", "passed": False},
            {"name": "dynamic_eq_playbook"},
        ],
    }

    result = offline_abx_gate.evaluate_abx_artifact(payload)
    codes = {failure["code"] for failure in result["failures"]}

    assert result["overall_pass"] is False
    assert result["technical_pass"] is False
    assert "safety_audit_failed" in codes
    assert "safety_audit_missing_status" in codes


def test_abx_gate_can_require_safety_audit_evidence():
    payload = {
        "seed": 42,
        "clips": [
            {"clip_id": "a", "variant": "A", "lufs": -16.0, "peak_dbfs": -3.0, "sha256": "a" * 64},
            {"clip_id": "b", "variant": "B", "lufs": -15.8, "peak_dbfs": -2.8, "sha256": "b" * 64},
        ],
    }
    config = offline_abx_gate.ABXGateConfig(require_safety_audits=True)

    result = offline_abx_gate.evaluate_abx_artifact(payload, config)

    assert result["overall_pass"] is False
    assert "missing_safety_audit" in {failure["code"] for failure in result["failures"]}


def test_abx_gate_cli_can_require_safety_audit(tmp_path):
    artifact = tmp_path / "artifact.json"
    report_path = tmp_path / "report.json"
    artifact.write_text(
        json.dumps(
            {
                "seed": 42,
                "clips": [
                    {"clip_id": "a", "variant": "A", "lufs": -16.0, "peak_dbfs": -3.0, "sha256": "a" * 64},
                    {"clip_id": "b", "variant": "B", "lufs": -15.8, "peak_dbfs": -2.8, "sha256": "b" * 64},
                ],
                "safety_audits": [
                    {
                        "name": "autoeq_audit_validation",
                        "passed": True,
                        "target_render_sha256": "b" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = offline_abx_gate.main([str(artifact), "--require-safety-audit", "--out", str(report_path)])
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["overall_pass"] is True
    assert report["safety_audits"]["required"] is True
    assert report["safety_audits"]["passed"] == ["autoeq_audit_validation"]
    assert report["safety_audits"]["with_provenance"] == ["autoeq_audit_validation"]


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
