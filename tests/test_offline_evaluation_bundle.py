from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "offline_evaluation_bundle.py"
SPEC = importlib.util.spec_from_file_location("offline_evaluation_bundle", MODULE_PATH)
offline_evaluation_bundle = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = offline_evaluation_bundle
SPEC.loader.exec_module(offline_evaluation_bundle)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, payload: dict) -> str:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return _sha256(text.encode("utf-8"))


def _write_bundle_fixture(tmp_path: Path, *, audit_target: str | None = None, gate_source: str | None = None):
    candidate_bytes = b"candidate render fixture\n"
    candidate_sha = _sha256(candidate_bytes)
    candidate_path = tmp_path / "candidate.wav"
    candidate_path.write_bytes(candidate_bytes)

    audit_payload = {
        "schema_version": "autoeq-audit-fixture-v1",
        "passed": True,
        "target_render_sha256": audit_target or candidate_sha,
    }
    audit_sha = _write_json(tmp_path / "autoeq_audit.json", audit_payload)

    abx_payload = {
        "schema_version": "automixer-abx-fixture-v1",
        "seed": 42,
        "clips": [
            {"variant": "A", "lufs": -16.1, "peak_dbfs": -2.0, "sha256": "a" * 64},
            {"variant": "B", "lufs": -16.0, "peak_dbfs": -2.1, "sha256": candidate_sha},
        ],
        "safety_audits": [
            {
                "name": "autoeq_audit_validation",
                "passed": True,
                "target_render_sha256": audit_target or candidate_sha,
            }
        ],
    }
    abx_sha = _write_json(tmp_path / "abx_artifact.json", abx_payload)

    gate_payload = {
        "schema_version": "automixer-abx-gate-v1",
        "overall_pass": True,
        "technical_pass": True,
        "offline_only": True,
        "live_write": False,
        "source_artifact_sha256": gate_source or abx_sha,
        "safety_audits": {
            "required": True,
            "with_provenance": ["autoeq_audit_validation"],
        },
    }
    gate_sha = _write_json(tmp_path / "abx_gate_report.json", gate_payload)

    bundle_payload = {
        "schema_version": offline_evaluation_bundle.BUNDLE_SCHEMA_VERSION,
        "candidate_render": {"path": "candidate.wav", "sha256": candidate_sha},
        "safety_audit": {"path": "autoeq_audit.json", "sha256": audit_sha},
        "abx_artifact": {"path": "abx_artifact.json", "sha256": abx_sha},
        "abx_gate_report": {"path": "abx_gate_report.json", "sha256": gate_sha},
        "tool_versions": {
            "autoeq_audit": "fixture",
            "offline_abx_gate": "fixture",
            "offline_evaluation_bundle": "fixture",
        },
    }
    bundle_path = tmp_path / "offline_evaluation_bundle.json"
    _write_json(bundle_path, bundle_payload)
    return bundle_path, bundle_payload


def test_offline_evaluation_bundle_accepts_complete_provenance(tmp_path):
    _bundle_path, bundle = _write_bundle_fixture(tmp_path)

    report = offline_evaluation_bundle.validate_bundle(bundle, base_dir=tmp_path)

    assert report["overall_pass"] is True
    assert report["offline_only"] is True
    assert report["live_write"] is False
    assert report["failures"] == []
    assert report["checksums"]["candidate_render_sha256"] == bundle["candidate_render"]["sha256"]
    assert report["relations"]["safety_audit_targets_candidate_render"] is True
    assert report["relations"]["abx_artifact_targets_candidate_render"] is True
    assert report["relations"]["gate_report_targets_abx_artifact"] is True


def test_offline_evaluation_bundle_rejects_audit_for_different_render(tmp_path):
    _bundle_path, bundle = _write_bundle_fixture(tmp_path, audit_target="c" * 64)

    report = offline_evaluation_bundle.validate_bundle(bundle, base_dir=tmp_path)
    codes = {failure["code"] for failure in report["failures"]}

    assert report["overall_pass"] is False
    assert "safety_audit_render_mismatch" in codes
    assert "abx_artifact_audit_provenance_missing" in codes


def test_offline_evaluation_bundle_rejects_stale_gate_report_source(tmp_path):
    _bundle_path, bundle = _write_bundle_fixture(tmp_path, gate_source="d" * 64)

    report = offline_evaluation_bundle.validate_bundle(bundle, base_dir=tmp_path)
    codes = {failure["code"] for failure in report["failures"]}

    assert report["overall_pass"] is False
    assert "abx_gate_report_source_mismatch" in codes
    assert report["relations"]["gate_report_targets_abx_artifact"] is False


def test_offline_evaluation_bundle_cli_writes_report(tmp_path):
    bundle_path, _bundle = _write_bundle_fixture(tmp_path)
    report_path = tmp_path / "bundle_report.json"

    exit_code = offline_evaluation_bundle.main([str(bundle_path), "--out", str(report_path)])
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["overall_pass"] is True
    assert report["schema_version"] == offline_evaluation_bundle.REPORT_SCHEMA_VERSION


def test_offline_evaluation_bundle_source_has_no_forbidden_live_write_calls():
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
        "unmute(",
    ]
    assert not any(token in source for token in forbidden)
