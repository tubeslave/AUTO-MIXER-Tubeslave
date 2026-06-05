"""Validate offline evaluation bundles for Automixer render evidence.

The bundle validator is intentionally file/JSON-only. It does not import mixer
clients, backend servers, OSC transports, or DSP apply modules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


BUNDLE_SCHEMA_VERSION = "automixer-offline-evaluation-bundle-v1"
REPORT_SCHEMA_VERSION = "automixer-offline-evaluation-bundle-report-v1"

CHECKSUM_FIELDS = ("sha256", "checksum", "audio_sha256", "file_sha256")
PASS_FIELDS = ("passed", "overall_pass", "technical_pass")
TARGET_CHECKSUM_FIELDS = (
    "target_render_sha256",
    "candidate_sha256",
    "candidate_audio_sha256",
    "checked_audio_sha256",
    "render_sha256",
    "audio_sha256",
)
SOURCE_ARTIFACT_FIELDS = (
    "source_artifact_sha256",
    "abx_artifact_sha256",
    "input_artifact_sha256",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _section(payload: Mapping[str, Any], *names: str) -> Mapping[str, Any]:
    for name in names:
        value = payload.get(name)
        if isinstance(value, Mapping):
            return value
    return {}


def _first_present(row: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if row.get(name) is not None:
            return row.get(name)
    return None


def _file_reference_path(row: Mapping[str, Any], base_dir: Path) -> Optional[Path]:
    raw_path = row.get("path") or row.get("file") or row.get("artifact")
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = base_dir / path
    return path


def _expected_checksum(row: Mapping[str, Any]) -> Optional[str]:
    value = _first_present(row, CHECKSUM_FIELDS)
    return str(value) if value else None


def _validate_file_reference(
    row: Mapping[str, Any],
    *,
    label: str,
    base_dir: Path,
    failures: list[dict[str, str]],
) -> tuple[Optional[Path], Optional[str]]:
    path = _file_reference_path(row, base_dir)
    expected = _expected_checksum(row)

    if path is None:
        failures.append(
            {
                "code": "missing_file_path",
                "detail": f"{label} must include a repo-local file path",
            }
        )
        return None, None
    if not expected:
        failures.append(
            {
                "code": "missing_file_checksum",
                "detail": f"{label} must include a sha256 checksum",
            }
        )
    if not path.is_file():
        failures.append(
            {
                "code": "missing_file",
                "detail": f"{label} file does not exist: {path}",
            }
        )
        return path, expected

    actual = _sha256_file(path)
    if expected and actual != expected:
        failures.append(
            {
                "code": "file_checksum_mismatch",
                "detail": f"{label} checksum mismatch: {expected} != {actual}",
            }
        )
    return path, actual


def _validate_json_reference(
    row: Mapping[str, Any],
    *,
    label: str,
    base_dir: Path,
    failures: list[dict[str, str]],
) -> tuple[Optional[Mapping[str, Any]], Optional[str]]:
    path, checksum = _validate_file_reference(
        row,
        label=label,
        base_dir=base_dir,
        failures=failures,
    )
    if path is None or not path.is_file():
        return None, checksum
    try:
        return _load_json(path), checksum
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        failures.append(
            {
                "code": "invalid_json_artifact",
                "detail": f"{label} is not a JSON object: {exc}",
            }
        )
        return None, checksum


def _pass_status(row: Mapping[str, Any]) -> Optional[bool]:
    for name in PASS_FIELDS:
        if name not in row:
            continue
        value = row[name]
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"pass", "passed", "true", "ok"}:
                return True
            if normalized in {"fail", "failed", "false", "blocked", "error"}:
                return False
        return None
    return None


def _checksum_values(row: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for name in CHECKSUM_FIELDS:
        value = row.get(name)
        if value:
            values.append(str(value))
    return values


def _target_checksum(row: Mapping[str, Any]) -> Optional[str]:
    value = _first_present(row, TARGET_CHECKSUM_FIELDS)
    if value:
        return str(value)

    for name in ("target", "candidate", "render", "subject"):
        nested = row.get(name)
        if isinstance(nested, Mapping):
            checksums = _checksum_values(nested)
            if checksums:
                return checksums[0]
    return None


def _normalize_variant(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"b", "after", "candidate", "processed", "test"}:
        return "B"
    if text in {"a", "before", "original", "reference", "baseline"}:
        return "A"
    return None


def _clip_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for name in ("clips", "clip_metrics"):
        value = payload.get(name)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, Mapping)]
    return []


def _audit_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for name in ("safety_audits", "audit_reports", "audits"):
        value = payload.get(name)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, Mapping)]
        if isinstance(value, Mapping):
            return [value]
    return []


def _candidate_render_checksums(payload: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for row in _clip_rows(payload):
        variant = _normalize_variant(_first_present(row, ("variant", "side", "label", "version")))
        if variant == "B":
            values.extend(_checksum_values(row))
    return values


def validate_bundle(
    payload: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Return a deterministic validation report for one offline bundle."""

    root = (base_dir or Path.cwd()).resolve()
    failures: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    if payload.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        failures.append(
            {
                "code": "unsupported_schema_version",
                "detail": f"schema_version must be {BUNDLE_SCHEMA_VERSION}",
            }
        )

    render_row = _section(payload, "candidate_render", "render")
    audit_row = _section(payload, "safety_audit", "safety_report", "audit")
    artifact_row = _section(payload, "abx_artifact")
    report_row = _section(payload, "abx_gate_report", "gate_report")

    if not render_row:
        failures.append({"code": "missing_bundle_section", "detail": "candidate_render is required"})
    if not audit_row:
        failures.append({"code": "missing_bundle_section", "detail": "safety_audit is required"})
    if not artifact_row:
        failures.append({"code": "missing_bundle_section", "detail": "abx_artifact is required"})
    if not report_row:
        failures.append({"code": "missing_bundle_section", "detail": "abx_gate_report is required"})

    _render_path, render_checksum = _validate_file_reference(
        render_row,
        label="candidate_render",
        base_dir=root,
        failures=failures,
    ) if render_row else (None, None)
    audit_payload, audit_checksum = _validate_json_reference(
        audit_row,
        label="safety_audit",
        base_dir=root,
        failures=failures,
    ) if audit_row else (None, None)
    abx_payload, abx_checksum = _validate_json_reference(
        artifact_row,
        label="abx_artifact",
        base_dir=root,
        failures=failures,
    ) if artifact_row else (None, None)
    gate_payload, gate_checksum = _validate_json_reference(
        report_row,
        label="abx_gate_report",
        base_dir=root,
        failures=failures,
    ) if report_row else (None, None)

    if audit_payload is not None:
        status = _pass_status(audit_payload)
        if status is not True:
            failures.append(
                {
                    "code": "safety_audit_not_passed",
                    "detail": "safety_audit must carry a passing status",
                }
            )
        audit_target = _target_checksum(audit_payload)
        if not audit_target:
            failures.append(
                {
                    "code": "safety_audit_missing_render_checksum",
                    "detail": "safety_audit must name the target render checksum",
                }
            )
        elif render_checksum and audit_target != render_checksum:
            failures.append(
                {
                    "code": "safety_audit_render_mismatch",
                    "detail": "safety_audit target render checksum does not match candidate_render",
                }
            )

    if abx_payload is not None and render_checksum:
        candidate_checksums = set(_candidate_render_checksums(abx_payload))
        if render_checksum not in candidate_checksums:
            failures.append(
                {
                    "code": "abx_artifact_candidate_mismatch",
                    "detail": "abx_artifact B/candidate checksum does not match candidate_render",
                }
            )
        attached_audit_targets = {
            target
            for row in _audit_rows(abx_payload)
            if (target := _target_checksum(row))
        }
        if render_checksum not in attached_audit_targets:
            failures.append(
                {
                    "code": "abx_artifact_audit_provenance_missing",
                    "detail": "abx_artifact must attach safety audit provenance for candidate_render",
                }
            )

    if gate_payload is not None:
        if gate_payload.get("overall_pass") is not True or gate_payload.get("technical_pass") is not True:
            failures.append(
                {
                    "code": "abx_gate_report_not_passed",
                    "detail": "abx_gate_report must pass overall and technical gates",
                }
            )
        if gate_payload.get("offline_only") is not True:
            failures.append(
                {
                    "code": "abx_gate_report_not_offline_only",
                    "detail": "abx_gate_report must be offline_only=true",
                }
            )
        if gate_payload.get("live_write") is not False:
            failures.append(
                {
                    "code": "abx_gate_report_live_write_not_false",
                    "detail": "abx_gate_report must carry live_write=false",
                }
            )
        source_checksum = _first_present(gate_payload, SOURCE_ARTIFACT_FIELDS)
        if source_checksum is None:
            source_checksum = _first_present(report_row, SOURCE_ARTIFACT_FIELDS)
        if not source_checksum:
            failures.append(
                {
                    "code": "abx_gate_report_missing_source_checksum",
                    "detail": "abx_gate_report must name the ABX artifact checksum it evaluated",
                }
            )
        elif abx_checksum and str(source_checksum) != abx_checksum:
            failures.append(
                {
                    "code": "abx_gate_report_source_mismatch",
                    "detail": "abx_gate_report source checksum does not match abx_artifact",
                }
            )

    tools = payload.get("tool_versions") or payload.get("tools")
    if not isinstance(tools, Mapping) or not tools:
        failures.append(
            {
                "code": "missing_tool_metadata",
                "detail": "bundle must include non-empty tool_versions or tools metadata",
            }
        )
    elif "offline_abx_gate" not in tools:
        warnings.append(
            {
                "code": "missing_offline_abx_gate_version",
                "detail": "tool metadata should include offline_abx_gate",
            }
        )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "overall_pass": not failures,
        "failures": failures,
        "warnings": warnings,
        "checksums": {
            "candidate_render_sha256": render_checksum,
            "safety_audit_sha256": audit_checksum,
            "abx_artifact_sha256": abx_checksum,
            "abx_gate_report_sha256": gate_checksum,
        },
        "relations": {
            "safety_audit_targets_candidate_render": not any(
                item["code"] == "safety_audit_render_mismatch" for item in failures
            ),
            "abx_artifact_targets_candidate_render": not any(
                item["code"] == "abx_artifact_candidate_mismatch" for item in failures
            ),
            "gate_report_targets_abx_artifact": not any(
                item["code"] == "abx_gate_report_source_mismatch" for item in failures
            ),
        },
        "offline_only": True,
        "live_write": False,
    }


def load_bundle(path: Path) -> Mapping[str, Any]:
    return _load_json(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate an offline Automixer evaluation bundle")
    parser.add_argument("bundle_json", type=Path)
    parser.add_argument("--out", type=Path, help="Write validation report JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    bundle_path = args.bundle_json
    report = validate_bundle(load_bundle(bundle_path), base_dir=bundle_path.parent)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if report["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
