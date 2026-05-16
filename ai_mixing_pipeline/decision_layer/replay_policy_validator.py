"""Replay-only policy and coherence validation for deterministic intelligence graphs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any

from .replay_checkpoint_manifest import ReplayCheckpointManifest
from .replay_executor import ReplayExecutorResult
from .replay_proposal_ranking import compare_replay_graph_checkpoints


REPLAY_POLICY_VALIDATION_SCHEMA_VERSION = "replay_policy_validation/v1"
_SEVERITY_ORDER = {"error": 3, "warning": 2, "info": 1}


@dataclass(frozen=True)
class ReplayValidationFinding:
    severity: str
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReplayPolicyValidationReport:
    schema_version: str
    ok: bool
    highest_severity: str
    findings: list[ReplayValidationFinding]
    manifest_id: str
    selected_proposal_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ok": self.ok,
            "highest_severity": self.highest_severity,
            "findings": [finding.to_dict() for finding in self.findings],
            "manifest_id": self.manifest_id,
            "selected_proposal_id": self.selected_proposal_id,
        }


def _stable_payload(payload: Any) -> Any:
    return json.loads(json.dumps(payload, sort_keys=True))


def _resolve_manifest(payload: ReplayCheckpointManifest | dict[str, Any]) -> ReplayCheckpointManifest:
    return payload if isinstance(payload, ReplayCheckpointManifest) else ReplayCheckpointManifest.from_dict(payload)


def _resolve_executor(payload: ReplayExecutorResult | dict[str, Any] | None) -> ReplayExecutorResult | None:
    if payload is None:
        return None
    return payload if isinstance(payload, ReplayExecutorResult) else ReplayExecutorResult.from_dict(payload)


def validate_replay_policy_consistency(
    manifest: ReplayCheckpointManifest | dict[str, Any],
    *,
    executor_result: ReplayExecutorResult | dict[str, Any] | None = None,
    baseline_manifest: ReplayCheckpointManifest | dict[str, Any] | None = None,
) -> ReplayPolicyValidationReport:
    resolved_manifest = _resolve_manifest(manifest)
    resolved_executor = _resolve_executor(executor_result)
    findings: list[ReplayValidationFinding] = []
    if not resolved_manifest.dry_run_only or resolved_manifest.live_mixer_writes:
        findings.append(
            ReplayValidationFinding(
                severity="error",
                code="live_write_manifest",
                message="Replay manifests must stay dry-run only.",
                details={
                    "dry_run_only": resolved_manifest.dry_run_only,
                    "live_mixer_writes": resolved_manifest.live_mixer_writes,
                },
            )
        )
    graph = resolved_manifest.graph_checkpoint
    timeline = graph.get("timeline_artifact") or {}
    if bool(timeline.get("dry_run_only", True)) is False or bool(timeline.get("live_mixer_writes", False)) is True:
        findings.append(
            ReplayValidationFinding(
                severity="error",
                code="policy_signature_mismatch",
                message="Graph checkpoint timeline is not dry-run safe.",
                details=_stable_payload(timeline),
            )
        )
    if resolved_executor is not None and resolved_executor.selected_proposal_id != graph.get("selected_proposal_id"):
        findings.append(
            ReplayValidationFinding(
                severity="error",
                code="executor_selection_mismatch",
                message="Executor selection drifted from the graph checkpoint.",
                details={
                    "executor_selected_proposal_id": resolved_executor.selected_proposal_id,
                    "graph_selected_proposal_id": graph.get("selected_proposal_id"),
                },
            )
        )
    if baseline_manifest is not None:
        baseline = _resolve_manifest(baseline_manifest)
        comparison = compare_replay_graph_checkpoints(baseline.graph_checkpoint, resolved_manifest.graph_checkpoint)
        if comparison["selected_proposal_changed"]:
            findings.append(
                ReplayValidationFinding(
                    severity="warning",
                    code="selection_drift",
                    message="Replay selection differs from the provided baseline manifest.",
                    details=comparison,
                )
            )
    if not resolved_manifest.replay_correlation_ids:
        findings.append(
            ReplayValidationFinding(
                severity="warning",
                code="missing_replay_correlation_ids",
                message="Replay artifacts should carry correlation ids for traceability.",
                details={},
            )
        )
    if not findings:
        findings.append(
            ReplayValidationFinding(
                severity="info",
                code="replay_safe",
                message="Manifest and executor are dry-run replay safe.",
                details={"manifest_id": resolved_manifest.manifest_id},
            )
        )
    highest = max(findings, key=lambda finding: _SEVERITY_ORDER.get(finding.severity, 0)).severity
    return ReplayPolicyValidationReport(
        schema_version=REPLAY_POLICY_VALIDATION_SCHEMA_VERSION,
        ok=all(finding.severity != "error" for finding in findings),
        highest_severity=highest,
        findings=findings,
        manifest_id=resolved_manifest.manifest_id,
        selected_proposal_id=str(graph.get("selected_proposal_id", "")),
    )
