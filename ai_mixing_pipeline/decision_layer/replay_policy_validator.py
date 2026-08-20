"""Replay-only policy and coherence validation for deterministic intelligence graphs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .replay_checkpoint_manifest import ReplayCheckpointManifest
from .replay_executor import ReplayExecutorResult
from .replay_proposal_ranking import (
    ProposalRankResult,
    ReplayDecisionProposal,
    ReplayGraphCheckpoint,
    compare_replay_graph_checkpoints,
    validate_feedback_mitigation_values,
    validate_fader_adjustment_values,
    validate_gain_adjustment_values,
)
from .replay_restore_loader import load_replay_rewind_context


REPLAY_POLICY_VALIDATION_SCHEMA_VERSION = "replay_policy_validation/v1"

_SEVERITY_ORDER = {
    "error": 0,
    "warning": 1,
    "info": 2,
}


@dataclass(frozen=True)
class ReplayValidationFinding:
    code: str
    severity: str
    message: str
    stage: str
    proposal_id: str = ""
    replay_correlation_id: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "stage": self.stage,
            "proposal_id": self.proposal_id,
            "replay_correlation_id": self.replay_correlation_id,
            "details": _stable_payload(self.details),
        }


@dataclass(frozen=True)
class ReplayPolicyValidationReport:
    schema_version: str
    manifest_id: str
    valid: bool
    dry_run_only: bool
    live_mixer_writes: bool
    findings: list[ReplayValidationFinding]
    summary: dict[str, Any] = field(default_factory=dict)
    drift_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "valid": self.valid,
            "dry_run_only": self.dry_run_only,
            "live_mixer_writes": self.live_mixer_writes,
            "summary": _stable_payload(self.summary),
            "drift_summary": _stable_payload(self.drift_summary),
            "findings": [finding.to_dict() for finding in self.findings],
        }


def _stable_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stable_payload(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_stable_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_stable_payload(item) for item in value]
    return value


def _event_seq(payload: dict[str, Any]) -> int | None:
    value = payload.get("event_seq")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_manifest(manifest: ReplayCheckpointManifest | dict[str, Any]) -> ReplayCheckpointManifest:
    return manifest if isinstance(manifest, ReplayCheckpointManifest) else ReplayCheckpointManifest.from_dict(dict(manifest))


def _resolve_executor(executor_result: ReplayExecutorResult | dict[str, Any] | None) -> ReplayExecutorResult | None:
    if executor_result is None or isinstance(executor_result, ReplayExecutorResult):
        return executor_result
    payload = dict(executor_result)
    raw_dry_run = payload.get("dry_run_only", True)
    if type(raw_dry_run) is not bool:
        raise ValueError("executor dry_run_only must be a JSON boolean")
    return ReplayExecutorResult(
        dry_run_only=raw_dry_run,
        proposals_executed=[str(item) for item in payload.get("proposals_executed") or ()],
        selected_proposal_id=payload.get("selected_proposal_id"),
        events=[] if payload.get("events") is None else [
            item if hasattr(item, "to_dict") else item for item in payload.get("events") or ()
        ],
        rollback_state=dict(payload.get("rollback_state") or {}),
        restored_state=dict(payload.get("restored_state") or {}) if payload.get("restored_state") is not None else None,
        state_snapshots=[dict(item) for item in payload.get("state_snapshots") or ()],
        trace_signature=str(payload.get("trace_signature", "")),
    )


def _executor_event_payloads(executor_result: ReplayExecutorResult | None) -> list[dict[str, Any]]:
    if executor_result is None:
        return []
    events: list[dict[str, Any]] = []
    for item in executor_result.events:
        if hasattr(item, "to_dict"):
            events.append(item.to_dict())
        elif isinstance(item, dict):
            events.append(dict(item))
    return events


def _ranking_payload(ranking: ProposalRankResult) -> list[dict[str, Any]]:
    return [dict(item) for item in ranking.ranked_proposals]


def _sha256_json(payload: Any, *, compact: bool = False) -> str:
    kwargs: dict[str, Any] = {"sort_keys": True, "ensure_ascii": True}
    if compact:
        kwargs["separators"] = (",", ":")
    return hashlib.sha256(json.dumps(payload, **kwargs).encode("utf-8")).hexdigest()


def validate_replay_policy_consistency(
    manifest: ReplayCheckpointManifest | dict[str, Any],
    *,
    executor_result: ReplayExecutorResult | dict[str, Any] | None = None,
    baseline_manifest: ReplayCheckpointManifest | dict[str, Any] | None = None,
) -> ReplayPolicyValidationReport:
    resolved_manifest = _resolve_manifest(manifest)
    graph_checkpoint = ReplayGraphCheckpoint.from_dict(dict(resolved_manifest.graph_checkpoint))
    proposals = list(graph_checkpoint.proposals)
    ranking = graph_checkpoint.ranking
    timeline_artifact = dict(graph_checkpoint.timeline_artifact or {})
    executor = _resolve_executor(executor_result)
    findings: list[ReplayValidationFinding] = []

    def add_finding(
        code: str,
        severity: str,
        message: str,
        stage: str,
        *,
        proposal_id: str = "",
        replay_correlation_id: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        findings.append(
            ReplayValidationFinding(
                code=code,
                severity=severity,
                message=message,
                stage=stage,
                proposal_id=proposal_id,
                replay_correlation_id=replay_correlation_id,
                details=dict(details or {}),
            )
        )

    trim_snapshot = dict(resolved_manifest.trim_snapshot or {})
    trim_transport = dict(trim_snapshot.get("transport_policy") or {})
    graph_metadata = dict(graph_checkpoint.replay_metadata or {})
    trim_metadata = dict(trim_snapshot.get("replay_metadata") or {})

    if not resolved_manifest.manifest_id.strip():
        add_finding(
            "missing_manifest_id",
            "error",
            "Replay manifest is missing its integrity identifier.",
            "policy",
        )
    if not trim_snapshot or not resolved_manifest.graph_checkpoint:
        add_finding(
            "empty_replay_manifest",
            "error",
            "Replay manifest must contain non-empty trim and graph checkpoints.",
            "policy",
        )

    manifest_payload = resolved_manifest.to_dict()
    claimed_manifest_id = str(manifest_payload.pop("manifest_id", ""))
    expected_manifest_id = "replay_manifest::" + hashlib.sha256(
        json.dumps(
            manifest_payload,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    if claimed_manifest_id and claimed_manifest_id != expected_manifest_id:
        add_finding(
            "manifest_integrity_mismatch",
            "error",
            "Replay manifest identifier does not match its canonical contents.",
            "policy",
            details={"expected_manifest_id": expected_manifest_id},
        )

    graph_core = {
        "version": graph_checkpoint.version,
        "proposals": [proposal.to_dict() for proposal in graph_checkpoint.proposals],
        "ranking": graph_checkpoint.ranking.to_dict(),
        "timeline_artifact": graph_checkpoint.timeline_artifact,
        "replay_metadata": graph_checkpoint.replay_metadata,
    }
    expected_checkpoint_id = f"replay_graph::{_sha256_json(graph_core, compact=True)[:16]}"
    if graph_checkpoint.checkpoint_id != expected_checkpoint_id:
        add_finding(
            "graph_checkpoint_integrity_mismatch",
            "error",
            "Replay graph checkpoint identifier does not match canonical contents.",
            "policy",
            details={"expected_checkpoint_id": expected_checkpoint_id},
        )

    if not resolved_manifest.dry_run_only:
        add_finding(
            "replay_policy_not_dry_run",
            "error",
            "Replay manifest is not marked dry-run-only.",
            "policy",
        )
    if resolved_manifest.live_mixer_writes:
        add_finding(
            "replay_policy_live_writes_present",
            "error",
            "Replay manifest reports live mixer writes.",
            "policy",
        )
    if type(trim_transport.get("dry_run_only")) is not bool:
        add_finding(
            "trim_transport_dry_run_marker_missing_or_invalid",
            "error",
            "Trim transport requires explicit boolean dry-run evidence.",
            "replay_state",
        )
    elif trim_transport["dry_run_only"] != resolved_manifest.dry_run_only:
        add_finding(
            "trim_transport_policy_mismatch",
            "error",
            "Trim snapshot transport policy disagrees with manifest dry-run state.",
            "replay_state",
            details={
                "trim_transport_dry_run_only": bool(trim_transport.get("dry_run_only", True)),
                "manifest_dry_run_only": bool(resolved_manifest.dry_run_only),
            },
        )
    if type(timeline_artifact.get("dry_run_only")) is not bool:
        add_finding(
            "timeline_dry_run_marker_missing_or_invalid",
            "error",
            "Replay timeline requires explicit boolean dry-run evidence.",
            "decision",
        )
    elif timeline_artifact["dry_run_only"] != resolved_manifest.dry_run_only:
        add_finding(
            "timeline_policy_mismatch",
            "error",
            "Replay timeline dry-run flag disagrees with manifest dry-run state.",
            "decision",
            details={
                "timeline_dry_run_only": bool(timeline_artifact.get("dry_run_only", True)),
                "manifest_dry_run_only": bool(resolved_manifest.dry_run_only),
            },
        )
    if type(timeline_artifact.get("live_mixer_writes")) is not bool:
        add_finding(
            "timeline_live_write_marker_missing_or_invalid",
            "error",
            "Replay timeline requires an explicit boolean live-write marker.",
            "decision",
        )
    elif timeline_artifact["live_mixer_writes"] != resolved_manifest.live_mixer_writes:
        add_finding(
            "timeline_live_write_mismatch",
            "error",
            "Replay timeline live-write flag disagrees with manifest live-write state.",
            "decision",
            details={
                "timeline_live_mixer_writes": bool(timeline_artifact.get("live_mixer_writes", False)),
                "manifest_live_mixer_writes": bool(resolved_manifest.live_mixer_writes),
            },
        )

    manifest_event_seq = resolved_manifest.event_seq
    trim_event_seq = _event_seq(trim_metadata)
    graph_event_seq = _event_seq(graph_metadata)
    seen_event_seq = {value for value in (manifest_event_seq, trim_event_seq, graph_event_seq) if value is not None}
    if len(seen_event_seq) > 1:
        add_finding(
            "replay_event_seq_mismatch",
            "error",
            "Replay event sequence differs across manifest, trim snapshot, and graph checkpoint.",
            "replay_state",
            details={
                "manifest_event_seq": manifest_event_seq,
                "trim_event_seq": trim_event_seq,
                "graph_event_seq": graph_event_seq,
            },
        )

    proposal_ids: set[str] = set()
    proposal_by_id: dict[str, ReplayDecisionProposal] = {}
    correlation_ids: set[str] = set()
    for proposal in proposals:
        if proposal.proposal_id in proposal_ids:
            add_finding(
                "duplicate_proposal_id",
                "error",
                "Duplicate replay proposal id detected.",
                "analyzer",
                proposal_id=proposal.proposal_id,
            )
        proposal_ids.add(proposal.proposal_id)
        proposal_by_id[proposal.proposal_id] = proposal
        if not proposal.replay_correlation_id:
            add_finding(
                "missing_replay_correlation_id",
                "error",
                "Replay proposal is missing replay_correlation_id.",
                "analyzer",
                proposal_id=proposal.proposal_id,
            )
        else:
            correlation_ids.add(proposal.replay_correlation_id)
        if not proposal.dry_run_only:
            add_finding(
                "proposal_not_dry_run",
                "error",
                "Replay proposal is not marked dry-run-only.",
                "analyzer",
                proposal_id=proposal.proposal_id,
                replay_correlation_id=proposal.replay_correlation_id,
            )
        if proposal.family == "feedback_mitigation":
            try:
                validate_feedback_mitigation_values(
                    channel=proposal.target.get("channel"),
                    action=proposal.action_type,
                    frequency_hz=proposal.target.get("frequency_hz"),
                    gain_db=proposal.requested_state.get("gain_db"),
                    q=proposal.requested_state.get("q"),
                    max_q=proposal.safety.get("max_q"),
                    max_cut_db=proposal.safety.get("max_cut_db"),
                )
            except ValueError as exc:
                add_finding(
                    "invalid_feedback_mitigation_bounds",
                    "error",
                    "Feedback mitigation proposal has malformed or unbounded values.",
                    "analyzer",
                    proposal_id=proposal.proposal_id,
                    replay_correlation_id=proposal.replay_correlation_id,
                    details={"validation_error": str(exc)},
                )
        if proposal.family == "gain_adjustment":
            raw_safety_step = (
                proposal.safety.get("max_step_db")
                if "max_step_db" in proposal.safety
                else proposal.safety.get("max_delta_db")
            )
            try:
                validate_gain_adjustment_values(
                    channel=proposal.target.get("channel"),
                    action=proposal.action_type,
                    current_gain_db=proposal.current_state.get("gain_db"),
                    requested_gain_db=proposal.requested_state.get("gain_db"),
                    max_step_db=raw_safety_step,
                )
            except ValueError as exc:
                add_finding(
                    "invalid_gain_adjustment_bounds",
                    "error",
                    "Gain adjustment proposal has malformed or missing safety bounds.",
                    "analyzer",
                    proposal_id=proposal.proposal_id,
                    replay_correlation_id=proposal.replay_correlation_id,
                    details={"validation_error": str(exc)},
                )
        if proposal.family == "fader_adjustment":
            raw_safety_step = (
                proposal.safety.get("max_step_db")
                if "max_step_db" in proposal.safety
                else proposal.safety.get("max_delta_db")
            )
            try:
                validate_fader_adjustment_values(
                    channel=proposal.target.get("channel"),
                    action=proposal.action_type,
                    current_fader_db=proposal.current_state.get("fader_db"),
                    requested_fader_db=proposal.requested_state.get("fader_db"),
                    max_step_db=raw_safety_step,
                )
            except ValueError as exc:
                add_finding(
                    "invalid_fader_adjustment_bounds",
                    "error",
                    "Fader adjustment proposal has malformed or missing safety bounds.",
                    "analyzer",
                    proposal_id=proposal.proposal_id,
                    replay_correlation_id=proposal.replay_correlation_id,
                    details={"validation_error": str(exc)},
                )

    if sorted(correlation_ids) != sorted(resolved_manifest.replay_correlation_ids):
        add_finding(
            "manifest_correlation_inventory_mismatch",
            "warning",
            "Manifest replay correlation inventory differs from graph proposal inventory.",
            "replay_state",
            details={
                "manifest_replay_correlation_ids": list(resolved_manifest.replay_correlation_ids),
                "graph_replay_correlation_ids": sorted(correlation_ids),
            },
        )

    analyzer_events: dict[str, dict[str, Any]] = {}
    critic_events: dict[str, list[dict[str, Any]]] = {}
    decision_events: dict[str, dict[str, Any]] = {}
    timeline = list(timeline_artifact.get("timeline") or ())
    expected_timeline_seq = list(range(1, len(timeline) + 1))
    actual_timeline_seq = [int(dict(item or {}).get("seq", 0)) for item in timeline]
    if actual_timeline_seq != expected_timeline_seq:
        add_finding(
            "timeline_event_seq_gap",
            "error",
            "Replay timeline event sequence is not contiguous.",
            "decision",
        )
    for item in timeline:
        event = dict(item or {})
        event_signature = str(event.pop("event_signature", ""))
        event.pop("seq", None)
        if event_signature != _sha256_json(event):
            add_finding(
                "timeline_event_integrity_mismatch",
                "error",
                "Replay timeline event signature does not match its contents.",
                str(event.get("stage", "decision")),
                proposal_id=str(event.get("proposal_id", "")),
            )
        proposal_id = str(event.get("proposal_id", ""))
        stage = str(event.get("stage", ""))
        if stage == "analyzer":
            analyzer_events[proposal_id] = event
        elif stage == "critic":
            critic_events.setdefault(proposal_id, []).append(event)
        elif stage == "decision":
            decision_events[proposal_id] = event

    for proposal in proposals:
        analyzer_event = analyzer_events.get(proposal.proposal_id)
        if analyzer_event is None:
            add_finding(
                "missing_analyzer_event",
                "error",
                "Replay timeline is missing analyzer event for proposal.",
                "analyzer",
                proposal_id=proposal.proposal_id,
                replay_correlation_id=proposal.replay_correlation_id,
            )
        else:
            if str(analyzer_event.get("replay_correlation_id", "")) != proposal.replay_correlation_id:
                add_finding(
                    "analyzer_correlation_mismatch",
                    "error",
                    "Analyzer event replay_correlation_id does not match proposal.",
                    "analyzer",
                    proposal_id=proposal.proposal_id,
                    replay_correlation_id=proposal.replay_correlation_id,
                )
            if str(analyzer_event.get("replay_signature", "")) != proposal.replay_signature:
                add_finding(
                    "analyzer_signature_mismatch",
                    "error",
                    "Analyzer event replay signature does not match proposal.",
                    "analyzer",
                    proposal_id=proposal.proposal_id,
                    replay_correlation_id=proposal.replay_correlation_id,
                )
        expected_critics = sorted(proposal.critic_scores)
        actual_critics = sorted(str(item.get("critic_name", "")) for item in critic_events.get(proposal.proposal_id, []))
        if expected_critics != actual_critics:
            add_finding(
                "critic_inventory_mismatch",
                "error",
                "Critic events do not match proposal critic inventory.",
                "critic",
                proposal_id=proposal.proposal_id,
                replay_correlation_id=proposal.replay_correlation_id,
                details={
                    "expected_critics": expected_critics,
                    "actual_critics": actual_critics,
                },
            )
        for critic_event in critic_events.get(proposal.proposal_id, []):
            if str(critic_event.get("replay_signature", "")) != proposal.replay_signature:
                add_finding(
                    "critic_signature_mismatch",
                    "error",
                    "Critic event replay signature does not match proposal.",
                    "critic",
                    proposal_id=proposal.proposal_id,
                    replay_correlation_id=proposal.replay_correlation_id,
                    details={"critic_name": critic_event.get("critic_name")},
                )

    ranked_proposals = _ranking_payload(ranking)
    expected_run_signature = _sha256_json(
        {
            "count": len(ranked_proposals),
            "run_input_signature": float(len(ranked_proposals)),
            "ranked": [
                (item.get("proposal_id"), item.get("confidence_weighted_score"))
                for item in ranked_proposals
            ],
        }
    )
    if ranking.run_signature != expected_run_signature:
        add_finding(
            "ranking_integrity_mismatch",
            "error",
            "Replay ranking signature does not match ranked contents.",
            "decision",
        )
    artifact_core = dict(timeline_artifact)
    artifact_signature = str(artifact_core.pop("artifact_signature", ""))
    if artifact_signature != _sha256_json(artifact_core):
        add_finding(
            "timeline_artifact_integrity_mismatch",
            "error",
            "Replay timeline artifact signature does not match its contents.",
            "decision",
        )
    selected_ranked = ranked_proposals[0]["proposal_id"] if ranked_proposals else None
    if selected_ranked != ranking.selected_proposal_id:
        add_finding(
            "ranking_selected_proposal_mismatch",
            "error",
            "Ranking selected proposal id is inconsistent with the highest-ranked proposal.",
            "decision",
            details={
                "selected_proposal_id": ranking.selected_proposal_id,
                "top_ranked_proposal_id": selected_ranked,
            },
        )

    expected_ranks = list(range(1, len(ranked_proposals) + 1))
    actual_ranks = [int(item.get("rank", 0)) for item in ranked_proposals]
    if actual_ranks != expected_ranks:
        add_finding(
            "ranking_order_gap",
            "error",
            "Ranking contains non-sequential rank values.",
            "decision",
            details={"actual_ranks": actual_ranks, "expected_ranks": expected_ranks},
        )

    for ranked in ranked_proposals:
        proposal_id = str(ranked.get("proposal_id", ""))
        proposal = proposal_by_id.get(proposal_id)
        event = decision_events.get(proposal_id)
        if proposal is None:
            add_finding(
                "ranking_unknown_proposal",
                "error",
                "Ranking references a proposal that is not present in the graph checkpoint.",
                "decision",
                proposal_id=proposal_id,
            )
            continue
        if str(ranked.get("replay_signature", "")) != proposal.replay_signature:
            add_finding(
                "ranking_proposal_signature_mismatch",
                "error",
                "Ranked proposal signature does not match canonical proposal contents.",
                "decision",
                proposal_id=proposal_id,
            )
        if event is None:
            add_finding(
                "missing_decision_event",
                "error",
                "Replay timeline is missing decision event for ranked proposal.",
                "decision",
                proposal_id=proposal_id,
                replay_correlation_id=proposal.replay_correlation_id,
            )
            continue
        if str(event.get("replay_correlation_id", "")) != proposal.replay_correlation_id:
            add_finding(
                "decision_correlation_mismatch",
                "error",
                "Decision event replay_correlation_id does not match proposal.",
                "decision",
                proposal_id=proposal_id,
                replay_correlation_id=proposal.replay_correlation_id,
            )
        if str(event.get("replay_signature", "")) != proposal.replay_signature:
            add_finding(
                "decision_signature_mismatch",
                "error",
                "Decision event replay signature does not match proposal.",
                "decision",
                proposal_id=proposal_id,
                replay_correlation_id=proposal.replay_correlation_id,
            )
        if int(event.get("rank", 0)) != int(ranked.get("rank", 0)):
            add_finding(
                "decision_rank_mismatch",
                "error",
                "Decision event rank does not match ranking payload.",
                "decision",
                proposal_id=proposal_id,
                replay_correlation_id=proposal.replay_correlation_id,
            )

    if executor is not None:
        if not executor.dry_run_only:
            add_finding(
                "executor_not_dry_run",
                "error",
                "Replay executor result is not marked dry-run-only.",
                "executor",
            )
        if executor.selected_proposal_id and executor.selected_proposal_id != ranking.selected_proposal_id:
            add_finding(
                "executor_selected_proposal_mismatch",
                "error",
                "Replay executor selected proposal does not match ranking selection.",
                "executor",
                proposal_id=str(executor.selected_proposal_id),
                details={
                    "executor_selected_proposal_id": executor.selected_proposal_id,
                    "ranking_selected_proposal_id": ranking.selected_proposal_id,
                },
            )
        executor_events = _executor_event_payloads(executor)
        expected_seq = list(range(1, len(executor_events) + 1))
        actual_seq = [int(item.get("seq", 0)) for item in executor_events]
        if actual_seq != expected_seq:
            add_finding(
                "executor_event_seq_gap",
                "error",
                "Replay executor events are not sequential.",
                "executor",
                details={"actual_seq": actual_seq, "expected_seq": expected_seq},
            )
        previous_post_signature: str | None = None
        for event in executor_events:
            calculated_pre = _sha256_json(event.get("pre_state") or {})
            calculated_post = _sha256_json(event.get("post_state") or {})
            if event.get("pre_signature") != calculated_pre or event.get("post_signature") != calculated_post:
                add_finding(
                    "executor_state_signature_mismatch",
                    "error",
                    "Replay executor state signature does not match event state.",
                    "executor",
                    proposal_id=str(event.get("proposal_id", "")),
                )
            if previous_post_signature is not None and event.get("pre_signature") != previous_post_signature:
                add_finding(
                    "executor_state_chain_broken",
                    "error",
                    "Replay executor state transition chain is discontinuous.",
                    "executor",
                    proposal_id=str(event.get("proposal_id", "")),
                )
            previous_post_signature = str(event.get("post_signature", ""))
            proposal_id = str(event.get("proposal_id", ""))
            proposal = proposal_by_id.get(proposal_id)
            if proposal is None:
                add_finding(
                    "executor_unknown_proposal",
                    "error",
                    "Replay executor event references unknown proposal.",
                    "executor",
                    proposal_id=proposal_id,
                )
                continue
            if str(event.get("replay_correlation_id", "")) != proposal.replay_correlation_id:
                add_finding(
                    "executor_correlation_mismatch",
                    "error",
                    "Replay executor event replay_correlation_id does not match proposal.",
                    "executor",
                    proposal_id=proposal_id,
                    replay_correlation_id=proposal.replay_correlation_id,
                )
            if event.get("dry_run") is not True:
                add_finding(
                    "executor_event_not_dry_run",
                    "error",
                    "Replay executor event is not marked dry-run.",
                    "executor",
                    proposal_id=proposal_id,
                    replay_correlation_id=proposal.replay_correlation_id,
                )
            if str(event.get("status", "")) == "blocked" and not str(event.get("blocked_reason", "")).strip():
                add_finding(
                    "executor_blocked_without_reason",
                    "error",
                    "Replay executor blocked an event without a blocked reason.",
                    "executor",
                    proposal_id=proposal_id,
                    replay_correlation_id=proposal.replay_correlation_id,
                )
        expected_trace_signature = _sha256_json(
            {
                "dry_run_only": True,
                "schema_version": "replay_executor/v1",
                "executed": executor_events,
                "state": executor.rollback_state,
            }
        )
        if executor.trace_signature != expected_trace_signature:
            add_finding(
                "executor_trace_integrity_mismatch",
                "error",
                "Replay executor trace signature does not match events and final state.",
                "executor",
            )

    if resolved_manifest.dry_run_only and not resolved_manifest.live_mixer_writes:
        try:
            load_replay_rewind_context(resolved_manifest)
        except Exception as exc:  # pragma: no cover - defensive path
            add_finding(
                "replay_restore_loader_failure",
                "error",
                "Replay manifest could not be restored into a deterministic rewind context.",
                "replay_state",
                details={"error": str(exc)},
            )

    drift_summary: dict[str, Any] = {}
    if baseline_manifest is not None:
        baseline = _resolve_manifest(baseline_manifest)
        baseline_graph = ReplayGraphCheckpoint.from_dict(dict(baseline.graph_checkpoint))
        drift_summary = compare_replay_graph_checkpoints(baseline_graph, graph_checkpoint)
        if baseline.dry_run_only != resolved_manifest.dry_run_only:
            add_finding(
                "drift_dry_run_policy_changed",
                "error",
                "Replay dry-run policy changed relative to baseline manifest.",
                "drift",
                details={
                    "baseline_dry_run_only": baseline.dry_run_only,
                    "candidate_dry_run_only": resolved_manifest.dry_run_only,
                },
            )
        if baseline.live_mixer_writes != resolved_manifest.live_mixer_writes:
            add_finding(
                "drift_live_write_policy_changed",
                "error",
                "Replay live-write policy changed relative to baseline manifest.",
                "drift",
                details={
                    "baseline_live_mixer_writes": baseline.live_mixer_writes,
                    "candidate_live_mixer_writes": resolved_manifest.live_mixer_writes,
                },
            )
        if drift_summary.get("selected_proposal_changed"):
            add_finding(
                "drift_selected_proposal_changed",
                "warning",
                "Replay decision selection changed relative to baseline.",
                "drift",
                proposal_id=str(drift_summary.get("candidate_selected_proposal_id", "")),
                details=drift_summary,
            )
        if drift_summary.get("proposal_ids_added") or drift_summary.get("proposal_ids_removed"):
            add_finding(
                "drift_proposal_inventory_changed",
                "warning",
                "Replay proposal inventory changed relative to baseline.",
                "drift",
                details=drift_summary,
            )
        if not drift_summary.get("same_timeline_signature", True) and not drift_summary.get("selected_proposal_changed"):
            add_finding(
                "drift_timeline_signature_changed",
                "warning",
                "Replay timeline signature changed relative to baseline.",
                "drift",
                details=drift_summary,
            )

    findings = sorted(
        findings,
        key=lambda item: (
            _SEVERITY_ORDER.get(item.severity, 99),
            item.stage,
            item.code,
            item.proposal_id,
            item.replay_correlation_id,
        ),
    )
    severity_counts: dict[str, int] = {}
    for item in findings:
        severity_counts[item.severity] = severity_counts.get(item.severity, 0) + 1

    summary = {
        "proposal_count": len(proposals),
        "timeline_event_count": len(timeline),
        "executor_event_count": len(_executor_event_payloads(executor)),
        "error_count": severity_counts.get("error", 0),
        "warning_count": severity_counts.get("warning", 0),
        "info_count": severity_counts.get("info", 0),
        "selected_proposal_id": ranking.selected_proposal_id,
        "ranking_run_signature": ranking.run_signature,
        "timeline_artifact_signature": timeline_artifact.get("artifact_signature"),
        "executor_trace_signature": executor.trace_signature if executor is not None else "",
    }
    return ReplayPolicyValidationReport(
        schema_version=REPLAY_POLICY_VALIDATION_SCHEMA_VERSION,
        manifest_id=resolved_manifest.manifest_id,
        valid=severity_counts.get("error", 0) == 0,
        dry_run_only=bool(resolved_manifest.dry_run_only),
        live_mixer_writes=bool(resolved_manifest.live_mixer_writes),
        findings=findings,
        summary=summary,
        drift_summary=drift_summary,
    )
