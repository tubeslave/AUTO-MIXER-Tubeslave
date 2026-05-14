"""Integrated replay-only rehearsal runner and artifact bundle builder."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from arrangement_automation.live_input_trim_controller import LiveInputTrimController
except ImportError:  # pragma: no cover - package import from repo root.
    from backend.arrangement_automation.live_input_trim_controller import LiveInputTrimController

from ai_mixing_pipeline.reports import write_json, write_shadow_mode_reports

from .replay_checkpoint_manifest import build_replay_checkpoint_manifest
from .replay_executor import build_replay_executor_from_rewind_context, simulate_replay_decisions
from .replay_policy_validator import validate_replay_policy_consistency
from .replay_proposal_ranking import (
    ReplayDecisionProposal,
    build_feedback_mitigation_proposals,
    build_fader_proposals,
    build_gain_proposals,
    build_operator_assistance_proposals,
    build_performer_stage_interaction_proposals,
    build_replay_graph_checkpoint,
    build_room_compensation_proposals,
    build_speech_priority_arbitration_proposals,
    rank_replay_proposals,
)
from .replay_restore_loader import load_replay_rewind_context


REPLAY_REHEARSAL_BUNDLE_SCHEMA_VERSION = "replay_rehearsal_bundle/v1"


def _stable_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stable_payload(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_stable_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_stable_payload(item) for item in value]
    return value


def _default_trim_config() -> dict[str, Any]:
    return {
        "automation": {
            "live_input_trim": {
                "run_background_loop": False,
                "analysis_only_mode": True,
                "live_apply_enabled": False,
                "confirm_live_apply": False,
            }
        }
    }


def _coerce_channels(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    return [int(item) for item in value]


def _coerce_channel_mapping(value: Any) -> dict[int, int]:
    if not isinstance(value, dict):
        return {}
    return {
        int(audio_channel): int(mixer_channel)
        for audio_channel, mixer_channel in value.items()
    }


def _build_trim_snapshot(scenario: dict[str, Any]) -> dict[str, Any]:
    snapshot = scenario.get("trim_snapshot")
    if isinstance(snapshot, dict) and snapshot:
        return _stable_payload(snapshot)

    trim_spec = dict(scenario.get("trim") or {})
    controller = LiveInputTrimController(config=_default_trim_config())
    controller.channels = _coerce_channels(trim_spec.get("channels") or scenario.get("channels") or [])
    if not controller.channels:
        controller.channels = [1]
    controller.channel_mapping = _coerce_channel_mapping(
        trim_spec.get("channel_mapping") or scenario.get("channel_mapping") or {channel: channel for channel in controller.channels}
    )
    controller.last_state = dict(trim_spec.get("last_state") or {})
    event_seq = trim_spec.get("event_seq", scenario.get("event_seq"))
    replay_metadata = {
        "scenario": str(scenario.get("scenario", "replay_rehearsal")),
        **dict(trim_spec.get("replay_metadata") or {}),
    }
    return controller.build_replay_snapshot(
        event_seq=int(event_seq) if event_seq is not None else None,
        replay_metadata=replay_metadata,
    ).to_dict()


def _collect_blocked_reasons_from_validation(validation_payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for finding in validation_payload.get("findings") or ():
        if str((finding or {}).get("severity", "")) == "error":
            code = str((finding or {}).get("code", "")).strip()
            if code and code not in reasons:
                reasons.append(code)
    return reasons


def _apply_critic_scores(
    proposals: list[ReplayDecisionProposal],
    scenario: dict[str, Any],
) -> list[ReplayDecisionProposal]:
    by_proposal = dict(scenario.get("critic_scores_by_proposal_id") or {})
    by_correlation = dict(scenario.get("critic_scores_by_replay_correlation_id") or {})
    enriched: list[ReplayDecisionProposal] = []
    for proposal in proposals:
        critic_scores = {}
        proposal_scores = by_proposal.get(proposal.proposal_id)
        if isinstance(proposal_scores, dict):
            critic_scores.update(proposal_scores)
        correlation_scores = by_correlation.get(proposal.replay_correlation_id)
        if isinstance(correlation_scores, dict):
            critic_scores.update(correlation_scores)
        if critic_scores:
            enriched.append(replace(proposal, critic_scores=_stable_payload(critic_scores)))
        else:
            enriched.append(proposal)
    return enriched


def _build_proposals(scenario: dict[str, Any]) -> list[ReplayDecisionProposal]:
    if isinstance(scenario.get("proposals"), list):
        return [
            item if isinstance(item, ReplayDecisionProposal) else ReplayDecisionProposal.from_dict(dict(item))
            for item in scenario.get("proposals") or ()
        ]

    inputs = dict(scenario.get("proposal_inputs") or {})
    proposals: list[ReplayDecisionProposal] = []
    proposals.extend(build_gain_proposals(recommendations=list(inputs.get("gain_recommendations") or ())))
    proposals.extend(build_fader_proposals(fader_updates=list(inputs.get("fader_updates") or ())))
    proposals.extend(
        build_speech_priority_arbitration_proposals(
            arbitration_payloads=list(inputs.get("speech_priority_arbitration") or ())
        )
    )
    proposals.extend(build_room_compensation_proposals(room_plans=list(inputs.get("room_compensation_plans") or ())))
    proposals.extend(build_feedback_mitigation_proposals(feedback_events=list(inputs.get("feedback_events") or ())))
    proposals.extend(
        build_operator_assistance_proposals(
            operator_recommendations=list(inputs.get("operator_recommendations") or ())
        )
    )
    proposals.extend(
        build_performer_stage_interaction_proposals(
            interaction_events=list(inputs.get("performer_stage_interactions") or ())
        )
    )
    return _apply_critic_scores(proposals, scenario)


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


@dataclass(frozen=True)
class ReplayRehearsalBundle:
    schema_version: str
    bundle_id: str
    scenario: str
    manifest_id: str
    selected_proposal_id: str | None
    replay_correlation_ids: list[str]
    stage_transitions: list[dict[str, Any]]
    governance: dict[str, Any]
    readiness: dict[str, Any]
    blocked_reasons: list[str]
    inventory: dict[str, Any]
    artifacts: dict[str, Any]
    manifest: dict[str, Any]
    rewind_context: dict[str, Any]
    executor: dict[str, Any]
    validation_report: dict[str, Any]
    shadow_reports: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "scenario": self.scenario,
            "manifest_id": self.manifest_id,
            "selected_proposal_id": self.selected_proposal_id,
            "replay_correlation_ids": list(self.replay_correlation_ids),
            "stage_transitions": _stable_payload(self.stage_transitions),
            "governance": _stable_payload(self.governance),
            "readiness": _stable_payload(self.readiness),
            "blocked_reasons": list(self.blocked_reasons),
            "inventory": _stable_payload(self.inventory),
            "artifacts": _stable_payload(self.artifacts),
            "manifest": _stable_payload(self.manifest),
            "rewind_context": _stable_payload(self.rewind_context),
            "executor": _stable_payload(self.executor),
            "validation_report": _stable_payload(self.validation_report),
            "shadow_reports": _stable_payload(self.shadow_reports),
            "metadata": _stable_payload(self.metadata),
        }


def run_replay_rehearsal(
    scenario: dict[str, Any] | str | Path,
    *,
    output_dir: str | Path,
) -> ReplayRehearsalBundle:
    scenario_payload = (
        json.loads(Path(scenario).read_text(encoding="utf-8"))
        if isinstance(scenario, (str, Path))
        else dict(scenario)
    )
    out_root = Path(output_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    stage_transitions: list[dict[str, Any]] = []

    def stage(name: str, **details: Any) -> None:
        stage_transitions.append(
            {
                "seq": len(stage_transitions) + 1,
                "stage": name,
                "details": _stable_payload(details),
            }
        )

    stage("scenario_loaded", scenario=scenario_payload.get("scenario", "replay_rehearsal"))

    trim_snapshot = _build_trim_snapshot(scenario_payload)
    stage(
        "trim_snapshot_built",
        event_seq=trim_snapshot.get("replay_metadata", {}).get("event_seq"),
        channel_count=len(trim_snapshot.get("channels") or ()),
    )

    proposals = _build_proposals(scenario_payload)
    stage("proposal_inventory_built", proposal_count=len(proposals))

    ranking = rank_replay_proposals(
        proposals,
        config=dict(scenario_payload.get("ranking_config") or {}),
        family_priority=dict(scenario_payload.get("family_priority") or {}),
    )
    graph = build_replay_graph_checkpoint(
        proposals,
        ranking,
        metadata={
            "scenario": str(scenario_payload.get("scenario", "replay_rehearsal")),
            "event_seq": scenario_payload.get("event_seq"),
            **dict(scenario_payload.get("graph_metadata") or {}),
        },
    )
    stage("proposal_ranking_complete", selected_proposal_id=ranking.selected_proposal_id)

    manifest = build_replay_checkpoint_manifest(
        trim_snapshot=trim_snapshot,
        graph_checkpoint=graph.to_dict(),
        critic_artifacts=dict(scenario_payload.get("critic_artifacts") or {}),
        metadata={
            "scenario": str(scenario_payload.get("scenario", "replay_rehearsal")),
            **dict(scenario_payload.get("manifest_metadata") or {}),
        },
    )
    manifest_payload = manifest.to_dict()
    stage("manifest_built", manifest_id=manifest.manifest_id)

    rewind_context = load_replay_rewind_context(manifest_payload)
    rewind_payload = rewind_context.to_dict()
    stage("rewind_context_restored", manifest_id=rewind_context.manifest_id)

    executor = simulate_replay_decisions(
        proposals,
        selected_proposal_id=ranking.selected_proposal_id,
        initial_state=build_replay_executor_from_rewind_context(rewind_payload),
    )
    executor_payload = executor.to_dict()
    stage("executor_simulated", trace_signature=executor.trace_signature)

    validation = validate_replay_policy_consistency(
        manifest_payload,
        executor_result=executor,
    )
    validation_payload = validation.to_dict()
    stage("policy_validated", valid=validation.valid)

    shadow_dir = out_root / "shadow_reports"
    shadow_report_paths = write_shadow_mode_reports(
        shadow_dir,
        manifest=manifest_payload,
        executor=executor,
        validation_report=validation_payload,
        operator_family=str(scenario_payload.get("operator_family", "operator_assistance")),
    )
    readiness_path = Path(shadow_report_paths["promotion_readiness"])
    readiness_payload = json.loads(readiness_path.read_text(encoding="utf-8"))
    comparison_payload = json.loads(Path(shadow_report_paths["shadow_mode_comparison"]).read_text(encoding="utf-8"))
    stage("shadow_reports_written", ready=readiness_payload.get("ready", False))

    blocked_reasons: list[str] = []
    blocked_reasons.extend(_collect_blocked_reasons_from_validation(validation_payload))
    for reason in readiness_payload.get("blocking_reasons") or ():
        text = str(reason).strip()
        if text and text not in blocked_reasons:
            blocked_reasons.append(text)

    governance = {
        "dry_run_only": bool(manifest.dry_run_only),
        "live_mixer_writes": bool(manifest.live_mixer_writes),
        "executor_dry_run_only": bool(executor.dry_run_only),
        "validation_valid": bool(validation.valid),
        "transport_mutation_allowed": False,
    }
    inventory = {
        "proposal_count": len(proposals),
        "selected_proposal_id": ranking.selected_proposal_id,
        "timeline_event_count": int((graph.timeline_artifact or {}).get("timeline_event_count", 0)),
        "executor_event_count": len(executor.events),
        "replay_correlation_count": len(manifest.replay_correlation_ids),
        "blocked_reason_count": len(blocked_reasons),
        "operator_disagreement_count": len(comparison_payload.get("disagreements") or ()),
    }

    artifact_paths = {
        "manifest": out_root / "replay_checkpoint_manifest.json",
        "rewind_context": out_root / "replay_rewind_context.json",
        "executor_result": out_root / "replay_executor_result.json",
        "validation_report": out_root / "replay_policy_validation_report.json",
        "graph_checkpoint": out_root / "replay_graph_checkpoint.json",
    }
    write_json(artifact_paths["manifest"], manifest_payload)
    write_json(artifact_paths["rewind_context"], rewind_payload)
    write_json(artifact_paths["executor_result"], executor_payload)
    write_json(artifact_paths["validation_report"], validation_payload)
    write_json(artifact_paths["graph_checkpoint"], graph.to_dict())

    shadow_reports = {
        name: _relative(Path(path), out_root)
        for name, path in shadow_report_paths.items()
    }
    artifacts = {
        key: _relative(path, out_root)
        for key, path in artifact_paths.items()
    }
    artifacts["shadow_reports_dir"] = _relative(shadow_dir, out_root)
    artifacts["shadow_reports"] = shadow_reports

    core_payload = {
        "schema_version": REPLAY_REHEARSAL_BUNDLE_SCHEMA_VERSION,
        "scenario": str(scenario_payload.get("scenario", "replay_rehearsal")),
        "manifest_id": manifest.manifest_id,
        "selected_proposal_id": ranking.selected_proposal_id,
        "replay_correlation_ids": list(manifest.replay_correlation_ids),
        "governance": governance,
        "readiness": {
            "ready": bool(readiness_payload.get("ready", False)),
            "promotion_readiness_score": int(readiness_payload.get("promotion_readiness_score", 0) or 0),
            "blocking_reasons": list(readiness_payload.get("blocking_reasons") or ()),
        },
        "blocked_reasons": blocked_reasons,
        "inventory": inventory,
        "shadow_reports": shadow_reports,
        "metadata": dict(scenario_payload.get("bundle_metadata") or {}),
    }
    bundle_digest = hashlib.sha256(
        json.dumps(_stable_payload(core_payload), sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    bundle = ReplayRehearsalBundle(
        schema_version=REPLAY_REHEARSAL_BUNDLE_SCHEMA_VERSION,
        bundle_id=f"replay_rehearsal::{bundle_digest[:16]}",
        scenario=str(scenario_payload.get("scenario", "replay_rehearsal")),
        manifest_id=manifest.manifest_id,
        selected_proposal_id=ranking.selected_proposal_id,
        replay_correlation_ids=list(manifest.replay_correlation_ids),
        stage_transitions=stage_transitions,
        governance=governance,
        readiness=readiness_payload,
        blocked_reasons=blocked_reasons,
        inventory=inventory,
        artifacts=artifacts,
        manifest=manifest_payload,
        rewind_context=rewind_payload,
        executor=executor_payload,
        validation_report=validation_payload,
        shadow_reports={
            "comparison": comparison_payload,
            "readiness": readiness_payload,
        },
        metadata=dict(scenario_payload.get("bundle_metadata") or {}),
    )
    write_json(out_root / "replay_rehearsal_bundle.json", bundle.to_dict())
    return bundle
