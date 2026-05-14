"""Replay-safe stress and scale simulation helpers for the intelligence graph.

This module is intentionally offline-only. It builds deterministic replay
workloads on top of the existing proposal -> ranking -> manifest -> executor ->
rewind pipeline and never reaches live transport code.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from typing import Any

try:
    from arrangement_automation.live_input_trim_controller import LiveInputTrimController
except ImportError:  # pragma: no cover - package import from repo root.
    from backend.arrangement_automation.live_input_trim_controller import LiveInputTrimController

from .replay_checkpoint_manifest import ReplayCheckpointManifest, build_replay_checkpoint_manifest
from .replay_executor import ReplayExecutorResult, build_replay_executor_from_rewind_context, simulate_replay_decisions
from .replay_policy_validator import ReplayPolicyValidationReport, validate_replay_policy_consistency
from .replay_proposal_ranking import (
    ProposalRankResult,
    ReplayDecisionProposal,
    ReplayGraphCheckpoint,
    build_fader_proposals,
    build_feedback_mitigation_proposals,
    build_gain_proposals,
    build_performer_stage_interaction_proposals,
    build_operator_assistance_proposals,
    build_replay_graph_checkpoint,
    build_replay_timeline_artifact,
    build_room_compensation_proposals,
    build_speech_priority_arbitration_proposals,
    rank_replay_proposals,
)
from .replay_restore_loader import ReplayRewindContext, load_replay_rewind_context
from ai_mixing_pipeline.reports import write_shadow_mode_reports


REPLAY_STRESS_SCALE_SCHEMA_VERSION = "replay_stress_scale/v1"


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


def _json_size_bytes(payload: Any) -> int:
    return len(json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8"))


def _trim_config() -> dict[str, Any]:
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


@dataclass(frozen=True)
class ReplayStressScenarioSpec:
    name: str
    analyzer_events: int = 24
    critic_passes: int = 3
    decision_burst_size: int = 24
    executor_batches: int = 2
    checkpoint_copies: int = 2
    rewind_iterations: int = 2
    branch_variants: int = 2
    scenario_repetitions: int = 2
    report_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "analyzer_events": int(self.analyzer_events),
            "critic_passes": int(self.critic_passes),
            "decision_burst_size": int(self.decision_burst_size),
            "executor_batches": int(self.executor_batches),
            "checkpoint_copies": int(self.checkpoint_copies),
            "rewind_iterations": int(self.rewind_iterations),
            "branch_variants": int(self.branch_variants),
            "scenario_repetitions": int(self.scenario_repetitions),
            "report_dir": self.report_dir,
        }


@dataclass(frozen=True)
class ReplayStressScenarioResult:
    schema_version: str
    spec: ReplayStressScenarioSpec
    governance: dict[str, Any]
    inventory: dict[str, Any]
    latencies_ms: dict[str, float]
    saturation: dict[str, Any]
    determinism: dict[str, Any]
    artifacts: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "spec": self.spec.to_dict(),
            "governance": dict(self.governance),
            "inventory": dict(self.inventory),
            "latencies_ms": dict(self.latencies_ms),
            "saturation": dict(self.saturation),
            "determinism": dict(self.determinism),
            "artifacts": dict(self.artifacts),
        }


@dataclass(frozen=True)
class ReplayStressSuiteResult:
    schema_version: str
    scenarios: list[ReplayStressScenarioResult]
    governance: dict[str, Any]
    aggregate: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
            "governance": dict(self.governance),
            "aggregate": dict(self.aggregate),
        }


def _critic_payload(pass_index: int, proposal_index: int) -> dict[str, dict[str, Any]]:
    overall = max(0.0, min(1.0, 0.45 + (0.03 * (proposal_index % 5)) - (0.01 * (pass_index % 2))))
    confidence = max(0.25, min(0.99, 0.72 + (0.02 * (pass_index % 3))))
    return {
        f"critic_pass_{pass_index:02d}": {
            "scores": {"overall": round(overall, 3)},
            "delta": {"overall": round(0.01 * ((proposal_index + pass_index) % 4), 3)},
            "confidence": round(confidence, 3),
        }
    }


def _enrich_proposals(
    proposals: list[ReplayDecisionProposal],
    *,
    critic_passes: int,
    variant_index: int,
) -> list[ReplayDecisionProposal]:
    enriched: list[ReplayDecisionProposal] = []
    for proposal_index, proposal in enumerate(proposals):
        critic_scores: dict[str, dict[str, Any]] = {}
        for pass_index in range(max(1, critic_passes)):
            critic_scores.update(_critic_payload(pass_index, proposal_index + variant_index))
        enriched.append(
            replace(
                proposal,
                critic_scores=critic_scores,
                metadata={
                    **dict(proposal.metadata),
                    "stress_variant": variant_index,
                    "critic_passes": critic_passes,
                },
            )
        )
    return enriched


def _build_trim_snapshot(*, spec: ReplayStressScenarioSpec) -> dict[str, Any]:
    controller = LiveInputTrimController(config=_trim_config())
    channel_count = max(2, min(spec.analyzer_events, 32))
    controller.channels = list(range(1, channel_count + 1))
    controller.channel_mapping = {channel: channel for channel in controller.channels}
    controller.last_state = {
        "applied": [
            {
                "channel": 1,
                "replay_correlation_id": f"corr::{spec.name}::seed",
                "target_trim_db": 0.0,
            }
        ]
    }
    return controller.build_replay_snapshot(
        event_seq=spec.analyzer_events,
        replay_metadata={
            "scenario": spec.name,
            "event_seq": spec.analyzer_events,
            "stress_scale": True,
        },
    ).to_dict()


def _build_proposals(
    *,
    spec: ReplayStressScenarioSpec,
    variant_index: int = 0,
) -> list[ReplayDecisionProposal]:
    proposals: list[ReplayDecisionProposal] = []
    total = max(1, spec.analyzer_events)
    gain_count = max(1, total // 4)
    fader_count = max(1, total // 4)
    feedback_count = max(1, total // 6)
    speech_count = max(1, total // 6)
    room_count = max(1, total // 8)
    performer_count = max(1, total // 7)
    operator_count = max(
        0,
        total - (gain_count + fader_count + feedback_count + speech_count + room_count + performer_count),
    )

    gain_recommendations = []
    for index in range(gain_count):
        gain_recommendations.append(
            {
                "source": "auto_soundcheck_engine",
                "mixer_channel": (index % 16) + 1,
                "current_trim_db": round((index % 3) * 0.25, 3),
                "recommended_target_trim_db": round(((index % 5) * 0.2) + 0.3, 3),
                "confidence": round(0.72 + ((index + variant_index) % 4) * 0.04, 3),
                "reason": f"stress_gain_{index}",
                "role": "lead_vocal" if index % 2 == 0 else "backing_vocal",
                "replay_correlation_id": f"corr::{spec.name}::gain::{variant_index}::{index}",
                "dry_run_only": True,
            }
        )
    proposals.extend(build_gain_proposals(recommendations=gain_recommendations))

    fader_updates = []
    for index in range(fader_count):
        fader_updates.append(
            {
                "source": "auto_fader_v2",
                "mixer_channel": ((index + gain_count) % 16) + 1,
                "current_fader_db": round(-6.0 + (index % 4), 3),
                "target_fader_db": round(-5.7 + (index % 4), 3),
                "confidence": round(0.65 + ((index + variant_index) % 5) * 0.03, 3),
                "reason": f"stress_fader_{index}",
                "live_apply_safe": False,
                "max_step_db": 0.5,
                "replay_correlation_id": f"corr::{spec.name}::fader::{variant_index}::{index}",
            }
        )
    proposals.extend(build_fader_proposals(fader_updates=fader_updates))

    feedback_events = []
    for index in range(feedback_count):
        feedback_events.append(
            {
                "event_id": f"feedback_{variant_index}_{index}",
                "source": "feedback_detector",
                "channel": ((index + 2) % 16) + 1,
                "action": "notch",
                "frequency_hz": 1600.0 + (index * 120.0),
                "magnitude_db": -4.0 - float(index % 4),
                "confidence": round(0.62 + ((index + variant_index) % 4) * 0.05, 3),
                "q": round(3.2 + (index % 3) * 0.4, 3),
                "operator_override_required": True,
                "live_apply_allowed": False,
                "replay_correlation_id": f"corr::{spec.name}::feedback::{variant_index}::{index}",
            }
        )
    proposals.extend(build_feedback_mitigation_proposals(feedback_events=feedback_events))

    speech_payloads = []
    for index in range(speech_count):
        speech_payloads.append(
            {
                "event_id": f"speech_{variant_index}_{index}",
                "source": "speech_priority_engine",
                "leader_channel": ((index + 3) % 16) + 1,
                "suppressed_channel": ((index + 7) % 16) + 1,
                "leader_boost_db": round(0.5 + (index % 3) * 0.1, 3),
                "suppressed_cut_db": round(1.0 + (index % 3) * 0.2, 3),
                "dominant_speaker": "lead_vocal",
                "confidence": round(0.6 + ((index + variant_index) % 4) * 0.04, 3),
                "live_apply_safe": False,
                "replay_correlation_id": f"corr::{spec.name}::speech::{variant_index}::{index}",
            }
        )
    proposals.extend(build_speech_priority_arbitration_proposals(arbitration_payloads=speech_payloads))

    room_plans = []
    for index in range(room_count):
        room_plans.append(
            {
                "event_id": f"room_{variant_index}_{index}",
                "room_classification": "reverberant" if index % 2 == 0 else "flat",
                "recommended_target": "master",
                "room_quality_indicator": "good" if index % 2 == 0 else "fair",
                "room_quality_score": round(0.58 + ((index + variant_index) % 4) * 0.05, 3),
                "planned_filters": [
                    {
                        "frequency": 180.0 + (index * 40.0),
                        "gain_db": round(-1.0 - (index % 2) * 0.8, 3),
                        "q": round(1.4 + (index % 3) * 0.2, 3),
                        "filter_type": "peak",
                    }
                ],
                "confidence_score": round(0.55 + ((index + variant_index) % 4) * 0.04, 3),
                "replay_correlation_id": f"corr::{spec.name}::room::{variant_index}::{index}",
            }
        )
    proposals.extend(build_room_compensation_proposals(room_plans=room_plans))

    operator_recommendations = []
    for index in range(operator_count):
        operator_recommendations.append(
            {
                "rec_id": f"operator_{variant_index}_{index}",
                "source": "operator_advice",
                "type": "scene_review",
                "target": f"bus_{(index % 4) + 1}",
                "message": f"Review branch {variant_index} action {index}",
                "confidence": round(0.35 + ((index + variant_index) % 5) * 0.03, 3),
                "urgency": "medium",
                "replay_correlation_id": f"corr::{spec.name}::operator::{variant_index}::{index}",
            }
        )
    proposals.extend(build_operator_assistance_proposals(operator_recommendations=operator_recommendations))

    performer_events = []
    for index in range(performer_count):
        performer_events.append(
            {
                "event_id": f"performer_{variant_index}_{index}",
                "source": "stage_interaction_model",
                "performer_id": f"p{(index % 4) + 1}",
                "mixer_channel": ((index + 6) % 16) + 1,
                "zone_from": "center",
                "zone_to": "stage_left" if index % 2 == 0 else "upstage",
                "orientation_deg": 45.0 + (index * 12.0),
                "orientation_change_deg": 10.0 if index % 2 == 0 else -5.0,
                "distance_m": 2.2 + (index * 0.1),
                "proximity_delta_m": 0.15 if index % 2 == 0 else -0.1,
                "movement_level_delta_db": 0.25 if index % 2 == 0 else -0.18,
                "movement_spill_delta_db": 0.12,
                "risk_profile": "high" if index % 3 == 0 else "medium",
                "event_type": "walk_towards_mic" if index % 2 == 0 else "backing_off",
                "requires_operator_review": True,
                "max_level_delta_db": 1.5,
                "max_spill_delta_db": 0.8,
                "confidence": 0.68 + ((index + variant_index) % 4) * 0.03,
                "replay_correlation_id": f"corr::{spec.name}::performer::{variant_index}::{index}",
            }
        )
    proposals.extend(build_performer_stage_interaction_proposals(interaction_events=performer_events))

    burst_slice = max(1, spec.decision_burst_size)
    proposals = proposals[:burst_slice]
    return _enrich_proposals(proposals, critic_passes=spec.critic_passes, variant_index=variant_index)


def _rank_variant(
    proposals: list[ReplayDecisionProposal],
    *,
    spec: ReplayStressScenarioSpec,
    variant_index: int,
) -> ProposalRankResult:
    return rank_replay_proposals(
        proposals,
        config={
            "critics": {
                f"critic_pass_{pass_index:02d}": {
                    "enabled": True,
                    "weight": round(1.0 - min(pass_index, 4) * 0.1, 3),
                }
                for pass_index in range(max(1, spec.critic_passes))
            }
        },
        family_priority={"fader_adjustment": 1.0 + (0.01 * variant_index)},
    )


def run_replay_stress_scenario(spec: ReplayStressScenarioSpec) -> ReplayStressScenarioResult:
    scenario_start = time.perf_counter()

    trim_snapshot = _build_trim_snapshot(spec=spec)

    analyzer_start = time.perf_counter()
    proposals = _build_proposals(spec=spec)
    analyzer_ms = _elapsed_ms(analyzer_start)

    ranking_start = time.perf_counter()
    ranking = _rank_variant(proposals, spec=spec, variant_index=0)
    ranking_repeat = _rank_variant(proposals, spec=spec, variant_index=0)
    ranking_ms = _elapsed_ms(ranking_start)

    timeline_start = time.perf_counter()
    timeline = build_replay_timeline_artifact(
        proposals,
        ranking,
        metadata={"scenario": spec.name, "event_seq": spec.analyzer_events, "scale_mode": True},
    )
    graph = build_replay_graph_checkpoint(
        proposals,
        ranking,
        metadata={"scenario": spec.name, "event_seq": spec.analyzer_events, "scale_mode": True},
    )
    timeline_ms = _elapsed_ms(timeline_start)

    manifest_start = time.perf_counter()
    manifests: list[ReplayCheckpointManifest] = []
    for copy_index in range(max(1, spec.checkpoint_copies)):
        manifests.append(
            build_replay_checkpoint_manifest(
                trim_snapshot=trim_snapshot,
                graph_checkpoint=graph.to_dict(),
                metadata={
                    "scenario": spec.name,
                    "copy_index": copy_index,
                    "scale_mode": True,
                },
            )
        )
    manifest_ms = _elapsed_ms(manifest_start)
    manifest = manifests[0]

    executor_start = time.perf_counter()
    executor_runs: list[ReplayExecutorResult] = []
    for _ in range(max(1, spec.executor_batches)):
        executor_runs.append(
            simulate_replay_decisions(
                proposals,
                selected_proposal_id=ranking.selected_proposal_id,
            )
        )
    executor_ms = _elapsed_ms(executor_start)
    executor = executor_runs[0]

    rewind_start = time.perf_counter()
    rewind_contexts: list[ReplayRewindContext] = []
    for _ in range(max(1, spec.rewind_iterations)):
        rewind_contexts.append(load_replay_rewind_context(manifest))
    rewind_ms = _elapsed_ms(rewind_start)
    rewind_context = rewind_contexts[0]

    validator_start = time.perf_counter()
    validator_report: ReplayPolicyValidationReport = validate_replay_policy_consistency(
        manifest,
        executor_result=executor,
    )
    validator_ms = _elapsed_ms(validator_start)

    branch_start = time.perf_counter()
    branch_signatures: list[str] = []
    for variant_index in range(max(1, spec.branch_variants)):
        variant_proposals = _build_proposals(spec=spec, variant_index=variant_index)
        variant_ranking = _rank_variant(variant_proposals, spec=spec, variant_index=variant_index)
        branch_signatures.append(variant_ranking.run_signature)
    branch_ms = _elapsed_ms(branch_start)

    concurrent_start = time.perf_counter()
    concurrent_summaries = []
    for repetition in range(max(1, spec.scenario_repetitions)):
        repeated_executor = simulate_replay_decisions(
            proposals,
            selected_proposal_id=ranking.selected_proposal_id,
            initial_state=build_replay_executor_from_rewind_context(rewind_context.to_dict()),
        )
        concurrent_summaries.append(
            {
                "index": repetition,
                "trace_signature": repeated_executor.trace_signature,
                "selected_proposal_id": repeated_executor.selected_proposal_id,
            }
        )
    concurrent_ms = _elapsed_ms(concurrent_start)

    latencies_ms = {
        "analyzer_build": analyzer_ms,
        "decision_ranking": ranking_ms,
        "timeline_checkpoint": timeline_ms,
        "manifest_build": manifest_ms,
        "executor_batch": executor_ms,
        "rewind_restore": rewind_ms,
        "policy_validation": validator_ms,
        "branch_compare": branch_ms,
        "concurrent_batch": concurrent_ms,
        "total": _elapsed_ms(scenario_start),
    }

    inventory = {
        "proposal_count": len(proposals),
        "timeline_event_count": len(timeline.get("timeline") or ()),
        "critic_event_count": sum(1 for item in timeline.get("timeline") or () if item.get("stage") == "critic"),
        "executor_event_count": len(executor.events),
        "manifest_count": len(manifests),
        "rewind_context_count": len(rewind_contexts),
        "branch_variant_count": len(branch_signatures),
        "concurrent_repetition_count": len(concurrent_summaries),
        "replay_correlation_count": len(manifest.replay_correlation_ids),
    }

    saturation = {
        "timeline_bytes": _json_size_bytes(timeline),
        "graph_checkpoint_bytes": _json_size_bytes(graph.to_dict()),
        "manifest_bytes": _json_size_bytes(manifest.to_dict()),
        "trim_snapshot_bytes": _json_size_bytes(trim_snapshot),
        "executor_trace_bytes": _json_size_bytes(executor.to_dict()),
        "max_critic_inventory_per_proposal": max(len(proposal.critic_scores) for proposal in proposals),
        "branch_signature_count": len(set(branch_signatures)),
        "executor_batch_trace_count": len({item.trace_signature for item in executor_runs}),
        "concurrent_trace_count": len({item["trace_signature"] for item in concurrent_summaries}),
    }

    determinism = {
        "ranking_stable": ranking.run_signature == ranking_repeat.run_signature,
        "selected_proposal_stable": ranking.selected_proposal_id == ranking_repeat.selected_proposal_id,
        "executor_trace_stable": len({item.trace_signature for item in executor_runs}) == 1,
        "concurrent_trace_stable": len({item["trace_signature"] for item in concurrent_summaries}) == 1,
        "rewind_manifest_stable": all(context.manifest_id == rewind_context.manifest_id for context in rewind_contexts),
        "policy_validation_valid": bool(validator_report.valid),
    }

    governance = {
        "dry_run_only": bool(manifest.dry_run_only),
        "live_mixer_writes": bool(manifest.live_mixer_writes),
        "executor_dry_run_only": bool(executor.dry_run_only),
        "transport_mutation_allowed": False,
        "runtime_modifications": False,
        "orchestration_changes": False,
    }

    artifacts = {
        "manifest_id": manifest.manifest_id,
        "selected_proposal_id": ranking.selected_proposal_id,
        "ranking_signature": ranking.run_signature,
        "executor_trace_signature": executor.trace_signature,
        "rewind_manifest_id": rewind_context.manifest_id,
        "validator_summary": dict(validator_report.summary),
        "branch_signatures": list(branch_signatures),
    }
    if spec.report_dir:
        report_paths = write_shadow_mode_reports(
            spec.report_dir,
            manifest=manifest.to_dict(),
            executor=executor,
            validation_report=validator_report.to_dict(),
        )
        artifacts["shadow_reports"] = report_paths

    return ReplayStressScenarioResult(
        schema_version=REPLAY_STRESS_SCALE_SCHEMA_VERSION,
        spec=spec,
        governance=governance,
        inventory=inventory,
        latencies_ms=latencies_ms,
        saturation=saturation,
        determinism=determinism,
        artifacts=artifacts,
    )


def run_replay_stress_suite(
    specs: list[ReplayStressScenarioSpec],
) -> ReplayStressSuiteResult:
    scenarios = [run_replay_stress_scenario(spec) for spec in specs]
    governance = {
        "all_dry_run_only": all(item.governance["dry_run_only"] for item in scenarios),
        "all_live_mixer_writes_disabled": all(not item.governance["live_mixer_writes"] for item in scenarios),
        "all_executor_dry_run_only": all(item.governance["executor_dry_run_only"] for item in scenarios),
    }
    aggregate = {
        "scenario_count": len(scenarios),
        "total_proposals": sum(item.inventory["proposal_count"] for item in scenarios),
        "total_timeline_events": sum(item.inventory["timeline_event_count"] for item in scenarios),
        "max_manifest_bytes": max(item.saturation["manifest_bytes"] for item in scenarios) if scenarios else 0,
        "max_timeline_bytes": max(item.saturation["timeline_bytes"] for item in scenarios) if scenarios else 0,
        "all_ranking_stable": all(item.determinism["ranking_stable"] for item in scenarios),
        "all_executor_traces_stable": all(item.determinism["executor_trace_stable"] for item in scenarios),
        "all_policy_validation_valid": all(item.determinism["policy_validation_valid"] for item in scenarios),
    }
    return ReplayStressSuiteResult(
        schema_version=REPLAY_STRESS_SCALE_SCHEMA_VERSION,
        scenarios=scenarios,
        governance=governance,
        aggregate=aggregate,
    )


__all__ = [
    "REPLAY_STRESS_SCALE_SCHEMA_VERSION",
    "ReplayStressScenarioResult",
    "ReplayStressScenarioSpec",
    "ReplayStressSuiteResult",
    "run_replay_stress_scenario",
    "run_replay_stress_suite",
]
