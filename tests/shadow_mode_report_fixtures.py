from __future__ import annotations

from typing import Any

from ai_mixing_pipeline.decision_layer.replay_checkpoint_manifest import (
    build_replay_checkpoint_manifest,
)
from ai_mixing_pipeline.decision_layer.replay_executor import simulate_replay_decisions
from ai_mixing_pipeline.decision_layer.replay_policy_validator import (
    validate_replay_policy_consistency,
)
from ai_mixing_pipeline.decision_layer.replay_proposal_ranking import (
    build_gain_proposals,
    build_operator_assistance_proposals,
    build_replay_graph_checkpoint,
    rank_replay_proposals,
)


def build_shadow_report_fixture_cases() -> list[dict[str, Any]]:
    operator_selected = _build_operator_selected_case()
    ai_selected = _build_ai_selected_case()
    return [operator_selected, ai_selected]


def _build_operator_selected_case() -> dict[str, Any]:
    gain_proposal = build_gain_proposals(
        recommendations=[
            {
                "source": "auto_soundcheck_engine",
                "mixer_channel": 1,
                "current_trim_db": 0.0,
                "recommended_target_trim_db": 1.2,
                "confidence": 0.84,
                "reason": "gain_stress",
                "replay_correlation_id": "corr::gain::test",
            }
        ]
    )[0]
    operator_proposal = build_operator_assistance_proposals(
        operator_recommendations=[
            {
                "source": "operator_console",
                "type": "manual_prompt",
                "rec_id": "operator::1",
                "message": "Hold vocals 1dB hotter.",
                "target": "lead_vocal",
                "confidence": 0.92,
                "replay_correlation_id": "corr::operator::test",
                "requires_confirmation": True,
            }
        ]
    )[0]
    ranking = rank_replay_proposals([gain_proposal, operator_proposal])
    graph = build_replay_graph_checkpoint(
        [gain_proposal, operator_proposal],
        ranking,
        metadata={"scenario": "report_writer"},
    )
    manifest = build_replay_checkpoint_manifest(
        trim_snapshot={
            "version": "replay_state_snapshot/v1",
            "snapshot_id": "replay_checkpoint::live_input_trim::1",
            "transport_policy": {"dry_run_only": True},
            "replay_metadata": {"scenario": "report_writer"},
            "replay_correlation_ids": ["corr::gain::test", "corr::operator::test"],
        },
        graph_checkpoint=graph.to_dict(),
        metadata={"scenario": "report_writer"},
    )
    executor = simulate_replay_decisions(
        [gain_proposal, operator_proposal],
        selected_proposal_id=ranking.selected_proposal_id,
    )
    validation = validate_replay_policy_consistency(
        manifest,
        executor_result=executor,
    )
    return {
        "name": "operator_selected_shadow_override",
        "generated_at": "2026-05-16T12:00:00+00:00",
        "manifest": manifest.to_dict(),
        "executor": executor.to_dict(),
        "validation": validation.to_dict(),
        "expected": {
            "selected_proposal_id": "corr::operator::test",
            "selected_family": "operator_assistance",
            "operator_count": 1,
            "ai_count": 1,
            "operator_action_capture_id": "corr::operator::test",
            "ai_would_do_this_id": "corr::gain::test",
            "disagreement_codes": ["operator_selected"],
            "disagreement_rate": 1.0,
            "disagreement_taxonomy_counts": {"operator_override": 1},
            "blocked_reasons": ["operator_confirmation_required"],
            "ready": True,
            "confidence_direction": "operator_above_ai",
            "confidence_delta": 0.08,
            "blocked_event_count": 1,
            "operator_source_payload": {
                "rec_id": "operator::1",
                "type": "manual_prompt",
                "message": "Hold vocals 1dB hotter.",
                "target": "lead_vocal",
                "requires_confirmation": True,
            },
            "ai_source_payload": {
                "mixer_channel": 1,
                "recommended_target_trim_db": 1.2,
                "reason": "gain_stress",
            },
        },
    }


def _build_ai_selected_case() -> dict[str, Any]:
    gain_proposal = build_gain_proposals(
        recommendations=[
            {
                "source": "auto_soundcheck_engine",
                "mixer_channel": 2,
                "current_trim_db": 0.0,
                "recommended_target_trim_db": 1.0,
                "confidence": 0.95,
                "reason": "gain_stability",
                "replay_correlation_id": "corr::gain::stable",
            }
        ]
    )[0]
    operator_proposal = build_operator_assistance_proposals(
        operator_recommendations=[
            {
                "source": "operator_console",
                "type": "manual_prompt",
                "rec_id": "operator::stable",
                "message": "Minor vocal bump.",
                "target": "lead_vocal",
                "confidence": 0.2,
                "replay_correlation_id": "corr::operator::stable",
            }
        ]
    )[0]
    ranking = rank_replay_proposals([gain_proposal, operator_proposal])
    graph = build_replay_graph_checkpoint(
        [gain_proposal, operator_proposal],
        ranking,
        metadata={"scenario": "ready_report"},
    )
    manifest = build_replay_checkpoint_manifest(
        trim_snapshot={
            "version": "replay_state_snapshot/v1",
            "snapshot_id": "replay_checkpoint::live_input_trim::2",
            "transport_policy": {"dry_run_only": True},
        },
        graph_checkpoint=graph.to_dict(),
        metadata={"scenario": "ready_report"},
    )
    executor = simulate_replay_decisions(
        [gain_proposal, operator_proposal],
        selected_proposal_id=ranking.selected_proposal_id,
    )
    validation = validate_replay_policy_consistency(
        manifest,
        executor_result=executor,
    )
    return {
        "name": "ai_selected_ready_shadow_run",
        "generated_at": "2026-05-16T12:30:00+00:00",
        "manifest": manifest.to_dict(),
        "executor": executor.to_dict(),
        "validation": validation.to_dict(),
        "expected": {
            "selected_proposal_id": "corr::gain::stable",
            "selected_family": "gain_adjustment",
            "operator_count": 1,
            "ai_count": 1,
            "operator_action_capture_id": "corr::operator::stable",
            "ai_would_do_this_id": "corr::gain::stable",
            "disagreement_codes": [],
            "disagreement_rate": 0.0,
            "disagreement_taxonomy_counts": {},
            "blocked_reasons": ["operator_confirmation_required"],
            "ready": True,
            "confidence_direction": "ai_above_operator",
            "confidence_delta": -0.75,
            "blocked_event_count": 1,
            "operator_source_payload": {
                "rec_id": "operator::stable",
                "type": "manual_prompt",
                "message": "Minor vocal bump.",
                "target": "lead_vocal",
            },
            "ai_source_payload": {
                "mixer_channel": 2,
                "recommended_target_trim_db": 1.0,
                "reason": "gain_stability",
            },
        },
    }
