from __future__ import annotations

import json

from ai_mixing_pipeline.decision_layer.replay_checkpoint_manifest import build_replay_checkpoint_manifest
from ai_mixing_pipeline.decision_layer.replay_executor import simulate_replay_decisions
from ai_mixing_pipeline.decision_layer.replay_policy_validator import validate_replay_policy_consistency
from ai_mixing_pipeline.decision_layer.replay_proposal_ranking import (
    build_gain_proposals,
    build_operator_assistance_proposals,
    build_replay_graph_checkpoint,
    rank_replay_proposals,
)
from ai_mixing_pipeline.reports.shadow_report_writer import write_shadow_mode_reports


def test_shadow_mode_reports_include_operator_ai_comparison_and_rollback_paths(tmp_path):
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
                "confidence": 0.92,
                "replay_correlation_id": "corr::operator::test",
            }
        ]
    )[0]
    ranking = rank_replay_proposals([gain_proposal, operator_proposal])
    graph = build_replay_graph_checkpoint([gain_proposal, operator_proposal], ranking, metadata={"scenario": "report_writer"})
    trim_snapshot = {
        "version": "replay_state_snapshot/v1",
        "transport_policy": {"dry_run_only": True},
        "replay_metadata": {"scenario": "report_writer"},
    }
    manifest = build_replay_checkpoint_manifest(
        trim_snapshot=trim_snapshot,
        graph_checkpoint=graph.to_dict(),
        metadata={"scenario": "report_writer"},
    )

    executor = simulate_replay_decisions([gain_proposal, operator_proposal], selected_proposal_id=ranking.selected_proposal_id)
    validation = validate_replay_policy_consistency(
        manifest,
        executor_result=executor,
    )

    report_paths = write_shadow_mode_reports(
        tmp_path,
        manifest=manifest.to_dict(),
        executor=executor,
        validation_report=validation.to_dict(),
    )

    comparison = json.loads((tmp_path / "shadow_mode_comparison.json").read_text(encoding="utf-8"))
    rollback = json.loads((tmp_path / "rollback_simulation_report.json").read_text(encoding="utf-8"))
    readiness = json.loads((tmp_path / "promotion_readiness_report.json").read_text(encoding="utf-8"))
    trajectory = json.loads((tmp_path / "confidence_trajectory.json").read_text(encoding="utf-8"))

    assert set(report_paths) == {
        "shadow_mode_comparison",
        "confidence_trajectory",
        "rollback_simulation",
        "promotion_readiness",
        "shadow_mode_summary",
    }
    assert comparison["operator_count"] == 1
    assert comparison["ai_count"] == 1
    assert comparison["disagreements"][0]["code"] == "operator_selected"
    assert any(item["proposal_id"] == "corr::operator::test" for item in comparison["operator_rows"]) is False
    assert len(rollback["possible_rollbacks"]) == len(executor.events) + 1
    assert readiness["schema_version"] == "replay_shadow_report/v1"
    assert len(trajectory["trajectory"]) == 2


def test_shadow_mode_reports_flag_readiness_as_ready_for_stable_shadow_workflow(tmp_path):
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
                "confidence": 0.2,
                "replay_correlation_id": "corr::operator::stable",
            }
        ]
    )[0]
    ranking = rank_replay_proposals([gain_proposal, operator_proposal])
    assert ranking.selected_proposal_id == gain_proposal.proposal_id
    graph = build_replay_graph_checkpoint([gain_proposal, operator_proposal], ranking, metadata={"scenario": "ready_report"})
    manifest = build_replay_checkpoint_manifest(
        trim_snapshot={"version": "replay_state_snapshot/v1", "transport_policy": {"dry_run_only": True}},
        graph_checkpoint=graph.to_dict(),
        metadata={"scenario": "ready_report"},
    )
    executor = simulate_replay_decisions([gain_proposal, operator_proposal], selected_proposal_id=ranking.selected_proposal_id)
    validation = validate_replay_policy_consistency(
        manifest,
        executor_result=executor,
    )

    report_paths = write_shadow_mode_reports(
        tmp_path,
        manifest=manifest.to_dict(),
        executor=executor,
        validation_report=validation.to_dict(),
    )

    comparison = json.loads((tmp_path / "shadow_mode_comparison.json").read_text(encoding="utf-8"))
    readiness = json.loads((tmp_path / "promotion_readiness_report.json").read_text(encoding="utf-8"))

    assert report_paths["promotion_readiness"] == str(tmp_path / "promotion_readiness_report.json")
    assert comparison["disagreements"] == []
    assert readiness["ready"] is True
    assert readiness["promotion_readiness_score"] >= 75
