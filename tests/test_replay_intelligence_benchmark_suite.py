"""Canonical replay-safe intelligence baseline benchmarks.

This suite captures deterministic baseline metrics for:
- analyzer tagging / proposal inventory
- critic score aggregation stability
- decision ranking stability
- executor simulation reproducibility
- rewind manifest coherence
- policy-validation completeness
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from ai_mixing_pipeline.decision_layer.replay_checkpoint_manifest import build_replay_checkpoint_manifest
from ai_mixing_pipeline.decision_layer.replay_executor import simulate_replay_decisions
from ai_mixing_pipeline.decision_layer.replay_policy_validator import (
    validate_replay_policy_consistency,
)
from ai_mixing_pipeline.decision_layer.replay_proposal_ranking import (
    build_feedback_mitigation_proposals,
    build_fader_proposals,
    build_gain_proposals,
    build_room_compensation_proposals,
    build_replay_graph_checkpoint,
    build_replay_timeline_artifact,
    build_speech_priority_arbitration_proposals,
    build_operator_assistance_proposals,
    rank_replay_proposals,
)
from ai_mixing_pipeline.decision_layer.replay_restore_loader import load_replay_rewind_context
from arrangement_automation.live_input_trim_controller import LiveInputTrimController


def _build_benchmark_case(
    *,
    ranking_config: dict[str, any] | None = None,
    family_priority: dict[str, float] | None = None,
):
    controller = LiveInputTrimController(
        config={
            "automation": {
                "live_input_trim": {
                    "run_background_loop": False,
                    "analysis_only_mode": True,
                    "live_apply_enabled": False,
                    "confirm_live_apply": False,
                }
            }
        }
    )
    controller.channels = [1, 2]
    controller.channel_mapping = {1: 1, 2: 2}
    controller.last_state = {}
    trim_snapshot = controller.build_replay_snapshot(
        event_seq=100,
        replay_metadata={"scenario": "bench"},
    ).to_dict()

    gain = replace(
        build_gain_proposals(
            recommendations=[
                {
                    "source": "auto_soundcheck_engine",
                    "mixer_channel": 1,
                    "current_trim_db": 0.0,
                    "recommended_target_trim_db": 0.8,
                    "confidence": 0.82,
                    "reason": "benchmark",
                    "role": "lead_vocal",
                    "replay_correlation_id": "corr::gain::1",
                    "dry_run_only": True,
                }
            ]
        )[0],
        critic_scores={
            "clarity": {"scores": {"overall": 0.61}, "delta": {"overall": 0.12}, "confidence": 0.84},
            "safety": {"safety_score": 0.91},
        },
    )

    fader = replace(
        build_fader_proposals(
            fader_updates=[
                {
                    "source": "auto_fader_v2",
                    "mixer_channel": 2,
                    "current_fader_db": -3.0,
                    "target_fader_db": -2.0,
                    "confidence": 0.77,
                    "reason": "benchmark",
                    "live_apply_safe": True,
                    "replay_correlation_id": "corr::fader::2",
                    "max_step_db": 0.4,
                }
            ]
        )[0],
        critic_scores={
            "audiobox_aesthetics": {"scores": {"overall": 0.45}, "delta": {"overall": 0.11}, "confidence": 0.7},
        },
    )

    feedback = replace(
        build_feedback_mitigation_proposals(
            feedback_events=[
                {
                    "event_id": "fb1",
                    "source": "feedback_detector",
                    "channel": 3,
                    "action": "notch",
                    "frequency_hz": 2500,
                    "magnitude_db": -6.2,
                    "confidence": 0.65,
                    "q": 4.2,
                    "replay_correlation_id": "corr::feedback::3",
                    "operator_override_required": True,
                    "live_apply_allowed": False,
                }
            ]
        )[0],
        critic_scores={"safety": {"safety_score": 0.88}},
    )

    speech = replace(
        build_speech_priority_arbitration_proposals(
            arbitration_payloads=[
                {
                    "event_id": "sp1",
                    "source": "speech_priority_engine",
                    "leader_channel": 4,
                    "suppressed_channel": 5,
                    "leader_boost_db": 0.7,
                    "suppressed_cut_db": 1.4,
                    "dominant_speaker": "lead_vocal",
                    "confidence": 0.71,
                    "live_apply_safe": False,
                    "replay_correlation_id": "corr::speech::4->5",
                }
            ]
        )[0],
        critic_scores={"mert": {"scores": {"overall": 0.51}, "delta": {"overall": 0.09}, "confidence": 0.67}},
    )

    room = replace(
        build_room_compensation_proposals(
            room_plans=[
                {
                    "event_id": "r1",
                    "room_classification": "reverberant",
                    "recommended_target": "master",
                    "room_quality_indicator": "good",
                    "room_quality_score": 0.72,
                    "planned_filters": [
                        {"frequency": 250.0, "gain_db": -2.0, "q": 1.8, "filter_type": "peak"}
                    ],
                    "confidence_score": 0.66,
                    "replay_correlation_id": "corr::room::master",
                }
            ]
        )[0],
        critic_scores={"clarity": {"scores": {"overall": 0.32}, "delta": {"overall": 0.06}, "confidence": 0.61}},
    )

    operator = build_operator_assistance_proposals(
        operator_recommendations=[
            {
                "rec_id": "op1",
                "source": "operator_advice",
                "type": "scene_reset",
                "target": "mix",
                "message": "Consider reset",
                "confidence": 0.42,
                "urgency": "medium",
            }
        ]
    )[0]

    proposals = [gain, fader, feedback, speech, room, operator]
    ranking = rank_replay_proposals(
        proposals,
        config={
            "critics": {
                "clarity": {"enabled": True, "weight": 1.0},
                "safety": {"enabled": True, "weight": 0.8},
                "audiobox_aesthetics": {"enabled": True, "weight": 0.6},
                "mert": {"enabled": True, "weight": 0.6},
            }
        }
        if ranking_config is None
        else ranking_config,
        family_priority=family_priority,
    )
    timeline = build_replay_timeline_artifact(proposals, ranking, metadata={"scenario": "benchmark", "event_seq": 100})
    graph = build_replay_graph_checkpoint(
        proposals,
        ranking,
        metadata={"scenario": "benchmark", "event_seq": 100, "path": "/tmp/replay_benchmark.wav"},
    )
    manifest = build_replay_checkpoint_manifest(
        trim_snapshot=trim_snapshot,
        graph_checkpoint=graph.to_dict(),
        metadata={"scenario": "benchmark_suite_v1"},
    )
    executor_one = simulate_replay_decisions(proposals)
    executor_two = simulate_replay_decisions(proposals)
    report = validate_replay_policy_consistency(manifest, executor_result=executor_one)

    return {
        "proposals": proposals,
        "ranking": ranking,
        "timeline": timeline,
        "manifest": manifest,
        "executor_one": executor_one,
        "executor_two": executor_two,
        "validator_report": report,
        "rewind_context": load_replay_rewind_context(manifest.to_dict()),
    }


def _build_metrics(case: dict[str, any]) -> dict[str, any]:
    return {
        "scenario": "replay_intelligence_benchmark_v1",
        "manifest": {
            "manifest_id": case["manifest"].manifest_id,
            "event_seq": case["manifest"].event_seq,
            "dry_run_only": case["manifest"].dry_run_only,
            "live_mixer_writes": case["manifest"].live_mixer_writes,
            "replay_correlation_count": len(case["manifest"].replay_correlation_ids),
            "replay_correlation_ids": sorted(case["manifest"].replay_correlation_ids),
            "timeline_event_count": case["timeline"]["timeline_event_count"],
            "proposal_count": case["timeline"]["proposal_count"],
        },
        "ranking": {
            "run_signature": case["ranking"].run_signature,
            "selected_proposal_id": case["ranking"].selected_proposal_id,
            "top_rank_family": case["ranking"].ranked_proposals[0]["family"],
            "deterministic_signature": f"{case['ranking'].ranked_proposals[0]['proposal_id']}::{case['ranking'].ranked_proposals[1]['proposal_id']}",
        },
        "executor": {
            "trace_signature": case["executor_one"].trace_signature,
            "trace_signature_repeat": case["executor_two"].trace_signature,
            "trace_stable": case["executor_one"].trace_signature == case["executor_two"].trace_signature,
            "proposals_executed_count": len(case["executor_one"].proposals_executed),
            "applied_ratio": round(len(case["executor_one"].proposals_executed) / float(case["timeline"]["proposal_count"]), 3),
            "restored_state_present": case["executor_one"].restored_state is not None,
            "first_state_gain": case["executor_one"].state_snapshots[0]["gains"].get("1"),
        },
        "validation": {
            "valid": bool(case["validator_report"].valid),
            "finding_count": len(case["validator_report"].findings),
            "drift_selected_changed": bool(case["validator_report"].drift_summary.get("selected_proposal_changed", False)),
            "selected_from_executor": case["executor_one"].selected_proposal_id,
        },
        "rewind": {
            "manifest_id": case["rewind_context"].manifest_id,
            "event_seq": case["rewind_context"].event_seq,
            "correlation_count": len(case["rewind_context"].replay_correlation_ids),
            "replay_dry_run_only": case["rewind_context"].dry_run_only,
        },
    }


def test_replay_intelligence_benchmark_matches_canonical_metrics() -> None:
    case = _build_benchmark_case()
    actual = _build_metrics(case)

    fixture_path = Path(__file__).parent / "fixtures" / "replay_intelligence_benchmark_v1.json"
    baseline = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert actual["manifest"] == baseline["manifest"]
    assert actual["ranking"]["run_signature"] == baseline["ranking"]["run_signature"]
    assert actual["ranking"]["selected_proposal_id"] == baseline["ranking"]["selected_proposal_id"]
    assert actual["executor"]["trace_signature"] == baseline["executor"]["trace_signature"]
    assert actual["executor"]["trace_signature_repeat"] == baseline["executor"]["trace_signature_repeat"]
    assert actual["executor"]["trace_stable"] is True
    assert actual["validation"]["valid"] is True
    assert actual["rewind"]["replay_dry_run_only"] is True
    assert actual["rewind"]["manifest_id"] == baseline["rewind"]["manifest_id"]


def test_replay_intelligence_baseline_detects_decision_drift() -> None:
    baseline = _build_benchmark_case()
    baseline_metrics = _build_metrics(baseline)
    candidate = _build_benchmark_case(
        ranking_config={
            "critics": {
                "clarity": {"enabled": True, "weight": 0.2},
                "safety": {"enabled": True, "weight": 0.2},
                "audiobox_aesthetics": {"enabled": True, "weight": 0.2},
                "mert": {"enabled": True, "weight": 0.2},
            }
        },
        family_priority={"fader_adjustment": 1.5},
    )
    candidate_metrics = _build_metrics(candidate)
    candidate_report = validate_replay_policy_consistency(
        candidate["manifest"],
        executor_result=candidate["executor_one"],
        baseline_manifest=baseline["manifest"].to_dict(),
    )

    assert candidate["manifest"].manifest_id != baseline["manifest"].manifest_id
    assert candidate["ranking"].run_signature != baseline["ranking"].run_signature
    assert candidate_metrics["ranking"]["selected_proposal_id"] != baseline_metrics["ranking"]["selected_proposal_id"]
    assert candidate_report.valid is True
    assert candidate_report.drift_summary.get("selected_proposal_changed") is True
    assert baseline["validator_report"].drift_summary == {}
