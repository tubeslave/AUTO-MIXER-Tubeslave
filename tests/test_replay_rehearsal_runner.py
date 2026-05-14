from __future__ import annotations

import json

from ai_mixing_pipeline.decision_layer.replay_rehearsal_runner import (
    REPLAY_REHEARSAL_BUNDLE_SCHEMA_VERSION,
    run_replay_rehearsal,
)


def _scenario() -> dict[str, object]:
    return {
        "scenario": "unit_rehearsal_bundle",
        "event_seq": 144,
        "trim": {
            "channels": [1, 2],
            "channel_mapping": {"1": 1, "2": 2},
            "last_state": {
                "applied": [
                    {
                        "channel": 1,
                        "target_trim_db": 0.6,
                        "replay_correlation_id": "corr::trim::seed",
                    }
                ]
            },
        },
        "proposal_inputs": {
            "gain_recommendations": [
                {
                    "source": "auto_soundcheck_engine",
                    "mixer_channel": 1,
                    "current_trim_db": 0.0,
                    "recommended_target_trim_db": 0.8,
                    "confidence": 0.91,
                    "reason": "unit_gain",
                    "role": "lead_vocal",
                    "replay_correlation_id": "corr::gain::1",
                    "dry_run_only": True,
                }
            ],
            "operator_recommendations": [
                {
                    "source": "operator_console",
                    "type": "manual_prompt",
                    "rec_id": "operator::1",
                    "message": "Hold current scene.",
                    "confidence": 0.2,
                    "replay_correlation_id": "corr::operator::1",
                }
            ],
        },
        "critic_scores_by_replay_correlation_id": {
            "corr::gain::1": {
                "clarity": {"scores": {"overall": 0.66}, "delta": {"overall": 0.12}, "confidence": 0.86},
                "safety": {"safety_score": 0.95},
            }
        },
        "ranking_config": {
            "critics": {
                "clarity": {"enabled": True, "weight": 1.0},
                "safety": {"enabled": True, "weight": 0.8},
            }
        },
    }


def test_replay_rehearsal_runner_writes_bundle_and_artifacts(tmp_path) -> None:
    bundle = run_replay_rehearsal(_scenario(), output_dir=tmp_path)

    assert bundle.schema_version == REPLAY_REHEARSAL_BUNDLE_SCHEMA_VERSION
    assert bundle.governance["dry_run_only"] is True
    assert bundle.governance["live_mixer_writes"] is False
    assert bundle.governance["executor_dry_run_only"] is True
    assert bundle.governance["transport_mutation_allowed"] is False
    assert bundle.inventory["proposal_count"] == 2
    assert bundle.inventory["replay_correlation_count"] >= 2
    assert bundle.blocked_reasons == []
    assert bundle.readiness["ready"] is True
    assert any(stage["stage"] == "shadow_reports_written" for stage in bundle.stage_transitions)
    assert "corr::gain::1" in bundle.replay_correlation_ids
    assert "corr::operator::1" in bundle.replay_correlation_ids

    bundle_path = tmp_path / "replay_rehearsal_bundle.json"
    manifest_path = tmp_path / "replay_checkpoint_manifest.json"
    executor_path = tmp_path / "replay_executor_result.json"
    validation_path = tmp_path / "replay_policy_validation_report.json"
    readiness_path = tmp_path / bundle.artifacts["shadow_reports"]["promotion_readiness"]

    assert bundle_path.exists()
    assert manifest_path.exists()
    assert executor_path.exists()
    assert validation_path.exists()
    assert readiness_path.exists()

    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert payload["bundle_id"] == bundle.bundle_id
    assert payload["manifest_id"] == bundle.manifest_id
    assert payload["shadow_reports"]["readiness"]["ready"] is True


def test_replay_rehearsal_runner_is_deterministic_for_same_inputs(tmp_path) -> None:
    first = run_replay_rehearsal(_scenario(), output_dir=tmp_path / "run_a")
    second = run_replay_rehearsal(_scenario(), output_dir=tmp_path / "run_b")

    assert first.bundle_id == second.bundle_id
    assert first.manifest_id == second.manifest_id
    assert first.executor["trace_signature"] == second.executor["trace_signature"]
    assert first.replay_correlation_ids == second.replay_correlation_ids
    assert first.blocked_reasons == second.blocked_reasons
