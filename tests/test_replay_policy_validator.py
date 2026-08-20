from __future__ import annotations

import pytest

from ai_mixing_pipeline.decision_layer.replay_checkpoint_manifest import (
    ReplayCheckpointManifest,
    build_replay_checkpoint_manifest,
)
from ai_mixing_pipeline.decision_layer.replay_executor import (
    build_replay_executor_from_rewind_context,
    simulate_replay_decisions,
)
from ai_mixing_pipeline.decision_layer.replay_policy_validator import (
    REPLAY_POLICY_VALIDATION_SCHEMA_VERSION,
    validate_replay_policy_consistency,
)
from ai_mixing_pipeline.decision_layer.replay_proposal_ranking import (
    ReplayDecisionProposal,
    build_fader_proposals,
    build_gain_proposals,
    build_replay_graph_checkpoint,
    rank_replay_proposals,
)
from ai_mixing_pipeline.decision_layer.replay_restore_loader import (
    load_replay_rewind_context,
)
from arrangement_automation.live_input_trim_controller import (
    LiveInputTrimController,
)


def _trim_config():
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


def _build_manifest(*, selected_family_priority: dict[str, float] | None = None):
    controller = LiveInputTrimController(config=_trim_config())
    controller.channels = [1, 2]
    controller.channel_mapping = {1: 1, 2: 2}
    controller.last_state = {
        "applied": [
            {
                "channel": 1,
                "replay_correlation_id": "corr::shared::gain",
                "target_trim_db": 0.5,
            }
        ]
    }
    trim_snapshot = controller.build_replay_snapshot(
        event_seq=34,
        replay_metadata={"stage": "policy_validator"},
    ).to_dict()

    gain = build_gain_proposals(
        recommendations=[
            {
                "source": "auto_soundcheck_engine",
                "mixer_channel": 1,
                "current_trim_db": 0.0,
                "recommended_target_trim_db": 0.5,
                "confidence": 0.88,
                "reason": "validator_gain",
                "role": "lead_vocal",
                "replay_correlation_id": "corr::shared::gain",
            }
        ]
    )[0]
    fader = build_fader_proposals(
        fader_updates=[
            {
                "source": "auto_fader",
                "mixer_channel": 2,
                "current_fader_db": -4.0,
                "target_fader_db": -3.5,
                "confidence": 0.6,
                "reason": "validator_fader",
                "live_apply_safe": False,
                "replay_correlation_id": "corr::shared::fader",
            }
        ]
    )[0]
    proposals = [gain, fader]
    ranking = rank_replay_proposals(
        proposals,
        family_priority=selected_family_priority,
    )
    graph_checkpoint = build_replay_graph_checkpoint(
        proposals,
        ranking,
        metadata={"event_seq": 34, "stage": "policy_validator"},
    ).to_dict()
    manifest = build_replay_checkpoint_manifest(
        trim_snapshot=trim_snapshot,
        graph_checkpoint=graph_checkpoint,
        critic_artifacts={
            "replay_correlation_ids": ["corr::shared::gain", "corr::shared::fader"],
            "validator_schema_hint": "replay_policy_validation/v1",
        },
        metadata={"scenario": "replay_policy_validator"},
    )
    return manifest, proposals, ranking


def test_replay_policy_validator_accepts_replay_safe_manifest_and_executor():
    manifest, proposals, ranking = _build_manifest()
    context = load_replay_rewind_context(manifest)
    executor = simulate_replay_decisions(
        proposals,
        selected_proposal_id=ranking.selected_proposal_id,
        initial_state=build_replay_executor_from_rewind_context(context.to_dict()),
    )

    report = validate_replay_policy_consistency(
        manifest,
        executor_result=executor,
    )

    assert report.schema_version == REPLAY_POLICY_VALIDATION_SCHEMA_VERSION
    assert report.valid is True
    assert report.summary["error_count"] == 0
    assert report.summary["selected_proposal_id"] == ranking.selected_proposal_id
    assert report.summary["executor_trace_signature"] == executor.trace_signature
    assert report.findings == []


def test_replay_policy_validator_rejects_unbounded_feedback_proposal_without_executor():
    base_manifest, _, _ = _build_manifest()
    proposal = ReplayDecisionProposal(
        proposal_id="feedback::notch::3::unbounded",
        family="feedback_mitigation",
        source_system="untrusted_restore",
        action_type="notch",
        target={"channel": 3, "frequency_hz": 2000.0},
        current_state={"magnitude_db": 0.0},
        requested_state={"gain_db": -1000.0, "q": 100.0},
        confidence=0.8,
        safety={"max_q": 8.0, "max_cut_db": 9.0},
        auto_apply_blocked=True,
    )
    ranking = rank_replay_proposals([proposal])
    graph_checkpoint = build_replay_graph_checkpoint(
        [proposal],
        ranking,
        metadata={"event_seq": 34, "stage": "policy_validator"},
    ).to_dict()
    manifest = build_replay_checkpoint_manifest(
        trim_snapshot=base_manifest.trim_snapshot,
        graph_checkpoint=graph_checkpoint,
        metadata=base_manifest.metadata,
    )

    report = validate_replay_policy_consistency(manifest)

    assert report.valid is False
    finding = next(item for item in report.findings if item.code == "invalid_feedback_mitigation_bounds")
    assert finding.proposal_id == proposal.proposal_id
    assert "exceeds declared max_q" in finding.details["validation_error"]


def test_replay_policy_validator_rejects_gain_proposal_with_missing_step_bound():
    base_manifest, _, _ = _build_manifest()
    proposal = ReplayDecisionProposal(
        proposal_id="gain::missing-bound::1",
        family="gain_adjustment",
        source_system="untrusted_restore",
        action_type="set_gain_db",
        target={"channel": 1, "kind": "gain"},
        current_state={"gain_db": 0.0},
        requested_state={"gain_db": 10.0},
        confidence=0.8,
        safety={},
        auto_apply_blocked=False,
    )
    ranking = rank_replay_proposals([proposal])
    graph_checkpoint = build_replay_graph_checkpoint(
        [proposal],
        ranking,
        metadata={"event_seq": 34, "stage": "policy_validator"},
    ).to_dict()
    manifest = build_replay_checkpoint_manifest(
        trim_snapshot=base_manifest.trim_snapshot,
        graph_checkpoint=graph_checkpoint,
        metadata=base_manifest.metadata,
    )

    report = validate_replay_policy_consistency(manifest)

    assert report.valid is False
    finding = next(item for item in report.findings if item.code == "invalid_gain_adjustment_bounds")
    assert finding.proposal_id == proposal.proposal_id
    assert "max_step_db must be an exact numeric value" in finding.details["validation_error"]


def test_replay_policy_validator_rejects_fader_proposal_with_missing_step_bound():
    base_manifest, _, _ = _build_manifest()
    proposal = ReplayDecisionProposal(
        proposal_id="fader::missing-bound::2",
        family="fader_adjustment",
        source_system="untrusted_restore",
        action_type="set_fader_db",
        target={"channel": 2, "kind": "fader"},
        current_state={"fader_db": -3.0},
        requested_state={"fader_db": -2.0},
        confidence=0.8,
        safety={},
        auto_apply_blocked=False,
    )
    ranking = rank_replay_proposals([proposal])
    graph_checkpoint = build_replay_graph_checkpoint(
        [proposal],
        ranking,
        metadata={"event_seq": 34, "stage": "policy_validator"},
    ).to_dict()
    manifest = build_replay_checkpoint_manifest(
        trim_snapshot=base_manifest.trim_snapshot,
        graph_checkpoint=graph_checkpoint,
        metadata=base_manifest.metadata,
    )

    report = validate_replay_policy_consistency(manifest)

    assert report.valid is False
    finding = next(
        item for item in report.findings if item.code == "invalid_fader_adjustment_bounds"
    )
    assert finding.proposal_id == proposal.proposal_id
    assert "max_step_db must be an exact numeric value" in finding.details["validation_error"]


def test_replay_policy_validator_detects_policy_signature_and_executor_mismatches():
    manifest, proposals, ranking = _build_manifest()
    graph_checkpoint = dict(manifest.graph_checkpoint)
    timeline_artifact = dict(graph_checkpoint["timeline_artifact"])
    timeline = [dict(item) for item in timeline_artifact["timeline"]]
    timeline[0]["replay_signature"] = "tampered-signature"
    timeline_artifact["timeline"] = timeline
    timeline_artifact["dry_run_only"] = False
    timeline_artifact["live_mixer_writes"] = True
    graph_checkpoint["timeline_artifact"] = timeline_artifact
    broken_manifest = build_replay_checkpoint_manifest(
        trim_snapshot={
            **manifest.trim_snapshot,
            "transport_policy": {
                **dict(manifest.trim_snapshot.get("transport_policy") or {}),
                "dry_run_only": True,
            },
        },
        graph_checkpoint=graph_checkpoint,
        critic_artifacts=manifest.critic_artifacts,
        metadata=manifest.metadata,
    )
    broken_executor = simulate_replay_decisions(
        proposals,
        selected_proposal_id=ranking.selected_proposal_id,
    ).to_dict()
    broken_executor["dry_run_only"] = False
    broken_executor["selected_proposal_id"] = "unknown::proposal"
    broken_executor["events"][0]["dry_run"] = False

    report = validate_replay_policy_consistency(
        broken_manifest,
        executor_result=broken_executor,
    )

    codes = {item.code for item in report.findings}
    assert report.valid is False
    assert "replay_policy_not_dry_run" in codes
    assert "replay_policy_live_writes_present" in codes
    assert "trim_transport_policy_mismatch" in codes
    assert "analyzer_signature_mismatch" in codes
    assert "executor_not_dry_run" in codes
    assert "executor_selected_proposal_mismatch" in codes
    assert "executor_event_not_dry_run" in codes


def test_replay_policy_validator_detects_selection_drift_against_baseline():
    baseline_manifest, _, _ = _build_manifest()
    candidate_manifest, _, _ = _build_manifest(
        selected_family_priority={"fader_adjustment": 2.0}
    )

    report = validate_replay_policy_consistency(
        candidate_manifest,
        baseline_manifest=baseline_manifest,
    )

    codes = {item.code for item in report.findings}
    assert report.valid is True
    assert "drift_selected_proposal_changed" in codes
    assert report.drift_summary["selected_proposal_changed"] is True


def test_replay_manifest_rejects_truthy_strings_missing_markers_and_tampering():
    manifest, _, _ = _build_manifest()
    payload = manifest.to_dict()
    payload["dry_run_only"] = "false"
    with pytest.raises(ValueError, match="JSON boolean"):
        ReplayCheckpointManifest.from_dict(payload)

    tampered = manifest.to_dict()
    tampered["graph_checkpoint"]["timeline_artifact"].pop("dry_run_only")
    report = validate_replay_policy_consistency(tampered)
    codes = {finding.code for finding in report.findings}
    assert report.valid is False
    assert "timeline_dry_run_marker_missing_or_invalid" in codes
    assert "manifest_integrity_mismatch" in codes
