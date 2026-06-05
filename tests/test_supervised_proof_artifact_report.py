import json
from pathlib import Path

from tools import evaluate_supervised_proof_artifacts as report_tool


def write_summary(run_dir: Path, payload: dict) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "summary.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def dry_run_summary(proof_id: str = "dry-gain-proof") -> dict:
    return {
        "run_id": proof_id,
        "status": "passed",
        "channel": 3,
        "delta_db": -0.25,
        "physical_wing": False,
        "write_policy": "fake_mixer_supervised_gate_only",
        "old_value": 1.0,
        "target_value": 0.75,
        "apply_result": {
            "apply_result": {
                "status": "applied",
                "channel": 3,
                "value_type": "gain",
                "old_value": 1.0,
                "effective_value": 0.75,
                "readback_status": "verified_from_physical_readback",
                "pre_read_status": "verified_from_physical_readback",
                "proof_status": "verified",
                "rollback_available": True,
                "rollback_value": 1.0,
            },
        },
        "rollback_result": {
            "rollback_status": "applied",
            "rollback_readback_status": "verified_from_physical_readback",
            "rollback_proof_status": "verified",
        },
        "cooldown_apply_result": {"apply_result": {"reason": "cooldown"}},
        "post_emergency_apply_result": {"apply_result": {"reason": "emergency_stop_active"}},
    }


def physical_summary(proof_id: str = "physical-fader-proof") -> dict:
    payload = dry_run_summary(proof_id)
    payload.update(
        {
            "channel": 4,
            "value_type": "fader",
            "physical_wing": True,
            "old_value": -12.0,
            "target_value": -12.25,
            "delta_db": -0.25,
            "pre_read_source": "physical_readback",
            "readback_source": "physical_readback",
            "rollback_readback_source": "physical_readback",
        }
    )
    payload["apply_result"]["apply_result"].update(
        {
            "channel": 4,
            "value_type": "fader",
            "old_value": -12.0,
            "effective_value": -12.25,
            "rollback_value": -12.0,
            "pre_read_source": "physical_readback",
            "readback_source": "physical_readback",
        }
    )
    return payload


def legacy_execute_wing_summary(proof_id: str = "legacy-physical-gain-proof") -> dict:
    payload = dry_run_summary(proof_id)
    payload.pop("physical_wing")
    payload.update(
        {
            "run_id": "20260523T104641Z_wing_operator_proposal_ch4_gain",
            "execute": True,
            "write_policy": "apply_operator_proposal_only",
            "pre_read_source": "physical_readback",
            "readback_source": "physical_readback",
            "rollback_readback_source": "physical_readback",
            "operator_reason": "operator approved small gain-down physical proof",
        }
    )
    payload["apply_result"]["apply_result"].update(
        {
            "pre_read_source": "physical_readback",
            "readback_source": "physical_readback",
        }
    )
    return payload


def test_proof_artifact_report_tool_is_read_only_and_transport_free():
    source = Path("tools/evaluate_supervised_proof_artifacts.py").read_text(encoding="utf-8")

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        "websockets",
        "socket",
        ".send(",
        "mixer_client",
        "set_fader",
        "set_gain",
        "requested_change",
        "requested_changes",
    ]
    for token in forbidden:
        assert token not in source


def test_discover_summary_paths_recurses_and_only_returns_summary_json(tmp_path: Path):
    first = write_summary(tmp_path / "a", dry_run_summary("a-proof"))
    second = write_summary(tmp_path / "nested" / "b", dry_run_summary("b-proof"))
    (tmp_path / "nested" / "b" / "events.jsonl").write_text("{}", encoding="utf-8")

    paths = report_tool.discover_summary_paths([tmp_path])

    assert paths == sorted([first, second])


def test_report_distinguishes_dry_run_physical_ignored_and_invalid_summaries(tmp_path: Path):
    dry = write_summary(tmp_path / "dry", dry_run_summary())
    physical = write_summary(tmp_path / "physical", physical_summary())
    legacy = write_summary(tmp_path / "legacy", legacy_execute_wing_summary())
    ignored = write_summary(tmp_path / "offline_validation", {"type": "offline_validation", "status": "passed"})
    invalid_dir = tmp_path / "invalid"
    invalid_dir.mkdir()
    invalid = invalid_dir / "summary.json"
    invalid.write_text("{not-json", encoding="utf-8")

    report = report_tool.build_supervised_proof_artifact_report(
        [dry, physical, legacy, ignored, invalid],
        require_physical=True,
    )

    assert report["status"] == "blocked"
    assert report["safety"]["live_write"] is False
    assert report["summary"]["source_summary_count"] == 5
    assert report["summary"]["artifact_count"] == 3
    assert report["summary"]["ignored_summary_count"] == 1
    assert report["summary"]["invalid_summary_count"] == 1
    assert report["summary"]["status_counts"]["dry_run_ready"] == 1
    assert report["summary"]["status_counts"]["supervised_candidate"] == 1
    assert report["summary"]["status_counts"]["blocked"] == 1
    assert report["summary"]["proof_mode_counts"]["legacy_ambiguous"] == 1
    assert report["summary"]["promotion_allowed_proof_ids"] == ["physical-fader-proof"]
    assert report["summary"]["dry_run_ready_proof_ids"] == ["dry-gain-proof"]
    assert report["summary"]["blocked_proof_ids"] == ["20260523T104641Z_wing_operator_proposal_ch4_gain"]
    assert report["summary"]["autonomous_ready_proof_ids"] == []
    assert report["ignored_summaries"] == [str(ignored)]
    assert report["invalid_summaries"] == [{"artifact_path": str(invalid), "reason": "json_load_error:JSONDecodeError"}]


def test_exit_code_requires_minimum_physical_candidates(tmp_path: Path):
    dry = write_summary(tmp_path / "dry", dry_run_summary())
    report = report_tool.build_supervised_proof_artifact_report([dry], require_physical=True)

    assert report_tool.exit_code_for_report(report, min_promotion_candidates=0) == 0
    assert report_tool.exit_code_for_report(report, min_promotion_candidates=1) == 2


def test_main_writes_report_and_returns_success_for_physical_candidate(tmp_path: Path):
    write_summary(tmp_path / "physical", physical_summary())
    write_summary(tmp_path / "legacy", legacy_execute_wing_summary())
    output_path = tmp_path / "report.json"

    exit_code = report_tool.main(
        [
            "--artifact-root",
            str(tmp_path),
            "--require-physical",
            "--min-promotion-candidates",
            "1",
            "--output",
            str(output_path),
            "--quiet",
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["report_path"] == str(output_path)
    assert report["summary"]["promotion_allowed_proof_ids"] == ["physical-fader-proof"]
    assert report["summary"]["autonomous_ready_proof_ids"] == []
    assert report["promotion_evidence"]["safety"]["live_widening_allowed"] is False
    assert report["promotion_evidence"]["summary"]["safety_row_review_candidate_ids"] == [
        "fader_supervised_evidence"
    ]
    assert report["proof_remediation_plan"]["safety"]["execution_allowed_by_plan"] is False
    assert report["proof_remediation_plan"]["status"] == "manual_remediation_required"
    assert report["proof_remediation_plan"]["summary"]["legacy_ambiguous_count"] == 1
    assert report["proof_remediation_plan"]["actions"][0]["action_type"] == "rerun_current_physical_schema"
    assert report["autonomy_evidence_overlay"]["safety"]["live_widening_allowed"] is False
    assert report["autonomy_evidence_overlay"]["summary"]["autonomous_ready_target_ids"] == []
    assert report["autonomy_evidence_overlay"]["summary"]["manual_safety_review_candidate_target_ids"] == [
        "fader_supervised_promotion"
    ]
    assert report["autonomy_evidence_overlay"]["summary"]["remediation_required_target_ids"] == [
        "gain_trim_supervised_promotion"
    ]
    assert report["autonomy_mode_transition_gate"]["safety"]["automatic_mode_change_allowed"] is False
    assert report["autonomy_mode_transition_gate"]["summary"]["autonomous_apply_allowed_modes"] == []
    assert report["autonomy_mode_transition_gate"]["summary"]["full_auto_readiness_gap_bucket_counts"]["read_surface"] > 0
    assert report["autonomy_mode_transition_gate"]["summary"]["supervised_current_live_write_kinds"] == [
        "fader",
        "gain",
    ]
    assert report["autonomy_mode_transition_gate"]["rows"][2]["mode"] == "supervised"
    assert report["autonomy_mode_transition_gate"]["rows"][2]["status"] == "limited_available"
    assert report["autonomy_mode_transition_gate"]["rows"][3]["mode"] == "auto"
    assert report["autonomy_mode_transition_gate"]["rows"][3]["status"] == "blocked"
    gap_rollup = report["summary"]["autonomy_gap_rollup"]
    assert gap_rollup["read_only"] is True
    assert gap_rollup["live_write"] is False
    assert gap_rollup["live_widening_allowed"] is False
    assert gap_rollup["full_autonomous_ready"] is False
    assert gap_rollup["source"] == "autonomy_mode_transition_gate.rows[mode=auto]"
    assert gap_rollup["full_autonomous_blocked"] is True
    assert gap_rollup["full_auto_gap_bucket_counts"]["read_surface"] > 0
    assert gap_rollup["full_auto_gap_bucket_counts"]["write_surface"] > 0
    assert "scene_preset_supervised_promotion" in gap_rollup["full_auto_gap_target_ids_by_bucket"]["read_surface"]
    assert gap_rollup["evidence_requirement_ids_by_action_family"]["bus_group_matrix"] == [
        "bus_group_assignment_cache_freshness",
        "bus_group_fader_readback",
        "matrix_send_readback",
        "main_bus_true_peak_headroom",
        "bus_group_physical_fixture",
        "hardware_readback_and_rollback_proof",
    ]
    assert gap_rollup["evidence_requirement_ids_by_action_family"]["mute"] == [
        "current_mute_cache_freshness",
        "mute_owner_intent_evidence",
        "signal_return_or_idle_window_evidence",
        "manual_mute_authority_review",
        "mute_physical_fixture",
        "hardware_readback_rollback_and_emergency_proof",
    ]
    assert "Add bus/group/matrix safety review" in (
        gap_rollup["next_required_evidence_by_action_family"]["bus_group_matrix"][0]
    )
    assert "fresh mute-state intent" in (
        gap_rollup["next_required_evidence_by_action_family"]["mute"][0]
    )
    assert gap_rollup["supervised_live_apply_allowed"] is True
    assert gap_rollup["supervised_current_live_write_kinds"] == ["fader", "gain"]
    assert "Full Autonomous remains blocked" in gap_rollup["claim"]
    assert report["next_wing_proof_plan"]["safety"]["execution_allowed_by_plan"] is False
    assert report["next_wing_proof_plan"]["status"] == "operator_proof_required"
    assert report["next_wing_proof_plan"]["summary"]["planned_value_types"] == ["gain"]
