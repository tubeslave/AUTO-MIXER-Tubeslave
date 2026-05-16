from __future__ import annotations

import json

from ai_mixing_pipeline.reports import write_shadow_mode_reports as package_write_shadow_mode_reports
from ai_mixing_pipeline.reports.shadow_report_writer import write_shadow_mode_reports
from tests.shadow_mode_report_fixtures import build_shadow_report_fixture_cases


def test_shadow_mode_reports_follow_fixture_schema_and_preserve_dry_run_evidence(tmp_path):
    for case in build_shadow_report_fixture_cases():
        case_dir = tmp_path / case["name"]
        report_paths = write_shadow_mode_reports(
            case_dir,
            case["manifest"],
            executor=case["executor"],
            validation_report=case["validation"],
            generated_at=case["generated_at"],
        )

        comparison = json.loads((case_dir / "shadow_mode_comparison.json").read_text(encoding="utf-8"))
        rollback = json.loads((case_dir / "rollback_simulation_report.json").read_text(encoding="utf-8"))
        readiness = json.loads((case_dir / "promotion_readiness_report.json").read_text(encoding="utf-8"))
        trajectory = json.loads((case_dir / "confidence_trajectory.json").read_text(encoding="utf-8"))
        summary = json.loads((case_dir / "shadow_mode_summary.json").read_text(encoding="utf-8"))
        expected = case["expected"]

        assert set(report_paths) == {
            "shadow_mode_comparison",
            "rollback_simulation",
            "promotion_readiness",
            "confidence_trajectory",
            "shadow_mode_summary",
        }
        assert comparison["selected_proposal_id"] == expected["selected_proposal_id"]
        assert comparison["selected_family"] == expected["selected_family"]
        assert comparison["operator_count"] == expected["operator_count"]
        assert comparison["ai_count"] == expected["ai_count"]
        if expected["operator_action_capture_id"] is None:
            assert comparison["operator_action_capture"] is None
        else:
            assert comparison["operator_action_capture"]["proposal_id"] == expected["operator_action_capture_id"]
        assert comparison["ai_would_do_this"]["proposal_id"] == expected["ai_would_do_this_id"]
        assert [item["code"] for item in comparison["disagreements"]] == expected["disagreement_codes"]
        assert comparison["disagreement_summary"]["disagreement_rate"] == expected["disagreement_rate"]
        assert comparison["disagreement_summary"]["classes"]["by_taxonomy"] == expected["disagreement_taxonomy_counts"]
        assert comparison["evidence"]["dry_run_only"] is True
        assert comparison["evidence"]["live_mixer_writes"] is False
        assert comparison["evidence"]["replay_correlation_ids"] == case["manifest"]["replay_correlation_ids"]

        if expected["operator_source_payload"] is None:
            assert comparison["operator_action_capture"] is None
        else:
            for key, value in expected["operator_source_payload"].items():
                assert comparison["operator_action_capture"]["source_payload"][key] == value
        for key, value in expected["ai_source_payload"].items():
            assert comparison["ai_would_do_this"]["source_payload"][key] == value

        assert rollback["trim_snapshot_id"] == case["manifest"]["trim_snapshot"]["snapshot_id"]
        assert rollback["graph_checkpoint_id"] == case["manifest"]["graph_checkpoint"]["checkpoint_id"]
        assert rollback["blocked_reasons"] == expected["blocked_reasons"]

        assert readiness["ready"] is expected["ready"]
        assert readiness["dry_run_only"] is True
        assert readiness["live_mixer_writes"] is False
        assert readiness["validation_counts"]["error"] == 0
        assert readiness["evidence"]["selected_proposal_id"] == expected["selected_proposal_id"]
        assert readiness["evidence"]["disagreement_codes"] == expected["disagreement_codes"]
        assert readiness["evidence"]["replay_correlation_ids"] == case["manifest"]["replay_correlation_ids"]
        assert readiness["evidence"]["validation_findings"]
        assert readiness["aggregation"]["disagreement_rate"] == expected["disagreement_rate"]
        assert readiness["aggregation"]["audit_coverage"]["complete"] is True
        assert readiness["aggregation"]["rollback_frequency"]["blocked_event_count"] == expected["blocked_event_count"]
        assert readiness["promotion_readiness_gates"]
        assert readiness["gate_summary"]["blocking_gate_count"] == 5

        assert trajectory["selected_proposal_id"] == expected["selected_proposal_id"]
        assert trajectory["selected_family"] == expected["selected_family"]
        assert trajectory["top_operator_proposal_id"] == expected["operator_action_capture_id"]
        assert trajectory["top_ai_proposal_id"] == expected["ai_would_do_this_id"]
        assert trajectory["trend_summary"]["direction"] == expected["confidence_direction"]
        assert trajectory["trend_summary"]["operator_vs_ai_delta"] == expected["confidence_delta"]

        assert summary["generated_at"] == case["generated_at"]
        assert summary["selected_proposal_id"] == expected["selected_proposal_id"]
        assert summary["disagreement_rate"] == expected["disagreement_rate"]
        assert summary["audit_coverage_rate"] == 1.0
        assert summary["files"]["comparison"] == "shadow_mode_comparison.json"


def test_reports_package_exports_shadow_report_writer() -> None:
    assert package_write_shadow_mode_reports is write_shadow_mode_reports
