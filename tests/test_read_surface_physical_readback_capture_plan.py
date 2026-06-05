from pathlib import Path

from backend.read_surface_inventory import ReadSurface
from backend.read_surface_physical_readback_capture_plan import (
    READ_SURFACE_PHYSICAL_READBACK_CAPTURE_PLAN_METHOD_ID,
    build_read_surface_physical_readback_capture_plan,
)
from backend.read_surface_remediation_plan import build_read_surface_remediation_plan


def test_capture_plan_module_stays_transport_free():
    source = Path("backend/read_surface_physical_readback_capture_plan.py").read_text(
        encoding="utf-8"
    )

    forbidden = [
        "WingClient",
        "DLiveClient",
        "EnhancedOSCClient",
        "import socket",
        "websockets",
        ".send(",
        "mixer_client",
        "set_fader",
        "set_gain",
        "set_eq",
        "set_compressor",
        "apply_operator_proposal",
    ]
    for token in forbidden:
        assert token not in source


def test_capture_plan_targets_current_next_physical_readback_gap_without_execution_authority():
    plan = build_read_surface_physical_readback_capture_plan()

    assert plan["type"] == "read_surface_physical_readback_capture_plan"
    assert plan["method_id"] == READ_SURFACE_PHYSICAL_READBACK_CAPTURE_PLAN_METHOD_ID
    assert plan["status"] == "operator_capture_required"
    assert plan["safety"] == {
        "read_only": True,
        "live_write": False,
        "mutates_console_state": False,
        "queries_console": False,
        "executes_tests": False,
        "operator_required": True,
        "execution_allowed_by_plan": False,
        "live_widening_allowed": False,
        "live_authority_granted": False,
        "full_autonomous_ready": False,
    }
    assert plan["summary"]["next_physical_readback_surface_id"] == "raw_mixer_state_cache"
    assert (
        plan["summary"]["next_physical_readback_action_id"]
        == "remediate_raw_mixer_state_cache"
    )
    assert plan["summary"]["target_action_consistency"] is True
    assert plan["summary"]["capture_step_count"] == 1
    assert "physical_readback_fixture" in plan["summary"]["required_artifact_contracts"]
    assert (
        "raw_mixer_state_physical_readback_contract"
        in plan["summary"]["required_artifact_contracts"]
    )
    assert "operator_must_capture_and_review_physical_readback_artifact" in plan["summary"]["blocked_by"]
    assert plan["summary"]["source_remediation_status"] == "read_remediation_required"
    assert plan["summary"]["source_fixture_status"] == "fixture_evidence_incomplete"

    step = plan["steps"][0]
    assert step["id"] == "capture_raw_mixer_state_cache_physical_readback"
    assert step["surface_id"] == "raw_mixer_state_cache"
    assert step["remediation_action_id"] == "remediate_raw_mixer_state_cache"
    assert step["target_console_surface"] == "all_cached_addresses"
    assert step["requires_connection"] is True
    assert step["operator_required"] is True
    assert step["execution_allowed_by_plan"] is False
    assert step["live_write"] is False
    assert step["mutates_console_state"] is False
    assert step["queries_console"] is False
    assert step["executes_tests"] is False
    assert step["contract_method_id"] == "raw_mixer_state_physical_readback_contract"
    assert "physical_readback_fixture_missing" in step["blockers_to_clear"]
    assert any("Raw mixer-state physical readback artifact" in item for item in step["expected_artifacts"])
    assert any("do not apply fader" in item for item in step["forbidden_actions"])
    assert any("do not transmit OSC" in item for item in step["forbidden_actions"])
    assert any("do not mark cached state as physical proof" in item for item in step["forbidden_actions"])


def test_capture_plan_reports_complete_when_no_physical_gap_remains():
    surface = ReadSurface(
        id="complete_fx",
        label="Complete FX",
        target_console_surface="fx",
        status="queryable",
        read_mode="osc_read_query",
        paths=("backend/wing_client.py",),
        entrypoints=("get_fx_overview",),
        read_fields=("fx_model",),
        current_coverage="Complete fixture row.",
    )
    fixture_snapshot = {
        "type": "read_surface_fixture_evidence",
        "status": "fixture_evidence_complete",
        "summary": {
            "physical_fixture_missing_surface_ids": [],
            "next_physical_readback_surface_id": None,
            "claim": "complete fixture evidence",
        },
        "surfaces": [
            {
                "surface_id": "complete_fx",
                "status": "fixture_evidence_present",
                "remaining_gaps": [],
            }
        ],
    }
    remediation_plan = build_read_surface_remediation_plan(
        [surface],
        fixture_evidence_snapshot=fixture_snapshot,
    )

    plan = build_read_surface_physical_readback_capture_plan(
        fixture_evidence_snapshot=fixture_snapshot,
        remediation_plan=remediation_plan,
    )

    assert plan["status"] == "read_surface_physical_readback_complete"
    assert plan["summary"]["next_physical_readback_surface_id"] is None
    assert plan["summary"]["next_physical_readback_action_id"] is None
    assert plan["summary"]["target_action_consistency"] is False
    assert plan["summary"]["capture_step_count"] == 0
    assert plan["summary"]["blocked_by"] == []
    assert plan["steps"] == []
    assert plan["safety"]["execution_allowed_by_plan"] is False
    assert plan["safety"]["live_authority_granted"] is False
    assert plan["safety"]["full_autonomous_ready"] is False


def test_capture_plan_normalizes_mismatched_remediation_target_without_claiming_execution():
    remediation_plan = build_read_surface_remediation_plan()
    mismatched = {
        **remediation_plan,
        "summary": {
            **remediation_plan["summary"],
            "next_physical_readback_surface_id": "dlive_state",
            "next_physical_readback_action_id": "remediate_dlive_state",
        },
    }

    plan = build_read_surface_physical_readback_capture_plan(
        remediation_plan=mismatched,
    )

    assert plan["status"] == "operator_capture_required"
    assert plan["summary"]["next_physical_readback_surface_id"] == "dlive_state"
    assert plan["summary"]["target_action_consistency"] is False
    assert "read_surface_next_physical_readback_target_inconsistent" in plan["summary"]["blocked_by"]
    assert plan["safety"]["execution_allowed_by_plan"] is False
    assert plan["safety"]["live_authority_granted"] is False
    assert plan["safety"]["live_write"] is False
