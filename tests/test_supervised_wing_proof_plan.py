from pathlib import Path

from backend.supervised_wing_proof_plan import build_supervised_wing_proof_plan


def evidence_snapshot(*rows):
    return {
        "type": "supervised_promotion_evidence",
        "safety": {
            "live_widening_allowed": False,
            "automatic_safety_update_allowed": False,
        },
        "rows": list(rows),
    }


def evidence_row(value_type: str, *, review_candidate: bool = False, blockers=None):
    return {
        "id": f"{value_type}_supervised_evidence",
        "value_type": value_type,
        "safety_row_review_candidate": review_candidate,
        "blockers": list(blockers or []),
    }


def test_supervised_wing_proof_plan_module_is_transport_free():
    source = Path("backend/supervised_wing_proof_plan.py").read_text(encoding="utf-8")

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        "websockets",
        "socket",
        ".send(",
        "mixer_client",
        "set_fader",
        "set_gain",
        "apply_operator_proposal",
        "requested_change",
        "requested_changes",
    ]
    for token in forbidden:
        assert token not in source


def test_plan_builds_gain_and_fader_physical_proof_steps_without_execution_authority():
    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(
            evidence_row("gain", blockers=["physical_candidate_missing"]),
            evidence_row("fader", blockers=["physical_candidate_missing", "blocked_artifacts_for_value_type"]),
        ),
        backend_url="ws://proof-backend",
        artifact_root="artifacts/proof",
        channels_by_value_type={"gain": (3,), "fader": (2, 16)},
    )

    assert plan["type"] == "supervised_wing_proof_plan"
    assert plan["status"] == "operator_proof_required"
    assert plan["safety"] == {
        "read_only": True,
        "live_write": False,
        "execution_allowed_by_plan": False,
        "live_widening_allowed": False,
        "automatic_safety_update_allowed": False,
        "full_autonomous_ready": False,
        "operator_required": True,
    }
    assert plan["summary"]["step_count"] == 3
    assert plan["summary"]["planned_value_types"] == ["fader", "gain"]
    assert plan["summary"]["planned_channels"] == [3, 2, 16]

    gain_step = plan["steps"][0]
    assert gain_step["id"] == "gain_physical_proof_ch3"
    assert gain_step["live_execution_required"] is True
    assert gain_step["execution_allowed_by_plan"] is False
    assert gain_step["operator_approval_required"] is True
    assert gain_step["delta_db"] == -0.25
    assert gain_step["command"][0] == "python3"
    assert "--execute" in gain_step["command"]
    assert "--prove-blocked-apply" in gain_step["command"]
    assert "--prove-emergency-stop" in gain_step["command"]
    assert "ws://proof-backend" in gain_step["command"]
    assert "Use gain-down proof only; gain-up remains out of scope for this plan." in gain_step["prerequisites"]

    fader_step = plan["steps"][1]
    assert "Review blocked prior proof artifacts for this value type before trusting a new result." in fader_step["prerequisites"]
    assert "blocked_artifacts_for_value_type" in fader_step["evidence_blockers_to_review"]


def test_plan_skips_value_type_that_already_has_manual_review_candidate():
    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(
            evidence_row("gain", review_candidate=True),
            evidence_row("fader", blockers=["physical_candidate_missing"]),
        ),
        channels_by_value_type={"fader": (4,)},
    )

    assert plan["status"] == "operator_proof_required"
    assert [step["value_type"] for step in plan["steps"]] == ["fader"]
    assert plan["steps"][0]["channel"] == 4


def test_plan_is_manual_review_ready_when_all_value_types_have_candidates():
    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(
            evidence_row("gain", review_candidate=True),
            evidence_row("fader", review_candidate=True),
        )
    )

    assert plan["status"] == "manual_review_ready"
    assert plan["summary"]["step_count"] == 0
    assert plan["steps"] == []
    assert plan["safety"]["execution_allowed_by_plan"] is False


def test_invalid_artifacts_block_plan_generation():
    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(
            evidence_row("gain", blockers=["invalid_proof_summaries_present"]),
            evidence_row("fader", blockers=["physical_candidate_missing"]),
        )
    )

    assert plan["status"] == "blocked_by_invalid_artifacts"
    assert plan["steps"] == []
    assert plan["summary"]["global_blockers"] == ["invalid_proof_summaries_present"]
