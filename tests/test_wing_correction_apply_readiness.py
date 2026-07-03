from backend.wing_correction_apply_readiness import (
    build_wing_correction_apply_readiness_snapshot,
    get_correction_apply_family,
    render_wing_correction_apply_readiness_markdown,
)


def test_wing_correction_apply_readiness_keeps_live_boundary_to_fader_gain():
    snapshot = build_wing_correction_apply_readiness_snapshot()

    assert snapshot["wing_deployment_boundary"]["live_allowed_write_kinds"] == [
        "fader",
        "gain",
    ]
    assert snapshot["wing_deployment_boundary"]["direct_live_widening_allowed"] is False
    assert snapshot["summary"]["supervised_policy_family_ids"] == [
        "fader_balance",
        "gain_trim",
    ]
    assert snapshot["summary"]["live_allowed_family_ids"] == []

    for family in snapshot["families"]:
        if family["family_id"] in {"fader_balance", "gain_trim"}:
            assert family["policy_allows_supervised_apply"] is True
            assert family["can_apply_live"] is False
            assert family["current_stage"] == "supervised_candidate"
            assert "physical_rollback_proof_required" in family["blocked_by"]
        else:
            assert family["policy_allows_supervised_apply"] is False
            assert family["can_apply_live"] is False
            assert "separate_hardware_proof_required" in family["blocked_by"]


def test_pan_is_next_candidate_but_not_applyable_on_wing_boundary():
    snapshot = build_wing_correction_apply_readiness_snapshot()
    pan = get_correction_apply_family("pan_spatial")

    assert pan is not None
    assert snapshot["summary"]["next_blocked_family_id"] == "fader_balance"
    assert pan.can_apply_live is False
    assert pan.current_stage == "proposal_only"
    assert "pan" in pan.write_kinds
    assert "not_in_wing_deployment_live_write_kinds" in pan.blocked_by
    assert "current_pan_physical_readback_required" == pan.readback_contract
    assert "previous_pan_value_rollback_required" == pan.rollback_contract


def test_eq_dynamics_gate_fx_have_explicit_apply_blockers():
    snapshot = build_wing_correction_apply_readiness_snapshot()
    families = {family["family_id"]: family for family in snapshot["families"]}

    expected = {
        "eq_filtering": "eq_band_identity_required",
        "dynamics_compressor": "makeup_gain_headroom_guard_required",
        "gate_expander": "chatter_safety_required",
        "fx_sends": "fx_parameter_schema_required",
    }
    for family_id, blocker in expected.items():
        family = families[family_id]
        assert family["can_apply_live"] is False
        assert blocker in family["blocked_by"]
        assert family["proposal_status"] == "operator_review_only"
        assert family["hardware_proof"].startswith("missing_")


def test_readiness_markdown_states_no_broad_live_widening():
    snapshot = build_wing_correction_apply_readiness_snapshot(
        include_method_status=False
    )
    markdown = render_wing_correction_apply_readiness_markdown(snapshot)

    assert "Current supervised policy write kinds: `fader`, `gain`" in markdown
    assert "Direct live widening allowed: `false`" in markdown
    assert "Release-ready live families: `0`" in markdown
    assert "Promote exactly one family at a time" in markdown
    assert (
        "| EQ, HPF, and filtering | `proposal_only` | `false` | `false` |"
        in markdown
    )
