from backend.shadow_gain_rides import build_shadow_gain_ride_candidates, shadow_gain_ride_trace


def _vocal_channel():
    return {
        "id": 2,
        "name": "Lead Vox",
        "role": "vocal",
        "group": "VOCALS",
        "status": "active",
        "lufs": -28.0,
        "peak_db": -12.0,
        "control_state": {"fader_db": -8.0},
    }


def _assert_operator_denial(payload):
    assert payload["can_apply_live"] is False
    assert payload["can_import_to_queue"] is False
    assert payload["queue_import_allowed"] is False
    assert payload["live_apply_allowed"] is False
    assert payload["read_only_or_dry_run_only"] is True
    assert payload["live_authority_granted"] is False
    assert payload["readiness_closure_allowed"] is False
    assert payload["queue_action"] == "not_queueable_until_promoted"
    assert payload["live_write"] is False
    assert payload["queue_mutation"] is False
    assert payload["mutates_mixer"] is False


def test_shadow_gain_ride_reports_current_wing_boundary_without_live_promotion():
    candidates = build_shadow_gain_ride_candidates([_vocal_channel()])

    assert len(candidates) == 1
    candidate = candidates[0]
    _assert_operator_denial(candidate)
    assert candidate["safety_limits"]["live_promotion_allowed"] is False
    assert candidate["safety_limits"]["allowed_live_write_kinds"] == ["fader", "gain"]

    trace = shadow_gain_ride_trace(candidate)
    _assert_operator_denial(trace)
    _assert_operator_denial(trace["metadata"])
    assert trace["proof_status"] == "shadow_only"
    assert trace["safety_limits"]["allowed_live_write_kinds"] == ["fader", "gain"]
