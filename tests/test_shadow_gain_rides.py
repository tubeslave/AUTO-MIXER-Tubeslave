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


def test_shadow_gain_ride_reports_current_wing_boundary_without_live_promotion():
    candidates = build_shadow_gain_ride_candidates([_vocal_channel()])

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["live_apply_allowed"] is False
    assert candidate["safety_limits"]["live_promotion_allowed"] is False
    assert candidate["safety_limits"]["allowed_live_write_kinds"] == ["fader", "gain"]

    trace = shadow_gain_ride_trace(candidate)
    assert trace["proof_status"] == "shadow_only"
    assert trace["safety_limits"]["allowed_live_write_kinds"] == ["fader", "gain"]
