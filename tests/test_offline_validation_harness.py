from pathlib import Path

from backend.offline_validation_harness import (
    OfflineChannelFixture,
    OfflineMixerState,
    review_proposal_safety,
    run_offline_validation,
    simulate_proposal_apply,
    write_offline_validation_artifacts,
)


def test_offline_validation_runs_without_live_writes_and_blocks_non_supervised_modes():
    result = run_offline_validation()

    assert result["status"] == "passed"
    assert result["connection_policy"] == "no_hardware_no_network_no_live_writes"
    assert result["live_write"] is False
    assert result["summary"]["supervised_proposal_count"] > 0
    assert result["summary"]["supervised_failure_count"] == 0
    assert result["summary"]["non_supervised_modes_blocked"] is True

    by_mode = {item["mode"]: item for item in result["mode_results"]}
    assert by_mode["analyze"]["applied_simulation_count"] == 0
    assert by_mode["assist"]["applied_simulation_count"] == 0
    assert by_mode["auto"]["applied_simulation_count"] == 0
    assert by_mode["supervised"]["applied_simulation_count"] == by_mode["supervised"]["proposal_count"]
    assert by_mode["supervised"]["emergency_stop_proven"] is True

    for proposal_result in by_mode["supervised"]["proposal_results"]:
        assert proposal_result["live_write"] is False
        assert proposal_result["safety_review"]["approved"] is True
        assert proposal_result["apply_result"]["proof_status"] == "verified"
        assert proposal_result["apply_result"]["readback_status"] == "verified_from_simulated_readback"
        assert proposal_result["apply_result"]["rollback_readback_status"] == "verified_from_simulated_readback"
        assert proposal_result["post_emergency_apply"]["reason"] == "emergency_stop_active"


def test_offline_validation_restores_fixture_values_after_simulated_rollback():
    fixture = [
        OfflineChannelFixture(
            16,
            "VOCAL",
            "Vocal",
            "VOCALS",
            fader_db=0.0,
            gain_db=-11.9,
            rms_db=-16.0,
            peak_db=-3.4,
            lufs=-17.7,
        )
    ]

    result = run_offline_validation(fixture, modes=("supervised",), total_channels=16)
    final_state = result["mode_results"][0]["final_state"]

    assert result["status"] == "passed"
    assert final_state["/ch/16/fdr"] == 0.0
    assert final_state["/ch/16/in/set/trim"] == -11.9


def test_offline_safety_review_blocks_upward_or_unsupported_changes():
    upward = {
        "id": "unsafe:fader_up",
        "requested_change": {
            "value_type": "fader",
            "channel": 1,
            "current_value": -5.0,
            "value": -3.0,
            "rollback_value": -5.0,
        },
    }
    unsupported = {
        "id": "unsafe:eq",
        "requested_change": {
            "value_type": "eq",
            "channel": 1,
            "current_value": 0.0,
            "value": -2.0,
            "rollback_value": 0.0,
        },
    }

    upward_review = review_proposal_safety(upward)
    unsupported_review = review_proposal_safety(unsupported)

    assert upward_review["approved"] is False
    assert "offline_harness_allows_downward_changes_only" in upward_review["blockers"]
    assert unsupported_review["approved"] is False
    assert "unsupported_value_type:eq" in unsupported_review["blockers"]


def test_simulated_apply_returns_blocked_when_safety_review_failed():
    mixer = OfflineMixerState([
        OfflineChannelFixture(1, "KICK", "Kick", "DRUMS", fader_db=-5.0),
    ])
    proposal = {
        "id": "unsafe:fader_up",
        "requested_change": {
            "value_type": "fader",
            "channel": 1,
            "current_value": -5.0,
            "value": -3.0,
            "rollback_value": -5.0,
        },
    }
    review = review_proposal_safety(proposal)
    result = simulate_proposal_apply(mixer, proposal, review)

    assert result["status"] == "blocked"
    assert result["reason"] == "safety_review_blocked"
    assert mixer.read(1, "fader") == -5.0


def test_offline_validation_writes_summary_and_events(tmp_path):
    result = run_offline_validation(modes=("supervised",))
    paths = write_offline_validation_artifacts(result, artifact_root=tmp_path, run_id="offline-test")

    summary = Path(paths["summary_path"])
    events = Path(paths["events_path"])

    assert summary.exists()
    assert events.exists()
    assert "offline_proposal_result" in events.read_text()


def test_offline_validation_harness_source_has_no_live_transport_imports():
    source = Path("backend/offline_validation_harness.py").read_text()
    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        "websockets",
        "socket",
        ".send(",
        "set_eq(",
        "set_compressor(",
    ]
    for token in forbidden:
        assert token not in source
