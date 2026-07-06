import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from auto_fx import AutoFXPlanner


class FakeMixer:
    def __init__(self):
        self.calls = []

    def set_fx_model(self, slot, model):
        self.calls.append(("set_fx_model", slot, model))

    def set_fx_on(self, slot, on):
        self.calls.append(("set_fx_on", slot, on))

    def set_fx_mix(self, slot, mix):
        self.calls.append(("set_fx_mix", slot, mix))

    def set_fx_parameter(self, slot, parameter, value):
        self.calls.append(("set_fx_parameter", slot, parameter, value))

    def set_fx_return_level(self, bus, level_db):
        self.calls.append(("set_fx_return_level", bus, level_db))

    def set_channel_send(self, channel, bus, level=None, on=None, mode=None):
        self.calls.append(("set_channel_send", channel, bus, level, on, mode))


def _assert_top_level_denial(result):
    assert result["live_write"] is False
    assert result["queue_mutation"] is False
    assert result["can_apply_live"] is False
    assert result["can_import_to_queue"] is False
    assert result["queue_import_allowed"] is False
    assert result["live_apply_allowed"] is False
    assert result["read_only_or_dry_run_only"] is True
    assert result["live_authority_granted"] is False
    assert result["readiness_closure_allowed"] is False
    assert result["queue_action"] == "not_queueable_until_promoted"
    assert result["mutates_mixer"] is False


def test_auto_fx_plan_contains_spatial_buses_and_priority_sends():
    planner = AutoFXPlanner(tempo_bpm=120.0)
    plan = planner.create_plan({
        1: "lead_vocal",
        2: "backing_vocal",
        3: "snare",
        4: "kick",
        5: "electric_guitar",
    })

    bus_names = {bus.name for bus in plan.buses}
    assert {"Vocal Plate", "Drum Room", "Tempo Delay", "Mod Doubler"} <= bus_names

    lead_sends = [send for send in plan.sends if send.channel_id == 1]
    assert {send.bus_id for send in lead_sends} == {13, 15}
    assert all(send.post_fader for send in plan.sends)

    kick_sends = [send for send in plan.sends if send.channel_id == 4]
    assert kick_sends == []


def test_tempo_delay_uses_tempo_for_dotted_eighth_and_quarter():
    planner = AutoFXPlanner(tempo_bpm=100.0)
    plan = planner.create_plan({1: "lead_vocal"})
    delay_bus = next(bus for bus in plan.buses if bus.name == "Tempo Delay")

    assert delay_bus.params["left_delay_ms"] == 450.0
    assert delay_bus.params["right_delay_ms"] == 600.0
    assert delay_bus.return_level_db == -0.5
    assert delay_bus.duck_source == "lead_vocal"


def test_default_fx_returns_are_audible_live_variant():
    planner = AutoFXPlanner(tempo_bpm=120.0)
    plan = planner.create_plan({1: "lead_vocal", 2: "snare"})

    levels = {bus.name: bus.return_level_db for bus in plan.buses}
    assert levels["Vocal Plate"] == -4.0
    assert levels["Drum Room"] == -4.0
    assert levels["Tempo Delay"] == -0.5
    assert levels["Mod Doubler"] == -11.0


def test_apply_to_mixer_is_fail_closed_and_returns_blocked_fx_commands():
    planner = AutoFXPlanner(tempo_bpm=120.0)
    plan = planner.create_plan({1: "lead_vocal", 2: "snare"})
    mixer = FakeMixer()

    result = planner.apply_to_mixer(mixer, plan)

    assert result["applied"] is False
    assert result["blocked"] is True
    assert result["calls"] == []
    assert result["reason"] == "fx_direct_apply_quarantined"
    assert result["write_policy"] == "proposal_only_quarantined"
    assert result["live_write_allowed"] is False
    _assert_top_level_denial(result)
    assert result["mixer_client_present"] is True
    assert mixer.calls == []

    blocked = result["blocked_calls"]
    assert all(call["cmd"] != "set_fx_on" for call in blocked)
    assert {"cmd": "set_fx_model", "slot": "FX1", "model": "PLATE"} in blocked
    assert {"cmd": "set_fx_model", "slot": "FX2", "model": "ROOM"} in blocked
    assert {
        "cmd": "set_channel_send",
        "channel": 1,
        "bus": 13,
        "level_db": -18.0,
        "on": 1,
        "mode": "POST",
    } in blocked
    assert {
        "cmd": "set_channel_send",
        "channel": 2,
        "bus": 14,
        "level_db": -17.0,
        "on": 1,
        "mode": "POST",
    } in blocked
    correction = result["operator_correction"]
    assert correction["objective_family"] == "fx_sends"
    assert correction["future_wing_readback_contract"]["target_kind"] == "send_or_fx_parameter"
    assert correction["bounded_delta"]["action_type"] == "auto_fx_apply_blocked"
    assert correction["bounded_delta"]["parameter_deltas"]["target_send_db"] == -17.0
    assert correction["bounded_delta"]["parameter_deltas"]["return_level_db"] == -0.5
    assert correction["bounded_delta"]["parameter_deltas"]["send_level_db"] == -17.0
    assert correction["confidence"] == 0.4
    assert correction["confidence_review"]["status"] == (
        "low_confidence_operator_review_required"
    )
    assert correction["source_evidence"]["evidence_completeness"]["status"] == (
        "complete"
    )
    assert correction["source_evidence"]["evidence_completeness"]["missing"] == []
    assert {
        "current_fx_send_return_cache_freshness",
        "current_fx_send_return_cache_no_live_authority",
        "current_fx_parameter_schema_hints",
        "fx_parameter_schema_no_live_authority",
        "source_identity_downstream_trust",
        "old_value_source",
        "rollback_fx_readback",
    } <= set(correction["source_evidence"]["required_evidence_fields"])
    assert correction["evidence_gate"]["status"] == (
        "required_evidence_complete_operator_review_still_required"
    )
    assert correction["operator_proposal_review_gate"]["evidence_gate_status"] == (
        "required_evidence_complete_operator_review_still_required"
    )
    assert correction["read_only_or_dry_run_only"] is True
    assert correction["live_apply_allowed"] is False
    assert correction["queue_import_allowed"] is False
    assert "monitor_safety" in correction["headroom_feedback_true_peak_checks"][
        "required_checks"
    ]
    assert correction["headroom_feedback_true_peak_checks"]["evidence_checks"][
        "current_fx_send_return_cache_no_live_authority"
    ] is True
    source_evidence = correction["source_evidence"]["evidence"]
    _assert_top_level_denial(source_evidence)
    assert source_evidence["send_level_limit"] == {
        "status": "passed",
        "min_send_db": -24.0,
        "max_send_db": -17.0,
        "allowed_min_send_db": -36.0,
        "allowed_max_send_db": -12.0,
        "post_fader_sends_only": True,
        "source": "legacy_auto_fx_plan_blocked_calls",
    }
    assert source_evidence["fx_return_headroom"] == {
        "status": "blocked_until_physical_return_readback",
        "planned_loudest_return_db": -0.5,
        "min_return_headroom_db": 3.0,
        "return_level_safety_status": "blocked",
        "source": "cached_or_physical_fx_read_required_before_promotion",
        "required_before_live_promotion": True,
    }
    assert source_evidence["return_level_safety"] == {
        "status": "blocked",
        "loudest_return_db": -0.5,
        "max_return_level_db": -3.0,
        "source": "legacy_auto_fx_plan_blocked_calls",
        "physical_return_readback_required_before_live_promotion": True,
    }
    assert source_evidence["dry_wet_bounds"] == {
        "fx_mix_percent": 100.0,
        "post_fader_sends_only": True,
        "loudest_send_db": -17.0,
        "loudest_return_db": -0.5,
        "send_level_limit_status": "passed",
        "return_level_safety_status": "blocked",
    }
    evidence_checks = correction["headroom_feedback_true_peak_checks"][
        "evidence_checks"
    ]
    for key in (
        "send_level_limit",
        "fx_return_headroom",
        "return_level_safety",
        "dry_wet_bounds",
        "monitor_safety",
        "feedback_risk",
        "intelligibility_risk",
        "old_value_source",
        "current_fx_send_return_cache_freshness",
        "current_fx_send_return_cache_no_live_authority",
        "fx_parameter_schema_no_live_authority",
        "current_fx_parameter_schema_no_live_authority",
        "current_fx_parameter_schema_hints",
        "source_identity_downstream_trust",
        "rollback_fx_readback",
    ):
        assert evidence_checks[key] == source_evidence[key]
    assert source_evidence["monitor_safety"] == "not_measured_in_legacy_auto_fx_apply_path"
    assert source_evidence["feedback_risk"] == "not_measured_in_legacy_auto_fx_apply_path"
    assert source_evidence["intelligibility_risk"] == "not_measured_in_legacy_auto_fx_apply_path"
    assert source_evidence["old_value_source"] == (
        "cached_or_physical_fx_read_required_before_promotion"
    )
    assert source_evidence["rollback_value_source"] == (
        "cached_or_physical_fx_read_required_before_promotion"
    )
    assert source_evidence["current_fx_send_return_cache_freshness"] == "not_provided"
    assert source_evidence["current_fx_send_return_cache_no_live_authority"] is True
    assert source_evidence["fx_parameter_schema_no_live_authority"] is True
    assert source_evidence["current_fx_parameter_schema_no_live_authority"] is True
    assert source_evidence["current_fx_parameter_schema_hints"] == {
        "status": "required_before_live_promotion",
        "source": "cached_or_physical_fx_read_required_before_promotion",
        "planned_fx_slots": {
            "FX1": {"model": "PLATE", "fx_type": "reverb"},
            "FX2": {"model": "ROOM", "fx_type": "reverb"},
            "FX3": {"model": "ST-DL", "fx_type": "delay"},
            "FX4": {"model": "CHORUS", "fx_type": "chorus"},
        },
        "requires_physical_schema_readback": True,
        "live_write_authority": False,
    }
    assert source_evidence["fx_slot_activation_contract"] == {
        "status": "unsupported_wing_fx_slot_on_path_excluded",
        "source": "wing_fw_3_1_fx_surface",
        "slot_activity_source": "loaded_model",
        "unsupported_action_excluded": "set_fx_on",
        "unsupported_address": "/fx/N/on",
        "live_write_authority": False,
    }
    assert source_evidence["source_identity_downstream_trust"] == {
        "status": "legacy_multi_source_unverified",
        "trusted_for_downstream": False,
        "source_roles": ["lead_vocal", "snare"],
        "operator_review_required": True,
    }
    assert source_evidence["rollback_fx_readback"] == {
        "status": "required_before_live_promotion",
        "source": "cached_or_physical_fx_read_required_before_promotion",
        "send_level_readback_required": True,
        "return_level_readback_required": True,
        "fx_parameter_readback_required": True,
        "monitor_send_readback_required": True,
    }
    assert "fx_return_and_send_readback_not_proven" in correction["safety_blockers"]
    assert planner.last_apply_result is result


def test_apply_to_mixer_without_mixer_still_reports_no_mutation():
    planner = AutoFXPlanner(tempo_bpm=120.0)
    plan = planner.create_plan({1: "lead_vocal"})

    result = planner.apply_to_mixer(None, plan)

    assert result["applied"] is False
    assert result["calls"] == []
    assert result["blocked"] is True
    assert result["reason"] == "no_mixer_client"
    assert result["mixer_client_present"] is False
    _assert_top_level_denial(result)
    assert result["blocked_calls"]
    assert result["operator_correction"]["objective_family"] == "fx_sends"
    assert result["operator_correction"]["live_apply_allowed"] is False
    assert result["operator_correction"]["queue_import_allowed"] is False


def test_auto_fx_blocked_calls_exclude_unsupported_wing_fx_on_action():
    planner = AutoFXPlanner(tempo_bpm=120.0)
    plan = planner.create_plan({1: "lead_vocal", 2: "snare"})

    result = planner.apply_to_mixer(None, plan)

    blocked_commands = [call["cmd"] for call in result["blocked_calls"]]
    assert "set_fx_model" in blocked_commands
    assert "set_fx_mix" in blocked_commands
    assert "set_channel_send" in blocked_commands
    assert "set_fx_on" not in blocked_commands
