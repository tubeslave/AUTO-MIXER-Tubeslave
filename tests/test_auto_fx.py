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


def test_apply_to_mixer_uses_fx_slots_and_post_fader_sends():
    planner = AutoFXPlanner(tempo_bpm=120.0)
    plan = planner.create_plan({1: "lead_vocal", 2: "snare"})
    mixer = FakeMixer()

    result = planner.apply_to_mixer(mixer, plan)

    assert result["applied"] is True
    assert ("set_fx_model", "FX1", "PLATE") in mixer.calls
    assert ("set_fx_model", "FX2", "ROOM") in mixer.calls
    assert any(call == ("set_channel_send", 1, 13, -18.0, 1, "POST") for call in mixer.calls)
    assert any(call == ("set_channel_send", 2, 14, -17.0, 1, "POST") for call in mixer.calls)
