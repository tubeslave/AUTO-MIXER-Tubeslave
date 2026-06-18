"""Fail-closed checks for legacy ODA agents."""

import asyncio
from pathlib import Path

from backend.agents.eq_agent import EQAgent
from backend.agents.gain_agent import GainAgent


class FakeEQMixer:
    def __init__(self):
        self.state = {
            "ch/3/eq/2/g": 14.0,
        }
        self.calls = []

    def send(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class FakeGainMixer:
    def __init__(self):
        self.calls = []
        self.trim = {4: 17.0}

    def get_channel_trim(self, channel):
        return self.trim.get(channel, 0.0)

    def send(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def test_eq_agent_records_blocked_evidence_without_live_write():
    mixer = FakeEQMixer()
    agent = EQAgent(mixer)

    asyncio.run(agent.act({"corrections": {3: {2: 4.0}}}))

    assert mixer.calls == []
    policy = agent.get_write_policy_state()
    assert policy["live_write_allowed"] is False
    assert policy["write_policy"] == "proposal_only_quarantined"
    assert policy["last_write_attempted"] is True
    assert policy["last_write_blocked"] is True
    assert policy["last_write_count"] == 0
    assert policy["last_blocked_calls"] == [
        {
            "cmd": "set_eq_band_gain",
            "path": "/ch/3/eq/2/g",
            "channel": 3,
            "band": 2,
            "current_gain_db": 14.0,
            "target_gain_db": 15.0,
            "delta_gain_db": 4.0,
            "bounded": True,
            "semantic": "legacy_eq_agent_band_gain",
        }
    ]
    states = agent.get_channel_states()
    assert states[3]["corrections"][2] == {
        "delta_gain_db": 4.0,
        "target_gain_db": 15.0,
        "blocked": True,
    }


def test_gain_agent_records_blocked_evidence_without_live_write():
    mixer = FakeGainMixer()
    agent = GainAgent(mixer, target_lufs=-18.0)

    asyncio.run(agent.act({"adjustments": {4: 3.0}}))

    assert mixer.calls == []
    policy = agent.get_write_policy_state()
    assert policy["live_write_allowed"] is False
    assert policy["write_policy"] == "proposal_only_quarantined"
    assert policy["last_write_attempted"] is True
    assert policy["last_write_blocked"] is True
    assert policy["last_write_count"] == 0
    assert policy["last_blocked_calls"] == [
        {
            "cmd": "set_channel_trim",
            "path": "/ch/4/trim",
            "channel": 4,
            "current_trim_db": 17.0,
            "target_trim_db": 18.0,
            "delta_trim_db": 3.0,
            "bounded": True,
            "target_lufs": -18.0,
            "semantic": "legacy_gain_agent_trim",
        }
    ]
    assert agent.get_channel_states()[4] == {
        "trim_db": 18.0,
        "adjustment": 3.0,
        "target_lufs": -18.0,
        "blocked": True,
    }


def test_eq_agent_source_has_no_direct_mixer_send():
    source = Path("backend/agents/eq_agent.py").read_text()

    assert "self._mixer.send(" not in source
    assert '"cmd": "set_eq_band_gain"' in source


def test_gain_agent_source_has_no_direct_mixer_send():
    source = Path("backend/agents/gain_agent.py").read_text()

    assert "self._mixer.send(" not in source
    assert '"cmd": "set_channel_trim"' in source
