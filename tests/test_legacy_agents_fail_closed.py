"""Fail-closed checks for legacy ODA agents."""

import asyncio
from pathlib import Path

from backend.agents.eq_agent import EQAgent


class FakeEQMixer:
    def __init__(self):
        self.state = {
            "ch/3/eq/2/g": 14.0,
        }
        self.calls = []

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


def test_eq_agent_source_has_no_direct_mixer_send():
    source = Path("backend/agents/eq_agent.py").read_text()

    assert "self._mixer.send(" not in source
    assert '"cmd": "set_eq_band_gain"' in source
