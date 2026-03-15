"""
Tests for backend.ai.agent — AIAgent safety clamps, rule-based routing,
tool execution, soundcheck, and feedback handling.

All tests run WITHOUT mixer hardware or LLM servers. Wing client is None,
and LLM calls are mocked where needed.
"""

import asyncio
import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.ai.agent import (
    AIAgent,
    MAX_GAIN_CHANGE_PER_STEP_DB,
    MAX_FADER_DB,
    MIN_FADER_DB,
    MAX_EQ_BOOST_DB,
    MAX_EQ_CUT_DB,
    TOOL_DEFINITIONS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent():
    """AIAgent with no wing client and default knowledge dir."""
    return AIAgent(wing_client=None)


# ---------------------------------------------------------------------------
# Safety clamp tests
# ---------------------------------------------------------------------------

class TestSafetyClamps:

    def test_clamp_fader_within_range(self):
        assert AIAgent._clamp_fader(0.0) == 0.0
        assert AIAgent._clamp_fader(-10.0) == -10.0

    def test_clamp_fader_max(self):
        assert AIAgent._clamp_fader(100.0) == MAX_FADER_DB

    def test_clamp_fader_min(self):
        assert AIAgent._clamp_fader(-200.0) == MIN_FADER_DB

    def test_clamp_eq_gain_within_range(self):
        assert AIAgent._clamp_eq_gain(0.0) == 0.0
        assert AIAgent._clamp_eq_gain(6.0) == 6.0
        assert AIAgent._clamp_eq_gain(-10.0) == -10.0

    def test_clamp_eq_gain_max(self):
        assert AIAgent._clamp_eq_gain(20.0) == MAX_EQ_BOOST_DB

    def test_clamp_eq_gain_min(self):
        assert AIAgent._clamp_eq_gain(-30.0) == MAX_EQ_CUT_DB

    def test_enforce_gain_limit_small_change(self, agent):
        """Small gain change should pass through unchanged."""
        agent._channel_state[1] = {"fader": 0.0}
        result = agent._enforce_gain_limit(1, 3.0)
        assert result == 3.0

    def test_enforce_gain_limit_large_change(self, agent):
        """Large gain change should be clamped to MAX_GAIN_CHANGE_PER_STEP_DB."""
        agent._channel_state[1] = {"fader": 0.0}
        result = agent._enforce_gain_limit(1, 20.0)
        assert result == MAX_GAIN_CHANGE_PER_STEP_DB

    def test_enforce_gain_limit_large_negative(self, agent):
        """Large negative gain change should be clamped."""
        agent._channel_state[1] = {"fader": 0.0}
        result = agent._enforce_gain_limit(1, -20.0)
        assert result == -MAX_GAIN_CHANGE_PER_STEP_DB

    def test_enforce_gain_limit_no_prior_state(self, agent):
        """When no prior state, current defaults to 0.0."""
        result = agent._enforce_gain_limit(99, 3.0)
        assert result == 3.0


# ---------------------------------------------------------------------------
# Tool execution tests
# ---------------------------------------------------------------------------

class TestToolExecution:

    def test_set_channel_fader(self, agent):
        async def _test():
            result = await agent.execute_tool("set_channel_fader", {"ch": 1, "value": -5.0})
            assert result["success"] is True
            assert result["fader"] == -5.0
        _run(_test())

    def test_set_channel_pan(self, agent):
        async def _test():
            result = await agent.execute_tool("set_channel_pan", {"ch": 1, "value": -30.0})
            assert result["success"] is True
            assert result["pan"] == -30.0
        _run(_test())

    def test_set_channel_mute(self, agent):
        async def _test():
            result = await agent.execute_tool("set_channel_mute", {"ch": 1, "muted": True})
            assert result["success"] is True
            assert result["muted"] is True
        _run(_test())

    def test_set_channel_hpf(self, agent):
        async def _test():
            result = await agent.execute_tool("set_channel_hpf", {"ch": 1, "frequency": 80.0})
            assert result["success"] is True
            assert result["hpf_frequency"] == 80.0
        _run(_test())

    def test_set_channel_eq(self, agent):
        async def _test():
            result = await agent.execute_tool(
                "set_channel_eq",
                {"ch": 1, "band": 1, "freq": 1000.0, "gain": 3.0, "q": 1.5},
            )
            assert result["success"] is True
            assert result["freq"] == 1000.0
        _run(_test())

    def test_set_channel_compressor(self, agent):
        async def _test():
            result = await agent.execute_tool(
                "set_channel_compressor",
                {"ch": 1, "threshold": -15.0, "ratio": 4.0, "attack": 10.0, "release": 100.0},
            )
            assert result["success"] is True
            assert result["ratio"] == 4.0
        _run(_test())

    def test_set_channel_gate(self, agent):
        async def _test():
            result = await agent.execute_tool(
                "set_channel_gate",
                {"ch": 1, "threshold": -30.0, "attack": 0.5, "hold": 50.0, "release": 200.0},
            )
            assert result["success"] is True
        _run(_test())

    def test_unknown_tool(self, agent):
        async def _test():
            result = await agent.execute_tool("nonexistent_tool", {})
            assert "error" in result
        _run(_test())

    def test_invalid_parameters(self, agent):
        async def _test():
            result = await agent.execute_tool("set_channel_fader", {"wrong_param": 5})
            assert "error" in result
        _run(_test())

    def test_get_channel_levels(self, agent):
        async def _test():
            # Set some state first
            await agent.set_channel_fader(1, -3.0)
            result = await agent.get_channel_levels(1)
            assert result["ch"] == 1
            assert result["fader"] == -3.0
        _run(_test())

    def test_get_mix_quality(self, agent):
        async def _test():
            result = await agent.get_mix_quality()
            assert "active_channels" in result
            assert "issues" in result
            assert "score" in result
        _run(_test())


# ---------------------------------------------------------------------------
# Parameter clamping within tools
# ---------------------------------------------------------------------------

class TestParameterClamping:

    def test_fader_clamped(self, agent):
        async def _test():
            result = await agent.set_channel_fader(1, 50.0)
            assert result["fader"] <= MAX_FADER_DB
        _run(_test())

    def test_eq_gain_clamped(self, agent):
        async def _test():
            result = await agent.set_channel_eq(1, 1, 1000.0, 25.0, 1.0)
            assert result["gain"] <= MAX_EQ_BOOST_DB
        _run(_test())

    def test_eq_freq_clamped(self, agent):
        async def _test():
            result = await agent.set_channel_eq(1, 1, 50000.0, 0.0, 1.0)
            assert result["freq"] <= 20000.0
        _run(_test())

    def test_pan_clamped(self, agent):
        async def _test():
            result = await agent.set_channel_pan(1, 200.0)
            assert result["pan"] <= 100.0
        _run(_test())

    def test_hpf_clamped(self, agent):
        async def _test():
            result = await agent.set_channel_hpf(1, 5000.0)
            assert result["hpf_frequency"] <= 2000.0
        _run(_test())


# ---------------------------------------------------------------------------
# Rule-based routing (Tier 1)
# ---------------------------------------------------------------------------

class TestRuleBasedRouting:

    def test_apply_preset_command(self, agent):
        result = agent._try_rule_based("apply kick to channel 1")
        assert result is not None
        assert len(result["actions"]) > 0
        assert result["actions"][0]["tool"] == "apply_preset"

    def test_mute_command(self, agent):
        result = agent._try_rule_based("mute channel 3")
        assert result is not None
        assert result["actions"][0]["tool"] == "set_channel_mute"
        assert result["actions"][0]["parameters"]["muted"] is True

    def test_unmute_command(self, agent):
        result = agent._try_rule_based("unmute channel 5")
        assert result is not None
        assert result["actions"][0]["parameters"]["muted"] is False

    def test_set_fader_command(self, agent):
        result = agent._try_rule_based("set channel 2 fader to -5")
        assert result is not None
        assert result["actions"][0]["tool"] == "set_channel_fader"
        assert result["actions"][0]["parameters"]["value"] == -5.0

    def test_set_pan_command(self, agent):
        result = agent._try_rule_based("set channel 4 pan to -30")
        assert result is not None
        assert result["actions"][0]["tool"] == "set_channel_pan"

    def test_set_hpf_command(self, agent):
        result = agent._try_rule_based("set channel 1 hpf to 80")
        assert result is not None
        assert result["actions"][0]["tool"] == "set_channel_hpf"

    def test_feedback_command(self, agent):
        result = agent._try_rule_based("feedback at 2500 hz")
        assert result is not None
        assert "feedback" in result["response"].lower()

    def test_soundcheck_command(self, agent):
        result = agent._try_rule_based("run soundcheck")
        assert result is not None
        assert "soundcheck" in result["response"].lower()

    def test_unmatched_command_returns_none(self, agent):
        result = agent._try_rule_based("tell me about the history of audio engineering")
        assert result is None


# ---------------------------------------------------------------------------
# Apply preset tool
# ---------------------------------------------------------------------------

class TestApplyPreset:

    def test_apply_kick_preset(self, agent):
        async def _test():
            result = await agent.apply_preset(1, "kick")
            assert result["success"] is True
            assert result["preset"] == "kick"
            assert "hpf" in result["applied"]
            assert "eq" in result["applied"]
            assert "compressor" in result["applied"]
        _run(_test())

    def test_apply_vocal_preset(self, agent):
        async def _test():
            result = await agent.apply_preset(5, "leadVocal")
            assert result["success"] is True
            # Gate should be enabled for vocals
            assert "gate" in result["applied"]
            gate = result["applied"]["gate"]
            assert gate.get("success", gate.get("enabled", None)) is not False
        _run(_test())

    def test_preset_tracks_instrument(self, agent):
        async def _test():
            await agent.apply_preset(3, "snare")
            assert agent._channel_instruments[3] == "snare"
        _run(_test())


# ---------------------------------------------------------------------------
# Soundcheck
# ---------------------------------------------------------------------------

class TestSoundcheck:

    def test_soundcheck_with_names(self, agent):
        async def _test():
            result = await agent.run_soundcheck(
                channels=[1, 2, 3],
                channel_names={1: "Kick", 2: "Snare", 3: "Lead Vox"},
            )
            assert result["classified"] > 0
            assert "summary" in result
        _run(_test())

    def test_soundcheck_skips_unnamed(self, agent):
        async def _test():
            result = await agent.run_soundcheck(
                channels=[1, 2],
                channel_names={1: "Kick"},  # Channel 2 has no name
            )
            assert result["channels"][2]["status"] == "skipped"
        _run(_test())

    def test_soundcheck_custom_classifier(self, agent):
        async def _test():
            def my_classifier(name):
                return "kick" if "bd" in name.lower() else None
            result = await agent.run_soundcheck(
                channels=[1],
                channel_names={1: "BD"},
                classifier_fn=my_classifier,
            )
            assert result["channels"][1]["instrument"] == "kick"
        _run(_test())

    def test_soundcheck_unclassified(self, agent):
        async def _test():
            def always_none(name):
                return None
            result = await agent.run_soundcheck(
                channels=[1],
                channel_names={1: "Unknown Thing"},
                classifier_fn=always_none,
            )
            assert result["channels"][1]["status"] == "unclassified"
            assert result["unclassified"] >= 1
        _run(_test())


# ---------------------------------------------------------------------------
# Default classifier
# ---------------------------------------------------------------------------

class TestDefaultClassifier:

    def test_recognizes_kick(self):
        result = AIAgent._default_classifier("Kick Drum")
        assert result == "kick"

    def test_recognizes_vocal(self):
        result = AIAgent._default_classifier("Lead Vox")
        assert result in ("leadVocal", "vocal")

    def test_recognizes_bass(self):
        result = AIAgent._default_classifier("Bass DI")
        assert result == "bass"

    def test_unknown_returns_none(self):
        result = AIAgent._default_classifier("XYZZY")
        assert result is None


# ---------------------------------------------------------------------------
# Feedback emergency
# ---------------------------------------------------------------------------

class TestFeedbackEmergency:

    def test_feedback_emergency_applies_notch(self, agent):
        async def _test():
            result = await agent.handle_feedback_emergency(1, 2500.0)
            assert result["action"] == "feedback_suppression"
            assert result["ch"] == 1
            assert result["frequency"] == 2500.0
            assert result["notch"]["gain"] < 0
            assert result["eq_result"]["success"] is True
        _run(_test())

    def test_feedback_emergency_uses_band_4(self, agent):
        async def _test():
            result = await agent.handle_feedback_emergency(2, 800.0)
            assert result["eq_result"]["band"] == 4
        _run(_test())


# ---------------------------------------------------------------------------
# Search knowledge tool
# ---------------------------------------------------------------------------

class TestSearchKnowledge:

    def test_search_returns_dict(self, agent):
        async def _test():
            result = await agent.search_knowledge("kick drum EQ")
            assert "query" in result
            assert "results" in result
            assert "count" in result
        _run(_test())


# ---------------------------------------------------------------------------
# Mix quality
# ---------------------------------------------------------------------------

class TestMixQuality:

    def test_hot_fader_flagged(self, agent):
        async def _test():
            await agent.set_channel_fader(1, 8.0)
            agent._channel_instruments[1] = "kick"
            metrics = await agent.get_mix_quality()
            assert any("hot" in issue.lower() for issue in metrics["issues"])
        _run(_test())

    def test_clean_mix_has_high_score(self, agent):
        async def _test():
            await agent.set_channel_fader(1, -5.0)
            await agent.set_channel_hpf(1, 80.0)
            agent._channel_instruments[1] = "leadVocal"
            metrics = await agent.get_mix_quality()
            assert metrics["score"] >= 80
        _run(_test())

    def test_empty_mix_no_issues(self, agent):
        async def _test():
            metrics = await agent.get_mix_quality()
            assert metrics["issues"] == []
            assert metrics["score"] == 100
        _run(_test())


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

class TestToolDefinitions:

    def test_eleven_tools_defined(self):
        assert len(TOOL_DEFINITIONS) == 11

    def test_all_tools_have_name_and_description(self):
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool

    def test_all_defined_tools_are_registered(self, agent):
        for tool in TOOL_DEFINITIONS:
            assert tool["name"] in agent._tools, (
                f"Tool '{tool['name']}' defined but not registered"
            )
