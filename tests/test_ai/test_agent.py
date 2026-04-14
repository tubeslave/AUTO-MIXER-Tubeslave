"""Tests for ai.agent module."""
import pytest
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from ai.agent import MixingAgent, AgentMode, AgentAction, AgentState


class TestAgentMode:
    """Tests for the AgentMode enum."""

    def test_enum_values(self):
        """AgentMode has the expected members and values."""
        assert AgentMode.AUTO.value == 'auto'
        assert AgentMode.SUGGEST.value == 'suggest'
        assert AgentMode.MANUAL.value == 'manual'
        assert len(AgentMode) == 3

    def test_enum_from_value(self):
        """AgentMode members can be created from string values."""
        assert AgentMode('auto') == AgentMode.AUTO
        assert AgentMode('suggest') == AgentMode.SUGGEST
        assert AgentMode('manual') == AgentMode.MANUAL


class TestMixingAgent:
    """Tests for the MixingAgent class."""

    def _make_agent(self, mode=AgentMode.SUGGEST):
        """Helper to create an agent with mocked dependencies."""
        mock_kb = MagicMock()
        mock_kb.search.return_value = []
        mock_rules = MagicMock()
        mock_rules.evaluate.return_value = []
        mock_llm = MagicMock()
        mock_mixer = MagicMock()

        agent = MixingAgent(
            knowledge_base=mock_kb,
            rule_engine=mock_rules,
            llm_client=mock_llm,
            mixer_client=mock_mixer,
            mode=mode,
            cycle_interval=0.1,
        )
        return agent, mock_kb, mock_rules, mock_llm, mock_mixer

    def test_init_with_mocked_deps(self):
        """MixingAgent initializes correctly with provided dependencies."""
        agent, mock_kb, mock_rules, mock_llm, mock_mixer = self._make_agent()
        assert agent.state.mode == AgentMode.SUGGEST
        assert agent.state.is_running is False
        assert agent.state.cycle_count == 0
        assert agent.kb is mock_kb
        assert agent.rules is mock_rules
        assert agent.llm is mock_llm
        assert agent.mixer is mock_mixer

    def test_set_mode(self):
        """set_mode changes the agent operating mode."""
        agent, *_ = self._make_agent()
        agent.set_mode(AgentMode.AUTO)
        assert agent.state.mode == AgentMode.AUTO
        agent.set_mode(AgentMode.MANUAL)
        assert agent.state.mode == AgentMode.MANUAL

    def test_set_confidence_threshold_clamps(self):
        """set_confidence_threshold clamps values to [0.0, 1.0]."""
        agent, *_ = self._make_agent()
        agent.set_confidence_threshold(0.8)
        assert agent._confidence_threshold == 0.8

        agent.set_confidence_threshold(-0.5)
        assert agent._confidence_threshold == 0.0

        agent.set_confidence_threshold(1.5)
        assert agent._confidence_threshold == 1.0

    def test_update_channel_state_and_summary(self):
        """Channel states can be updated and retrieved via get_channel_summary."""
        agent, *_ = self._make_agent()
        agent.update_channel_state(1, {
            'instrument': 'lead_vocal',
            'rms_db': -18.0,
            'peak_db': -6.0,
            'is_muted': False,
        })
        agent.update_channel_state(2, {
            'instrument': 'kick',
            'rms_db': -12.0,
            'peak_db': -4.0,
            'is_muted': False,
        })

        summary = agent.get_channel_summary()
        assert 1 in summary
        assert summary[1]['instrument'] == 'lead_vocal'
        assert summary[2]['rms_db'] == -12.0
        assert len(summary) == 2

    def test_batch_update_channel_states(self):
        """update_channel_states_batch updates multiple channels at once."""
        agent, *_ = self._make_agent()
        batch = {
            10: {'instrument': 'snare', 'rms_db': -15.0, 'peak_db': -8.0, 'is_muted': False},
            11: {'instrument': 'hi_hat', 'rms_db': -22.0, 'peak_db': -14.0, 'is_muted': False},
        }
        agent.update_channel_states_batch(batch)
        assert len(agent.state.channel_states) == 2
        assert agent.state.channel_states[10]['instrument'] == 'snare'

    def test_get_status_structure(self):
        """get_status returns a dict with all expected fields."""
        agent, *_ = self._make_agent()
        status = agent.get_status()
        expected_keys = {
            'mode', 'is_running', 'cycle_count', 'last_cycle_time_ms',
            'pending_actions', 'applied_actions', 'total_actions_history',
            'channels_tracked', 'confidence_threshold', 'cycle_interval_ms',
            'recent_errors', 'kb_first',
        }
        assert expected_keys.issubset(set(status.keys()))
        assert status['mode'] == 'suggest'
        assert status['is_running'] is False
        assert status['kb_first'] is False

    def test_kb_first_skips_rule_engine_init(self):
        mock_kb = MagicMock()
        mock_kb.search.return_value = []
        agent = MixingAgent(
            knowledge_base=mock_kb,
            rule_engine=None,
            llm_client=None,
            mixer_client=None,
            mode=AgentMode.SUGGEST,
            kb_first=True,
        )
        assert agent.rules is None
        assert agent.kb_first is True

    def test_dismiss_pending_actions(self):
        """Pending actions can be dismissed individually and in bulk."""
        agent, *_ = self._make_agent()
        # Manually add pending actions
        agent.state.pending_actions = [
            AgentAction(action_type='reduce_gain', channel=1, parameters={'amount_db': -3},
                        reason='test1', source='rule_engine'),
            AgentAction(action_type='mute_channel', channel=2, parameters={},
                        reason='test2', source='rule_engine'),
        ]
        assert len(agent.state.pending_actions) == 2

        # Dismiss first action
        assert agent.dismiss_action(0) is True
        assert len(agent.state.pending_actions) == 1

        # Dismiss invalid index
        assert agent.dismiss_action(99) is False

        # Dismiss all
        agent.state.pending_actions.append(
            AgentAction(action_type='adjust_gain', channel=3, parameters={},
                        reason='test3', source='rule_engine'),
        )
        agent.dismiss_all_pending()
        assert len(agent.state.pending_actions) == 0

    def test_stop_sets_running_false(self):
        """stop() sets is_running to False."""
        agent, *_ = self._make_agent()
        agent.state.is_running = True
        agent.stop()
        assert agent.state.is_running is False
