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

    def test_suggest_mode_queues_pending_actions_instead_of_replacing(self):
        """New suggestions are appended while existing pending actions stay stable."""
        agent, *_ = self._make_agent()
        first = AgentAction(
            action_type='adjust_gain',
            channel=1,
            parameters={'adjustment_db': -1.0},
            reason='first',
            source='rule_engine',
        )
        second = AgentAction(
            action_type='apply_hpf',
            channel=2,
            parameters={'frequency': 100},
            reason='second',
            source='rule_engine',
        )

        agent._queue_pending_actions([first])
        agent._queue_pending_actions([second])

        assert agent.state.pending_actions[0] is first
        assert agent.state.pending_actions[1] is second

    def test_suggest_mode_updates_duplicate_pending_action_in_place(self):
        """Equivalent suggestions refresh their data without changing list length."""
        agent, *_ = self._make_agent()
        first = AgentAction(
            action_type='adjust_gain',
            channel=1,
            parameters={'adjustment_db': -1.0},
            reason='old',
            source='rule_engine',
        )
        updated = AgentAction(
            action_type='adjust_gain',
            channel=1,
            parameters={'adjustment_db': -0.5},
            reason='new',
            source='rule_engine',
        )

        agent._queue_pending_actions([first])
        agent._queue_pending_actions([updated])

        assert len(agent.state.pending_actions) == 1
        assert agent.state.pending_actions[0].reason == 'new'
        assert agent.state.pending_actions[0].parameters['adjustment_db'] == -0.5

    def test_dismissed_pending_action_is_not_requeued_immediately(self):
        """Dismissed suggestions have a short cooldown before they can reappear."""
        agent, *_ = self._make_agent()
        action = AgentAction(
            action_type='adjust_gain',
            channel=1,
            parameters={'adjustment_db': -1.0},
            reason='test',
            source='rule_engine',
        )

        agent._queue_pending_actions([action])
        assert agent.dismiss_action(0) is True
        agent._queue_pending_actions([action])

        assert agent.state.pending_actions == []

    def test_stop_sets_running_false(self):
        """stop() sets is_running to False."""
        agent, *_ = self._make_agent()
        agent.state.is_running = True
        agent.stop()
        assert agent.state.is_running is False

    def test_decide_passes_auto_apply_protocol_context_to_llm(self):
        """LLM decisions receive auto-apply safety context from the knowledge base."""
        agent, mock_kb, _mock_rules, mock_llm, _mock_mixer = self._make_agent()

        class Entry:
            content = "Auto apply protocol: use small reversible moves."
            relevance_score = 1.0
            metadata = {"title": "Auto Apply Protocol"}

        mock_kb.search.return_value = [Entry()]
        mock_llm.get_mix_recommendation.return_value = {
            "gain_db": -8.0,
            "reason": "vocal peak is high",
            "expected_effect": "lower vocal peak level",
            "rollback_hint": "restore previous fader",
            "risk": "low",
        }

        actions = agent._decide({
            "channels": {
                1: {
                    "instrument": "lead_vocal",
                    "needs_attention": True,
                    "peak_db": -3.0,
                    "rms_db": -16.0,
                }
            }
        })

        search_queries = [call.args[0] for call in mock_kb.search.call_args_list]
        assert any("auto apply safety" in query for query in search_queries)
        assert any(
            tuple(call.kwargs.get("category", ())) == agent.LLM_CONTEXT_CATEGORIES
            for call in mock_kb.search.call_args_list
        )
        mock_llm.get_mix_recommendation.assert_called_once()
        context = mock_llm.get_mix_recommendation.call_args.args[1]
        assert "Auto apply protocol" in context[0]
        assert actions[0].risk == "low"
        assert actions[0].expected_effect == "lower vocal peak level"

    def test_decide_ignores_idle_noise_floor(self):
        """Idle virtual-device noise must not trigger auto gain or LLM fallback."""
        agent, mock_kb, _mock_rules, mock_llm, _mock_mixer = self._make_agent()

        actions = agent._decide({
            "channels": {
                1: {
                    "instrument": "ch_1",
                    "needs_attention": True,
                    "peak_db": -70.0,
                    "rms_db": -95.0,
                    "channel_armed": False,
                }
            }
        })

        assert actions == []
        mock_llm.get_mix_recommendation.assert_not_called()

    def test_decide_skips_llm_fallback_defaults(self):
        """Unavailable LLM fallback defaults must not be auto-applied as AI decisions."""
        agent, mock_kb, _mock_rules, mock_llm, _mock_mixer = self._make_agent()
        mock_llm.get_mix_recommendation.return_value = {
            "gain_db": -12.0,
            "reason": "Fallback defaults (LLM unavailable)",
            "llm_available": False,
        }

        actions = agent._decide({
            "channels": {
                1: {
                    "instrument": "lead_vocal",
                    "needs_attention": True,
                    "peak_db": -3.0,
                    "rms_db": -16.0,
                    "channel_armed": True,
                }
            }
        })

        assert all(action.source != "llm" for action in actions)

    @pytest.mark.asyncio
    async def test_llm_recommendation_applies_structured_params(self):
        """LLM recommendations are actionable after approval/apply."""
        agent, *_ = self._make_agent()
        class FakeMixer:
            def __init__(self):
                self.calls = []

            def set_fader(self, *args):
                self.calls.append(("set_fader", args))

            def set_pan(self, *args):
                self.calls.append(("set_pan", args))

            def set_eq_band(self, *args, **kwargs):
                self.calls.append(("set_eq_band", args, kwargs))

            def set_compressor(self, *args, **kwargs):
                self.calls.append(("set_compressor", args, kwargs))

        mixer = FakeMixer()
        agent.mixer = mixer

        action = AgentAction(
            action_type='llm_recommendation',
            channel=1,
            parameters={
                'gain_db': -7.0,
                'pan': 0.2,
                'eq_bands': [{'freq': 3000, 'gain_db': 2.0, 'q': 1.2}],
                'comp_threshold': -18.0,
                'comp_ratio': 3.0,
                'comp_attack_ms': 10.0,
                'comp_release_ms': 120.0,
            },
            reason='LLM recommendation',
            source='llm',
        )

        await agent._act([action])

        assert ("set_fader", (1, -7.0)) in mixer.calls
        assert ("set_pan", (1, 0.2)) in mixer.calls
        assert ("set_eq_band", (1, 1), {"freq": 3000, "gain": 2.0, "q": 1.2}) in mixer.calls
        assert (
            "set_compressor",
            (1,),
            {
                "threshold_db": -18.0,
                "ratio": 3.0,
                "attack_ms": 10.0,
                "release_ms": 120.0,
                "enabled": True,
            },
        ) in mixer.calls
        assert agent.state.applied_actions[-1] is action

    @pytest.mark.asyncio
    async def test_auto_actions_are_limited_before_apply(self):
        """Large automatic fader and EQ moves are constrained before mixer writes."""
        agent, *_ = self._make_agent(mode=AgentMode.AUTO)

        class FakeMixer:
            def __init__(self):
                self.calls = []

            def get_fader(self, channel):
                return 0.0

            def set_fader(self, *args):
                self.calls.append(("set_fader", args))

            def set_eq_band(self, *args, **kwargs):
                self.calls.append(("set_eq_band", args, kwargs))

        mixer = FakeMixer()
        agent.mixer = mixer
        actions = agent._prepare_actions([
            AgentAction(
                action_type='adjust_gain',
                channel=1,
                parameters={'adjustment_db': -12.0},
                confidence=1.0,
            ),
            AgentAction(
                action_type='set_eq',
                channel=1,
                parameters={'band': 1, 'freq': 3000, 'gain': 9.0, 'q': 1.2},
                confidence=1.0,
            ),
        ])

        await agent._act(actions)

        assert ("set_fader", (1, -1.0)) in mixer.calls
        assert ("set_eq_band", (1, 1), {"freq": 3000.0, "gain": 2.0, "q": 1.2}) in mixer.calls

    @pytest.mark.asyncio
    async def test_positive_fader_moves_require_configured_ceiling(self):
        """Automatic boosts can exceed unity only when safety limits allow it."""
        agent, *_ = self._make_agent(mode=AgentMode.AUTO)
        agent.configure_safety_limits(max_fader_step_db=2.0, max_fader_db=6.0)

        class FakeMixer:
            def __init__(self):
                self.calls = []

            def get_fader(self, channel):
                return 0.0

            def set_fader(self, *args):
                self.calls.append(("set_fader", args))

        mixer = FakeMixer()
        agent.mixer = mixer
        actions = agent._prepare_actions([
            AgentAction(
                action_type='adjust_gain',
                channel=1,
                parameters={'adjustment_db': 4.0},
                confidence=1.0,
            )
        ])

        await agent._act(actions)

        assert ("set_fader", (1, 2.0)) in mixer.calls

    @pytest.mark.asyncio
    async def test_emergency_stop_blocks_mixer_writes(self):
        """Emergency stop prevents pending writes and records an audit event."""
        agent, *_ = self._make_agent(mode=AgentMode.AUTO)
        mixer = MagicMock()
        agent.mixer = mixer
        agent.emergency_stop()

        await agent._act([
            AgentAction(action_type='mute_channel', channel=1, parameters={}, confidence=1.0)
        ])

        mixer.set_mute.assert_not_called()
        assert agent.get_status()['emergency_stop'] is True
        assert agent.get_action_audit_log(5)[0]['status'] == 'blocked'
