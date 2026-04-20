"""Tests for ai.rule_engine module."""
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from ai.rule_engine import RuleEngine, Rule, RulePriority, RuleResult


class TestRuleEngine:
    """Tests for the RuleEngine class."""

    def test_default_rules_loaded(self):
        """RuleEngine loads the expected set of default rules on init."""
        engine = RuleEngine()
        rule_names = [r.name for r in engine.rules]
        assert 'feedback_protection' in rule_names
        assert 'clipping_protection' in rule_names
        assert 'vocal_presence' in rule_names
        assert 'gain_staging' in rule_names
        assert 'mute_unused' in rule_names
        assert len(engine.rules) >= 8

    def test_feedback_protection_triggers(self):
        """feedback_protection rule triggers when feedback_detected is True."""
        engine = RuleEngine()
        engine.reset_cooldowns()
        state = {
            'feedback_detected': True,
            'feedback_channel': 3,
            'feedback_frequency': 2500.0,
        }
        results = engine.evaluate(state)
        feedback_results = [r for r in results if r.rule_name == 'feedback_protection']
        assert len(feedback_results) == 1
        result = feedback_results[0]
        assert result.triggered is True
        assert result.priority == RulePriority.CRITICAL
        assert result.parameters['channel'] == 3
        assert result.parameters['amount_db'] == -3.0
        assert result.confidence == 0.95

    def test_clipping_protection_triggers_on_high_peak(self):
        """clipping_protection rule triggers when true_peak_db exceeds -0.5 dBTP."""
        engine = RuleEngine()
        engine.reset_cooldowns()
        state = {
            'true_peak_db': 0.0,
            'channel_id': 7,
        }
        results = engine.evaluate(state)
        clipping_results = [r for r in results if r.rule_name == 'clipping_protection']
        assert len(clipping_results) == 1
        assert clipping_results[0].triggered is True
        assert clipping_results[0].priority == RulePriority.CRITICAL

    def test_no_rules_trigger_on_safe_state(self):
        """No rules trigger on a state with safe, nominal values."""
        engine = RuleEngine()
        engine.reset_cooldowns()
        state = {
            'feedback_detected': False,
            'true_peak_db': -12.0,
            'peak_db': -12.0,
            'rms_db': -20.0,
            'instrument': 'lead_vocal',
            'lufs_momentary': -16.0,
            'mix_lufs': -18.0,
            'band_energy': {'sub': -30},
            'dynamic_range_db': 15,
            'is_muted': False,
            'channel_armed': True,
            'stereo_balance': 0.0,
            'sibilance_ratio': 0.1,
        }
        results = engine.evaluate(state)
        assert len(results) == 0

    def test_vocal_presence_emits_actionable_gain_adjustment(self):
        """vocal_presence emits an adjustment_db that the agent can apply."""
        engine = RuleEngine()
        engine.reset_cooldowns()
        state = {
            'feedback_detected': False,
            'true_peak_db': -12.0,
            'peak_db': -12.0,
            'rms_db': -23.0,
            'instrument': 'lead_vocal',
            'channel_id': 14,
            'lufs_momentary': -22.0,
            'mix_lufs': -18.0,
            'band_energy': {'sub': -30},
            'dynamic_range_db': 15,
            'is_muted': False,
            'channel_armed': True,
            'stereo_balance': 0.0,
            'sibilance_ratio': 0.1,
        }
        results = engine.evaluate(state)
        vocal_results = [r for r in results if r.rule_name == 'vocal_presence']
        assert len(vocal_results) == 1
        assert vocal_results[0].parameters['channel'] == 14
        assert vocal_results[0].parameters['target_lufs'] == pytest.approx(-16.0)
        assert vocal_results[0].parameters['adjustment_db'] == pytest.approx(3.0)

    def test_dynamic_range_uses_instrument_specific_kick_settings(self):
        """dynamic_range uses kick-oriented compressor settings instead of the old generic vocal profile."""
        engine = RuleEngine()
        engine.reset_cooldowns()
        state = {
            'instrument': 'kick',
            'channel_id': 5,
            'peak_db': -6.0,
            'rms_db': -18.0,
            'lufs_momentary': -18.0,
            'dynamic_range_db': 28.0,
            'channel_armed': True,
        }

        results = engine.evaluate(state)
        dynamic = [r for r in results if r.rule_name == 'dynamic_range']

        assert len(dynamic) == 1
        assert dynamic[0].parameters['ratio'] == pytest.approx(4.0)
        assert dynamic[0].parameters['attack_ms'] == pytest.approx(20.0)
        assert dynamic[0].parameters['release_ms'] == pytest.approx(60.0)
        assert dynamic[0].parameters['threshold_db'] == pytest.approx(-14.0)

    def test_dynamic_range_skips_playback_sources(self):
        """Playback sources should not get generic dynamic-range compression from the rule engine."""
        engine = RuleEngine()
        engine.reset_cooldowns()
        state = {
            'instrument': 'playback',
            'channel_id': 9,
            'peak_db': -4.0,
            'rms_db': -12.0,
            'lufs_momentary': -12.0,
            'dynamic_range_db': 30.0,
            'channel_armed': True,
        }

        results = engine.evaluate(state)

        assert all(r.rule_name != 'dynamic_range' for r in results)

    def test_mute_unused_skips_lead_vocal(self):
        """Lead vocals should not be hard-muted by the generic idle-noise rule."""
        engine = RuleEngine()
        engine.reset_cooldowns()
        state = {
            'instrument': 'lead_vocal',
            'channel_id': 1,
            'rms_db': -70.0,
            'peak_db': -50.0,
            'is_muted': False,
            'channel_armed': True,
        }

        results = engine.evaluate(state)

        assert all(r.rule_name != 'mute_unused' for r in results)

    def test_mute_unused_allows_close_tom_mics(self):
        """Close tom mics remain eligible for idle auto-mute."""
        engine = RuleEngine()
        engine.reset_cooldowns()
        state = {
            'instrument': 'rack_tom',
            'channel_id': 12,
            'rms_db': -70.0,
            'peak_db': -50.0,
            'is_muted': False,
            'channel_armed': True,
        }

        results = engine.evaluate(state)
        muted = [r for r in results if r.rule_name == 'mute_unused']

        assert len(muted) == 1
        assert muted[0].parameters['channel'] == 12

    def test_add_and_remove_custom_rule(self):
        """Custom rules can be added and removed by name."""
        engine = RuleEngine()
        initial_count = len(engine.rules)

        custom_rule = Rule(
            name='test_custom_rule',
            description='A test custom rule',
            priority=RulePriority.LOW,
            condition=lambda state: state.get('test_flag', False),
            action=lambda state: RuleResult(
                rule_name='test_custom_rule',
                triggered=True,
                action='test_action',
                parameters={},
                priority=RulePriority.LOW,
                confidence=0.9,
                reason='Test triggered',
            ),
        )
        engine.add_rule(custom_rule)
        assert len(engine.rules) == initial_count + 1

        removed = engine.remove_rule('test_custom_rule')
        assert removed is True
        assert len(engine.rules) == initial_count

        # Removing non-existent rule returns False
        assert engine.remove_rule('nonexistent') is False

    def test_enable_disable_rule(self):
        """Rules can be enabled and disabled by name, affecting evaluation."""
        engine = RuleEngine()
        engine.reset_cooldowns()

        # Disable feedback_protection
        assert engine.enable_rule('feedback_protection', False) is True

        state = {
            'feedback_detected': True,
            'feedback_channel': 1,
            'feedback_frequency': 3000.0,
        }
        results = engine.evaluate(state)
        feedback_results = [r for r in results if r.rule_name == 'feedback_protection']
        assert len(feedback_results) == 0

        # Re-enable it
        engine.enable_rule('feedback_protection', True)
        results = engine.evaluate(state)
        feedback_results = [r for r in results if r.rule_name == 'feedback_protection']
        assert len(feedback_results) == 1

    def test_results_sorted_by_priority(self):
        """Evaluation results are returned sorted by priority (CRITICAL first)."""
        engine = RuleEngine()
        engine.reset_cooldowns()
        state = {
            'feedback_detected': True,
            'feedback_channel': 1,
            'feedback_frequency': 1000.0,
            'true_peak_db': 0.5,
            'channel_id': 1,
            'peak_db': -2.0,
            'rms_db': -70.0,
            'is_muted': False,
            'channel_armed': True,
        }
        results = engine.evaluate(state)
        assert len(results) >= 2
        # Verify ordering: each result's priority value should be <= the next
        for i in range(len(results) - 1):
            assert results[i].priority.value <= results[i + 1].priority.value

    def test_get_rules_returns_dicts(self):
        """get_rules returns rule info as a list of dicts."""
        engine = RuleEngine()
        rules_list = engine.get_rules()
        assert isinstance(rules_list, list)
        assert len(rules_list) > 0
        first = rules_list[0]
        assert 'name' in first
        assert 'description' in first
        assert 'priority' in first
        assert 'enabled' in first
        assert 'cooldown_sec' in first
