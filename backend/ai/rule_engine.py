"""
Rule engine for automated mixing decisions.
Implements production rules based on live sound engineering best practices.
"""
import logging
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RulePriority(Enum):
    CRITICAL = 1    # Safety (feedback, clipping)
    HIGH = 2        # Core mixing (gain staging, vocal presence)
    MEDIUM = 3      # Enhancement (EQ, dynamics)
    LOW = 4         # Polish (reverb, stereo image)
    SUGGESTION = 5  # Advisory only


@dataclass
class RuleResult:
    """Result of evaluating a rule."""
    rule_name: str
    triggered: bool
    action: str  # Description of recommended action
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: RulePriority = RulePriority.MEDIUM
    confidence: float = 1.0
    reason: str = ''


@dataclass
class Rule:
    """A single production rule."""
    name: str
    description: str
    priority: RulePriority
    condition: Callable[[Dict], bool]
    action: Callable[[Dict], RuleResult]
    enabled: bool = True
    cooldown_sec: float = 1.0
    _last_fired: float = 0.0


class RuleEngine:
    """Evaluates mixing rules against current state."""

    def __init__(self):
        self.rules: List[Rule] = []
        self._build_default_rules()
        logger.info(f"Rule engine initialized with {len(self.rules)} rules")

    def _build_default_rules(self):
        """Build default mixing rules based on live sound engineering best practices."""

        # Rule: Feedback detection — immediate gain reduction
        self.rules.append(Rule(
            name='feedback_protection',
            description='Reduce gain when feedback is detected on a channel',
            priority=RulePriority.CRITICAL,
            condition=lambda state: state.get('feedback_detected', False),
            action=lambda state: RuleResult(
                rule_name='feedback_protection',
                triggered=True,
                action='reduce_gain',
                parameters={
                    'channel': state.get('feedback_channel', 0),
                    'amount_db': -3.0,
                    'frequency': state.get('feedback_frequency', 0),
                },
                priority=RulePriority.CRITICAL,
                confidence=0.95,
                reason=f"Feedback detected at {state.get('feedback_frequency', 0):.0f}Hz on ch{state.get('feedback_channel', 0)}"
            )
        ))

        # Rule: Clipping protection — reduce gain when true peak exceeds -0.5 dBTP
        self.rules.append(Rule(
            name='clipping_protection',
            description='Reduce gain when signal clips or approaches 0dBFS',
            priority=RulePriority.CRITICAL,
            condition=lambda state: state.get('true_peak_db', -100) > -0.5,
            action=lambda state: RuleResult(
                rule_name='clipping_protection',
                triggered=True,
                action='reduce_gain',
                parameters={
                    'channel': state.get('channel_id', 0),
                    'amount_db': min(-1.0, -0.5 - state.get('true_peak_db', 0)),
                },
                priority=RulePriority.CRITICAL,
                confidence=1.0,
                reason=f"True peak at {state.get('true_peak_db', 0):.1f}dBTP exceeds -0.5dBTP limit"
            )
        ))

        # Rule: Vocal presence — ensure lead vocal sits above instrument bus
        self.rules.append(Rule(
            name='vocal_presence',
            description='Ensure lead vocal sits above instruments in the mix',
            priority=RulePriority.HIGH,
            condition=lambda state: (
                state.get('instrument') == 'lead_vocal' and
                state.get('lufs_momentary', -100) < state.get('mix_lufs', -100) - 2
            ),
            action=lambda state: RuleResult(
                rule_name='vocal_presence',
                triggered=True,
                action='adjust_gain',
                parameters={
                    'channel': state.get('channel_id', 0),
                    'target_relative_db': 0.0,
                },
                priority=RulePriority.HIGH,
                confidence=0.8,
                reason='Lead vocal below mix level by more than 2dB LUFS'
            )
        ))

        # Rule: Low-end buildup on non-bass instruments
        self.rules.append(Rule(
            name='low_end_buildup',
            description='Apply HPF when low-end energy is excessive on non-bass sources',
            priority=RulePriority.MEDIUM,
            condition=lambda state: (
                state.get('band_energy', {}).get('sub', -100) > -6 and
                state.get('instrument') not in ('kick', 'bass_guitar', 'synth_bass', 'floor_tom')
            ),
            action=lambda state: RuleResult(
                rule_name='low_end_buildup',
                triggered=True,
                action='apply_hpf',
                parameters={
                    'channel': state.get('channel_id', 0),
                    'frequency': _hpf_frequency_for_instrument(state.get('instrument', '')),
                    'slope': 18,
                },
                priority=RulePriority.MEDIUM,
                confidence=0.7,
                reason=f"Excessive sub energy ({state.get('band_energy', {}).get('sub', 0):.1f}dB) on {state.get('instrument', 'unknown')}"
            )
        ))

        # Rule: Dynamic range control — compress when range too wide
        self.rules.append(Rule(
            name='dynamic_range',
            description='Apply compression when dynamic range exceeds 24dB',
            priority=RulePriority.MEDIUM,
            condition=lambda state: state.get('dynamic_range_db', 0) > 24,
            action=lambda state: RuleResult(
                rule_name='dynamic_range',
                triggered=True,
                action='adjust_compressor',
                parameters={
                    'channel': state.get('channel_id', 0),
                    'threshold_db': state.get('lufs_momentary', -20) + 6,
                    'ratio': 3.0,
                    'attack_ms': 10.0,
                    'release_ms': 100.0,
                },
                priority=RulePriority.MEDIUM,
                confidence=0.7,
                reason=f"Dynamic range {state.get('dynamic_range_db', 0):.1f}dB exceeds 24dB target"
            )
        ))

        # Rule: Gain staging — maintain proper headroom
        self.rules.append(Rule(
            name='gain_staging',
            description='Ensure proper gain staging with -12dB peak target',
            priority=RulePriority.HIGH,
            condition=lambda state: (
                state.get('peak_db', -100) > -6 or
                state.get('peak_db', -100) < -30
            ),
            action=lambda state: RuleResult(
                rule_name='gain_staging',
                triggered=True,
                action='adjust_gain',
                parameters={
                    'channel': state.get('channel_id', 0),
                    'target_peak_db': -12.0,
                    'adjustment_db': -12.0 - state.get('peak_db', -12),
                },
                priority=RulePriority.HIGH,
                confidence=0.85,
                reason=f"Peak level at {state.get('peak_db', 0):.1f}dB (target: -12dB)"
            )
        ))

        # Rule: Mute unused channels to reduce noise floor
        self.rules.append(Rule(
            name='mute_unused',
            description='Mute channels with no signal to reduce noise floor',
            priority=RulePriority.LOW,
            condition=lambda state: (
                state.get('rms_db', -100) < -60 and
                not state.get('is_muted', False) and
                state.get('channel_armed', True)
            ),
            action=lambda state: RuleResult(
                rule_name='mute_unused',
                triggered=True,
                action='mute_channel',
                parameters={'channel': state.get('channel_id', 0)},
                priority=RulePriority.LOW,
                confidence=0.6,
                reason='No signal detected (RMS < -60dB), muting to reduce noise'
            ),
            cooldown_sec=5.0,
        ))

        # Rule: Unmute when signal returns
        self.rules.append(Rule(
            name='unmute_active',
            description='Unmute channels when signal returns above threshold',
            priority=RulePriority.HIGH,
            condition=lambda state: (
                state.get('rms_db', -100) > -40 and
                state.get('is_muted', False) and
                state.get('auto_muted', False)
            ),
            action=lambda state: RuleResult(
                rule_name='unmute_active',
                triggered=True,
                action='unmute_channel',
                parameters={'channel': state.get('channel_id', 0)},
                priority=RulePriority.HIGH,
                confidence=0.8,
                reason=f"Signal returned (RMS: {state.get('rms_db', 0):.1f}dB), unmuting channel"
            ),
            cooldown_sec=0.5,
        ))

        # Rule: De-esser engagement for vocals
        self.rules.append(Rule(
            name='deesser_engagement',
            description='Engage de-esser when sibilance energy high on vocal channels',
            priority=RulePriority.MEDIUM,
            condition=lambda state: (
                state.get('instrument') in ('lead_vocal', 'backing_vocal') and
                state.get('band_energy', {}).get('presence', -100) > -3 and
                state.get('sibilance_ratio', 0) > 0.4
            ),
            action=lambda state: RuleResult(
                rule_name='deesser_engagement',
                triggered=True,
                action='adjust_deesser',
                parameters={
                    'channel': state.get('channel_id', 0),
                    'frequency': 6500,
                    'threshold_db': -20,
                    'ratio': 4.0,
                },
                priority=RulePriority.MEDIUM,
                confidence=0.7,
                reason='Excessive sibilance detected on vocal channel'
            ),
            cooldown_sec=3.0,
        ))

        # Rule: Stereo balance check
        self.rules.append(Rule(
            name='stereo_balance',
            description='Warn when stereo image is significantly imbalanced',
            priority=RulePriority.LOW,
            condition=lambda state: abs(state.get('stereo_balance', 0.0)) > 0.3,
            action=lambda state: RuleResult(
                rule_name='stereo_balance',
                triggered=True,
                action='suggest_pan_adjustment',
                parameters={
                    'current_balance': state.get('stereo_balance', 0.0),
                    'suggested_correction': -state.get('stereo_balance', 0.0) * 0.5,
                },
                priority=RulePriority.LOW,
                confidence=0.5,
                reason=f"Stereo image imbalanced ({state.get('stereo_balance', 0.0):+.2f}), consider panning corrections"
            ),
            cooldown_sec=10.0,
        ))

    def evaluate(self, state: Dict) -> List[RuleResult]:
        """Evaluate all enabled rules against current mixer state.
        Returns triggered results sorted by priority (critical first)."""
        results = []
        now = time.time()

        for rule in self.rules:
            if not rule.enabled:
                continue
            if now - rule._last_fired < rule.cooldown_sec:
                continue
            try:
                if rule.condition(state):
                    result = rule.action(state)
                    results.append(result)
                    rule._last_fired = now
            except Exception as e:
                logger.error(f"Rule '{rule.name}' evaluation error: {e}")

        # Sort by priority value (1=CRITICAL first, then 2=HIGH, etc.)
        results.sort(key=lambda r: r.priority.value)
        return results

    def add_rule(self, rule: Rule):
        """Add a custom rule to the engine."""
        self.rules.append(rule)
        logger.info(f"Added rule: {rule.name} (priority: {rule.priority.name})")

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name."""
        for i, rule in enumerate(self.rules):
            if rule.name == name:
                self.rules.pop(i)
                logger.info(f"Removed rule: {name}")
                return True
        return False

    def enable_rule(self, name: str, enabled: bool = True) -> bool:
        """Enable or disable a rule by name."""
        for rule in self.rules:
            if rule.name == name:
                rule.enabled = enabled
                logger.info(f"Rule '{name}' {'enabled' if enabled else 'disabled'}")
                return True
        return False

    def get_rules(self) -> List[Dict]:
        """Get all rules as dicts for API/UI display."""
        return [
            {
                'name': r.name,
                'description': r.description,
                'priority': r.priority.name,
                'enabled': r.enabled,
                'cooldown_sec': r.cooldown_sec,
            }
            for r in self.rules
        ]

    def reset_cooldowns(self):
        """Reset all rule cooldown timers."""
        for rule in self.rules:
            rule._last_fired = 0.0


def _hpf_frequency_for_instrument(instrument: str) -> int:
    """Return recommended HPF cutoff frequency for a given instrument type."""
    hpf_map = {
        'lead_vocal': 80,
        'backing_vocal': 100,
        'acoustic_guitar': 80,
        'electric_guitar': 80,
        'keys_piano': 60,
        'organ': 40,
        'snare': 80,
        'hi_hat': 200,
        'overhead': 100,
        'rack_tom': 80,
        'floor_tom': 60,
        'percussion': 100,
        'strings': 80,
        'brass': 80,
        'woodwind': 100,
        'harmonica': 200,
        'banjo': 100,
        'mandolin': 120,
        'violin': 150,
        'viola': 100,
        'cello': 50,
    }
    return hpf_map.get(instrument, 100)
