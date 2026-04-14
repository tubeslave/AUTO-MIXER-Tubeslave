"""
Mixing Agent -- Observe-Decide-Act autonomous mixing controller.
Integrates knowledge base, rule engine, and LLM for intelligent mixing.
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AgentMode(Enum):
    AUTO = 'auto'        # Fully autonomous — applies all changes
    SUGGEST = 'suggest'  # Suggest but don't apply — waits for approval
    MANUAL = 'manual'    # Only respond to explicit queries


@dataclass
class AgentAction:
    """An action the agent wants to take on the mixer."""
    action_type: str  # 'set_gain', 'set_eq', 'set_comp', 'set_pan', 'mute', 'unmute', etc.
    channel: int
    parameters: Dict[str, Any]
    priority: int = 3  # 1=highest
    confidence: float = 0.5
    reason: str = ''
    source: str = 'rule_engine'  # 'rule_engine', 'llm', 'knowledge_base'
    timestamp: float = 0.0


@dataclass
class AgentState:
    """Current agent state tracking."""
    mode: AgentMode = AgentMode.SUGGEST
    is_running: bool = False
    cycle_count: int = 0
    last_cycle_time: float = 0.0
    pending_actions: List[AgentAction] = field(default_factory=list)
    applied_actions: List[AgentAction] = field(default_factory=list)
    channel_states: Dict[int, Dict] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class MixingAgent:
    """Autonomous mixing agent using ODA (Observe-Decide-Act) pattern.

    The agent continuously monitors mixer state, evaluates rules,
    optionally consults the LLM, and either applies or suggests
    mixing adjustments depending on mode.
    """

    def __init__(self,
                 knowledge_base=None,
                 rule_engine=None,
                 llm_client=None,
                 mixer_client=None,
                 mode: AgentMode = AgentMode.SUGGEST,
                 cycle_interval: float = 0.5,
                 kb_first: bool = False):
        self.kb_first = kb_first

        if knowledge_base is None:
            try:
                from .knowledge_base import KnowledgeBase
                knowledge_base = KnowledgeBase()
            except Exception as e:
                logger.warning(f"Could not init knowledge base: {e}")

        if rule_engine is None and not kb_first:
            try:
                from .rule_engine import RuleEngine
                rule_engine = RuleEngine()
            except Exception as e:
                logger.warning(f"Could not init rule engine: {e}")

        self.kb = knowledge_base
        self.rules = rule_engine
        self.llm = llm_client
        self.mixer = mixer_client
        self.state = AgentState(mode=mode)
        self.cycle_interval = cycle_interval
        self._confidence_threshold = 0.6
        self._max_actions_per_cycle = 5
        self._action_history: List[AgentAction] = []
        self._llm_query_interval = 10.0  # seconds between LLM consultations
        self._last_llm_query = 0.0
        logger.info(f"MixingAgent initialized in {mode.value} mode")

    async def start(self):
        """Start the agent ODA loop."""
        self.state.is_running = True
        self.state.errors.clear()
        logger.info("Agent started")
        while self.state.is_running:
            try:
                cycle_start = time.time()
                await self._run_cycle()
                self.state.cycle_count += 1
                self.state.last_cycle_time = time.time() - cycle_start
                elapsed = time.time() - cycle_start
                sleep_time = max(0.01, self.cycle_interval - elapsed)
                await asyncio.sleep(sleep_time)
            except asyncio.CancelledError:
                break
            except Exception as e:
                error_msg = f"Agent cycle error: {e}"
                logger.error(error_msg)
                self.state.errors.append(error_msg)
                if len(self.state.errors) > 50:
                    self.state.errors = self.state.errors[-25:]
                await asyncio.sleep(1.0)
        logger.info("Agent stopped")

    def stop(self):
        """Stop the agent loop."""
        self.state.is_running = False

    async def _run_cycle(self):
        """Single ODA cycle: Observe -> Decide -> Act."""
        # OBSERVE
        observations = await self._observe()

        # DECIDE
        actions = self._decide(observations)

        # ACT
        if self.state.mode == AgentMode.AUTO:
            await self._act(actions)
        elif self.state.mode == AgentMode.SUGGEST:
            self.state.pending_actions = actions

    async def _observe(self) -> Dict:
        """Observe current mixer state.

        Reads channel states from the shared state dict which is
        populated by the metering/OSC subsystem.
        """
        observations = {
            'timestamp': time.time(),
            'channels': {},
        }

        for ch_id, ch_state in self.state.channel_states.items():
            observations['channels'][ch_id] = dict(ch_state)

        return observations

    def _decide(self, observations: Dict) -> List[AgentAction]:
        """Make mixing decisions based on observations.

        Evaluates rule engine for each channel, enriches decisions
        with knowledge base context, and optionally queries LLM
        for complex situations.
        """
        actions: List[AgentAction] = []
        now = time.time()

        for ch_id, ch_state in observations.get('channels', {}).items():
            # Phase 1: Rule engine evaluation
            if self.rules:
                rule_results = self.rules.evaluate(ch_state)
                for result in rule_results:
                    if result.triggered and result.confidence >= self._confidence_threshold:
                        action = AgentAction(
                            action_type=result.action,
                            channel=result.parameters.get('channel', ch_id),
                            parameters=result.parameters,
                            priority=result.priority.value,
                            confidence=result.confidence,
                            reason=result.reason,
                            source='rule_engine',
                            timestamp=now,
                        )
                        actions.append(action)

            # Phase 2: Knowledge base context enrichment
            if self.kb and ch_state.get('instrument'):
                instrument = ch_state['instrument']
                kb_results = self.kb.search(
                    f"{instrument} mixing live sound",
                    n_results=2,
                    category='instrument_profiles'
                )
                for entry in kb_results:
                    if entry.relevance_score > 0.7:
                        logger.debug(
                            f"KB context for ch{ch_id} ({instrument}): "
                            f"{entry.metadata.get('title', '')}"
                        )

            # Phase 3: LLM consultation for complex scenarios (rate-limited)
            if (self.llm and
                    now - self._last_llm_query > self._llm_query_interval and
                    ch_state.get('needs_attention', False)):
                self._last_llm_query = now
                try:
                    context_entries = []
                    if self.kb:
                        kb_hits = self.kb.search(
                            f"{ch_state.get('instrument', '')} live mixing issue",
                            n_results=2
                        )
                        context_entries = [e.content for e in kb_hits]
                    rec = self.llm.get_mix_recommendation(ch_state, context_entries)
                    if rec and rec.get('gain_db') is not None:
                        action = AgentAction(
                            action_type='llm_recommendation',
                            channel=ch_id,
                            parameters=rec,
                            priority=3,
                            confidence=0.6,
                            reason=rec.get('reason', 'LLM recommendation'),
                            source='llm',
                            timestamp=now,
                        )
                        actions.append(action)
                except Exception as e:
                    logger.debug(f"LLM consultation error: {e}")

        # Sort by priority (lower number = higher priority), then confidence
        actions.sort(key=lambda a: (a.priority, -a.confidence))

        # Limit actions per cycle to prevent overwhelming the mixer
        return actions[:self._max_actions_per_cycle]

    async def _act(self, actions: List[AgentAction]):
        """Execute decided actions on the mixer.

        Uses method names available on both WingClient and DLiveClient
        (DLiveClient provides compatibility aliases).
        """
        for action in actions:
            try:
                applied = False
                if self.mixer:
                    if action.action_type == 'reduce_gain':
                        amount = action.parameters.get('amount_db', -3)
                        current = self._safe_get_fader(action.channel)
                        target = max(-100.0, current + amount)
                        self._safe_set_fader(action.channel, target)
                        applied = True
                    elif action.action_type == 'adjust_gain':
                        adj = action.parameters.get('adjustment_db', 0)
                        current = self._safe_get_fader(action.channel)
                        target = max(-100.0, min(10.0, current + adj))
                        self._safe_set_fader(action.channel, target)
                        applied = True
                    elif action.action_type == 'mute_channel':
                        self.mixer.set_mute(action.channel, True)
                        applied = True
                    elif action.action_type == 'unmute_channel':
                        self.mixer.set_mute(action.channel, False)
                        applied = True
                    elif action.action_type == 'apply_hpf':
                        freq = action.parameters.get('frequency', 80)
                        if hasattr(self.mixer, 'set_channel_hpf'):
                            self.mixer.set_channel_hpf(action.channel, freq)
                        elif hasattr(self.mixer, 'set_hpf'):
                            self.mixer.set_hpf(action.channel, freq)
                        applied = True
                    elif action.action_type == 'adjust_compressor':
                        if hasattr(self.mixer, 'set_channel_compressor'):
                            self.mixer.set_channel_compressor(
                                action.channel,
                                threshold=action.parameters.get('threshold_db', -18),
                                ratio=action.parameters.get('ratio', 3.0),
                                attack=action.parameters.get('attack_ms', 10),
                                release=action.parameters.get('release_ms', 100),
                            )
                        applied = True
                    elif action.action_type == 'adjust_deesser':
                        if hasattr(self.mixer, 'set_channel_deesser'):
                            self.mixer.set_channel_deesser(
                                action.channel,
                                frequency=action.parameters.get('frequency', 6500),
                                threshold=action.parameters.get('threshold_db', -20),
                                ratio=action.parameters.get('ratio', 4.0),
                            )
                        applied = True
                    elif action.action_type == 'set_eq':
                        band = action.parameters.get('band', 1)
                        self.mixer.set_eq_band(
                            action.channel, band,
                            freq=action.parameters.get('freq', 1000.0),
                            gain=action.parameters.get('gain', 0.0),
                            q=action.parameters.get('q', 1.0),
                        )
                        applied = True

                if applied:
                    self._action_history.append(action)
                    self.state.applied_actions.append(action)

                    if len(self._action_history) > 1000:
                        self._action_history = self._action_history[-500:]
                    if len(self.state.applied_actions) > 100:
                        self.state.applied_actions = self.state.applied_actions[-50:]

                    logger.debug(
                        f"Applied: {action.action_type} on ch{action.channel} "
                        f"(confidence={action.confidence:.2f}, reason={action.reason})"
                    )
            except Exception as e:
                error_msg = f"Action error: {action.action_type} ch{action.channel}: {e}"
                logger.error(error_msg)
                self.state.errors.append(error_msg)

    def _safe_set_fader(self, channel: int, value_db: float):
        """Set fader using whichever method the mixer client provides."""
        value_db = max(-100.0, min(0.0, value_db))
        if hasattr(self.mixer, 'set_channel_fader_db'):
            self.mixer.set_channel_fader_db(channel, value_db)
        elif hasattr(self.mixer, 'set_fader'):
            self.mixer.set_fader(channel, value_db)

    def _safe_get_fader(self, channel: int) -> float:
        """Get fader value using whichever method the mixer client provides."""
        if hasattr(self.mixer, 'get_fader'):
            return self.mixer.get_fader(channel)
        return -100.0

    def update_channel_state(self, channel_id: int, state: Dict):
        """Update observed state for a channel. Called by the metering subsystem."""
        self.state.channel_states[channel_id] = state

    def update_channel_states_batch(self, states: Dict[int, Dict]):
        """Batch update multiple channel states at once."""
        self.state.channel_states.update(states)

    def set_mode(self, mode: AgentMode):
        """Change agent mode."""
        old_mode = self.state.mode
        self.state.mode = mode
        logger.info(f"Agent mode changed from {old_mode.value} to {mode.value}")

    def set_confidence_threshold(self, threshold: float):
        """Set minimum confidence threshold for actions (0.0 to 1.0)."""
        self._confidence_threshold = max(0.0, min(1.0, threshold))

    def approve_action(self, action_idx: int) -> bool:
        """Approve a pending action (in SUGGEST mode). Returns True if approved."""
        if 0 <= action_idx < len(self.state.pending_actions):
            action = self.state.pending_actions.pop(action_idx)
            # Schedule the action for execution
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._act([action]))
                else:
                    loop.run_until_complete(self._act([action]))
            except RuntimeError:
                # No event loop running — execute synchronously via new loop
                asyncio.run(self._act([action]))
            return True
        return False

    def approve_all_pending(self) -> int:
        """Approve all pending actions. Returns count of approved actions."""
        count = len(self.state.pending_actions)
        if count > 0:
            actions = list(self.state.pending_actions)
            self.state.pending_actions.clear()
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._act(actions))
                else:
                    loop.run_until_complete(self._act(actions))
            except RuntimeError:
                asyncio.run(self._act(actions))
        return count

    def dismiss_action(self, action_idx: int) -> bool:
        """Dismiss a pending action without applying it."""
        if 0 <= action_idx < len(self.state.pending_actions):
            self.state.pending_actions.pop(action_idx)
            return True
        return False

    def dismiss_all_pending(self):
        """Dismiss all pending actions."""
        self.state.pending_actions.clear()

    def get_status(self) -> Dict:
        """Get current agent status for UI display."""
        return {
            'mode': self.state.mode.value,
            'is_running': self.state.is_running,
            'cycle_count': self.state.cycle_count,
            'last_cycle_time_ms': round(self.state.last_cycle_time * 1000, 2),
            'pending_actions': len(self.state.pending_actions),
            'applied_actions': len(self.state.applied_actions),
            'total_actions_history': len(self._action_history),
            'channels_tracked': len(self.state.channel_states),
            'confidence_threshold': self._confidence_threshold,
            'cycle_interval_ms': round(self.cycle_interval * 1000, 2),
            'recent_errors': self.state.errors[-5:] if self.state.errors else [],
            'kb_first': self.kb_first,
        }

    def get_pending_actions(self) -> List[Dict]:
        """Get pending actions formatted for UI display."""
        return [
            {
                'index': i,
                'type': a.action_type,
                'channel': a.channel,
                'parameters': a.parameters,
                'confidence': round(a.confidence, 3),
                'reason': a.reason,
                'source': a.source,
                'timestamp': a.timestamp,
            }
            for i, a in enumerate(self.state.pending_actions)
        ]

    def get_action_history(self, limit: int = 50) -> List[Dict]:
        """Get recent action history."""
        recent = self._action_history[-limit:]
        return [
            {
                'type': a.action_type,
                'channel': a.channel,
                'parameters': a.parameters,
                'confidence': round(a.confidence, 3),
                'reason': a.reason,
                'source': a.source,
                'timestamp': a.timestamp,
            }
            for a in reversed(recent)
        ]

    def get_channel_summary(self) -> Dict[int, Dict]:
        """Get summary of all tracked channels."""
        summary = {}
        for ch_id, state in self.state.channel_states.items():
            summary[ch_id] = {
                'instrument': state.get('instrument', 'unknown'),
                'rms_db': state.get('rms_db', -100),
                'peak_db': state.get('peak_db', -100),
                'is_muted': state.get('is_muted', False),
            }
        return summary
