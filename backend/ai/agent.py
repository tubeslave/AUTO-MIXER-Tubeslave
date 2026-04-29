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
    risk: str = 'medium'
    expected_effect: str = ''
    rollback_hint: str = ''


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

    LLM_CONTEXT_CATEGORIES = (
        'agent_auto_apply_protocol',
        'instrument_profiles',
        'live_sound_checklist',
        'mixing_rules',
        'troubleshooting',
    )

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
                knowledge_base = KnowledgeBase(allowed_categories=KnowledgeBase.AGENT_RUNTIME_CATEGORIES)
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
        self._action_audit_log: List[Dict[str, Any]] = []
        self._dismissed_action_until: Dict[tuple, float] = {}
        self._llm_query_interval = 10.0  # seconds between LLM consultations
        self._last_llm_query = 0.0
        self._dismissed_action_cooldown_sec = 30.0
        self._max_pending_actions = 50
        self._emergency_stop = False
        self._llm_context_categories = tuple(self.LLM_CONTEXT_CATEGORIES)
        self._max_fader_step_db = 1.0
        self._max_fader_db = 0.0
        self._max_eq_step_db = 2.0
        self._max_comp_threshold_step_db = 3.0
        self._max_pan_step = 0.25
        self._allowed_action_types = {
            'reduce_gain',
            'adjust_gain',
            'mute_channel',
            'unmute_channel',
            'apply_hpf',
            'adjust_compressor',
            'adjust_deesser',
            'set_eq',
            'llm_recommendation',
        }
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

    def emergency_stop(self):
        """Immediately prevent further mixer writes while keeping diagnostics available."""
        self._emergency_stop = True
        self.state.mode = AgentMode.MANUAL
        self.state.pending_actions.clear()
        self.state.is_running = False
        self._record_audit(None, "emergency_stop", "AI agent emergency stop")

    def clear_emergency_stop(self):
        """Allow the agent to run again after an operator reset."""
        self._emergency_stop = False

    async def _run_cycle(self):
        """Single ODA cycle: Observe -> Decide -> Act."""
        if self._emergency_stop:
            return

        # OBSERVE
        observations = await self._observe()

        # DECIDE
        actions = await asyncio.to_thread(self._decide, observations)
        actions = self._prepare_actions(actions)

        # ACT
        if self.state.mode == AgentMode.AUTO:
            await self._act(actions)
        elif self.state.mode == AgentMode.SUGGEST:
            self._queue_pending_actions(actions)

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
            if not self._has_actionable_signal(ch_state):
                continue

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
                            (
                                f"{ch_state.get('instrument', '')} live mixing issue "
                                "auto apply safety gain EQ compressor decision protocol"
                            ),
                            n_results=3,
                            category=self._llm_context_categories,
                        )
                        context_entries = [e.content for e in kb_hits]
                    rec = self.llm.get_mix_recommendation(ch_state, context_entries)
                    if rec:
                        if rec.get('llm_available') is False:
                            logger.info(
                                "Skipping LLM fallback defaults for ch%s because no real LLM backend is available",
                                ch_id,
                            )
                            continue
                        action = AgentAction(
                            action_type='llm_recommendation',
                            channel=ch_id,
                            parameters=rec,
                            priority=3,
                            confidence=0.6,
                            reason=rec.get('reason', 'LLM recommendation'),
                            source='llm',
                            timestamp=now,
                            risk=rec.get('risk', 'medium'),
                            expected_effect=rec.get('expected_effect', ''),
                            rollback_hint=rec.get('rollback_hint', ''),
                        )
                        actions.append(action)
                except Exception as e:
                    logger.debug(f"LLM consultation error: {e}")

        # Sort by priority (lower number = higher priority), then confidence
        actions.sort(key=lambda a: (a.priority, -a.confidence))

        # Limit actions per cycle to prevent overwhelming the mixer
        return actions[:self._max_actions_per_cycle]

    def _has_actionable_signal(self, ch_state: Dict[str, Any]) -> bool:
        """Return True only when meter data indicates a real signal to mix."""
        if ch_state.get('feedback_detected', False):
            return True
        if ch_state.get('channel_armed', False):
            return True

        try:
            peak_db = float(ch_state.get('peak_db', -200.0))
        except (TypeError, ValueError):
            peak_db = -200.0
        try:
            rms_db = float(ch_state.get('rms_db', -200.0))
        except (TypeError, ValueError):
            rms_db = -200.0
        return peak_db > -45.0 or rms_db > -50.0

    def _prepare_actions(self, actions: List[AgentAction]) -> List[AgentAction]:
        """Validate and constrain actions before suggestion or application."""
        prepared: List[AgentAction] = []
        for action in actions:
            normalized = self._normalize_action(action)
            if normalized is not None:
                prepared.append(normalized)
        return prepared[:self._max_actions_per_cycle]

    def _queue_pending_actions(self, actions: List[AgentAction]):
        """Merge new suggestions into a stable pending-action queue.

        The UI approves actions by index, so replacing the whole list on every
        cycle makes buttons race the agent loop. Existing equivalent actions are
        refreshed in place; new actions are appended and remain pending until an
        operator approves or dismisses them.
        """
        if not actions:
            return

        now = time.time()
        self._dismissed_action_until = {
            key: until
            for key, until in self._dismissed_action_until.items()
            if until > now
        }

        pending_by_key = {
            self._action_key(action): idx
            for idx, action in enumerate(self.state.pending_actions)
        }

        for action in actions:
            key = self._action_key(action)
            if self._dismissed_action_until.get(key, 0.0) > now:
                continue
            existing_idx = pending_by_key.get(key)
            if existing_idx is not None:
                self.state.pending_actions[existing_idx] = action
                continue
            if len(self.state.pending_actions) >= self._max_pending_actions:
                self._record_audit(action, "skipped", "Pending action queue is full")
                continue
            pending_by_key[key] = len(self.state.pending_actions)
            self.state.pending_actions.append(action)

    def _action_key(self, action: AgentAction) -> tuple:
        """Stable identity for deduplicating equivalent pending suggestions."""
        params = action.parameters or {}
        if action.action_type == 'set_eq':
            detail = params.get('band', 1)
        elif action.action_type == 'apply_hpf':
            detail = 'hpf'
        elif action.action_type == 'adjust_compressor':
            detail = 'compressor'
        elif action.action_type == 'llm_recommendation':
            detail = (
                bool(params.get('gain_db') is not None),
                bool(params.get('eq_bands')),
                bool(params.get('comp_threshold') is not None or params.get('threshold_db') is not None),
                bool(params.get('pan') is not None),
            )
        else:
            detail = None
        return (action.action_type, int(action.channel), action.source, detail)

    def _normalize_action(self, action: AgentAction) -> Optional[AgentAction]:
        if action.action_type not in self._allowed_action_types:
            self._reject_action(action, f"Unsupported action type: {action.action_type}")
            return None

        try:
            channel = int(action.channel)
        except (TypeError, ValueError):
            self._reject_action(action, f"Invalid channel: {action.channel}")
            return None
        if channel <= 0:
            self._reject_action(action, f"Invalid channel: {channel}")
            return None

        params = dict(action.parameters or {})
        confidence = max(0.0, min(1.0, float(action.confidence)))

        if action.action_type == 'reduce_gain':
            params['amount_db'] = max(-self._max_fader_step_db, min(0.0, float(params.get('amount_db', -1.0))))
        elif action.action_type == 'adjust_gain':
            adjustment = float(params.get('adjustment_db', 0.0))
            params['adjustment_db'] = max(-self._max_fader_step_db, min(self._max_fader_step_db, adjustment))
        elif action.action_type == 'apply_hpf':
            params['frequency'] = self._clamp_float(params.get('frequency', 80.0), 20.0, 400.0)
        elif action.action_type == 'adjust_compressor':
            params = self._normalize_compressor_params(params)
        elif action.action_type == 'set_eq':
            params['gain'] = self._clamp_float(params.get('gain', 0.0), -self._max_eq_step_db, self._max_eq_step_db)
            params['freq'] = self._clamp_float(params.get('freq', 1000.0), 20.0, 20000.0)
            params['q'] = self._clamp_float(params.get('q', 1.0), 0.2, 10.0)
        elif action.action_type == 'llm_recommendation':
            params = self._normalize_llm_recommendation(params)

        return AgentAction(
            action_type=action.action_type,
            channel=channel,
            parameters=params,
            priority=int(action.priority),
            confidence=confidence,
            reason=action.reason,
            source=action.source,
            timestamp=action.timestamp or time.time(),
            risk=action.risk,
            expected_effect=action.expected_effect,
            rollback_hint=action.rollback_hint,
        )

    def _normalize_llm_recommendation(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(rec)
        if normalized.get('gain_db') is not None:
            normalized['gain_db'] = self._clamp_float(normalized['gain_db'], -100.0, 10.0)
        if normalized.get('pan') is not None:
            normalized['pan'] = self._clamp_float(normalized['pan'], -1.0, 1.0)

        eq_bands = []
        for idx, band in enumerate(normalized.get('eq_bands') or [], start=1):
            if not isinstance(band, dict):
                continue
            eq_bands.append({
                'band': int(band.get('band', idx)),
                'freq': self._clamp_float(band.get('freq', 1000.0), 20.0, 20000.0),
                'gain_db': self._clamp_float(
                    band.get('gain_db', band.get('gain', 0.0)),
                    -self._max_eq_step_db,
                    self._max_eq_step_db,
                ),
                'q': self._clamp_float(band.get('q', 1.0), 0.2, 10.0),
            })
        normalized['eq_bands'] = eq_bands[:4]

        comp_threshold = normalized.get('comp_threshold', normalized.get('threshold_db'))
        if comp_threshold is not None:
            normalized['comp_threshold'] = self._clamp_float(comp_threshold, -60.0, 0.0)
        comp_ratio = normalized.get('comp_ratio', normalized.get('ratio'))
        if comp_ratio is not None:
            normalized['comp_ratio'] = self._clamp_float(comp_ratio, 1.0, 20.0)
        normalized['comp_attack_ms'] = self._clamp_float(
            normalized.get('comp_attack_ms', normalized.get('attack_ms', 10.0)),
            0.1,
            200.0,
        )
        normalized['comp_release_ms'] = self._clamp_float(
            normalized.get('comp_release_ms', normalized.get('release_ms', 100.0)),
            5.0,
            5000.0,
        )
        return normalized

    def _normalize_compressor_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(params)
        normalized['threshold_db'] = self._clamp_float(normalized.get('threshold_db', -18.0), -60.0, 0.0)
        normalized['ratio'] = self._clamp_float(normalized.get('ratio', 3.0), 1.0, 20.0)
        normalized['attack_ms'] = self._clamp_float(normalized.get('attack_ms', 10.0), 0.1, 200.0)
        normalized['release_ms'] = self._clamp_float(normalized.get('release_ms', 100.0), 5.0, 5000.0)
        return normalized

    def _clamp_float(self, value: Any, minimum: float, maximum: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = minimum
        return max(minimum, min(maximum, numeric))

    def _reject_action(self, action: AgentAction, reason: str):
        self.state.errors.append(f"Rejected action: {reason}")
        if len(self.state.errors) > 50:
            self.state.errors = self.state.errors[-25:]
        self._record_audit(action, "rejected", reason)

    async def _act(self, actions: List[AgentAction]):
        """Execute decided actions on the mixer.

        Uses method names available on both WingClient and DLiveClient
        (DLiveClient provides compatibility aliases).
        """
        for action in actions:
            try:
                applied = False
                if self._emergency_stop:
                    self._record_audit(action, "blocked", "Emergency stop is active")
                    continue
                if self.mixer:
                    if action.action_type == 'reduce_gain':
                        amount = action.parameters.get('amount_db', -3)
                        current = self._safe_get_fader(action.channel) or 0.0
                        target = max(-100.0, current + amount)
                        self._safe_set_fader(action.channel, target)
                        applied = True
                    elif action.action_type == 'adjust_gain':
                        adj = action.parameters.get('adjustment_db', 0)
                        current = self._safe_get_fader(action.channel) or 0.0
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
                        threshold = self._limited_compressor_threshold(
                            action.channel,
                            action.parameters.get('threshold_db', -18),
                        )
                        if hasattr(self.mixer, 'set_channel_compressor'):
                            self.mixer.set_channel_compressor(
                                action.channel,
                                threshold=threshold,
                                ratio=action.parameters.get('ratio', 3.0),
                                attack=action.parameters.get('attack_ms', 10),
                                release=action.parameters.get('release_ms', 100),
                            )
                            applied = True
                        elif hasattr(self.mixer, 'set_compressor'):
                            self.mixer.set_compressor(
                                action.channel,
                                threshold_db=threshold,
                                ratio=action.parameters.get('ratio', 3.0),
                                attack_ms=action.parameters.get('attack_ms', 10),
                                release_ms=action.parameters.get('release_ms', 100),
                                enabled=True,
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
                    elif action.action_type == 'llm_recommendation':
                        applied = self._apply_llm_recommendation(action)

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
                    self._record_audit(action, "applied", "Applied to mixer")
                else:
                    self._record_audit(action, "skipped", "No compatible mixer command was applied")
            except Exception as e:
                error_msg = f"Action error: {action.action_type} ch{action.channel}: {e}"
                logger.error(error_msg)
                self.state.errors.append(error_msg)
                self._record_audit(action, "error", str(e))

    def _safe_set_fader(self, channel: int, value_db: float):
        """Set fader using whichever method the mixer client provides."""
        value_db = max(-100.0, min(self._max_fader_db, value_db))
        current = self._safe_get_fader(channel)
        if current is not None:
            value_db = max(current - self._max_fader_step_db, min(current + self._max_fader_step_db, value_db))
        if hasattr(self.mixer, 'set_channel_fader_db'):
            self.mixer.set_channel_fader_db(channel, value_db)
        elif hasattr(self.mixer, 'set_fader'):
            self.mixer.set_fader(channel, value_db)

    def _apply_llm_recommendation(self, action: AgentAction) -> bool:
        """Apply a structured LLM recommendation using safe mixer adapters."""
        rec = action.parameters
        applied = False

        if rec.get('gain_db') is not None:
            self._safe_set_fader(action.channel, float(rec['gain_db']))
            applied = True

        if rec.get('pan') is not None and hasattr(self.mixer, 'set_pan'):
            current_pan = None
            if hasattr(self.mixer, 'get_pan'):
                try:
                    candidate = self.mixer.get_pan(action.channel)
                    if isinstance(candidate, (int, float)):
                        current_pan = float(candidate)
                except Exception:
                    current_pan = None
            target_pan = float(rec['pan'])
            if current_pan is not None:
                target_pan = max(current_pan - self._max_pan_step, min(current_pan + self._max_pan_step, target_pan))
            self.mixer.set_pan(action.channel, target_pan)
            applied = True

        eq_bands = rec.get('eq_bands') or []
        if eq_bands and hasattr(self.mixer, 'set_eq_band'):
            for idx, band in enumerate(eq_bands[:4], start=1):
                self.mixer.set_eq_band(
                    action.channel,
                    band.get('band', idx),
                    freq=band.get('freq', 1000.0),
                    gain=band.get('gain_db', band.get('gain', 0.0)),
                    q=band.get('q', 1.0),
                )
            applied = True

        comp_threshold = rec.get('comp_threshold', rec.get('threshold_db'))
        comp_ratio = rec.get('comp_ratio', rec.get('ratio'))
        if comp_threshold is not None and hasattr(self.mixer, 'set_compressor'):
            comp_threshold = self._limited_compressor_threshold(action.channel, comp_threshold)
            self.mixer.set_compressor(
                action.channel,
                threshold_db=comp_threshold,
                ratio=comp_ratio or 3.0,
                attack_ms=rec.get('comp_attack_ms', rec.get('attack_ms', 10.0)),
                release_ms=rec.get('comp_release_ms', rec.get('release_ms', 100.0)),
                enabled=True,
            )
            applied = True

        return applied

    def _limited_compressor_threshold(self, channel: int, target: float) -> float:
        target = self._clamp_float(target, -60.0, 0.0)
        if not hasattr(self.mixer, 'get_compressor_threshold'):
            return target
        try:
            current = self.mixer.get_compressor_threshold(channel)
        except Exception:
            return target
        if not isinstance(current, (int, float)):
            return target
        current = float(current)
        return max(current - self._max_comp_threshold_step_db, min(current + self._max_comp_threshold_step_db, target))

    def _safe_get_fader(self, channel: int) -> float:
        """Get fader value using whichever method the mixer client provides."""
        if hasattr(self.mixer, 'get_fader'):
            try:
                value = self.mixer.get_fader(channel)
                if isinstance(value, (int, float)):
                    return float(value)
            except Exception:
                return None
        return None

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

    def configure_safety_limits(self, **limits):
        """Update runtime safety limits for automatic mixer writes."""
        if 'max_fader_step_db' in limits:
            self._max_fader_step_db = self._clamp_float(limits['max_fader_step_db'], 0.1, 12.0)
        if 'max_fader_db' in limits:
            self._max_fader_db = self._clamp_float(limits['max_fader_db'], -100.0, 10.0)
        if 'max_eq_step_db' in limits:
            self._max_eq_step_db = self._clamp_float(limits['max_eq_step_db'], 0.1, 12.0)
        if 'max_comp_threshold_step_db' in limits:
            self._max_comp_threshold_step_db = self._clamp_float(limits['max_comp_threshold_step_db'], 0.1, 12.0)
        if 'max_pan_step' in limits:
            self._max_pan_step = self._clamp_float(limits['max_pan_step'], 0.01, 1.0)

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
            action = self.state.pending_actions.pop(action_idx)
            self._dismissed_action_until[self._action_key(action)] = (
                time.time() + self._dismissed_action_cooldown_sec
            )
            return True
        return False

    def dismiss_all_pending(self):
        """Dismiss all pending actions."""
        until = time.time() + self._dismissed_action_cooldown_sec
        for action in self.state.pending_actions:
            self._dismissed_action_until[self._action_key(action)] = until
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
            'emergency_stop': self._emergency_stop,
            'safety_limits': {
                'max_fader_step_db': self._max_fader_step_db,
                'max_fader_db': self._max_fader_db,
                'max_eq_step_db': self._max_eq_step_db,
                'max_comp_threshold_step_db': self._max_comp_threshold_step_db,
                'max_pan_step': self._max_pan_step,
            },
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
                'risk': a.risk,
                'expected_effect': a.expected_effect,
                'rollback_hint': a.rollback_hint,
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
                'risk': a.risk,
                'expected_effect': a.expected_effect,
                'rollback_hint': a.rollback_hint,
            }
            for a in reversed(recent)
        ]

    def get_action_audit_log(self, limit: int = 50) -> List[Dict]:
        return list(reversed(self._action_audit_log[-limit:]))

    def _record_audit(self, action: Optional[AgentAction], status: str, detail: str):
        entry = {
            'timestamp': time.time(),
            'status': status,
            'detail': detail,
        }
        if action is not None:
            entry.update({
                'type': action.action_type,
                'channel': action.channel,
                'parameters': action.parameters,
                'confidence': round(action.confidence, 3),
                'reason': action.reason,
                'source': action.source,
                'risk': action.risk,
            })
        self._action_audit_log.append(entry)
        if len(self._action_audit_log) > 1000:
            self._action_audit_log = self._action_audit_log[-500:]

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
